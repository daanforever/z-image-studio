"""Atomic, trainer-polled command queue for active training jobs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from zimage.training.contracts import CommandEnvelope, UpdateClassification
from zimage.training.jobs import CONFIG_FILE, save_job_config
from zimage.training.schema import (
    TrainingConfigError,
    classify_job_update,
    load_job_document,
    validate_job_document,
)

COMMANDS_DIRECTORY = "commands"
QUARANTINE_DIRECTORY = "quarantine"
SEQUENCE_FILE = ".next-id"
SEQUENCE_LOCK_FILE = ".sequence.lock"
_COMMAND_WIDTH = 20
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_RETRY_SECONDS = 0.01

CommandHandler = Callable[[CommandEnvelope], None]


class CommandQueue:
    """A filesystem queue that advances only when ``consume`` is called."""

    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir)
        self.commands_dir = self.job_dir / COMMANDS_DIRECTORY
        self.commands_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, kind: str, payload: Mapping[str, Any]) -> CommandEnvelope:
        if not isinstance(kind, str) or not kind:
            raise ValueError("command kind must be nonempty text")
        if not isinstance(payload, Mapping):
            raise TypeError("command payload must be a mapping")

        with _sequence_lock(self.commands_dir):
            command_id = self._reserve_id()
            envelope = CommandEnvelope(
                command_id=command_id,
                kind=kind,
                payload=dict(payload),
                created_at=time.time(),
            )
            destination = self.commands_dir / _command_filename(command_id)
            _atomic_write_json(destination, asdict(envelope))
        return envelope

    def enqueue_update(self, document: Mapping[str, Any]) -> CommandEnvelope:
        """Write a separate candidate; validation occurs at consumption."""
        if not isinstance(document, Mapping):
            raise TypeError("job update must be a mapping")
        return self.enqueue("update", {"config": dict(document)})

    def save_idle_update(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, write canonical YAML, and discard pending command JSON."""
        if not isinstance(document, Mapping):
            raise TypeError("job update must be a mapping")
        validated = _validate_idle_candidate(self.job_dir, document)
        with _sequence_lock(self.commands_dir):
            saved = save_job_config(self.job_dir, validated)
            self._discard_pending_json()
        return saved

    def _discard_pending_json(self) -> None:
        for path in self.commands_dir.glob("*.json"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def pending(self) -> list[Path]:
        """Return candidates in deterministic command-ID order."""
        candidates: list[tuple[int, Path]] = []
        malformed_names: list[Path] = []
        for path in self.commands_dir.glob("*.json"):
            command_id = _id_from_filename(path)
            if command_id is None:
                malformed_names.append(path)
            else:
                candidates.append((command_id, path))
        for path in sorted(malformed_names, key=lambda item: item.name):
            self._quarantine(path)
        return [path for _, path in sorted(candidates, key=lambda item: item[0])]

    def consume(self, handler: CommandHandler | None = None) -> list[CommandEnvelope]:
        """Consume the current queue once; this method performs no watching."""
        consumed: list[CommandEnvelope] = []
        for candidate in self.pending():
            try:
                envelope = _load_envelope(candidate)
                expected_id = _id_from_filename(candidate)
                if envelope.command_id != expected_id:
                    raise ValueError("command ID does not match filename")
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                self._quarantine(candidate)
                continue

            if handler is None:
                try:
                    self._apply(envelope)
                except (TrainingConfigError, TypeError, ValueError):
                    self._quarantine(candidate)
                    continue
            else:
                handler(envelope)

            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
            consumed.append(envelope)
        return consumed

    def _apply(self, envelope: CommandEnvelope) -> None:
        if envelope.kind != "update":
            raise ValueError(f"unsupported command kind: {envelope.kind}")
        if set(envelope.payload) != {"config"}:
            raise ValueError("update command payload must contain only config")
        document = envelope.payload["config"]
        if not isinstance(document, Mapping):
            raise TypeError("update config must be a mapping")
        save_job_config(self.job_dir, document)

    def _reserve_id(self) -> int:
        sequence_path = self.commands_dir / SEQUENCE_FILE
        try:
            current = int(sequence_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, OSError, ValueError):
            current = _largest_known_id(self.commands_dir)
        command_id = current + 1
        _atomic_write_text(sequence_path, f"{command_id}\n")
        return command_id

    def _quarantine(self, candidate: Path) -> Path:
        quarantine = self.commands_dir / QUARANTINE_DIRECTORY
        quarantine.mkdir(exist_ok=True)
        destination = quarantine / f"{candidate.name}.invalid"
        if destination.exists():
            destination = quarantine / f"{candidate.name}.{uuid4().hex}.invalid"
        os.replace(candidate, destination)
        return destination


def enqueue_command(
    job_dir: str | Path, kind: str, payload: Mapping[str, Any]
) -> CommandEnvelope:
    return CommandQueue(job_dir).enqueue(kind, payload)


def enqueue_update(
    job_dir: str | Path, document: Mapping[str, Any]
) -> CommandEnvelope:
    return CommandQueue(job_dir).enqueue_update(document)


def save_idle_update(
    job_dir: str | Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    return CommandQueue(job_dir).save_idle_update(document)


def consume_commands(
    job_dir: str | Path, handler: CommandHandler | None = None
) -> list[CommandEnvelope]:
    return CommandQueue(job_dir).consume(handler)


def _validate_idle_candidate(
    job_dir: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    validated = validate_job_document(document)
    current_path = Path(job_dir) / CONFIG_FILE
    if current_path.is_file():
        current = load_job_document(current_path)
        classification, changed = classify_job_update(current, validated)
        if classification is UpdateClassification.REJECTED_IMMUTABLE:
            raise TrainingConfigError(
                "rejected immutable fields: " + ", ".join(changed)
            )
    return validated


def _load_envelope(path: Path) -> CommandEnvelope:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("command must be an object")
    if set(raw) != {"command_id", "kind", "payload", "created_at"}:
        raise ValueError("command fields do not match the envelope contract")
    command_id = raw["command_id"]
    kind = raw["kind"]
    payload = raw["payload"]
    created_at = raw["created_at"]
    if isinstance(command_id, bool) or not isinstance(command_id, int) or command_id < 1:
        raise TypeError("command_id must be a positive integer")
    if not isinstance(kind, str) or not kind:
        raise TypeError("kind must be nonempty text")
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
        raise TypeError("created_at must be numeric")
    return CommandEnvelope(command_id, kind, payload, float(created_at))


def _command_filename(command_id: int) -> str:
    return f"{command_id:0{_COMMAND_WIDTH}d}.json"


def _id_from_filename(path: Path) -> int | None:
    stem = path.stem
    if len(stem) != _COMMAND_WIDTH or not stem.isascii() or not stem.isdigit():
        return None
    value = int(stem)
    return value if value > 0 else None


def _largest_known_id(commands_dir: Path) -> int:
    largest = 0
    for path in commands_dir.glob("*.json"):
        command_id = _id_from_filename(path)
        if command_id is not None:
            largest = max(largest, command_id)
    return largest


@contextmanager
def _sequence_lock(commands_dir: Path) -> Iterator[None]:
    """Serialize sequence updates across Windows and POSIX processes."""
    lock_path = commands_dir / SEQUENCE_LOCK_FILE
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                _acquire_file_lock(descriptor)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring command queue lock: {lock_path}"
                    )
                time.sleep(_LOCK_RETRY_SECONDS)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            _release_file_lock(descriptor)
    finally:
        os.close(descriptor)


def _acquire_file_lock(descriptor: int) -> None:
    if os.name == "nt":
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(descriptor: int) -> None:
    if os.name == "nt":
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _atomic_write_json(target: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        target,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
