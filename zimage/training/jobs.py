"""Filesystem lifecycle and runtime orchestration for training jobs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import yaml
from slugify import slugify

from zimage.training.contracts import (
    JobState,
    JobStatus,
    RuntimeGuard,
    UpdateClassification,
)
from zimage.training.job_log import LOGS_DIR, job_log_session
from zimage.training.schema import (
    TrainingConfigError,
    classify_job_update,
    job_create_template,
    load_job_document,
    load_job_document_for_classify,
    validate_job_document,
)

CONFIG_FILE = "config.yaml"
STATE_FILE = "state.json"
JOB_DIRECTORIES = ("commands", "checkpoints", "previews", LOGS_DIR)
JOB_ROOT_ENTRIES = (CONFIG_FILE, STATE_FILE, *JOB_DIRECTORIES)

_WINDOWS_RESERVED_STEMS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}

Backend = Callable[[Path], int | None]


def resolve_job_id(base_name: str) -> str:
    """Resolve a user-visible base name to its stable job directory ID.

    Duplicate indexing belongs exclusively in this API when it is added later.
    """
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("job name must be nonempty text")

    windows_name = base_name.rstrip(" .")
    windows_stem = windows_name.split(".", 1)[0].rstrip(" ").casefold()
    if windows_stem in _WINDOWS_RESERVED_STEMS:
        raise ValueError("job name is reserved by Windows")

    job_id = slugify(base_name, lowercase=True, separator="-", allow_unicode=False)
    job_id = job_id.strip("-")
    if not job_id:
        raise ValueError("job name does not contain usable characters")
    if job_id.rstrip(" .").split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS:
        raise ValueError("job name is reserved by Windows")
    return job_id


def resolve_job_path(jobs_dir: str | Path, job_id: str) -> Path:
    """Return a direct child of ``jobs_dir`` for an already-resolved ID."""
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job ID must be nonempty text")
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("job ID must be a direct child name")
    return Path(jobs_dir) / job_id


def create_or_open_job(base_name: str, jobs_dir: str | Path) -> Path:
    """Create a job exactly once, or open the existing slug directory."""
    job_id = resolve_job_id(base_name)
    root = resolve_job_path(jobs_dir, job_id)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not root.is_dir():
            raise NotADirectoryError(root)
        return root

    document = job_create_template()
    document["job_name"] = base_name
    document = validate_job_document(document)
    # Schema validation normalizes surrounding whitespace; creation must retain
    # the user's display name byte-for-byte in the one canonical field.
    document["job_name"] = base_name
    for directory in JOB_DIRECTORIES:
        (root / directory).mkdir()
    _atomic_write_yaml(root / CONFIG_FILE, document)
    write_job_state(
        root,
        JobState(job_id=job_id, status=JobStatus.STOPPED, step=0, epoch=0),
    )
    return root


def create_job(base_name: str, jobs_dir: str | Path) -> Path:
    """Compatibility spelling for create/open semantics."""
    return create_or_open_job(base_name, jobs_dir)


def load_job_config(job_dir: str | Path) -> dict[str, Any]:
    return load_job_document(Path(job_dir) / CONFIG_FILE)


def save_job_config(
    job_dir: str | Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and atomically replace an idle job's canonical YAML.

    When a complete checkpoint exists (or always vs the current on-disk job),
    reject immutable LoRA / base-model field changes via ``classify_job_update``.
    """

    root = Path(job_dir)
    validated = validate_job_document(document)
    current_path = root / CONFIG_FILE
    if current_path.is_file():
        current = load_job_document_for_classify(current_path)
        if current is not None:
            classification, changed = classify_job_update(current, validated)
            if classification is UpdateClassification.REJECTED_IMMUTABLE:
                raise TrainingConfigError(
                    "rejected immutable fields: " + ", ".join(changed)
                )
    _atomic_write_yaml(current_path, validated)
    return validated


def load_job_state(job_dir: str | Path) -> JobState:
    target = Path(job_dir) / STATE_FILE
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError
        expected = {"job_id", "status", "step", "epoch", "last_error", "exit_code"}
        if set(raw) != expected:
            raise ValueError("unexpected state fields")
        return JobState(
            job_id=_required_text(raw["job_id"], "job_id"),
            status=JobStatus(raw["status"]),
            step=_required_nonnegative_int(raw["step"], "step"),
            epoch=_required_nonnegative_int(raw["epoch"], "epoch"),
            last_error=_optional_text(raw["last_error"], "last_error"),
            exit_code=_optional_int(raw["exit_code"], "exit_code"),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigError(f"invalid job state: {target}") from exc


def write_job_state(job_dir: str | Path, state: JobState) -> None:
    """Atomically persist only the operational ``JobState`` fields."""
    payload = asdict(state)
    payload["status"] = state.status.value
    _atomic_write_json(Path(job_dir) / STATE_FILE, payload)


class JobController:
    """Run injected backends under an injected foundation ``RuntimeGuard``."""

    def __init__(
        self,
        runtime_guard: RuntimeGuard,
        *,
        cache_backend: Backend | None = None,
        run_backend: Backend | None = None,
    ) -> None:
        if not isinstance(runtime_guard, RuntimeGuard):
            raise TypeError("runtime_guard must implement RuntimeGuard")
        self.runtime_guard = runtime_guard
        self.cache_backend = cache_backend
        self.run_backend = run_backend

    def cache(self, job_dir: str | Path) -> int:
        """Run the cache backend under the GPU lease, without trainer job status writes."""

        root = Path(job_dir)
        with job_log_session(root):
            backend = self.cache_backend
            if backend is None:
                raise RuntimeError("cache backend is not configured")
            if not self.runtime_guard.acquire():
                raise RuntimeError("training runtime is already in use")
            try:
                result = backend(root)
                return 0 if result is None else int(result)
            finally:
                self.runtime_guard.release()

    def run(self, job_dir: str | Path) -> int:
        root = Path(job_dir)
        with job_log_session(root):
            backend = self.run_backend
            if backend is None:
                raise RuntimeError("run backend is not configured")
            if not self.runtime_guard.acquire():
                state = load_job_state(root)
                write_job_state(
                    root,
                    JobState(
                        job_id=state.job_id,
                        status=JobStatus.FAILED,
                        step=state.step,
                        epoch=state.epoch,
                        last_error="training runtime is already in use",
                        exit_code=1,
                    ),
                )
                raise RuntimeError("training runtime is already in use")
            try:
                state = load_job_state(root)
                write_job_state(
                    root,
                    JobState(
                        job_id=state.job_id,
                        status=JobStatus.RUNNING,
                        step=state.step,
                        epoch=state.epoch,
                        last_error=state.last_error,
                    ),
                )
                try:
                    result = backend(root)
                    exit_code = 0 if result is None else int(result)
                except Exception as exc:
                    latest = load_job_state(root)
                    write_job_state(
                        root,
                        JobState(
                            job_id=latest.job_id,
                            status=JobStatus.FAILED,
                            step=latest.step,
                            epoch=latest.epoch,
                            last_error=str(exc),
                            exit_code=1,
                        ),
                    )
                    raise
                latest = load_job_state(root)
                final_status = (
                    JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                )
                write_job_state(
                    root,
                    JobState(
                        job_id=latest.job_id,
                        status=final_status,
                        step=latest.step,
                        epoch=latest.epoch,
                        last_error=latest.last_error,
                        exit_code=exit_code,
                    ),
                )
                return exit_code
            finally:
                self.runtime_guard.release()


def _atomic_write_yaml(target: Path, document: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(
        dict(document),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    _atomic_write_text(target, text)


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


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be nonempty text")
    return value


def _required_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value
