"""Append-only per-job training log at ``logs/job.log``.

Gradio-free filesystem helper: the trainer process creates ``logs/`` lazily
and tees Python stdout/stderr into the file. The Gradio parent never opens
this file for write. This module does not import the optimizer loop or GPU
helpers.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

LOGS_DIR = "logs"
LOG_FILE = "job.log"
TRAINING_LOGGER_NAME = "zimage.training"
DEFAULT_READ_LIMIT = 64 * 1024
TAIL_WINDOW_BYTES = 256 * 1024
TRUNCATED_MARKER = "... [truncated] ...\n"
SESSION_BANNER_PREFIX = "===== session start"

__all__ = [
    "DEFAULT_READ_LIMIT",
    "LOGS_DIR",
    "LOG_FILE",
    "JobLogChunk",
    "SESSION_BANNER_PREFIX",
    "TAIL_WINDOW_BYTES",
    "TRAINING_LOGGER_NAME",
    "TRUNCATED_MARKER",
    "job_log_path",
    "job_log_session",
    "read_job_log_chunk",
]


@dataclass(frozen=True)
class JobLogChunk:
    """One incremental read of ``logs/job.log``."""

    chunk: str
    next_offset: int
    reset: bool


def job_log_path(job_dir: str | Path) -> Path:
    """Return ``{job_dir}/logs/job.log``. ``job_dir`` is already resolved."""

    return Path(job_dir) / LOGS_DIR / LOG_FILE


@contextmanager
def job_log_session(job_dir: str | Path) -> Iterator[Path]:
    """Create ``logs/`` if needed, append a banner, and tee stdio into the file.

    The log file receives CR-overwrite lines with CSI/OSC stripped:
    in-progress ``\\r`` updates stay in memory until a newline or session
    end. ``flush`` does not commit the in-memory line. The original
    console streams receive unmodified text. Logger ``zimage.training``
    gets a ``StreamHandler`` on the teed stdout (no ``FileHandler``) and
    ``propagate`` is off so a parent logger cannot duplicate lines.
    Streams, handler, and propagate are restored in ``finally``.
    """

    path = job_log_path(job_dir)
    path.parent.mkdir(exist_ok=True)
    log_file = path.open(
        "a",
        encoding="utf-8",
        errors="replace",
        newline="\n",
        buffering=1,
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = logging.getLogger(TRAINING_LOGGER_NAME)
    handler: logging.Handler | None = None
    previous_level = logger.level
    previous_propagate = logger.propagate
    tee_out: _TeeStream | None = None
    tee_err: _TeeStream | None = None
    try:
        tee_out = _TeeStream(original_stdout, log_file)
        tee_err = _TeeStream(original_stderr, log_file)
        sys.stdout = tee_out
        sys.stderr = tee_err
        handler = logging.StreamHandler(tee_out)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        sys.stdout.write(_session_banner())
        sys.stdout.flush()
        yield path
    except Exception:
        if handler is not None:
            logger.exception("job failed")
        raise
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        if tee_out is not None:
            tee_out.commit_pending()
        if tee_err is not None:
            tee_err.commit_pending()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def read_job_log_chunk(
    job_dir: str | Path,
    offset: int,
    limit: int = DEFAULT_READ_LIMIT,
) -> JobLogChunk:
    """Read a bounded UTF-8 chunk of ``{job_dir}/logs/job.log``.

    Callers pass a job directory already produced by ``resolve_job_path``.
    The path is always ``job_dir / logs / job.log``; no caller-supplied
    relative log path is accepted.
    """

    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")

    path = job_log_path(job_dir)
    if not path.is_file():
        if offset < 0:
            return JobLogChunk(chunk="", next_offset=0, reset=True)
        return JobLogChunk(chunk="", next_offset=0, reset=False)

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        reset = False
        prefix = ""
        start = offset
        if offset < 0 or offset > size:
            start = max(0, size - TAIL_WINDOW_BYTES)
            reset = True
            if start > 0:
                prefix = TRUNCATED_MARKER
        handle.seek(start)
        raw = handle.read(limit)

    holdback = _incomplete_utf8_suffix_len(raw)
    # Only hold incomplete bytes when the read hit ``limit`` (a cut). At
    # EOF, consume the remainder with ``errors="replace"`` so a trailing
    # fragment cannot stall ``next_offset``.
    if holdback and len(raw) < limit:
        holdback = 0
    complete = raw[: len(raw) - holdback] if holdback else raw
    text = complete.decode("utf-8", errors="replace")
    return JobLogChunk(
        chunk=prefix + text,
        next_offset=start + len(complete),
        reset=reset,
    )


def _session_banner() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{SESSION_BANNER_PREFIX} {stamp} pid={os.getpid()} =====\n"


def _incomplete_utf8_suffix_len(data: bytes) -> int:
    """Return 1–3 if ``data`` ends on an incomplete UTF-8 sequence, else 0."""

    if not data:
        return 0
    index = len(data) - 1
    while index >= 0 and data[index] & 0xC0 == 0x80:
        index -= 1
        if len(data) - index > 4:
            return 0
    if index < 0:
        return 0
    width = _utf8_lead_width(data[index])
    if width <= 0:
        return 0
    have = len(data) - index
    if have < width:
        return have
    return 0


def _utf8_lead_width(byte: int) -> int:
    if byte < 0x80:
        return 1
    if byte < 0xC0:
        return 0
    if byte < 0xE0:
        return 2
    if byte < 0xF0:
        return 3
    if byte < 0xF8:
        return 4
    return 0


def _skip_escape(text: str, start: int) -> int | None:
    """Return the length of a CSI/OSC (or other ESC) sequence, or None if incomplete."""

    remaining = len(text) - start
    if remaining < 2:
        return None
    kind = text[start + 1]
    if kind == "[":
        index = start + 2
        while index < len(text):
            code = ord(text[index])
            if 0x20 <= code <= 0x3F:
                index += 1
                continue
            if 0x40 <= code <= 0x7E:
                return index + 1 - start
            return index - start
        return None
    if kind == "]":
        index = start + 2
        while index < len(text):
            char = text[index]
            if char == "\x07":
                return index + 1 - start
            if char == "\x1b":
                if index + 1 >= len(text):
                    return None
                if text[index + 1] == "\\":
                    return index + 2 - start
                index += 1
                continue
            if char in "\r\n":
                return index - start
            index += 1
        return None
    return 2


class _JobLogLineBuffer:
    """Collapse CR overwrites and strip CSI/OSC into committed file lines.

    ``\\r`` moves to column 0 and the next characters overwrite in place
    without clearing the rest of the line. ``\\n`` emits the current line.
    ``flush`` is not handled here: callers must not commit on flush.
    """

    def __init__(self) -> None:
        self._chars: list[str] = []
        self._column = 0
        self._pending = ""

    def feed(self, data: str) -> str:
        if not data:
            return ""
        text = self._pending + data
        self._pending = ""
        committed: list[str] = []
        index = 0
        length = len(text)
        while index < length:
            char = text[index]
            if char == "\x1b":
                skipped = _skip_escape(text, index)
                if skipped is None:
                    self._pending = text[index:]
                    break
                index += skipped if skipped > 0 else 1
                continue
            if char == "\r":
                self._column = 0
                index += 1
                continue
            if char == "\n":
                committed.append("".join(self._chars) + "\n")
                self._chars = []
                self._column = 0
                index += 1
                continue
            self._put(char)
            index += 1
        return "".join(committed)

    def close(self) -> str:
        self._pending = ""
        if not self._chars:
            return ""
        result = "".join(self._chars) + "\n"
        self._chars = []
        self._column = 0
        return result

    def _put(self, char: str) -> None:
        if self._column < len(self._chars):
            self._chars[self._column] = char
        else:
            self._chars.append(char)
        self._column += 1


class _TeeStream:
    """Write unmodified text to the console and overwrite-normalized text to the log."""

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file
        self._buffer = _JobLogLineBuffer()

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = os.fsdecode(data) if isinstance(data, bytes) else str(data)
        written = self._original.write(data)
        committed = self._buffer.feed(data)
        if committed:
            self._log_file.write(committed)
            self._log_file.flush()
        return written if isinstance(written, int) else len(data)

    def commit_pending(self) -> None:
        leftover = self._buffer.close()
        if leftover:
            self._log_file.write(leftover)
            self._log_file.flush()

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        check = getattr(self._original, "isatty", None)
        return bool(check()) if callable(check) else False

    @property
    def encoding(self) -> str | None:
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:
        return getattr(self._original, "errors", "replace")

    @property
    def name(self) -> str:
        return getattr(self._original, "name", "<job-log-tee>")

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        return None

    @property
    def closed(self) -> bool:
        return bool(getattr(self._log_file, "closed", False))

    def fileno(self) -> int:
        raise OSError("fileno is not supported on a job-log tee")
