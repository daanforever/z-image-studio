"""OS-backed exclusive GPU lease shared by inference and training.

The lock file is a PID lease serialized with the same Windows ``msvcrt`` /
POSIX ``fcntl`` critical section used by the command queue. A live holder
blocks ``acquire``. A stale PID (process already exited) is taken over so a
crashed or killed owner cannot pin the GPU.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from zimage.training.contracts import RuntimeGuard

__all__ = [
    "LOCK_ENV_VAR",
    "LOCK_FILENAME",
    "FileRuntimeGuard",
    "create_runtime_guard",
    "pid_is_alive",
    "resolve_runtime_lock_path",
]

if os.name == "nt":
    import ctypes
    import msvcrt
else:
    import fcntl

LOCK_FILENAME = ".gpu.lease"
LOCK_ENV_VAR = "ZIMAGE_RUNTIME_LOCK"
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_RETRY_SECONDS = 0.01
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5

_REGISTRY_LOCK = threading.Lock()
_LOCAL_HOLDERS: dict[str, int] = {}


class FileRuntimeGuard:
    """One exclusive cross-process GPU lease backed by a lock file.

    ``acquire`` is thread-bound and reentrant on the holder thread: a nested
    call on that thread increments depth and returns True. Any other thread
    in this process — including one using this same instance — gets False
    while depth is positive. ``release`` decrements depth and clears the
    lock file only on the last matching release from the holder thread.

    ``is_held`` is instance-level (depth > 0), not thread-bound. Another
    thread must call ``acquire`` to learn whether it may take the lease.
    """

    def __init__(self, lock_path: str | Path) -> None:
        self.lock_path = Path(lock_path)
        self._depth = 0
        self._holder_ident: int | None = None
        self._thread_lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to take the lease. True if this thread now holds it."""
        ident = threading.get_ident()
        with self._thread_lock:
            if self._depth > 0:
                if self._holder_ident == ident:
                    self._depth += 1
                    return True
                return False
            key = _lock_key(self.lock_path)
            with _REGISTRY_LOCK:
                if key in _LOCAL_HOLDERS:
                    return False
                try:
                    with _critical_section(self.lock_path) as descriptor:
                        holder = _read_holder_pid(descriptor)
                        if (
                            holder is not None
                            and holder != os.getpid()
                            and pid_is_alive(holder)
                        ):
                            return False
                        _write_holder_pid(descriptor, os.getpid())
                        _LOCAL_HOLDERS[key] = id(self)
                        self._depth = 1
                        self._holder_ident = ident
                        return True
                except TimeoutError:
                    return False

    def release(self) -> None:
        """Decrement depth; clear the lock file only on the last holder release."""
        with self._thread_lock:
            if self._depth == 0 or self._holder_ident != threading.get_ident():
                return
            if self._depth > 1:
                self._depth -= 1
                return
            key = _lock_key(self.lock_path)
            with _REGISTRY_LOCK:
                try:
                    with _critical_section(self.lock_path) as descriptor:
                        if _read_holder_pid(descriptor) == os.getpid():
                            _write_holder_pid(descriptor, None)
                except TimeoutError:
                    pass
                finally:
                    self._depth = 0
                    self._holder_ident = None
                    if _LOCAL_HOLDERS.get(key) == id(self):
                        del _LOCAL_HOLDERS[key]

    def is_held(self) -> bool:
        """Whether this instance currently holds the lease (depth > 0).

        Instance-level only. Cross-thread callers must use ``acquire``.
        """
        with self._thread_lock:
            return self._depth > 0


def create_runtime_guard(
    lock_path: str | Path | None = None,
) -> RuntimeGuard:
    """Build the process-wide GPU lease. Does not import Gradio."""
    path = Path(lock_path) if lock_path is not None else resolve_runtime_lock_path()
    return FileRuntimeGuard(path)


def resolve_runtime_lock_path() -> Path:
    """Resolve the exclusive lease file without importing Gradio.

    Preference order:
    1. ``ZIMAGE_RUNTIME_LOCK`` if set
    2. ``.gpu.lease`` under the configured training ``jobs_dir``
    3. ``.gpu.lease`` next to the root ``config.yaml`` if training paths fail
    """
    override = os.environ.get(LOCK_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    from zimage.config import ROOT

    try:
        from zimage.training.schema import (
            TrainingConfigError,
            resolve_training_paths,
        )

        configured = Path(resolve_training_paths().jobs_dir)
        jobs_dir = configured if configured.is_absolute() else (ROOT / configured)
        return jobs_dir.resolve() / LOCK_FILENAME
    except (OSError, TrainingConfigError):
        return (ROOT / LOCK_FILENAME).resolve()


def pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` still names a running process."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        ctypes.c_uint32,
        ctypes.c_bool,
        ctypes.c_uint32,
    ]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool

    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if handle:
        exit_code = ctypes.c_uint32()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        if ok:
            return exit_code.value == _STILL_ACTIVE
        return True
    return ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def _lock_key(lock_path: Path) -> str:
    return os.path.normcase(str(Path(lock_path).resolve()))


@contextmanager
def _critical_section(lock_path: Path) -> Iterator[int]:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                _acquire_file_lock(descriptor)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring GPU lease: {path}")
                time.sleep(_LOCK_RETRY_SECONDS)
        try:
            yield descriptor
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


def _read_holder_pid(descriptor: int) -> int | None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = os.read(descriptor, 256)
    text = data.decode("ascii", errors="ignore").replace("\0", " ").strip()
    if not text:
        return None
    token = text.split()[0]
    if not token.isdigit():
        return None
    pid = int(token)
    return pid if pid > 0 else None


def _write_holder_pid(descriptor: int, pid: int | None) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    payload = b"\0" if pid is None else f"{pid}\n".encode("ascii")
    os.write(descriptor, payload)
    os.fsync(descriptor)
