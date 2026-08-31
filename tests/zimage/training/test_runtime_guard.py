from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from zimage.training.cli import RUNTIME_GUARD_FACTORY
from zimage.training.contracts import RuntimeGuard
from zimage.training.runtime_guard import (
    LOCK_FILENAME,
    create_runtime_guard,
    FileRuntimeGuard,
    pid_is_alive,
    resolve_runtime_lock_path,
)

ROOT = Path(__file__).resolve().parents[3]


def test_factory_contract_matches_cli_entrypoint():
    assert RUNTIME_GUARD_FACTORY == "zimage.training.runtime_guard:create_runtime_guard"
    guard = create_runtime_guard(Path("unused").resolve() / "will-not-create-until-acquire")
    assert isinstance(guard, RuntimeGuard)
    assert hasattr(guard, "acquire")
    assert hasattr(guard, "release")
    assert hasattr(guard, "is_held")


def test_acquire_is_exclusive_second_acquire_fails(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    first = FileRuntimeGuard(lock_path)
    second = FileRuntimeGuard(lock_path)

    assert first.acquire() is True
    assert first.is_held() is True
    assert second.acquire() is False
    assert second.is_held() is False
    assert create_runtime_guard(lock_path).acquire() is False


def test_release_allows_another_caller_to_acquire(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    first = FileRuntimeGuard(lock_path)
    second = FileRuntimeGuard(lock_path)

    assert first.acquire() is True
    first.release()
    assert first.is_held() is False
    assert second.acquire() is True
    second.release()


def test_same_thread_nested_acquire_release(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    guard = FileRuntimeGuard(lock_path)
    other = FileRuntimeGuard(lock_path)

    assert guard.acquire() is True
    assert guard.acquire() is True
    assert guard.is_held() is True
    assert other.acquire() is False
    guard.release()
    assert guard.is_held() is True
    assert other.acquire() is False
    guard.release()
    assert guard.is_held() is False
    assert other.acquire() is True
    other.release()


def test_other_thread_acquire_is_false_while_instance_held(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    guard = FileRuntimeGuard(lock_path)
    held = threading.Event()
    released = threading.Event()
    other_result: dict[str, bool] = {}

    def holder() -> None:
        assert guard.acquire() is True
        held.set()
        assert released.wait(timeout=5)
        guard.release()

    def other() -> None:
        assert held.wait(timeout=5)
        other_result["acquire"] = guard.acquire()
        other_result["is_held"] = guard.is_held()
        guard.release()
        other_result["still_exclusive"] = FileRuntimeGuard(lock_path).acquire()
        released.set()

    first = threading.Thread(target=holder)
    second = threading.Thread(target=other)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert other_result["acquire"] is False
    assert other_result["is_held"] is True
    assert other_result["still_exclusive"] is False
    assert guard.is_held() is False
    recovered = FileRuntimeGuard(lock_path)
    assert recovered.acquire() is True
    recovered.release()


def test_stale_lock_recovery_takes_over_dead_pid(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = dead.pid
    dead.wait(timeout=10)
    assert not pid_is_alive(dead_pid)
    lock_path.write_text(f"{dead_pid}\n", encoding="ascii")

    guard = FileRuntimeGuard(lock_path)
    assert guard.acquire() is True
    assert guard.is_held() is True
    assert lock_path.read_text(encoding="ascii").strip() == str(os.getpid())
    guard.release()


def test_cross_process_holder_blocks_then_death_unblocks(tmp_path):
    lock_path = tmp_path / LOCK_FILENAME
    ready = tmp_path / "ready"
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from zimage.training.runtime_guard import FileRuntimeGuard\n"
        f"guard = FileRuntimeGuard(Path({str(lock_path)!r}))\n"
        "assert guard.acquire()\n"
        f"Path({str(ready)!r}).write_text('1', encoding='ascii')\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=str(ROOT))
    try:
        _wait_until(ready.is_file, message="child lease")
        other = FileRuntimeGuard(lock_path)
        assert other.acquire() is False
    finally:
        child.terminate()
        child.wait(timeout=10)

    recovered = FileRuntimeGuard(lock_path)
    assert recovered.acquire() is True
    recovered.release()


def test_create_runtime_guard_resolves_jobs_dir_without_gradio(
    monkeypatch, tmp_path
):
    source = Path(__file__).resolve().parents[3] / "zimage" / "training" / "runtime_guard.py"
    assert "import gradio" not in source.read_text(encoding="utf-8")

    jobs = tmp_path / "configured-jobs"

    class Paths:
        jobs_dir = str(jobs)

    monkeypatch.setattr(
        "zimage.training.schema.resolve_training_paths",
        lambda: Paths(),
    )
    monkeypatch.delenv("ZIMAGE_RUNTIME_LOCK", raising=False)

    guard = create_runtime_guard()
    assert guard.lock_path == (jobs / LOCK_FILENAME).resolve()
    assert guard.acquire() is True
    assert (jobs / LOCK_FILENAME).is_file()
    guard.release()


def test_create_runtime_guard_honors_environment_override(monkeypatch, tmp_path):
    lock_path = tmp_path / "override.lease"
    monkeypatch.setenv("ZIMAGE_RUNTIME_LOCK", str(lock_path))

    resolved = resolve_runtime_lock_path()
    guard = create_runtime_guard()
    assert resolved == lock_path.resolve()
    assert guard.lock_path == lock_path.resolve()
    assert guard.acquire() is True
    guard.release()


def _wait_until(predicate, timeout=5.0, message="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {message}")
