from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from zimage.training.cli import (
    TRAIN_CWD,
    TRAIN_SCRIPT,
    TrainingProcessSpec,
    build_process_spec,
)
from zimage.training.contracts import JobState, JobStatus
from zimage.training.jobs import create_or_open_job, load_job_state, write_job_state
from zimage.training.runtime_guard import FileRuntimeGuard
from zimage.ui.training_process import (
    TrainingProcessManager,
    create_training_process_manager,
)

ROOT = Path(__file__).resolve().parents[3]


def test_training_process_module_does_not_import_gradio_or_loop():
    source = (
        Path(__file__).resolve().parents[3]
        / "zimage"
        / "ui"
        / "training_process.py"
    ).read_text(encoding="utf-8")
    assert "import gradio" not in source
    assert "from gradio" not in source
    assert "zimage.training.modeling" not in source
    assert "zimage.training.loop" not in source


def test_start_uses_exact_build_process_spec_argv_cwd_env(tmp_path):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    spec = build_process_spec("run", "job")
    recorded: dict = {}
    started = threading.Event()

    class RecordingPopen:
        def __init__(self, args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            self.args = args
            self.pid = 4242
            self.returncode = None
            self._done = threading.Event()
            started.set()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            finished = self._done.wait(timeout)
            if timeout is not None and not finished:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return self.returncode

        def terminate(self):
            self.returncode = 1
            self._done.set()

        def kill(self):
            self.returncode = 1
            self._done.set()

    manager = TrainingProcessManager(jobs_dir=jobs, popen=RecordingPopen)
    manager.start("job")
    assert started.wait(timeout=2)
    try:
        assert recorded["args"] == [spec.executable, *spec.argv]
        assert Path(recorded["kwargs"]["cwd"]) == spec.cwd
        assert spec.cwd == TRAIN_CWD
        assert Path(spec.argv[1]) == TRAIN_SCRIPT
        env = recorded["kwargs"]["env"]
        for key, value in spec.environment.items():
            assert env[key] == value
        assert spec.environment == {"PYTHONUNBUFFERED": "1"}
        assert manager.is_running()
        with pytest.raises(RuntimeError, match="already running"):
            manager.start("job")
    finally:
        manager.stop()


def test_duplicate_start_stop_sets_stopped_without_checkpoint(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    before = {path.name for path in (root / "checkpoints").iterdir()}
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_script_spec("import time; time.sleep(60)"),
    )

    manager.start("job")
    assert manager.is_running()
    with pytest.raises(RuntimeError, match="already running"):
        manager.start("job")

    manager.stop()

    assert not manager.is_running()
    state = load_job_state(root)
    assert state.status is JobStatus.STOPPED
    assert {path.name for path in (root / "checkpoints").iterdir()} == before
    assert list((root / "checkpoints").iterdir()) == []


def test_crash_sets_failed_last_error_and_exit_code(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_script_spec("raise SystemExit(9)"),
    )

    manager.start("job")
    assert manager.wait(timeout=10) == 9

    state = load_job_state(root)
    raw = (root / "state.json").read_text(encoding="utf-8")
    assert state.status is JobStatus.FAILED
    assert state.exit_code == 9
    assert state.last_error == "trainer exited with code 9"
    assert '"exit_code": 9' in raw
    assert "trainer exited with code 9" in raw


def test_normal_completion_releases_lease(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    lock_path = tmp_path / ".gpu.lease"
    guard = FileRuntimeGuard(lock_path)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        runtime_guard=guard,
        spec_factory=_lease_child_spec(lock_path, hold=False, exit_code=0),
    )

    manager.start("job")
    assert manager.wait(timeout=10) == 0
    assert load_job_state(root).status is JobStatus.COMPLETED
    assert guard.is_held() is False
    assert FileRuntimeGuard(lock_path).acquire() is True


def test_stop_releases_lease_held_by_killed_child(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    lock_path = tmp_path / ".gpu.lease"
    ready = tmp_path / "held"
    guard = FileRuntimeGuard(lock_path)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        runtime_guard=guard,
        spec_factory=_lease_child_spec(lock_path, hold=True, ready=ready),
    )

    manager.start("job")
    _wait_until(ready.is_file, message="child GPU lease")
    assert FileRuntimeGuard(lock_path).acquire() is False

    manager.stop()

    assert load_job_state(root).status is JobStatus.STOPPED
    assert list((root / "checkpoints").iterdir()) == []
    assert FileRuntimeGuard(lock_path).acquire() is True


def test_start_joins_previous_watcher_before_new_child(tmp_path):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    events: list[str] = []
    first_wait = threading.Event()
    first = _ControllablePopen(events=events, label="first", wait_gate=first_wait)

    class Factory:
        count = 0

        def __call__(self, *args, **kwargs):
            self.count += 1
            events.append(f"spawn-{self.count}")
            if self.count == 1:
                return first
            return _ControllablePopen(events=events, label="second")

    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=Factory(),
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    first.returncode = 0

    def release_first() -> None:
        time.sleep(0.1)
        assert "spawn-2" not in events
        first_wait.set()

    releaser = threading.Thread(target=release_first)
    releaser.start()
    manager.start("job")
    releaser.join(timeout=5)
    try:
        assert events.index("first-waited") < events.index("spawn-2")
        assert load_job_state(jobs / "job").status is JobStatus.RUNNING
    finally:
        manager.stop()


def test_stale_watcher_ignores_replaced_process(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    first_wait = threading.Event()
    first = _ControllablePopen(wait_gate=first_wait)
    replacement = _ControllablePopen()
    spawned = {"n": 0}

    def popen(*args, **kwargs):
        spawned["n"] += 1
        assert spawned["n"] == 1
        return first

    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=popen,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    watcher = manager._watcher
    manager._process = replacement
    first.returncode = 7
    first_wait.set()
    assert watcher is not None
    watcher.join(timeout=5)
    assert not watcher.is_alive()
    state = load_job_state(root)
    assert state.status is JobStatus.RUNNING
    assert state.last_error is None
    manager.stop()
    assert load_job_state(root).status is JobStatus.STOPPED


def test_stop_does_not_overwrite_completed_status(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_script_spec("raise SystemExit(0)"),
    )
    manager.start("job")
    assert manager.wait(timeout=10) == 0
    assert load_job_state(root).status is JobStatus.COMPLETED

    manager.stop()

    state = load_job_state(root)
    assert state.status is JobStatus.COMPLETED
    assert state.exit_code == 0
    assert state.last_error is None


def test_stop_does_not_overwrite_failed_status(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_script_spec("raise SystemExit(9)"),
    )
    manager.start("job")
    assert manager.wait(timeout=10) == 9
    before = load_job_state(root)
    assert before.status is JobStatus.FAILED
    assert before.last_error == "trainer exited with code 9"
    assert before.exit_code == 9

    manager.stop()

    state = load_job_state(root)
    assert state.status is JobStatus.FAILED
    assert state.last_error == "trainer exited with code 9"
    assert state.exit_code == 9


def test_stop_writes_stopped_when_state_still_running(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    wait_gate = threading.Event()
    child = _ControllablePopen(wait_gate=wait_gate)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=lambda *args, **kwargs: child,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    child.returncode = 1
    manager.stop()
    state = load_job_state(root)
    assert state.status is JobStatus.STOPPED
    assert state.last_error is None


def test_stop_kills_alive_process_but_keeps_failed_state(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    wait_gate = threading.Event()
    child = _ControllablePopen(wait_gate=wait_gate)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=lambda *args, **kwargs: child,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    assert manager.is_running()
    latest = load_job_state(root)
    write_job_state(
        root,
        JobState(
            job_id=latest.job_id,
            status=JobStatus.FAILED,
            step=latest.step,
            epoch=latest.epoch,
            last_error="controller failed first",
            exit_code=3,
        ),
    )

    manager.stop()

    assert child.returncode is not None
    assert not manager.is_running()
    state = load_job_state(root)
    assert state.status is JobStatus.FAILED
    assert state.last_error == "controller failed first"
    assert state.exit_code == 3


def test_watch_keeps_controller_last_error_and_fills_exit_code(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_failed_state_spec(root, last_error="dataset is empty"),
    )
    manager.start("job")
    assert manager.wait(timeout=10) == 9

    state = load_job_state(root)
    assert state.status is JobStatus.FAILED
    assert state.last_error == "dataset is empty"
    assert state.exit_code == 9


def test_watch_does_not_replace_failed_last_error_when_exit_code_present(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        spec_factory=_failed_state_spec(
            root, last_error="oom in sampler", exit_code=1, process_exit=4
        ),
    )
    manager.start("job")
    assert manager.wait(timeout=10) == 4

    state = load_job_state(root)
    assert state.status is JobStatus.FAILED
    assert state.last_error == "oom in sampler"
    assert state.exit_code == 1


def test_create_training_process_manager_factory(tmp_path):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    manager = create_training_process_manager(
        jobs_dir=jobs,
        spec_factory=_script_spec("raise SystemExit(0)"),
    )
    manager.start("job")
    assert manager.wait(timeout=10) == 0
    assert manager.jobs_dir == jobs.resolve()


def _script_spec(body: str):
    def factory(*argv):
        assert argv == ("run", "job")
        return TrainingProcessSpec(
            executable=sys.executable,
            argv=("-u", "-c", body),
            cwd=ROOT,
            environment={"PYTHONUNBUFFERED": "1"},
        )

    return factory


def _failed_state_spec(
    job_dir: Path,
    *,
    last_error: str,
    exit_code: int | None = None,
    process_exit: int = 9,
):
    exit_repr = "None" if exit_code is None else repr(exit_code)
    body = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from zimage.training.contracts import JobState, JobStatus\n"
        "from zimage.training.jobs import load_job_state, write_job_state\n"
        f"root = Path({str(job_dir)!r})\n"
        "latest = load_job_state(root)\n"
        "write_job_state(\n"
        "    root,\n"
        "    JobState(\n"
        "        job_id=latest.job_id,\n"
        "        status=JobStatus.FAILED,\n"
        "        step=latest.step,\n"
        "        epoch=latest.epoch,\n"
        f"        last_error={last_error!r},\n"
        f"        exit_code={exit_repr},\n"
        "    ),\n"
        ")\n"
        f"raise SystemExit({process_exit})\n"
    )
    return _script_spec(body)


class _ControllablePopen:
    def __init__(
        self,
        *args,
        events: list[str] | None = None,
        label: str = "child",
        wait_gate: threading.Event | None = None,
        **kwargs,
    ):
        self.args = args
        self.pid = id(self)
        self.returncode = None
        self._events = events
        self._label = label
        self._done = threading.Event()
        self._wait_gate = wait_gate

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._wait_gate is not None and timeout is None:
            self._wait_gate.wait(timeout=5)
            if self._events is not None:
                self._events.append(f"{self._label}-waited")
            return self.returncode
        if timeout is not None:
            if self.returncode is not None:
                self._done.set()
                if self._wait_gate is not None:
                    self._wait_gate.set()
                return self.returncode
            finished = self._done.wait(timeout)
            if not finished:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return self.returncode
        finished = self._done.wait()
        return self.returncode

    def terminate(self):
        self.returncode = 1
        self._done.set()
        if self._wait_gate is not None:
            self._wait_gate.set()

    def kill(self):
        self.returncode = 1
        self._done.set()
        if self._wait_gate is not None:
            self._wait_gate.set()


def _lease_child_spec(lock_path: Path, *, hold: bool, ready: Path | None = None, exit_code: int = 0):
    ready_line = (
        f"Path({str(ready)!r}).write_text('1', encoding='ascii')\n"
        if ready is not None
        else ""
    )
    sleep_line = "time.sleep(60)\n" if hold else ""
    body = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from zimage.training.runtime_guard import FileRuntimeGuard\n"
        f"guard = FileRuntimeGuard(Path({str(lock_path)!r}))\n"
        "assert guard.acquire()\n"
        f"{ready_line}"
        f"{sleep_line}"
        f"raise SystemExit({exit_code})\n"
    )
    return _script_spec(body)


def _wait_until(predicate, timeout=5.0, message="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {message}")


def test_stop_vs_watcher_completed_keeps_completed(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    exit_gate = threading.Event()
    process = _ControllablePopen(label="child", wait_gate=exit_gate)

    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=lambda *a, **k: process,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    process.returncode = 0
    exit_gate.set()
    _wait_until(
        lambda: load_job_state(root).status is JobStatus.COMPLETED,
        message="watcher COMPLETED",
    )
    manager.stop()
    latest = load_job_state(root)
    assert latest.status is JobStatus.COMPLETED
    assert latest.exit_code == 0


def test_stop_and_watcher_barrier_after_exit_0_is_completed(tmp_path):
    """Stop races the watcher after the child has already exited 0."""
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    exit_gate = threading.Event()
    process = _ControllablePopen(label="child", wait_gate=exit_gate)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=lambda *a, **k: process,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    barrier = threading.Barrier(2)
    real_finalize = manager._finalize_terminal

    def gated_finalize(*args, **kwargs):
        barrier.wait(timeout=5)
        return real_finalize(*args, **kwargs)

    manager._finalize_terminal = gated_finalize
    process.returncode = 0
    exit_gate.set()
    stopper = threading.Thread(target=manager.stop)
    stopper.start()
    stopper.join(timeout=5)
    assert not stopper.is_alive()
    latest = load_job_state(root)
    assert latest.status is JobStatus.COMPLETED
    assert latest.exit_code == 0


def test_stop_and_watcher_barrier_keeps_failed_after_nonzero_exit(tmp_path):
    """Stop must not overwrite FAILED after a non-zero child exit."""
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    exit_gate = threading.Event()
    process = _ControllablePopen(label="child", wait_gate=exit_gate)
    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=lambda *a, **k: process,
        spec_factory=_script_spec("pass"),
    )
    manager.start("job")
    latest = load_job_state(root)
    write_job_state(
        root,
        JobState(
            job_id=latest.job_id,
            status=JobStatus.FAILED,
            step=latest.step,
            epoch=latest.epoch,
            last_error="trainer exited with code 9",
            exit_code=9,
        ),
    )
    barrier = threading.Barrier(2)
    real_finalize = manager._finalize_terminal

    def gated_finalize(*args, **kwargs):
        barrier.wait(timeout=5)
        return real_finalize(*args, **kwargs)

    manager._finalize_terminal = gated_finalize
    process.returncode = 9
    exit_gate.set()
    stopper = threading.Thread(target=manager.stop)
    stopper.start()
    stopper.join(timeout=5)
    assert not stopper.is_alive()
    state = load_job_state(root)
    assert state.status is JobStatus.FAILED
    assert state.last_error == "trainer exited with code 9"
    assert state.exit_code == 9


def test_start_refuses_while_previous_watcher_alive(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    create_or_open_job("other", jobs)

    class StickyWatcher:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            return None

        def start(self) -> None:
            return None

    sticky = StickyWatcher()
    real_thread = threading.Thread

    class ThreadFactory:
        count = 0

        def __call__(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                return sticky
            return real_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", ThreadFactory())

    spawns = {"n": 0}

    class CountingPopen(_ControllablePopen):
        def __init__(self, *args, **kwargs):
            spawns["n"] += 1
            super().__init__(label=f"p{spawns['n']}")

    def any_job_spec(operation, job_id):
        assert operation == "run"
        return TrainingProcessSpec(
            executable=sys.executable,
            argv=("-u", "-c", "pass"),
            cwd=ROOT,
            environment={"PYTHONUNBUFFERED": "1"},
        )

    manager = TrainingProcessManager(
        jobs_dir=jobs,
        popen=CountingPopen,
        spec_factory=any_job_spec,
    )
    manager.start("job")
    assert spawns["n"] == 1
    # Child still "running"; clear process handle so Start reaches the watcher check.
    manager._process = None
    with pytest.raises(RuntimeError, match="watcher is still alive"):
        manager.start("other")
    assert spawns["n"] == 1
    sticky._alive = False
    manager.start("other")
    assert spawns["n"] == 2
    manager.stop()
