"""Windows-safe trainer subprocess lifecycle. No Gradio and no training loop.

The process manager launches the frozen CLI command from
``build_process_spec("run", job_id)`` and tracks exit status in memory only
(no persistent training logs).

Lease ownership: ``python -u train.py run`` acquires the GPU lease via
``create_runtime_guard`` inside ``JobController.run``. This manager does not
hold the lease while the child is alive (a held parent lease would make the
frozen CLI fail to acquire). After Immediate Stop, crash, or normal
completion it reclaims a dead-holder lock so inference can acquire again.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from zimage.training.cli import build_process_spec
from zimage.training.contracts import JobState, JobStatus, RuntimeGuard
from zimage.training.jobs import load_job_state, resolve_job_path, write_job_state
from zimage.training.runtime_guard import create_runtime_guard

__all__ = [
    "TrainingProcessManager",
    "create_training_process_manager",
]

PopenFactory = Callable[..., subprocess.Popen[Any]]
SpecFactory = Callable[..., Any]
_TERMINATE_TIMEOUT_SECONDS = 5.0
_WATCHER_JOIN_SECONDS = 5.0


class TrainingProcessManager:
    """Start/stop one trainer subprocess and persist operational job state."""

    def __init__(
        self,
        *,
        jobs_dir: str | Path | None = None,
        runtime_guard: RuntimeGuard | None = None,
        popen: PopenFactory | None = None,
        spec_factory: SpecFactory | None = None,
    ) -> None:
        self._jobs_dir = _resolve_jobs_dir(jobs_dir)
        self._guard = runtime_guard
        self._popen = popen or subprocess.Popen
        self._spec_factory = spec_factory or build_process_spec
        self._lock = threading.Lock()
        self._process: subprocess.Popen[Any] | None = None
        self._watcher: threading.Thread | None = None
        self._job_id: str | None = None
        self._job_dir: Path | None = None
        self._returncode: int | None = None
        self._stop_requested = False

    @property
    def jobs_dir(self) -> Path:
        return self._jobs_dir

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def job_dir(self) -> Path | None:
        return self._job_dir

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def start(self, job_id: str) -> None:
        """Start ``python -u train.py run {job_id}``. Duplicate Start fails."""
        with self._lock:
            if self.is_running():
                raise RuntimeError("training is already running")
            previous_watcher = self._watcher
            self._watcher = None

        if previous_watcher is not None:
            previous_watcher.join(timeout=_WATCHER_JOIN_SECONDS)
            if previous_watcher.is_alive():
                with self._lock:
                    # Keep the live watcher so a later Start can refuse again.
                    self._watcher = previous_watcher
                raise RuntimeError(
                    "previous training watcher is still alive; refusing Start"
                )

        with self._lock:
            if self.is_running():
                raise RuntimeError("training is already running")
            if self._watcher is not None and self._watcher.is_alive():
                raise RuntimeError(
                    "previous training watcher is still alive; refusing Start"
                )
            job_dir = resolve_job_path(self._jobs_dir, job_id)
            if not job_dir.is_dir():
                raise FileNotFoundError(f"job does not exist: {job_id}")
            load_job_state(job_dir)
            spec = self._spec_factory("run", job_id)
            environment = os.environ.copy()
            environment.update(spec.environment)
            _write_status(
                job_dir,
                JobStatus.RUNNING,
                last_error=None,
                exit_code=None,
                expected_current=None,
            )
            try:
                process = self._popen(
                    [spec.executable, *spec.argv],
                    cwd=os.fspath(spec.cwd),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                _write_status(
                    job_dir,
                    JobStatus.FAILED,
                    last_error=str(exc),
                    exit_code=1,
                    expected_current=JobStatus.RUNNING,
                )
                raise
            self._process = process
            self._job_id = job_id
            self._job_dir = job_dir
            self._returncode = None
            self._stop_requested = False
            watcher = threading.Thread(
                target=self._watch,
                args=(process, job_dir),
                name=f"training-process-{job_id}",
                daemon=True,
            )
            self._watcher = watcher
            watcher.start()

    def stop(self) -> None:
        """Immediate Stop: kill the trainer, then finalize under the lock."""
        with self._lock:
            process = self._process
            job_dir = self._job_dir
            watcher = self._watcher
            self._stop_requested = True
        if process is not None and process.poll() is None:
            _terminate_process(process)
        code = None
        if process is not None:
            try:
                code = process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        if job_dir is not None:
            self._finalize_terminal(process, job_dir, code)
        self._reclaim_lease()
        if watcher is not None:
            watcher.join(timeout=_WATCHER_JOIN_SECONDS)
        with self._lock:
            if self._process is process:
                self._process = None

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the child exits. Used by tests and Integration."""
        process = self._process
        if process is None:
            return self._returncode
        code = process.wait(timeout=timeout)
        watcher = self._watcher
        if watcher is not None:
            watcher.join(timeout=_WATCHER_JOIN_SECONDS)
        return code

    def _watch(self, process: subprocess.Popen[Any], job_dir: Path) -> None:
        """Reap the spawned child. Ignore if ``start`` replaced that process."""
        code = process.wait()
        if self._finalize_terminal(process, job_dir, code):
            self._reclaim_lease()

    def _finalize_terminal(
        self,
        process: subprocess.Popen[Any] | None,
        job_dir: Path,
        code: int | None,
    ) -> bool:
        """Apply the single terminal status. Stop and the watcher both call this.

        Exit 0 is always COMPLETED, even when Stop already set a request.
        A requested stop with a non-zero code stays STOPPED. Any other
        non-zero exit is FAILED. Does not overwrite a written FAILED.
        Returns False when ``process`` is a replaced child (do not reclaim).
        """
        with self._lock:
            if process is not None and self._process is not process:
                return False
            if code is None:
                code = self._returncode
            else:
                self._returncode = code
            latest = load_job_state(job_dir)
            if code == 0:
                if latest.status is JobStatus.FAILED:
                    if latest.exit_code is None:
                        _write_status(
                            job_dir,
                            JobStatus.FAILED,
                            last_error=latest.last_error,
                            exit_code=0,
                            expected_current=JobStatus.FAILED,
                        )
                elif latest.status is not JobStatus.COMPLETED:
                    _write_status(
                        job_dir,
                        JobStatus.COMPLETED,
                        exit_code=0,
                        expected_current=latest.status,
                    )
            elif latest.status is JobStatus.RUNNING:
                if self._stop_requested:
                    _write_status(
                        job_dir,
                        JobStatus.STOPPED,
                        expected_current=JobStatus.RUNNING,
                    )
                elif code is not None:
                    _write_status(
                        job_dir,
                        JobStatus.FAILED,
                        last_error=f"trainer exited with code {code}",
                        exit_code=code,
                    )
            elif latest.status is JobStatus.FAILED and latest.exit_code is None:
                if code is not None:
                    _write_status(
                        job_dir,
                        JobStatus.FAILED,
                        last_error=latest.last_error,
                        exit_code=code,
                        expected_current=JobStatus.FAILED,
                    )
            if self._process is process:
                self._process = None
            return True

    def _ensure_guard(self) -> RuntimeGuard:
        if self._guard is None:
            self._guard = create_runtime_guard()
        return self._guard

    def _reclaim_lease(self) -> None:
        """Release our hold, or take over a dead PID left by a killed child."""
        guard = self._ensure_guard()
        if guard.is_held():
            guard.release()
            return
        if guard.acquire():
            guard.release()


def create_training_process_manager(
    *,
    jobs_dir: str | Path | None = None,
    runtime_guard: RuntimeGuard | None = None,
    popen: PopenFactory | None = None,
    spec_factory: SpecFactory | None = None,
) -> TrainingProcessManager:
    """Construct a manager. ``jobs_dir`` defaults to the root training paths."""
    return TrainingProcessManager(
        jobs_dir=jobs_dir,
        runtime_guard=runtime_guard,
        popen=popen,
        spec_factory=spec_factory,
    )


def _resolve_jobs_dir(override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    from zimage.config import ROOT
    from zimage.training.schema import resolve_training_paths

    configured = Path(resolve_training_paths().jobs_dir)
    if configured.is_absolute():
        return configured
    return (ROOT / configured).resolve()


_UNSET = object()


def _write_status(
    job_dir: Path,
    status: JobStatus,
    *,
    last_error: str | None | object = _UNSET,
    exit_code: int | None | object = _UNSET,
    expected_current: JobStatus | None | object = _UNSET,
) -> bool:
    """Compare-and-set job status. Returns False when the precondition fails.

    Terminal callers (Stop / watcher) must hold ``TrainingProcessManager._lock``
    around the read/CAS/write. Terminal statuses only write when the current
    status matches ``expected_current`` (default: RUNNING). Filling
    ``exit_code`` on an existing FAILED may pass ``expected_current=FAILED``.
    Passing ``expected_current=None`` skips the check (used when entering
    RUNNING from Start).
    """

    latest = load_job_state(job_dir)
    if expected_current is _UNSET:
        expected_current = JobStatus.RUNNING
    if expected_current is not None and latest.status is not expected_current:
        return False
    write_job_state(
        job_dir,
        JobState(
            job_id=latest.job_id,
            status=status,
            step=latest.step,
            epoch=latest.epoch,
            last_error=latest.last_error if last_error is _UNSET else last_error,
            exit_code=latest.exit_code if exit_code is _UNSET else exit_code,
        ),
    )
    return True


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
