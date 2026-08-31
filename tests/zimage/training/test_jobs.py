from __future__ import annotations

import json

import pytest
import yaml

from zimage.training.contracts import JobState, JobStatus
from zimage.training.jobs import (
    JOB_ROOT_ENTRIES,
    JobController,
    create_or_open_job,
    load_job_config,
    load_job_state,
    resolve_job_id,
    save_job_config,
    write_job_state,
)
from zimage.training.schema import TrainingConfigError


def test_resolve_job_id_transliterates_and_normalizes_separators():
    assert resolve_job_id("Мой стиль__V2") == "moi-stil-v2"


@pytest.mark.parametrize("name", ["", " ", "💥", "\u200b"])
def test_resolve_job_id_rejects_empty_input_or_result(name):
    with pytest.raises(ValueError):
        resolve_job_id(name)


@pytest.mark.parametrize(
    "name",
    ["CON", "con", "Con.txt", "PRN. ", "aux.json...", "NUL ", "COM1", "lpt9.log"],
)
def test_resolve_job_id_rejects_windows_reserved_names(name):
    with pytest.raises(ValueError):
        resolve_job_id(name)


def test_create_has_exact_layout_and_preserves_original_name(tmp_path):
    root = create_or_open_job("  Мой Стиль  ", tmp_path / "jobs")

    assert root.name == "moi-stil"
    assert {path.name for path in root.iterdir()} == set(JOB_ROOT_ENTRIES)
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "commands",
        "checkpoints",
        "previews",
        "logs",
    }
    assert not (root / "logs" / "job.log").exists()
    persisted = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["job_name"] == "  Мой Стиль  "
    assert load_job_state(root) == JobState(
        job_id="moi-stil",
        status=JobStatus.STOPPED,
        step=0,
        epoch=0,
    )


def test_create_opens_existing_job_without_overwriting_any_file(tmp_path):
    root = create_or_open_job("Same Name", tmp_path)
    config = root / "config.yaml"
    state = root / "state.json"
    config.write_text("sentinel config", encoding="utf-8")
    state.write_text("sentinel state", encoding="utf-8")

    reopened = create_or_open_job("same-name", tmp_path)

    assert reopened == root
    assert config.read_text(encoding="utf-8") == "sentinel config"
    assert state.read_text(encoding="utf-8") == "sentinel state"


def test_idle_save_validates_before_atomic_replacement(tmp_path):
    root = create_or_open_job("job", tmp_path)
    original = (root / "config.yaml").read_text(encoding="utf-8")
    update = load_job_config(root)
    update["seed"] = 123

    saved = save_job_config(root, update)

    assert saved["seed"] == 123
    assert load_job_config(root)["seed"] == 123
    assert not list(root.glob("*.tmp"))

    update["precision"] = "invalid"
    with pytest.raises(TrainingConfigError):
        save_job_config(root, update)
    assert load_job_config(root)["seed"] == 123
    assert (root / "config.yaml").read_text(encoding="utf-8") != original


@pytest.mark.parametrize(
    "status", [JobStatus.RUNNING, JobStatus.STOPPED, JobStatus.COMPLETED, JobStatus.FAILED]
)
def test_state_json_contains_only_operational_fields(tmp_path, status):
    root = create_or_open_job("job", tmp_path)
    state = JobState(
        job_id="job",
        status=status,
        step=7,
        epoch=2,
        last_error="failure" if status is JobStatus.FAILED else None,
        exit_code=1 if status is JobStatus.FAILED else None,
    )

    write_job_state(root, state)

    raw = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert set(raw) == {
        "job_id",
        "status",
        "step",
        "epoch",
        "last_error",
        "exit_code",
    }
    assert raw["status"] == status.value
    assert load_job_state(root) == state


class RecordingGuard:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.calls = []

    def acquire(self) -> bool:
        self.calls.append("acquire")
        return self.acquired

    def release(self) -> None:
        self.calls.append("release")

    def is_held(self) -> bool:
        return self.acquired


def test_controller_requires_injected_runtime_guard():
    with pytest.raises(TypeError):
        JobController(object())  # type: ignore[arg-type]


def test_controller_uses_injected_guard_and_backend(tmp_path):
    root = create_or_open_job("job", tmp_path)
    guard = RecordingGuard()
    backend_calls = []
    controller = JobController(
        guard,
        run_backend=lambda path: backend_calls.append(path) or 0,
    )

    assert controller.run(root) == 0
    assert guard.calls == ["acquire", "release"]
    assert backend_calls == [root]
    assert load_job_state(root).status is JobStatus.COMPLETED
    assert (root / "logs" / "job.log").is_file()
    assert "job.log" not in {path.name for path in root.iterdir()}
    assert {path.name for path in root.iterdir()} == set(JOB_ROOT_ENTRIES)


def test_controller_cache_does_not_acquire_gpu_lease(tmp_path):
    root = create_or_open_job("job", tmp_path)
    guard = RecordingGuard()
    backend_calls = []
    controller = JobController(
        guard,
        cache_backend=lambda path: backend_calls.append(path) or 0,
    )

    assert controller.cache(root) == 0
    assert guard.calls == []
    assert backend_calls == [root]
    assert (root / "logs" / "job.log").is_file()
    assert "job.log" not in {path.name for path in root.iterdir()}


def test_controller_cache_raises_when_backend_missing(tmp_path):
    root = create_or_open_job("job", tmp_path)
    guard = RecordingGuard()
    controller = JobController(guard)

    with pytest.raises(RuntimeError, match="cache backend is not configured"):
        controller.cache(root)
    assert guard.calls == []


def test_run_busy_guard_writes_failed_and_does_not_release(tmp_path):
    root = create_or_open_job("job", tmp_path)
    initial = JobState("job", JobStatus.STOPPED, step=4, epoch=2)
    write_job_state(root, initial)
    guard = RecordingGuard(acquired=False)
    controller = JobController(guard, run_backend=lambda _: pytest.fail("called"))

    with pytest.raises(RuntimeError, match="already in use"):
        controller.run(root)

    latest = load_job_state(root)
    assert latest.status is JobStatus.FAILED
    assert latest.step == 4
    assert latest.epoch == 2
    assert latest.last_error == "training runtime is already in use"
    assert latest.exit_code == 1
    assert guard.calls == ["acquire"]


def test_run_preserves_backend_last_error_when_exit_nonzero(tmp_path):
    root = create_or_open_job("job", tmp_path)
    guard = RecordingGuard()

    def backend(path):
        write_job_state(
            path,
            JobState(
                "job",
                JobStatus.RUNNING,
                step=7,
                epoch=1,
                last_error="preview failed; additionally, release failed",
            ),
        )
        return 1

    controller = JobController(guard, run_backend=backend)

    assert controller.run(root) == 1
    latest = load_job_state(root)
    assert latest.status is JobStatus.FAILED
    assert latest.exit_code == 1
    assert latest.last_error == "preview failed; additionally, release failed"
    assert latest.step == 7
    assert guard.calls == ["acquire", "release"]


def test_run_preserves_backend_updated_progress_in_final_state(tmp_path):
    root = create_or_open_job("job", tmp_path)
    guard = RecordingGuard()

    def backend(path):
        write_job_state(path, JobState("job", JobStatus.RUNNING, step=19, epoch=3))
        return 0

    controller = JobController(guard, run_backend=backend)

    assert controller.run(root) == 0
    assert load_job_state(root) == JobState(
        "job",
        JobStatus.COMPLETED,
        step=19,
        epoch=3,
        exit_code=0,
    )
    assert guard.calls == ["acquire", "release"]


def test_two_controller_runs_append_two_session_banners(tmp_path):
    root = create_or_open_job("job", tmp_path)
    controller = JobController(
        RecordingGuard(),
        run_backend=lambda _: 0,
    )

    assert controller.run(root) == 0
    assert controller.run(root) == 0

    text = (root / "logs" / "job.log").read_text(encoding="utf-8")
    assert text.count("===== session start") == 2
    assert {path.name for path in root.iterdir()} == set(JOB_ROOT_ENTRIES)


def test_idle_save_rejects_immutable_lora_after_checkpoint(tmp_path):
    import torch

    from zimage.training.checkpoints import NativeLoraCheckpointWriter
    from zimage.training.contracts import NativeAdapterMetadata

    root = create_or_open_job("job", tmp_path)
    config = load_job_config(root)
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / "step-1",
        lora_state={
            "to_q.lora_A.weight": torch.ones(2, 16),
            "to_q.lora_B.weight": torch.ones(16, 2),
        },
        metadata=NativeAdapterMetadata(
            adapter_name="default",
            base_model_name_or_path=str(
                config["model"]["main_transformer"]["path"]
            ),
            base_model_revision=config["model"]["main_transformer"].get("revision"),
            peft_config={
                "r": int(config["lora"]["rank"]),
                "lora_alpha": float(config["lora"]["alpha"]),
                "lora_dropout": 0.0,
                "target_modules": list(config["lora"]["targets"]),
                "peft_type": "LORA",
            },
            optimizer_step=1,
        ),
    )
    mutated = load_job_config(root)
    mutated["lora"]["alpha"] = float(mutated["lora"]["alpha"]) + 1.0
    with pytest.raises(TrainingConfigError, match="immutable|lora.alpha"):
        save_job_config(root, mutated)
    assert load_job_config(root)["lora"]["alpha"] == config["lora"]["alpha"]

    lr_ok = load_job_config(root)
    lr_ok["optimizer"]["learning_rate"] = 5.0e-5
    saved = save_job_config(root, lr_ok)
    assert saved["optimizer"]["learning_rate"] == 5.0e-5
    assert load_job_config(root)["optimizer"]["learning_rate"] == 5.0e-5
