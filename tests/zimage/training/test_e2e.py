"""Studio training integration tests.

CPU-mocked config → cache → step → checkpoint → preview paths live here.
Those tests do not execute CUDA. The opt-in Blackwell smoke is the
real-weight GPU path and wraps production setup/step observers.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
from PIL import Image

from tests.zimage.training.test_loop import (
    RecordingSampler,
    RecordingWriter,
    injections,
    make_job,
)
from zimage.engine import pipeline as pipeline_mod
from zimage.engine.pipeline import GPU_LEASE_HELD_MESSAGE, generate_image
from zimage.training.checkpoints import (
    load_latest_lora_state,
    load_lora_state,
    validate_checkpoint_directory,
)
from zimage.training.contracts import JobStatus
from zimage.training.jobs import (
    JOB_ROOT_ENTRIES,
    JobController,
    create_or_open_job,
    load_job_config,
    load_job_state,
    save_job_config,
)
from zimage.training.loop import cache_job, run_job
from zimage.training.runtime_guard import FileRuntimeGuard
from zimage.training.schema import job_create_template
import zimage.ui.handlers as handlers


ALLOWED_JOB_ENTRIES = set(JOB_ROOT_ENTRIES)
HARDWARE_SMOKE_FLAG = "ZIMAGE_RUN_HARDWARE_SMOKE"
REAL_MAIN_MODEL = "ZIMAGE_REAL_MAIN_MODEL"
REAL_SAMPLING_MODEL = "ZIMAGE_REAL_SAMPLING_MODEL"
_ONE_GIB = 1024 ** 3


def _resolve_named_module(root, target: str):
    """Find a configured LoRA target without assuming PEFT wrapper layout."""

    named = dict(root.named_modules())
    if target in named:
        return named[target]
    matches = [
        (name, module)
        for name, module in named.items()
        if name == target or name.endswith(f".{target}")
    ]
    if not matches:
        available = sorted(named)
        raise AssertionError(
            f"configured LoRA target {target!r} not found among {available[:40]}"
        )
    matches.sort(key=lambda item: len(item[0]))
    return matches[0][1]


def _module_is_torchao_float8_linear(module) -> bool:
    from torchao.float8.float8_linear import Float8Linear

    stack = [module]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current) is Float8Linear or isinstance(current, Float8Linear):
            return True
        getter = getattr(current, "get_base_layer", None)
        if callable(getter):
            base = getter()
            if base is not None:
                stack.append(base)
        nested = getattr(current, "base_layer", None)
        if nested is not None:
            stack.append(nested)
        children = getattr(current, "children", None)
        if callable(children):
            stack.extend(list(children()))
    return False


def _assert_trainable_lora_bf16_cuda(model) -> None:
    import torch

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    for name, parameter in trainable:
        assert parameter.dtype is torch.bfloat16, name
        assert parameter.device.type == "cuda", name


def _assert_observed_setup(setup, *, lora_targets: list[str]) -> None:
    assert setup.fp8_enabled is True
    for target in lora_targets:
        module = _resolve_named_module(setup.transformer, target)
        assert _module_is_torchao_float8_linear(module), (
            f"{target} was not converted to TorchAO Float8Linear"
        )
    _assert_trainable_lora_bf16_cuda(setup.transformer)


def _assert_observed_flow_result(result) -> None:
    import torch

    assert result.loss.device.type == "cuda"
    assert result.loss.dtype == torch.float32
    assert result.noisy_latent.device.type == "cuda"
    assert result.model_pred.device.type == "cuda"
    assert result.packed_inputs
    assert all(tensor.device.type == "cuda" for tensor in result.packed_inputs)


def _job_names(job_dir: Path) -> set[str]:
    return {child.name for child in job_dir.iterdir()}


def test_mocked_e2e_cache_step_checkpoint_preview(tmp_path: Path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events)
    root = make_job(
        tmp_path,
        max_steps=1,
        checkpoint_every=1,
        sampling={
            "common_parameters": {
                "num_inference_steps": 9,
                "guidance_scale": 0.0,
                "time_shift": 3.0,
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "prompt": "shared",
                "negative_prompt": "",
            },
            "samples": [{"prompt": "one"}],
        },
    )

    assert cache_job(root, **injections(events)) == 0
    assert (
        run_job(
            root,
            **injections(
                events,
                checkpoint_writer=writer,
                preview_sampler=sampler,
            ),
        )
        == 0
    )

    assert events.count("write") == 1
    assert events.count("sample") == 1
    assert events.index("write") < events.index("sample")
    assert writer.saved[0].path.is_dir()
    assert (root / "previews" / "step-1" / "00.png").is_file() or sampler.calls
    assert _job_names(root) <= ALLOWED_JOB_ENTRIES
    assert "metrics" not in _job_names(root)
    assert "debug" not in _job_names(root)
    assert not any(name.endswith(".log") for name in _job_names(root))
    for child in root.iterdir():
        assert child.name in ALLOWED_JOB_ENTRIES


def test_natural_completion_final_save_then_sample(tmp_path: Path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events)
    root = make_job(tmp_path, max_steps=3, checkpoint_every=2)

    assert (
        run_job(
            root,
            **injections(
                events,
                checkpoint_writer=writer,
                preview_sampler=sampler,
            ),
        )
        == 0
    )
    assert events.count("write") == 2
    assert events.count("sample") == 2
    assert events.index("write") < events.index("sample")
    assert [saved.metadata.optimizer_step for saved in writer.saved] == [2, 3]


def test_sampling_error_leaves_checkpoint_and_fails(tmp_path: Path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events, fail=True)
    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)

    assert (
        run_job(
            root,
            **injections(
                events,
                checkpoint_writer=writer,
                preview_sampler=sampler,
            ),
        )
        != 0
    )
    assert events.index("write") < events.index("sample")
    kept = writer.saved[0].path / "kept.txt"
    assert kept.is_file()
    assert load_job_state(root).last_error == "preview failed"


def test_generate_rejected_while_lease_held(monkeypatch, tmp_path: Path):
    """CPU-mocked lease rejection; does not run CUDA kernels."""

    lock_path = tmp_path / "gpu.lease"
    monkeypatch.setenv("ZIMAGE_RUNTIME_LOCK", str(lock_path))
    monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)
    monkeypatch.setattr(pipeline_mod, "runtime_status", lambda: {
        "demo": False,
        "cuda": False,
        "torch": True,
        "device": "cpu",
        "device_name": "CPU",
        "torch_version": "2.0",
        "cuda_built": "",
        "loaded": False,
    })
    monkeypatch.setattr(pipeline_mod, "resolve_device", lambda _device: "cpu")

    holder = FileRuntimeGuard(lock_path)
    assert holder.acquire() is True
    try:
        with pytest.raises(RuntimeError, match="Training owns the GPU"):
            generate_image("prompt", seed=1, outputs_dir=tmp_path)
        assert GPU_LEASE_HELD_MESSAGE.startswith("Training owns the GPU")
    finally:
        holder.release()
        monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)


def test_generate_rejected_during_start_fence(monkeypatch, tmp_path: Path):
    """CPU-mocked start-fence rejection; does not run CUDA kernels."""

    monkeypatch.setenv("ZIMAGE_RUNTIME_LOCK", str(tmp_path / "gpu.lease"))
    monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)
    monkeypatch.setattr(pipeline_mod, "runtime_status", lambda: {
        "demo": False,
        "cuda": False,
        "torch": True,
        "device": "cpu",
        "device_name": "CPU",
        "torch_version": "2.0",
        "cuda_built": "",
        "loaded": False,
    })
    monkeypatch.setattr(pipeline_mod, "resolve_device", lambda _device: "cpu")
    pipeline_mod.set_training_start_fence()
    try:
        with pytest.raises(RuntimeError, match="Training owns the GPU"):
            generate_image("prompt", seed=1, outputs_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Training owns the GPU"):
            pipeline_mod.ensure_pipeline("model", "cpu")
        assert GPU_LEASE_HELD_MESSAGE.startswith("Training owns the GPU")
    finally:
        pipeline_mod.clear_training_start_fence()
        monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)


def test_training_start_handoff_unloads_then_starts(tmp_path: Path, monkeypatch):
    """CPU-mocked UI start handoff. The 'cuda' event is a stub callback, not a GPU run."""

    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    order: list = []

    class FakeGuard:
        def acquire(self):
            order.append("acquire")
            return True

        def release(self):
            order.append("release")

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            order.append(("start", job_id))

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: order.append("request_stop"))
    monkeypatch.setattr(handlers, "unload_pipeline", lambda: order.append("unload"))
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: order.append("cuda"))
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", lambda: 4242)

    handlers.start_training_job("job")
    assert order == [
        "request_stop",
        "acquire",
        "release",
        "unload",
        "cuda",
        ("start", "job"),
    ]


def test_immediate_stop_does_not_write_checkpoint(tmp_path: Path, monkeypatch):
    """CPU-mocked UI stop callback; does not run CUDA kernels."""

    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    before = {path.name for path in (root / "checkpoints").iterdir()}
    order: list = []

    class FakeManager:
        def stop(self):
            order.append("stop")

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    handlers.stop_training_job("job")
    assert order == ["stop"]
    assert {path.name for path in (root / "checkpoints").iterdir()} == before
    assert list((root / "checkpoints").iterdir()) == []


def test_job_directory_has_no_metrics_debug_or_top_level_logs(tmp_path: Path):
    root = make_job(tmp_path, max_steps=1)
    assert cache_job(root, **injections()) == 0
    assert run_job(root, **injections()) == 0
    names = _job_names(root)
    assert names <= ALLOWED_JOB_ENTRIES
    assert names >= {
        "config.yaml",
        "state.json",
        "commands",
        "checkpoints",
        "previews",
        "logs",
    }
    forbidden = ("metrics", "debug", "log", "metrics.json", "debug.json")
    assert not (names & set(forbidden))
    assert (root / "logs").is_dir()
    assert not any(child.suffix in {".log", ".metrics"} for child in root.iterdir())
    assert not (root / "logs" / "job.log").exists()


@pytest.mark.skipif(
    os.getenv(HARDWARE_SMOKE_FLAG) != "1",
    reason=f"set {HARDWARE_SMOKE_FLAG}=1 to run the Blackwell hardware smoke test",
)
def test_real_blackwell_fp8_warm_start_turbo_preview_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Run two local-only real-model jobs through checkpoint warm-start.

    Observers wrap production ``setup_main_transformer`` and
    ``official_flow_matching_step``; they call the originals and do not
    inject a fake model, optimizer, writer, sampler, or guard.
    """

    import torch
    import zimage.training.loop as loop_module

    if not torch.cuda.is_available():
        pytest.skip("Blackwell hardware smoke test requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        pytest.skip(
            "Blackwell hardware smoke test requires CUDA compute capability "
            f">= 12.0, got {capability[0]}.{capability[1]}"
        )

    def required_local_snapshot(variable: str) -> Path:
        value = os.getenv(variable)
        if not value:
            pytest.skip(f"missing environment variable: {variable}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            pytest.skip(f"{variable} must be an absolute local snapshot path: {path}")
        path = path.resolve()
        if not path.is_dir():
            pytest.skip(f"{variable} local snapshot path does not exist: {path}")
        return path

    main_model = required_local_snapshot(REAL_MAIN_MODEL)
    sampling_model = required_local_snapshot(REAL_SAMPLING_MODEL)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    dataset_dir = (tmp_path / "dataset").resolve()
    dataset_dir.mkdir()
    image_path = dataset_dir / "sample.png"
    Image.new("RGB", (256, 256), color=(36, 72, 108)).save(image_path)
    image_path.with_suffix(".txt").write_text(
        "тестовый стиль Blackwell",
        encoding="utf-8",
    )
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (256, 256)

    jobs_dir = (tmp_path / "jobs").resolve()
    job_dir = create_or_open_job("blackwell-hardware-smoke", jobs_dir)
    config = job_create_template()
    config.update(
        {
            "job_name": "blackwell-hardware-smoke",
            "main_transformer": {"path": str(main_model), "revision": None},
            "sampling_transformer": {
                "path": str(sampling_model),
                "revision": None,
            },
            "datasets": [
                {
                    "name": str(dataset_dir),
                    "default_caption": "",
                }
            ],
            "lora": {
                "rank": 1,
                "alpha": 1,
                "dropout": 0.0,
                "targets": ["layers.0.attention.to_q"],
            },
            "precision": "fp8",
            "gradient_checkpointing": True,
            "max_steps": 1,
            "checkpoint_every": 1,
            "max_sequence_length": 64,
            "sampling": {
                "common_parameters": {
                    "num_inference_steps": 1,
                    "guidance_scale": 0.0,
                    "time_shift": 3.0,
                    "width": 256,
                    "height": 256,
                    "seed": 12345,
                    "prompt": "a small blue geometric sculpture",
                    "negative_prompt": "",
                },
                "samples": [
                    {"prompt": "a small blue geometric sculpture"},
                ],
            },
        }
    )
    save_job_config(job_dir, config)

    lock_path = (tmp_path / "blackwell-smoke.gpu.lease").resolve()
    first_guard: FileRuntimeGuard | None = None
    second_guard: FileRuntimeGuard | None = None
    probe_guard: FileRuntimeGuard | None = None
    observations = {"setups": 0, "steps": 0}
    real_setup = loop_module.setup_main_transformer
    real_step = loop_module.official_flow_matching_step

    def observe_setup(transformer, **kwargs):
        setup = real_setup(transformer, **kwargs)
        _assert_observed_setup(setup, lora_targets=list(kwargs["lora"]["targets"]))
        observations["setups"] += 1
        return setup

    def observe_step(*args, **kwargs):
        result = real_step(*args, **kwargs)
        _assert_observed_flow_result(result)
        observations["steps"] += 1
        return result

    monkeypatch.setattr(loop_module, "setup_main_transformer", observe_setup)
    monkeypatch.setattr(loop_module, "official_flow_matching_step", observe_step)

    def real_backend(root: Path) -> int:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        code = run_job(root, datasets_dir=dataset_dir, device="cuda")
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        assert peak > _ONE_GIB, (
            f"expected a meaningful CUDA peak allocation (>1 GiB), got {peak} bytes"
        )
        return code

    try:
        first_guard = FileRuntimeGuard(lock_path)
        first_controller = JobController(first_guard, run_backend=real_backend)
        assert first_controller.run(job_dir) == 0
        assert not first_guard.is_held()
        assert observations["setups"] == 1
        assert observations["steps"] >= 1

        first_state = load_job_state(job_dir)
        assert first_state.status is JobStatus.COMPLETED
        assert first_state.step == 1
        first_checkpoint = job_dir / "checkpoints" / "step-1"
        assert validate_checkpoint_directory(first_checkpoint) == first_checkpoint
        first_loaded = load_lora_state(first_checkpoint)
        assert first_loaded.metadata.optimizer_step == 1
        assert first_loaded.state_dict
        assert load_latest_lora_state(job_dir).path == first_checkpoint
        assert (job_dir / "previews" / "step-1" / "00.png").is_file()
        assert not any(
            path.name.casefold().startswith("optimizer")
            for path in (job_dir / "checkpoints").rglob("*")
        )

        probe_guard = FileRuntimeGuard(lock_path)
        assert probe_guard.acquire()
        probe_guard.release()
        assert not probe_guard.is_held()

        updated = load_job_config(job_dir)
        updated["max_steps"] = 2
        save_job_config(job_dir, updated)

        second_guard = FileRuntimeGuard(lock_path)
        second_controller = JobController(second_guard, run_backend=real_backend)
        assert second_controller.run(job_dir) == 0
        assert not second_guard.is_held()
        assert observations["setups"] == 2
        assert observations["steps"] >= 2

        second_state = load_job_state(job_dir)
        assert second_state.status is JobStatus.COMPLETED
        assert second_state.step == 2
        second_checkpoint = job_dir / "checkpoints" / "step-2"
        assert validate_checkpoint_directory(second_checkpoint) == second_checkpoint
        second_loaded = load_lora_state(second_checkpoint)
        assert second_loaded.metadata.optimizer_step == 2
        assert second_loaded.state_dict
        latest = load_latest_lora_state(job_dir)
        assert latest is not None
        assert latest.path == second_checkpoint
        assert latest.metadata.optimizer_step == 2
        assert (job_dir / "previews" / "step-2" / "00.png").is_file()
        assert not any(
            path.name.casefold().startswith("optimizer")
            for path in (job_dir / "checkpoints").rglob("*")
        )
        assert _job_names(job_dir) == ALLOWED_JOB_ENTRIES

        probe_guard = FileRuntimeGuard(lock_path)
        assert probe_guard.acquire()
        probe_guard.release()
        assert not probe_guard.is_held()
    finally:
        for guard in (probe_guard, second_guard, first_guard):
            if guard is not None:
                guard.release()
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
