from __future__ import annotations

import gc
import logging
import subprocess
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from zimage.training.commands import consume_commands, enqueue_update
from zimage.training.contracts import (
    JobState,
    JobStatus,
    NativeAdapterMetadata,
    SavedCheckpoint,
)
from zimage.training.jobs import (
    create_or_open_job,
    load_job_config,
    load_job_state,
    save_job_config,
    write_job_state,
)
from zimage.training.checkpoints import NativeLoraCheckpointWriter
from zimage.training.loop import (
    _construct_accelerator,
    _move_optimizer_state_tensors,
    _resolve_training_device,
    _validate_prepared_training_runtime,
    _write_checkpoint_then_sample,
    cache_job,
    cache_config_from_components,
    get_scheduler_sigmas,
    official_flow_matching_step,
    pack_zimage_hidden_states,
    run_job,
)
from zimage.training.schema import TrainingConfigError
from zimage.training.modeling import (
    ComponentLoaders,
    ModelSource,
    ModelSources,
    TrainingModelComponents,
    TrainingModelLifecycle,
)


class PassthroughAccelerator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def prepare(self, *args):
        return args

    def accumulate(self, *models):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model


class Factory:
    def __init__(self, factory) -> None:
        self._factory = factory
        self.created = []

    def from_pretrained(self, *args, **kwargs):
        obj = self._factory()
        self.created.append(obj)
        return obj


class FakeDistribution:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent


class FakeVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = {"shift_factor": 0.0, "scaling_factor": 1.0}
        self.revision = "fake-revision"
        self.dtype = torch.bfloat16
        self.moved_to: list[object] = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def encode(self, pixels):
        _batch, _channels, height, width = pixels.shape
        latent = torch.ones(
            (1, 16, height // 8, width // 8),
            dtype=torch.bfloat16,
        )
        return SimpleNamespace(latent_dist=FakeDistribution(latent))


class FakeTokenizer:
    def __init__(self) -> None:
        self.config = {"padding_side": "left"}

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )


class FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = {"model_type": "qwen3"}
        self.moved_to: list[object] = []

    def forward(self, **kwargs):
        batch, length = kwargs["input_ids"].shape
        hidden = torch.ones((batch, length, 2560), dtype=torch.float32)
        return SimpleNamespace(hidden_states=[hidden, hidden, hidden])

    def to(self, *args, **kwargs):
        self.moved_to.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)


class FakeScheduler:
    def __init__(self, num_train_timesteps: int = 4) -> None:
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.timesteps = torch.linspace(1000.0, 250.0, num_train_timesteps)
        self.sigmas = torch.linspace(1.0, 0.25, num_train_timesteps)


class FakeTransformer(torch.nn.Module):
    def __init__(self, events: list | None = None, *, trainable: bool = True) -> None:
        super().__init__()
        self.events = events if events is not None else []
        self.calls: list[dict] = []
        self.revision = "fake-revision"
        if trainable:
            self.lora_scale = torch.nn.Parameter(torch.tensor(0.25))

    def enable_gradient_checkpointing(self) -> None:
        self.events.append("checkpoint")

    def add_adapter(self, config, adapter_name="default") -> None:
        self.events.append(("adapter", adapter_name, config))
        self.peft_config = {adapter_name: config}
        if not hasattr(self, "lora_scale"):
            self.lora_scale = torch.nn.Parameter(torch.tensor(0.25))
        else:
            self.lora_scale.requires_grad_(True)

    def forward(self, hidden_states, timestep=None, cap_feats=None, return_dict=False, **kwargs):
        self.events.append("forward")
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "cap_feats": cap_feats,
            }
        )
        preds = []
        scale = self.lora_scale.reshape(*([1] * hidden_states[0].ndim))
        for hidden in hidden_states:
            preds.append(torch.ones_like(hidden) * 0 + scale)
        return (preds,)


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, params, events: list, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.events = events

    def step(self, closure=None):
        self.events.append(("step", float(self.param_groups[0]["lr"])))
        return super().step(closure)


class RecordingWriter:
    def __init__(self, events: list) -> None:
        self.events = events
        self.saved: list[SavedCheckpoint] = []

    def write_atomic(self, *, destination, lora_state, metadata):
        self.events.append("write")
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / "kept.txt").write_text("checkpoint", encoding="utf-8")
        saved = SavedCheckpoint(path, metadata)
        self.saved.append(saved)
        return saved


class RecordingSampler:
    def __init__(self, events: list, *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.calls: list[tuple] = []

    def sample_unfused(self, *, checkpoint, parameters, destination):
        self.events.append("sample")
        path = Path(destination)
        self.calls.append((checkpoint, dict(parameters), path))
        if self.fail:
            raise RuntimeError("preview failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preview", encoding="utf-8")
        return path


class CleanupRecordingSampler(RecordingSampler):
    def __init__(
        self,
        events: list,
        *,
        fail: bool = False,
        main_transformer=None,
    ) -> None:
        super().__init__(events, fail=fail)
        self.main_transformer = main_transformer

    def sample_unfused(self, *, checkpoint, parameters, destination):
        self.events.append("sample")
        try:
            assert self.main_transformer is not None
            assert self.main_transformer.residency == "cpu"
            if self.fail:
                raise RuntimeError("preview failed")
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("preview", encoding="utf-8")
            self.calls.append((checkpoint, dict(parameters), path))
            return path
        finally:
            self.events.append("sampler_cleanup")


class RecordingHook:
    def __init__(self, events: list) -> None:
        self.events = events
        self.steps: list[int] = []

    def on_optimizer_step(self, boundary) -> None:
        self.events.append("hook")
        self.steps.append(boundary.state.step)


def make_loaders(events: list) -> ComponentLoaders:
    return ComponentLoaders(
        vae=Factory(FakeVae),
        tokenizer=Factory(FakeTokenizer),
        text_encoder=Factory(FakeTextEncoder),
        transformer=Factory(lambda: FakeTransformer(events, trainable=False)),
        scheduler=Factory(FakeScheduler),
    )


def make_dataset(tmp_path: Path, name: str = "dataset") -> Path:
    dataset = tmp_path / name
    dataset.mkdir()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(dataset / "a.png")
    (dataset / "a.txt").write_text("a caption", encoding="utf-8")
    return dataset.resolve()


def make_job(tmp_path: Path, **overrides) -> Path:
    dataset = make_dataset(tmp_path)
    root = create_or_open_job("job", tmp_path / "jobs")
    config = load_job_config(root)
    config["datasets"] = [{"name": str(dataset), "default_caption": "fallback"}]
    config["max_steps"] = 2
    config["epochs"] = 50
    config["precision"] = "bf16"
    config["gradient_checkpointing"] = False
    config["checkpoint_every"] = 100
    config.update(overrides)
    save_job_config(root, config)
    return root


def injections(events: list | None = None, **extra):
    events = events if events is not None else []
    payload = {
        "loaders": make_loaders(events),
        "device": "cpu",
        "fp8_capable": False,
        "accelerator": PassthroughAccelerator(),
        "lora_config_factory": lambda **kwargs: SimpleNamespace(**kwargs),
        "optimizer_factory": lambda params, **kwargs: RecordingAdamW(
            params, events, **kwargs
        ),
        "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
        "set_lora_state": lambda _model, _state: None,
        "training_hook": RecordingHook(events),
        "checkpoint_writer": None,
        "preview_sampler": None,
        "load_latest_adapter": None,
    }
    payload.update(extra)
    return payload


def default_sampler_injections(events: list | None = None, **extra):
    payload = injections(events, **extra)
    del payload["preview_sampler"]
    return payload


def default_loader_injections(events: list | None = None, **extra):
    """Keep production ``_default_load_latest_adapter``; do not disable it."""

    payload = injections(events, **extra)
    del payload["load_latest_adapter"]
    return payload


def _optimizer_files(job_dir: Path) -> list[Path]:
    root = Path(job_dir) / "checkpoints"
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold().startswith("optimizer")
    ]


class FakeDefaultSampler:
    last = None
    events: list | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        self.used_factory = False
        self.prompt_paths = dict(kwargs.get("prompt_paths") or {})
        self.negative_prompt_paths = dict(kwargs.get("negative_prompt_paths") or {})
        self.common_parameters = dict(kwargs.get("common_parameters") or {})
        FakeDefaultSampler.last = self

    @classmethod
    def from_components(cls, **kwargs):
        inst = cls(**kwargs)
        inst.used_factory = True
        return inst

    def sample_unfused(self, *, checkpoint, parameters, destination):
        if FakeDefaultSampler.events is not None:
            FakeDefaultSampler.events.append("sample")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preview", encoding="utf-8")
        self.calls.append((checkpoint, dict(parameters), path))
        return path


def _install_fake_default_sampler(monkeypatch, sampler_cls=FakeDefaultSampler):
    import zimage.training.loop as loop_module

    FakeDefaultSampler.last = None
    FakeDefaultSampler.events = None
    real_impl = loop_module._training_impl

    def fake_impl(leaf, name):
        if leaf == "sampling" and name == "UnfusedPreviewSampler":
            return sampler_cls
        return real_impl(leaf, name)

    monkeypatch.setattr(loop_module, "_training_impl", fake_impl)
    return loop_module


def test_official_density_weighting_noising_5d_input_negate_and_fp32_loss():
    latent = torch.ones(16, 2, 2)
    noise = torch.full((1, 16, 2, 2), 2.0)
    scheduler = FakeScheduler(num_train_timesteps=1)
    scheduler.timesteps = torch.tensor([500.0])
    scheduler.sigmas = torch.tensor([0.5])
    transformer = FakeTransformer()
    prompt = torch.ones(3, 2560)

    result = official_flow_matching_step(
        transformer=transformer,
        scheduler=scheduler,
        latent=latent,
        prompt_embedding=prompt,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        noise=noise,
    )

    packed = result.packed_inputs
    assert isinstance(packed, list) and len(packed) == 1
    assert packed[0].shape == (16, 1, 2, 2)
    assert torch.allclose(packed[0][:, 0], torch.full((16, 2, 2), 1.5))
    assert torch.allclose(result.noisy_latent, torch.full((1, 16, 2, 2), 1.5))
    assert torch.allclose(result.timestep_normalized, torch.tensor([0.5]))
    assert transformer.calls[0]["hidden_states"][0].shape == (16, 1, 2, 2)
    assert torch.allclose(transformer.calls[0]["timestep"], torch.tensor([0.5]))
    assert torch.allclose(result.model_pred, torch.full((1, 16, 2, 2), -0.25))
    assert torch.allclose(result.target, torch.ones(1, 16, 2, 2))
    assert torch.allclose(result.weighting, torch.ones_like(result.sigmas))
    assert result.loss.dtype == torch.float32
    assert torch.allclose(result.loss, torch.tensor(1.5625))
    assert pack_zimage_hidden_states(torch.zeros(1, 16, 2, 2))[0].shape == (16, 1, 2, 2)


def test_get_scheduler_sigmas_equality_index_and_broadcast():
    scheduler = FakeScheduler(num_train_timesteps=5)
    # Timestep values are valid-looking raw indices (2, 7) that must not be
    # used as sigma subscripts. Equality mapping is the official contract.
    scheduler.timesteps = torch.tensor([1000.0, 2.0, 500.0, 7.0, 1.0])
    scheduler.sigmas = torch.tensor([1.0, 0.9, 0.5, 0.3, 0.1])
    requested = torch.tensor([7.0, 2.0, 500.0])

    got = get_scheduler_sigmas(
        scheduler, requested, n_dim=4, dtype=torch.float32
    )

    expected = torch.tensor([0.3, 0.9, 0.5]).reshape(3, 1, 1, 1)
    assert got.shape == (3, 1, 1, 1)
    assert got.dtype == torch.float32
    assert torch.equal(got, expected)

    # Timestep 2.0 is schedule index 1 (sigma 0.9). Raw-value indexing
    # would take sigmas[2] == 0.5, or a single-timestep impl would not
    # return three broadcast rows.
    assert torch.equal(scheduler.sigmas[int(requested[1].item())], scheduler.sigmas[2])
    assert torch.equal(got[1, 0, 0, 0], scheduler.sigmas[1].to(dtype=got.dtype))
    assert not torch.equal(got[1, 0, 0, 0], scheduler.sigmas[2].to(dtype=got.dtype))
    assert got.shape[0] == requested.shape[0] == 3

    got_5d = get_scheduler_sigmas(
        scheduler, requested, n_dim=5, dtype=torch.float32
    )
    assert got_5d.shape == (3, 1, 1, 1, 1)
    assert torch.equal(got_5d.flatten(), expected.flatten())


def test_run_job_forwards_official_weighting_fields(monkeypatch, tmp_path):
    recorded = []

    def density(**kwargs):
        recorded.append(("density", dict(kwargs)))
        return torch.zeros(kwargs["batch_size"])

    def weighting(**kwargs):
        recorded.append(("weighting", dict(kwargs)))
        return torch.ones_like(kwargs["sigmas"])

    monkeypatch.setattr(
        "zimage.training.loop.compute_density_for_timestep_sampling", density
    )
    monkeypatch.setattr(
        "zimage.training.loop.compute_loss_weighting_for_sd3", weighting
    )
    root = make_job(
        tmp_path,
        max_steps=1,
        weighting_scheme="logit_normal",
        logit_mean=0.5,
        logit_std=0.25,
        mode_scale=1.5,
    )

    assert run_job(root, **injections()) == 0
    density_kwargs = next(item[1] for item in recorded if item[0] == "density")
    weighting_kwargs = next(item[1] for item in recorded if item[0] == "weighting")
    assert density_kwargs["weighting_scheme"] == "logit_normal"
    assert density_kwargs["logit_mean"] == 0.5
    assert density_kwargs["logit_std"] == 0.25
    assert density_kwargs["mode_scale"] == 1.5
    assert density_kwargs["batch_size"] == 1
    assert weighting_kwargs["weighting_scheme"] == "logit_normal"
    assert weighting_kwargs["sigmas"] is not None


def test_max_steps_wins_over_epochs_and_is_absolute(tmp_path):
    root = make_job(tmp_path, max_steps=4, epochs=100)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=2, epoch=0))

    assert run_job(root, **injections()) == 0

    state = load_job_state(root)
    assert state.step == 4
    # Loop-level RUNNING is expected: JobController owns terminal COMPLETED.
    assert state.status is JobStatus.RUNNING
    assert state.status is not JobStatus.COMPLETED


def test_default_warm_start_applies_latest_lora_before_fresh_optimizer(tmp_path):
    events: list = []
    applied: dict[str, torch.Tensor] = {}
    created: list[RecordingAdamW] = []
    initial_states: list[dict] = []
    scale_at_step: list[float] = []
    marker = 7.5
    lora_state = {
        "to_q.lora_A.weight": torch.full((2, 16), marker),
        "to_q.lora_B.weight": torch.full((16, 2), -1.25),
    }

    root = make_job(tmp_path, max_steps=4)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=3, epoch=0))
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / "step-3",
        lora_state=lora_state,
        metadata=NativeAdapterMetadata(
            adapter_name="default",
            base_model_name_or_path=str(
                load_job_config(root)["model"]["main_transformer"]["path"]
            ),
            base_model_revision=load_job_config(root)["model"]["main_transformer"].get(
                "revision"
            ),
            peft_config={
                "r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "peft_type": "LORA",
            },
            optimizer_step=3,
        ),
    )
    assert _optimizer_files(root) == []

    def set_lora_state(model, state_dict):
        events.append("apply")
        applied.update(
            {key: tensor.detach().cpu().clone() for key, tensor in state_dict.items()}
        )
        model.lora_scale.data.fill_(marker)

    class WarmStartAdamW(RecordingAdamW):
        def __init__(self, params, **kwargs) -> None:
            super().__init__(params, events, **kwargs)
            created.append(self)
            initial_states.append(dict(self.state))
            events.append(("optimizer_created", len(self.state)))

        def step(self, closure=None):
            param = next(iter(self.param_groups[0]["params"]))
            scale_at_step.append(float(param.detach()))
            return super().step(closure)

    payload = default_loader_injections(
        events,
        set_lora_state=set_lora_state,
        optimizer_factory=lambda params, **kwargs: WarmStartAdamW(params, **kwargs),
    )
    assert "load_latest_adapter" not in payload

    assert run_job(root, **payload) == 0

    apply_at = events.index("apply")
    created_at = next(
        index for index, item in enumerate(events) if item[0] == "optimizer_created"
    )
    step_at = next(index for index, item in enumerate(events) if item[0] == "step")
    assert apply_at < created_at < step_at
    assert applied
    assert any("lora" in key.lower() for key in applied)
    loaded_a = next(
        tensor
        for key, tensor in applied.items()
        if key.endswith("to_q.lora_A.weight")
    )
    assert torch.allclose(loaded_a, lora_state["to_q.lora_A.weight"])
    assert len(created) == 1
    assert initial_states == [{}]
    assert scale_at_step == [marker]
    assert load_job_state(root).step == 4
    assert _optimizer_files(root) == []


def test_commands_are_polled_only_after_optimizer_step(tmp_path):
    events: list = []
    consumed = {"n": 0}

    def consume(job_dir, handler=None):
        consumed["n"] += 1
        events.append("consume")
        return consume_commands(job_dir, handler)

    root = make_job(tmp_path, max_steps=2)
    assert run_job(root, **injections(events, consume_commands=consume)) == 0

    step_indexes = [index for index, item in enumerate(events) if item[0] == "step"]
    consume_indexes = [index for index, item in enumerate(events) if item == "consume"]
    forward_indexes = [index for index, item in enumerate(events) if item == "forward"]
    assert len(step_indexes) == 2
    assert len(consume_indexes) == 2
    assert len(forward_indexes) == 2
    for consume_at, step_at, forward_at in zip(
        consume_indexes, step_indexes, forward_indexes
    ):
        assert forward_at < step_at < consume_at
    assert consume_indexes[0] < forward_indexes[1]


def test_apply_at_step_updates_optimizer_learning_rate(tmp_path):
    events: list = []
    root = make_job(tmp_path, max_steps=2)
    updated = load_job_config(root)
    updated["optimizer"]["learning_rate"] = 5.0e-5
    enqueue_update(root, updated)

    assert run_job(root, **injections(events)) == 0

    step_lrs = [item[1] for item in events if item[0] == "step"]
    assert step_lrs == [1.0e-4, 5.0e-5]
    assert load_job_config(root)["optimizer"]["learning_rate"] == 5.0e-5


def test_rebuild_required_tears_down_and_rebuilds(monkeypatch, tmp_path):
    loads = {"n": 0}
    setups = {"n": 0}
    accelerators: list[object] = []
    import zimage.training.loop as loop_module

    real_load = loop_module.load_training_components
    real_setup = loop_module.setup_main_transformer

    def counting_load(*args, **kwargs):
        loads["n"] += 1
        return real_load(*args, **kwargs)

    def counting_setup(*args, **kwargs):
        setups["n"] += 1
        return real_setup(*args, **kwargs)

    def accel_factory(**kwargs):
        accel = PassthroughAccelerator("cpu")
        accelerators.append(accel)
        return accel

    monkeypatch.setattr(loop_module, "load_training_components", counting_load)
    monkeypatch.setattr(loop_module, "setup_main_transformer", counting_setup)

    root = make_job(tmp_path, max_steps=2)
    updated = load_job_config(root)
    updated["max_sequence_length"] = 256
    enqueue_update(root, updated)

    old_runtime: dict = {}

    real_rebuild = loop_module._rebuild_runtime

    def wrapping_rebuild(job_dir, runtime, injected):
        old_runtime["accel"] = runtime.get("accelerator")
        old_runtime["sampler_before"] = runtime.get("preview_sampler")
        rebuilt = real_rebuild(job_dir, runtime, injected)
        old_runtime["sampler_after"] = runtime.get("preview_sampler")
        old_runtime["new_accel"] = rebuilt.get("accelerator")
        return rebuilt

    monkeypatch.setattr(loop_module, "_rebuild_runtime", wrapping_rebuild)

    payload = injections(accelerator_factory=accel_factory)
    # Drop prebuilt accelerator so rebuild factory path is exercised for both builds.
    del payload["accelerator"]
    assert run_job(root, **payload) == 0
    assert loads["n"] == 2
    assert setups["n"] == 2
    assert load_job_config(root)["max_sequence_length"] == 256
    assert old_runtime["accel"] is not None
    assert old_runtime["new_accel"] is not None
    assert old_runtime["accel"] is not old_runtime["new_accel"]
    assert old_runtime["sampler_after"] is None
    assert len(accelerators) >= 2


def test_apply_adapter_state_strips_prefixes_and_rejects_key_mismatch(monkeypatch):
    import zimage.training.loop as loop_module

    applied: dict[str, object] = {}

    class Result:
        unexpected_keys = []
        missing_keys = ["base_model.model.to_q.weight"]

    def fake_set(model, state_dict, adapter_name="default"):
        applied.update(state_dict)
        return Result()

    monkeypatch.setattr(
        "peft.set_peft_model_state_dict",
        fake_set,
        raising=False,
    )
    # Import path used inside _apply_adapter_state
    import peft

    monkeypatch.setattr(peft, "set_peft_model_state_dict", fake_set)

    transformer = FakeTransformer()
    transformer.peft_config = {"default": object()}
    state = {
        "transformer.base_model.model.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.base_model.model.to_q.lora_B.weight": torch.ones(4, 2),
    }
    loop_module._apply_adapter_state(transformer, state, "default", {})
    assert "to_q.lora_A.weight" in applied
    assert "transformer.base_model.model.to_q.lora_A.weight" not in applied

    class BadResult:
        unexpected_keys = ["extra.lora_A.weight"]
        missing_keys = ["to_q.lora_A.weight"]

    monkeypatch.setattr(peft, "set_peft_model_state_dict", lambda *a, **k: BadResult())
    with pytest.raises(TrainingConfigError, match="key mismatch"):
        loop_module._apply_adapter_state(transformer, state, "default", {})


def test_apply_adapter_state_requires_peft_config():
    import zimage.training.loop as loop_module

    transformer = FakeTransformer()
    with pytest.raises(TrainingConfigError, match="no peft_config"):
        loop_module._apply_adapter_state(
            transformer,
            {"to_q.lora_A.weight": torch.ones(1, 1)},
            "default",
            {},
        )


def test_tiny_peft_native_warm_start_applies_exact_lora_tensors(tmp_path):
    """Native writer → default loader → production ``_apply_adapter_state``."""
    from peft import get_peft_model_state_dict

    import zimage.training.loop as loop_module
    from zimage.training.sampling import _peft_compatible_state
    from tests.zimage.training.test_sampling import (
        TinyTransformer,
        _filled_state,
        _lora_config,
        _peft_mapping,
    )

    adapter_name = "default"
    config = _lora_config()
    donor = TinyTransformer().to(dtype=torch.bfloat16)
    donor.add_adapter(config, adapter_name=adapter_name)
    written = _filled_state(donor, adapter_name, sign=1.0)

    job_dir = tmp_path / "job"
    NativeLoraCheckpointWriter().write_atomic(
        destination=job_dir / "checkpoints" / "step-1",
        lora_state=written,
        metadata=NativeAdapterMetadata(
            adapter_name=adapter_name,
            base_model_name_or_path="org/z-image",
            base_model_revision=None,
            peft_config=_peft_mapping(config),
            optimizer_step=1,
        ),
    )

    loaded = loop_module._default_load_latest_adapter(job_dir)
    assert loaded is not None

    recipient = TinyTransformer().to(dtype=torch.bfloat16)
    recipient.add_adapter(_lora_config(), adapter_name=adapter_name)
    loop_module._apply_adapter_state(
        recipient, loaded.state_dict, adapter_name, {}
    )

    expected = _peft_compatible_state(written)
    applied = _peft_compatible_state(
        get_peft_model_state_dict(recipient, adapter_name=adapter_name)
    )
    assert set(applied) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(applied[key].detach().cpu(), tensor.detach().cpu())


def test_warm_start_rejects_rank_alpha_metadata_mismatch(tmp_path):
    root = make_job(tmp_path, max_steps=1)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=1, epoch=0))
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / "step-1",
        lora_state={
            "to_q.lora_A.weight": torch.ones(2, 16),
            "to_q.lora_B.weight": torch.ones(16, 2),
        },
        metadata=NativeAdapterMetadata(
            adapter_name="default",
            base_model_name_or_path="org/z-image",
            base_model_revision="abc123",
            peft_config={
                "r": 99,
                "lora_alpha": 99,
                "lora_dropout": 0.0,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "peft_type": "LORA",
            },
            optimizer_step=1,
        ),
    )
    applied = []

    def set_lora_state(model, state_dict):
        applied.append(state_dict)

    with pytest.raises(TrainingConfigError, match="warm-start|rank"):
        run_job(
            root,
            **default_loader_injections(set_lora_state=set_lora_state),
        )
    assert applied == []


def test_default_warm_start_without_set_lora_state_loads_native_weights(
    monkeypatch, tmp_path
):
    """Production path: no set_lora_state injection; strip Diffusers prefixes."""
    import peft
    import zimage.training.loop as loop_module

    marker = 3.25
    lora_state = {
        "transformer.to_q.lora_A.weight": torch.full((2, 16), marker),
        "transformer.to_q.lora_B.weight": torch.full((16, 2), -0.5),
    }
    root = make_job(tmp_path, max_steps=1)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=1, epoch=0))
    config = load_job_config(root)
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / "step-1",
        lora_state={
            "to_q.lora_A.weight": torch.full((2, 16), marker),
            "to_q.lora_B.weight": torch.full((16, 2), -0.5),
        },
        metadata=NativeAdapterMetadata(
            adapter_name="default",
            base_model_name_or_path=str(config["model"]["main_transformer"]["path"]),
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

    applied: dict[str, torch.Tensor] = {}

    class Ok:
        unexpected_keys = []
        missing_keys = []

    def fake_set(model, state_dict, adapter_name="default"):
        applied.update(
            {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        )
        if hasattr(model, "lora_scale"):
            model.lora_scale.data.fill_(marker)
        return Ok()

    monkeypatch.setattr(peft, "set_peft_model_state_dict", fake_set)

    real_setup = loop_module.setup_main_transformer

    def setup_with_peft(*args, **kwargs):
        result = real_setup(*args, **kwargs)
        result.transformer.peft_config = {"default": object()}
        return result

    monkeypatch.setattr(loop_module, "setup_main_transformer", setup_with_peft)

    payload = default_loader_injections()
    del payload["set_lora_state"]
    assert "set_lora_state" not in payload
    assert run_job(root, **payload) == 0
    assert "to_q.lora_A.weight" in applied
    assert torch.allclose(applied["to_q.lora_A.weight"], torch.full((2, 16), marker))


def test_rebuild_export_failure_does_not_warm_start_older_checkpoint(
    monkeypatch, tmp_path
):
    import zimage.training.loop as loop_module

    root = make_job(tmp_path, max_steps=2)
    updated = load_job_config(root)
    updated["max_sequence_length"] = 256
    enqueue_update(root, updated)

    warm_calls = []
    build_count = {"n": 0}
    real_build = loop_module._build_runtime

    def counting_build(*args, **kwargs):
        build_count["n"] += 1
        return real_build(*args, **kwargs)

    def boom_lora_state(*args, **kwargs):
        raise RuntimeError("export boom")

    def track_warm(*args, **kwargs):
        warm_calls.append(build_count["n"])
        return None

    monkeypatch.setattr(loop_module, "_build_runtime", counting_build)
    monkeypatch.setattr(loop_module, "_lora_state", boom_lora_state)
    monkeypatch.setattr(loop_module, "_maybe_warm_start", track_warm)

    assert run_job(root, **injections()) == 1
    # Warm-start must not run on a rebuild fallback path (only initial build).
    assert warm_calls == [1]
    err = load_job_state(root).last_error or ""
    assert "export" in err.lower() or "rebuild" in err.lower()
    assert build_count["n"] == 1


def test_lora_state_has_no_name_matching_fallback():
    source = Path(__file__).resolve().parents[3] / "zimage" / "training" / "loop.py"
    text = source.read_text(encoding="utf-8")
    assert 'except Exception:\n        preserved = None' not in text
    assert '"lora" in name.lower()' not in text
    assert "preserved = None" not in text.split("def _rebuild_runtime")[1].split(
        "def _teardown_runtime"
    )[0]


def test_immutable_main_and_lora_topology_are_rejected(tmp_path):
    root = make_job(tmp_path, max_steps=2)
    updated = load_job_config(root)
    updated["lora"]["rank"] = 8
    enqueue_update(root, updated)

    assert run_job(root, **injections()) == 0

    persisted = load_job_config(root)
    assert persisted["lora"]["rank"] == 8
    state = load_job_state(root)
    assert state.status is JobStatus.RUNNING
    assert state.step == 2
    assert state.last_error is None or "rejected immutable" not in state.last_error
    assert not list((root / "commands").glob("*.json"))


def test_checkpoint_writer_then_preview_sampler_order(tmp_path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events)
    root = make_job(
        tmp_path,
        max_steps=1,
        checkpoint_every=1,
        sampling={
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "time_shift": 3.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "prompt": "shared",
            "negative_prompt": "",
            "samples": [{"prompt": "one"}, {"prompt": "two", "seed": 7}],
        },
    )

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
    assert events.count("sample") == 2
    assert events.index("write") < events.index("sample")
    assert writer.saved[0].metadata.optimizer_step == 1
    assert isinstance(writer.saved[0].metadata, NativeAdapterMetadata)
    assert [call[1]["prompt"] for call in sampler.calls] == ["one", "two"]
    assert sampler.calls[1][1]["seed"] == 7
    assert sampler.calls[0][0] is writer.saved[0]
    assert [call[2] for call in sampler.calls] == [
        root / "previews" / "00001-00-sample.jpg",
        root / "previews" / "00001-01-sample.jpg",
    ]
    assert not any(child.is_dir() for child in (root / "previews").iterdir())


def test_preview_paths_follow_sampling_image_format_png(tmp_path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events)
    root = make_job(
        tmp_path,
        max_steps=1,
        checkpoint_every=1,
        sampling={
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "time_shift": 3.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "prompt": "shared",
            "negative_prompt": "",
            "image_format": "png",
            "samples": [{"prompt": "one"}],
        },
    )

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
    assert [call[2] for call in sampler.calls] == [
        root / "previews" / "00001-00-sample.png",
    ]


def test_cuda_checkpoint_preview_handoff_exact_order(tmp_path):
    events: list[str] = []
    writer = RecordingWriter(events)
    transformer = FakeTransformer()
    transformer.residency = "cuda"
    sampler = CleanupRecordingSampler(
        events,
        main_transformer=transformer,
    )
    optimizer = SimpleNamespace(
        state={"parameter": {"momentum": torch.tensor(1.0)}}
    )
    root = make_job(
        tmp_path,
        sampling={
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
            "time_shift": 1.0,
            "width": 16,
            "height": 16,
            "seed": 1,
            "prompt": "one",
            "negative_prompt": "",
            "samples": [{"prompt": "one"}],
        },
    )
    config = load_job_config(root)
    runtime = {
        "config": config,
        "transformer": transformer,
        "optimizer": optimizer,
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
    }
    last_optimizer_device = {"value": None}

    def move_transformer(model, device):
        model.residency = device.type
        events.append(f"main_{device.type}")

    def move_optimizer(tensor, device):
        if last_optimizer_device["value"] != device.type:
            last_optimizer_device["value"] = device.type
            events.append(f"optimizer_{device.type}")
        return tensor

    def synchronize():
        events.append("sync")

    result = _write_checkpoint_then_sample(
        root,
        JobState("job", JobStatus.RUNNING, step=1, epoch=0),
        runtime,
        {
            "device": "cuda",
            "checkpoint_writer": writer,
            "preview_sampler": sampler,
            "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
            "training_transformer_mover": move_transformer,
            "optimizer_tensor_mover": move_optimizer,
            "cuda_synchronize": synchronize,
            "cuda_empty_cache": lambda: events.append("empty_cache"),
            "garbage_collect": lambda: events.append("gc"),
        },
    )

    assert result == 0
    assert events == [
        "write",
        "sync",
        "main_cpu",
        "optimizer_cpu",
        "gc",
        "empty_cache",
        "sample",
        "sampler_cleanup",
        "main_cuda",
        "optimizer_cuda",
        "sync",
    ]
    assert runtime["optimizer"] is optimizer
    assert sampler.main_transformer is transformer


def test_cuda_handoff_wraps_all_previews_once(tmp_path):
    events: list[str] = []
    writer = RecordingWriter(events)
    transformer = FakeTransformer()
    transformer.residency = "cuda"
    sampler = CleanupRecordingSampler(events, main_transformer=transformer)
    optimizer = SimpleNamespace(state={"p": torch.tensor(1.0)})
    root = make_job(
        tmp_path,
        sampling={
            "prompt": "shared",
            "samples": [{"prompt": "one"}, {"prompt": "two"}],
        },
    )
    runtime = {
        "config": load_job_config(root),
        "transformer": transformer,
        "optimizer": optimizer,
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
    }
    moved_optimizer_to: list[str] = []

    def move_transformer(model, device):
        model.residency = device.type
        events.append(f"main_{device.type}")

    def move_optimizer(tensor, device):
        moved_optimizer_to.append(device.type)
        return tensor

    assert (
        _write_checkpoint_then_sample(
            root,
            JobState("job", JobStatus.RUNNING, step=1, epoch=0),
            runtime,
            {
                "device": torch.device("cuda"),
                "checkpoint_writer": writer,
                "preview_sampler": sampler,
                "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
                "training_transformer_mover": move_transformer,
                "optimizer_tensor_mover": move_optimizer,
                "cuda_synchronize": lambda: None,
                "cuda_empty_cache": lambda: None,
                "garbage_collect": lambda: None,
            },
        )
        == 0
    )
    assert events.count("main_cpu") == 1
    assert events.count("main_cuda") == 1
    assert moved_optimizer_to == ["cpu", "cuda"]
    assert events.count("sample") == 2
    assert events.count("sampler_cleanup") == 2
    assert events.index("main_cpu") < events.index("sample")
    assert events.index("main_cuda") > max(
        index for index, item in enumerate(events) if item == "sampler_cleanup"
    )


def test_cuda_handoff_restores_after_preview_failure_and_keeps_checkpoint(tmp_path):
    events: list[str] = []
    writer = RecordingWriter(events)
    transformer = FakeTransformer()
    transformer.residency = "cuda"
    sampler = CleanupRecordingSampler(
        events,
        fail=True,
        main_transformer=transformer,
    )
    optimizer = SimpleNamespace(state={"p": torch.tensor(1.0)})
    root = make_job(tmp_path)
    runtime = {
        "config": load_job_config(root),
        "transformer": transformer,
        "optimizer": optimizer,
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
    }

    def move_transformer(model, device):
        model.residency = device.type
        events.append(f"main_{device.type}")

    result = _write_checkpoint_then_sample(
        root,
        JobState("job", JobStatus.RUNNING, step=1, epoch=0),
        runtime,
        {
            "device": "cuda",
            "checkpoint_writer": writer,
            "preview_sampler": sampler,
            "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
            "training_transformer_mover": move_transformer,
            "optimizer_tensor_mover": lambda tensor, _device: tensor,
            "cuda_synchronize": lambda: None,
            "cuda_empty_cache": lambda: None,
            "garbage_collect": lambda: None,
        },
    )

    assert result == 1
    assert events.index("sampler_cleanup") < events.index("main_cuda")
    assert transformer.residency == "cuda"
    assert writer.saved[0].path.joinpath("kept.txt").is_file()
    assert load_job_state(root).last_error == "preview failed"


def test_cuda_restore_failure_is_reported_and_optimizer_restore_is_attempted(tmp_path):
    events: list[str] = []
    writer = RecordingWriter(events)
    transformer = FakeTransformer()
    transformer.residency = "cuda"
    sampler = CleanupRecordingSampler(events, main_transformer=transformer)
    optimizer = SimpleNamespace(state={"p": torch.tensor(1.0)})
    root = make_job(tmp_path)
    runtime = {
        "config": load_job_config(root),
        "transformer": transformer,
        "optimizer": optimizer,
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
    }
    optimizer_moves: list[str] = []

    def move_transformer(model, device):
        if device.type == "cuda":
            raise RuntimeError("restore boom")
        model.residency = device.type

    def move_optimizer(tensor, device):
        optimizer_moves.append(device.type)
        return tensor

    result = _write_checkpoint_then_sample(
        root,
        JobState("job", JobStatus.RUNNING, step=1, epoch=0),
        runtime,
        {
            "device": "cuda",
            "checkpoint_writer": writer,
            "preview_sampler": sampler,
            "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
            "training_transformer_mover": move_transformer,
            "optimizer_tensor_mover": move_optimizer,
            "cuda_synchronize": lambda: None,
            "cuda_empty_cache": lambda: None,
            "garbage_collect": lambda: None,
        },
    )

    assert result == 1
    assert optimizer_moves == ["cpu", "cuda"]
    assert writer.saved[0].path.joinpath("kept.txt").is_file()
    error = load_job_state(root).last_error
    assert error is not None
    assert "failed to restore training runtime after previews" in error
    assert "main transformer: restore boom" in error


def test_nested_optimizer_state_tensor_movement():
    tensor_values = {
        "mapping": torch.tensor(1.0),
        "list": torch.tensor(2.0),
        "tuple": torch.tensor(3.0),
    }
    optimizer = SimpleNamespace(
        state={
            "parameter": {
                "mapping": tensor_values["mapping"],
                "containers": [
                    tensor_values["list"],
                    (tensor_values["tuple"], "unchanged"),
                ],
            }
        }
    )
    moved: list[tuple[float, str]] = []

    def move_tensor(tensor, device):
        moved.append((tensor.item(), device.type))
        return tensor + 10

    _move_optimizer_state_tensors(
        optimizer,
        torch.device("cpu"),
        {"optimizer_tensor_mover": move_tensor},
    )

    nested = optimizer.state["parameter"]
    assert moved == [(1.0, "cpu"), (2.0, "cpu"), (3.0, "cpu")]
    assert nested["mapping"].item() == 11.0
    assert nested["containers"][0].item() == 12.0
    assert nested["containers"][1][0].item() == 13.0
    assert nested["containers"][1][1] == "unchanged"


def test_cpu_preview_path_does_not_invoke_cuda_handoff(tmp_path):
    events: list[str] = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events)
    transformer = FakeTransformer()
    root = make_job(tmp_path)
    runtime = {
        "config": load_job_config(root),
        "transformer": transformer,
        "optimizer": SimpleNamespace(state={}),
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU preview invoked CUDA handoff")

    assert (
        _write_checkpoint_then_sample(
            root,
            JobState("job", JobStatus.RUNNING, step=1, epoch=0),
            runtime,
            {
                "device": "cpu",
                "checkpoint_writer": writer,
                "preview_sampler": sampler,
                "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
                "training_transformer_mover": forbidden,
                "optimizer_tensor_mover": forbidden,
                "cuda_synchronize": forbidden,
                "cuda_empty_cache": forbidden,
                "garbage_collect": forbidden,
            },
        )
        == 0
    )
    assert events == ["write", "sample"]


def test_sampling_failure_returns_nonzero_and_keeps_checkpoint(tmp_path):
    events: list = []
    writer = RecordingWriter(events)
    sampler = RecordingSampler(events, fail=True)
    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)

    result = run_job(
        root,
        **injections(
            events,
            checkpoint_writer=writer,
            preview_sampler=sampler,
        ),
    )

    assert result != 0
    assert events.index("write") < events.index("sample")
    kept = writer.saved[0].path / "kept.txt"
    assert kept.is_file()
    assert kept.read_text(encoding="utf-8") == "checkpoint"
    assert load_job_state(root).last_error == "preview failed"


def test_cache_config_from_components_uses_effective_text_encoder_precision():
    job = {
        "max_sequence_length": 32,
        "precision": "fp8",
        "model": {"main_transformer": {"revision": "rev"}},
    }

    def components(text_encoder):
        return TrainingModelComponents(
            sources=ModelSources(ModelSource("main"), ModelSource("main")),
            vae=FakeVae(),
            tokenizer=FakeTokenizer(),
            text_encoder=text_encoder,
            training_scheduler=object(),
            main_transformer=FakeTransformer([], trainable=False),
            sampling_transformer=FakeTransformer([], trainable=False),
            sampling_scheduler=object(),
        )

    unmarked = cache_config_from_components(job, components(FakeTextEncoder()))
    assert unmarked.text_encoder_precision == "bf16"

    marked_encoder = FakeTextEncoder()
    marked_encoder._quantized_precision = "fp8"
    marked = cache_config_from_components(job, components(marked_encoder))
    assert marked.text_encoder_precision == "fp8"


def test_load_lifecycle_passes_quantize_capable_from_fp8_capable(monkeypatch):
    import zimage.training.loop as loop_module

    captured: dict[str, object] = {}
    real_load = loop_module.load_training_components

    def wrapping_load(job, **kwargs):
        captured["quantize_capable"] = kwargs.get("quantize_capable")
        forwarded = dict(kwargs)
        forwarded["quantize_capable"] = False
        return real_load(job, **forwarded)

    monkeypatch.setattr(loop_module, "load_training_components", wrapping_load)
    job = {"model": {"main_transformer": {"path": "org/main", "revision": None}}}
    loaders = make_loaders([])

    injected = {"loaders": loaders, "fp8_capable": True}
    loop_module._load_lifecycle(job, injected)
    assert captured["quantize_capable"] is True
    assert captured["quantize_capable"] == loop_module._fp8_capable(injected)

    injected = {"loaders": loaders, "fp8_capable": False}
    loop_module._load_lifecycle(job, injected)
    assert captured["quantize_capable"] is False
    assert captured["quantize_capable"] == loop_module._fp8_capable(injected)


def test_cache_job_prepares_cache_without_running_optimizer(tmp_path):
    events: list = []
    root = make_job(tmp_path)
    loaders = make_loaders(events)

    assert (
        cache_job(
            root,
            loaders=loaders,
            device="cpu",
            fp8_capable=False,
        )
        == 0
    )

    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    cached = list((root / ".cache" / "dataset").rglob("*.safetensors"))
    assert cached
    previewed = list((root / ".cache" / "preview").rglob("*.safetensors"))
    assert previewed
    assert not (dataset / ".cache").exists()
    assert load_job_state(root).step == 0
    assert "step" not in events
    assert "forward" not in events
    moved = loaders.text_encoder.created[0].moved_to
    assert moved[-1] == "cpu"
    assert "cuda" not in moved


def _add_dataset_image(dataset: Path, name: str, *, color=(40, 50, 60), caption: str) -> None:
    Image.new("RGB", (16, 16), color).save(dataset / name)
    (dataset / Path(name).with_suffix(".txt")).write_text(caption, encoding="utf-8")


def test_runtime_stores_job_cache_paths_without_cached_samples(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module
    from zimage.training.cache import CachedSample

    captured: dict[str, object] = {}
    real_optimize = loop_module._optimize

    def wrapping_optimize(job_dir, state, holder, injected):
        runtime = holder["runtime"]
        captured["has_cached"] = "cached" in runtime
        paths = list(runtime["cache_paths"])
        captured["paths"] = paths
        captured["contains_sample"] = any(isinstance(item, CachedSample) for item in paths)
        cache_root = (Path(job_dir) / ".cache" / "dataset").resolve()
        captured["parents"] = [Path(path).resolve().parent for path in paths]
        captured["cache_root"] = cache_root
        return real_optimize(job_dir, state, holder, injected)

    monkeypatch.setattr(loop_module, "_optimize", wrapping_optimize)
    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections()) == 0
    assert captured["has_cached"] is False
    assert captured["contains_sample"] is False
    assert captured["paths"]
    assert captured["parents"] == [captured["cache_root"]] * len(captured["paths"])


def test_one_cpu_load_per_training_step_and_device_move_keeps_cache_cpu(
    tmp_path, monkeypatch
):
    import zimage.training.loop as loop_module

    loads: list[Path] = []
    real_load = loop_module.load_cache
    real_step = loop_module.official_flow_matching_step

    def tracking_load(path):
        loads.append(Path(path))
        sample = real_load(path)
        assert sample.latent.device.type == "cpu"
        assert sample.prompt_embedding.device.type == "cpu"
        return sample

    def tracking_step(**kwargs):
        result = real_step(**kwargs)
        assert kwargs["latent"].device.type == "cpu"
        assert kwargs["prompt_embedding"].device.type == "cpu"
        return result

    monkeypatch.setattr(loop_module, "load_cache", tracking_load)
    monkeypatch.setattr(loop_module, "official_flow_matching_step", tracking_step)
    root = make_job(tmp_path, max_steps=3)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    _add_dataset_image(dataset, "b.png", caption="b caption")
    assert run_job(root, **injections()) == 0
    assert len(loads) == 3
    reloaded = real_load(loads[0])
    assert reloaded.latent.device.type == "cpu"
    assert reloaded.prompt_embedding.device.type == "cpu"


def test_duplicate_cache_paths_preserve_order_across_epochs_resume_and_rebuild(
    tmp_path, monkeypatch
):
    import zimage.training.loop as loop_module

    captured_paths: list[list[Path]] = []
    loads: list[Path] = []
    real_build = loop_module._build_runtime
    real_load = loop_module.load_cache

    def tracking_build(*args, **kwargs):
        runtime = real_build(*args, **kwargs)
        captured_paths.append([Path(path) for path in runtime["cache_paths"]])
        return runtime

    def tracking_load(path):
        loads.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(loop_module, "_build_runtime", tracking_build)
    monkeypatch.setattr(loop_module, "load_cache", tracking_load)

    root = make_job(tmp_path, max_steps=4)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    _add_dataset_image(dataset, "copy.png", color=(10, 20, 30), caption="a caption")
    updated = load_job_config(root)
    updated["max_sequence_length"] = 256
    enqueue_update(root, updated)

    assert run_job(root, **injections()) == 0
    assert len(captured_paths) == 2
    first, rebuilt = captured_paths
    assert len(first) == 2
    assert first[0] == first[1]
    assert rebuilt == first
    assert loads[:2] == first
    assert load_job_state(root).epoch == 2


def test_resume_uses_step_modulo_cache_paths(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    loads: list[Path] = []
    captured: dict[str, list[Path]] = {}
    real_load = loop_module.load_cache
    real_optimize = loop_module._optimize

    def wrapping_optimize(job_dir, state, holder, injected):
        captured["paths"] = [Path(path) for path in holder["runtime"]["cache_paths"]]
        return real_optimize(job_dir, state, holder, injected)

    def tracking_load(path):
        loads.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(loop_module, "_optimize", wrapping_optimize)
    monkeypatch.setattr(loop_module, "load_cache", tracking_load)

    root = make_job(tmp_path, max_steps=3)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    _add_dataset_image(dataset, "b.png", caption="b caption")
    _write_job_checkpoint(root, step=1)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=2, epoch=0))

    assert run_job(root, **default_loader_injections()) == 0
    paths = captured["paths"]
    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert loads == [paths[1], paths[0]]


def test_sample_and_result_released_before_checkpoint(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    refs: list = []
    real_load = loop_module.load_cache
    real_step = loop_module.official_flow_matching_step
    real_write = loop_module._write_checkpoint_then_sample

    def tracking_load(path):
        sample = real_load(path)
        refs.append(weakref.ref(sample))
        return sample

    def tracking_step(**kwargs):
        result = real_step(**kwargs)
        refs.append(weakref.ref(result))
        return result

    def tracking_write(*args, **kwargs):
        gc.collect()
        assert refs and all(ref() is None for ref in refs)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(loop_module, "load_cache", tracking_load)
    monkeypatch.setattr(loop_module, "official_flow_matching_step", tracking_step)
    monkeypatch.setattr(loop_module, "_write_checkpoint_then_sample", tracking_write)

    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)
    assert run_job(root, **injections()) == 0
    assert refs


def test_sample_and_result_released_before_preview(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    refs: list = []
    real_load = loop_module.load_cache
    real_step = loop_module.official_flow_matching_step

    def tracking_load(path):
        sample = real_load(path)
        refs.append(weakref.ref(sample))
        return sample

    def tracking_step(**kwargs):
        result = real_step(**kwargs)
        refs.append(weakref.ref(result))
        return result

    class ProbeSampler:
        def sample_unfused(self, *, checkpoint, parameters, destination):
            gc.collect()
            assert refs and all(ref() is None for ref in refs)
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("preview", encoding="utf-8")
            return path

    monkeypatch.setattr(loop_module, "load_cache", tracking_load)
    monkeypatch.setattr(loop_module, "official_flow_matching_step", tracking_step)

    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)
    writer = RecordingWriter([])
    assert (
        run_job(
            root,
            **injections(checkpoint_writer=writer, preview_sampler=ProbeSampler()),
        )
        == 0
    )
    assert refs
    assert writer.saved


def test_corrupt_cache_at_use_raises_without_reencoding(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module
    from zimage.training.cache import CacheError

    prepares = {"n": 0}
    real_prepare = loop_module._prepare_cache
    real_build = loop_module._build_runtime

    def counting_prepare(*args, **kwargs):
        prepares["n"] += 1
        return real_prepare(*args, **kwargs)

    def build_then_corrupt(*args, **kwargs):
        runtime = real_build(*args, **kwargs)
        for path in runtime["cache_paths"]:
            Path(path).write_bytes(b"not-a-safetensors-file")
        return runtime

    monkeypatch.setattr(loop_module, "_prepare_cache", counting_prepare)
    monkeypatch.setattr(loop_module, "_build_runtime", build_then_corrupt)

    root = make_job(tmp_path, max_steps=1)
    with pytest.raises(CacheError, match="cannot read cache"):
        run_job(root, **injections())
    assert prepares["n"] == 1


def _track_place_park(monkeypatch, events: list, *, call_real: bool = True):
    import zimage.training.loop as loop_module

    real_place = loop_module.TrainingModelLifecycle.place_cache_modules
    real_park = loop_module.TrainingModelLifecycle.park_cache_modules

    def tracking_place(self, device, *, vae=True):
        events.append(("place", torch.device(device).type))
        if call_real:
            return real_place(self, device, vae=vae)
        return None

    def tracking_park(self):
        events.append("park")
        if call_real:
            return real_park(self)
        return None

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle, "place_cache_modules", tracking_place
    )
    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle, "park_cache_modules", tracking_park
    )
    return loop_module


def test_stale_cache_places_before_encode_and_parks_after(tmp_path, monkeypatch):
    events: list = []
    _track_place_park(monkeypatch, events)

    class TrackingVae(FakeVae):
        def encode(self, pixels):
            events.append("encode")
            return super().encode(pixels)

    loaders = ComponentLoaders(
        vae=Factory(TrackingVae),
        tokenizer=Factory(FakeTokenizer),
        text_encoder=Factory(FakeTextEncoder),
        transformer=Factory(lambda: FakeTransformer([], trainable=False)),
        scheduler=Factory(FakeScheduler),
    )
    root = make_job(tmp_path)

    assert cache_job(root, loaders=loaders, device="cpu", fp8_capable=False) == 0
    assert events == [("place", "cpu"), "encode", "park"]
    moved = loaders.text_encoder.created[0].moved_to
    assert "cuda" not in moved
    assert torch.device(moved[-1]).type == "cpu"


def test_valid_cache_never_places_cache_modules(tmp_path, monkeypatch):
    root = make_job(tmp_path)
    assert cache_job(root, **injections()) == 0

    events: list = []

    def forbid_place(self, device):
        events.append("place")
        raise AssertionError("place_cache_modules must not run for VALID cache")

    def forbid_park(self):
        events.append("park")
        raise AssertionError("park_cache_modules must not run when place was skipped")

    import zimage.training.loop as loop_module

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle, "place_cache_modules", forbid_place
    )
    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle, "park_cache_modules", forbid_park
    )

    assert cache_job(root, **injections()) == 0
    assert events == []


def _recording_gpu_probe(phases: list):
    def probe(phase, components=None):
        phases.append(phase)

    return probe


def test_stale_cache_job_probe_records_place_and_end(tmp_path):
    phases: list[str] = []
    root = make_job(tmp_path)
    assert (
        cache_job(root, **injections(gpu_usage_probe=_recording_gpu_probe(phases)))
        == 0
    )
    assert phases == ["cache_place", "cache_encode", "cache_end"]


def test_stale_cache_job_probe_records_each_sample(tmp_path, caplog):
    root = make_job(tmp_path)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    Image.new("RGB", (48, 32), (1, 2, 3)).save(dataset / "b.png")
    (dataset / "b.txt").write_text("b caption", encoding="utf-8")
    phases: list[str] = []
    with caplog.at_level(logging.INFO, logger="zimage.training"):
        assert (
            cache_job(
                root, **injections(gpu_usage_probe=_recording_gpu_probe(phases))
            )
            == 0
        )
    assert phases == [
        "cache_place",
        "cache_encode",
        "cache_encode",
        "cache_end",
    ]
    encode_lines = [
        record.message
        for record in caplog.records
        if record.message.startswith("cache encode n=")
    ]
    assert len(encode_lines) == 2
    assert "n=1 samples=2" in encode_lines[0]
    assert "a.png" in encode_lines[0]
    assert "size=16x16" in encode_lines[0]
    assert "n=2 samples=2" in encode_lines[1]
    assert "b.png" in encode_lines[1]
    assert "size=48x32" in encode_lines[1]


def test_stale_cache_reclaims_cuda_cache_per_sample(tmp_path):
    root = make_job(tmp_path)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    Image.new("RGB", (48, 32), (1, 2, 3)).save(dataset / "b.png")
    (dataset / "b.txt").write_text("b caption", encoding="utf-8")
    calls = {"gc": 0, "empty_cache": 0}

    assert (
        cache_job(
            root,
            **injections(
                garbage_collect=lambda: calls.__setitem__("gc", calls["gc"] + 1),
                cuda_empty_cache=lambda: calls.__setitem__(
                    "empty_cache", calls["empty_cache"] + 1
                ),
            ),
        )
        == 0
    )
    assert calls == {"gc": 2, "empty_cache": 2}


def test_warm_cache_does_not_reclaim_cuda_cache(tmp_path):
    root = make_job(tmp_path)
    assert cache_job(root, **injections()) == 0
    calls = {"gc": 0, "empty_cache": 0}

    assert (
        cache_job(
            root,
            **injections(
                garbage_collect=lambda: calls.__setitem__("gc", calls["gc"] + 1),
                cuda_empty_cache=lambda: calls.__setitem__(
                    "empty_cache", calls["empty_cache"] + 1
                ),
            ),
        )
        == 0
    )
    assert calls == {"gc": 0, "empty_cache": 0}


def test_warm_run_reclaims_cuda_cache_per_step(tmp_path):
    root = make_job(tmp_path)
    assert cache_job(root, **injections()) == 0
    calls = {"gc": 0, "empty_cache": 0}

    assert (
        run_job(
            root,
            **injections(
                garbage_collect=lambda: calls.__setitem__("gc", calls["gc"] + 1),
                cuda_empty_cache=lambda: calls.__setitem__(
                    "empty_cache", calls["empty_cache"] + 1
                ),
            ),
        )
        == 0
    )
    assert calls == {"gc": 2, "empty_cache": 2}


def test_step_probe_after_collect_before_empty_cache(tmp_path):
    root = make_job(tmp_path, max_steps=1)
    assert cache_job(root, **injections()) == 0
    events: list[str] = []

    def probe(phase, context=None):
        events.append(f"probe:{phase}")

    assert (
        run_job(
            root,
            **injections(
                garbage_collect=lambda: events.append("gc"),
                cuda_empty_cache=lambda: events.append("empty_cache"),
                gpu_usage_probe=probe,
            ),
        )
        == 0
    )
    step_at = events.index("probe:step")
    assert events[step_at - 1 : step_at + 2] == ["gc", "probe:step", "empty_cache"]


def test_cache_encode_probe_after_collect_before_empty_cache(tmp_path):
    events: list[str] = []

    def probe(phase, context=None):
        events.append(f"probe:{phase}")

    assert (
        cache_job(
            make_job(tmp_path),
            **injections(
                garbage_collect=lambda: events.append("gc"),
                cuda_empty_cache=lambda: events.append("empty_cache"),
                gpu_usage_probe=probe,
            ),
        )
        == 0
    )
    encode_at = events.index("probe:cache_encode")
    assert events[encode_at - 1 : encode_at + 2] == [
        "gc",
        "probe:cache_encode",
        "empty_cache",
    ]


def test_cache_job_park_failure_does_not_mask_encode_error(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    def boom_prepare(
        samples,
        encoder,
        config,
        *,
        job_dir,
        on_before_encode=None,
        **kwargs,
    ):
        if on_before_encode is not None:
            on_before_encode()
        raise RuntimeError("encode boom")

    def boom_park(self):
        raise RuntimeError("park boom")

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle, "park_cache_modules", boom_park
    )
    with pytest.raises(RuntimeError, match="encode boom"):
        cache_job(make_job(tmp_path), **injections(prepare_cache=boom_prepare))


def test_stale_run_job_probe_phase_order(tmp_path):
    phases: list[str] = []
    root = make_job(tmp_path, max_steps=1)
    assert (
        run_job(root, **injections(gpu_usage_probe=_recording_gpu_probe(phases)))
        == 0
    )
    assert "step_peak" not in phases
    assert phases == [
        "load",
        "cache_place",
        "cache_encode",
        "cache_end",
        "train_placed",
        "step",
        "teardown",
        "summary",
    ]


def test_warm_cache_run_job_probe_omits_cache_place(tmp_path):
    root = make_job(tmp_path, max_steps=1)
    assert cache_job(root, **injections()) == 0
    phases: list[str] = []
    assert (
        run_job(root, **injections(gpu_usage_probe=_recording_gpu_probe(phases)))
        == 0
    )
    assert "cache_place" not in phases
    assert "cache_encode" not in phases
    assert "cache_end" not in phases
    assert phases == ["load", "train_placed", "step", "teardown", "summary"]


def test_warm_cache_default_sampler_probe_omits_cache_place(
    tmp_path, monkeypatch
):
    _install_fake_default_sampler(monkeypatch)
    root = make_job(tmp_path, max_steps=1)
    assert cache_job(root, **injections()) == 0
    phases: list[str] = []
    payload = default_sampler_injections(
        gpu_usage_probe=_recording_gpu_probe(phases)
    )
    assert run_job(root, **payload) == 0
    assert phases == [
        "load",
        "train_placed",
        "step",
        "teardown",
        "summary",
    ]
    text_encoder = payload["loaders"].text_encoder.created[0]
    vae = payload["loaders"].vae.created[0]
    assert all(torch.device(target).type != "cuda" for target in text_encoder.moved_to)
    assert all(torch.device(target).type != "cuda" for target in vae.moved_to)


def test_run_job_preview_probe_phases(tmp_path):
    phases: list[str] = []
    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)
    assert (
        run_job(
            root,
            **injections(
                gpu_usage_probe=_recording_gpu_probe(phases),
                checkpoint_writer=RecordingWriter([]),
                preview_sampler=RecordingSampler([]),
            ),
        )
        == 0
    )
    assert "step_peak" not in phases
    assert phases == [
        "load",
        "cache_place",
        "cache_encode",
        "cache_end",
        "train_placed",
        "step",
        "preview_end",
        "teardown",
        "summary",
    ]


def test_run_job_probes_each_optimizer_step(tmp_path):
    phases: list[str] = []
    root = make_job(tmp_path, max_steps=3)
    assert (
        run_job(root, **injections(gpu_usage_probe=_recording_gpu_probe(phases)))
        == 0
    )
    assert phases.count("step") == 3


def test_write_checkpoint_probe_preview_pause_end_restore(tmp_path):
    phases: list[str] = []
    events: list[str] = []
    writer = RecordingWriter(events)
    transformer = FakeTransformer()
    transformer.residency = "cuda"
    sampler = CleanupRecordingSampler(events, main_transformer=transformer)
    optimizer = SimpleNamespace(state={"p": torch.tensor(1.0)})
    root = make_job(tmp_path)
    runtime = {
        "config": load_job_config(root),
        "transformer": transformer,
        "optimizer": optimizer,
        "accelerator": PassthroughAccelerator(),
        "setup": SimpleNamespace(adapter_name="default"),
        "last_error": None,
        "components": SimpleNamespace(),
    }

    def move_transformer(model, device):
        model.residency = device.type
        events.append(f"main_{device.type}")

    def move_optimizer(tensor, device):
        return tensor

    assert (
        _write_checkpoint_then_sample(
            root,
            JobState("job", JobStatus.RUNNING, step=1, epoch=0),
            runtime,
            {
                "device": torch.device("cuda"),
                "checkpoint_writer": writer,
                "preview_sampler": sampler,
                "get_lora_state": lambda _model: {"lora": torch.tensor(1.0)},
                "training_transformer_mover": move_transformer,
                "optimizer_tensor_mover": move_optimizer,
                "cuda_synchronize": lambda: None,
                "cuda_empty_cache": lambda: None,
                "garbage_collect": lambda: None,
                "gpu_usage_probe": _recording_gpu_probe(phases),
            },
        )
        == 0
    )
    assert phases == ["preview_pause", "preview_end", "restore"]


def test_raising_gpu_probe_does_not_fail_job(tmp_path):
    def boom(phase, components=None):
        raise RuntimeError("probe failed")

    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections(gpu_usage_probe=boom)) == 0


def test_stale_cache_without_cuda_or_device_inject_requires_cuda(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    events: list = []
    _track_place_park(monkeypatch, events)
    root = make_job(tmp_path)
    payload = injections()
    del payload["device"]

    with pytest.raises(TrainingConfigError, match="training requires CUDA"):
        cache_job(root, **payload)

    assert events == []


def test_encode_exception_still_parks_cache_modules(tmp_path, monkeypatch):
    events: list = []
    _track_place_park(monkeypatch, events)

    class BoomVae(FakeVae):
        def encode(self, pixels):
            events.append("encode")
            raise RuntimeError("encode failed")

    loaders = ComponentLoaders(
        vae=Factory(BoomVae),
        tokenizer=Factory(FakeTokenizer),
        text_encoder=Factory(FakeTextEncoder),
        transformer=Factory(lambda: FakeTransformer([], trainable=False)),
        scheduler=Factory(FakeScheduler),
    )
    root = make_job(tmp_path)

    from zimage.training.cache import CacheError

    with pytest.raises(CacheError, match="size=16x16") as caught:
        cache_job(root, loaders=loaders, device="cpu", fp8_capable=False)

    assert "a.png" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "encode failed" in str(caught.value.__cause__)
    assert events == [("place", "cpu"), "encode", "park"]
    assert torch.device(loaders.text_encoder.created[0].moved_to[-1]).type == "cpu"


def test_warm_preview_cache_does_not_place_text_encoder(tmp_path, monkeypatch):
    _install_fake_default_sampler(monkeypatch)
    root = make_job(
        tmp_path,
        max_steps=1,
        sampling=_sampling_block(prompt="preview only"),
    )
    assert cache_job(root, **injections()) == 0

    events: list = []
    _track_place_park(monkeypatch, events)
    payload = default_sampler_injections()
    assert run_job(root, **payload) == 0
    assert events == []
    vae = payload["loaders"].vae.created[0]
    text_encoder = payload["loaders"].text_encoder.created[0]
    assert all(torch.device(target).type != "cuda" for target in vae.moved_to)
    assert all(torch.device(target).type != "cuda" for target in text_encoder.moved_to)


def test_stale_cache_preview_runs_after_place_before_park(tmp_path, monkeypatch):
    events: list = []
    loop_module = _track_place_park(monkeypatch, events)
    _install_fake_default_sampler(monkeypatch)
    real_preview = loop_module.prepare_preview_prompt_cache

    def tracking_preview(prompts, encoder, config, *, job_dir):
        events.append("preview")
        return real_preview(prompts, encoder, config, job_dir=job_dir)

    monkeypatch.setattr(loop_module, "prepare_preview_prompt_cache", tracking_preview)

    class TrackingVae(FakeVae):
        def encode(self, pixels):
            events.append("encode")
            return super().encode(pixels)

    root = make_job(
        tmp_path,
        max_steps=1,
        sampling=_sampling_block(prompt="preview after cache"),
    )
    payload = default_sampler_injections()
    payload["loaders"] = ComponentLoaders(
        vae=Factory(TrackingVae),
        tokenizer=Factory(FakeTokenizer),
        text_encoder=Factory(FakeTextEncoder),
        transformer=Factory(lambda: FakeTransformer([], trainable=False)),
        scheduler=Factory(FakeScheduler),
    )

    assert run_job(root, **payload) == 0
    assert events == [("place", "cpu"), "encode", "preview", "park"]


def test_loop_does_not_import_gradio_or_implementations():
    code = r"""
import sys
import zimage.training.loop as loop

forbidden = (
    "gradio",
    "zimage.ui",
    "zimage.training.checkpoints",
    "zimage.training.sampling",
    "zimage.engine.quantization",
    "zimage.engine.lora",
)
loaded = [name for name in forbidden if name in sys.modules]
src = open(loop.__file__, encoding="utf-8").read()
needles = (
    "gradio",
    "zimage.ui",
    "zimage.training.checkpoints",
    "zimage.training.sampling",
    "apply_quantization",
    "zimage.engine.lora",
)
source_hits = [needle for needle in needles if needle in src]
if loaded or source_hits:
    print("loaded=" + ",".join(loaded), file=sys.stderr)
    print("source=" + ",".join(source_hits), file=sys.stderr)
    raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_job_uses_fakes_and_does_not_download(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("production weight download")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    monkeypatch.setattr(
        "zimage.training.modeling.ComponentLoaders.defaults",
        lambda: (_ for _ in ()).throw(AssertionError("defaults would download")),
    )
    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections()) == 0


def test_default_preview_sampler_uses_from_components_without_pipeline(monkeypatch):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    components = SimpleNamespace(
        sampling_transformer=object(),
        sampling_scheduler=object(),
        vae=object(),
        main_transformer=object(),
    )
    embeddings = {"one": Path("one.safetensors")}
    negatives = {"neg": Path("neg.safetensors")}
    common = {"prompt": "one", "width": 8}

    sampler = loop_module._default_preview_sampler(
        {
            "components": components,
            "config": {
                "sampling": common,
                "lora": {"targets": ["to_k"]},
            },
            "preview_prompt_paths": embeddings,
            "preview_negative_paths": negatives,
        },
        {"device": "cpu"},
    )

    assert sampler is not None
    assert sampler.used_factory is True
    assert sampler.kwargs["transformer"] is components.sampling_transformer
    assert sampler.kwargs["scheduler"] is components.sampling_scheduler
    assert sampler.kwargs["vae"] is components.vae
    assert sampler.kwargs["prompt_paths"]["one"] is embeddings["one"]
    assert sampler.kwargs["negative_prompt_paths"]["neg"] is negatives["neg"]
    assert sampler.kwargs["common_parameters"] == common
    assert sampler.kwargs["device"] == "cpu"
    assert sampler.kwargs["target_modules"] == ["to_k"]
    assert sampler.kwargs["main_transformer"] is components.main_transformer
    assert "pipeline" not in sampler.kwargs
    assert callable(sampler.kwargs["gpu_usage_probe"])


def test_default_preview_sampler_falls_back_to_init_kwargs(monkeypatch):
    import zimage.training.loop as loop_module

    class InitOnlySampler:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    _install_fake_default_sampler(monkeypatch, InitOnlySampler)
    components = SimpleNamespace(
        sampling_transformer="xf",
        sampling_scheduler="sched",
        vae="vae",
        main_transformer="main",
    )

    sampler = loop_module._default_preview_sampler(
        {
            "components": components,
            "config": {"sampling": {}, "lora": {"targets": []}},
            "preview_prompt_paths": {"p": Path("p.safetensors")},
        },
        {"device": "cpu"},
    )

    assert sampler is not None
    assert sampler.kwargs["transformer"] == "xf"
    assert sampler.kwargs["scheduler"] == "sched"
    assert sampler.kwargs["vae"] == "vae"
    assert sampler.kwargs["prompt_paths"] == {"p": Path("p.safetensors")}
    assert callable(sampler.kwargs["gpu_usage_probe"])


def test_default_preview_sampler_binds_probe_and_accumulates_preview_peak(
    monkeypatch,
):
    import zimage.training.loop as loop_module
    from zimage.training.gpu_usage import GpuProbeContext

    _install_fake_default_sampler(monkeypatch)
    loop_module._reset_gpu_usage_job_peaks()
    seen: list[tuple[str, int | None]] = []

    def probe(phase, context=None):
        peak = getattr(context, "phase_peak_bytes", None)
        seen.append((phase, None if peak is None else int(peak)))

    components = SimpleNamespace(
        sampling_transformer=object(),
        sampling_scheduler=object(),
        vae=object(),
        main_transformer=object(),
    )
    sampler = loop_module._default_preview_sampler(
        {
            "components": components,
            "config": {"sampling": {}, "lora": {"targets": []}},
        },
        {"device": "cpu", "gpu_usage_probe": probe},
    )
    assert sampler is not None
    bound = sampler.kwargs["gpu_usage_probe"]
    bound("preview_run", GpuProbeContext(phase_peak_bytes=100))
    bound("preview_run", GpuProbeContext(phase_peak_bytes=300))
    bound("preview_run", GpuProbeContext(phase_peak_bytes=200))
    assert seen == [
        ("preview_run", 100),
        ("preview_run", 300),
        ("preview_run", 200),
    ]
    assert loop_module._GPU_USAGE_JOB_PEAKS["preview"] == 300


def test_cli_path_builds_default_sampler_before_releasing_text(monkeypatch, tmp_path):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    order: list[str] = []
    real_preview = loop_module.prepare_preview_prompt_cache
    real_lifecycle = loop_module.TrainingModelLifecycle

    def tracking_preview(prompts, encoder, config, *, job_dir):
        order.append("prepare")
        return real_preview(prompts, encoder, config, job_dir=job_dir)

    class TrackingLifecycle(real_lifecycle):
        def release_text_resources(self):
            order.append("release")
            return super().release_text_resources()

    monkeypatch.setattr(loop_module, "prepare_preview_prompt_cache", tracking_preview)
    monkeypatch.setattr(loop_module, "TrainingModelLifecycle", TrackingLifecycle)

    events: list = []
    FakeDefaultSampler.events = events
    writer = RecordingWriter(events)
    root = make_job(
        tmp_path,
        max_steps=1,
        checkpoint_every=1,
        sampling={
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "time_shift": 3.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "prompt": "shared",
            "negative_prompt": "neg",
            "samples": [{"prompt": "one"}, {"prompt": "two", "seed": 7}],
        },
    )

    assert (
        run_job(
            root,
            **default_sampler_injections(events, checkpoint_writer=writer),
        )
        == 0
    )

    assert "prepare" in order
    assert order.index("prepare") < order.index("release")
    sampler = FakeDefaultSampler.last
    assert sampler is not None
    assert sampler.used_factory is True
    assert set(sampler.kwargs["prompt_paths"]) == {"one", "two"}
    assert set(sampler.kwargs["negative_prompt_paths"]) == {"neg"}
    assert all(isinstance(path, Path) for path in sampler.kwargs["prompt_paths"].values())
    assert all(
        isinstance(path, Path) for path in sampler.kwargs["negative_prompt_paths"].values()
    )
    assert sampler.kwargs["transformer"] is not None
    assert sampler.kwargs["scheduler"] is not None
    assert sampler.kwargs["vae"] is not None
    assert events.count("write") == 1
    assert events.count("sample") == 2
    assert events.index("write") < events.index("sample")
    assert [call[1]["prompt"] for call in sampler.calls] == ["one", "two"]
    assert sampler.calls[0][0] is writer.saved[0]


def test_explicit_none_preview_sampler_skips_default_factory(monkeypatch, tmp_path):
    import zimage.training.loop as loop_module

    called = {"n": 0}

    def boom(runtime, injected=None):
        called["n"] += 1
        raise AssertionError("default sampler should not run")

    monkeypatch.setattr(loop_module, "_default_preview_sampler", boom)
    events: list = []
    writer = RecordingWriter(events)
    root = make_job(tmp_path, max_steps=1, checkpoint_every=1)

    assert (
        run_job(
            root,
            **injections(events, checkpoint_writer=writer, preview_sampler=None),
        )
        == 0
    )
    assert called["n"] == 0
    assert events.count("write") == 1
    assert "sample" not in events
    assert writer.saved
    assert not list((root / "previews").rglob("*"))


def test_natural_completion_final_save_and_sample_with_default_sampler(
    monkeypatch, tmp_path
):
    _install_fake_default_sampler(monkeypatch)
    events: list = []
    FakeDefaultSampler.events = events
    writer = RecordingWriter(events)
    root = make_job(tmp_path, max_steps=2, checkpoint_every=100)

    assert (
        run_job(root, **default_sampler_injections(events, checkpoint_writer=writer))
        == 0
    )
    assert events.count("write") == 1
    sampler = FakeDefaultSampler.last
    assert sampler is not None
    assert sampler.calls
    assert writer.saved[0].metadata.optimizer_step == 2


def test_production_without_cuda_or_explicit_device_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = make_job(tmp_path, max_steps=1)
    payload = injections()
    del payload["device"]

    with pytest.raises(
        (TrainingConfigError, RuntimeError),
        match="training requires CUDA.*CPU is unsupported",
    ):
        run_job(root, **payload)


def test_explicit_cuda_without_cuda_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = make_job(tmp_path, max_steps=1)

    with pytest.raises(
        (TrainingConfigError, RuntimeError),
        match="training requires CUDA",
    ):
        run_job(root, **injections(device="cuda"))


def test_explicit_cpu_device_injection_is_permitted_for_tests(tmp_path):
    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections(device="cpu")) == 0
    assert load_job_state(root).step == 1


def test_accelerator_factory_receives_cuda_bf16_and_grad_accum():
    captured: dict[str, object] = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return PassthroughAccelerator("cuda")

    accelerator = _construct_accelerator(
        torch.device("cuda"),
        {"accelerator_factory": factory},
    )

    assert isinstance(accelerator, PassthroughAccelerator)
    assert captured["mixed_precision"] == "bf16"
    assert captured["gradient_accumulation_steps"] == 1
    assert captured.get("cpu") is not True


def test_prepare_device_mismatch_fails_before_first_step(tmp_path):
    root = make_job(tmp_path, max_steps=1)

    with pytest.raises(
        (TrainingConfigError, RuntimeError),
        match="does not match target training device",
    ):
        run_job(
            root,
            **injections(
                device="cpu",
                accelerator=PassthroughAccelerator("cuda"),
            ),
        )
    assert load_job_state(root).step == 0


def test_trainable_parameter_device_mismatch_fails_before_first_step(tmp_path):
    class MetaPrepareAccelerator(PassthroughAccelerator):
        def prepare(self, *args):
            transformer = args[0]
            for name, parameter in list(transformer.named_parameters()):
                if not parameter.requires_grad:
                    continue
                replaced = torch.nn.Parameter(
                    torch.empty(parameter.shape, device="meta"),
                    requires_grad=True,
                )
                module = transformer
                parts = name.split(".")
                for part in parts[:-1]:
                    module = getattr(module, part)
                module.register_parameter(parts[-1], replaced)
            return args

    root = make_job(tmp_path, max_steps=1)

    with pytest.raises(
        (TrainingConfigError, RuntimeError),
        match="trainable adapter parameters must remain on",
    ):
        run_job(
            root,
            **injections(
                device="cpu",
                accelerator=MetaPrepareAccelerator("cpu"),
            ),
        )
    assert load_job_state(root).step == 0


def test_explicit_fp8_capable_cuda_path_cannot_silently_disable_fp8(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import zimage.training.loop as loop_module

    def fake_setup(transformer, **kwargs):
        if not hasattr(transformer, "lora_scale"):
            transformer.lora_scale = torch.nn.Parameter(torch.tensor(0.25))
        return SimpleNamespace(
            transformer=transformer,
            requested_precision="fp8",
            effective_precision="bf16",
            fp8_enabled=False,
            gradient_checkpointing_enabled=False,
            adapter_name="default",
        )

    monkeypatch.setattr(loop_module, "setup_main_transformer", fake_setup)
    monkeypatch.setattr(
        "zimage.training.modeling.quantize_text_encoder",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "zimage.training.modeling.quantize_sampling_transformer",
        lambda *_args, **_kwargs: None,
    )
    root = make_job(tmp_path, max_steps=1, precision="fp8")
    # Materialize cache on CPU so injecting device="cuda" does not place.
    assert cache_job(root, **injections(device="cpu")) == 0

    with pytest.raises(
        (TrainingConfigError, RuntimeError),
        match="cannot disable fp8",
    ):
        run_job(
            root,
            **injections(
                device="cuda",
                fp8_capable=True,
                accelerator=PassthroughAccelerator("cuda"),
            ),
        )
    assert load_job_state(root).step == 0


def test_bf16_cuda_path_accepted(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = _resolve_training_device({})
    assert device.type == "cuda"

    captured: dict[str, object] = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return PassthroughAccelerator("cuda")

    _construct_accelerator(device, {"accelerator_factory": factory})
    assert captured["mixed_precision"] == "bf16"
    assert captured["gradient_accumulation_steps"] == 1

    _validate_prepared_training_runtime(
        accelerator=PassthroughAccelerator("cuda"),
        transformer=torch.nn.Module(),
        setup=SimpleNamespace(fp8_enabled=False, adapter_name="default"),
        device=device,
        job={"precision": "bf16"},
        injected={"fp8_capable": True},
    )


def _cuda_prepared_runtime_kwargs(**overrides):
    payload = {
        "accelerator": PassthroughAccelerator("cuda"),
        "transformer": torch.nn.Module(),
        "setup": SimpleNamespace(fp8_enabled=False, adapter_name="default"),
        "device": torch.device("cuda"),
        "job": {"precision": "bf16"},
        "injected": {},
    }
    payload.update(overrides)
    return payload


def test_leftover_cuda_vae_fails_prepared_runtime():
    components = SimpleNamespace(
        vae=SimpleNamespace(device=torch.device("cuda")),
        text_encoder=None,
        sampling_transformer=None,
    )
    with pytest.raises(TrainingConfigError, match="vae on CUDA"):
        _validate_prepared_training_runtime(
            **_cuda_prepared_runtime_kwargs(components=components)
        )


def test_invalid_yaml_update_sets_last_error_and_consumes_command(tmp_path):
    root = make_job(tmp_path, max_steps=2)
    # Enqueue a structurally broken update payload via raw command file.
    from zimage.training.commands import enqueue_update

    bad = load_job_config(root)
    bad["optimizer"]["learning_rate"] = "not-a-number"
    enqueue_update(root, bad)

    assert run_job(root, **injections()) == 0
    state = load_job_state(root)
    assert state.last_error is not None
    assert "learning_rate" in state.last_error or "number" in state.last_error.lower()
    assert not list((root / "commands").glob("*.json"))


def test_resume_rewinds_step_to_latest_checkpoint(tmp_path):
    root = make_job(tmp_path, max_steps=2)
    config = load_job_config(root)
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / "step-1",
        lora_state={
            "to_q.lora_A.weight": torch.ones(2, 16),
            "to_q.lora_B.weight": torch.ones(16, 2),
        },
        metadata=NativeAdapterMetadata(
            adapter_name="default",
            base_model_name_or_path=str(config["model"]["main_transformer"]["path"]),
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
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=2, epoch=0))
    steps: list[int] = []

    class Hook:
        def on_optimizer_step(self, boundary):
            steps.append(boundary.state.step)

    assert run_job(root, **default_loader_injections(training_hook=Hook())) == 0
    # Rewound to checkpoint step 1, then ran through max_steps=2 → one step.
    assert steps == [2]
    assert load_job_state(root).step == 2


def test_mid_run_prompt_refresh_serial_te_handoff(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    events: list[str] = []
    after_handoff: dict[str, object] = {}
    last_optimizer_device = {"value": None}

    real_reload = loop_module.TrainingModelLifecycle.reload_text_resources_on_cpu
    real_preview = loop_module.prepare_preview_prompt_cache
    real_release = loop_module.TrainingModelLifecycle.release_text_resources
    real_place = loop_module.TrainingModelLifecycle.place_cache_modules
    real_apply = loop_module._apply_hot_runtime
    placed = {"device": None, "vae": None}

    def tracking_reload(self, *args, **kwargs):
        tokenizer, encoder = real_reload(self, *args, **kwargs)
        events.append("te_loaded")
        assert self.text_resources_loaded
        if hasattr(encoder, "moved_to") and encoder.moved_to:
            assert torch.device(encoder.moved_to[-1]).type == "cpu"
        return tokenizer, encoder

    def tracking_place(self, device, *, vae=True):
        target = torch.device(device)
        if vae:
            return real_place(self, device, vae=vae)
        # Record residency without allocating CUDA tensors.
        events.append("te_place")
        placed["device"] = target
        placed["vae"] = vae
        encoder = self.components.text_encoder
        if encoder is None:
            return None
        encoder.residency = target.type
        if target.type != "cuda":
            encoder.to(device)
        return None

    def tracking_preview(prompts, encoder, config, *, job_dir):
        if placed["device"] is None:
            return real_preview(prompts, encoder, config, job_dir=job_dir)
        events.append("te_encode")
        target = placed["device"]
        assert placed["vae"] is False
        residency = getattr(
            encoder.text_encoder,
            "residency",
            next(encoder.text_encoder.parameters()).device.type,
        )
        assert residency == target.type
        assert next(encoder.text_encoder.parameters()).device.type == "cpu"
        return real_preview(prompts, encoder, config, job_dir=job_dir)

    def tracking_release(self):
        result = real_release(self)
        events.append("te_released")
        assert not self.text_resources_loaded
        return result

    def tracking_apply(runtime, reload, injected):
        before = len(events)
        result = real_apply(runtime, reload, injected)
        slice_events = list(events[before:])
        if "te_encode" not in slice_events:
            return result
        after_handoff["mid_run"] = slice_events
        lifecycle = runtime["lifecycle"]
        after_handoff["tokenizer"] = lifecycle.components.tokenizer
        after_handoff["text_encoder"] = lifecycle.components.text_encoder
        after_handoff["text_resources_loaded"] = lifecycle.text_resources_loaded
        after_handoff["prompts"] = dict(
            runtime.get("preview_prompt_paths") or {}
        )
        transformer = runtime["transformer"]
        after_handoff["main_residency"] = getattr(transformer, "residency", None)
        after_handoff["main_devices"] = {
            name: param.device.type
            for name, param in transformer.named_parameters()
        }
        optimizer_devices: list[str] = []
        for bucket in runtime["optimizer"].state.values():
            if not isinstance(bucket, dict):
                continue
            for item in bucket.values():
                if isinstance(item, torch.Tensor):
                    optimizer_devices.append(item.device.type)
        after_handoff["optimizer_devices"] = optimizer_devices
        return result

    def move_transformer(model, device):
        # Record residency without allocating CUDA tensors.
        model.residency = device.type
        events.append(f"main_{device.type}")
        if device.type != "cuda":
            model.to(device)

    def move_optimizer(tensor, device):
        if last_optimizer_device["value"] != device.type:
            last_optimizer_device["value"] = device.type
            events.append(f"optimizer_{device.type}")
        if device.type == "cuda":
            return tensor
        return tensor.to(device=device)

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle,
        "reload_text_resources_on_cpu",
        tracking_reload,
    )
    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle,
        "place_cache_modules",
        tracking_place,
    )
    monkeypatch.setattr(
        loop_module, "prepare_preview_prompt_cache", tracking_preview
    )
    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle,
        "release_text_resources",
        tracking_release,
    )
    monkeypatch.setattr(loop_module, "_apply_hot_runtime", tracking_apply)
    # Training steps stay on the injected CPU runtime device. Only the serial
    # handoff consults this helper, so fake movers can restore residency
    # without allocating CUDA.
    monkeypatch.setattr(
        loop_module,
        "_training_device_from_runtime",
        lambda _runtime, _injected: torch.device("cuda"),
    )

    root = make_job(
        tmp_path,
        max_steps=2,
        sampling={
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "time_shift": 3.0,
            "width": 64,
            "height": 64,
            "seed": 42,
            "prompt": "old prompt",
            "negative_prompt": "",
            "samples": [{"prompt": "old prompt"}],
        },
    )
    updated = load_job_config(root)
    updated["sampling"]["prompt"] = "brand new prompt"
    updated["sampling"]["samples"] = [{"prompt": "brand new prompt"}]
    enqueue_update(root, updated)

    class CapturingSampler:
        def __init__(self):
            self.prompt_paths = {"old prompt": Path("old")}
            self.negative_prompt_paths = {}
            self.common_parameters = {}

        def sample_unfused(self, **kwargs):
            return kwargs.get("destination")

        def release_after_preview(self):
            events.append("sampler_release")

    sampler = CapturingSampler()
    payload = injections(
        events,
        preview_sampler=sampler,
        training_transformer_mover=move_transformer,
        optimizer_tensor_mover=move_optimizer,
        cuda_synchronize=lambda: events.append("sync"),
        cuda_empty_cache=lambda: events.append("empty_cache"),
        garbage_collect=lambda: events.append("gc"),
    )
    assert run_job(root, **payload) == 0

    mid_run = after_handoff["mid_run"]
    assert mid_run == [
        "sync",
        "main_cpu",
        "optimizer_cpu",
        "gc",
        "empty_cache",
        "sampler_release",
        "te_loaded",
        "te_place",
        "te_encode",
        "te_released",
        "main_cuda",
        "optimizer_cuda",
        "sync",
    ]
    assert placed["device"] is not None
    assert placed["device"].type == "cuda"
    assert placed["vae"] is False
    assert after_handoff["tokenizer"] is None
    assert after_handoff["text_encoder"] is None
    assert after_handoff["text_resources_loaded"] is False
    assert "brand new prompt" in after_handoff["prompts"]
    assert "brand new prompt" in sampler.prompt_paths
    assert all(
        isinstance(path, Path) for path in after_handoff["prompts"].values()
    )
    assert all(
        not isinstance(value, torch.Tensor)
        for value in after_handoff["prompts"].values()
    )
    vae = payload["loaders"].vae.created[0]
    assert all(torch.device(target).type != "cuda" for target in vae.moved_to)
    assert after_handoff["main_residency"] == "cuda"
    assert set(after_handoff["main_devices"].values()) <= {"cpu"}
    assert set(after_handoff["optimizer_devices"] or ["cpu"]) <= {"cpu"}
    assert load_job_config(root)["sampling"]["prompt"] == (
        "brand new prompt"
    )
    assert load_job_state(root).status is JobStatus.RUNNING or load_job_state(
        root
    ).step == 2


def test_serial_refresh_passes_precision_and_converts_reloaded_encoder(
    monkeypatch, tmp_path
):
    import zimage.training.loop as loop_module
    from zimage.training.cache import CacheConfig, load_preview_cache

    converted = []

    def fake_quantize(module, *, precision="fp8"):
        converted.append(
            (id(module), getattr(module, "_quantized_precision", None), precision)
        )
        setattr(module, "_quantized_precision", precision)

    monkeypatch.setattr("zimage.training.modeling.quantize_text_encoder", fake_quantize)
    captured = {}
    real_reload = loop_module.TrainingModelLifecycle.reload_text_resources_on_cpu

    def tracking_reload(self, *args, **kwargs):
        captured.update(kwargs)
        tokenizer, encoder = real_reload(self, *args, **kwargs)
        captured["encoder"] = encoder
        return tokenizer, encoder

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle,
        "reload_text_resources_on_cpu",
        tracking_reload,
    )

    components = TrainingModelComponents(
        sources=ModelSources(ModelSource("main"), ModelSource("main")),
        vae=FakeVae(),
        tokenizer=None,
        text_encoder=None,
        training_scheduler=object(),
        main_transformer=FakeTransformer([], trainable=False),
        sampling_transformer=FakeTransformer([], trainable=False),
        sampling_scheduler=object(),
    )
    lifecycle = TrainingModelLifecycle(components)
    sampler = SimpleNamespace(
        prompt_paths={},
        negative_prompt_paths={},
        common_parameters={},
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    cache_config = CacheConfig(
        main_revision="fake-revision",
        vae_config={"shift_factor": 0.0, "scaling_factor": 1.0},
        text_encoder_config={"model_type": "qwen3"},
        tokenizer_config={"padding_side": "left"},
        qwen_chat_template=dict(loop_module.QWEN_CHAT_TEMPLATE),
        max_sequence_length=4,
    )
    runtime = {
        "lifecycle": lifecycle,
        "config": {
            "precision": " FP8 ",
            "max_sequence_length": 4,
            "sampling": {
                "prompt": "refresh me",
                "negative_prompt": "",
                "samples": [{"prompt": "refresh me"}],
            },
        },
        "transformer": components.main_transformer,
        "device": "cpu",
        "job_dir": job_dir,
        "cache_config": cache_config,
        "preview_prompt_paths": {},
        "preview_negative_paths": {},
    }
    injected = {
        "loaders": make_loaders([]),
        "fp8_capable": True,
        "device": "cpu",
        "preview_sampler": sampler,
    }

    loop_module._refresh_preview_prompt_cache_serial(runtime, injected)

    assert captured["precision"] == "fp8"
    assert captured["quantize_capable"] is True
    assert captured["quantize_capable"] == loop_module._fp8_capable(injected)
    first_id = converted[0][0]
    assert converted == [(first_id, None, "fp8")]
    assert first_id == id(captured["encoder"])
    cached_path = runtime["preview_prompt_paths"]["refresh me"]
    embedded, _metadata = load_preview_cache(cached_path)
    assert torch.isfinite(embedded).all()
    assert lifecycle.text_resources_loaded is False
    assert lifecycle.components.text_encoder is None
    assert "refresh me" in sampler.prompt_paths
    assert sampler.prompt_paths["refresh me"] == cached_path

    captured.clear()
    loop_module._refresh_preview_prompt_cache_serial(runtime, injected)
    assert converted == [(first_id, None, "fp8")]
    assert "precision" not in captured

    runtime["config"]["sampling"]["prompt"] = "refresh me too"
    runtime["config"]["sampling"]["samples"] = [{"prompt": "refresh me too"}]
    loop_module._refresh_preview_prompt_cache_serial(runtime, injected)
    assert converted[1][0] != first_id
    assert converted[1][1] is None
    assert captured["quantize_capable"] is True
    assert "refresh me too" in runtime["preview_prompt_paths"]


def _sampling_block(*, prompt: str, negative: str = "", **sample_extra) -> dict:
    return {
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "time_shift": 3.0,
        "width": 64,
        "height": 64,
        "seed": 42,
        "prompt": "",
        "negative_prompt": "",
        "samples": [{"prompt": prompt, "negative_prompt": negative, **sample_extra}],
    }


def _adapter_metadata_for_job(config, *, step: int, **overrides) -> NativeAdapterMetadata:
    peft = {
        "r": int(config["lora"]["rank"]),
        "lora_alpha": float(config["lora"]["alpha"]),
        "lora_dropout": float(config["lora"]["dropout"]),
        "target_modules": list(config["lora"]["targets"]),
        "peft_type": "LORA",
    }
    peft.update(overrides.pop("peft", {}))
    return NativeAdapterMetadata(
        adapter_name=overrides.get("adapter_name", "default"),
        base_model_name_or_path=overrides.get(
            "base_model_name_or_path",
            str(config["model"]["main_transformer"]["path"]),
        ),
        base_model_revision=overrides.get(
            "base_model_revision",
            config["model"]["main_transformer"].get("revision"),
        ),
        peft_config=peft,
        optimizer_step=step,
    )


def _write_job_checkpoint(root: Path, *, step: int, **metadata_overrides) -> None:
    config = load_job_config(root)
    NativeLoraCheckpointWriter().write_atomic(
        destination=root / "checkpoints" / f"step-{step}",
        lora_state={
            "to_q.lora_A.weight": torch.ones(2, 16),
            "to_q.lora_B.weight": torch.ones(16, 2),
        },
        metadata=_adapter_metadata_for_job(config, step=step, **metadata_overrides),
    )


def test_per_sample_prompt_updates_stores_without_rebuild(monkeypatch, tmp_path):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    rebuilds = {"n": 0}
    real_rebuild = loop_module._rebuild_runtime

    def tracking_rebuild(*args, **kwargs):
        rebuilds["n"] += 1
        return real_rebuild(*args, **kwargs)

    monkeypatch.setattr(loop_module, "_rebuild_runtime", tracking_rebuild)

    events: list = []
    FakeDefaultSampler.events = events
    root = make_job(
        tmp_path,
        max_steps=2,
        sampling=_sampling_block(prompt="old sample prompt"),
    )
    updated = load_job_config(root)
    updated["sampling"]["samples"] = [{"prompt": "new sample prompt"}]
    enqueue_update(root, updated)

    assert (
        run_job(root, **default_sampler_injections(events, checkpoint_writer=None))
        == 0
    )
    assert rebuilds["n"] == 0
    sampler = FakeDefaultSampler.last
    assert sampler is not None
    assert "new sample prompt" in sampler.prompt_paths
    new_path = Path(sampler.prompt_paths["new sample prompt"])
    preview_root = (root / ".cache" / "preview").resolve()
    assert new_path.is_file()
    assert new_path.resolve().parent == preview_root
    files = sorted(preview_root.glob("*.safetensors"))
    assert new_path.resolve() in [path.resolve() for path in files]
    assert len(files) >= 2
    assert load_job_state(root).step == 2
    assert load_job_state(root).status is JobStatus.RUNNING


def test_per_sample_nonempty_negative_updates_stores_without_rebuild(
    monkeypatch, tmp_path
):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    rebuilds = {"n": 0}
    real_rebuild = loop_module._rebuild_runtime

    def tracking_rebuild(*args, **kwargs):
        rebuilds["n"] += 1
        return real_rebuild(*args, **kwargs)

    monkeypatch.setattr(loop_module, "_rebuild_runtime", tracking_rebuild)

    events: list = []
    FakeDefaultSampler.events = events
    root = make_job(
        tmp_path,
        max_steps=2,
        sampling=_sampling_block(prompt="keep prompt"),
    )
    updated = load_job_config(root)
    updated["sampling"]["samples"] = [
        {"prompt": "keep prompt", "negative_prompt": "new sample negative"}
    ]
    enqueue_update(root, updated)

    assert (
        run_job(root, **default_sampler_injections(events, checkpoint_writer=None))
        == 0
    )
    assert rebuilds["n"] == 0
    sampler = FakeDefaultSampler.last
    assert sampler is not None
    assert "keep prompt" in sampler.prompt_paths
    assert "new sample negative" in sampler.negative_prompt_paths
    keep_path = Path(sampler.prompt_paths["keep prompt"])
    neg_path = Path(sampler.negative_prompt_paths["new sample negative"])
    preview_root = (root / ".cache" / "preview").resolve()
    assert keep_path.is_file()
    assert neg_path.is_file()
    assert keep_path.resolve() != neg_path.resolve()
    assert keep_path.resolve().parent == preview_root
    assert neg_path.resolve().parent == preview_root
    assert load_job_state(root).step == 2


def test_sampling_size_seed_time_shift_and_empty_negative_do_not_load_te(
    monkeypatch, tmp_path
):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    te_loads = {"n": 0}
    refresh_calls = {"n": 0}
    real_reload = loop_module.TrainingModelLifecycle.reload_text_resources_on_cpu
    real_refresh = loop_module._refresh_preview_prompt_cache_serial

    def counting_reload(self, *args, **kwargs):
        te_loads["n"] += 1
        return real_reload(self, *args, **kwargs)

    def counting_refresh(runtime, injected):
        refresh_calls["n"] += 1
        return real_refresh(runtime, injected)

    monkeypatch.setattr(
        loop_module.TrainingModelLifecycle,
        "reload_text_resources_on_cpu",
        counting_reload,
    )
    monkeypatch.setattr(
        loop_module, "_refresh_preview_prompt_cache_serial", counting_refresh
    )

    events: list = []
    FakeDefaultSampler.events = events
    root = make_job(
        tmp_path,
        max_steps=2,
        sampling=_sampling_block(prompt="stable prompt", negative="old negative"),
    )
    updated = load_job_config(root)
    updated["sampling"]["width"] = 32
    updated["sampling"]["height"] = 32
    updated["sampling"]["seed"] = 99
    updated["sampling"]["time_shift"] = 1.25
    updated["sampling"]["negative_prompt"] = ""
    updated["sampling"]["samples"] = [
        {"prompt": "stable prompt", "negative_prompt": "", "seed": 7}
    ]
    enqueue_update(root, updated)

    assert (
        run_job(root, **default_sampler_injections(events, checkpoint_writer=None))
        == 0
    )
    assert refresh_calls["n"] == 0
    assert te_loads["n"] == 0
    assert load_job_state(root).step == 2


def test_runtime_and_sampler_store_preview_paths_without_tensors(
    tmp_path, monkeypatch
):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    captured: dict[str, object] = {}
    real_optimize = loop_module._optimize

    def wrapping_optimize(job_dir, state, holder, injected):
        runtime = holder["runtime"]
        captured["prompt"] = dict(runtime["preview_prompt_paths"])
        captured["negative"] = dict(runtime["preview_negative_paths"])
        captured["has_embed_key"] = "preview_prompt_embeddings" in runtime
        return real_optimize(job_dir, state, holder, injected)

    monkeypatch.setattr(loop_module, "_optimize", wrapping_optimize)
    root = make_job(
        tmp_path,
        max_steps=1,
        sampling=_sampling_block(prompt="one", negative="neg"),
    )
    assert run_job(root, **default_sampler_injections()) == 0
    prompt_paths = captured["prompt"]
    negative_paths = captured["negative"]
    assert "one" in prompt_paths
    assert "neg" in negative_paths
    assert captured["has_embed_key"] is False
    assert all(isinstance(path, Path) for path in prompt_paths.values())
    assert all(isinstance(path, Path) for path in negative_paths.values())
    assert all(not isinstance(value, torch.Tensor) for value in prompt_paths.values())
    preview_root = (root / ".cache" / "preview").resolve()
    assert all(Path(path).resolve().parent == preview_root for path in prompt_paths.values())
    sampler = FakeDefaultSampler.last
    assert sampler is not None
    assert set(sampler.prompt_paths) == {"one"}
    assert set(sampler.negative_prompt_paths) == {"neg"}
    assert all(
        not isinstance(value, torch.Tensor) for value in sampler.prompt_paths.values()
    )


def test_positive_and_negative_share_one_preview_file(tmp_path):
    root = make_job(
        tmp_path,
        sampling=_sampling_block(prompt="shared text", negative="shared text"),
    )
    assert cache_job(root, **injections()) == 0
    files = sorted((root / ".cache" / "preview").rglob("*.safetensors"))
    assert len(files) == 1


def test_rebuild_reuses_compatible_preview_prompt_files(tmp_path, monkeypatch):
    import zimage.training.loop as loop_module

    _install_fake_default_sampler(monkeypatch)
    root = make_job(
        tmp_path,
        max_steps=2,
        sampling=_sampling_block(prompt="keep preview"),
    )
    assert cache_job(root, **injections()) == 0
    preview_root = root / ".cache" / "preview"
    before = {path: path.read_bytes() for path in preview_root.rglob("*.safetensors")}
    assert before
    encodes = {"n": 0}
    real_preview = loop_module.prepare_preview_prompt_cache

    def counting_preview(prompts, encoder, config, *, job_dir):
        encodes["n"] += 1
        return real_preview(prompts, encoder, config, job_dir=job_dir)

    monkeypatch.setattr(loop_module, "prepare_preview_prompt_cache", counting_preview)
    updated = load_job_config(root)
    updated["gradient_checkpointing"] = True
    enqueue_update(root, updated)
    assert run_job(root, **default_sampler_injections()) == 0
    assert encodes["n"] == 0
    after = {path: path.read_bytes() for path in preview_root.rglob("*.safetensors")}
    assert after == before


def test_epochs_mode_rewinds_step_and_epoch_and_repeats_tail(monkeypatch, tmp_path):
    import zimage.training.loop as loop_module

    root = make_job(tmp_path, max_steps=None, epochs=2)
    dataset = Path(load_job_config(root)["datasets"][0]["name"])
    Image.new("RGB", (16, 16), (40, 50, 60)).save(dataset / "b.png")
    (dataset / "b.txt").write_text("b caption", encoding="utf-8")
    _write_job_checkpoint(root, step=3)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=4, epoch=2))

    rewound: dict[str, int] = {}
    real_rewind = loop_module._rewind_state_to_checkpoint

    def tracking_rewind(*args, **kwargs):
        state = real_rewind(*args, **kwargs)
        rewound["step"] = state.step
        rewound["epoch"] = state.epoch
        return state

    monkeypatch.setattr(loop_module, "_rewind_state_to_checkpoint", tracking_rewind)

    steps: list[int] = []
    epochs: list[int] = []

    class Hook:
        def on_optimizer_step(self, boundary):
            steps.append(boundary.state.step)
            epochs.append(boundary.state.epoch)

    assert run_job(root, **default_loader_injections(training_hook=Hook())) == 0
    assert rewound == {"step": 3, "epoch": 1}
    assert steps == [4]
    assert epochs == [2]
    state = load_job_state(root)
    assert state.step == 4
    assert state.epoch == 2


def test_rebuild_teardown_error_aborts_before_new_accelerator(monkeypatch, tmp_path):
    import zimage.training.loop as loop_module

    class FailingReleaseSampler(FakeDefaultSampler):
        def release_after_preview(self):
            raise RuntimeError("sampler teardown boom")

    _install_fake_default_sampler(monkeypatch, FailingReleaseSampler)
    constructed = {"n": 0}
    builds = {"n": 0}
    real_construct = loop_module._construct_accelerator
    real_build = loop_module._build_runtime

    def counting_construct(device, injected):
        constructed["n"] += 1
        return real_construct(device, injected)

    def counting_build(*args, **kwargs):
        builds["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(loop_module, "_construct_accelerator", counting_construct)
    monkeypatch.setattr(loop_module, "_build_runtime", counting_build)

    root = make_job(tmp_path, max_steps=2)
    updated = load_job_config(root)
    updated["max_sequence_length"] = 256
    enqueue_update(root, updated)

    result = run_job(root, **default_sampler_injections())
    assert result == 1
    assert constructed["n"] == 0
    assert builds["n"] == 1
    error = load_job_state(root).last_error or ""
    assert "rebuild teardown failed" in error
    assert "sampler teardown boom" in error


def test_warm_start_rejects_missing_metadata(tmp_path):
    applied: list = []

    def set_lora_state(_model, state_dict):
        applied.append(state_dict)

    def loader(_job_dir):
        return {
            "state_dict": {"to_q.lora_A.weight": torch.ones(2, 16)},
            "metadata": {"adapter_name": "default"},
        }

    root = make_job(tmp_path, max_steps=1)
    with pytest.raises(TrainingConfigError, match="warm-start checkpoint metadata"):
        run_job(
            root,
            **injections(set_lora_state=set_lora_state, load_latest_adapter=loader),
        )
    assert applied == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"base_model_revision": "other-rev"}, "revision"),
        ({"adapter_name": "other-adapter"}, "adapter"),
        ({"peft": {"lora_dropout": 0.9}}, "dropout"),
    ],
)
def test_warm_start_rejects_revision_adapter_dropout_mismatch(
    tmp_path, overrides, match
):
    root = make_job(tmp_path, max_steps=1)
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=1, epoch=0))
    _write_job_checkpoint(root, step=1, **overrides)
    applied: list = []

    def set_lora_state(_model, state_dict):
        applied.append(state_dict)

    with pytest.raises(TrainingConfigError, match=f"warm-start|{match}"):
        run_job(
            root,
            **default_loader_injections(set_lora_state=set_lora_state),
        )
    assert applied == []
