from __future__ import annotations

import gc
import inspect
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.loaders import PeftAdapterMixin
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

from zimage.training.checkpoints import (
    write_atomic,
    step_checkpoint_dir,
)
from zimage.training.contracts import (
    NativeAdapterMetadata,
    PreviewSampler,
    SavedCheckpoint,
)
from zimage.training.sampling import (
    PreviewSamplingError,
    UnfusedPreviewSampler,
    _quantize_float8_weight_only,
    resolve_preview_parameters,
)
from zimage.training.schema import merge_sample_parameters


class TinyTransformer(PeftAdapterMixin, torch.nn.Module):
    def __init__(self, *, width: int = 16) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(width, width, bias=False)
        self.proj_out = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.to_q(hidden_states))


class TinyVAE(torch.nn.Module):
    def __init__(self, *, width: int = 16) -> None:
        super().__init__()
        self.decoder = torch.nn.Linear(width, width, bias=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


class RecordingPipeline:
    def __init__(self, transformer: TinyTransformer) -> None:
        self.transformer = transformer
        self.vae = SimpleNamespace(label="shared-vae")
        self.scheduler = SimpleNamespace(config=SimpleNamespace(shift=3.0), shift=3.0)
        self.text_encoder = None
        self.tokenizer = None
        self.calls: list[dict] = []
        self.fail = False
        self.events: list[str] | None = None

    def __call__(self, **kwargs):
        if self.events is not None:
            self.events.append("forward")
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("forced sampling failure")
        hidden = kwargs.get("prompt_embeds")
        if isinstance(hidden, list) and hidden and isinstance(hidden[0], torch.Tensor):
            tokens = hidden[0]
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            self.transformer(tokens)
        width = int(kwargs.get("width", 8))
        height = int(kwargs.get("height", 8))
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (20, 40, 60))])


def _lora_config(*, rank: int = 2, alpha: int = 8) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        init_lora_weights=True,
        target_modules=["to_q"],
    )


def _adapter_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None:
            names.update(lora_a.keys())
    return names


def _assert_unfused(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "merged"):
            assert module.merged is False
        if hasattr(module, "merged_adapters"):
            assert module.merged_adapters == []


def _filled_state(model: TinyTransformer, adapter_name: str, sign: float) -> dict:
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    return {
        key: torch.full_like(value, sign * 0.25 if ".lora_B." in key else 1.0)
        for key, value in state.items()
    }


def _peft_mapping(config: LoraConfig) -> dict:
    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
        "peft_type": "LORA",
    }


def _write_adapter_checkpoint(
    job: Path,
    *,
    step: int,
    sign: float,
    adapter_name: str = "preview",
    alpha: int = 8,
) -> SavedCheckpoint:
    config = _lora_config(alpha=alpha)
    donor = TinyTransformer().to(dtype=torch.bfloat16)
    donor.add_adapter(config, adapter_name=adapter_name)
    state = _filled_state(donor, adapter_name, sign)
    return write_atomic(
        destination=step_checkpoint_dir(job, step),
        lora_state=state,
        metadata=NativeAdapterMetadata(
            adapter_name=adapter_name,
            base_model_name_or_path="org/z-image",
            base_model_revision=None,
            peft_config=_peft_mapping(config),
            optimizer_step=step,
        ),
    )


def _embeddings() -> dict[str, torch.Tensor]:
    return {
        "a cat": torch.ones(3, 16, dtype=torch.bfloat16),
        "subject": torch.full((3, 16), 2.0, dtype=torch.bfloat16),
        "": torch.zeros(3, 16, dtype=torch.bfloat16),
    }


def test_sampler_implements_preview_sampler_protocol():
    sampler = UnfusedPreviewSampler(transformer=TinyTransformer())
    assert isinstance(sampler, PreviewSampler)


def test_unfused_adapter_replacement_leaves_base_weights_unchanged(tmp_path):
    torch.manual_seed(23)
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    first = _write_adapter_checkpoint(job, step=1, sign=1.0, adapter_name="first")
    second = _write_adapter_checkpoint(job, step=2, sign=-1.0, adapter_name="second")

    transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
    base_q = transformer.to_q.weight.detach().clone()
    base_out = transformer.proj_out.weight.detach().clone()
    hidden = torch.randn(2, 16, dtype=torch.bfloat16)
    base_output = transformer(hidden).detach()

    pipeline = RecordingPipeline(transformer)
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        vae=pipeline.vae,
        scheduler=pipeline.scheduler,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        target_modules=["to_q"],
    )

    sampler.load_unfused_adapter(first)
    first_output = transformer(hidden).detach()
    assert _adapter_names(transformer) == {"first"}
    _assert_unfused(transformer)

    sampler.load_unfused_adapter(second)
    second_output = transformer(hidden).detach()
    repeated = transformer(hidden).detach()

    assert _adapter_names(transformer) == {"second"}
    assert set(transformer.peft_config) == {"second"}
    assert all("first" not in name for name, _ in transformer.named_parameters())
    _assert_unfused(transformer)
    assert torch.equal(transformer.to_q.get_base_layer().weight, base_q)
    assert torch.equal(transformer.proj_out.weight, base_out)
    assert not torch.equal(first_output, second_output)
    assert not torch.equal(base_output, second_output)
    assert torch.equal(second_output, repeated)


def test_merge_common_and_per_sample_parameters_is_deterministic():
    common = {
        "guidance_scale": 0.0,
        "time_shift": 3.0,
        "num_inference_steps": 9,
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "prompt": "",
        "negative_prompt": "",
    }
    sample = {"prompt": "subject", "seed": 7, "width": 512, "time_shift": 4.5}
    merged = resolve_preview_parameters(common, sample)
    assert merged == merge_sample_parameters(common, sample)
    assert merged["prompt"] == "subject"
    assert merged["seed"] == 7
    assert merged["width"] == 512
    assert merged["time_shift"] == 4.5
    assert merged["height"] == 1024
    assert merged["guidance_scale"] == 0.0
    again = resolve_preview_parameters(common, sample)
    assert again == merged


def test_sample_unfused_uses_merged_parameters_and_preencoded_embeds(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=3, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    common = {
        "guidance_scale": 0.0,
        "time_shift": 3.0,
        "num_inference_steps": 9,
        "width": 64,
        "height": 32,
        "seed": 42,
        "prompt": "a cat",
        "negative_prompt": "",
    }
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        vae=pipeline.vae,
        scheduler=pipeline.scheduler,
        prompt_embeddings=_embeddings(),
        common_parameters=common,
    )
    destination = tmp_path / "previews" / "step-3.png"
    path = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "subject", "seed": 11, "width": 48},
        destination=destination,
    )
    assert path == destination
    assert destination.is_file()
    with Image.open(destination) as preview:
        assert preview.format == "PNG"
    assert sampler.last_parameters["prompt"] == "subject"
    assert sampler.last_parameters["seed"] == 11
    assert sampler.last_parameters["width"] == 48
    assert sampler.last_parameters["height"] == 32
    assert sampler.last_parameters["time_shift"] == 3.0
    assert pipeline.scheduler.shift == 3.0
    assert pipeline.calls
    call = pipeline.calls[0]
    assert call["prompt"] is None
    assert call["prompt_embeds"][0].shape[-1] == 16
    assert torch.equal(call["prompt_embeds"][0], _embeddings()["subject"].to(call["prompt_embeds"][0].device))
    assert pipeline.text_encoder is None
    assert pipeline.tokenizer is None
    assert pipeline.vae.label == "shared-vae"


def test_sample_unfused_jpeg_destination_writes_jpeg_and_unlinks_siblings(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
    )
    destination = tmp_path / "00001-00-sample.jpg"
    png_sibling = tmp_path / "00001-00-sample.png"
    jpeg_sibling = tmp_path / "00001-00-sample.jpeg"
    png_sibling.write_bytes(b"old-png")
    jpeg_sibling.write_bytes(b"old-jpeg")
    other_step = tmp_path / "00002-00-sample.png"
    other_step.write_bytes(b"keep")

    path = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=destination,
    )
    assert path == destination
    assert destination.is_file()
    assert not png_sibling.exists()
    assert not jpeg_sibling.exists()
    assert other_step.read_bytes() == b"keep"
    with Image.open(destination) as preview:
        assert preview.format == "JPEG"


def test_write_preview_image_png_codec_unlinks_jpeg_siblings(tmp_path):
    from zimage.training.sampling import _write_preview_image

    destination = tmp_path / "00001-00-sample.png"
    jpg_sibling = tmp_path / "00001-00-sample.jpg"
    jpeg_sibling = tmp_path / "00001-00-sample.jpeg"
    jpg_sibling.write_bytes(b"old-jpg")
    jpeg_sibling.write_bytes(b"old-jpeg")
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    _write_preview_image(image, destination)
    assert destination.is_file()
    assert not jpg_sibling.exists()
    assert not jpeg_sibling.exists()
    with Image.open(destination) as preview:
        assert preview.format == "PNG"


def test_sampling_failure_does_not_delete_checkpoint(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=8, sign=1.0)
    weights = checkpoint.path / "pytorch_lora_weights.safetensors"
    before = weights.read_bytes()
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    pipeline.fail = True
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
    )
    with pytest.raises(PreviewSamplingError, match="forced sampling failure"):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "broken.png",
        )
    assert checkpoint.path.is_dir()
    assert weights.is_file()
    assert weights.read_bytes() == before
    assert not (tmp_path / "broken.png").exists()


def test_preview_forward_and_cleanup_failure_are_chained(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=8, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    pipeline.fail = True
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
    )

    def boom_release():
        raise RuntimeError("cleanup boom")

    sampler.release_after_preview = boom_release  # type: ignore[method-assign]
    with pytest.raises(PreviewSamplingError, match="forced sampling failure") as caught:
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "broken.png",
        )
    message = str(caught.value)
    assert "additionally" in message
    assert "cleanup boom" in message or "release_after_preview" in message


def test_coerce_image_rejects_size_mismatch_without_resize():
    from zimage.training.sampling import _coerce_image

    result = SimpleNamespace(images=[Image.new("RGB", (8, 8), (1, 2, 3))])
    with pytest.raises(PreviewSamplingError, match="does not match"):
        _coerce_image(result, width=16, height=16)


def test_empty_negative_zeros_nonempty_missing_raises(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    embeds = _embeddings()
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=embeds,
        negative_prompt_embeddings={},
        common_parameters={
            "prompt": "a cat",
            "negative_prompt": "",
            "width": 16,
            "height": 16,
        },
    )
    path = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat", "negative_prompt": ""},
        destination=tmp_path / "ok.png",
    )
    assert path.is_file()
    assert pipeline.calls
    neg = pipeline.calls[0]["negative_prompt_embeds"][0]
    assert torch.equal(neg, torch.zeros_like(neg))

    with pytest.raises(PreviewSamplingError, match="missing negative_prompt"):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat", "negative_prompt": "foo"},
            destination=tmp_path / "missing-neg.png",
        )


def test_missing_preencoded_embedding_does_not_reload_text_encoder(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "unknown caption"},
    )
    with pytest.raises(PreviewSamplingError, match="does not reload a text encoder"):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "unknown caption"},
            destination=tmp_path / "nope.png",
        )
    assert pipeline.calls == []
    assert checkpoint.path.is_dir()


def test_sampling_module_does_not_import_fuse_loader():
    source = Path(__file__).resolve().parents[3] / "zimage" / "training" / "sampling.py"
    text = source.read_text(encoding="utf-8")
    assert "zimage.engine.lora" not in text
    assert "zimage.engine.pipeline" not in text
    assert "apply_quantization" not in text
    assert "fuse_lora" not in text
    assert "sync_lora_adapters" not in text

    code = """
import sys
import zimage.training.sampling
forbidden = [name for name in ("zimage.engine.lora", "zimage.engine.quantization", "zimage.engine.pipeline") if name in sys.modules]
if forbidden:
    print(",".join(forbidden), file=sys.stderr)
raise SystemExit(bool(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_from_components_builds_sampler_without_pipeline():
    transformer = TinyTransformer()
    scheduler = SimpleNamespace(label="sampling-scheduler")
    vae = SimpleNamespace(label="shared-vae")
    embeddings = _embeddings()
    common = {"prompt": "a cat", "time_shift": 3.0}
    sampler = UnfusedPreviewSampler.from_components(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        prompt_embeddings=embeddings,
        negative_prompt_embeddings={"": embeddings[""]},
        common_parameters=common,
        device="cpu",
        target_modules=["to_q"],
        main_transformer=transformer,
    )
    assert isinstance(sampler, PreviewSampler)
    assert sampler.pipeline is None
    assert sampler.transformer is transformer
    assert sampler.scheduler is scheduler
    assert sampler.vae is vae
    assert sampler.prompt_embeddings["a cat"] is embeddings["a cat"]
    assert sampler.negative_prompt_embeddings[""] is embeddings[""]
    assert sampler.common_parameters == common
    assert sampler.device == "cpu"
    assert sampler.target_modules == ["to_q"]
    assert sampler.main_transformer is transformer
    assert sampler._gpu_usage_probe is None
    signature = inspect.signature(UnfusedPreviewSampler.from_components)
    assert list(signature.parameters) == [
        "transformer",
        "scheduler",
        "vae",
        "prompt_embeddings",
        "negative_prompt_embeddings",
        "common_parameters",
        "device",
        "target_modules",
        "main_transformer",
        "quantizer",
        "device_mover",
        "cuda_cleanup",
        "gpu_usage_probe",
    ]


def test_sample_unfused_probes_preview_run_before_release(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    events: list[str] = []
    pipeline.events = events
    contexts: list[object] = []

    def probe(phase, context=None):
        events.append(f"probe:{phase}")
        contexts.append(context)

    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        gpu_usage_probe=probe,
    )
    original_release = sampler.release_after_preview

    def tracking_release():
        events.append("release_after_preview")
        original_release()

    sampler.release_after_preview = tracking_release  # type: ignore[method-assign]
    path = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=tmp_path / "preview.png",
    )
    assert path.is_file()
    assert "probe:preview_run" in events
    assert "release_after_preview" in events
    assert events.index("forward") < events.index("probe:preview_run")
    assert events.index("probe:preview_run") < events.index("release_after_preview")
    assert contexts
    assert getattr(contexts[0], "phase_peak_bytes", None) is not None


def test_preview_run_probe_error_does_not_fail_sample(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    released = {"n": 0}

    def boom(phase, context=None):
        raise RuntimeError("probe failed")

    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        gpu_usage_probe=boom,
    )
    original_release = sampler.release_after_preview

    def tracking_release():
        released["n"] += 1
        original_release()

    sampler.release_after_preview = tracking_release  # type: ignore[method-assign]
    path = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=tmp_path / "preview.png",
    )
    assert path.is_file()
    assert released["n"] == 1


def test_failed_pipeline_still_probes_then_releases(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=1, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    pipeline.fail = True
    events: list[str] = []
    pipeline.events = events

    def probe(phase, context=None):
        events.append(f"probe:{phase}")

    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        gpu_usage_probe=probe,
    )
    original_release = sampler.release_after_preview

    def tracking_release():
        events.append("release_after_preview")
        original_release()

    sampler.release_after_preview = tracking_release  # type: ignore[method-assign]
    with pytest.raises(PreviewSamplingError, match="forced sampling failure"):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "broken.png",
        )
    assert events.index("forward") < events.index("probe:preview_run")
    assert events.index("probe:preview_run") < events.index("release_after_preview")


def test_preview_time_shift_installs_flow_match_scheduler(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=4, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    original = pipeline.scheduler
    sampler = UnfusedPreviewSampler.from_components(
        transformer=transformer,
        scheduler=original,
        vae=pipeline.vae,
        prompt_embeddings=_embeddings(),
        common_parameters={
            "prompt": "a cat",
            "width": 16,
            "height": 16,
            "time_shift": 3.0,
        },
        device="cpu",
        target_modules=["to_q"],
        main_transformer=transformer,
    )
    sampler.pipeline = pipeline

    first = tmp_path / "shift-3.png"
    sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=first,
    )
    first_scheduler = pipeline.scheduler
    assert first_scheduler is not original
    assert isinstance(first_scheduler, FlowMatchEulerDiscreteScheduler)
    assert first_scheduler.shift == 3.0
    assert first_scheduler.config.shift == 3.0
    assert first_scheduler.config.num_train_timesteps == 1000
    assert "time_shift" not in pipeline.calls[0]

    second = tmp_path / "shift-45.png"
    sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat", "time_shift": 4.5},
        destination=second,
    )
    second_scheduler = pipeline.scheduler
    assert isinstance(second_scheduler, FlowMatchEulerDiscreteScheduler)
    assert second_scheduler.shift == 4.5
    assert second_scheduler.config.shift == 4.5
    assert second_scheduler.config.num_train_timesteps == 1000
    assert not torch.equal(first_scheduler.sigmas, second_scheduler.sigmas)
    assert "time_shift" not in pipeline.calls[1]
    assert first.is_file()
    assert second.is_file()
    assert checkpoint.path.is_dir()


def _cuda_lifecycle_sampler(
    transformer: TinyTransformer,
    pipeline: RecordingPipeline,
    events: list,
) -> UnfusedPreviewSampler:
    original_add = transformer.add_adapter
    original_delete = transformer.delete_adapters

    def add_adapter(*args, **kwargs):
        events.append("add_adapter")
        return original_add(*args, **kwargs)

    def delete_adapters(*args, **kwargs):
        events.append("remove_adapter")
        return original_delete(*args, **kwargs)

    transformer.add_adapter = add_adapter
    transformer.delete_adapters = delete_adapters
    pipeline.events = events

    def move(component, device):
        label = "transformer" if component is transformer else "vae"
        events.append(f"move_{label}_{device.type}")

    return UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        vae=pipeline.vae,
        prompt_embeddings={"a cat": object(), "": object()},
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        device="cuda",
        quantizer=lambda model: events.append("quantize"),
        device_mover=move,
        cuda_cleanup=lambda device: events.append("cuda_cleanup"),
    )


def test_cuda_preview_orders_quantization_adapter_and_device_lifecycle(
    tmp_path, monkeypatch
):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=11, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    events: list[str] = []
    sampler = _cuda_lifecycle_sampler(transformer, pipeline, events)
    monkeypatch.setattr(
        "zimage.training.sampling._make_generator",
        lambda device, seed: SimpleNamespace(device=device, seed=seed),
    )

    sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=tmp_path / "cuda-preview.png",
    )

    assert events.index("quantize") < events.index("add_adapter")
    assert events.index("add_adapter") < events.index("move_transformer_cuda")
    assert events.index("move_transformer_cuda") < events.index("forward")
    assert events.index("move_vae_cuda") < events.index("forward")
    assert events.index("forward") < events.index("remove_adapter")
    assert events.index("remove_adapter") < events.index("move_transformer_cpu", 2)
    assert events.index("remove_adapter") < events.index("move_vae_cpu", 2)
    assert events[-1] == "cuda_cleanup"
    assert _adapter_names(transformer) == set()


def test_cuda_preview_cleans_up_after_forward_failure(tmp_path, monkeypatch):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=12, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    pipeline.fail = True
    events: list[str] = []
    sampler = _cuda_lifecycle_sampler(transformer, pipeline, events)
    monkeypatch.setattr(
        "zimage.training.sampling._make_generator",
        lambda device, seed: SimpleNamespace(device=device, seed=seed),
    )

    with pytest.raises(PreviewSamplingError, match="forced sampling failure"):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "failed-preview.png",
        )

    assert "remove_adapter" in events
    assert events.index("forward") < events.index("remove_adapter")
    assert events.index("remove_adapter") < events.index("move_transformer_cpu", 2)
    assert events.index("remove_adapter") < events.index("move_vae_cpu", 2)
    assert events[-1] == "cuda_cleanup"
    assert _adapter_names(transformer) == set()


def test_cuda_preview_quantizes_base_once_across_adapter_replacements(
    tmp_path, monkeypatch
):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    first = _write_adapter_checkpoint(job, step=13, sign=1.0, adapter_name="first")
    second = _write_adapter_checkpoint(job, step=14, sign=-1.0, adapter_name="second")
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)
    events: list[str] = []
    sampler = _cuda_lifecycle_sampler(transformer, pipeline, events)
    monkeypatch.setattr(
        "zimage.training.sampling._make_generator",
        lambda device, seed: SimpleNamespace(device=device, seed=seed),
    )

    for index, checkpoint in enumerate((first, second), start=1):
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters={"prompt": "a cat"},
            destination=tmp_path / f"{index}.png",
        )

    assert events.count("quantize") == 1
    assert events.count("add_adapter") == 2
    assert events.count("remove_adapter") == 2
    assert _adapter_names(transformer) == set()


def test_cpu_preview_skips_quantization_and_never_fuses(tmp_path):
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    checkpoint = _write_adapter_checkpoint(job, step=15, sign=1.0)
    transformer = TinyTransformer().to(dtype=torch.bfloat16)
    pipeline = RecordingPipeline(transformer)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preview must not quantize or fuse")

    transformer.fuse_lora = forbidden
    sampler = UnfusedPreviewSampler(
        transformer=transformer,
        pipeline=pipeline,
        vae=pipeline.vae,
        prompt_embeddings=_embeddings(),
        common_parameters={"prompt": "a cat", "width": 16, "height": 16},
        device="cpu",
        quantizer=forbidden,
    )

    result = sampler.sample_unfused(
        checkpoint=checkpoint,
        parameters={"prompt": "a cat"},
        destination=tmp_path / "cpu-preview.png",
    )

    assert result.is_file()
    assert _adapter_names(transformer) == set()


def test_float8_preview_quantizer_exposes_peft_requantizer():
    from peft.tuners.lora.torchao import TorchaoLoraLinear
    from torchao.quantization import Float8WeightOnlyConfig

    transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
    _quantize_float8_weight_only(transformer)
    getter = transformer.hf_quantizer.quantization_config.get_apply_tensor_subclass
    assert callable(getter)
    assert isinstance(getter(), Float8WeightOnlyConfig)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformer.add_adapter(_lora_config(), adapter_name="preview")
    assert not any(
        "get_apply_tensor_subclass" in str(item.message) for item in caught
    )
    layers = [
        module
        for module in transformer.modules()
        if isinstance(module, TorchaoLoraLinear)
    ]
    assert layers
    assert all(module.get_apply_tensor_subclass is not None for module in layers)
    assert all(
        isinstance(module.get_apply_tensor_subclass(), Float8WeightOnlyConfig)
        for module in layers
    )


def test_cuda_survivor_scan_does_not_probe_deprecated_reduce_op():
    import torch.distributed  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survivors = _cuda_tensor_survivors()
    assert isinstance(survivors, list)
    assert not any("reduce_op" in str(item.message) for item in caught)


_FP8_MIN_CAPABILITY = (8, 9)
_CUDA_MEMORY_SLACK_BYTES = 8 * 1024 * 1024


def _fp8_cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("real CUDA preview lifecycle requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability < _FP8_MIN_CAPABILITY:
        pytest.skip(
            "real CUDA preview lifecycle requires FP8 compute capability "
            f"8.9+, got {capability}"
        )


def _reclaim_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _warmup_cuda_fp8_context() -> None:
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    module = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    quantize_(module, Float8WeightOnlyConfig())
    module.to(device="cuda")
    hidden = torch.ones(1, 16, device="cuda", dtype=torch.bfloat16)
    module(hidden)
    del module, hidden
    _reclaim_cuda()


def _base_weight(model: torch.nn.Module) -> torch.Tensor:
    layer = model.to_q
    getter = getattr(layer, "get_base_layer", None)
    if callable(getter):
        layer = getter()
    return layer.weight


def _parameter_devices(module: torch.nn.Module) -> set[torch.device]:
    return {parameter.device for parameter in module.parameters()}


def _assert_module_device_type(module: torch.nn.Module, device_type: str) -> None:
    devices = _parameter_devices(module)
    assert devices, f"{type(module).__name__} has no parameters"
    assert {device.type for device in devices} == {device_type}


def _flatten_tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        items: list[torch.Tensor] = []
        for item in value:
            items.extend(_flatten_tensors(item))
        return items
    return []


def _cuda_tensor_survivors() -> list[tuple]:
    survivors: list[tuple] = []
    for obj in gc.get_objects():
        try:
            # type()/issubclass avoids torch.is_tensor() / isinstance(..., Tensor)
            # on gc objects such as torch.distributed.reduce_op, whose deprecated
            # __getattribute__ fires on any isinstance probe.
            if not issubclass(type(obj), torch.Tensor):
                continue
            if obj.device.type == "cuda":
                survivors.append((tuple(obj.shape), str(obj.dtype), str(obj.device)))
        except Exception:
            continue
    return survivors


def _forbid_fuse_or_merge(model: torch.nn.Module):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unfused CUDA preview must not fuse or merge")

    model.fuse_lora = forbidden
    for name in ("merge_and_unload", "merge_adapter", "merge"):
        if hasattr(model, name):
            setattr(model, name, forbidden)


class CudaLifecyclePipeline:
    """Lightweight pipeline that still consumes the real transformer and VAE."""

    def __init__(self, transformer: TinyTransformer, vae: TinyVAE) -> None:
        self.transformer = transformer
        self.vae = vae
        self.scheduler = SimpleNamespace(config=SimpleNamespace(shift=3.0), shift=3.0)
        self.text_encoder = None
        self.tokenizer = None
        self.fail = False
        self.forwards = 0

    def __call__(self, **kwargs):
        self._assert_live_cuda_contract(kwargs)
        hidden = kwargs.get("prompt_embeds")
        tokens = hidden[0] if isinstance(hidden, list) and hidden else hidden
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        hidden_out = self.transformer(tokens)
        self.vae(hidden_out)
        self.forwards += 1
        if self.fail:
            raise RuntimeError("forced sampling failure")
        width = int(kwargs.get("width", 8))
        height = int(kwargs.get("height", 8))
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (20, 40, 60))])

    def _assert_live_cuda_contract(self, kwargs: dict) -> None:
        transformer_devices = _parameter_devices(self.transformer)
        vae_devices = _parameter_devices(self.vae)
        assert {device.type for device in transformer_devices} == {"cuda"}
        assert vae_devices == transformer_devices
        cuda_device = next(iter(transformer_devices))
        base = _base_weight(self.transformer)
        assert base.device == cuda_device
        prompt_tensors = _flatten_tensors(kwargs.get("prompt_embeds"))
        negative_tensors = _flatten_tensors(kwargs.get("negative_prompt_embeds"))
        assert prompt_tensors
        assert negative_tensors
        assert all(tensor.device == cuda_device for tensor in prompt_tensors)
        assert all(tensor.device == cuda_device for tensor in negative_tensors)
        assert _adapter_names(self.transformer)
        _assert_unfused(self.transformer)


def test_sample_unfused_real_cuda_lifecycle_quantizes_once_and_restores_cpu(tmp_path):
    """Production sampler lifecycle on tiny CUDA modules. No injected mover/quantizer."""

    _fp8_cuda_or_skip()
    from torchao.quantization import Float8Tensor

    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    first = _write_adapter_checkpoint(job, step=21, sign=1.0, adapter_name="first")
    second = _write_adapter_checkpoint(job, step=22, sign=-1.0, adapter_name="second")

    _warmup_cuda_fp8_context()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    transformer = None
    vae = None
    pipeline = None
    sampler = None
    metrics: dict[str, object] = {}
    try:
        transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
        vae = TinyVAE().to(dtype=torch.bfloat16).eval()
        pipeline = CudaLifecyclePipeline(transformer, vae)
        _forbid_fuse_or_merge(transformer)

        sampler = UnfusedPreviewSampler(
            transformer=transformer,
            pipeline=pipeline,
            vae=vae,
            prompt_embeddings=_embeddings(),
            common_parameters={"prompt": "a cat", "width": 16, "height": 16},
            device="cuda",
            target_modules=["to_q"],
        )
        assert sampler._quantizer is None
        assert sampler._device_mover is None
        assert sampler._cuda_cleanup is None
        assert sampler._fp8_quantized is False
        assert not isinstance(_base_weight(transformer), Float8Tensor)
        _assert_module_device_type(transformer, "cpu")
        _assert_module_device_type(vae, "cpu")

        first_path = sampler.sample_unfused(
            checkpoint=first,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "first.png",
        )
        assert first_path.is_file()
        assert sampler._fp8_quantized is True
        requantizer = transformer.hf_quantizer.quantization_config.get_apply_tensor_subclass
        assert callable(requantizer)
        assert type(requantizer()).__name__ == "Float8WeightOnlyConfig"
        assert _adapter_names(transformer) == set()
        _assert_module_device_type(transformer, "cpu")
        _assert_module_device_type(vae, "cpu")
        first_weight = _base_weight(transformer)
        assert isinstance(first_weight, Float8Tensor)
        assert first_weight.device.type == "cpu"
        first_weight_id = id(first_weight)

        second_path = sampler.sample_unfused(
            checkpoint=second,
            parameters={"prompt": "a cat"},
            destination=tmp_path / "second.png",
        )
        assert second_path.is_file()
        assert sampler._fp8_quantized is True
        second_weight = _base_weight(transformer)
        assert isinstance(second_weight, Float8Tensor)
        assert id(second_weight) == first_weight_id
        assert _adapter_names(transformer) == set()
        _assert_module_device_type(transformer, "cpu")
        _assert_module_device_type(vae, "cpu")
        assert pipeline.forwards == 2

        pipeline.fail = True
        with pytest.raises(PreviewSamplingError, match="forced sampling failure"):
            sampler.sample_unfused(
                checkpoint=first,
                parameters={"prompt": "a cat"},
                destination=tmp_path / "failed.png",
            )
        assert not (tmp_path / "failed.png").exists()
        assert pipeline.forwards == 3
        assert _adapter_names(transformer) == set()
        _assert_module_device_type(transformer, "cpu")
        _assert_module_device_type(vae, "cpu")
        assert sampler._fp8_quantized is True
        assert isinstance(_base_weight(transformer), Float8Tensor)
        assert id(_base_weight(transformer)) == first_weight_id
    finally:
        metrics["peak_allocated"] = torch.cuda.max_memory_allocated()
        metrics["peak_reserved"] = torch.cuda.max_memory_reserved()
        sampler = None
        pipeline = None
        transformer = None
        vae = None
        _reclaim_cuda()
        metrics["post_allocated"] = torch.cuda.memory_allocated()
        metrics["post_reserved"] = torch.cuda.memory_reserved()
        metrics["survivors"] = _cuda_tensor_survivors()

    assert metrics["survivors"] == []
    assert metrics["post_allocated"] <= baseline_alloc + _CUDA_MEMORY_SLACK_BYTES
    assert metrics["post_reserved"] <= baseline_reserved + _CUDA_MEMORY_SLACK_BYTES
