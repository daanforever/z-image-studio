from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from zimage.training.cache import CacheConfig, encode_sample
from zimage.training.dataset import DatasetSample
from zimage.training.gpu_usage import (
    GpuUsageProbe,
    GpuUsageSnapshot,
    format_gpu_usage,
    snapshot_gpu_usage,
)
from zimage.training.modeling import (
    ModelSource,
    ModelSources,
    TrainingModelComponents,
    TrainingModelLifecycle,
)
from zimage.training.schema import CACHE_PROMPT_EMBED_HIDDEN_SIZE


def _fake_cuda(*, available: bool, allocated=0, reserved=0, peak=0, calls=None):
    tracked = calls if calls is not None else []

    def memory_allocated():
        tracked.append("allocated")
        return allocated

    def memory_reserved():
        tracked.append("reserved")
        return reserved

    def max_memory_allocated():
        tracked.append("peak")
        return peak

    return SimpleNamespace(
        is_available=lambda: available,
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
        max_memory_allocated=max_memory_allocated,
    )


def _components(*, vae="cpu", text_encoder="cpu", transformer="cpu"):
    return SimpleNamespace(
        vae=SimpleNamespace(device=vae),
        text_encoder=SimpleNamespace(device=text_encoder),
        main_transformer=SimpleNamespace(device=transformer),
    )


def test_snapshot_cuda_absent_zeros_and_skips_memory_apis():
    calls: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=False, allocated=99, calls=calls))
    snap = snapshot_gpu_usage(
        "cache_place",
        _components(vae="cpu", text_encoder="cpu", transformer="cpu"),
        torch_module=fake,
    )
    assert snap.phase == "cache_place"
    assert snap.cuda_available is False
    assert snap.allocated_bytes == 0
    assert snap.reserved_bytes == 0
    assert snap.peak_allocated_bytes == 0
    assert snap.module_devices == {
        "vae": "cpu",
        "text_encoder": "cpu",
        "transformer": "cpu",
    }
    assert calls == []


def test_snapshot_cuda_present_reads_bytes_and_module_devices():
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=1024, reserved=2048, peak=4096)
    )
    snap = snapshot_gpu_usage(
        "cache_place",
        _components(vae="cuda", text_encoder="cuda", transformer="cpu"),
        torch_module=fake,
    )
    assert snap.phase == "cache_place"
    assert snap.cuda_available is True
    assert snap.allocated_bytes == 1024
    assert snap.reserved_bytes == 2048
    assert snap.peak_allocated_bytes == 4096
    assert snap.module_devices == {
        "vae": "cuda",
        "text_encoder": "cuda",
        "transformer": "cpu",
    }


def test_snapshot_cuda_present_via_monkeypatch(monkeypatch):
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=7, reserved=8, peak=9)
    )
    monkeypatch.setattr("zimage.training.gpu_usage.torch", fake)
    snap = snapshot_gpu_usage("train_placed", _components(transformer="cuda"))
    assert snap.cuda_available is True
    assert (snap.allocated_bytes, snap.reserved_bytes, snap.peak_allocated_bytes) == (
        7,
        8,
        9,
    )
    assert snap.module_devices["transformer"] == "cuda"


def test_snapshot_missing_components_are_none():
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    snap = snapshot_gpu_usage("teardown", None, torch_module=fake)
    assert snap.module_devices == {
        "vae": "none",
        "text_encoder": "none",
        "transformer": "none",
    }


def test_snapshot_released_text_encoder_is_none():
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    components = SimpleNamespace(
        vae=SimpleNamespace(device="cpu"),
        text_encoder=None,
        main_transformer=SimpleNamespace(device="cuda"),
    )
    snap = snapshot_gpu_usage("train_placed", components, torch_module=fake)
    assert snap.module_devices == {
        "vae": "cpu",
        "text_encoder": "none",
        "transformer": "cuda",
    }


def test_format_gpu_usage_stable_line():
    snap = GpuUsageSnapshot(
        phase="cache_place",
        cuda_available=True,
        allocated_bytes=1024,
        reserved_bytes=2048,
        peak_allocated_bytes=4096,
        module_devices={
            "vae": "cuda",
            "text_encoder": "cuda",
            "transformer": "cpu",
        },
    )
    assert format_gpu_usage(snap) == (
        "gpu usage phase=cache_place cuda=1 allocated=1024 "
        "reserved=2048 peak_allocated=4096 "
        "vae=cuda text_encoder=cuda transformer=cpu"
    )


def test_snapshot_cuda_is_available_error_does_not_raise():
    class BoomCuda:
        def is_available(self):
            raise RuntimeError("driver exploded")

    snap = snapshot_gpu_usage(
        "teardown", torch_module=SimpleNamespace(cuda=BoomCuda())
    )
    assert snap.cuda_available is False
    assert snap.allocated_bytes == 0


def test_probe_swallows_logger_errors():
    class BoomLogger:
        def info(self, _msg):
            raise RuntimeError("log failed")

    GpuUsageProbe(logger=BoomLogger())("teardown")


_CUDA_MEMORY_SLACK_BYTES = 8 * 1024 * 1024


class _TinyLinearVae(torch.nn.Module):
    """Linear VAE stand-in whose weights occupy CUDA memory when placed."""

    def __init__(self, *, width: int = 64) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
        self.dtype = torch.bfloat16
        self.fail_encode = False

    def encode(self, pixels: torch.Tensor):
        if self.fail_encode:
            raise RuntimeError("forced encode failure")
        batch, _channels, height, width = pixels.shape
        _ = self.proj(pixels.new_ones((batch, self.proj.in_features)))
        latent = pixels.new_zeros((batch, 16, height // 8, width // 8))
        return SimpleNamespace(latent_dist=_TinyLatentDist(latent))


class _TinyLatentDist:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent


class _TinyLinearTextEncoder(torch.nn.Module):
    """Linear text encoder; 2560×2560 bf16 weights exceed the 8 MiB slack."""

    def __init__(self, *, hidden_size: int = CACHE_PROMPT_EMBED_HIDDEN_SIZE) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        projected = self.proj(
            torch.ones(
                batch,
                self.proj.in_features,
                device=input_ids.device,
                dtype=self.proj.weight.dtype,
            )
        )
        hidden = projected.unsqueeze(1).expand(batch, length, -1)
        return SimpleNamespace(hidden_states=[hidden, hidden, hidden])


class _TinyCpuTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)


class _TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[10, 20, 30, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )


def _reclaim_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _warmup_cuda_linear_context() -> None:
    """Establish cuBLAS workspace so cache encode does not inflate baseline."""

    module = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    module.to(device="cuda")
    hidden = torch.ones(1, 16, device="cuda", dtype=torch.bfloat16)
    module(hidden)
    del module, hidden
    _reclaim_cuda()


def _device_type(module: torch.nn.Module) -> str:
    return next(module.parameters()).device.type


def test_tiny_cuda_cache_place_encode_park():
    """Real CUDA place → encode → park; VRAM rises then returns within slack."""

    if not torch.cuda.is_available():
        pytest.skip("tiny-CUDA cache place/encode/park requires CUDA")

    vae = None
    text_encoder = None
    transformer = None
    lifecycle = None
    try:
        _warmup_cuda_linear_context()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        vae = _TinyLinearVae()
        text_encoder = _TinyLinearTextEncoder()
        transformer = _TinyCpuTransformer()
        lifecycle = TrainingModelLifecycle(
            TrainingModelComponents(
                sources=ModelSources(ModelSource("tiny"), ModelSource("tiny")),
                vae=vae,
                tokenizer=_TinyTokenizer(),
                text_encoder=text_encoder,
                training_scheduler=object(),
                main_transformer=transformer,
                sampling_transformer=transformer,
                sampling_scheduler=object(),
            )
        )
        adapter = lifecycle.cache_encoder()
        cuda = torch.device("cuda")
        image = Image.new("RGB", (16, 16), (0, 127, 255))
        sample = DatasetSample(
            image_path=Path("image.png"),
            caption="tiny cuda cache",
            dataset_path=Path("."),
        )
        config = CacheConfig(
            main_revision="tiny",
            vae_config={"shift_factor": 0.0, "scaling_factor": 1.0},
            text_encoder_config={},
            tokenizer_config={},
            qwen_chat_template={},
            max_sequence_length=4,
        )

        lifecycle.place_cache_modules(cuda)
        try:
            assert _device_type(vae) == "cuda"
            assert _device_type(text_encoder) == "cuda"
            assert _device_type(transformer) == "cpu"
            assert torch.cuda.memory_allocated() > baseline

            latent, prompt = encode_sample(sample, image, adapter, config)
            assert latent.device.type == "cpu"
            assert latent.dtype is torch.bfloat16
            assert prompt.device.type == "cpu"
            assert prompt.dtype is torch.bfloat16

            lifecycle.prepare_preview_prompt_embeddings(
                ["tiny cuda preview"],
                max_sequence_length=config.max_sequence_length,
            )
            assert _device_type(text_encoder) == "cuda"
        finally:
            lifecycle.park_cache_modules()

        assert _device_type(vae) == "cpu"
        assert _device_type(text_encoder) == "cpu"
        assert _device_type(transformer) == "cpu"
        assert torch.cuda.memory_allocated() <= baseline + _CUDA_MEMORY_SLACK_BYTES

        lifecycle.place_cache_modules(cuda)
        vae.fail_encode = True
        try:
            with pytest.raises(RuntimeError, match="forced encode failure"):
                encode_sample(sample, image, adapter, config)
        finally:
            lifecycle.park_cache_modules()

        assert _device_type(vae) == "cpu"
        assert _device_type(text_encoder) == "cpu"
        assert torch.cuda.memory_allocated() <= baseline + _CUDA_MEMORY_SLACK_BYTES
    finally:
        if lifecycle is not None:
            lifecycle.park_cache_modules()
        del vae, text_encoder, transformer, lifecycle
        _reclaim_cuda()
