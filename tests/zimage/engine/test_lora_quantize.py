"""Real PEFT fuse + torchao quantize on a tiny Linear (no Z-Image weights)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from zimage.engine.lora import LoraSpec, parse_lora_specs, reset_lora_adapters, sync_lora_adapters
from zimage.engine.quantization import apply_quantization

IN = 8
RANK = 2
SCALE = 0.5
# B[0,0]*A[0,0] = 3*2 = 6; fused W[0,0] = SCALE * 6 = 3 when alpha == rank.
DELTA = 6.0


def _make_transformer():
    from diffusers.loaders import PeftAdapterMixin
    from diffusers.models.modeling_utils import ModelMixin

    class Tiny(ModelMixin, PeftAdapterMixin):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(IN, IN, bias=False)
            nn.init.zeros_(self.proj.weight)

    return Tiny()


def _ab_tensors():
    a = torch.zeros(RANK, IN)
    b = torch.zeros(IN, RANK)
    a[0, 0] = 2.0
    b[0, 0] = 3.0
    return a, b


class TinyPipe:
    def __init__(self):
        self.transformer = _make_transformer()

    def load_lora_weights(self, state_dict, weight_name=None, adapter_name=None):
        config = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["proj"], bias="none")
        inject_adapter_in_model(config, self.transformer, adapter_name=adapter_name)
        with torch.no_grad():
            self.transformer.proj.lora_A[adapter_name].weight.copy_(state_dict["lora_A"])
            self.transformer.proj.lora_B[adapter_name].weight.copy_(state_dict["lora_B"])

    def set_adapters(self, names, adapter_weights=None):
        from diffusers.utils.peft_utils import set_weights_and_activate_adapters

        weights = adapter_weights if adapter_weights is not None else [1.0] * len(names)
        set_weights_and_activate_adapters(self.transformer, names, weights)

    def fuse_lora(self, adapter_names=None, lora_scale=1.0):
        self.transformer.fuse_lora(lora_scale, adapter_names=adapter_names)

    def unload_lora_weights(self):
        self.transformer.unload_lora()

    def to(self, device):
        self.transformer.to(device)
        return self


def _write_ab_lora(path: Path) -> None:
    a, b = _ab_tensors()
    save_file({"lora_A": a, "lora_B": b}, str(path))


def _spec(path: Path, scale: float = SCALE) -> LoraSpec:
    return LoraSpec(
        path=path,
        filename=path.name,
        adapter_name=path.stem,
        scale=scale,
    )


def _has_lora_params(module) -> bool:
    return any("lora_" in name for name, _ in module.named_parameters())


def _proj_weight_value(module) -> float:
    weight = module.proj.weight
    if hasattr(weight, "dequantize"):
        weight = weight.dequantize()
    tensor = weight.detach()
    if tensor.ndim == 0:
        return float(tensor)
    return float(tensor.reshape(IN, IN)[0, 0].cpu().float())


@pytest.fixture
def reset_lora():
    reset_lora_adapters()
    yield
    reset_lora_adapters()


def test_fuse_applies_strength_once_then_unloads(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_ab_lora(path)
    pipe = TinyPipe()
    sync_lora_adapters(pipe, (_spec(path, SCALE),))

    assert not _has_lora_params(pipe.transformer)
    assert isinstance(pipe.transformer.proj, nn.Linear)
    assert _proj_weight_value(pipe.transformer) == pytest.approx(SCALE * DELTA)
    assert _proj_weight_value(pipe.transformer) != pytest.approx((SCALE**2) * DELTA)


def test_fuse_then_int8_quantizes_fused_weights(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_ab_lora(path)
    pipe = TinyPipe()
    sync_lora_adapters(pipe, (_spec(path, SCALE),))
    fused = _proj_weight_value(pipe.transformer)

    applied = apply_quantization(pipe, "int8", quantize_text_encoder=False)
    assert applied == "transformer"
    assert not _has_lora_params(pipe.transformer)
    assert "Int8" in type(pipe.transformer.proj.weight).__name__

    x = torch.zeros(1, IN)
    x[0, 0] = 1.0
    y = pipe.transformer.proj(x)
    assert float(y[0, 0].cpu()) == pytest.approx(fused, rel=0.05, abs=0.15)


def test_two_loras_fuse_additively_before_quantize(tmp_path: Path, reset_lora):
    first = tmp_path / "one.safetensors"
    second = tmp_path / "two.safetensors"
    _write_ab_lora(first)
    _write_ab_lora(second)
    pipe = TinyPipe()
    specs = parse_lora_specs(
        str(tmp_path),
        ["one.safetensors", "two.safetensors"],
        [["one.safetensors", SCALE], ["two.safetensors", 1.0]],
    )
    # parse_lora_specs needs real files in a directory listing; write via helper already did.
    sync_lora_adapters(pipe, specs)

    expected = SCALE * DELTA + 1.0 * DELTA
    assert _proj_weight_value(pipe.transformer) == pytest.approx(expected)
    apply_quantization(pipe, "int8", quantize_text_encoder=False)
    x = torch.zeros(1, IN)
    x[0, 0] = 1.0
    y = pipe.transformer.proj(x)
    assert float(y[0, 0].cpu()) == pytest.approx(expected, rel=0.05, abs=0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU fuse/quantize")
def test_fuse_and_quantize_on_cuda(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_ab_lora(path)
    pipe = TinyPipe()
    sync_lora_adapters(pipe, (_spec(path, SCALE),), device="cuda")
    assert next(pipe.transformer.parameters()).device.type == "cuda"
    assert _proj_weight_value(pipe.transformer) == pytest.approx(SCALE * DELTA)

    apply_quantization(pipe, "int8", quantize_text_encoder=False, device="cuda")
    assert not _has_lora_params(pipe.transformer)
    param = next(pipe.transformer.parameters())
    assert param.device.type == "cuda"
    x = torch.zeros(1, IN, device="cuda")
    x[0, 0] = 1.0
    y = pipe.transformer.proj(x)
    assert float(y[0, 0].cpu()) == pytest.approx(SCALE * DELTA, rel=0.05, abs=0.15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU fuse/quantize")
def test_fuse_and_fp8_quantize_on_cuda(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_ab_lora(path)
    pipe = TinyPipe()
    sync_lora_adapters(pipe, (_spec(path, SCALE),), device="cuda")
    apply_quantization(pipe, "fp8", quantize_text_encoder=False, device="cuda")
    assert not _has_lora_params(pipe.transformer)
    x = torch.zeros(1, IN, device="cuda")
    x[0, 0] = 1.0
    y = pipe.transformer.proj(x)
    assert float(y[0, 0].cpu()) == pytest.approx(SCALE * DELTA, rel=0.05, abs=0.2)
