from __future__ import annotations

import inspect
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from diffusers.loaders import PeftAdapterMixin

from zimage.training.quantization import (
    _QUANTIZED_PRECISION_ATTR,
    is_sampling_transformer_quantized,
    quantize_sampling_transformer,
    quantize_text_encoder,
    quantized_precision,
    should_quantize_at_load,
)
from zimage.training.sampling import _quantize_float8_weight_only


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)


class TinyTransformer(PeftAdapterMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(16, 16, bias=False)


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
        init_lora_weights=True,
        target_modules=["to_q"],
    )


def test_should_quantize_at_load():
    assert should_quantize_at_load("fp8", True) is True
    assert should_quantize_at_load(" FP8 ", True) is True
    assert should_quantize_at_load("fp8", False) is False
    assert should_quantize_at_load("bf16", True) is False


def test_unsupported_helper_precision_raises_value_error():
    module = TinyEncoder()
    with pytest.raises(ValueError, match="unsupported quantization precision"):
        quantize_text_encoder(module, precision="int8")
    with pytest.raises(ValueError, match="unsupported quantization precision"):
        quantize_sampling_transformer(module, precision="bf16")
    assert quantized_precision(module) is None


def test_real_text_encoder_conversion_is_idempotent_via_marker(caplog, monkeypatch):
    from torchao.quantization import Float8Tensor, quantize_ as real_quantize

    module = TinyEncoder().to(dtype=torch.bfloat16).eval()
    calls: list[object] = []

    def counting_quantize(target, config):
        calls.append(type(config).__name__)
        return real_quantize(target, config)

    monkeypatch.setattr("torchao.quantization.quantize_", counting_quantize)
    caplog.set_level(logging.INFO, logger="zimage.training")

    quantize_text_encoder(module, precision="fp8")
    assert quantized_precision(module) == "fp8"
    assert getattr(module, _QUANTIZED_PRECISION_ATTR) == "fp8"
    assert isinstance(module.proj.weight, Float8Tensor)
    assert calls == ["Float8WeightOnlyConfig"]
    assert "quantize text encoder precision=fp8" in caplog.text
    assert "DynamicActivation" not in "".join(calls)
    weight_id = id(module.proj.weight)

    caplog.clear()
    quantize_text_encoder(module, precision="fp8")
    assert calls == ["Float8WeightOnlyConfig"]
    assert id(module.proj.weight) == weight_id
    assert "quantize text encoder" not in caplog.text


def test_stub_conversion_sets_marker_and_skips_second_call(caplog, monkeypatch):
    module = TinyEncoder()
    calls: list[object] = []

    def fake_quantize(target, config):
        calls.append(type(config).__name__)

    monkeypatch.setattr("torchao.quantization.quantize_", fake_quantize)
    caplog.set_level(logging.INFO, logger="zimage.training")

    quantize_text_encoder(module)
    assert calls == ["Float8WeightOnlyConfig"]
    assert quantized_precision(module) == "fp8"
    assert is_sampling_transformer_quantized(module) is True
    assert "quantize text encoder precision=fp8" in caplog.text

    caplog.clear()
    quantize_text_encoder(module)
    assert calls == ["Float8WeightOnlyConfig"]
    assert "quantize text encoder" not in caplog.text


def test_stub_marker_skips_conversion_without_torchao_subclass(monkeypatch):
    module = TinyEncoder()
    setattr(module, _QUANTIZED_PRECISION_ATTR, "fp8")
    calls: list[int] = []
    monkeypatch.setattr(
        "torchao.quantization.quantize_",
        lambda *_args, **_kwargs: calls.append(1),
    )

    quantize_sampling_transformer(module)
    assert calls == []
    assert quantized_precision(module) == "fp8"


def test_failed_conversion_does_not_set_marker(monkeypatch, caplog):
    module = TinyEncoder()

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced conversion failure")

    monkeypatch.setattr("torchao.quantization.quantize_", boom)
    caplog.set_level(logging.INFO, logger="zimage.training")
    with pytest.raises(RuntimeError, match="forced conversion failure"):
        quantize_text_encoder(module)
    assert quantized_precision(module) is None
    assert not hasattr(module, _QUANTIZED_PRECISION_ATTR)
    assert "quantize text encoder" not in caplog.text


def test_torchao_subclass_is_detected_without_marker(monkeypatch):
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    module = TinyEncoder().to(dtype=torch.bfloat16).eval()
    quantize_(module, Float8WeightOnlyConfig())
    assert not hasattr(module, _QUANTIZED_PRECISION_ATTR)
    assert quantized_precision(module) == "fp8"
    assert is_sampling_transformer_quantized(module) is True

    calls: list[int] = []
    monkeypatch.setattr(
        "torchao.quantization.quantize_",
        lambda *_args, **_kwargs: calls.append(1),
    )
    quantize_sampling_transformer(module)
    assert calls == []


def test_sampling_helper_attaches_peft_requantizer():
    from peft.tuners.lora.torchao import TorchaoLoraLinear
    from torchao.quantization import Float8WeightOnlyConfig

    transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
    quantize_sampling_transformer(transformer)
    getter = transformer.hf_quantizer.quantization_config.get_apply_tensor_subclass
    assert callable(getter)
    assert isinstance(getter(), Float8WeightOnlyConfig)

    transformer.add_adapter(_lora_config(), adapter_name="preview")
    layers = [
        child
        for child in transformer.modules()
        if isinstance(child, TorchaoLoraLinear)
    ]
    assert layers
    assert all(
        isinstance(child.get_apply_tensor_subclass(), Float8WeightOnlyConfig)
        for child in layers
    )


def test_public_helper_names_omit_precision_literals():
    import zimage.training.quantization as quantization

    literals = ("fp8", "float8", "int8", "bf16")
    public = [
        (name, obj)
        for name, obj in vars(quantization).items()
        if not name.startswith("_") and inspect.isfunction(obj)
    ]
    assert public
    for name, obj in public:
        lowered = name.lower()
        assert not any(token in lowered for token in literals), name
        for parameter in inspect.signature(obj).parameters:
            lowered = parameter.lower()
            assert not any(token in lowered for token in literals), f"{name}.{parameter}"


def test_quantize_float8_weight_only_delegates_to_sampling_helper():
    transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
    _quantize_float8_weight_only(transformer)
    assert quantized_precision(transformer) == "fp8"
    getter = transformer.hf_quantizer.quantization_config.get_apply_tensor_subclass
    assert callable(getter)


def test_training_quantization_import_stays_isolated():
    source = (
        Path(__file__).resolve().parents[3] / "zimage" / "training" / "quantization.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "zimage.engine" not in text
    assert "apply_quantization" not in text
    assert "_scheme_for" not in text
    assert "zimage.training.loop" not in text
    assert "zimage.training.modeling" not in text
    assert "zimage.training.sampling" not in text

    code = """
import sys
import zimage.training.quantization
forbidden = (
    "zimage.engine.quantization",
    "zimage.engine.pipeline",
    "zimage.engine.lora",
    "zimage.training.loop",
    "zimage.training.modeling",
    "zimage.training.sampling",
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    print(",".join(loaded), file=sys.stderr)
raise SystemExit(bool(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_quantization_sampling_modeling_loop_have_no_import_cycle():
    code = """
import zimage.training.quantization
import zimage.training.sampling
import zimage.training.modeling
import zimage.training.loop
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
