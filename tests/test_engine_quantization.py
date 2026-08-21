from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from zimage.engine.quantization import (
    apply_int8_quantization,
    apply_quantization,
    is_fp8_precision,
    is_int8_precision,
    is_quantized_precision,
    require_fp8_device,
    require_torchao,
    try_import_torchao,
)


def test_is_int8_precision():
    assert is_int8_precision("int8")
    assert is_int8_precision("INT8WO")
    assert is_int8_precision(" q8 ")
    assert not is_int8_precision("bfloat16")
    assert not is_int8_precision("fp8")
    assert not is_int8_precision("")
    assert not is_int8_precision(None)


def test_is_fp8_precision():
    assert is_fp8_precision("fp8")
    assert is_fp8_precision("FLOAT8")
    assert is_fp8_precision("float8dq")
    assert is_fp8_precision("fp8dq")
    assert not is_fp8_precision("int8")
    assert not is_fp8_precision("bfloat16")


def test_is_quantized_precision():
    assert is_quantized_precision("int8")
    assert is_quantized_precision("fp8")
    assert not is_quantized_precision("float16")


def test_try_import_torchao_available():
    module = try_import_torchao()
    assert module is not None
    assert hasattr(module, "__name__")


def test_try_import_torchao_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchao", None)
    assert try_import_torchao() is None


def test_require_torchao_ok(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    require_torchao()


def test_require_torchao_missing(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: None)
    with pytest.raises(RuntimeError, match="pip install torchao"):
        require_torchao()


def test_require_fp8_device_rejects_cpu():
    with pytest.raises(RuntimeError, match="fp8 requires CUDA"):
        require_fp8_device("cpu")


def test_require_fp8_device_rejects_missing_torch(monkeypatch):
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        require_fp8_device("cuda")


def test_require_fp8_device_rejects_cpu_cuda_runtime(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        require_fp8_device("cuda")


def test_require_fp8_device_rejects_old_gpu(monkeypatch):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda _i: (8, 6),
            get_device_name=lambda _i: "NVIDIA GeForce RTX 3080",
        )
    )
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    with pytest.raises(RuntimeError, match="8.9"):
        require_fp8_device("cuda")


def test_require_fp8_device_allows_ada(monkeypatch):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda _i: (8, 9),
            get_device_name=lambda _i: "NVIDIA GeForce RTX 4090",
        )
    )
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    require_fp8_device("cuda")


def test_require_fp8_device_allows_blackwell(monkeypatch):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda _i: (12, 0),
            get_device_name=lambda _i: "NVIDIA GeForce RTX 5080",
        )
    )
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    require_fp8_device("cuda")


def _pipe_with_module(attr="transformer"):
    class Transformer:
        def eval(self):
            self.evaluated = True
            return self

    class Pipe:
        def __init__(self):
            setattr(self, attr, Transformer())

    return Pipe()


def test_apply_int8_quantization_weight_only(monkeypatch):
    calls = []
    pipe = _pipe_with_module()

    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setattr(
        "zimage.engine.quantization._int8_weight_only_scheme",
        lambda: "int8wo-scheme",
    )

    def fake_quantize(module, scheme):
        calls.append((module, scheme))

    fake_quant = SimpleNamespace(quantize_=fake_quantize)
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_quant)

    assert apply_int8_quantization(pipe) == "transformer"
    assert calls == [(pipe.transformer, "int8wo-scheme")]
    assert pipe.transformer.evaluated is True


def test_apply_fp8_quantization(monkeypatch):
    calls = []
    pipe = _pipe_with_module()

    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setattr("zimage.engine.quantization._fp8_scheme", lambda: "fp8-scheme")

    def fake_quantize(module, scheme):
        calls.append((module, scheme))

    fake_quant = SimpleNamespace(quantize_=fake_quantize)
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_quant)

    assert apply_quantization(pipe, "fp8") == "transformer"
    assert calls == [(pipe.transformer, "fp8-scheme")]


def test_apply_quantization_uses_dit_module(monkeypatch):
    pipe = _pipe_with_module("dit")
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torchao.quantization",
        SimpleNamespace(quantize_=lambda _module, _scheme: None),
    )
    assert apply_quantization(pipe, "int8") == "dit"


def test_apply_quantization_uses_unet_module(monkeypatch):
    pipe = _pipe_with_module("unet")
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torchao.quantization",
        SimpleNamespace(quantize_=lambda _module, _scheme: None),
    )
    assert apply_quantization(pipe, "int8") == "unet"


def test_apply_int8_quantization_requires_dit(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    with pytest.raises(RuntimeError, match="no transformer"):
        apply_int8_quantization(SimpleNamespace())


def test_apply_quantization_requires_quantize_(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace())
    with pytest.raises(RuntimeError, match="quantize_"):
        apply_quantization(_pipe_with_module(), "int8")


def test_apply_quantization_wraps_torchao_errors(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")

    def boom(_module, _scheme):
        raise ValueError("bad tensor")

    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=boom))
    with pytest.raises(RuntimeError, match="Failed to apply torchao int8 .*transformer"):
        apply_quantization(_pipe_with_module(), "int8")
