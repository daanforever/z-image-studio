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
    should_quantize,
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


def test_should_quantize():
    assert should_quantize("fp8", True, True)
    assert should_quantize("int8", True, False)
    assert should_quantize("fp8", False, True)
    assert not should_quantize("fp8", False, False)
    assert not should_quantize("bfloat16", True, True)


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


class _EvalModule:
    def eval(self):
        self.evaluated = True
        return self


def _pipe_with_module(attr="transformer", text_encoder=False):
    class Pipe:
        def __init__(self):
            setattr(self, attr, _EvalModule())
            if text_encoder:
                self.text_encoder = _EvalModule()

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


def test_apply_quantization_text_encoder_alone_is_not_enough(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    pipe = SimpleNamespace(text_encoder=_EvalModule())
    with pytest.raises(RuntimeError, match="no transformer"):
        apply_quantization(pipe, "int8")


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


def _install_quantize(monkeypatch, quantize_fn):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    monkeypatch.setitem(sys.modules, "torchao", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torchao.quantization",
        SimpleNamespace(quantize_=quantize_fn),
    )


def test_apply_quantization_includes_text_encoder(monkeypatch):
    calls = []
    pipe = _pipe_with_module(text_encoder=True)
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")

    def fake_quantize(module, scheme):
        calls.append((module, scheme))

    _install_quantize(monkeypatch, fake_quantize)

    assert apply_quantization(pipe, "int8") == "transformer, text_encoder"
    assert calls == [
        (pipe.transformer, "scheme"),
        (pipe.text_encoder, "scheme"),
    ]
    assert pipe.transformer.evaluated is True
    assert pipe.text_encoder.evaluated is True


def test_apply_quantization_includes_text_encoder_2(monkeypatch):
    pipe = _pipe_with_module()
    pipe.text_encoder_2 = _EvalModule()
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    calls = []

    def fake_quantize(module, scheme):
        calls.append(module)

    _install_quantize(monkeypatch, fake_quantize)
    assert apply_quantization(pipe, "fp8") == "transformer, text_encoder_2"
    assert calls == [pipe.transformer, pipe.text_encoder_2]


def test_apply_quantization_text_encoder_error_names_module(monkeypatch):
    pipe = _pipe_with_module(text_encoder=True)
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")

    def fake_quantize(module, _scheme):
        if module is pipe.text_encoder:
            raise ValueError("te failed")

    _install_quantize(monkeypatch, fake_quantize)
    with pytest.raises(RuntimeError, match="Failed to apply torchao int8 .*text_encoder"):
        apply_quantization(pipe, "int8")


def test_apply_quantization_skips_missing_text_encoder(monkeypatch):
    pipe = _pipe_with_module()
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    calls = []
    _install_quantize(monkeypatch, lambda module, scheme: calls.append((module, scheme)))
    assert apply_quantization(pipe, "int8") == "transformer"
    assert calls == [(pipe.transformer, "scheme")]


def test_apply_quantization_transformer_only(monkeypatch):
    pipe = _pipe_with_module(text_encoder=True)
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    calls = []
    _install_quantize(monkeypatch, lambda module, _scheme: calls.append(module))
    assert (
        apply_quantization(pipe, "int8", quantize_transformer=True, quantize_text_encoder=False)
        == "transformer"
    )
    assert calls == [pipe.transformer]


def test_apply_quantization_text_encoder_only(monkeypatch):
    pipe = _pipe_with_module(text_encoder=True)
    monkeypatch.setattr("zimage.engine.quantization._scheme_for", lambda _name: "scheme")
    calls = []
    _install_quantize(monkeypatch, lambda module, _scheme: calls.append(module))
    assert (
        apply_quantization(pipe, "fp8", quantize_transformer=False, quantize_text_encoder=True)
        == "text_encoder"
    )
    assert calls == [pipe.text_encoder]


def test_apply_quantization_no_modules_selected(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    with pytest.raises(RuntimeError, match="no modules selected"):
        apply_quantization(_pipe_with_module(text_encoder=True), "int8", False, False)
