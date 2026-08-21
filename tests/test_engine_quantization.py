from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from zimage.engine.quantization import (
    apply_int8_quantization,
    is_int8_precision,
    require_torchao,
)


def test_is_int8_precision():
    assert is_int8_precision("int8")
    assert is_int8_precision("INT8WO")
    assert is_int8_precision(" q8 ")
    assert not is_int8_precision("bfloat16")
    assert not is_int8_precision("")
    assert not is_int8_precision(None)


def test_require_torchao_missing(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: None)
    with pytest.raises(RuntimeError, match="pip install torchao"):
        require_torchao()


def test_apply_int8_quantization_weight_only(monkeypatch):
    calls = []

    class Linear:
        pass

    class Transformer:
        def __init__(self):
            self.layer = Linear()

    class Pipe:
        def __init__(self):
            self.transformer = Transformer()

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

    pipe = Pipe()
    assert apply_int8_quantization(pipe) == "transformer"
    assert calls == [(pipe.transformer, "int8wo-scheme")]


def test_apply_int8_quantization_requires_dit(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization.try_import_torchao", lambda: object())
    with pytest.raises(RuntimeError, match="no transformer"):
        apply_int8_quantization(SimpleNamespace())
