from __future__ import annotations

import logging
import sys
import types

import pytest

from zimage.engine.quantization import (
    _fp8_scheme,
    _int8_weight_only_scheme,
    _scheme_for,
    _scheme_label,
)


def _install_quant(monkeypatch, **attrs):
    quant = types.ModuleType("torchao.quantization")
    for name, value in attrs.items():
        setattr(quant, name, value)
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", quant)
    return quant


def test_int8_scheme_prefers_config(monkeypatch):
    class Int8WeightOnlyConfig:
        pass

    _install_quant(monkeypatch, Int8WeightOnlyConfig=Int8WeightOnlyConfig)
    assert isinstance(_int8_weight_only_scheme(), Int8WeightOnlyConfig)


def test_int8_scheme_falls_back_to_factory(monkeypatch):
    _install_quant(monkeypatch, int8_weight_only=lambda: "legacy-int8")
    assert _int8_weight_only_scheme() == "legacy-int8"


def test_fp8_scheme_prefers_dynamic_config(monkeypatch):
    class Float8DynamicActivationFloat8WeightConfig:
        pass

    _install_quant(
        monkeypatch,
        Float8DynamicActivationFloat8WeightConfig=Float8DynamicActivationFloat8WeightConfig,
    )
    assert isinstance(_fp8_scheme(), Float8DynamicActivationFloat8WeightConfig)


def test_fp8_scheme_falls_back_to_dynamic_factory(monkeypatch):
    _install_quant(
        monkeypatch,
        float8_dynamic_activation_float8_weight=lambda: "legacy-dyn",
    )
    assert _fp8_scheme() == "legacy-dyn"


def test_fp8_scheme_falls_back_to_weight_only_config(monkeypatch, caplog):
    class Float8WeightOnlyConfig:
        pass

    _install_quant(monkeypatch, Float8WeightOnlyConfig=Float8WeightOnlyConfig)
    with caplog.at_level(logging.WARNING, logger="zimage"):
        scheme = _fp8_scheme()
    assert isinstance(scheme, Float8WeightOnlyConfig)
    assert "weight-only fp8" in caplog.text


def test_fp8_scheme_falls_back_to_weight_only_factory(monkeypatch, caplog):
    _install_quant(monkeypatch, float8_weight_only=lambda: "legacy-wo")
    with caplog.at_level(logging.WARNING, logger="zimage"):
        scheme = _fp8_scheme()
    assert scheme == "legacy-wo"
    assert "weight-only fp8" in caplog.text


def test_scheme_for_dispatches(monkeypatch):
    monkeypatch.setattr("zimage.engine.quantization._fp8_scheme", lambda: "fp8-s")
    monkeypatch.setattr("zimage.engine.quantization._int8_weight_only_scheme", lambda: "int8-s")
    assert _scheme_for("fp8") == "fp8-s"
    assert _scheme_for("INT8WO") == "int8-s"
    with pytest.raises(RuntimeError, match="No torchao scheme"):
        _scheme_for("float32")


def test_scheme_label_uses_name_or_type():
    def factory():
        return None

    class EmptyName:
        __name__ = ""

    assert _scheme_label(factory) == "factory"
    assert _scheme_label(EmptyName()) == "EmptyName"
    assert _scheme_label(object()) == "object"
