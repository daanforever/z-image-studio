from __future__ import annotations

from types import SimpleNamespace

from zimage.engine.runtime import dtype_from_name, resolve_device, runtime_status


def test_runtime_status_demo_env(monkeypatch):
    monkeypatch.setenv("ZIMAGE_DEMO", "1")
    status = runtime_status()
    assert status["demo"] is True
    assert "ZIMAGE_DEMO" in status["demo_reason"]
    assert status["device"] == "cpu"


def test_runtime_status_missing_torch(monkeypatch):
    monkeypatch.delenv("ZIMAGE_DEMO", raising=False)
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: None)
    status = runtime_status()
    assert status["demo"] is True
    assert status["demo_reason"] == "PyTorch is not installed"


def test_runtime_status_cpu_torch(monkeypatch):
    monkeypatch.delenv("ZIMAGE_DEMO", raising=False)
    fake = SimpleNamespace(
        __version__="2.10.0+cpu",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    status = runtime_status()
    assert status["torch"] is True
    assert status["cuda"] is False
    assert status["cpu_torch_on_nvidia"] is True
    assert "cu130" in status["demo_reason"]


def test_runtime_status_cuda(monkeypatch):
    monkeypatch.delenv("ZIMAGE_DEMO", raising=False)
    props = SimpleNamespace(total_memory=16 * (1024**3))
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _i: "NVIDIA GeForce RTX 5080",
        get_device_properties=lambda _i: props,
        memory_allocated=lambda _i: 1.5 * (1024**3),
    )
    fake = SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=fake_cuda,
    )
    monkeypatch.setattr("zimage.engine.runtime.try_import_torch", lambda: fake)
    status = runtime_status()
    assert status["cuda"] is True
    assert status["device"] == "cuda"
    assert status["device_name"] == "NVIDIA GeForce RTX 5080"
    assert status["vram"].startswith("1.5 / 16.0")
    assert status["cpu_torch_on_nvidia"] is False


def test_resolve_device_auto_cuda(monkeypatch):
    monkeypatch.setattr(
        "zimage.engine.runtime.runtime_status",
        lambda: {"demo": False, "cuda": True},
    )
    assert resolve_device("auto") == "cuda"
    assert resolve_device("") == "cuda"


def test_resolve_device_falls_back_from_cuda(monkeypatch):
    monkeypatch.setattr(
        "zimage.engine.runtime.runtime_status",
        lambda: {"demo": False, "cuda": False},
    )
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_demo(monkeypatch):
    monkeypatch.setattr(
        "zimage.engine.runtime.runtime_status",
        lambda: {"demo": True, "cuda": False},
    )
    assert resolve_device("auto") == "demo"


def test_dtype_from_name():
    import torch

    assert dtype_from_name("float16") is torch.float16
    assert dtype_from_name("fp32") is torch.float32
    assert dtype_from_name("unknown") is torch.bfloat16
