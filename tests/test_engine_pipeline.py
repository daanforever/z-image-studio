from __future__ import annotations

import sys
import types
from pathlib import Path

from PIL import Image

from zimage.engine import ensure_pipeline, generate_image, save_image, unload_pipeline
from zimage.engine import pipeline as pipeline_mod


def test_save_image_writes_png(tmp_path: Path):
    image = Image.new("RGB", (16, 16), "red")
    path = save_image(image, seed=99, outputs_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
    assert "99" in path.name
    loaded = Image.open(path)
    assert loaded.size == (16, 16)


def test_generate_image_demo_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZIMAGE_DEMO", "1")
    image, seed, status = generate_image(
        "studio test",
        width=512,
        height=512,
        seed=123,
        outputs_dir=tmp_path,
    )
    assert seed == 123
    assert status["demo"] is True
    assert status["loaded"] is False
    assert Path(status["saved"]).exists()
    assert image.size == (512, 512)


def test_ensure_pipeline_demo_skips_load(monkeypatch):
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "demo")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": True, "demo_reason": "forced"},
    )
    pipe, status = ensure_pipeline("Tongyi-MAI/Z-Image-Turbo", "auto")
    assert pipe is None
    assert status["demo"] is True


def test_ensure_pipeline_reuses_cached_pipe(monkeypatch):
    fake_pipe = object()
    loads = {"n": 0}

    def fake_load(*_args, **_kwargs):
        loads["n"] += 1
        return fake_pipe

    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": False, "cuda": False},
    )
    monkeypatch.setattr("zimage.engine.pipeline.load_pipeline", fake_load)
    pipeline_mod._pipe = None
    pipeline_mod._pipe_key = None
    try:
        first, status_a = ensure_pipeline("model-a", "cpu", "float32", False, False)
        second, status_b = ensure_pipeline("model-a", "cpu", "float32", False, False)
        assert first is fake_pipe
        assert second is fake_pipe
        assert loads["n"] == 1
        assert status_a["loaded"] is True
        assert status_b["model"] == "model-a"
    finally:
        unload_pipeline()
    assert pipeline_mod._pipe is None


def test_load_pipeline_cpu_offload_and_tiling(monkeypatch):
    class Pipe:
        def __init__(self):
            self.moved_to = None
            self.offloaded = False
            self.tiled = False

        def to(self, device):
            self.moved_to = device
            return self

        def enable_model_cpu_offload(self):
            self.offloaded = True

        def enable_vae_tiling(self):
            self.tiled = True

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = Pipe
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    cpu_pipe = pipeline_mod.load_pipeline("model", "cpu", "float32", False, True)
    assert cpu_pipe.moved_to == "cpu"
    assert cpu_pipe.tiled is True
    assert cpu_pipe.offloaded is False

    cuda_pipe = pipeline_mod.load_pipeline("model", "cuda", "float32", True, False)
    assert cuda_pipe.offloaded is True
    assert cuda_pipe.moved_to is None


def test_load_pipeline_int8_quantizes_before_device(monkeypatch):
    order = []

    class Pipe:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            order.append(("to", device))
            self.moved_to = device
            return self

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            order.append("load")
            return cls()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = Pipe
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)

    def fake_apply(pipe):
        order.append("quant")
        return "transformer"

    monkeypatch.setattr("zimage.engine.pipeline.apply_int8_quantization", fake_apply)

    pipe = pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)
    assert pipe.moved_to == "cuda"
    assert order == ["load", "quant", ("to", "cuda")]


def test_load_pipeline_int8_requires_torchao(monkeypatch):
    monkeypatch.setattr(
        "zimage.engine.pipeline.require_torchao",
        lambda: (_ for _ in ()).throw(RuntimeError("int8 precision requires torchao")),
    )
    try:
        pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "torchao" in str(exc)


def test_load_pipeline_falls_back_and_hints(monkeypatch):
    class ZImagePipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("ZImagePipeline is missing")

    class DiffusionPipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("weights missing")

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = ZImagePipeline
    fake_diffusers.DiffusionPipeline = DiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    try:
        pipeline_mod.load_pipeline("some-model", "cpu", "float32", False, False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        message = str(exc)
        assert "some-model" in message
        assert "diffusers" in message
