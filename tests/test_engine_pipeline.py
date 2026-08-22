from __future__ import annotations

from pathlib import Path

from PIL import Image

from zimage.engine import ensure_pipeline, save_image, unload_pipeline
from zimage.engine import pipeline as pipeline_mod


def test_save_image_writes_png(tmp_path: Path):
    image = Image.new("RGB", (16, 16), "red")
    path = save_image(image, seed=99, outputs_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
    assert "99" in path.name
    loaded = Image.open(path)
    assert loaded.size == (16, 16)


def test_save_image_uses_default_outputs_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("zimage.engine.pipeline.OUTPUTS_DIR", tmp_path)
    image = Image.new("RGB", (8, 8), "blue")
    path = save_image(image, seed=7)
    assert path.parent == tmp_path
    assert path.exists()


def test_ensure_pipeline_demo_skips_load(monkeypatch):
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "demo")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": True, "demo_reason": "forced"},
    )
    pipe, status = ensure_pipeline("Tongyi-MAI/Z-Image-Turbo", "auto")
    assert pipe is None
    assert status["demo"] is True


def test_ensure_pipeline_reuses_cached_pipe(monkeypatch, reset_pipeline):
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

    first, status_a = ensure_pipeline("model-a", "cpu", "float32", False, False)
    second, status_b = ensure_pipeline("model-a", "cpu", "float32", False, False)
    assert first is fake_pipe
    assert second is fake_pipe
    assert loads["n"] == 1
    assert status_a["loaded"] is True
    assert status_a["precision"] == "float32"
    assert status_b["model"] == "model-a"

    unload_pipeline()
    assert pipeline_mod._pipe is None


def test_ensure_pipeline_reloads_on_precision_change(monkeypatch, reset_pipeline):
    loads: list[str] = []

    def fake_load(_model, _device, dtype_name, _cpu_offload, _vae_tiling, **_kwargs):
        loads.append(dtype_name)
        return object()

    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": False, "cuda": False},
    )
    monkeypatch.setattr("zimage.engine.pipeline.load_pipeline", fake_load)
    monkeypatch.setattr("zimage.engine.pipeline._reclaim_memory", lambda: None)

    ensure_pipeline("model-a", "cpu", "float32", False, False)
    ensure_pipeline("model-a", "cpu", "int8wo", False, False)
    assert loads == ["float32", "int8"]


def test_ensure_pipeline_reloads_on_offload_change(monkeypatch, reset_pipeline):
    keys: list[bool] = []

    def fake_load(_model, _device, _dtype, cpu_offload, _vae_tiling, **_kwargs):
        keys.append(cpu_offload)
        return object()

    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": False, "cuda": True},
    )
    monkeypatch.setattr("zimage.engine.pipeline.load_pipeline", fake_load)
    monkeypatch.setattr("zimage.engine.pipeline._reclaim_memory", lambda: None)

    ensure_pipeline("model-a", "cuda", "float32", False, False)
    ensure_pipeline("model-a", "cuda", "float32", True, False)
    assert keys == [False, True]


def test_ensure_pipeline_reloads_on_quantize_targets_change(monkeypatch, reset_pipeline):
    loads: list[tuple[bool, bool]] = []

    def fake_load(*_args, **kwargs):
        loads.append(
            (kwargs.get("quantize_transformer", True), kwargs.get("quantize_text_encoder", True))
        )
        return object()

    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": False, "cuda": True},
    )
    monkeypatch.setattr("zimage.engine.pipeline.load_pipeline", fake_load)
    monkeypatch.setattr("zimage.engine.pipeline._reclaim_memory", lambda: None)

    ensure_pipeline("model-a", "cuda", "fp8", False, False, True, True)
    ensure_pipeline("model-a", "cuda", "fp8", False, False, True, True)
    ensure_pipeline("model-a", "cuda", "fp8", False, False, True, False)
    assert loads == [(True, True), (True, False)]


def test_pipeline_key_changes_with_lora_identity():
    from zimage.engine.lora import LoraSpec

    base = dict(
        model_id="model-a",
        device="cuda",
        dtype_name="fp8",
        cpu_offload=False,
        vae_tiling=False,
        quantize_transformer=True,
        quantize_text_encoder=True,
    )
    spec_a = LoraSpec(
        path=Path("/loras/style.safetensors"),
        filename="style.safetensors",
        adapter_name="style",
        scale=0.8,
    )
    spec_b = LoraSpec(
        path=Path("/loras/style.safetensors"),
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )
    without = pipeline_mod._pipeline_key(**base, loras=())
    with_a = pipeline_mod._pipeline_key(**base, loras=(spec_a,))
    with_b = pipeline_mod._pipeline_key(**base, loras=(spec_b,))
    assert without != with_a
    assert with_a != with_b
    assert without[-1] == ()
    assert with_a[-1] == ((str(spec_a.path), "style", 0.8),)
    # Quantize flags stay enabled when LoRA is present (fuse-then-quantize).
    assert with_a[5] is True
    assert with_a[6] is True


def test_ensure_pipeline_reloads_on_lora_change(monkeypatch, reset_pipeline):
    from zimage.engine.lora import LoraSpec

    loads: list[tuple] = []
    spec = LoraSpec(
        path=Path("/loras/style.safetensors"),
        filename="style.safetensors",
        adapter_name="style",
        scale=0.8,
    )

    def fake_load(*_args, **kwargs):
        loads.append(tuple(kwargs.get("loras") or ()))
        return object()

    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": False, "cuda": True},
    )
    monkeypatch.setattr("zimage.engine.pipeline.load_pipeline", fake_load)
    monkeypatch.setattr("zimage.engine.pipeline._reclaim_memory", lambda: None)

    ensure_pipeline("model-a", "cuda", "fp8", False, False, loras=())
    ensure_pipeline("model-a", "cuda", "fp8", False, False, loras=(spec,))
    ensure_pipeline("model-a", "cuda", "fp8", False, False, loras=(spec,))
    ensure_pipeline("model-a", "cuda", "fp8", False, False, loras=())
    assert loads == [(), (spec,), ()]


def test_reclaim_memory_without_torch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    pipeline_mod._reclaim_memory()
