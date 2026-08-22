from __future__ import annotations

from pathlib import Path

from PIL import Image

from zimage.engine import (
    clear_output_images,
    delete_output_image,
    ensure_pipeline,
    list_output_images,
    save_image,
    unload_pipeline,
)
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


def test_list_output_images_missing_dir(tmp_path: Path):
    missing = tmp_path / "no-such-outputs"
    assert list_output_images(outputs_dir=missing) == []


def test_list_output_images_empty_dir(tmp_path: Path):
    assert list_output_images(outputs_dir=tmp_path) == []


def test_list_output_images_newest_first_png_only(tmp_path: Path):
    import os

    older = tmp_path / "zimage-old.png"
    newer = tmp_path / "zimage-new.png"
    skip = tmp_path / "notes.txt"
    older.write_bytes(b"\x89PNG\r\n\x1a\n")
    newer.write_bytes(b"\x89PNG\r\n\x1a\n")
    skip.write_text("ignore", encoding="utf-8")
    older_mtime = 1_700_000_000.0
    newer_mtime = 1_700_000_100.0
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))
    paths = list_output_images(outputs_dir=tmp_path)
    assert paths == [str(newer), str(older)]

def test_list_output_images_respects_limit(tmp_path: Path):
    import os
    import time

    base = time.time()
    for i in range(5):
        path = tmp_path / f"zimage-{i}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        os.utime(path, (base + i, base + i))
    paths = list_output_images(outputs_dir=tmp_path, limit=3)
    assert len(paths) == 3
    assert paths[0].endswith("zimage-4.png")
    assert paths[-1].endswith("zimage-2.png")


def test_list_output_images_caps_at_gallery_limit(tmp_path: Path, monkeypatch):
    import os

    monkeypatch.setattr("zimage.engine.pipeline.GALLERY_LIMIT", 3)
    for i in range(5):
        path = tmp_path / f"zimage-{i}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))
    paths = list_output_images(outputs_dir=tmp_path)
    assert len(paths) == 3


def test_delete_output_image_removes_png(tmp_path: Path):
    path = tmp_path / "zimage-1.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    deleted = delete_output_image(path, outputs_dir=tmp_path)
    assert deleted == path.resolve()
    assert not path.exists()


def test_delete_output_image_missing_file_succeeds(tmp_path: Path):
    missing = tmp_path / "gone.png"
    deleted = delete_output_image(missing, outputs_dir=tmp_path)
    assert deleted == missing.resolve()
    assert not missing.exists()


def test_delete_output_image_refuses_outside_dir(tmp_path: Path):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        assert delete_output_image(outside, outputs_dir=tmp_path) is None
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_delete_output_image_refuses_non_png(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("keep", encoding="utf-8")
    assert delete_output_image(path, outputs_dir=tmp_path) is None
    assert path.exists()


def test_clear_output_images_missing_dir(tmp_path: Path):
    missing = tmp_path / "no-such-outputs"
    assert clear_output_images(outputs_dir=missing) == 0


def test_clear_output_images_empty_dir(tmp_path: Path):
    assert clear_output_images(outputs_dir=tmp_path) == 0


def test_clear_output_images_removes_image_suffixes_keeps_others(tmp_path: Path):
    png = tmp_path / "a.png"
    jpg = tmp_path / "b.JPG"
    jpeg = tmp_path / "c.jpeg"
    txt = tmp_path / "notes.txt"
    nested = tmp_path / "sub"
    nested.mkdir()
    nested_png = nested / "nested.png"
    for path in (png, jpg, jpeg, nested_png):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    txt.write_text("keep", encoding="utf-8")

    deleted = clear_output_images(outputs_dir=tmp_path)
    assert deleted == 3
    assert not png.exists()
    assert not jpg.exists()
    assert not jpeg.exists()
    assert txt.exists()
    assert nested_png.exists()


def test_clear_output_images_skips_symlink_outside(tmp_path: Path):
    outside = tmp_path.parent / "outside-clear.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        # Symlinks may require elevated privileges on Windows.
        outside.unlink(missing_ok=True)
        return
    try:
        deleted = clear_output_images(outputs_dir=tmp_path)
        assert deleted == 0
        assert outside.exists()
        assert link.exists()
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


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
