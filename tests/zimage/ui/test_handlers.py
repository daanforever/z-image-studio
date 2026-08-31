from __future__ import annotations

import inspect
import time
from pathlib import Path

from PIL import Image

import gradio as gr
import pytest

from zimage.config import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, OUTPUTS_DIR
from zimage.ui.handlers import (
    _image_progress,
    clear_preview_images,
    delete_preview_image,
    generate,
    load_gallery,
    load_gallery_with_index,
    load_model,
    refresh_loras,
    request_stop,
    restore_ui_prefs,
    save_ui_prefs,
    set_gallery_index,
    sync_lora_weights,
    unload_model,
)
import zimage.ui.handlers as handlers


def test_load_gallery_returns_disk_paths(monkeypatch):
    captured = {}

    def fake_list(outputs_dir=None):
        captured["outputs_dir"] = outputs_dir
        return ["outputs/a.png", "outputs/b.png"]

    monkeypatch.setattr("zimage.ui.handlers.list_output_images", fake_list)
    assert load_gallery() == ["outputs/a.png", "outputs/b.png"]
    assert captured["outputs_dir"] == OUTPUTS_DIR


def test_load_gallery_uses_output_dir(monkeypatch):
    captured = {}

    def fake_list(outputs_dir=None):
        captured["outputs_dir"] = outputs_dir
        return []

    monkeypatch.setattr("zimage.ui.handlers.list_output_images", fake_list)
    load_gallery(r"d:\Projects\DeepSeek\z-image-studio\outputs" + "\\")
    assert captured["outputs_dir"] == Path("d:/Projects/DeepSeek/z-image-studio/outputs")


def test_load_gallery_with_index_empty(monkeypatch):
    monkeypatch.setattr("zimage.ui.handlers.list_output_images", lambda outputs_dir=None: [])
    items, index = load_gallery_with_index()
    assert items == []
    assert index is None


def test_load_gallery_with_index_resets_to_zero(monkeypatch):
    monkeypatch.setattr(
        "zimage.ui.handlers.list_output_images",
        lambda outputs_dir=None: ["a.png", "b.png"],
    )
    items, index = load_gallery_with_index()
    assert items == ["a.png", "b.png"]
    assert index == 0


def test_set_gallery_index_from_select():
    class Evt:
        index = 2

    assert set_gallery_index(Evt()) == 2


def test_delete_preview_image_removes_selected(tmp_path: Path, monkeypatch):
    newer = tmp_path / "zimage-new.png"
    older = tmp_path / "zimage-old.png"
    newer.write_bytes(b"\x89PNG\r\n\x1a\n")
    older.write_bytes(b"\x89PNG\r\n\x1a\n")
    import os

    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status=None, extra="": extra)

    items, index, status = delete_preview_image(
        [str(newer), str(older)],
        0,
        str(tmp_path),
    )
    assert items == [str(older)]
    assert index == 0
    assert not newer.exists()
    assert older.exists()
    assert "Deleted" in status
    assert "zimage-new.png" in status


def test_delete_preview_image_empty_gallery(monkeypatch):
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status=None, extra="": extra)
    items, index, status = delete_preview_image([], 0, "./outputs")
    assert items == []
    assert index is None
    assert "No image to delete" in status


def test_delete_preview_image_falls_back_to_disk_list(tmp_path: Path, monkeypatch):
    disk = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    for path in disk:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")
    captured = {}

    def fake_list(outputs_dir=None):
        captured["outputs_dir"] = outputs_dir
        return list(disk)

    def fake_delete(path, outputs_dir=None):
        captured["deleted"] = (str(path), outputs_dir)
        Path(path).unlink()
        disk[:] = [p for p in disk if p != str(path)]
        return Path(path).resolve()

    monkeypatch.setattr("zimage.ui.handlers.list_output_images", fake_list)
    monkeypatch.setattr("zimage.ui.handlers.delete_output_image", fake_delete)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status=None, extra="": extra)

    # PIL images have no path — fall back to disk listing by index.
    items, index, status = delete_preview_image(
        [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")],
        1,
        str(tmp_path),
    )
    assert captured["deleted"][0] == str(tmp_path / "b.png")
    assert captured["deleted"][1] == tmp_path
    assert items == [str(tmp_path / "a.png")]
    assert index == 0
    assert "Deleted" in status


def test_delete_preview_image_refuses_outside_path(tmp_path: Path, monkeypatch):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr("zimage.ui.handlers.list_output_images", lambda outputs_dir=None: [])
    monkeypatch.setattr("zimage.ui.handlers.delete_output_image", lambda *_a, **_k: None)
    try:
        with pytest.raises(gr.Error, match="Output dir"):
            delete_preview_image([str(outside)], 0, str(tmp_path))
    finally:
        outside.unlink(missing_ok=True)


def test_clear_preview_images_clears_and_resets_gallery(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_clear(outputs_dir=None):
        captured["outputs_dir"] = outputs_dir
        return 4

    monkeypatch.setattr("zimage.ui.handlers.clear_output_images", fake_clear)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status=None, extra="": extra or "ready",
    )
    items, index, status = clear_preview_images(str(tmp_path))
    assert items == []
    assert index is None
    assert captured["outputs_dir"] == tmp_path
    assert "Cleared 4 images" in status
    assert str(tmp_path) in status


def test_clear_preview_images_empty_warns(monkeypatch):
    monkeypatch.setattr("zimage.ui.handlers.clear_output_images", lambda outputs_dir=None: 0)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status=None, extra="": extra or "ready",
    )
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda msg: warnings.append(msg))
    items, index, status = clear_preview_images("./outputs")
    assert items == []
    assert index is None
    assert status == "No images to clear."
    assert warnings == ["No images to clear."]


def test_clear_preview_images_uses_default_output_dir(monkeypatch):
    captured = {}

    def fake_clear(outputs_dir=None):
        captured["outputs_dir"] = outputs_dir
        return 1

    monkeypatch.setattr("zimage.ui.handlers.clear_output_images", fake_clear)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status=None, extra="": extra or "ready",
    )
    clear_preview_images(None)
    assert captured["outputs_dir"] == OUTPUTS_DIR


def _drain(gen):
    last = None
    for last in gen:
        pass
    return last


def _generate(
    *,
    prompt="a cat",
    resolution="512x384 (4:3)",
    seed=1,
    random_seed=False,
    steps=9,
    guidance=0.0,
    time_shift=3.0,
    model_id="model",
    device="cpu",
    dtype_name="float32",
    cpu_offload=False,
    vae_tiling=False,
    quantize_modules=None,
    batch_count=1,
    output_dir=DEFAULT_OUTPUT_DIR,
    gallery=None,
    lora_dir="",
    lora_names=None,
    lora_weights=None,
    image_format="jpeg",
    progress=None,
):
    return _drain(
        generate(
            prompt,
            resolution,
            seed,
            random_seed,
            steps,
            guidance,
            time_shift,
            model_id,
            device,
            dtype_name,
            cpu_offload,
            vae_tiling,
            quantize_modules,
            batch_count,
            output_dir,
            gallery,
            lora_dir,
            lora_names,
            lora_weights,
            image_format,
            progress=progress,
        )
    )


def test_generate_requires_prompt():
    with pytest.raises(gr.Error, match="Enter a prompt"):
        _generate(prompt="  ")


def test_generate_requires_prompt_when_none():
    with pytest.raises(gr.Error, match="Enter a prompt"):
        _generate(prompt=None)


def test_generate_rejects_invalid_batch_count():
    with pytest.raises(gr.Error, match="Batch count"):
        _generate(batch_count=0)
    with pytest.raises(gr.Error, match="Batch count"):
        _generate(batch_count=10_000)
    with pytest.raises(gr.Error, match="Batch count"):
        _generate(batch_count=None)
    with pytest.raises(gr.Error, match="Batch count"):
        _generate(batch_count="x")


def test_generate_success_prepends_gallery(monkeypatch):
    fake = Image.new("RGB", (8, 8), "blue")
    previous = Image.new("RGB", (8, 8), "green")

    def fake_generate_image(*_args, **kwargs):
        assert kwargs["width"] == 512
        assert kwargs["height"] == 384
        assert kwargs["seed"] == 42
        assert kwargs["outputs_dir"] == Path("outputs")
        assert kwargs["image_format"] == "jpeg"
        return fake, 42, {"device": "cpu", "device_name": "CPU", "loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    items, used, seed, status = _generate(
        prompt="a cat",
        resolution="512x384 (4:3)",
        seed=42,
        gallery=[previous],
    )
    assert items[0] is fake
    assert items[1] is previous
    assert used == "42"
    assert seed == 42
    assert status == "ok"


def test_generate_forwards_image_format(monkeypatch):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured["image_format"] = kwargs["image_format"]
        return Image.new("RGB", (4, 4), "red"), 1, {"device": "cpu", "loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    _generate(image_format="png")
    assert captured["image_format"] == "png"


def test_generate_passes_windows_output_dir(monkeypatch):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured["outputs_dir"] = kwargs["outputs_dir"]
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    _generate(output_dir=r"d:\Projects\DeepSeek\z-image-studio\outputs" + "\\")
    assert captured["outputs_dir"] == Path("d:/Projects/DeepSeek/z-image-studio/outputs")


def test_generate_empty_output_dir_falls_back(monkeypatch):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured["outputs_dir"] = kwargs["outputs_dir"]
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    _generate(output_dir="")
    assert captured["outputs_dir"] == OUTPUTS_DIR


def test_generate_random_seed_and_default_model(monkeypatch):
    captured = {}
    fake = Image.new("RGB", (4, 4), "red")

    def fake_generate_image(*_args, **kwargs):
        captured.update(kwargs)
        return fake, kwargs["seed"], {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    monkeypatch.setattr("zimage.ui.handlers.random.randint", lambda _a, _b: 99)

    items, used, seed, _status = _generate(
        random_seed=True,
        seed=1,
        model_id="   ",
        gallery=None,
    )
    assert captured["seed"] == 99
    assert captured["model_id"] == DEFAULT_MODEL
    assert items == [fake]
    assert used == "99"
    assert seed == 99


def test_generate_batch_incremental_seeds(monkeypatch):
    seeds = []
    fakes = [Image.new("RGB", (2, 2), c) for c in ("red", "green", "blue")]

    def fake_generate_image(*_args, **kwargs):
        seeds.append(kwargs["seed"])
        return fakes[len(seeds) - 1], kwargs["seed"], {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    items, used, seed, _status = _generate(seed=10, batch_count=3, gallery=None)
    assert seeds == [10, 11, 12]
    assert used == "10–12"
    assert seed == 12
    assert items == list(reversed(fakes))


def test_generate_batch_random_start_then_increment(monkeypatch):
    seeds = []

    def fake_generate_image(*_args, **kwargs):
        seeds.append(kwargs["seed"])
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    monkeypatch.setattr("zimage.ui.handlers.random.randint", lambda _a, _b: 50)

    items, used, seed, _status = _generate(random_seed=True, batch_count=2)
    assert seeds == [50, 51]
    assert used == "50–51"
    assert seed == 51
    assert len(items) == 2


def test_generate_batch_stop_keeps_frames(monkeypatch):
    call_n = {"n": 0}

    def fake_generate_image(*_args, **kwargs):
        call_n["n"] += 1
        if call_n["n"] == 2:
            request_stop()
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status, extra="": extra or "ok",
    )
    handlers._stop_event.clear()

    items, used, seed, status = _generate(seed=7, batch_count=5)
    assert call_n["n"] == 2
    assert len(items) == 2
    assert used == "7–8"
    assert seed == 8
    assert "Stopped after 2 of 5" in status


def test_generate_caps_gallery_at_limit(monkeypatch):
    monkeypatch.setattr("zimage.ui.handlers.GALLERY_LIMIT", 12)
    fake = Image.new("RGB", (2, 2), "white")
    previous = [Image.new("RGB", (2, 2), "black") for _ in range(12)]

    monkeypatch.setattr(
        "zimage.ui.handlers.generate_image",
        lambda *_args, **_kwargs: (fake, 1, {}),
    )
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    items, _used, _seed, _status = _generate(gallery=previous)
    assert len(items) == 12
    assert items[0] is fake
    assert items[-1] is previous[10]
    assert all(item is not previous[11] for item in items)


def test_generate_batch_caps_gallery_at_limit(monkeypatch):
    monkeypatch.setattr("zimage.ui.handlers.GALLERY_LIMIT", 12)
    previous = [Image.new("RGB", (2, 2), "black") for _ in range(10)]
    created = []

    def fake_generate_image(*_args, **kwargs):
        img = Image.new("RGB", (2, 2), "white")
        created.append(img)
        return img, kwargs["seed"], {}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    items, _used, _seed, _status = _generate(seed=1, batch_count=5, gallery=previous)
    assert len(items) == 12
    assert items[0] is created[-1]
    assert items[4] is created[0]


def test_generate_mid_batch_error_keeps_frames(monkeypatch):
    call_n = {"n": 0}
    first = Image.new("RGB", (2, 2), "red")

    def fake_generate_image(*_args, **kwargs):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return first, kwargs["seed"], {"loaded": True}
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status, extra="": extra or "ok",
    )

    gen = generate(
        "a cat",
        "512x384 (4:3)",
        3,
        False,
        9,
        0.0,
        3.0,
        "model",
        "cpu",
        "float32",
        False,
        False,
        None,
        3,
        None,
        progress=None,
    )
    first_yield = next(gen)
    assert first_yield[0][0] is first
    with pytest.raises(gr.Error, match="CUDA OOM"):
        _drain(gen)


def test_generate_offline_hint(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("Cannot reach hub: local_files_only is set")

    monkeypatch.setattr("zimage.ui.handlers.generate_image", boom)
    with pytest.raises(gr.Error, match="HF_HUB_OFFLINE=0"):
        _generate()


def test_generate_generic_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr("zimage.ui.handlers.generate_image", boom)
    with pytest.raises(gr.Error, match="CUDA OOM") as exc_info:
        _generate()
    assert "HF_HUB_OFFLINE" not in str(exc_info.value)


def test_load_model_success(monkeypatch):
    monkeypatch.setattr(
        "zimage.ui.handlers.ensure_pipeline",
        lambda *_args, **_kwargs: (object(), {"loaded": True, "device": "cpu"}),
    )
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "loaded-ok")
    assert load_model("model", "cpu", "float32", False, False) == "loaded-ok"


def test_load_model_passes_quantize_flags(monkeypatch):
    captured = {}

    def fake_ensure(*_args, **kwargs):
        captured.update(kwargs)
        return object(), {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.ensure_pipeline", fake_ensure)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    load_model("model", "cpu", "int8", False, False, ["text encoder"])
    assert captured["quantize_transformer"] is False
    assert captured["quantize_text_encoder"] is True


def test_generate_passes_quantize_flags(monkeypatch):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured.update(kwargs)
        return Image.new("RGB", (2, 2), "white"), 1, {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    _generate(quantize_modules=["transformer"])
    assert captured["quantize_transformer"] is True
    assert captured["quantize_text_encoder"] is False


def test_load_model_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("Failed to load model")

    monkeypatch.setattr("zimage.ui.handlers.ensure_pipeline", boom)
    with pytest.raises(gr.Error, match="Failed to load model"):
        load_model("bad-model", "cpu", "float32", False, False)


def test_unload_model(monkeypatch):
    called = {"n": 0}

    def fake_unload():
        called["n"] += 1

    monkeypatch.setattr("zimage.ui.handlers.unload_pipeline", fake_unload)
    monkeypatch.setattr(
        "zimage.ui.handlers.runtime_status",
        lambda: {
            "demo": False,
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": "x",
            "cuda_built": "no",
        },
    )
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": extra)
    text = unload_model()
    assert called["n"] == 1
    assert "unloaded" in text.lower()


def test_image_progress_maps_fraction_into_batch():
    recorded = []

    def progress(value, desc=""):
        recorded.append((value, desc))

    report = _image_progress(progress, index=1, count=4)
    report(0.5, desc="Generating…")
    assert recorded == [(0.375, "Image 2 / 4 — Generating…")]


def test_image_progress_none_returns_none():
    assert _image_progress(None, 0, 1) is None


def test_generate_batch1_progress_monotone_ends_at_one(monkeypatch):
    recorded = []

    def fake_generate_image(*_args, **kwargs):
        progress = kwargs["progress"]
        assert progress is not None
        progress(0.5, desc="mid")
        progress(1.0, desc="Done")
        return Image.new("RGB", (2, 2), "white"), 1, {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    def progress(value, desc=""):
        recorded.append(value)

    _generate(seed=1, batch_count=1, progress=progress)
    assert recorded == [0.5, 1.0]
    assert all(recorded[i] <= recorded[i + 1] for i in range(len(recorded) - 1))


def test_generate_batch3_progress_no_early_one(monkeypatch):
    recorded = []

    def fake_generate_image(*_args, **kwargs):
        progress = kwargs["progress"]
        progress(0.5, desc="mid")
        progress(1.0, desc="Done")
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    def progress(value, desc=""):
        recorded.append((value, desc))

    _generate(seed=10, batch_count=3, progress=progress)

    values = [v for v, _ in recorded]
    # Per image: 0.5 and 1.0 mapped into thirds, plus mid-batch re-set after yields.
    assert (1 / 6) in values
    assert (1 / 3) in values
    assert 0.5 in values
    assert (2 / 3) in values
    assert (5 / 6) in values
    assert 1.0 in values
    # 1.0 only after the last image finishes (may appear once at the end).
    assert values.count(1.0) == 1
    assert values[-1] == 1.0
    assert all(v < 1.0 for v in values[:-1])


def test_generate_stop_progress_stays_below_one(monkeypatch):
    recorded = []
    call_n = {"n": 0}

    def fake_generate_image(*_args, **kwargs):
        call_n["n"] += 1
        progress = kwargs["progress"]
        progress(1.0, desc="Done")
        if call_n["n"] == 2:
            request_stop()
        return Image.new("RGB", (2, 2), "white"), kwargs["seed"], {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr(
        "zimage.ui.handlers.format_status",
        lambda status, extra="": extra or "ok",
    )
    handlers._stop_event.clear()

    def progress(value, desc=""):
        recorded.append(value)

    _generate(seed=7, batch_count=5, progress=progress)
    assert call_n["n"] == 2
    assert recorded
    assert recorded[-1] < 1.0
    assert 1.0 not in recorded


def test_generate_passes_progress_wrapper(monkeypatch):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured["progress"] = kwargs.get("progress")
        if kwargs["progress"] is not None:
            kwargs["progress"](1.0, desc="Done")
        return Image.new("RGB", (2, 2), "white"), 1, {}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")

    def progress(value, desc=""):
        pass

    _generate(progress=progress)
    assert captured["progress"] is not None
    assert captured["progress"] is not progress


def test_refresh_loras_keeps_existing_files(tmp_path: Path):
    (tmp_path / "alpha.safetensors").write_bytes(b"")
    (tmp_path / "beta.safetensors").write_bytes(b"")
    normalized, dropdown, weights = refresh_loras(
        str(tmp_path),
        ["alpha.safetensors", "gone.safetensors"],
        [["alpha.safetensors", 0.8], ["gone.safetensors", 0.5]],
    )
    assert normalized == Path(tmp_path).as_posix()
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert labels == ["alpha.safetensors", "beta.safetensors"]
    assert dropdown.value == ["alpha.safetensors"]
    assert dropdown.allow_custom_value is True
    assert weights == [["alpha.safetensors", 0.8]]


def test_refresh_loras_normalizes_windows_file_path(tmp_path: Path):
    (tmp_path / "alpha.safetensors").write_bytes(b"")
    windows_like = str(tmp_path / "alpha.safetensors").replace("/", "\\")
    normalized, dropdown, weights = refresh_loras(windows_like, None, None)
    assert normalized == Path(tmp_path).as_posix()
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert labels == ["alpha.safetensors"]
    assert weights == []


def test_refresh_loras_normalizes_windows_dir_path(tmp_path: Path):
    (tmp_path / "alpha.safetensors").write_bytes(b"")
    windows_like = str(tmp_path).replace("/", "\\")
    normalized, dropdown, weights = refresh_loras(windows_like, ["alpha.safetensors"], None)
    assert normalized == Path(tmp_path).as_posix()
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert labels == ["alpha.safetensors"]
    assert dropdown.value == ["alpha.safetensors"]
    assert weights == [["alpha.safetensors", 1.0]]


def test_refresh_loras_normalizes_windows_dir_path_trailing_slash(tmp_path: Path):
    (tmp_path / "alpha.safetensors").write_bytes(b"")
    windows_like = str(tmp_path).replace("/", "\\") + "\\"
    normalized, dropdown, weights = refresh_loras(windows_like, None, None)
    assert normalized == Path(tmp_path).as_posix()
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert labels == ["alpha.safetensors"]
    assert weights == []


def test_sync_lora_weights_preserves_and_defaults():
    rows = sync_lora_weights(
        ["style.safetensors", "char.safetensors"],
        [["style.safetensors", 0.8], ["old.safetensors", 0.2]],
    )
    assert rows == [["style.safetensors", 0.8], ["char.safetensors", 1.0]]


def test_generate_passes_lora_specs(monkeypatch, tmp_path: Path):
    captured = {}
    (tmp_path / "style.safetensors").write_bytes(b"")

    def fake_generate_image(*_args, **kwargs):
        captured.update(kwargs)
        return Image.new("RGB", (2, 2), "white"), 1, {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    _generate(
        lora_dir=str(tmp_path),
        lora_names=["style.safetensors"],
        lora_weights=[["style.safetensors", 0.8]],
    )
    specs = captured["loras"]
    assert len(specs) == 1
    assert specs[0].filename == "style.safetensors"
    assert specs[0].scale == 0.8


def test_generate_lora_error_is_gr_error(monkeypatch, tmp_path: Path):
    (tmp_path / "style.safetensors").write_bytes(b"")

    def boom(*_args, **_kwargs):
        raise RuntimeError("bad adapter")

    monkeypatch.setattr("zimage.ui.handlers.generate_image", boom)
    with pytest.raises(gr.Error, match="bad adapter"):
        _generate(
            lora_dir=str(tmp_path),
            lora_names=["style.safetensors"],
            lora_weights=[["style.safetensors", 1.0]],
        )


def test_refresh_loras_from_fixture(tiny_lora_dir: Path):
    normalized, dropdown, weights = refresh_loras(str(tiny_lora_dir), None, None)
    assert normalized == Path(tiny_lora_dir).as_posix()
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert "tiny_zimage_lora.safetensors" in labels
    assert weights == []


def test_save_ui_prefs_normalizes_lora_dir():
    from zimage.prefs import load_ui_prefs

    save_ui_prefs(
        "a cat",
        "1024x768 (4:3)",
        9,
        1,
        "./outputs",
        "jpeg",
        42,
        True,
        DEFAULT_MODEL,
        "auto",
        "fp8",
        ["transformer", "text encoder"],
        False,
        False,
        r"D:\loras\style",
        [],
        [],
        0.0,
        3.0,
    )
    prefs = load_ui_prefs()
    assert prefs["prompt"] == "a cat"
    assert prefs["lora_dir"] == "D:/loras/style"


def test_save_ui_prefs_drops_missing_adapter(tmp_path: Path):
    from zimage.prefs import load_ui_prefs

    (tmp_path / "alpha.safetensors").write_bytes(b"")
    save_ui_prefs(
        "a cat",
        "1024x768 (4:3)",
        9,
        1,
        "./outputs",
        "jpeg",
        42,
        True,
        DEFAULT_MODEL,
        "auto",
        "fp8",
        ["transformer", "text encoder"],
        False,
        False,
        str(tmp_path),
        ["alpha.safetensors", "gone.safetensors"],
        [["alpha.safetensors", 0.8], ["gone.safetensors", 0.5]],
        0.0,
        3.0,
    )
    prefs = load_ui_prefs()
    assert prefs["lora_adapters"] == ["alpha.safetensors"]
    assert prefs["lora_weights"] == [["alpha.safetensors", 0.8]]
    assert "gone.safetensors" not in prefs["lora_adapters"]
    assert all(row[0] != "gone.safetensors" for row in prefs["lora_weights"])


def test_save_ui_prefs_handles_none_prompt():
    from zimage.prefs import load_ui_prefs

    save_ui_prefs(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    prefs = load_ui_prefs()
    assert prefs["prompt"] == ""
    assert prefs["lora_dir"] == ""


def test_restore_ui_prefs_without_file():
    (
        prompt,
        _resolution,
        _steps,
        _batch,
        _output_dir,
        _image_format,
        _seed,
        _random_seed,
        model_id,
        _device,
        _precision,
        _quantize_modules,
        _cpu_offload,
        _vae_tiling,
        lora_dir,
        adapters,
        weights,
        _guidance,
        _time_shift,
    ) = restore_ui_prefs()
    assert prompt == ""
    assert lora_dir == ""
    assert list(adapters.choices or []) == []
    assert weights == []
    assert model_id == DEFAULT_MODEL


def test_restore_ui_prefs_all_fields(tiny_lora_dir: Path):
    from zimage.prefs import save_ui_prefs as dump_prefs

    dump_prefs(
        {
            "prompt": "sunset",
            "resolution": "1280x720 (16:9)",
            "steps": 11,
            "batch": 2,
            "output_dir": "./custom-out",
            "image_format": "png",
            "seed": 7,
            "random_seed": False,
            "model_id": "local/model",
            "device": "cpu",
            "precision": "float16",
            "quantize_modules": ["transformer"],
            "cpu_offload": True,
            "vae_tiling": True,
            "lora_dir": str(tiny_lora_dir),
            "lora_adapters": ["tiny_zimage_lora.safetensors"],
            "lora_weights": [["tiny_zimage_lora.safetensors", 0.55]],
            "guidance": 1.2,
            "time_shift": 4.0,
        }
    )
    (
        prompt,
        resolution,
        steps,
        batch,
        output_dir,
        image_format,
        seed,
        random_seed,
        model_id,
        device,
        precision,
        quantize_modules,
        cpu_offload,
        vae_tiling,
        lora_dir,
        adapters,
        weights,
        guidance,
        time_shift,
    ) = restore_ui_prefs()
    assert prompt == "sunset"
    assert resolution == "1280x720 (16:9)"
    assert steps == 11
    assert batch == 2
    assert output_dir == "./custom-out"
    assert image_format == "png"
    assert seed == 7
    assert random_seed is False
    assert model_id == "local/model"
    assert device == "cpu"
    assert precision == "float16"
    assert quantize_modules == ["transformer"]
    assert cpu_offload is True
    assert vae_tiling is True
    assert lora_dir == Path(tiny_lora_dir).as_posix()
    assert adapters.value == ["tiny_zimage_lora.safetensors"]
    assert weights == [["tiny_zimage_lora.safetensors", 0.55]]
    assert guidance == 1.2
    assert time_shift == 4.0


def test_restore_ui_prefs_drops_missing_adapter(tiny_lora_dir: Path):
    from zimage.prefs import save_ui_prefs as dump_prefs

    dump_prefs(
        {
            "lora_dir": str(tiny_lora_dir),
            "lora_adapters": ["tiny_zimage_lora.safetensors", "gone.safetensors"],
            "lora_weights": [
                ["tiny_zimage_lora.safetensors", 0.4],
                ["gone.safetensors", 0.9],
            ],
        }
    )
    result = restore_ui_prefs()
    adapters = result[15]
    weights = result[16]
    assert adapters.value == ["tiny_zimage_lora.safetensors"]
    assert weights == [["tiny_zimage_lora.safetensors", 0.4]]


def test_generate_passes_real_fixture_lora(monkeypatch, tiny_lora_dir: Path):
    captured = {}

    def fake_generate_image(*_args, **kwargs):
        captured.update(kwargs)
        return Image.new("RGB", (2, 2), "white"), 1, {"loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status, extra="": "ok")
    _generate(
        lora_dir=str(tiny_lora_dir),
        lora_names=["tiny_zimage_lora.safetensors"],
        lora_weights=[["tiny_zimage_lora.safetensors", 0.55]],
    )
    specs = captured["loras"]
    assert len(specs) == 1
    assert specs[0].filename == "tiny_zimage_lora.safetensors"
    assert specs[0].scale == 0.55
    assert specs[0].path.is_file()
    assert specs[0].path.stat().st_size > 1024


def test_load_model_does_not_accept_lora():
    params = inspect.signature(load_model).parameters
    assert "lora_dir" not in params
    assert "loras" not in params
    assert "lora_names" not in params


def test_list_training_jobs_scans_jobs_dir(tmp_path: Path, monkeypatch):
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("Beta", jobs)
    create_or_open_job("Alpha", jobs)
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    assert handlers.list_training_jobs() == ["alpha", "beta"]


def test_create_or_open_training_job_does_not_rewrite(tmp_path: Path, monkeypatch):
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    before = (root / "config.yaml").read_bytes()
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    payload = handlers.create_or_open_training_job("job")
    assert (root / "config.yaml").read_bytes() == before
    assert payload["job_id"] == "job"
    assert "job_name:" in payload["config_text"]
    assert payload["state"]["status"] == "stopped"


def test_save_and_queue_training_yaml(tmp_path: Path, monkeypatch):
    from zimage.training.commands import consume_commands
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.jobs import create_or_open_job, load_job_config, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    loaded = handlers.load_training_job("job")
    text = loaded["config_text"]
    assert handlers.validate_training_yaml("job", text) == {"ok": True}
    invalid = handlers.validate_training_yaml("job", "not: [valid")
    assert invalid["ok"] is False
    saved = handlers.save_training_yaml("job", text)
    assert saved["mode"] == "saved"
    # Idle / STOPPED queue_training_update writes YAML (CLI parity), no commands.
    idle_queue = handlers.queue_training_update("job", text)
    assert idle_queue["mode"] == "saved"
    assert list((root / "commands").glob("*.json")) == []
    write_job_state(root, JobState("job", JobStatus.RUNNING, step=1, epoch=0))
    queued = handlers.queue_training_update("job", text)
    assert queued["mode"] == "queued"
    envelopes = consume_commands(root)
    assert len(envelopes) == 1
    assert envelopes[0].kind == "update"
    assert load_job_config(root)["job_name"]


def test_save_training_yaml_overwrites_legacy_top_level_transformers(
    tmp_path: Path, monkeypatch
):
    import yaml

    from zimage.training.jobs import create_or_open_job, load_job_config
    from zimage.training.schema import KNOWN_MAIN_SOURCE, job_create_template

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    document = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    model = document.pop("model")
    document["main_transformer"] = model["main_transformer"]
    document["sampling_transformer"] = model["sampling_transformer"]
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            document, default_flow_style=False, allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    nested = job_create_template()
    nested["job_name"] = "job"
    text = yaml.safe_dump(
        nested, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    assert handlers.validate_training_yaml("job", text) == {"ok": True}
    saved = handlers.save_training_yaml("job", text)

    assert saved["mode"] == "saved"
    persisted = yaml.safe_load(saved["config_text"])
    assert "main_transformer" not in persisted
    assert persisted["model"]["main_transformer"]["path"] == KNOWN_MAIN_SOURCE
    assert load_job_config(root)["model"]["main_transformer"]["path"] == KNOWN_MAIN_SOURCE


def test_save_training_yaml_queues_when_state_running(tmp_path: Path, monkeypatch):
    from zimage.training.commands import consume_commands
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.jobs import create_or_open_job, load_job_config, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    before = (root / "config.yaml").read_bytes()
    write_job_state(root, JobState("job", JobStatus.RUNNING, step=1, epoch=0))
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    text = handlers.load_training_job("job")["config_text"]
    result = handlers.save_training_yaml("job", text)
    assert result["mode"] == "queued"
    assert (root / "config.yaml").read_bytes() == before
    envelopes = consume_commands(root)
    assert len(envelopes) == 1
    assert envelopes[0].kind == "update"
    assert load_job_config(root)["job_name"]


def test_idle_save_discards_stale_queued_update(tmp_path: Path, monkeypatch):
    import yaml

    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.jobs import create_or_open_job, load_job_config, write_job_state
    from zimage.training.schema import TrainingConfigError

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)

    write_job_state(root, JobState("job", JobStatus.RUNNING, step=1, epoch=0))
    stale = load_job_config(root)
    stale["seed"] = 11
    queued = handlers.queue_training_update("job", yaml.safe_dump(stale))
    assert queued["mode"] == "queued"
    assert list((root / "commands").glob("*.json"))

    write_job_state(
        root, JobState("job", JobStatus.COMPLETED, step=1, epoch=0, exit_code=0)
    )
    idle = load_job_config(root)
    idle["seed"] = 22
    saved = handlers.save_training_yaml("job", yaml.safe_dump(idle))
    assert saved["mode"] == "saved"
    assert load_job_config(root)["seed"] == 22
    assert list((root / "commands").glob("*.json")) == []

    write_job_state(root, JobState("job", JobStatus.RUNNING, step=1, epoch=0))
    handlers.queue_training_update("job", yaml.safe_dump(stale))
    write_job_state(
        root, JobState("job", JobStatus.COMPLETED, step=1, epoch=0, exit_code=0)
    )
    invalid = load_job_config(root)
    invalid["precision"] = "invalid"
    with pytest.raises(TrainingConfigError):
        handlers.save_training_yaml("job", yaml.safe_dump(invalid))
    leftover = list((root / "commands").glob("*.json"))
    assert len(leftover) == 1
    assert load_job_config(root)["seed"] == 22


def test_start_training_job_handoff_sequence(tmp_path: Path, monkeypatch):
    from zimage.engine.pipeline import training_start_fence_is_set
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    order: list = []
    fence_at: dict[str, bool] = {}

    class FakeGuard:
        def acquire(self):
            order.append("acquire")
            fence_at["acquire"] = training_start_fence_is_set()
            return True

        def release(self):
            fence_at["release"] = training_start_fence_is_set()
            order.append("release")

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            fence_at["start"] = training_start_fence_is_set()
            order.append(("start", job_id))

        def stop(self):
            order.append("stop")

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: order.append("request_stop"))
    monkeypatch.setattr(handlers, "unload_pipeline", lambda: order.append("unload"))
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: order.append("cuda"))
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", lambda: 4242)

    payload = handlers.start_training_job("job")
    assert payload["job_id"] == "job"
    assert order == [
        "request_stop",
        "acquire",
        "release",
        "unload",
        "cuda",
        ("start", "job"),
    ]
    assert fence_at["acquire"] is False
    assert fence_at["release"] is True
    assert fence_at["start"] is True
    assert training_start_fence_is_set() is False
    handlers.stop_training_job("job")
    assert order[-1] == "stop"


def test_start_training_keeps_fence_until_foreign_holder_pid(tmp_path: Path, monkeypatch):
    from zimage.engine.pipeline import training_start_fence_is_set
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    fence_before_holder: list[bool] = []
    reads = {"n": 0}

    def fake_lease_reader():
        reads["n"] += 1
        fence_before_holder.append(training_start_fence_is_set())
        if reads["n"] < 3:
            return None
        return 4242

    class FakeGuard:
        def acquire(self):
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            assert training_start_fence_is_set() is True

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: None)
    monkeypatch.setattr(handlers, "unload_pipeline", lambda: None)
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: None)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", fake_lease_reader)

    handlers.start_training_job("job")
    assert reads["n"] >= 3
    assert fence_before_holder[0] is True
    assert fence_before_holder[1] is True
    assert training_start_fence_is_set() is False


def test_start_training_clears_fence_when_start_fails(tmp_path: Path, monkeypatch):
    from zimage.engine.pipeline import training_start_fence_is_set
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)

    class FakeGuard:
        def acquire(self):
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            raise RuntimeError("spawn failed")

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: None)
    monkeypatch.setattr(handlers, "unload_pipeline", lambda: None)
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: None)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())

    with pytest.raises(RuntimeError, match="spawn failed"):
        handlers.start_training_job("job")
    assert training_start_fence_is_set() is False


def test_start_training_clears_fence_when_holder_wait_times_out(tmp_path: Path, monkeypatch):
    from zimage.engine.pipeline import training_start_fence_is_set
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)

    class FakeGuard:
        def acquire(self):
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    class FakeManager:
        def __init__(self):
            self.stopped = False
            self.job_id = None

        def is_running(self):
            return False

        def start(self, job_id):
            self.job_id = job_id

        def stop(self):
            self.stopped = True
            self.job_id = None

    manager = FakeManager()
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: None)
    monkeypatch.setattr(handlers, "unload_pipeline", lambda: None)
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: None)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: manager)
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", lambda: None)
    monkeypatch.setattr(handlers, "_LEASE_WAIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(handlers, "_LEASE_WAIT_POLL_SECONDS", 0.01)

    with pytest.raises(RuntimeError, match="Timed out waiting for the trainer"):
        handlers.start_training_job("job")
    assert training_start_fence_is_set() is False
    assert manager.stopped is True


def test_wait_for_gpu_lease_does_not_acquire_during_start_fence(monkeypatch):
    from zimage.engine.pipeline import (
        clear_training_start_fence,
        set_training_start_fence,
    )

    acquired: list[bool] = []

    class FakeGuard:
        def acquire(self):
            acquired.append(True)
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    set_training_start_fence()
    try:
        with pytest.raises(RuntimeError, match="training is already running"):
            handlers._wait_for_gpu_lease()
        assert acquired == []
    finally:
        clear_training_start_fence()


def test_wait_for_gpu_lease_releases_if_fence_set_after_acquire(monkeypatch):
    from zimage.engine.pipeline import (
        clear_training_start_fence,
        set_training_start_fence,
    )

    released: list[bool] = []

    class FakeGuard:
        def acquire(self):
            set_training_start_fence()
            return True

        def release(self):
            released.append(True)

        def is_held(self):
            return False

    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    try:
        with pytest.raises(RuntimeError, match="training is already running"):
            handlers._wait_for_gpu_lease()
        assert released == [True]
    finally:
        clear_training_start_fence()


def test_live_foreign_lease_pid_reads_file_runtime_guard(tmp_path: Path, monkeypatch):
    from zimage.training.runtime_guard import LOCK_ENV_VAR

    lock = tmp_path / ".gpu.lease"
    monkeypatch.setenv(LOCK_ENV_VAR, str(lock))
    monkeypatch.setattr(
        "zimage.training.runtime_guard.pid_is_alive",
        lambda pid: pid == 4242,
    )
    lock.write_text("4242\n", encoding="ascii")
    assert handlers._live_foreign_lease_pid() == 4242
    lock.write_text("99999\n", encoding="ascii")
    assert handlers._live_foreign_lease_pid() is None
    lock.write_bytes(b"\0")
    assert handlers._live_foreign_lease_pid() is None



def test_duplicate_start_does_not_wait(tmp_path: Path, monkeypatch):
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    waited: list[str] = []

    class SlowGuard:
        def acquire(self):
            waited.append("acquire")
            time.sleep(5)
            return False

        def release(self):
            waited.append("release")

        def is_held(self):
            return False

    class RunningManager:
        def is_running(self):
            return True

        def start(self, job_id):
            waited.append("start")

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: waited.append("request_stop"))
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: SlowGuard())
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: RunningManager())

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="training is already running"):
        handlers.start_training_job("job")
    assert time.monotonic() - started < 1.0
    assert waited == []


def test_load_model_rejected_during_start_fence():
    from zimage.engine.pipeline import (
        clear_training_start_fence,
        set_training_start_fence,
    )

    set_training_start_fence()
    try:
        with pytest.raises(gr.Error, match="Training owns the GPU"):
            load_model("model", "cpu", "float32", False, False)
    finally:
        clear_training_start_fence()


def test_generate_rejected_during_start_fence():
    from zimage.engine.pipeline import (
        clear_training_start_fence,
        set_training_start_fence,
    )

    set_training_start_fence()
    try:
        with pytest.raises(gr.Error, match="Training owns the GPU"):
            _generate()
    finally:
        clear_training_start_fence()


def test_start_training_blocks_generate_and_load_during_fence(tmp_path: Path, monkeypatch):
    from zimage.engine.pipeline import generate_image as real_generate
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    blocked: list[str] = []

    class FakeGuard:
        def acquire(self):
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            return None

    def during_unload():
        with pytest.raises(gr.Error, match="Training owns the GPU"):
            load_model("model", "cpu", "float32", False, False)
        blocked.append("load")
        with pytest.raises(RuntimeError, match="Training owns the GPU"):
            real_generate("prompt", seed=1, outputs_dir=tmp_path)
        blocked.append("generate")

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: None)
    monkeypatch.setattr(handlers, "unload_pipeline", during_unload)
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: None)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", lambda: 4242)
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {
            "demo": False,
            "cuda": False,
            "torch": True,
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": "2.0",
            "cuda_built": "",
            "loaded": False,
        },
    )
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")

    handlers.start_training_job("job")
    assert blocked == ["load", "generate"]


def test_poll_training_state_lists_previews(tmp_path: Path, monkeypatch):
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    preview = root / "previews" / "00001-00-sample.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    payload = handlers.poll_training_state("job")
    assert payload["state"]["job_id"] == "job"
    assert payload["previews"] == [str(preview)]


def test_poll_training_log_reads_chunk(tmp_path: Path, monkeypatch):
    from zimage.training.job_log import LOG_FILE, LOGS_DIR
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    log_path = root / LOGS_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "hello log\n"
    log_path.write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    payload = handlers.poll_training_log("job", -1)
    assert payload["chunk"] == text
    assert payload["reset"] is True
    assert payload["next_offset"] == len(text.encode("utf-8"))
    second = handlers.poll_training_log("job", payload["next_offset"])
    assert second["chunk"] == ""
    assert second["reset"] is False
    assert second["next_offset"] == payload["next_offset"]


def test_clear_training_log_truncates_existing_file(tmp_path: Path, monkeypatch):
    from zimage.training.job_log import LOG_FILE, LOGS_DIR
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    log_path = root / LOGS_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"hello log\n")
    nested = root / "previews" / "step-1" / "00.png"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"\x89PNG\r\n\x1a\n")
    flat = root / "previews" / "00001-00-sample.png"
    flat.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    payload = handlers.clear_training_log("job")
    assert log_path.stat().st_size == 0
    previews = root / "previews"
    assert previews.is_dir()
    assert list(previews.iterdir()) == []
    assert not nested.exists()
    assert not flat.exists()
    assert payload["job_id"] == "job"
    assert payload["previews"] == []
    assert payload["state"]["status"] == "stopped"
    assert payload["state"]["step"] == 0


def test_clear_training_log_missing_file_does_not_raise(tmp_path: Path, monkeypatch):
    from zimage.training.jobs import create_or_open_job

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    payload = handlers.clear_training_log("job")
    assert payload["job_id"] == "job"


def test_clear_training_log_unknown_job_fails(tmp_path: Path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        handlers.clear_training_log("missing")


def test_clear_training_log_resets_progress_and_returns_job_view(
    tmp_path: Path, monkeypatch
):
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.job_log import LOG_FILE, LOGS_DIR
    from zimage.training.jobs import create_or_open_job, load_job_state, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    log_path = root / LOGS_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"hello log\n")
    (root / "previews" / "00001-00-sample.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    ckpt = root / "checkpoints" / "step-3" / "adapter.bin"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    write_job_state(
        root,
        JobState(
            "job",
            JobStatus.COMPLETED,
            step=3,
            epoch=1,
            last_error="old",
            exit_code=0,
        ),
    )
    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)

    payload = handlers.clear_training_log("job")

    assert log_path.stat().st_size == 0
    assert list((root / "previews").iterdir()) == []
    checkpoints = root / "checkpoints"
    assert checkpoints.is_dir()
    assert list(checkpoints.iterdir()) == []
    assert not ckpt.exists()
    state = load_job_state(root)
    assert state.status is JobStatus.STOPPED
    assert state.step == 0
    assert state.epoch == 0
    assert state.last_error is None
    assert state.exit_code is None
    assert payload["job_id"] == "job"
    assert "job_name:" in payload["config_text"]
    assert payload["state"]["status"] == "stopped"
    assert payload["state"]["step"] == 0
    assert payload["state"]["epoch"] == 0
    assert payload["previews"] == []
    assert "message" not in payload


def test_clear_training_log_refuses_when_this_job_running(tmp_path: Path, monkeypatch):
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.job_log import LOG_FILE, LOGS_DIR
    from zimage.training.jobs import create_or_open_job, load_job_state, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    log_path = root / LOGS_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_bytes = b"hello log\n"
    log_path.write_bytes(log_bytes)
    preview = root / "previews" / "00001-00-sample.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\n")
    ckpt = root / "checkpoints" / "step-3" / "adapter.bin"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    write_job_state(root, JobState("job", JobStatus.RUNNING, step=3, epoch=1))
    state_bytes = (root / "state.json").read_bytes()

    class RunningManager:
        job_id = "job"

        def is_running(self):
            return True

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: RunningManager())
    monkeypatch.setattr(handlers, "training_start_fence_is_set", lambda: False)

    with pytest.raises(RuntimeError, match="Stop first"):
        handlers.clear_training_log("job")

    assert log_path.read_bytes() == log_bytes
    assert preview.is_file()
    assert ckpt.is_file()
    assert (root / "state.json").read_bytes() == state_bytes
    assert load_job_state(root).step == 3


def test_clear_training_log_refuses_when_start_fence_set(tmp_path: Path, monkeypatch):
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.job_log import LOG_FILE, LOGS_DIR
    from zimage.training.jobs import create_or_open_job, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    log_path = root / LOGS_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_bytes = b"hello log\n"
    log_path.write_bytes(log_bytes)
    preview = root / "previews" / "00001-00-sample.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\n")
    ckpt = root / "checkpoints" / "step-3" / "adapter.bin"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    write_job_state(root, JobState("job", JobStatus.COMPLETED, step=3, epoch=1, exit_code=0))
    state_bytes = (root / "state.json").read_bytes()

    class IdleManager:
        job_id = None

        def is_running(self):
            return False

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: IdleManager())
    monkeypatch.setattr(handlers, "training_start_fence_is_set", lambda: True)

    with pytest.raises(RuntimeError, match="Stop first"):
        handlers.clear_training_log("job")

    assert log_path.read_bytes() == log_bytes
    assert preview.is_file()
    assert ckpt.is_file()
    assert (root / "state.json").read_bytes() == state_bytes


def test_clear_training_log_succeeds_when_other_job_running(
    tmp_path: Path, monkeypatch
):
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.jobs import create_or_open_job, load_job_state, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    ckpt = root / "checkpoints" / "step-2" / "adapter.bin"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    write_job_state(root, JobState("job", JobStatus.STOPPED, step=2, epoch=0))

    class OtherManager:
        job_id = "other"

        def is_running(self):
            return True

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: OtherManager())
    monkeypatch.setattr(handlers, "training_start_fence_is_set", lambda: False)

    payload = handlers.clear_training_log("job")

    assert payload["state"]["step"] == 0
    assert load_job_state(root).status is JobStatus.STOPPED
    assert list((root / "checkpoints").iterdir()) == []


def test_clear_training_log_succeeds_when_stale_job_id_not_running(
    tmp_path: Path, monkeypatch
):
    from zimage.training.jobs import create_or_open_job, load_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)

    class StaleManager:
        job_id = "job"

        def is_running(self):
            return False

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: StaleManager())
    monkeypatch.setattr(handlers, "training_start_fence_is_set", lambda: False)

    payload = handlers.clear_training_log("job")

    assert payload["job_id"] == "job"
    assert load_job_state(root).step == 0


def test_clear_training_log_succeeds_when_crashed_running_state(
    tmp_path: Path, monkeypatch
):
    from zimage.training.contracts import JobState, JobStatus
    from zimage.training.jobs import create_or_open_job, load_job_state, write_job_state

    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    ckpt = root / "checkpoints" / "step-4" / "adapter.bin"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    write_job_state(root, JobState("job", JobStatus.RUNNING, step=4, epoch=1))

    class DeadManager:
        job_id = "job"

        def is_running(self):
            return False

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: DeadManager())
    monkeypatch.setattr(handlers, "training_start_fence_is_set", lambda: False)

    payload = handlers.clear_training_log("job")

    state = load_job_state(root)
    assert state.status is JobStatus.STOPPED
    assert state.step == 0
    assert list((root / "checkpoints").iterdir()) == []
    assert payload["state"]["status"] == "stopped"
    assert payload["state"]["step"] == 0


def test_poll_training_log_read_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        handlers,
        "_require_job_dir",
        lambda job_id: (_ for _ in ()).throw(OSError("missing")),
    )
    payload = handlers.poll_training_log("job", 12)
    assert payload == {"chunk": "", "next_offset": 12, "reset": False}

    monkeypatch.setattr(handlers, "_require_job_dir", lambda job_id: Path("/tmp/job"))
    monkeypatch.setattr(
        handlers,
        "read_job_log_chunk",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read")),
    )
    payload = handlers.poll_training_log("job", 7)
    assert payload == {"chunk": "", "next_offset": 7, "reset": False}


def test_training_callbacks_are_canonical_production_functions():
    from zimage.ui.training_panel import TrainingCallbacks, noop_training_callbacks

    bundle = handlers.training_callbacks()
    assert isinstance(bundle, TrainingCallbacks)
    assert bundle.start_job is handlers.start_training_job
    assert bundle.stop_job is handlers.stop_training_job
    assert bundle.save_yaml is handlers.save_training_yaml
    assert bundle.list_jobs is handlers.list_training_jobs
    assert bundle.create_or_open is handlers.create_or_open_training_job
    assert bundle.queue_update is handlers.queue_training_update
    assert bundle.load_job is handlers.load_training_job
    assert bundle.validate_yaml is handlers.validate_training_yaml
    assert bundle.poll_state is handlers.poll_training_state
    assert bundle.poll_log is handlers.poll_training_log
    assert bundle.clear_log is handlers.clear_training_log

    noop = noop_training_callbacks()
    assert bundle.start_job is not noop.start_job
    assert bundle.stop_job is not noop.stop_job
    assert bundle.save_yaml is not noop.save_yaml
    assert bundle.list_jobs is not noop.list_jobs
    assert bundle.create_or_open is not noop.create_or_open
    assert bundle.queue_update is not noop.queue_update
    assert bundle.poll_log is not noop.poll_log
    assert bundle.clear_log is not noop.clear_log


def test_start_training_unloads_cached_pipeline_and_next_inference_reloads(
    tmp_path: Path, monkeypatch, reset_pipeline
):
    """Training start clears the inference singleton; the next load is lazy.

    Same ``zimage.engine.pipeline._pipe`` that ``ensure_pipeline`` caches.
    Generate is not invoked while the fake job still holds the GPU lease.
    """
    from zimage.engine import pipeline as pipeline_mod
    from zimage.engine.pipeline import ensure_pipeline, training_start_fence_is_set
    from zimage.training.jobs import create_or_open_job
    from zimage.training.runtime_guard import FileRuntimeGuard

    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    lock_path = tmp_path / "gpu.lease"
    monkeypatch.setenv("ZIMAGE_RUNTIME_LOCK", str(lock_path))
    monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)
    pipeline_mod.clear_training_start_fence()
    pipeline_mod._lease_local.depth = 0

    cached = object()
    pipeline_mod._pipe = cached
    pipeline_mod._pipe_key = ("seeded-before-training",)
    assert handlers.unload_pipeline is pipeline_mod.unload_pipeline

    loads: list[object] = []

    def fake_load(*_args, **_kwargs):
        pipe = object()
        loads.append(pipe)
        return pipe

    def non_demo_status():
        return {
            "demo": False,
            "cuda": False,
            "torch": True,
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": "2.0",
            "cuda_built": "",
            "loaded": False,
        }

    monkeypatch.setattr(pipeline_mod, "load_pipeline", fake_load)
    monkeypatch.setattr(pipeline_mod, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(pipeline_mod, "runtime_status", non_demo_status)
    monkeypatch.setattr(pipeline_mod, "_reclaim_memory", lambda: None)

    fence_at: dict[str, bool] = {}
    pipe_at_start: dict[str, object] = {}

    class FakeGuard:
        def acquire(self):
            return True

        def release(self):
            return None

        def is_held(self):
            return False

    class FakeManager:
        def is_running(self):
            return False

        def start(self, job_id):
            fence_at["start"] = training_start_fence_is_set()
            pipe_at_start["pipe"] = pipeline_mod._pipe
            pipe_at_start["key"] = pipeline_mod._pipe_key
            assert job_id == "job"

        def stop(self):
            return None

    monkeypatch.setattr(handlers, "_jobs_dir", lambda: jobs)
    monkeypatch.setattr(handlers, "request_stop", lambda: None)
    monkeypatch.setattr(handlers, "_create_handoff_guard", lambda: FakeGuard())
    monkeypatch.setattr(handlers, "_sync_and_empty_cuda", lambda: None)
    monkeypatch.setattr(handlers, "_get_training_process_manager", lambda: FakeManager())
    monkeypatch.setattr(handlers, "_live_foreign_lease_pid", lambda: 4242)

    payload = handlers.start_training_job("job")
    assert payload["job_id"] == "job"
    assert pipe_at_start["pipe"] is None
    assert pipe_at_start["key"] is None
    assert pipeline_mod._pipe is None
    assert pipeline_mod._pipe_key is None
    assert fence_at["start"] is True
    assert training_start_fence_is_set() is False
    assert loads == []

    child_lease = FileRuntimeGuard(lock_path)
    assert child_lease.acquire() is True
    try:
        with pytest.raises(RuntimeError, match="Training owns the GPU"):
            ensure_pipeline("model-a", "cpu", "float32", False, False)
        with pytest.raises(gr.Error, match="Training owns the GPU"):
            load_model("model-a", "cpu", "float32", False, False)
        assert loads == []
        assert pipeline_mod._pipe is None
    finally:
        child_lease.release()
        monkeypatch.setattr(pipeline_mod, "_INFERENCE_GUARD", None)
        pipeline_mod._lease_local.depth = 0

    first, status_a = ensure_pipeline("model-a", "cpu", "float32", False, False)
    second, status_b = ensure_pipeline("model-a", "cpu", "float32", False, False)
    assert first is not cached
    assert first is second
    assert first is pipeline_mod._pipe
    assert loads == [first]
    assert status_a["loaded"] is True
    assert status_b["model"] == "model-a"
    assert training_start_fence_is_set() is False
