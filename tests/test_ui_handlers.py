from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image

import gradio as gr
import pytest

from zimage.config import DEFAULT_MODEL
from zimage.ui.handlers import (
    _image_progress,
    generate,
    load_model,
    refresh_loras,
    request_stop,
    sync_lora_weights,
    unload_model,
)
import zimage.ui.handlers as handlers


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
    gallery=None,
    lora_dir="",
    lora_names=None,
    lora_weights=None,
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
            gallery,
            lora_dir,
            lora_names,
            lora_weights,
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


def test_generate_caps_gallery_at_twelve(monkeypatch):
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


def test_generate_batch_caps_gallery_at_twelve(monkeypatch):
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
    dropdown, weights = refresh_loras(
        str(tmp_path),
        ["alpha.safetensors", "gone.safetensors"],
        [["alpha.safetensors", 0.8], ["gone.safetensors", 0.5]],
    )
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert labels == ["alpha.safetensors", "beta.safetensors"]
    assert dropdown.value == ["alpha.safetensors"]
    assert weights == [["alpha.safetensors", 0.8]]


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
    dropdown, weights = refresh_loras(str(tiny_lora_dir), None, None)
    labels = []
    for choice in dropdown.choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    assert "tiny_zimage_lora.safetensors" in labels
    assert weights == []


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
