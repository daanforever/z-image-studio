from __future__ import annotations

from PIL import Image

import gradio as gr
import pytest

from zimage.config import DEFAULT_MODEL
from zimage.ui.handlers import generate, load_model, unload_model


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
    gallery=None,
    progress=None,
):
    return generate(
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
        gallery,
        progress=progress,
    )


def test_generate_requires_prompt():
    with pytest.raises(gr.Error, match="Enter a prompt"):
        _generate(prompt="  ")


def test_generate_requires_prompt_when_none():
    with pytest.raises(gr.Error, match="Enter a prompt"):
        _generate(prompt=None)


def test_generate_success_prepends_gallery(monkeypatch):
    fake = Image.new("RGB", (8, 8), "blue")
    previous = Image.new("RGB", (8, 8), "green")

    def fake_generate_image(*_args, **kwargs):
        assert kwargs["width"] == 512
        assert kwargs["height"] == 384
        assert kwargs["seed"] == 42
        return fake, 42, {"device": "cpu", "device_name": "CPU", "loaded": True}

    monkeypatch.setattr("zimage.ui.handlers.generate_image", fake_generate_image)
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status: "ok")

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
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status: "ok")
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


def test_generate_caps_gallery_at_twelve(monkeypatch):
    fake = Image.new("RGB", (2, 2), "white")
    previous = [Image.new("RGB", (2, 2), "black") for _ in range(12)]

    monkeypatch.setattr(
        "zimage.ui.handlers.generate_image",
        lambda *_args, **_kwargs: (fake, 1, {}),
    )
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status: "ok")

    items, _used, _seed, _status = _generate(gallery=previous)
    assert len(items) == 12
    assert items[0] is fake
    assert items[-1] is previous[10]
    assert all(item is not previous[11] for item in items)


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
    monkeypatch.setattr("zimage.ui.handlers.format_status", lambda status: "loaded-ok")
    assert load_model("model", "cpu", "float32", False, False) == "loaded-ok"


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
