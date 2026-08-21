from __future__ import annotations

from PIL import Image

import gradio as gr
import pytest

from app import parse_args
from zimage.ui.handlers import generate, load_model, unload_model
from zimage.ui.layout import build_ui
from zimage.ui.theme import CUSTOM_CSS, build_theme


def test_parse_args_defaults():
    args = parse_args([])
    assert args.host == "0.0.0.0"
    assert args.share is False


def test_parse_args_overrides():
    args = parse_args(["--host", "127.0.0.1", "--port", "8000", "--share"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.share is True


def test_generate_requires_prompt():
    with pytest.raises(gr.Error, match="Enter a prompt"):
        generate(
            "  ",
            "512x384 (4:3)",
            1,
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
            progress=None,
        )


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

    items, used, seed, status = generate(
        "a cat",
        "512x384 (4:3)",
        42,
        False,
        9,
        0.0,
        3.0,
        "Tongyi-MAI/Z-Image-Turbo",
        "cpu",
        "float32",
        False,
        False,
        [previous],
        progress=None,
    )
    assert items[0] is fake
    assert items[1] is previous
    assert used == "42"
    assert seed == 42
    assert status == "ok"


def test_generate_offline_hint(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("Cannot reach hub: local_files_only is set")

    monkeypatch.setattr("zimage.ui.handlers.generate_image", boom)
    with pytest.raises(gr.Error, match="HF_HUB_OFFLINE=0"):
        generate(
            "a cat",
            "512x384 (4:3)",
            1,
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
            progress=None,
        )


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


def test_build_theme_and_css():
    theme = build_theme()
    assert theme is not None
    assert "#generate-btn" in CUSTOM_CSS
    assert "#status-md" in CUSTOM_CSS


def test_build_ui_constructs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    assert demo is not None
