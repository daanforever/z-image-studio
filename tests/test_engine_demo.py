from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from zimage.engine.demo import demo_image, wrap_text


def test_wrap_text_empty():
    image = Image.new("RGB", (64, 64))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    assert wrap_text("", font, 40, draw) == ""


def test_wrap_text_splits_long_line():
    image = Image.new("RGB", (200, 80))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = wrap_text("one two three four five six seven eight", font, 50, draw)
    assert "\n" in wrapped
    assert wrapped.split()[0] == "one"


def test_wrap_text_stops_after_ten_lines():
    image = Image.new("RGB", (40, 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = wrap_text(" ".join(f"w{i}" for i in range(20)), font, 1, draw)
    lines = wrapped.split("\n")
    assert len(lines) <= 11
    assert any("…" in line for line in lines)


def test_demo_image_clamps_size_and_is_rgb():
    image = demo_image("a test prompt", width=64, height=8000, seed=7, reason="no weights")
    assert image.mode == "RGB"
    assert image.size == (512, 1536)


def test_demo_image_empty_prompt_and_missing_font(monkeypatch):
    fallback = ImageFont.load_default()
    monkeypatch.setattr(
        "zimage.engine.demo.ImageFont.truetype",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no font")),
    )
    monkeypatch.setattr("zimage.engine.demo.ImageFont.load_default", lambda: fallback)
    image = demo_image("   ", width=8000, height=64, seed=1, reason="")
    assert image.mode == "RGB"
    assert image.size == (1536, 512)
