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


def test_demo_image_clamps_size_and_is_rgb():
    image = demo_image("a test prompt", width=64, height=8000, seed=7, reason="no weights")
    assert image.mode == "RGB"
    assert image.size == (512, 1536)
