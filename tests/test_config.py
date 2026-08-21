from __future__ import annotations

from pathlib import Path

from zimage.config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    PRECISION_CHOICES,
    is_truthy,
    load_dotenv,
    parse_resolution,
)


def test_precision_choices_include_int8():
    assert PRECISION_CHOICES == ["bfloat16", "float16", "float32", "int8"]


def test_parse_resolution_preset():
    assert parse_resolution("1024x768 (4:3)") == (1024, 768)


def test_parse_resolution_multiplication_sign():
    assert parse_resolution("1280×720") == (1280, 720)


def test_parse_resolution_spaces_and_star():
    assert parse_resolution("  512 * 384  ") == (512, 384)


def test_parse_resolution_fallback():
    assert parse_resolution("not-a-size") == (DEFAULT_WIDTH, DEFAULT_HEIGHT)


def test_is_truthy():
    assert is_truthy("1")
    assert is_truthy("true")
    assert is_truthy("YES")
    assert is_truthy("on")
    assert not is_truthy("0")
    assert not is_truthy("false")
    assert not is_truthy("")
    assert not is_truthy(None)


def test_load_dotenv_setdefault(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ZIMAGE_TEST_A=from-file\n# comment\nZIMAGE_TEST_B=keep\n", encoding="utf-8")
    monkeypatch.setenv("ZIMAGE_TEST_B", "already")
    monkeypatch.delenv("ZIMAGE_TEST_A", raising=False)

    load_dotenv(env_file)

    import os

    assert os.environ["ZIMAGE_TEST_A"] == "from-file"
    assert os.environ["ZIMAGE_TEST_B"] == "already"
