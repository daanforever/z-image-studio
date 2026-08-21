from __future__ import annotations

import os
from pathlib import Path

from zimage.config import (
    DEFAULT_DTYPE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    PRECISION_CHOICES,
    canonical_precision,
    is_truthy,
    load_dotenv,
    parse_resolution,
)


def test_precision_choices_include_quantized():
    assert PRECISION_CHOICES == ["fp8", "bfloat16", "float16", "float32", "int8"]
    assert DEFAULT_DTYPE == "fp8"


def test_canonical_precision_aliases():
    assert canonical_precision("INT8WO") == "int8"
    assert canonical_precision("float8dq") == "fp8"
    assert canonical_precision("fp8dq") == "fp8"
    assert canonical_precision("fp16") == "float16"
    assert canonical_precision("bf16") == "bfloat16"
    assert canonical_precision("half") == "float16"
    assert canonical_precision("fp32") == "float32"
    assert canonical_precision("q8") == "int8"
    assert canonical_precision("int8_weight_only") == "int8"
    assert canonical_precision("unknown") == "bfloat16"
    assert canonical_precision(None) == "bfloat16"
    assert canonical_precision("   ") == "bfloat16"


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

    assert os.environ["ZIMAGE_TEST_A"] == "from-file"
    assert os.environ["ZIMAGE_TEST_B"] == "already"


def test_load_dotenv_missing_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ZIMAGE_TEST_MISSING", raising=False)
    load_dotenv(tmp_path / "nope.env")
    assert "ZIMAGE_TEST_MISSING" not in os.environ


def test_load_dotenv_strips_quotes_and_skips_junk(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# comment\nZIMAGE_TEST_Q=\"quoted\"\nZIMAGE_TEST_S='single'\nNOTHING\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ZIMAGE_TEST_Q", raising=False)
    monkeypatch.delenv("ZIMAGE_TEST_S", raising=False)
    load_dotenv(env_file)
    assert os.environ["ZIMAGE_TEST_Q"] == "quoted"
    assert os.environ["ZIMAGE_TEST_S"] == "single"
