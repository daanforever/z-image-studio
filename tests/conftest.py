from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zimage.config import load_dotenv as _load_dotenv  # noqa: E402, F401

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TINY_LORA_DIR = FIXTURES_DIR / "loras"
TINY_LORA_FILE = TINY_LORA_DIR / "tiny_zimage_lora.safetensors"


@pytest.fixture
def reset_pipeline():
    from zimage.engine.lora import reset_lora_adapters
    from zimage.engine.pipeline import unload_pipeline

    unload_pipeline()
    reset_lora_adapters()
    yield
    unload_pipeline()
    reset_lora_adapters()


@pytest.fixture
def tiny_lora_dir() -> Path:
    assert TINY_LORA_FILE.is_file(), f"Missing fixture: {TINY_LORA_FILE}"
    return TINY_LORA_DIR
