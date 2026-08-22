"""Atomic YAML document store for the project root config.yaml."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

from zimage.config import ROOT

CONFIG_YAML = ROOT / "config.yaml"

_lock = threading.Lock()


def config_path() -> Path:
    return CONFIG_YAML


def load_document(path: Path | None = None) -> dict[str, Any]:
    """Load the full YAML document. Missing / invalid → empty dict."""
    target = path or CONFIG_YAML
    if not target.is_file():
        return {}
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def dump_document(data: dict[str, Any], path: Path | None = None) -> None:
    """Atomically write the full YAML document (tmp + replace)."""
    target = path or CONFIG_YAML
    payload = data if isinstance(data, dict) else {}
    text = yaml.safe_dump(
        payload,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with _lock:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)


def update_section(section: str, value: dict[str, Any], path: Path | None = None) -> None:
    """Replace one top-level section, leaving other keys untouched."""
    target = path or CONFIG_YAML
    with _lock:
        doc = load_document(target)
        doc[section] = value if isinstance(value, dict) else {}
        text = yaml.safe_dump(
            doc,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
