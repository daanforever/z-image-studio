from __future__ import annotations

from pathlib import Path

import yaml

from zimage.prefs import load_ui_prefs, save_ui_prefs
from zimage.prefs.store import dump_document, load_document, update_section


def test_load_document_missing_file():
    assert load_document() == {}


def test_load_document_empty_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    assert load_document() == {}


def test_load_document_broken_yaml(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(":\n  - broken: [", encoding="utf-8")
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    assert load_document() == {}


def test_load_document_non_dict_root(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    assert load_document() == {}

    path.write_text("42\n", encoding="utf-8")
    assert load_document() == {}


def test_roundtrip_ui_section(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    monkeypatch.setattr("zimage.prefs.CONFIG_YAML", path)
    save_ui_prefs({"prompt": "hello", "steps": 7})
    loaded = load_ui_prefs()
    assert loaded["prompt"] == "hello"
    assert loaded["steps"] == 7
    doc = load_document()
    assert "ui" in doc
    assert doc["ui"]["prompt"] == "hello"


def test_save_preserves_other_sections(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    monkeypatch.setattr("zimage.prefs.CONFIG_YAML", path)
    dump_document({"engine": {"foo": 1}, "ui": {"prompt": "old"}})
    save_ui_prefs({"prompt": "new"})
    doc = load_document()
    assert doc["engine"] == {"foo": 1}
    assert doc["ui"]["prompt"] == "new"


def test_ui_section_non_dict_uses_defaults(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    monkeypatch.setattr("zimage.prefs.CONFIG_YAML", path)
    dump_document({"ui": ["not", "a", "dict"]})
    prefs = load_ui_prefs()
    assert prefs["prompt"] == ""
    assert "steps" in prefs


def test_atomic_write_leaves_no_tmp(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    monkeypatch.setattr("zimage.prefs.CONFIG_YAML", path)
    save_ui_prefs({"prompt": "atomic"})
    assert path.is_file()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ui"]["prompt"] == "atomic"


def test_update_section_only_touches_named_key(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("zimage.prefs.store.CONFIG_YAML", path)
    dump_document({"a": 1, "b": 2})
    update_section("b", {"x": 3})
    doc = load_document()
    assert doc == {"a": 1, "b": {"x": 3}}
