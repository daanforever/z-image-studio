"""Server-side UI / app preferences persisted in config.yaml."""

from zimage.prefs.schema import (
    UI_PREF_KEYS,
    UI_SECTION,
    coerce_ui_prefs,
    serialize_ui_prefs,
    ui_pref_defaults,
)
from zimage.prefs.store import (
    CONFIG_YAML,
    config_path,
    dump_document,
    load_document,
    update_section,
)


def load_ui_prefs() -> dict:
    """Load and coerce the ui section from config.yaml."""
    doc = load_document()
    section = doc.get(UI_SECTION)
    return coerce_ui_prefs(section if isinstance(section, dict) else {})


def save_ui_prefs(data: dict) -> None:
    """Write the ui section; other top-level sections are preserved."""
    update_section(UI_SECTION, serialize_ui_prefs(data))


__all__ = [
    "CONFIG_YAML",
    "UI_PREF_KEYS",
    "UI_SECTION",
    "coerce_ui_prefs",
    "config_path",
    "dump_document",
    "load_document",
    "load_ui_prefs",
    "save_ui_prefs",
    "serialize_ui_prefs",
    "ui_pref_defaults",
    "update_section",
]
