from __future__ import annotations

import inspect
import warnings

from zimage.ui.handlers import generate
from zimage.ui.layout import build_ui


def _elem_ids(demo) -> set[str | None]:
    return {getattr(block, "elem_id", None) for block in demo.blocks.values()}


def test_build_ui_constructs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The parameters have been moved from the Blocks constructor",
            category=UserWarning,
        )
        demo = build_ui()
    assert demo is not None


def test_build_ui_has_navbar(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    ids = _elem_ids(demo)
    assert "studio-navbar" in ids
    assert "studio-brand" in ids
    assert "studio-navbar-actions" in ids
    assert "studio-stop-btn" in ids
    brand = next(
        block
        for block in demo.blocks.values()
        if getattr(block, "elem_id", None) == "studio-brand"
    )
    assert "Studio" in str(brand.value)


def test_generate_event_show_progress_minimal(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    generate_fns = [
        fn
        for fn in demo.fns.values()
        if getattr(getattr(fn, "fn", None), "__name__", None) == "generate"
    ]
    assert len(generate_fns) == 1
    assert generate_fns[0].show_progress == "minimal"


def test_generate_progress_default_disables_track_tqdm():
    default = inspect.signature(generate).parameters["progress"].default
    assert getattr(default, "track_tqdm", None) is False
