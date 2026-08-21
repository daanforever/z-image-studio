from __future__ import annotations

import warnings

from zimage.ui.layout import build_ui


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
