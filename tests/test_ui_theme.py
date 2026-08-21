from __future__ import annotations

from zimage.ui.theme import CUSTOM_CSS, appearance_kwargs, build_theme


def test_build_theme_and_css():
    theme = build_theme()
    assert theme is not None
    assert "#generate-btn" in CUSTOM_CSS
    assert "#status-md" in CUSTOM_CSS
    appearance = appearance_kwargs()
    assert appearance["css"] == CUSTOM_CSS
    assert appearance["theme"] is not None
    assert "color-scheme" in appearance["head"]
