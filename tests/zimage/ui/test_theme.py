from __future__ import annotations

from zimage.ui.theme import CUSTOM_CSS, CUSTOM_JS, appearance_kwargs, build_theme


def test_build_theme_and_css():
    theme = build_theme()
    assert theme is not None
    assert "#generate-btn" in CUSTOM_CSS
    assert "#status-md" in CUSTOM_CSS
    assert "#studio-navbar" in CUSTOM_CSS
    assert "#studio-navbar-actions" in CUSTOM_CSS
    assert "#studio-clear-btn" in CUSTOM_CSS
    assert "#studio-stop-btn" in CUSTOM_CSS
    assert "#studio-navbar-actions button" in CUSTOM_CSS
    assert "border-radius: 0.5rem" in CUSTOM_CSS
    assert ".studio-brand" in CUSTOM_CSS
    assert ".studio-footer" in CUSTOM_CSS
    assert "text-align: center" in CUSTOM_CSS
    assert "64ch" not in CUSTOM_CSS
    assert "margin: 0 auto" in CUSTOM_CSS
    assert ".gradio-container .app" in CUSTOM_CSS
    assert "#output-gallery .media-button" in CUSTOM_CSS
    assert "cursor: zoom-in" in CUSTOM_CSS
    assert "#output-gallery .gallery-container" in CUSTOM_CSS
    assert "height: 640px" in CUSTOM_CSS
    assert "#output-gallery .grid-wrap" in CUSTOM_CSS
    assert "overflow-y: auto" in CUSTOM_CSS
    assert "#output-gallery .thumbnails" in CUSTOM_CSS
    assert "overflow-x: auto" in CUSTOM_CSS
    appearance = appearance_kwargs()
    assert appearance["css"] == CUSTOM_CSS
    assert appearance["js"] == CUSTOM_JS
    assert appearance["theme"] is not None
    assert "color-scheme" in appearance["head"]


def test_theme_uses_system_fonts():
    theme = build_theme()
    assert "ui-sans-serif" in theme.font
    assert "ui-monospace" in theme.font_mono
    assert not any(
        url and "fonts.googleapis.com" in url for url in theme._stylesheets
    )
    assert "IBM Plex" not in CUSTOM_CSS
    assert "Instrument Sans" not in CUSTOM_CSS
    assert "ui-monospace" in CUSTOM_CSS


def test_gallery_preview_click_opens_fullscreen():
    assert "#output-gallery .media-button" in CUSTOM_JS
    assert 'button[aria-label="Fullscreen"]' in CUSTOM_JS
    assert "stopImmediatePropagation" in CUSTOM_JS


def test_training_log_pre_is_scoped_monospace():
    assert "#studio-training-job-log pre" in CUSTOM_CSS
    assert "pre.studio-training-job-log-pre" in CUSTOM_CSS
    assert "#studio-training-job-log .html-container" in CUSTOM_CSS
    assert "#studio-training-job-log .prose" in CUSTOM_CSS
    assert (
        "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace "
        "!important" in CUSTOM_CSS
    )
    assert "white-space: pre;" in CUSTOM_CSS
    assert "overflow: auto;" in CUSTOM_CSS
    assert "pre-wrap" not in CUSTOM_CSS
    assert "word-break: break-word" not in CUSTOM_CSS
    assert not any(
        line.lstrip().startswith(prefix)
        for line in CUSTOM_CSS.splitlines()
        for prefix in (".prose", ".html-container")
    )


def test_training_log_delta_helper_in_custom_js():
    assert "__zimageApplyTrainingLogDelta" in CUSTOM_JS
    assert "studio-training-job-log" in CUSTOM_JS
    assert "querySelector" in CUSTOM_JS
    assert "textContent" in CUSTOM_JS
    assert "scrollTop" in CUSTOM_JS
    assert "scrollHeight" in CUSTOM_JS
    assert "MutationObserver" not in CUSTOM_JS
