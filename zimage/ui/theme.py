"""Gradio theme and CSS: achromatic darkroom so generated images stay the focus."""

from __future__ import annotations

from typing import Any

from zimage import config as _config  # noqa: F401 — telemetry defaults before Gradio
import gradio as gr

COLOR_SCHEME_HEAD = '<meta name="color-scheme" content="dark light">'

CUSTOM_CSS = """
.gradio-container {
    max-width: 1240px !important;
    width: 100%;
    margin: 0 auto !important;
}
.gradio-container .app {
    max-width: 100% !important;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
}
#studio-navbar {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 1rem;
    width: 100%;
    min-height: 3.25rem;
    margin: 0 0 1.15rem !important;
    padding: 0.15rem 0 0.85rem !important;
    border-bottom: 1px solid var(--border-color-primary);
    background: transparent;
    overflow: visible !important;
}
#studio-navbar .block,
#studio-navbar .form,
#studio-navbar .html-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-width: 0;
}
.studio-brand {
    display: inline-block;
    font-size: 1.125rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: var(--body-text-color);
    user-select: none;
}
#studio-navbar-actions,
#studio-navbar-generate,
#studio-navbar-training {
    display: flex !important;
    flex: 0 0 auto !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 0.5rem;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    overflow: visible !important;
}
#studio-navbar-actions > .block,
#studio-navbar-actions > .form,
#studio-navbar-generate > .block,
#studio-navbar-generate > .form,
#studio-navbar-training > .block,
#studio-navbar-training > .form {
    flex: 0 0 auto !important;
    width: auto !important;
    max-width: none !important;
}
#studio-navbar-shared {
    display: none !important;
}
#studio-navbar-generate button,
#studio-navbar-training button {
    box-sizing: border-box !important;
    width: 2.25rem !important;
    height: 2.25rem !important;
    min-width: 2.25rem !important;
    max-width: 2.25rem !important;
    padding: 0 !important;
    border-radius: 0.5rem !important;
    font-size: 0 !important;
    line-height: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#studio-stop-btn::before,
#studio-training-stop::before {
    content: "";
    display: block;
    width: 0.7em;
    height: 0.7em;
    font-size: 1rem;
    background: currentColor;
    border-radius: 0.08em;
}
#studio-training-start::before {
    content: "";
    display: block;
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 0.38em 0 0.38em 0.62em;
    border-color: transparent transparent transparent currentColor;
    font-size: 1rem;
}
#studio-training-clear::before {
    content: "";
    display: block;
    box-sizing: border-box;
    width: 0.85em;
    height: 0.42em;
    font-size: 1rem;
    border: 0.09em solid currentColor;
    border-radius: 0.1em;
    background: linear-gradient(90deg, currentColor 0 0.22em, transparent 0.22em);
    transform: rotate(-35deg);
}
#studio-clear-btn {
    position: relative !important;
}
/* Broom: bristle head (::before) + angled handle (::after) */
#studio-clear-btn::before {
    content: "";
    display: block;
    box-sizing: border-box;
    width: 0.7em;
    height: 0.5em;
    font-size: 1rem;
    margin-top: 0.32em;
    margin-left: -0.18em;
    border-radius: 0.06em 0.06em 0.2em 0.2em;
    transform: rotate(-40deg);
    background:
        linear-gradient(90deg, transparent 0.12em, currentColor 0.12em, currentColor 0.2em, transparent 0.2em) 0 0.16em / 100% 0.34em no-repeat,
        linear-gradient(90deg, transparent 0.28em, currentColor 0.28em, currentColor 0.36em, transparent 0.36em) 0 0.16em / 100% 0.34em no-repeat,
        linear-gradient(90deg, transparent 0.44em, currentColor 0.44em, currentColor 0.52em, transparent 0.52em) 0 0.16em / 100% 0.34em no-repeat,
        linear-gradient(currentColor, currentColor) 0 0 / 100% 0.16em no-repeat;
}
#studio-clear-btn::after {
    content: "";
    display: block;
    position: absolute;
    top: 0.28em;
    left: calc(50% + 0.08em);
    width: 0.12em;
    height: 0.85em;
    background: currentColor;
    border-radius: 0.06em;
    transform: translateX(-50%) rotate(-40deg);
    transform-origin: 50% 0;
}
#generate-btn {
    min-height: 48px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
#status-md {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.875rem;
}
.studio-footer,
.studio-footer .prose,
.studio-footer .md {
    max-width: none !important;
    width: 100%;
    text-align: center;
}
.studio-footer p {
    max-width: none;
    margin-left: auto;
    margin-right: auto;
    color: var(--body-text-color-subdued);
}
.dark #output-gallery,
.dark #output-gallery .grid-wrap,
.dark #output-gallery .image-container,
.dark #output-gallery .thumbnail-item,
.dark #output-gallery .empty {
    background: #0a0a0a !important;
}
#output-gallery .gallery-container {
    height: 640px;
    overflow: hidden;
}
#output-gallery .grid-wrap {
    overflow-y: auto;
    max-height: 640px;
}
#output-gallery .thumbnails {
    overflow-x: auto;
    overflow-y: hidden;
    flex-wrap: nowrap;
}
#output-gallery .media-button {
    cursor: zoom-in;
}
#output-gallery.fullscreen .media-button {
    cursor: zoom-out;
}
#output-gallery.fullscreen .gallery-container,
#output-gallery.fullscreen .grid-wrap {
    height: 100% !important;
    max-height: none !important;
}
footer { display: none !important; }
#studio-training-log-delta {
    display: none !important;
}
#studio-training-job-log .html-container,
#studio-training-job-log .prose,
#studio-training-job-log pre,
pre.studio-training-job-log-pre {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 0.8125rem;
    color: #e6e6e6;
    background: #0a0a0a;
    white-space: pre;
    overflow: auto;
}
#studio-training-job-log pre,
pre.studio-training-job-log-pre {
    box-sizing: border-box;
    margin: 0;
    padding: 0.75rem 1rem;
    min-height: 22.5rem;
    max-height: 30rem;
    line-height: 1.45;
}
#studio-training-job {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    overflow: visible;
}
#studio-training-save {
    flex: 0 0 auto !important;
    width: auto !important;
}
#studio-training-create-open,
#studio-training-save,
#studio-training-start,
#studio-training-stop,
#studio-training-clear {
    position: relative;
    overflow: visible;
}
#studio-training-create-open:hover::after,
#studio-training-create-open:focus-within::after,
#studio-training-save:hover::after,
#studio-training-save:focus-within::after,
#studio-training-start:hover::after,
#studio-training-start:focus-within::after,
#studio-training-stop:hover::after,
#studio-training-stop:focus-within::after,
#studio-training-clear:hover::after,
#studio-training-clear:focus-within::after {
    pointer-events: none;
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 30;
    box-sizing: border-box;
    margin-top: 0.35rem;
    max-width: 20rem;
    padding: 0.45rem 0.65rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 0.375rem;
    background: var(--block-background-fill);
    color: var(--body-text-color);
    font-size: 0.75rem;
    font-weight: 400;
    line-height: 1.35;
    text-align: left;
    white-space: normal;
}
#studio-training-create-open:hover::after,
#studio-training-create-open:focus-within::after {
    content: "Create a new job from the Job name, or open that slug if it already exists.";
}
#studio-training-save:hover::after,
#studio-training-save:focus-within::after {
    content: "Save config.yaml. If the job is running, queue the update.";
}
#studio-training-start:hover::after,
#studio-training-start:focus-within::after {
    content: "Start training for the selected job. Stops Generate first if it is running.";
}
#studio-training-stop:hover::after,
#studio-training-stop:focus-within::after {
    content: "Stop the running training job.";
}
#studio-training-clear:hover::after,
#studio-training-clear:focus-within::after {
    content: "Clear the training log for the selected job.";
}
#studio-training-start:hover::after,
#studio-training-start:focus-within::after,
#studio-training-stop:hover::after,
#studio-training-stop:focus-within::after,
#studio-training-clear:hover::after,
#studio-training-clear:focus-within::after {
    left: auto;
    right: 0;
}
"""

# Gradio Gallery preview click cycles prev/next; map it onto the existing fullscreen control.
CUSTOM_JS = """
(() => {
  if (window.__zimageGalleryFullscreenClick) return;
  window.__zimageGalleryFullscreenClick = true;
  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const media = target.closest("#output-gallery .media-button");
      if (!media) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      const gallery = media.closest("#output-gallery");
      const toggle = gallery && gallery.querySelector(
        'button[aria-label="Fullscreen"], button[aria-label="Exit fullscreen mode"]'
      );
      if (toggle) toggle.click();
    },
    true
  );
})();
window.__zimageApplyTrainingLogDelta = function (delta) {
  if (delta == null || delta === "") return;
  let payload = delta;
  if (typeof delta === "string") {
    const trimmed = delta.trim();
    if (!trimmed) return;
    try {
      payload = JSON.parse(trimmed);
    } catch (err) {
      return;
    }
  }
  if (typeof payload !== "object") return;
  const chunk = payload.chunk == null ? "" : String(payload.chunk);
  const reset = Boolean(payload.reset);
  if (!reset && chunk === "") return;
  const seen = typeof delta === "string" ? delta : JSON.stringify(payload);
  if (seen === window.__zimageTrainingLogDeltaSeen) return;
  window.__zimageTrainingLogDeltaSeen = seen;
  const host = document.getElementById("studio-training-job-log");
  if (!host) return;
  const pre = host.tagName === "PRE" ? host : host.querySelector("pre.studio-training-job-log-pre, pre");
  if (!pre) return;
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight <= 16;
  if (reset) pre.textContent = "";
  if (chunk) pre.textContent += chunk;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
};
"""


def appearance_kwargs() -> dict[str, Any]:
    """App-level Gradio 6 launch() parameters (theme, css, js, head)."""
    return {
        "theme": build_theme(),
        "css": CUSTOM_CSS,
        "js": CUSTOM_JS,
        "head": COLOR_SCHEME_HEAD,
    }


def build_theme() -> gr.themes.Base:
    """Neutral darkroom: true dark gray + off-white accent.

    Achromatic surfaces (no blue/warm cast) so generated images are not
    color-shifted. Accent is light gray, not a hue: it reads as the CTA
    without competing with the frame.
    """
    return gr.themes.Base(
        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="neutral",
        font=["ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    ).set(
        # Page — Material #121212, not #000 and not blue-tinted slate
        body_background_fill="*neutral_50",
        body_background_fill_dark="#121212",
        body_text_color="*neutral_800",
        body_text_color_dark="*neutral_200",
        body_text_color_subdued="*neutral_500",
        body_text_color_subdued_dark="*neutral_400",
        background_fill_primary="white",
        background_fill_primary_dark="#1c1c1c",
        background_fill_secondary="*neutral_100",
        background_fill_secondary_dark="#262626",
        border_color_primary="*neutral_200",
        border_color_primary_dark="#333333",
        color_accent="*neutral_200",
        color_accent_soft="*neutral_100",
        color_accent_soft_dark="#262626",
        # Blocks
        block_background_fill="white",
        block_background_fill_dark="#1c1c1c",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_border_color_dark="#2e2e2e",
        block_label_background_fill="*neutral_100",
        block_label_background_fill_dark="#262626",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_300",
        block_label_text_weight="500",
        block_title_text_color="*neutral_700",
        block_title_text_color_dark="*neutral_300",
        block_title_text_weight="500",
        block_shadow="none",
        block_shadow_dark="none",
        # Inputs
        input_background_fill="white",
        input_background_fill_dark="#262626",
        input_border_color="*neutral_200",
        input_border_color_dark="#404040",
        input_border_color_focus="*neutral_800",
        input_border_color_focus_dark="*neutral_300",
        input_shadow="none",
        input_shadow_focus="0 0 0 3px *neutral_200",
        input_shadow_focus_dark="0 0 0 3px #333333",
        # CTA: charcoal on light, off-white on dark (not pure #fff)
        button_primary_background_fill="*neutral_800",
        button_primary_background_fill_hover="*neutral_900",
        button_primary_background_fill_dark="*neutral_200",
        button_primary_background_fill_hover_dark="*neutral_100",
        button_primary_text_color="white",
        button_primary_text_color_dark="*neutral_900",
        button_primary_border_color="*neutral_800",
        button_primary_border_color_dark="*neutral_200",
        button_secondary_background_fill="white",
        button_secondary_background_fill_dark="#262626",
        button_secondary_background_fill_hover="*neutral_50",
        button_secondary_background_fill_hover_dark="#333333",
        button_secondary_text_color="*neutral_700",
        button_secondary_text_color_dark="*neutral_200",
        button_secondary_border_color="*neutral_200",
        button_secondary_border_color_dark="#404040",
        button_cancel_background_fill="*neutral_200",
        button_cancel_background_fill_dark="#333333",
        button_cancel_text_color="*neutral_800",
        button_cancel_text_color_dark="*neutral_100",
        checkbox_background_color_selected="*neutral_800",
        checkbox_background_color_selected_dark="*neutral_200",
        checkbox_border_color_selected="*neutral_800",
        checkbox_border_color_selected_dark="*neutral_200",
        checkbox_label_background_fill_selected="*neutral_100",
        checkbox_label_background_fill_selected_dark="#333333",
        checkbox_label_text_color_selected="*neutral_900",
        checkbox_label_text_color_selected_dark="*neutral_100",
        slider_color="*neutral_800",
        slider_color_dark="*neutral_300",
        loader_color="*neutral_700",
        loader_color_dark="*neutral_300",
        link_text_color="*neutral_800",
        link_text_color_dark="*neutral_300",
        shadow_drop="none",
        shadow_drop_lg="none",
        panel_border_width="1px",
        panel_border_color="*neutral_200",
        panel_border_color_dark="#2e2e2e",
    )
