"""Gradio theme and CSS: achromatic darkroom so generated images stay the focus."""

from __future__ import annotations

import gradio as gr

CUSTOM_CSS = """
.gradio-container { max-width: 1240px !important; }
.gradio-container::before {
    content: "";
    display: block;
    height: 1px;
    margin: 0 0 1.15rem;
    background: var(--border-color-primary);
}
#generate-btn {
    min-height: 48px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
#status-md {
    font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.875rem;
}
.studio-hero h1 {
    letter-spacing: -0.03em;
    font-weight: 600;
    margin-bottom: 0.35rem !important;
}
.studio-hero p {
    max-width: 64ch;
    color: var(--body-text-color-subdued);
}
.dark #output-gallery,
.dark #output-gallery .grid-wrap,
.dark #output-gallery .image-container,
.dark #output-gallery .thumbnail-item,
.dark #output-gallery .empty {
    background: #0a0a0a !important;
}
footer { display: none !important; }
"""


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
        font=[
            gr.themes.GoogleFont("Instrument Sans", weights=(400, 500, 600, 700)),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("IBM Plex Mono", weights=(400, 500)),
            "ui-monospace",
            "monospace",
        ],
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
