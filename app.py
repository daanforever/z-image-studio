"""Gradio UI for local Z-Image-Turbo inference."""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys

import gradio as gr

from config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    EXAMPLE_PROMPTS,
    RESOLUTION_PRESETS,
    load_dotenv,
    parse_resolution,
)
from engine import ensure_pipeline, generate_image, resolve_device, runtime_status, unload_pipeline

load_dotenv()

log = logging.getLogger("zimage")


def _ensure_console_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _plain_status(markdown: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown)
    text = text.replace("`", "").strip()
    return re.sub(r"\n{2,}", "\n", text)


def _log_status(markdown: str, status: dict) -> None:
    _ensure_console_logging()
    text = _plain_status(markdown)
    if not text:
        return
    text = text.replace("\n", "\n      ")
    warning = bool(status.get("demo") or status.get("cpu_torch_on_nvidia"))
    if warning:
        log.warning(text)
    else:
        log.info(text)


def _log_error(message: str) -> None:
    _ensure_console_logging()
    log.error(message)


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


def format_status(status: dict | None = None, extra: str = "") -> str:
    status = status or runtime_status()
    if status.get("demo"):
        reason = status.get("demo_reason") or "demo mode"
        markdown = (
            f"**Mode:** demo (model not loaded)\n\n"
            f"{reason}\n\n"
            f"For real generation, install CUDA-enabled PyTorch and set the path "
            f"to `Tongyi-MAI/Z-Image-Turbo`."
        )
        _log_status(markdown, status)
        return markdown

    device = status.get("device") or resolve_device(DEFAULT_DEVICE)
    name = status.get("device_name") or device
    lines = [
        f"**Device:** `{device}` · {name}",
        f"**PyTorch:** {status.get('torch_version') or '—'} · CUDA build: {status.get('cuda_built') or 'no'}",
    ]
    if status.get("vram"):
        lines.append(f"**VRAM:** {status['vram']}")
    if status.get("loaded"):
        lines.append(f"**Model:** `{status.get('model', DEFAULT_MODEL)}` · loaded")
    else:
        lines.append("**Model:** not in memory yet (loads on first generation)")
    if status.get("cpu_torch_on_nvidia"):
        lines.append(
            "\n⚠ PyTorch was built **without CUDA**. RTX 5080 needs a GPU build:\n"
            "`pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130`"
        )
    if status.get("saved"):
        lines.append(f"**Saved:** `{status['saved']}`")
    if extra:
        lines.append(extra)
    markdown = "\n".join(lines)
    _log_status(markdown, status)
    return markdown


def load_model(model_id: str, device: str, dtype_name: str, cpu_offload: bool, vae_tiling: bool):
    try:
        _, status = ensure_pipeline(model_id, device, dtype_name, cpu_offload, vae_tiling)
        return format_status(status)
    except Exception as exc:  # noqa: BLE001
        _log_error(str(exc))
        raise gr.Error(str(exc)) from exc


def unload_model():
    unload_pipeline()
    status = runtime_status()
    status["loaded"] = False
    return format_status(status, extra="Model unloaded from memory.")


def generate(
    prompt: str,
    resolution: str,
    seed: int,
    random_seed: bool,
    steps: int,
    guidance: float,
    time_shift: float,
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    gallery: list | None,
    progress=gr.Progress(track_tqdm=True),
):
    prompt = (prompt or "").strip()
    if not prompt:
        _log_error("Enter a prompt.")
        raise gr.Error("Enter a prompt.")

    used_seed = random.randint(1, 2_147_483_647) if random_seed else int(seed)
    width, height = parse_resolution(resolution)

    try:
        image, used_seed, status = generate_image(
            prompt,
            model_id=model_id.strip() or DEFAULT_MODEL,
            device=device,
            dtype_name=dtype_name,
            width=width,
            height=height,
            steps=int(steps),
            guidance=float(guidance),
            seed=used_seed,
            time_shift=float(time_shift),
            cpu_offload=cpu_offload,
            vae_tiling=vae_tiling,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "offline" in message.lower() or "local_files_only" in message.lower():
            message += (
                " Hugging Face network access is disabled. Set HF_HUB_OFFLINE=0 "
                "or provide a full local snapshot."
            )
        _log_error(message)
        raise gr.Error(message) from exc

    items = [image] + list(gallery or [])
    return items[:12], str(used_seed), int(used_seed), format_status(status)


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


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Z-Image-Turbo Studio",
        theme=build_theme(),
        css=CUSTOM_CSS,
        fill_height=True,
        head='<meta name="color-scheme" content="dark light">',
    ) as demo:
        gr.Markdown(
            """
# Z-Image-Turbo Studio
Local Gradio UI for **Tongyi-MAI/Z-Image-Turbo** via `diffusers`.
The model works best in English and Chinese; Russian is supported but weaker.
Turbo: **9 steps**, `guidance_scale = 0` — CFG is already baked in during distillation.
            """,
            elem_classes=["studio-hero"],
        )

        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                    placeholder="Describe the shot: lighting, lens, composition, on-image text…",
                )
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=RESOLUTION_PRESETS,
                        value=DEFAULT_RESOLUTION,
                        label="Resolution",
                    )
                    steps = gr.Slider(1, 20, value=DEFAULT_STEPS, step=1, label="Steps (Turbo = 9)")
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    random_seed = gr.Checkbox(value=True, label="Random seed")
                generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt, label="Examples")

                with gr.Accordion("Model & device", open=True):
                    model_id = gr.Textbox(
                        value=DEFAULT_MODEL,
                        label="Model (Hugging Face ID or snapshot path)",
                        info="e.g. Tongyi-MAI/Z-Image-Turbo or E:\\Backup\\huggingface\\hub\\models--Tongyi-MAI--Z-Image-Turbo\\snapshots\\…",
                    )
                    with gr.Row():
                        device = gr.Radio(
                            choices=["auto", "cuda", "cpu"],
                            value=DEFAULT_DEVICE if DEFAULT_DEVICE in {"auto", "cuda", "cpu"} else "auto",
                            label="Device",
                        )
                        dtype_name = gr.Radio(
                            choices=["bfloat16", "float16", "float32"],
                            value=DEFAULT_DTYPE if DEFAULT_DTYPE in {"bfloat16", "float16", "float32"} else "bfloat16",
                            label="Precision",
                        )
                    with gr.Row():
                        cpu_offload = gr.Checkbox(value=False, label="CPU offload (saves VRAM)")
                        vae_tiling = gr.Checkbox(value=False, label="VAE tiling")
                    with gr.Row():
                        load_btn = gr.Button("Load model")
                        unload_btn = gr.Button("Unload")

                with gr.Accordion("Advanced", open=False):
                    guidance = gr.Slider(
                        0.0,
                        8.0,
                        value=DEFAULT_GUIDANCE,
                        step=0.1,
                        label="Guidance scale",
                        info="Keep at 0 for Turbo. For full Z-Image, 3–5 is typical.",
                    )
                    time_shift = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="Time shift")

            with gr.Column(scale=6):
                status = gr.Markdown(format_status(), elem_id="status-md")
                gallery = gr.Gallery(
                    label="Output",
                    columns=1,
                    height=640,
                    object_fit="contain",
                    preview=True,
                    format="png",
                    elem_id="output-gallery",
                )
                used_seed = gr.Textbox(label="Used seed", interactive=False)

        load_btn.click(
            load_model,
            inputs=[model_id, device, dtype_name, cpu_offload, vae_tiling],
            outputs=status,
        )
        unload_btn.click(unload_model, outputs=status)
        generate_btn.click(
            generate,
            inputs=[
                prompt,
                resolution,
                seed,
                random_seed,
                steps,
                guidance,
                time_shift,
                model_id,
                device,
                dtype_name,
                cpu_offload,
                vae_tiling,
                gallery,
            ],
            outputs=[gallery, used_seed, seed, status],
        )

        gr.Markdown(
            "Images are saved to `outputs/`. "
            "This is not Disty0/q8 quantized weights — official **Z-Image-Turbo** only."
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Gradio studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    _ensure_console_logging()
    args = parse_args()
    demo = build_ui()
    log.info("Starting server at http://%s:%s", args.host, args.port)
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        _log_error(f"Failed to start server: {exc}")
        raise
