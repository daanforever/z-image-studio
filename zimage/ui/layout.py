"""Gradio layout for Z-Image-Turbo Studio."""

from __future__ import annotations

import gradio as gr

from zimage.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    EXAMPLE_PROMPTS,
    PRECISION_CHOICES,
    RESOLUTION_PRESETS,
)
from zimage.ui.handlers import generate, load_model, unload_model
from zimage.ui.status import format_status


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Z-Image-Turbo Studio",
        fill_height=True,
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
                            choices=PRECISION_CHOICES,
                            value=DEFAULT_DTYPE if DEFAULT_DTYPE in PRECISION_CHOICES else "bfloat16",
                            label="Precision",
                            info="fp8 / int8: torchao on the DiT (official checkpoint). fp8 needs Ada 8.9+ / Blackwell.",
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
            "**fp8** / **int8** quantize the official **Z-Image-Turbo** DiT with torchao — "
            "not Disty0/SDNQ checkpoints."
        )
    return demo
