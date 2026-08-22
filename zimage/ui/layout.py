"""Gradio layout for Z-Image-Turbo Studio."""

from __future__ import annotations

import gradio as gr

from zimage.config import (
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_LORA_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUANTIZE_MODULES,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    EXAMPLE_PROMPTS,
    MAX_BATCH,
    PRECISION_CHOICES,
    QUANTIZE_CHOICES,
    RESOLUTION_PRESETS,
)
from zimage.engine.lora import normalize_lora_dir
from zimage.ui.handlers import (
    generate,
    load_gallery,
    load_model,
    refresh_loras,
    request_stop,
    sync_lora_weights,
    unload_model,
)
from zimage.ui.status import format_status


def build_navbar() -> gr.Button:
    """Bootstrap-style top bar: brand on the left, Stop on the right."""
    with gr.Row(elem_id="studio-navbar", equal_height=True, min_height=52):
        gr.HTML(
            '<span class="studio-brand">Studio</span>',
            elem_id="studio-brand",
        )
        with gr.Row(elem_id="studio-navbar-actions", equal_height=True):
            stop_btn = gr.Button(
                "Stop",
                variant="stop",
                elem_id="studio-stop-btn",
                size="sm",
            )
    return stop_btn


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Z-Image-Turbo Studio",
        fill_height=True,
    ) as demo:
        stop_btn = build_navbar()
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
                batch_count = gr.Number(
                    value=DEFAULT_BATCH,
                    precision=0,
                    minimum=1,
                    maximum=MAX_BATCH,
                    label="Batch",
                    info="Images to generate with incremental seeds (seed, seed+1, …).",
                )
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    random_seed = gr.Checkbox(value=True, label="Random seed")
                generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")

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
                            value=DEFAULT_DTYPE if DEFAULT_DTYPE in PRECISION_CHOICES else "fp8",
                            label="Precision",
                            info="fp8 / int8: torchao on the checked modules (official checkpoint). fp8 needs Ada 8.9+ / Blackwell and cannot use CPU offload.",
                        )
                    quantize_modules = gr.CheckboxGroup(
                        choices=QUANTIZE_CHOICES,
                        value=DEFAULT_QUANTIZE_MODULES,
                        label="quantize",
                        elem_id="studio-quantize",
                    )
                    with gr.Row():
                        cpu_offload = gr.Checkbox(value=False, label="CPU offload (saves VRAM)")
                        vae_tiling = gr.Checkbox(value=False, label="VAE tiling")
                    with gr.Row():
                        load_btn = gr.Button("Load model")
                        unload_btn = gr.Button("Unload")

                with gr.Accordion("LoRA", open=True, elem_id="studio-lora"):
                    lora_dir = gr.Textbox(
                        value=normalize_lora_dir(DEFAULT_LORA_DIR),
                        label="Directory",
                        info=(
                            "Local folder of .safetensors / .pt adapters "
                            "(a pasted file path uses its parent folder). "
                            "Leave empty for the base model. "
                            "Adapters are fused into the base weights before "
                            "quantization (VRAM ≈ base model); changing adapters "
                            "or strength reloads the model."
                        ),
                        elem_id="studio-lora-dir",
                    )
                    lora_refresh = gr.Button("Refresh", elem_id="studio-lora-refresh")
                    lora_adapters = gr.Dropdown(
                        choices=[],
                        value=[],
                        multiselect=True,
                        label="Adapters",
                        info="Select one or more files. Empty = no LoRA.",
                        elem_id="studio-lora-adapters",
                    )
                    lora_weights = gr.Dataframe(
                        headers=["LoRA", "Strength"],
                        datatype=["str", "number"],
                        value=[],
                        column_count=2,
                        interactive=True,
                        label="Strength",
                        elem_id="studio-lora-weights",
                    )

                with gr.Accordion("Advanced", open=False):
                    guidance = gr.Slider(
                        0.0,
                        8.0,
                        value=DEFAULT_GUIDANCE,
                        step=0.1,
                        label="Guidance scale",
                        info="Keep at 0 for Turbo. For full Z-Image, 3–5 is typical.",
                    )
                    time_shift = gr.Slider(
                        1.0,
                        10.0,
                        value=3.0,
                        step=0.1,
                        label="Time shift",
                        info="3 is the Turbo default. Toward 10: more steps on composition. Toward 1: more on fine detail.",
                    )

            with gr.Column(scale=6):
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
                status = gr.Markdown(format_status(), elem_id="status-md")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=prompt,
                    label="Examples",
                    elem_id="studio-examples",
                )

        load_btn.click(
            load_model,
            inputs=[model_id, device, dtype_name, cpu_offload, vae_tiling, quantize_modules],
            outputs=status,
        )
        unload_btn.click(unload_model, outputs=status)
        lora_refresh.click(
            refresh_loras,
            inputs=[lora_dir, lora_adapters, lora_weights],
            outputs=[lora_dir, lora_adapters, lora_weights],
        )
        lora_dir.submit(
            refresh_loras,
            inputs=[lora_dir, lora_adapters, lora_weights],
            outputs=[lora_dir, lora_adapters, lora_weights],
        )
        lora_dir.blur(
            refresh_loras,
            inputs=[lora_dir, lora_adapters, lora_weights],
            outputs=[lora_dir, lora_adapters, lora_weights],
        )
        lora_adapters.change(
            sync_lora_weights,
            inputs=[lora_adapters, lora_weights],
            outputs=lora_weights,
        )
        generate_event = generate_btn.click(
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
                quantize_modules,
                batch_count,
                gallery,
                lora_dir,
                lora_adapters,
                lora_weights,
            ],
            outputs=[gallery, used_seed, seed, status],
            show_progress="minimal",
            show_progress_on=status,
        )
        stop_btn.click(request_stop, cancels=[generate_event])
        demo.load(load_gallery, outputs=gallery)

        gr.Markdown(
            """
Local Gradio UI for **Tongyi-MAI/Z-Image-Turbo** via `diffusers`.
The model works best in English and Chinese; Russian is supported but weaker.<br>
Turbo: **9 steps**, `guidance_scale = 0` — CFG is already baked in during distillation.

Images are saved to `outputs/`.<br>
**fp8** / **int8** quantize the checked modules of official **Z-Image-Turbo**
with torchao.
            """,
            elem_classes=["studio-footer"],
        )
    return demo
