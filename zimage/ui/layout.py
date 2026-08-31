"""Gradio layout for Z-Image-Turbo Studio."""

from __future__ import annotations

from dataclasses import dataclass

from zimage.config import (
    EXAMPLE_PROMPTS,
    IMAGE_FORMAT_CHOICES,
    MAX_BATCH,
    PRECISION_CHOICES,
    QUANTIZE_CHOICES,
    RESOLUTION_PRESETS,
)
import gradio as gr

from zimage.prefs import load_ui_prefs
from zimage.ui.handlers import (
    _clamp_lora_selection,
    cancel_generate_for_training,
    clear_preview_images,
    delete_preview_image,
    generate,
    load_gallery_with_index,
    load_model,
    refresh_loras,
    request_stop,
    restore_ui_prefs,
    save_ui_prefs,
    set_gallery_index,
    sync_lora_weights,
    training_callbacks,
    unload_model,
)
from zimage.ui.status import format_status
from zimage.ui.training_panel import build_training_panel

SYNC_STUDIO_TAB_URL_JS = (
    "(...args) => {"
    " const url = new URL(window.location.href);"
    " const training = document.querySelector('#studio-tab-training-button')"
    "?.getAttribute('aria-selected') === 'true';"
    " if (training) { url.searchParams.set('tab', 'training'); }"
    " else { url.searchParams.delete('tab'); }"
    " history.replaceState(null, '', url.pathname + url.search + url.hash);"
    "}"
)


@dataclass
class StudioNavbar:
    """Top-bar components: shared (empty), generate, and training action rows."""

    clear_btn: gr.Button
    generate_stop_btn: gr.Button
    training_start_btn: gr.Button
    training_stop_btn: gr.Button
    training_clear_btn: gr.Button
    shared_actions: gr.Row
    generate_actions: gr.Row
    training_actions: gr.Row


def build_navbar() -> StudioNavbar:
    """Bootstrap-style top bar: brand on the left, tab-specific actions on the right."""
    with gr.Row(elem_id="studio-navbar", equal_height=True, min_height=52):
        gr.HTML(
            '<span class="studio-brand">Studio</span>',
            elem_id="studio-brand",
        )
        with gr.Row(elem_id="studio-navbar-actions", equal_height=True):
            with gr.Row(elem_id="studio-navbar-shared") as shared_actions:
                pass
            with gr.Row(
                elem_id="studio-navbar-generate",
                visible=True,
            ) as generate_actions:
                clear_btn = gr.Button(
                    "Clear",
                    elem_id="studio-clear-btn",
                    size="sm",
                )
                generate_stop_btn = gr.Button(
                    "Stop",
                    variant="stop",
                    elem_id="studio-stop-btn",
                    size="sm",
                )
            with gr.Row(
                elem_id="studio-navbar-training",
                visible=False,
            ) as training_actions:
                training_start_btn = gr.Button(
                    "Start",
                    variant="primary",
                    elem_id="studio-training-start",
                    size="sm",
                    visible=True,
                )
                training_stop_btn = gr.Button(
                    "Stop",
                    variant="stop",
                    elem_id="studio-training-stop",
                    size="sm",
                    visible=False,
                )
                training_clear_btn = gr.Button(
                    "Clear",
                    elem_id="studio-training-clear",
                    size="sm",
                    visible=True,
                )
    return StudioNavbar(
        clear_btn=clear_btn,
        generate_stop_btn=generate_stop_btn,
        training_start_btn=training_start_btn,
        training_stop_btn=training_stop_btn,
        training_clear_btn=training_clear_btn,
        shared_actions=shared_actions,
        generate_actions=generate_actions,
        training_actions=training_actions,
    )


def _studio_tab_from_query(value: object) -> str:
    if isinstance(value, str) and value.lower() == "training":
        return "training"
    return "generate"


def _navbar_tab_visibility(show_training: bool):
    return gr.update(visible=not show_training), gr.update(visible=show_training)


def on_studio_tab(evt: gr.SelectData):
    """Show Generate or Training navbar rows from the selected tab index."""
    show_training = getattr(evt, "index", None) == 1 or getattr(evt, "value", None) in {
        "training",
        "Training",
    }
    return _navbar_tab_visibility(show_training)


def restore_studio_tab(request: gr.Request | None = None):
    if request is None:
        tab_id = "generate"
    else:
        tab_id = _studio_tab_from_query(
            dict(getattr(request, "query_params", None) or {}).get("tab")
        )
    generate_actions, training_actions = _navbar_tab_visibility(tab_id == "training")
    return gr.update(selected=tab_id), generate_actions, training_actions


def _gallery_buttons(delete_btn, *, share: bool) -> list:
    buttons: list = []
    if share:
        buttons.append("share")
    buttons.extend(["download", "download_all", "fullscreen", delete_btn])
    return buttons


def _pref_inputs(
    prompt,
    resolution,
    steps,
    batch_count,
    output_dir,
    image_format,
    seed,
    random_seed,
    model_id,
    device,
    dtype_name,
    quantize_modules,
    cpu_offload,
    vae_tiling,
    lora_dir,
    lora_adapters,
    lora_weights,
    guidance,
    time_shift,
) -> list:
    return [
        prompt,
        resolution,
        steps,
        batch_count,
        output_dir,
        image_format,
        seed,
        random_seed,
        model_id,
        device,
        dtype_name,
        quantize_modules,
        cpu_offload,
        vae_tiling,
        lora_dir,
        lora_adapters,
        lora_weights,
        guidance,
        time_shift,
    ]


def build_ui(*, share: bool = False) -> gr.Blocks:
    prefs = load_ui_prefs()
    lora_dir_value, lora_files, lora_kept, lora_weight_rows = _clamp_lora_selection(
        prefs["lora_dir"],
        prefs["lora_adapters"],
        prefs["lora_weights"],
    )
    with gr.Blocks(
        title="Z-Image-Turbo Studio",
        fill_height=True,
    ) as demo:
        navbar = build_navbar()
        with gr.Tabs(elem_id="studio-tabs") as tabs:
            with gr.Tab("Generate", id="generate", elem_id="studio-tab-generate"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            value=prefs["prompt"],
                            placeholder="Describe the shot: lighting, lens, composition, on-image text…",
                            elem_id="studio-prompt",
                        )
                        with gr.Row():
                            resolution = gr.Dropdown(
                                choices=RESOLUTION_PRESETS,
                                value=prefs["resolution"],
                                label="Resolution",
                            )
                            steps = gr.Slider(
                                1,
                                20,
                                value=prefs["steps"],
                                step=1,
                                label="Steps (Turbo = 9)",
                            )
                        with gr.Row():
                            batch_count = gr.Number(
                                value=prefs["batch"],
                                precision=0,
                                minimum=1,
                                maximum=MAX_BATCH,
                                label="Batch",
                                info="Images to generate with incremental seeds (seed, seed+1, …).",
                            )
                            output_dir = gr.Textbox(
                                value=prefs["output_dir"],
                                label="Output dir",
                                info="Folder for saved images (Windows paths accepted).",
                                elem_id="studio-output-dir",
                            )
                            image_format = gr.Radio(
                                choices=IMAGE_FORMAT_CHOICES,
                                value=prefs["image_format"],
                                label="Format",
                                info="JPEG is smaller; PNG is lossless.",
                                elem_id="studio-image-format",
                            )
                        with gr.Row():
                            seed = gr.Number(value=prefs["seed"], precision=0, label="Seed")
                            random_seed = gr.Checkbox(value=prefs["random_seed"], label="Random seed")
                        generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")

                        with gr.Accordion("Model & device", open=True):
                            model_id = gr.Textbox(
                                value=prefs["model_id"],
                                label="Model (Hugging Face ID or snapshot path)",
                                info="e.g. Tongyi-MAI/Z-Image-Turbo or E:\\Backup\\huggingface\\hub\\models--Tongyi-MAI--Z-Image-Turbo\\snapshots\\…",
                            )
                            with gr.Row():
                                device = gr.Radio(
                                    choices=["auto", "cuda", "cpu"],
                                    value=prefs["device"],
                                    label="Device",
                                )
                                dtype_name = gr.Radio(
                                    choices=PRECISION_CHOICES,
                                    value=prefs["precision"],
                                    label="Precision",
                                    info="fp8 / int8: torchao on the checked modules (official checkpoint). fp8 needs Ada 8.9+ / Blackwell and cannot use CPU offload.",
                                )
                            quantize_modules = gr.CheckboxGroup(
                                choices=QUANTIZE_CHOICES,
                                value=prefs["quantize_modules"],
                                label="quantize",
                                elem_id="studio-quantize",
                            )
                            with gr.Row():
                                cpu_offload = gr.Checkbox(
                                    value=prefs["cpu_offload"],
                                    label="CPU offload (saves VRAM)",
                                )
                                vae_tiling = gr.Checkbox(
                                    value=prefs["vae_tiling"],
                                    label="VAE tiling",
                                )
                            with gr.Row():
                                load_btn = gr.Button("Load model")
                                unload_btn = gr.Button("Unload")

                        with gr.Accordion("LoRA", open=True, elem_id="studio-lora"):
                            lora_dir = gr.Textbox(
                                value=lora_dir_value,
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
                                choices=lora_files,
                                value=lora_kept,
                                multiselect=True,
                                allow_custom_value=True,
                                label="Adapters",
                                info="Select one or more files. Empty = no LoRA.",
                                elem_id="studio-lora-adapters",
                            )
                            lora_weights = gr.Dataframe(
                                headers=["LoRA", "Strength"],
                                datatype=["str", "number"],
                                value=lora_weight_rows,
                                column_count=2,
                                interactive=True,
                                label="Strength",
                                elem_id="studio-lora-weights",
                            )

                        with gr.Accordion("Advanced", open=False):
                            guidance = gr.Slider(
                                0.0,
                                8.0,
                                value=prefs["guidance"],
                                step=0.1,
                                label="Guidance scale",
                                info="Keep at 0 for Turbo. For full Z-Image, 3–5 is typical.",
                            )
                            time_shift = gr.Slider(
                                1.0,
                                10.0,
                                value=prefs["time_shift"],
                                step=0.1,
                                label="Time shift",
                                info="3 is the Turbo default. Toward 10: more steps on composition. Toward 1: more on fine detail.",
                            )

                    with gr.Column(scale=6):
                        gallery_index = gr.State(0)
                        delete_btn = gr.Button(
                            "Delete",
                            render=False,
                            elem_id="studio-gallery-delete",
                        )
                        gallery = gr.Gallery(
                            label="Output",
                            columns=1,
                            height=640,
                            object_fit="contain",
                            preview=True,
                            format="png",
                            elem_id="output-gallery",
                            buttons=_gallery_buttons(delete_btn, share=share),
                        )
                        used_seed = gr.Textbox(label="Used seed", interactive=False)
                        status = gr.Markdown(format_status(), elem_id="status-md")
                        gr.Examples(
                            examples=EXAMPLE_PROMPTS,
                            inputs=prompt,
                            label="Examples",
                            elem_id="studio-examples",
                        )

            with gr.Tab("Training", id="training", elem_id="studio-tab-training"):
                training = build_training_panel(
                    callbacks=training_callbacks(),
                    start_btn=navbar.training_start_btn,
                    stop_btn=navbar.training_stop_btn,
                    clear_btn=navbar.training_clear_btn,
                )
        tabs.select(
            on_studio_tab,
            outputs=[navbar.generate_actions, navbar.training_actions],
        ).then(None, js=SYNC_STUDIO_TAB_URL_JS)
        demo.load(
            restore_studio_tab,
            outputs=[tabs, navbar.generate_actions, navbar.training_actions],
        )

        pref_inputs = _pref_inputs(
            prompt,
            resolution,
            steps,
            batch_count,
            output_dir,
            image_format,
            seed,
            random_seed,
            model_id,
            device,
            dtype_name,
            quantize_modules,
            cpu_offload,
            vae_tiling,
            lora_dir,
            lora_adapters,
            lora_weights,
            guidance,
            time_shift,
        )
        pref_outputs = list(pref_inputs)

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
        lora_dir_submit = lora_dir.submit(
            refresh_loras,
            inputs=[lora_dir, lora_adapters, lora_weights],
            outputs=[lora_dir, lora_adapters, lora_weights],
        )
        lora_dir_blur = lora_dir.blur(
            refresh_loras,
            inputs=[lora_dir, lora_adapters, lora_weights],
            outputs=[lora_dir, lora_adapters, lora_weights],
        )
        lora_adapters.change(
            sync_lora_weights,
            inputs=[lora_adapters, lora_weights],
            outputs=lora_weights,
        )

        for control in (
            prompt,
            resolution,
            steps,
            batch_count,
            output_dir,
            image_format,
            seed,
            random_seed,
            model_id,
            device,
            dtype_name,
            quantize_modules,
            cpu_offload,
            vae_tiling,
            lora_adapters,
            lora_weights,
            guidance,
            time_shift,
        ):
            control.change(save_ui_prefs, inputs=pref_inputs)

        lora_dir_submit.then(save_ui_prefs, inputs=pref_inputs)
        lora_dir_blur.then(save_ui_prefs, inputs=pref_inputs)

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
                output_dir,
                gallery,
                lora_dir,
                lora_adapters,
                lora_weights,
                image_format,
            ],
            outputs=[gallery, used_seed, seed, status],
            show_progress="minimal",
            show_progress_on=status,
        )
        generate_btn.click(save_ui_prefs, inputs=pref_inputs)
        navbar.generate_stop_btn.click(request_stop, cancels=[generate_event])
        training.start_btn.click(cancel_generate_for_training, cancels=[generate_event])
        navbar.clear_btn.click(
            clear_preview_images,
            inputs=[output_dir],
            outputs=[gallery, gallery_index, status],
        )
        generate_event.then(lambda: 0, outputs=gallery_index)
        delete_btn.click(
            delete_preview_image,
            inputs=[gallery, gallery_index, output_dir],
            outputs=[gallery, gallery_index, status],
        )
        gallery.select(set_gallery_index, outputs=gallery_index)
        demo.load(
            restore_ui_prefs,
            outputs=pref_outputs,
        ).then(
            load_gallery_with_index,
            inputs=[output_dir],
            outputs=[gallery, gallery_index],
        )

        gr.Markdown(
            """
Local Gradio UI for **Tongyi-MAI/Z-Image-Turbo** via `diffusers`.
The model works best in English and Chinese; Russian is supported but weaker.<br>
Turbo: **9 steps**, `guidance_scale = 0` — CFG is already baked in during distillation.

Images are saved to the **Output dir** field (default `./outputs`) as **JPEG** by default (PNG optional).<br>
**fp8** / **int8** quantize the checked modules of official **Z-Image-Turbo**
with torchao.
            """,
            elem_classes=["studio-footer"],
        )
    return demo
