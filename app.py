"""Gradio UI for local Z-Image-Turbo inference."""

from __future__ import annotations

import argparse
import random
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

CUSTOM_CSS = """
.gradio-container { max-width: 1240px !important; }
#generate-btn { min-height: 52px; font-size: 1.05rem; font-weight: 600; }
#status-md { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
footer { display: none !important; }
"""


def format_status(status: dict | None = None, extra: str = "") -> str:
    status = status or runtime_status()
    if status.get("demo"):
        reason = status.get("demo_reason") or "demo mode"
        return (
            f"**Mode:** demo (model not loaded)\n\n"
            f"{reason}\n\n"
            f"For real generation, install CUDA-enabled PyTorch and set the path "
            f"to `Tongyi-MAI/Z-Image-Turbo`."
        )

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
            "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`"
        )
    if status.get("saved"):
        lines.append(f"**Saved:** `{status['saved']}`")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def load_model(model_id: str, device: str, dtype_name: str, cpu_offload: bool, vae_tiling: bool):
    try:
        _, status = ensure_pipeline(model_id, device, dtype_name, cpu_offload, vae_tiling)
        return format_status(status)
    except Exception as exc:  # noqa: BLE001
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
        raise gr.Error(message) from exc

    items = [image] + list(gallery or [])
    return items[:12], str(used_seed), int(used_seed), format_status(status)


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("DM Sans"),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Z-Image-Turbo Studio", fill_height=True) as demo:
        gr.Markdown(
            """
# Z-Image-Turbo Studio
Local Gradio UI for **Tongyi-MAI/Z-Image-Turbo** via `diffusers`.
The model works best in English and Chinese; Russian is supported but weaker.
Turbo: **9 steps**, `guidance_scale = 0` — CFG is already baked in during distillation.
            """
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
    args = parse_args()
    demo = build_ui()
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=build_theme(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(f"Failed to start server: {exc}", file=sys.stderr)
        raise
