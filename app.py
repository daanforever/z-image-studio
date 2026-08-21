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
        reason = status.get("demo_reason") or "демо-режим"
        return (
            f"**Режим:** демо (модель не загружена)\n\n"
            f"{reason}\n\n"
            f"Для настоящей генерации поставьте CUDA-PyTorch и укажите путь "
            f"к `Tongyi-MAI/Z-Image-Turbo`."
        )

    device = status.get("device") or resolve_device(DEFAULT_DEVICE)
    name = status.get("device_name") or device
    lines = [
        f"**Устройство:** `{device}` · {name}",
        f"**PyTorch:** {status.get('torch_version') or '—'} · CUDA build: {status.get('cuda_built') or 'нет'}",
    ]
    if status.get("vram"):
        lines.append(f"**VRAM:** {status['vram']}")
    if status.get("loaded"):
        lines.append(f"**Модель:** `{status.get('model', DEFAULT_MODEL)}` · загружена")
    else:
        lines.append("**Модель:** ещё не в памяти (загрузится при первой генерации)")
    if status.get("cpu_torch_on_nvidia"):
        lines.append(
            "\n⚠ Сборка PyTorch **без CUDA**. На RTX 5080 нужна GPU-сборка:\n"
            "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`"
        )
    if status.get("saved"):
        lines.append(f"**Файл:** `{status['saved']}`")
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
    return format_status(status, extra="Модель выгружена из памяти.")


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
        raise gr.Error("Введите промпт.")

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
                " Сеть для Hugging Face выключена. Поставьте HF_HUB_OFFLINE=0 "
                "или укажите полный локальный snapshot."
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
Локальный Gradio UI для **Tongyi-MAI/Z-Image-Turbo** через `diffusers`.
Модель понимает английский и китайский; русский тоже можно, но слабее.
Turbo: **9 шагов**, `guidance_scale = 0` — CFG уже «запечён» при дистилляции.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Промпт",
                    lines=5,
                    placeholder="Опишите кадр: свет, оптика, композиция, текст на картинке…",
                )
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=RESOLUTION_PRESETS,
                        value=DEFAULT_RESOLUTION,
                        label="Разрешение",
                    )
                    steps = gr.Slider(1, 20, value=DEFAULT_STEPS, step=1, label="Шаги (Turbo = 9)")
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    random_seed = gr.Checkbox(value=True, label="Случайный seed")
                generate_btn = gr.Button("Сгенерировать", variant="primary", elem_id="generate-btn")
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt, label="Примеры")

                with gr.Accordion("Модель и устройство", open=True):
                    model_id = gr.Textbox(
                        value=DEFAULT_MODEL,
                        label="Модель (Hugging Face id или путь к snapshot)",
                        info="Например Tongyi-MAI/Z-Image-Turbo или E:\\Backup\\huggingface\\hub\\models--Tongyi-MAI--Z-Image-Turbo\\snapshots\\…",
                    )
                    with gr.Row():
                        device = gr.Radio(
                            choices=["auto", "cuda", "cpu"],
                            value=DEFAULT_DEVICE if DEFAULT_DEVICE in {"auto", "cuda", "cpu"} else "auto",
                            label="Устройство",
                        )
                        dtype_name = gr.Radio(
                            choices=["bfloat16", "float16", "float32"],
                            value=DEFAULT_DTYPE if DEFAULT_DTYPE in {"bfloat16", "float16", "float32"} else "bfloat16",
                            label="Точность",
                        )
                    with gr.Row():
                        cpu_offload = gr.Checkbox(value=False, label="CPU offload (экономия VRAM)")
                        vae_tiling = gr.Checkbox(value=False, label="VAE tiling")
                    with gr.Row():
                        load_btn = gr.Button("Загрузить модель")
                        unload_btn = gr.Button("Выгрузить")

                with gr.Accordion("Дополнительно", open=False):
                    guidance = gr.Slider(
                        0.0,
                        8.0,
                        value=DEFAULT_GUIDANCE,
                        step=0.1,
                        label="Guidance scale",
                        info="Для Turbo оставляйте 0. Для полной Z-Image обычно 3–5.",
                    )
                    time_shift = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="Time shift")

            with gr.Column(scale=6):
                status = gr.Markdown(format_status(), elem_id="status-md")
                gallery = gr.Gallery(
                    label="Результат",
                    columns=1,
                    height=640,
                    object_fit="contain",
                    preview=True,
                    format="png",
                )
                used_seed = gr.Textbox(label="Использованный seed", interactive=False)

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
            "Картинки сохраняются в папку `outputs/`. "
            "Это не квантованные Disty0/q8 — только официальный **Z-Image-Turbo**."
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
        print(f"Не удалось запустить сервер: {exc}", file=sys.stderr)
        raise
