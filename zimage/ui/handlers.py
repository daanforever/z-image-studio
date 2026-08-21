"""Gradio event handlers: load / unload / generate."""

from __future__ import annotations

import random

import gradio as gr

from zimage.config import DEFAULT_MODEL, parse_resolution
from zimage.engine import ensure_pipeline, generate_image, runtime_status, unload_pipeline
from zimage.ui.log import log_error
from zimage.ui.status import format_status


def load_model(model_id: str, device: str, dtype_name: str, cpu_offload: bool, vae_tiling: bool):
    try:
        _, status = ensure_pipeline(model_id, device, dtype_name, cpu_offload, vae_tiling)
        return format_status(status)
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
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
        log_error("Enter a prompt.")
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
        log_error(message)
        raise gr.Error(message) from exc

    items = [image] + list(gallery or [])
    return items[:12], str(used_seed), int(used_seed), format_status(status)
