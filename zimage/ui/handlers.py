"""Gradio event handlers: load / unload / generate."""

from __future__ import annotations

import random
import threading
from collections.abc import Generator

import gradio as gr

from zimage.config import DEFAULT_BATCH, DEFAULT_MODEL, MAX_BATCH, parse_quantize_modules, parse_resolution
from zimage.engine import ensure_pipeline, generate_image, runtime_status, unload_pipeline
from zimage.ui.log import log_error
from zimage.ui.status import format_status

_stop_event = threading.Event()


def request_stop() -> None:
    """Signal the active batch to stop after the current image (no rollback)."""
    _stop_event.set()


def _parse_batch_count(batch_count) -> int:
    if batch_count is None:
        log_error("Batch count must be an integer between 1 and 9999.")
        raise gr.Error("Batch count must be an integer between 1 and 9999.")
    try:
        count = int(batch_count)
    except (TypeError, ValueError) as exc:
        log_error("Batch count must be an integer between 1 and 9999.")
        raise gr.Error("Batch count must be an integer between 1 and 9999.") from exc
    if count < 1 or count > MAX_BATCH:
        log_error(f"Batch count must be between 1 and {MAX_BATCH}.")
        raise gr.Error(f"Batch count must be between 1 and {MAX_BATCH}.")
    return count


def _format_used_seed(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}–{end}"


def _offline_hint(message: str) -> str:
    if "offline" in message.lower() or "local_files_only" in message.lower():
        return (
            message
            + " Hugging Face network access is disabled. Set HF_HUB_OFFLINE=0 "
            "or provide a full local snapshot."
        )
    return message


def _image_progress(progress, index: int, count: int):
    """Map a per-image 0..1 fraction onto the overall batch progress bar."""
    if progress is None:
        return None

    def report(fraction, desc="") -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        overall = (index + clamped) / count
        label = f"Image {index + 1} / {count}"
        if desc:
            label = f"{label} — {desc}"
        progress(overall, desc=label)

    return report


def load_model(
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    quantize_modules=None,
):
    quantize_transformer, quantize_text_encoder = parse_quantize_modules(quantize_modules)
    try:
        _, status = ensure_pipeline(
            model_id,
            device,
            dtype_name,
            cpu_offload,
            vae_tiling,
            quantize_transformer=quantize_transformer,
            quantize_text_encoder=quantize_text_encoder,
        )
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
    quantize_modules=None,
    batch_count=DEFAULT_BATCH,
    gallery: list | None = None,
    progress=gr.Progress(),
) -> Generator[tuple, None, None]:
    prompt = (prompt or "").strip()
    if not prompt:
        log_error("Enter a prompt.")
        raise gr.Error("Enter a prompt.")

    count = _parse_batch_count(batch_count)
    quantize_transformer, quantize_text_encoder = parse_quantize_modules(quantize_modules)
    _stop_event.clear()

    base_seed = random.randint(1, 2_147_483_647) if random_seed else int(seed)
    width, height = parse_resolution(resolution)
    model = model_id.strip() or DEFAULT_MODEL
    items = list(gallery or [])
    last_seed = base_seed
    last_status: dict | None = None
    produced = 0
    stopped = False

    for i in range(count):
        if _stop_event.is_set():
            stopped = True
            break

        current_seed = base_seed + i
        image_progress = _image_progress(progress, i, count)

        try:
            image, used_seed, status = generate_image(
                prompt,
                model_id=model,
                device=device,
                dtype_name=dtype_name,
                width=width,
                height=height,
                steps=int(steps),
                guidance=float(guidance),
                seed=current_seed,
                time_shift=float(time_shift),
                cpu_offload=cpu_offload,
                vae_tiling=vae_tiling,
                quantize_transformer=quantize_transformer,
                quantize_text_encoder=quantize_text_encoder,
                progress=image_progress,
            )
        except Exception as exc:  # noqa: BLE001
            message = _offline_hint(str(exc))
            log_error(message)
            if produced:
                extra = f"Stopped after {produced} of {count}: {message}"
                yield (
                    items[:12],
                    _format_used_seed(base_seed, last_seed),
                    int(last_seed),
                    format_status(last_status, extra=extra),
                )
            raise gr.Error(message) from exc

        last_seed = int(used_seed)
        last_status = status
        produced += 1
        items = [image] + items
        yield (
            items[:12],
            _format_used_seed(base_seed, last_seed),
            int(last_seed),
            format_status(status),
        )

        # Keep the bar visible between streamed gallery updates mid-batch.
        if progress is not None and i + 1 < count and not _stop_event.is_set():
            progress((i + 1) / count, desc=f"Image {i + 2} / {count}")

    if stopped:
        extra = f"Stopped after {produced} of {count}."
        yield (
            items[:12],
            _format_used_seed(base_seed, last_seed) if produced else "",
            int(last_seed) if produced else int(seed) if seed is not None else base_seed,
            format_status(last_status, extra=extra),
        )
