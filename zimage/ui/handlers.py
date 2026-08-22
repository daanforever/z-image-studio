"""Gradio event handlers: load / unload / generate."""

from __future__ import annotations

import random
import threading
from collections.abc import Generator
from pathlib import Path

from zimage.config import (
    DEFAULT_BATCH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    GALLERY_LIMIT,
    MAX_BATCH,
    parse_output_dir,
    parse_quantize_modules,
    parse_resolution,
)
import gradio as gr

from zimage.engine import (
    delete_output_image,
    ensure_pipeline,
    generate_image,
    list_output_images,
    runtime_status,
    unload_pipeline,
)
from zimage.engine.lora import (
    DEFAULT_STRENGTH,
    list_lora_files,
    normalize_lora_dir,
    parse_lora_specs,
    weights_map,
)
from zimage.ui.log import log_error
from zimage.ui.status import format_status

_stop_event = threading.Event()


def request_stop() -> None:
    """Signal the active batch to stop after the current image (no rollback)."""
    _stop_event.set()


def load_gallery(output_dir=None) -> list[str]:
    """Populate the Output gallery from disk on page load."""
    return list_output_images(outputs_dir=parse_output_dir(output_dir))


def load_gallery_with_index(output_dir=None) -> tuple[list[str], int | None]:
    """Populate the Output gallery and reset selection to the newest item."""
    items = load_gallery(output_dir)
    return items, 0 if items else None


def _gallery_item_path(item) -> str | None:
    """Extract a filesystem path from a Gradio gallery item, if present."""
    if item is None:
        return None
    if isinstance(item, (str, Path)):
        text = str(item).strip()
        return text or None
    if isinstance(item, dict):
        for key in ("path", "name", "orig_name"):
            value = item.get(key)
            if value:
                return str(value)
        image = item.get("image")
        if isinstance(image, dict):
            for key in ("path", "name", "orig_name"):
                value = image.get(key)
                if value:
                    return str(value)
        if isinstance(image, str) and image.strip():
            return image.strip()
    if isinstance(item, (list, tuple)) and item:
        return _gallery_item_path(item[0])
    return None


def _path_under_dir(path: str | Path, directory: Path) -> bool:
    try:
        Path(path).resolve().relative_to(directory.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def set_gallery_index(evt: gr.SelectData) -> int | None:
    """Track the previewed gallery index from select events."""
    index = getattr(evt, "index", None)
    if index is None:
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def delete_preview_image(gallery, selected_index, output_dir=None):
    """Remove the previewed PNG from disk and refresh the gallery."""
    items = list(gallery or [])
    outputs_path = parse_output_dir(output_dir)

    if not items:
        gr.Warning("No image to delete.")
        return items, None, format_status(extra="No image to delete.")

    try:
        index = 0 if selected_index is None else int(selected_index)
    except (TypeError, ValueError):
        index = -1
    if index < 0 or index >= len(items):
        gr.Warning("No image to delete.")
        return items, selected_index, format_status(extra="No image to delete.")

    candidate = _gallery_item_path(items[index])
    if candidate is None or not _path_under_dir(candidate, outputs_path):
        disk_paths = list_output_images(outputs_dir=outputs_path)
        if index < len(disk_paths):
            candidate = disk_paths[index]

    if not candidate:
        gr.Warning("No image to delete.")
        return items, selected_index, format_status(extra="No image to delete.")

    deleted = delete_output_image(candidate, outputs_dir=outputs_path)
    if deleted is None:
        log_error(f"Refused to delete path outside Output dir: {candidate}")
        raise gr.Error("Cannot delete a file outside the Output dir.")

    remaining = list_output_images(outputs_dir=outputs_path)
    if remaining:
        new_index = min(index, len(remaining) - 1)
    else:
        new_index = None
    return (
        remaining,
        new_index,
        format_status(extra=f"Deleted `{deleted}`."),
    )


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


def _as_name_list(names) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        text = names.strip()
        return [text] if text else []
    return [str(item).strip() for item in names if str(item).strip()]


def refresh_loras(directory, selected=None, current_df=None):
    normalized = normalize_lora_dir(directory)
    files = list_lora_files(normalized)
    kept = [name for name in _as_name_list(selected) if name in files]
    return (
        normalized,
        gr.Dropdown(choices=files, value=kept, multiselect=True),
        sync_lora_weights(kept, current_df),
    )


def sync_lora_weights(selected, current_df=None):
    previous = weights_map(current_df)
    rows = []
    for name in _as_name_list(selected):
        rows.append([name, previous.get(name, DEFAULT_STRENGTH)])
    return rows


def save_ui_prefs(prompt, lora_dir) -> dict[str, str]:
    """Snapshot Prompt and LoRA Directory for browser localStorage."""
    return {
        "prompt": "" if prompt is None else str(prompt),
        "lora_dir": normalize_lora_dir(lora_dir),
    }


def restore_ui_prefs(prefs):
    """Restore Prompt / LoRA Directory and rescan adapters from the saved dir."""
    data = prefs if isinstance(prefs, dict) else {}
    prompt = "" if data.get("prompt") is None else str(data.get("prompt"))
    lora_dir, adapters, weights = refresh_loras(data.get("lora_dir"), None, None)
    return prompt, lora_dir, adapters, weights


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
    output_dir=DEFAULT_OUTPUT_DIR,
    gallery: list | None = None,
    lora_dir: str = "",
    lora_names=None,
    lora_weights=None,
    progress=gr.Progress(),
) -> Generator[tuple, None, None]:
    prompt = (prompt or "").strip()
    if not prompt:
        log_error("Enter a prompt.")
        raise gr.Error("Enter a prompt.")

    count = _parse_batch_count(batch_count)
    outputs_path = parse_output_dir(output_dir)
    quantize_transformer, quantize_text_encoder = parse_quantize_modules(quantize_modules)
    _stop_event.clear()

    base_seed = random.randint(1, 2_147_483_647) if random_seed else int(seed)
    width, height = parse_resolution(resolution)
    model = model_id.strip() or DEFAULT_MODEL
    loras = parse_lora_specs(lora_dir, lora_names, lora_weights)
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
                loras=loras,
                progress=image_progress,
                outputs_dir=outputs_path,
            )
        except Exception as exc:  # noqa: BLE001
            message = _offline_hint(str(exc))
            log_error(message)
            if produced:
                extra = f"Stopped after {produced} of {count}: {message}"
                yield (
                    items[:GALLERY_LIMIT],
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
            items[:GALLERY_LIMIT],
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
            items[:GALLERY_LIMIT],
            _format_used_seed(base_seed, last_seed) if produced else "",
            int(last_seed) if produced else int(seed) if seed is not None else base_seed,
            format_status(last_status, extra=extra),
        )
