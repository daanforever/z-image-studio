"""Gradio event handlers: load / unload / generate / training callbacks."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Mapping

import yaml

from zimage.config import (
    DEFAULT_BATCH,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    GALLERY_LIMIT,
    MAX_BATCH,
    canonical_image_format,
    parse_output_dir,
    parse_quantize_modules,
    parse_resolution,
)
import gradio as gr

from zimage.engine import (
    clear_output_images,
    delete_output_image,
    ensure_pipeline,
    generate_image,
    list_output_images,
    runtime_status,
    unload_pipeline,
)
from zimage.engine.pipeline import (
    GPU_LEASE_HELD_MESSAGE,
    clear_training_start_fence,
    set_training_start_fence,
    training_start_fence_is_set,
)
from zimage.engine.lora import (
    DEFAULT_STRENGTH,
    list_lora_files,
    normalize_lora_dir,
    parse_lora_specs,
    weights_map,
)
from zimage.prefs import load_ui_prefs, save_ui_prefs as dump_ui_prefs
from zimage.training.commands import enqueue_update, save_idle_update
from zimage.training.contracts import JobStatus
from zimage.training.job_log import read_job_log_chunk
from zimage.training.jobs import (
    CONFIG_FILE,
    create_or_open_job,
    load_job_state,
    resolve_job_path,
)
from zimage.training.schema import TrainingConfigError, validate_job_document
from zimage.ui.log import log_error
from zimage.ui.status import format_status
from zimage.ui.training_panel import TrainingCallbacks
from zimage.ui.training_process import (
    TrainingProcessManager,
    create_training_process_manager,
)

_stop_event = threading.Event()
_LEASE_WAIT_TIMEOUT_SECONDS = 300.0
_LEASE_WAIT_POLL_SECONDS = 0.05
_training_manager: TrainingProcessManager | None = None
_PREVIEW_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def request_stop() -> None:
    """Signal the active batch to stop after the current image (no rollback)."""
    _stop_event.set()


def cancel_generate_for_training() -> None:
    """Training Start hook: stop the in-flight Generate batch."""
    request_stop()


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
    """Remove the previewed image from disk and refresh the gallery."""
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


def clear_preview_images(output_dir=None):
    """Remove all generated images from the Output dir and refresh the gallery."""
    outputs_path = parse_output_dir(output_dir)
    deleted = clear_output_images(outputs_dir=outputs_path)
    if deleted == 0:
        gr.Warning("No images to clear.")
        return [], None, format_status(extra="No images to clear.")
    return (
        [],
        None,
        format_status(extra=f"Cleared {deleted} images from `{outputs_path}`."),
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


def _clamp_lora_selection(directory, selected=None, current_df=None):
    """Normalize LoRA dir, keep on-disk adapters, and sync their weights."""
    normalized = normalize_lora_dir(directory)
    files = list_lora_files(normalized)
    kept = [name for name in _as_name_list(selected) if name in files]
    return normalized, files, kept, sync_lora_weights(kept, current_df)


def refresh_loras(directory, selected=None, current_df=None):
    normalized, files, kept, weights = _clamp_lora_selection(
        directory, selected, current_df
    )
    return (
        normalized,
        gr.Dropdown(
            choices=files,
            value=kept,
            multiselect=True,
            allow_custom_value=True,
        ),
        weights,
    )


def sync_lora_weights(selected, current_df=None):
    previous = weights_map(current_df)
    rows = []
    for name in _as_name_list(selected):
        rows.append([name, previous.get(name, DEFAULT_STRENGTH)])
    return rows


def save_ui_prefs(
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
) -> None:
    """Persist all editable UI fields to config.yaml."""
    normalized, _files, kept, weights = _clamp_lora_selection(
        lora_dir, lora_adapters, lora_weights
    )
    dump_ui_prefs(
        {
            "prompt": "" if prompt is None else str(prompt),
            "resolution": resolution,
            "steps": steps,
            "batch": batch_count,
            "output_dir": "" if output_dir is None else str(output_dir),
            "image_format": image_format,
            "seed": seed,
            "random_seed": random_seed,
            "model_id": model_id,
            "device": device,
            "precision": dtype_name,
            "quantize_modules": quantize_modules,
            "cpu_offload": cpu_offload,
            "vae_tiling": vae_tiling,
            "lora_dir": normalized,
            "lora_adapters": kept,
            "lora_weights": weights,
            "guidance": guidance,
            "time_shift": time_shift,
        }
    )


def restore_ui_prefs():
    """Restore all editable UI fields from config.yaml and rescan LoRAs."""
    data = load_ui_prefs()
    lora_dir, adapters, weights = refresh_loras(
        data.get("lora_dir"),
        data.get("lora_adapters"),
        data.get("lora_weights"),
    )
    return (
        data.get("prompt", ""),
        data.get("resolution"),
        data.get("steps"),
        data.get("batch"),
        data.get("output_dir"),
        data.get("image_format"),
        data.get("seed"),
        data.get("random_seed"),
        data.get("model_id"),
        data.get("device"),
        data.get("precision"),
        data.get("quantize_modules"),
        data.get("cpu_offload"),
        data.get("vae_tiling"),
        lora_dir,
        adapters,
        weights,
        data.get("guidance"),
        data.get("time_shift"),
    )


def load_model(
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    quantize_modules=None,
):
    if training_start_fence_is_set():
        raise gr.Error(GPU_LEASE_HELD_MESSAGE)
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
    image_format=DEFAULT_IMAGE_FORMAT,
    progress=gr.Progress(),
) -> Generator[tuple, None, None]:
    prompt = (prompt or "").strip()
    if not prompt:
        log_error("Enter a prompt.")
        raise gr.Error("Enter a prompt.")

    count = _parse_batch_count(batch_count)
    outputs_path = parse_output_dir(output_dir)
    quantize_transformer, quantize_text_encoder = parse_quantize_modules(quantize_modules)
    fmt = canonical_image_format(image_format)
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
                image_format=fmt,
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


def training_callbacks() -> TrainingCallbacks:
    """Wire the Training tab to jobs + the process manager. No panel logic."""
    return TrainingCallbacks(
        list_jobs=list_training_jobs,
        create_or_open=create_or_open_training_job,
        load_job=load_training_job,
        validate_yaml=validate_training_yaml,
        save_yaml=save_training_yaml,
        start_job=start_training_job,
        stop_job=stop_training_job,
        poll_state=poll_training_state,
        poll_log=poll_training_log,
        queue_update=queue_training_update,
    )


def list_training_jobs() -> list[str]:
    """Scan ``jobs_dir`` for job folders that contain ``config.yaml``."""
    root = _jobs_dir()
    if not root.is_dir():
        return []
    jobs: list[str] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and (child / CONFIG_FILE).is_file():
            jobs.append(child.name)
    return jobs


def create_or_open_training_job(name: str) -> dict[str, Any]:
    """Create a job or open the existing slug without rewriting files."""
    job_dir = create_or_open_job(name, _jobs_dir())
    return _job_view(job_dir)


def load_training_job(job_id: str) -> dict[str, Any]:
    """Load canonical YAML, operational state, and preview paths."""
    return _job_view(_require_job_dir(job_id))


def validate_training_yaml(job_id: str, text: str) -> dict[str, Any]:
    """Validate editor text. Return ``ok`` / ``error``."""
    _require_job_dir(job_id)
    try:
        validate_job_document(_parse_job_yaml(text))
    except (TrainingConfigError, TypeError, ValueError, yaml.YAMLError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def save_training_yaml(job_id: str, text: str) -> dict[str, Any]:
    """Idle validated atomic save. Running ``state.json`` queues an update."""
    job_dir = _require_job_dir(job_id)
    if load_job_state(job_dir).status is JobStatus.RUNNING:
        return queue_training_update(job_id, text)
    save_idle_update(job_dir, _parse_job_yaml(text))
    return {
        "mode": "saved",
        "config_text": (job_dir / CONFIG_FILE).read_text(encoding="utf-8"),
        "state": _state_mapping(load_job_state(job_dir)),
        "previews": _list_previews(job_dir),
    }


def queue_training_update(job_id: str, text: str) -> dict[str, Any]:
    """Enqueue when RUNNING; otherwise idle-save like the CLI path."""
    job_dir = _require_job_dir(job_id)
    if load_job_state(job_dir).status is not JobStatus.RUNNING:
        save_idle_update(job_dir, _parse_job_yaml(text))
        return {
            "mode": "saved",
            "config_text": (job_dir / CONFIG_FILE).read_text(encoding="utf-8"),
            "state": _state_mapping(load_job_state(job_dir)),
            "previews": _list_previews(job_dir),
        }
    enqueue_update(job_dir, _parse_job_yaml(text))
    return {
        "mode": "queued",
        "state": _state_mapping(load_job_state(job_dir)),
        "previews": _list_previews(job_dir),
    }


def start_training_job(job_id: str) -> dict[str, Any]:
    """Handoff the GPU lease, unload inference, then start the trainer child."""
    job_dir = _require_job_dir(job_id)
    manager = _get_training_process_manager()
    if manager.is_running() or training_start_fence_is_set():
        raise RuntimeError("training is already running")
    request_stop()
    guard = _wait_for_gpu_lease()
    set_training_start_fence()
    try:
        guard.release()
        unload_pipeline()
        _sync_and_empty_cuda()
        manager.start(job_id)
        try:
            _wait_for_foreign_gpu_holder()
        except Exception:
            # Lease-wait timeout / failure: kill the child before clearing the
            # fence so Generate cannot race a still-alive trainer.
            manager.stop()
            raise
    finally:
        # Child holds the GPU lease, or start failed / timed out.
        clear_training_start_fence()
    return _job_view(job_dir, message="Start requested.")


def stop_training_job(job_id: str) -> dict[str, Any]:
    """Immediate Stop: kill the trainer child. No extra checkpoint."""
    job_dir = _require_job_dir(job_id)
    manager = _get_training_process_manager()
    active = getattr(manager, "job_id", None)
    if active is not None and active != job_id:
        raise RuntimeError(
            f"stop refused: manager is running job {active!r}, not {job_id!r}"
        )
    manager.stop()
    return _job_view(job_dir, message="Stop requested.")


def poll_training_state(job_id: str) -> dict[str, Any]:
    """Return ``state.json`` plus preview image paths under ``previews/``."""
    job_dir = _require_job_dir(job_id)
    return {
        "state": _state_mapping(load_job_state(job_dir)),
        "previews": _list_previews(job_dir),
    }


def poll_training_log(job_id: str, offset: int) -> dict[str, Any]:
    """Return a bounded ``logs/job.log`` delta. Never raises into the poller."""
    try:
        job_dir = _require_job_dir(job_id)
        result = read_job_log_chunk(job_dir, offset)
    except Exception:
        return {"chunk": "", "next_offset": offset, "reset": False}
    return {
        "chunk": result.chunk,
        "next_offset": result.next_offset,
        "reset": result.reset,
    }


def _jobs_dir() -> Path:
    from zimage.config import ROOT
    from zimage.training.schema import resolve_training_paths

    configured = Path(resolve_training_paths().jobs_dir)
    if configured.is_absolute():
        return configured
    return (ROOT / configured).resolve()


def _get_training_process_manager() -> TrainingProcessManager:
    global _training_manager
    if _training_manager is None:
        _training_manager = create_training_process_manager(jobs_dir=_jobs_dir())
    return _training_manager


def _create_handoff_guard():
    from zimage.training.runtime_guard import create_runtime_guard

    return create_runtime_guard()


def _live_foreign_lease_pid() -> int | None:
    """Return another live PID that currently holds the GPU lease, if any.

    Reads the FileRuntimeGuard lock file without calling ``acquire``, so an
    in-flight start cannot steal the GPU while waiting for the child.
    """
    import os

    from zimage.training.runtime_guard import create_runtime_guard, pid_is_alive

    guard = create_runtime_guard()
    path = Path(guard.lock_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="ascii", errors="ignore")
    except OSError:
        return None
    token = text.replace("\0", " ").strip().split()
    if not token or not token[0].isdigit():
        return None
    pid = int(token[0])
    if pid <= 0 or pid == os.getpid() or not pid_is_alive(pid):
        return None
    return pid


def _wait_for_foreign_gpu_holder() -> None:
    """Block until a live foreign PID holds the GPU lease, or time out."""
    deadline = time.monotonic() + _LEASE_WAIT_TIMEOUT_SECONDS
    while True:
        if _live_foreign_lease_pid() is not None:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for the trainer to take the GPU lease."
            )
        time.sleep(_LEASE_WAIT_POLL_SECONDS)


def _wait_for_gpu_lease():
    """Wait until this process can acquire the lease (inference finished).

    Does not acquire — and does not return a held lease — while the training
    start fence is set.
    """
    guard = _create_handoff_guard()
    deadline = time.monotonic() + _LEASE_WAIT_TIMEOUT_SECONDS
    while True:
        if training_start_fence_is_set():
            raise RuntimeError("training is already running")
        if guard.acquire():
            if training_start_fence_is_set():
                guard.release()
                raise RuntimeError("training is already running")
            return guard
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for the GPU lease; stop Generate before starting training."
            )
        time.sleep(_LEASE_WAIT_POLL_SECONDS)


def _sync_and_empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _require_job_dir(job_id: str) -> Path:
    job_dir = resolve_job_path(_jobs_dir(), job_id)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"job does not exist: {job_id}")
    return job_dir


def _job_view(job_dir: Path, *, message: str = "") -> dict[str, Any]:
    payload = {
        "job_id": job_dir.name,
        "config_text": (job_dir / CONFIG_FILE).read_text(encoding="utf-8"),
        "state": _state_mapping(load_job_state(job_dir)),
        "previews": _list_previews(job_dir),
    }
    if message:
        payload["message"] = message
    return payload


def _state_mapping(state: Any) -> dict[str, Any]:
    status = getattr(state, "status", "")
    return {
        "job_id": state.job_id,
        "status": getattr(status, "value", status),
        "step": state.step,
        "epoch": state.epoch,
        "last_error": state.last_error,
        "exit_code": state.exit_code,
    }


def _list_previews(job_dir: Path) -> list[str]:
    root = job_dir / "previews"
    if not root.is_dir():
        return []
    paths = [
        child
        for child in root.rglob("*")
        if child.is_file() and child.suffix.lower() in _PREVIEW_SUFFIXES
    ]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in paths]


def _parse_job_yaml(text: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"invalid job YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TrainingConfigError("YAML must contain a mapping")
    return document

