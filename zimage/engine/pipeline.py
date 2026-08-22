"""Lazy Z-Image-Turbo pipeline: CUDA inference with a demo fallback."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from zimage.config import (
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_SEQ,
    DEFAULT_MODEL,
    DEFAULT_SHIFT,
    GALLERY_LIMIT,
    OUTPUTS_DIR,
    canonical_precision,
    is_truthy,
)
from zimage.engine.demo import demo_image
from zimage.engine.lora import (
    LoraSpec,
    lora_identity_key,
    reset_lora_adapters,
    status_loras,
    sync_lora_adapters,
)
from zimage.engine.quantization import (
    apply_quantization,
    is_fp8_precision,
    is_quantized_precision,
    require_fp8_device,
    require_torchao,
    should_quantize,
)
from zimage.engine.runtime import dtype_from_name, resolve_device, runtime_status

_lock = threading.Lock()
_pipe: Any = None
_pipe_key: tuple | None = None


def _instantiate_pipeline(model_id: str, kwargs: dict[str, Any]):
    last_error: Exception | None = None
    try:
        from diffusers import ZImagePipeline

        return ZImagePipeline.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — fallback is intentional
        last_error = exc
        try:
            from diffusers import DiffusionPipeline

            return DiffusionPipeline.from_pretrained(model_id, **kwargs)
        except Exception as fallback_exc:  # noqa: BLE001
            hint = ""
            if last_error is not None and (
                "ZImagePipeline" in type(last_error).__name__ or "ZImagePipeline" in str(last_error)
            ):
                hint = (
                    " Install a recent diffusers: "
                    "pip install git+https://github.com/huggingface/diffusers"
                )
            raise RuntimeError(
                f"Failed to load model {model_id}: {fallback_exc}.{hint}"
            ) from fallback_exc


def _reclaim_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _loaded_status(model_id: str, device: str, dtype_name: str) -> dict[str, Any]:
    status = runtime_status()
    status["loaded"] = True
    status["model"] = model_id
    status["device"] = device
    status["precision"] = canonical_precision(dtype_name)
    return status


def _pipeline_key(
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    quantize_transformer: bool,
    quantize_text_encoder: bool,
    loras: tuple[LoraSpec, ...] | list[LoraSpec] | None = (),
) -> tuple:
    dtype_name = canonical_precision(dtype_name)
    if not is_quantized_precision(dtype_name):
        quantize_transformer = False
        quantize_text_encoder = False
    else:
        quantize_transformer = bool(quantize_transformer)
        quantize_text_encoder = bool(quantize_text_encoder)
    return (
        model_id.strip(),
        device,
        dtype_name,
        cpu_offload,
        vae_tiling,
        quantize_transformer,
        quantize_text_encoder,
        lora_identity_key(loras),
    )


def load_pipeline(
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
    loras: tuple[LoraSpec, ...] | list[LoraSpec] | None = (),
):
    import torch

    dtype_name = canonical_precision(dtype_name)
    lora_specs = tuple(loras or ())
    quantize = should_quantize(dtype_name, quantize_transformer, quantize_text_encoder)
    if quantize:
        require_torchao()
    if quantize and is_fp8_precision(dtype_name):
        require_fp8_device(device)
        if cpu_offload:
            raise RuntimeError(
                "fp8 cannot be combined with CPU offload. "
                "Disable CPU offload, or use int8 if you need offload."
            )

    dtype = dtype_from_name(dtype_name)
    if device == "cpu" and dtype in {torch.bfloat16, torch.float16}:
        dtype = torch.float32

    local_only = is_truthy(os.environ.get("HF_HUB_OFFLINE"))
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_only,
    }

    pipe = _instantiate_pipeline(model_id, kwargs)

    # Fuse then quantize on GPU when the model will live there. With CPU
    # offload, stay on CPU so peak VRAM is already the reduced footprint.
    work_on_gpu = device == "cuda" and not cpu_offload
    if lora_specs:
        sync_lora_adapters(pipe, lora_specs, device=("cuda" if work_on_gpu else None))
        _reclaim_memory()

    quantize_on_gpu = quantize and work_on_gpu
    if quantize and not quantize_on_gpu:
        apply_quantization(
            pipe,
            dtype_name,
            quantize_transformer=quantize_transformer,
            quantize_text_encoder=quantize_text_encoder,
        )

    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        if quantize_on_gpu:
            apply_quantization(
                pipe,
                dtype_name,
                quantize_transformer=quantize_transformer,
                quantize_text_encoder=quantize_text_encoder,
                device=device,
            )
            _reclaim_memory()
        pipe.to(device)

    if vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    return pipe


def ensure_pipeline(
    model_id: str,
    device: str,
    dtype_name: str = DEFAULT_DTYPE,
    cpu_offload: bool = False,
    vae_tiling: bool = False,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
    loras: tuple[LoraSpec, ...] | list[LoraSpec] | None = (),
):
    global _pipe, _pipe_key

    resolved = resolve_device(device)
    if resolved == "demo":
        return None, runtime_status()

    dtype_name = canonical_precision(dtype_name)
    lora_specs = tuple(loras or ())
    key = _pipeline_key(
        model_id,
        resolved,
        dtype_name,
        cpu_offload,
        vae_tiling,
        quantize_transformer,
        quantize_text_encoder,
        loras=lora_specs,
    )
    with _lock:
        if _pipe is not None and _pipe_key == key:
            return _pipe, _loaded_status(model_id, resolved, dtype_name)
        _pipe = None
        _pipe_key = None
        reset_lora_adapters()
        _reclaim_memory()
        _pipe = load_pipeline(
            model_id,
            resolved,
            dtype_name,
            cpu_offload,
            vae_tiling,
            quantize_transformer=quantize_transformer,
            quantize_text_encoder=quantize_text_encoder,
            loras=lora_specs,
        )
        _pipe_key = key
        return _pipe, _loaded_status(model_id, resolved, dtype_name)


def unload_pipeline() -> None:
    global _pipe, _pipe_key
    with _lock:
        _pipe = None
        _pipe_key = None
    reset_lora_adapters()
    _reclaim_memory()


def save_image(image: Image.Image, seed: int, outputs_dir: Path | None = None) -> Path:
    directory = outputs_dir or OUTPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"zimage-{stamp}-{seed}.png"
    image.save(path)
    return path


def list_output_images(
    outputs_dir: Path | None = None,
    limit: int | None = None,
) -> list[str]:
    """Newest-first PNG paths under outputs_dir, capped at limit."""
    directory = outputs_dir or OUTPUTS_DIR
    if not directory.is_dir():
        return []
    cap = GALLERY_LIMIT if limit is None else int(limit)
    paths = [
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() == ".png"
    ]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(path) for path in paths[: max(0, cap)]]


def delete_output_image(
    path: str | Path,
    outputs_dir: Path | None = None,
) -> Path | None:
    """Unlink a PNG under outputs_dir. Returns the path, or None if refused."""
    directory = (outputs_dir or OUTPUTS_DIR).resolve()
    try:
        target = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if target.suffix.lower() != ".png":
        return None
    try:
        target.relative_to(directory)
    except ValueError:
        return None
    if target.exists() and not target.is_file():
        return None
    try:
        target.unlink(missing_ok=True)
    except OSError:
        return None
    return target


def _pipeline_cache_hit(
    model_id: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    vae_tiling: bool,
    quantize_transformer: bool,
    quantize_text_encoder: bool,
    loras: tuple[LoraSpec, ...] | list[LoraSpec] | None = (),
) -> bool:
    key = _pipeline_key(
        model_id,
        device,
        dtype_name,
        cpu_offload,
        vae_tiling,
        quantize_transformer,
        quantize_text_encoder,
        loras=loras,
    )
    return _pipe is not None and _pipe_key == key


def _invoke_pipe(pipe, kwargs: dict[str, Any]):
    try:
        return pipe(**kwargs, max_sequence_length=DEFAULT_MAX_SEQ)
    except TypeError:
        return pipe(**kwargs)


def generate_image(
    prompt: str,
    *,
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype_name: str = DEFAULT_DTYPE,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int = 42,
    time_shift: float = DEFAULT_SHIFT,
    cpu_offload: bool = False,
    vae_tiling: bool = False,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
    loras: tuple[LoraSpec, ...] | list[LoraSpec] | None = (),
    progress=None,
    outputs_dir: Path | None = None,
) -> tuple[Image.Image, int, dict[str, Any]]:
    status = runtime_status()
    resolved = resolve_device(device)
    lora_specs = tuple(loras or ())

    if resolved == "demo" or status["demo"]:
        if progress is not None:
            progress(0.5, desc="Demo…")
        image = demo_image(prompt, width, height, seed, status.get("demo_reason", ""))
        if progress is not None:
            progress(0.98, desc="Saving…")
        path = save_image(image, seed, outputs_dir=outputs_dir)
        status["saved"] = str(path)
        status["loaded"] = False
        status["demo"] = True
        if progress is not None:
            progress(1.0, desc="Done")
        return image, seed, status

    needs_load = not _pipeline_cache_hit(
        model_id,
        resolved,
        dtype_name,
        cpu_offload,
        vae_tiling,
        quantize_transformer,
        quantize_text_encoder,
        loras=lora_specs,
    )
    if progress is not None and needs_load:
        progress(0.0, desc="Loading model…")

    pipe, _status = ensure_pipeline(
        model_id,
        resolved,
        dtype_name,
        cpu_offload,
        vae_tiling,
        quantize_transformer=quantize_transformer,
        quantize_text_encoder=quantize_text_encoder,
        loras=lora_specs,
    )

    if progress is not None and needs_load:
        progress(0.02, desc="Loading model…")

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    if hasattr(pipe, "scheduler"):
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=time_shift)
        except Exception:
            pass

    gen_device = "cuda" if resolved == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    total_steps = max(int(steps), 1)

    def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        if progress is not None:
            frac = min(0.05 + 0.90 * ((step_index + 1) / total_steps), 0.95)
            progress(frac, desc=f"Generating… {step_index + 1}/{total_steps}")
        return callback_kwargs

    if progress is not None:
        progress(0.05, desc="Generating…")

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "generator": generator,
    }
    # Turbo ignores CFG; keep the arg for API compatibility.
    try:
        result = _invoke_pipe(pipe, {**kwargs, "callback_on_step_end": on_step_end})
    except TypeError:
        # Pipeline rejected callback_on_step_end; keep a coarse "Generating…" until return.
        result = _invoke_pipe(pipe, kwargs)

    image = result.images[0]
    if progress is not None:
        progress(0.98, desc="Saving…")
    path = save_image(image, seed, outputs_dir=outputs_dir)
    status = _loaded_status(model_id, resolved, dtype_name)
    status["loras"] = status_loras(lora_specs)
    status["saved"] = str(path)
    if progress is not None:
        progress(1.0, desc="Done")
    return image, seed, status
