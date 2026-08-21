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
    OUTPUTS_DIR,
    is_truthy,
)
from zimage.engine.demo import demo_image
from zimage.engine.runtime import dtype_from_name, resolve_device, runtime_status

_lock = threading.Lock()
_pipe: Any = None
_pipe_key: tuple | None = None


def load_pipeline(model_id: str, device: str, dtype_name: str, cpu_offload: bool, vae_tiling: bool):
    import torch

    dtype = dtype_from_name(dtype_name)
    if device == "cpu" and dtype in {torch.bfloat16, torch.float16}:
        dtype = torch.float32

    local_only = is_truthy(os.environ.get("HF_HUB_OFFLINE"))
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_only,
    }

    last_error: Exception | None = None
    try:
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — fallback is intentional
        last_error = exc
        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
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

    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
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
):
    global _pipe, _pipe_key

    resolved = resolve_device(device)
    if resolved == "demo":
        return None, runtime_status()

    key = (model_id.strip(), resolved, dtype_name, cpu_offload, vae_tiling)
    with _lock:
        if _pipe is not None and _pipe_key == key:
            status = runtime_status()
            status["loaded"] = True
            status["model"] = model_id
            return _pipe, status
        _pipe = load_pipeline(model_id, resolved, dtype_name, cpu_offload, vae_tiling)
        _pipe_key = key
        status = runtime_status()
        status["loaded"] = True
        status["model"] = model_id
        status["device"] = resolved
        return _pipe, status


def unload_pipeline() -> None:
    global _pipe, _pipe_key
    with _lock:
        _pipe = None
        _pipe_key = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_image(image: Image.Image, seed: int, outputs_dir: Path | None = None) -> Path:
    directory = outputs_dir or OUTPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"zimage-{stamp}-{seed}.png"
    image.save(path)
    return path


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
    progress=None,
    outputs_dir: Path | None = None,
) -> tuple[Image.Image, int, dict[str, Any]]:
    status = runtime_status()
    resolved = resolve_device(device)

    if resolved == "demo" or status["demo"]:
        image = demo_image(prompt, width, height, seed, status.get("demo_reason", ""))
        path = save_image(image, seed, outputs_dir=outputs_dir)
        status["saved"] = str(path)
        status["loaded"] = False
        status["demo"] = True
        return image, seed, status

    if progress is not None:
        progress(0.05, desc="Loading model…")

    pipe, status = ensure_pipeline(model_id, resolved, dtype_name, cpu_offload, vae_tiling)

    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    if hasattr(pipe, "scheduler"):
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=time_shift)
        except Exception:
            pass

    gen_device = "cuda" if resolved == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    if progress is not None:
        progress(0.2, desc="Generating…")

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
        result = pipe(**kwargs, max_sequence_length=DEFAULT_MAX_SEQ)
    except TypeError:
        result = pipe(**kwargs)

    image = result.images[0]
    path = save_image(image, seed, outputs_dir=outputs_dir)
    status = runtime_status()
    status["loaded"] = True
    status["model"] = model_id
    status["saved"] = str(path)
    status["device"] = resolved
    if progress is not None:
        progress(1.0, desc="Done")
    return image, seed, status
