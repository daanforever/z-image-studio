"""Lazy Z-Image-Turbo pipeline: CUDA inference with a demo fallback."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from config import (
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_SEQ,
    DEFAULT_MODEL,
    DEFAULT_SHIFT,
    OUTPUTS_DIR,
    is_truthy,
)

_lock = threading.Lock()
_pipe: Any = None
_pipe_key: tuple | None = None


def runtime_status() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": False,
        "torch_version": "",
        "cuda": False,
        "cuda_built": "",
        "device": "cpu",
        "device_name": "CPU",
        "vram": "",
        "demo": False,
        "demo_reason": "",
        "cpu_torch_on_nvidia": False,
    }

    if is_truthy(os.environ.get("ZIMAGE_DEMO")):
        info["demo"] = True
        info["demo_reason"] = "Включён ZIMAGE_DEMO=1"
        return info

    try:
        import torch
    except ImportError:
        info["demo"] = True
        info["demo_reason"] = "PyTorch не установлен"
        return info

    info["torch"] = True
    info["torch_version"] = torch.__version__
    info["cuda_built"] = torch.version.cuda or ""
    info["cuda"] = bool(torch.cuda.is_available())

    if info["cuda"]:
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        info["vram"] = f"{allocated:.1f} / {total:.1f} ГБ"
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"
        if "+cpu" in torch.__version__ or not info["cuda_built"]:
            info["cpu_torch_on_nvidia"] = True
            info["demo_reason"] = (
                "Установлен CPU-PyTorch без CUDA. Для RTX 50xx: "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
            )

    return info


def resolve_device(requested: str) -> str:
    status = runtime_status()
    if status["demo"]:
        return "demo"
    if requested in {"auto", "", None}:
        return "cuda" if status["cuda"] else "cpu"
    if requested == "cuda" and not status["cuda"]:
        return "cpu"
    return requested


def _dtype_from_name(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get((name or "bfloat16").lower(), torch.bfloat16)


def _load_pipeline(model_id: str, device: str, dtype_name: str, cpu_offload: bool, vae_tiling: bool):
    import torch

    dtype = _dtype_from_name(dtype_name)
    if device == "cpu" and dtype in {torch.bfloat16, torch.float16}:
        dtype = torch.float32

    local_only = is_truthy(os.environ.get("HF_HUB_OFFLINE"))
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_only,
    }

    pipe = None
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
            if "ZImagePipeline" in type(last_error).__name__ or "ZImagePipeline" in str(last_error):
                hint = (
                    " Установите свежий diffusers: "
                    "pip install git+https://github.com/huggingface/diffusers"
                )
            raise RuntimeError(
                f"Не удалось загрузить модель {model_id}: {fallback_exc}.{hint}"
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
        _pipe = _load_pipeline(model_id, resolved, dtype_name, cpu_offload, vae_tiling)
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
        import torch
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _demo_image(prompt: str, width: int, height: int, seed: int, reason: str) -> Image.Image:
    width = max(512, min(width, 1536))
    height = max(512, min(height, 1536))
    image = Image.new("RGB", (width, height), "#12100c")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(18 + 28 * t)
        g = int(14 + 10 * t)
        b = int(8 + 4 * (1 - t))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    accent = "#e8a54b"
    draw.rectangle([0, 0, 8, height], fill=accent)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 36)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()

    margin = 48
    draw.text((margin, margin), "Z-Image-Turbo · демо", font=title_font, fill=accent)
    draw.text(
        (margin, margin + 56),
        reason or "Модель не загружена — интерфейс работает без весов.",
        font=small_font,
        fill="#c9bba8",
    )

    wrapped = _wrap_text(prompt.strip() or "(пустой промпт)", body_font, width - margin * 2, draw)
    draw.multiline_text((margin, margin + 110), wrapped, font=body_font, fill="#f4efe6", spacing=8)
    draw.text(
        (margin, height - 72),
        f"{width}×{height}   seed {seed}",
        font=small_font,
        fill="#8a7d6d",
    )
    return image


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) >= 10:
                current += "…"
                break
    lines.append(current)
    return "\n".join(lines[:11])


def save_image(image: Image.Image, seed: int) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUTS_DIR / f"zimage-{stamp}-{seed}.png"
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
) -> tuple[Image.Image, int, dict[str, Any]]:
    status = runtime_status()
    resolved = resolve_device(device)

    if resolved == "demo" or status["demo"]:
        image = _demo_image(prompt, width, height, seed, status.get("demo_reason", ""))
        path = save_image(image, seed)
        status["saved"] = str(path)
        status["loaded"] = False
        status["demo"] = True
        return image, seed, status

    if progress is not None:
        progress(0.05, desc="Загрузка модели…")

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
        progress(0.2, desc="Генерация…")

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
    path = save_image(image, seed)
    status = runtime_status()
    status["loaded"] = True
    status["model"] = model_id
    status["saved"] = str(path)
    status["device"] = resolved
    if progress is not None:
        progress(1.0, desc="Готово")
    return image, seed, status
