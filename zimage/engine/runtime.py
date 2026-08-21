"""Device / dtype detection without loading the diffusion pipeline."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any

from zimage.config import CUDA_REINSTALL_CMD, is_truthy


def try_import_torch() -> ModuleType | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


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
        info["demo_reason"] = "ZIMAGE_DEMO=1 is enabled"
        return info

    torch = try_import_torch()
    if torch is None:
        info["demo"] = True
        info["demo_reason"] = "PyTorch is not installed"
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
        info["vram"] = f"{allocated:.1f} / {total:.1f} GB"
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"
        if "+cpu" in torch.__version__ or not info["cuda_built"]:
            info["cpu_torch_on_nvidia"] = True
            info["demo_reason"] = (
                "CPU-only PyTorch is installed (no CUDA). For RTX 50xx: "
                + CUDA_REINSTALL_CMD
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


def dtype_from_name(name: str):
    import torch

    from zimage.engine.quantization import is_quantized_precision

    if is_quantized_precision(name):
        # Compute dtype for torchao quantized weights (int8 / fp8).
        return torch.bfloat16

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get((name or "bfloat16").lower(), torch.bfloat16)
