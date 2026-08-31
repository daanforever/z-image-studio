"""Child-process VRAM and module-device snapshots for training jobs.

Reads ``torch.cuda`` only when ``is_available()`` is true. Never calls
nvidia-smi or pynvml. Snapshot and log helpers never raise into the job.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger("zimage.training")

MODULE_KEYS = ("vae", "text_encoder", "transformer")
_UNITS = ("B", "KB", "MB", "GB")


def format_bytes(n: int) -> str:
    value = float(n)
    unit = "B"
    for next_unit in _UNITS[1:]:
        if value < 1024:
            break
        value /= 1024
        unit = next_unit
    if unit == "B":
        return f"{int(value)}B"
    return f"{value:.1f}{unit}"


@dataclass(frozen=True)
class GpuUsageSnapshot:
    """One child-process GPU usage sample at a named job phase."""

    phase: str
    cuda_available: bool
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    module_devices: Mapping[str, str]


def snapshot_gpu_usage(
    phase: str,
    components: Any = None,
    *,
    torch_module: Any = None,
) -> GpuUsageSnapshot:
    """Capture CUDA stats and VAE / text-encoder / transformer devices.

    ``torch_module`` is a test injection; production reads the real
    ``torch``. CUDA APIs are not called when ``is_available()`` is false.
    """

    empty = {key: "none" for key in MODULE_KEYS}
    try:
        cuda_available, allocated, reserved, peak = _cuda_stats(torch_module)
        devices = _module_devices(components)
        return GpuUsageSnapshot(
            phase=str(phase),
            cuda_available=cuda_available,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak,
            module_devices=devices,
        )
    except Exception:
        return GpuUsageSnapshot(
            phase=str(phase),
            cuda_available=False,
            allocated_bytes=0,
            reserved_bytes=0,
            peak_allocated_bytes=0,
            module_devices=empty,
        )


def format_gpu_usage(snapshot: GpuUsageSnapshot) -> str:
    """Stable ``job.log`` line for one snapshot."""

    devices = snapshot.module_devices
    return (
        f"gpu usage phase={snapshot.phase} "
        f"cuda={int(snapshot.cuda_available)} "
        f"allocated={format_bytes(snapshot.allocated_bytes)} "
        f"reserved={format_bytes(snapshot.reserved_bytes)} "
        f"peak_allocated={format_bytes(snapshot.peak_allocated_bytes)} "
        f"vae={devices.get('vae', 'none')} "
        f"text_encoder={devices.get('text_encoder', 'none')} "
        f"transformer={devices.get('transformer', 'none')}"
    )


class GpuUsageProbe:
    """Log a snapshot for a phase. Callable so jobs can inject a replacement."""

    def __init__(
        self,
        *,
        torch_module: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._torch = torch_module
        self._log = logger if logger is not None else log

    def __call__(self, phase: str, components: Any = None) -> None:
        self.record(phase, components)

    def record(self, phase: str, components: Any = None) -> None:
        try:
            snapshot = snapshot_gpu_usage(
                phase, components, torch_module=self._torch
            )
            self._log.info(format_gpu_usage(snapshot))
        except Exception:
            return


def record_gpu_usage(phase: str, components: Any = None) -> None:
    """Default probe used when the job does not inject ``gpu_usage_probe``."""

    GpuUsageProbe()(phase, components)


def _cuda_stats(torch_module: Any | None) -> tuple[bool, int, int, int]:
    mod = torch if torch_module is None else torch_module
    cuda = getattr(mod, "cuda", None)
    try:
        available = bool(cuda is not None and cuda.is_available())
    except Exception:
        return False, 0, 0, 0
    if not available:
        return False, 0, 0, 0
    try:
        return (
            True,
            int(cuda.memory_allocated()),
            int(cuda.memory_reserved()),
            int(cuda.max_memory_allocated()),
        )
    except Exception:
        return True, 0, 0, 0


def _module_devices(components: Any) -> dict[str, str]:
    modules = {
        "vae": _component_attr(components, "vae"),
        "text_encoder": _component_attr(components, "text_encoder"),
        "transformer": _component_attr(
            components, "main_transformer", "transformer"
        ),
    }
    return {key: _module_device_label(module) for key, module in modules.items()}


def _component_attr(components: Any, *names: str) -> Any:
    if components is None:
        return None
    for name in names:
        if isinstance(components, Mapping) and name in components:
            return components[name]
        value = getattr(components, name, None)
        if value is not None:
            return value
    return None


def _module_device_label(module: Any) -> str:
    if module is None:
        return "none"
    device = getattr(module, "device", None)
    if device is not None:
        return _device_type(device)
    try:
        return _device_type(next(module.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return "cpu"


def _device_type(device: Any) -> str:
    kind = getattr(device, "type", None)
    if kind is not None:
        return str(kind)
    text = str(device)
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("cpu"):
        return "cpu"
    return text
