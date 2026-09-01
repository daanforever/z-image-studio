"""Child-process VRAM and module-device snapshots for training jobs.

Reads ``torch.cuda`` only when ``is_available()`` is true. Queries
nvidia-smi for board-level used/total memory; failures become zeros.
Snapshot and log helpers never raise into the job. The default probe
does not run a garbage-collection pass before sampling.

GPU probe toggles live in root ``config.yaml`` (``training.debug``)
and job ``config.yaml`` (``debug``); job keys override root. There
are no environment variables or a parallel config source. See
``resolve_gpu_usage_settings``. When ``debug.detailed`` is true,
``DetailedGpuUsageProbe`` adds per-module CUDA nbytes, leftover
tensor groups, and a caching-allocator line (``inactive_split`` is
fragmentation). Leftover groups are live Python tensors and are not
expected to sum to ``nvidia_used``; use ``reserved − allocated`` and
``inactive_split``. Compact ``phase=step`` / ``cache_encode`` run after
the job's garbage-collection pass and before ``empty_cache``.
Otherwise the compact line is the only log output.
"""

from __future__ import annotations

import gc
import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from zimage.training.schema import GpuUsageSettings

log = logging.getLogger("zimage.training")

MODULE_KEYS = ("vae", "text_encoder", "transformer", "sampling_transformer")
NBYTES_BUCKETS = (
    "vae",
    "text_encoder",
    "main_transformer",
    "sampling_transformer",
    "optimizer_state",
    "preview_embed_maps",
)
TOP_LEFTOVER_GROUPS = 40
_UNITS = ("B", "KB", "MB", "GB")
_MIB = 1024 * 1024
_ALLOCATOR_STATS = (
    ("allocated", "allocated_bytes.all.current", True),
    ("active", "active_bytes.all.current", True),
    ("inactive_split", "inactive_split_bytes.all.current", True),
    ("reserved", "reserved_bytes.all.current", True),
    ("retries", "num_alloc_retries", False),
    ("ooms", "num_ooms", False),
)


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


@dataclass
class GpuProbeContext:
    """Runtime objects for a GPU usage probe.

    ``phase_peak_bytes`` is the allocator max measured inside a
    ``PeakMemoryScope`` (or ``None`` if the phase was not scoped).
    """

    components: Any = None
    optimizer: Any = None
    transformer: Any = None
    phase_peak_bytes: int | None = None
    preview_prompt_embeddings: Any = None
    preview_negative_embeddings: Any = None
    preview_sampler: Any = None


@dataclass(frozen=True)
class GpuUsageSnapshot:
    """One child-process GPU usage sample at a named job phase.

    ``peak_allocated_bytes`` is the process-lifetime CUDA allocator max
    (``torch.cuda.max_memory_allocated``), not reset between phases.
    ``phase_peak_allocated_bytes`` is the max since ``PeakMemoryScope``
    enter (0 if the phase was not scoped). When a scope peak is provided
    it is never below current ``allocated_bytes``. ``nvidia_used_bytes`` is
    nvidia-smi used memory at this instant (includes CUDA context and
    memory outside the PyTorch allocator).
    """

    phase: str
    cuda_available: bool
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    module_devices: Mapping[str, str]
    phase_peak_allocated_bytes: int = 0
    nvidia_used_bytes: int = 0
    nvidia_total_bytes: int = 0


class PeakMemoryScope:
    """Reset CUDA peak stats, run work, synchronize, then read the peak.

    On enter: ``reset_peak_memory_stats``. On exit: ``synchronize`` then
    ``max_memory_allocated``. Missing CUDA is a no-op with ``peak_bytes`` 0.
    """

    def __init__(self, *, torch_module: Any = None) -> None:
        self._torch = torch_module
        self.peak_bytes = 0

    def __enter__(self) -> PeakMemoryScope:
        cuda = _cuda_module(self._torch)
        if cuda is None:
            return self
        try:
            cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, *exc: object) -> bool:
        cuda = _cuda_module(self._torch)
        if cuda is None:
            return False
        try:
            cuda.synchronize()
            self.peak_bytes = int(cuda.max_memory_allocated())
        except Exception:
            self.peak_bytes = 0
        return False


def snapshot_gpu_usage(
    phase: str,
    components: Any = None,
    *,
    torch_module: Any = None,
    context: Any = None,
) -> GpuUsageSnapshot:
    """Capture CUDA stats, nvidia-smi, and module devices.

    ``torch_module`` is a test injection; production reads the real
    ``torch``. CUDA APIs are not called when ``is_available()`` is false.
    A ``GpuProbeContext`` may be passed as ``context`` or as the second
    positional argument (components shorthand otherwise).
    """

    empty = {key: "none" for key in MODULE_KEYS}
    try:
        probe_context, probe_components = _normalize_probe_arg(components, context)
        cuda_available, allocated, reserved, peak = _cuda_stats(torch_module)
        devices = _module_devices(probe_components)
        phase_peak = 0
        if probe_context is not None and probe_context.phase_peak_bytes is not None:
            phase_peak = max(int(probe_context.phase_peak_bytes), allocated)
        nvidia_used, nvidia_total = (0, 0)
        if cuda_available:
            nvidia_used, nvidia_total = _nvidia_memory(torch_module=torch_module)
        return GpuUsageSnapshot(
            phase=str(phase),
            cuda_available=cuda_available,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak,
            module_devices=devices,
            phase_peak_allocated_bytes=phase_peak,
            nvidia_used_bytes=nvidia_used,
            nvidia_total_bytes=nvidia_total,
        )
    except Exception:
        return GpuUsageSnapshot(
            phase=str(phase),
            cuda_available=False,
            allocated_bytes=0,
            reserved_bytes=0,
            peak_allocated_bytes=0,
            module_devices=empty,
            phase_peak_allocated_bytes=0,
            nvidia_used_bytes=0,
            nvidia_total_bytes=0,
        )


def format_gpu_usage(snapshot: GpuUsageSnapshot) -> str:
    """Stable compact ``job.log`` line for one snapshot."""

    devices = snapshot.module_devices
    return (
        f"gpu usage phase={snapshot.phase} "
        f"cuda={int(snapshot.cuda_available)} "
        f"allocated={format_bytes(snapshot.allocated_bytes)} "
        f"reserved={format_bytes(snapshot.reserved_bytes)} "
        f"peak_allocated={format_bytes(snapshot.peak_allocated_bytes)} "
        f"phase_peak={format_bytes(snapshot.phase_peak_allocated_bytes)} "
        f"nvidia_used={format_bytes(snapshot.nvidia_used_bytes)} "
        f"nvidia_total={format_bytes(snapshot.nvidia_total_bytes)} "
        f"vae={devices.get('vae', 'none')} "
        f"text_encoder={devices.get('text_encoder', 'none')} "
        f"transformer={devices.get('transformer', 'none')} "
        f"sampling_transformer={devices.get('sampling_transformer', 'none')}"
    )


def collect_module_nbytes(context: Any = None) -> dict[str, int]:
    """CUDA nbytes per named bucket. CPU and missing modules are zero."""

    nbytes, _ids = _module_nbytes_and_ids(context)
    return nbytes


def collect_leftover_groups(
    context: Any = None,
    *,
    top: int = TOP_LEFTOVER_GROUPS,
) -> tuple[list[dict[str, Any]], int]:
    """Top leftover CUDA tensor groups not in named buckets.

    ``leftover_nbytes`` is the full leftover total, not only the top groups.
    """

    _nbytes, named_ids = _module_nbytes_and_ids(context)
    return _leftover_groups(named_ids, top=top)


def format_gpu_usage_detailed(
    snapshot: GpuUsageSnapshot,
    nbytes: Mapping[str, int],
    leftover_groups: Sequence[Mapping[str, Any]] | None = None,
    leftover_nbytes: int = 0,
    allocator_stats: Mapping[str, int] | None = None,
) -> str:
    """Compact line plus nbytes buckets, optional allocator stats, leftover."""

    groups = leftover_groups if leftover_groups is not None else ()
    lines = [
        format_gpu_usage(snapshot),
        (
            "gpu usage   "
            f"vae={format_bytes(int(nbytes.get('vae', 0)))} "
            f"text_encoder={format_bytes(int(nbytes.get('text_encoder', 0)))} "
            f"main_transformer={format_bytes(int(nbytes.get('main_transformer', 0)))} "
            f"sampling_transformer="
            f"{format_bytes(int(nbytes.get('sampling_transformer', 0)))} "
            f"optimizer_state="
            f"{format_bytes(int(nbytes.get('optimizer_state', 0)))} "
            f"preview_embed_maps="
            f"{format_bytes(int(nbytes.get('preview_embed_maps', 0)))} "
            f"leftover={format_bytes(int(leftover_nbytes))}"
        ),
    ]
    if allocator_stats is not None:
        lines.append(_format_allocator_line(allocator_stats))
    for group in groups:
        lines.append(
            "gpu usage   leftover "
            f"shape={group['shape']} "
            f"dtype={group['dtype']} "
            f"count={group['count']} "
            f"nbytes={format_bytes(int(group['nbytes']))}"
        )
    return "\n".join(lines)


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

    def __call__(self, phase: str, context: Any = None) -> None:
        """``context`` may be a ``GpuProbeContext`` or components (shorthand)."""

        self.record(phase, context)

    def record(self, phase: str, context: Any = None) -> None:
        try:
            snapshot = snapshot_gpu_usage(
                phase, context, torch_module=self._torch
            )
            self._log.info(format_gpu_usage(snapshot))
        except Exception:
            return


class DetailedGpuUsageProbe(GpuUsageProbe):
    """Compact snapshot plus named CUDA buckets, allocator stats, leftover."""

    def record(self, phase: str, context: Any = None) -> None:
        try:
            snapshot = snapshot_gpu_usage(
                phase, context, torch_module=self._torch
            )
            nbytes, named_ids = _module_nbytes_and_ids(context)
            leftover: list[dict[str, Any]] = []
            leftover_nbytes = 0
            if snapshot.cuda_available:
                leftover, leftover_nbytes = _leftover_groups(named_ids)
            text = format_gpu_usage_detailed(
                snapshot,
                nbytes,
                leftover,
                leftover_nbytes,
                allocator_stats=_allocator_stats(self._torch),
            )
            for line in text.splitlines():
                self._log.info(line)
        except Exception:
            return


def _default_gpu_usage_probe(
    settings: GpuUsageSettings,
    *,
    torch_module: Any = None,
    logger: logging.Logger | None = None,
) -> GpuUsageProbe:
    """Return the YAML-selected probe. ``detailed`` chooses the subclass."""

    if settings.detailed:
        return DetailedGpuUsageProbe(torch_module=torch_module, logger=logger)
    return GpuUsageProbe(torch_module=torch_module, logger=logger)


def record_gpu_usage(phase: str, context: Any = None) -> None:
    """Default probe used when the job does not inject ``gpu_usage_probe``."""

    GpuUsageProbe()(phase, context)


def _normalize_probe_arg(
    components: Any,
    context: Any,
) -> tuple[GpuProbeContext | None, Any]:
    if isinstance(components, GpuProbeContext):
        return components, components.components
    if isinstance(context, GpuProbeContext):
        probe_components = (
            components if components is not None else context.components
        )
        return context, probe_components
    if context is not None and components is None:
        return None, context
    return None, components


def _cuda_module(torch_module: Any | None) -> Any | None:
    mod = torch if torch_module is None else torch_module
    cuda = getattr(mod, "cuda", None)
    try:
        if cuda is None or not cuda.is_available():
            return None
    except Exception:
        return None
    return cuda


def _allocator_stats(torch_module: Any | None = None) -> dict[str, int] | None:
    """Return caching-allocator stats, or ``None`` when CUDA/stats are missing."""

    cuda = _cuda_module(torch_module)
    reader = getattr(cuda, "memory_stats", None) if cuda is not None else None
    if not callable(reader):
        return None
    try:
        raw = reader()
        return {name: int(raw.get(key, 0)) for name, key, _bytes in _ALLOCATOR_STATS}
    except Exception:
        return None


def _format_allocator_line(stats: Mapping[str, int]) -> str:
    parts = []
    for name, _key, as_bytes in _ALLOCATOR_STATS:
        value = int(stats.get(name, 0))
        text = format_bytes(value) if as_bytes else str(value)
        parts.append(f"{name}={text}")
    return "gpu usage   allocator " + " ".join(parts)


def _cuda_stats(torch_module: Any | None) -> tuple[bool, int, int, int]:
    cuda = _cuda_module(torch_module)
    if cuda is None:
        return False, 0, 0, 0
    try:
        cuda.synchronize()
        return (
            True,
            int(cuda.memory_allocated()),
            int(cuda.memory_reserved()),
            int(cuda.max_memory_allocated()),
        )
    except Exception:
        return True, 0, 0, 0


def _nvidia_memory(
    device_index: int | None = None,
    *,
    torch_module: Any = None,
) -> tuple[int, int]:
    """Return ``(used_bytes, total_bytes)`` from ``nvidia-smi -i {index}``.

    ``device_index`` defaults to ``torch.cuda.current_device()``. Any error
    (missing binary, parse failure, CUDA error) becomes ``(0, 0)``.
    """

    try:
        index = device_index
        if index is None:
            mod = torch if torch_module is None else torch_module
            index = int(mod.cuda.current_device())
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            return 0, 0
        return _parse_nvidia_smi_memory(completed.stdout)
    except Exception:
        return 0, 0


def _parse_nvidia_smi_memory(text: str) -> tuple[int, int]:
    """Parse nvidia-smi CSV ``used, total`` (MiB) into bytes."""

    line = text.strip().splitlines()[0]
    used_raw, total_raw = (part.strip() for part in line.split(",", 1))
    return _nvidia_field_to_bytes(used_raw), _nvidia_field_to_bytes(total_raw)


def _nvidia_field_to_bytes(field: str) -> int:
    token = field.strip().split()[0]
    return int(float(token)) * _MIB


def _module_devices(components: Any) -> dict[str, str]:
    modules = {
        "vae": _component_attr(components, "vae"),
        "text_encoder": _component_attr(components, "text_encoder"),
        "transformer": _component_attr(
            components, "main_transformer", "transformer"
        ),
        "sampling_transformer": _component_attr(
            components, "sampling_transformer"
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


def _probe_context(context: Any) -> GpuProbeContext:
    if isinstance(context, GpuProbeContext):
        return context
    probe_context, components = _normalize_probe_arg(context, None)
    if probe_context is not None:
        return probe_context
    return GpuProbeContext(components=components)


def _named_bucket_tensors(context: Any) -> dict[str, list[Any]]:
    probe = _probe_context(context)
    components = probe.components
    main = probe.transformer
    if main is None:
        main = _component_attr(components, "main_transformer", "transformer")
    preview_tensors = _nested_tensors(probe.preview_prompt_embeddings)
    preview_tensors.extend(_nested_tensors(probe.preview_negative_embeddings))
    sampler = probe.preview_sampler
    if sampler is not None:
        preview_tensors.extend(
            _nested_tensors(getattr(sampler, "prompt_embeddings", None))
        )
        preview_tensors.extend(
            _nested_tensors(getattr(sampler, "negative_prompt_embeddings", None))
        )
    return {
        "vae": _module_tensors(_component_attr(components, "vae")),
        "text_encoder": _module_tensors(
            _component_attr(components, "text_encoder")
        ),
        "main_transformer": _module_tensors(main),
        "sampling_transformer": _module_tensors(
            _component_attr(components, "sampling_transformer")
        ),
        "optimizer_state": _optimizer_tensors(probe.optimizer),
        "preview_embed_maps": preview_tensors,
    }


def _module_nbytes_and_ids(context: Any) -> tuple[dict[str, int], set[int]]:
    nbytes = {key: 0 for key in NBYTES_BUCKETS}
    named_ids: set[int] = set()
    for key, tensors in _named_bucket_tensors(context).items():
        amount, ids = _cuda_nbytes_and_ids(tensors)
        nbytes[key] = amount
        named_ids.update(ids)
    return nbytes, named_ids


def _module_tensors(module: Any) -> list[Any]:
    if module is None:
        return []
    found: list[Any] = []
    try:
        found.extend(list(module.parameters(recurse=True)))
    except Exception:
        pass
    try:
        found.extend(list(module.buffers(recurse=True)))
    except Exception:
        pass
    return found


def _optimizer_tensors(optimizer: Any) -> list[Any]:
    if optimizer is None:
        return []
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        return []
    found: list[Any] = []
    for value in state.values():
        found.extend(_nested_tensors(value))
    return found


def _nested_tensors(value: Any) -> list[Any]:
    found: list[Any] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        try:
            if isinstance(item, torch.Tensor):
                found.append(item)
                return
        except Exception:
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)

    walk(value)
    return found


def _is_cuda_tensor(obj: Any) -> bool:
    try:
        cls = type(obj)
        if cls is not torch.Tensor and not issubclass(cls, torch.Tensor):
            return False
        return obj.device.type == "cuda"
    except Exception:
        return False


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.nbytes)
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _cuda_nbytes_and_ids(tensors: list[Any]) -> tuple[int, set[int]]:
    ids: set[int] = set()
    seen_storage: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if not _is_cuda_tensor(tensor):
            continue
        ids.add(id(tensor))
        try:
            nbytes = _tensor_nbytes(tensor)
        except Exception:
            continue
        try:
            key = (int(tensor.untyped_storage().data_ptr()), nbytes)
            if key in seen_storage:
                continue
            seen_storage.add(key)
        except Exception:
            pass
        total += nbytes
    return total, ids


def _leftover_groups(
    named_ids: set[int],
    *,
    top: int = TOP_LEFTOVER_GROUPS,
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[tuple[int, ...], str], dict[str, int]] = {}
    total = 0
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if not _is_cuda_tensor(obj):
                continue
            token = id(obj)
            if token in named_ids or token in seen:
                continue
            seen.add(token)
            nbytes = _tensor_nbytes(obj)
            key = (tuple(int(dim) for dim in obj.shape), str(obj.dtype))
        except Exception:
            continue
        bucket = groups.setdefault(key, {"count": 0, "nbytes": 0})
        bucket["count"] += 1
        bucket["nbytes"] += nbytes
        total += nbytes
    ranked = sorted(
        groups.items(), key=lambda item: item[1]["nbytes"], reverse=True
    )
    leftover = [
        {
            "shape": list(shape),
            "dtype": dtype,
            "count": stats["count"],
            "nbytes": stats["nbytes"],
        }
        for (shape, dtype), stats in ranked[:top]
    ]
    return leftover, total
