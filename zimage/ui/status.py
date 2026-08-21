"""Markdown status block for the Gradio panel (also logged to the console)."""

from __future__ import annotations

from zimage.config import CUDA_REINSTALL_CMD, DEFAULT_DEVICE, DEFAULT_MODEL
from zimage.engine import resolve_device, runtime_status
from zimage.ui.log import log_status


def _format_strength(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _format_loras(loras) -> str:
    if not loras:
        return "none"
    parts = []
    for item in loras:
        name = item.get("name") if isinstance(item, dict) else item[0]
        strength = item.get("strength") if isinstance(item, dict) else item[1]
        parts.append(f"{name} ({_format_strength(strength)})")
    return ", ".join(parts)


def format_status(status: dict | None = None, extra: str = "") -> str:
    status = status or runtime_status()
    if status.get("demo"):
        reason = status.get("demo_reason") or "demo mode"
        markdown = (
            f"**Mode:** demo (model not loaded)\n\n"
            f"{reason}\n\n"
            f"For real generation, install CUDA-enabled PyTorch and set the path "
            f"to `Tongyi-MAI/Z-Image-Turbo`."
        )
        log_status(markdown, status)
        return markdown

    device = status.get("device") or resolve_device(DEFAULT_DEVICE)
    name = status.get("device_name") or device
    lines = [
        f"**Device:** `{device}` · {name}",
        f"**PyTorch:** {status.get('torch_version') or '—'} · CUDA build: {status.get('cuda_built') or 'no'}",
    ]
    if status.get("vram"):
        lines.append(f"**VRAM:** {status['vram']}")
    if status.get("loaded"):
        precision = status.get("precision")
        loaded = f"**Model:** `{status.get('model', DEFAULT_MODEL)}` · loaded"
        if precision:
            loaded += f" · `{precision}`"
        lines.append(loaded)
        lines.append(f"**LoRA:** {_format_loras(status.get('loras'))}")
    else:
        lines.append("**Model:** not in memory yet (loads on first generation)")
    if status.get("cpu_torch_on_nvidia"):
        lines.append(
            "\n⚠ PyTorch was built **without CUDA**. RTX 5080 needs a GPU build:\n"
            f"`{CUDA_REINSTALL_CMD}`"
        )
    if status.get("saved"):
        lines.append(f"**Saved:** `{status['saved']}`")
    if extra:
        lines.append(extra)
    markdown = "\n".join(lines)
    log_status(markdown, status)
    return markdown
