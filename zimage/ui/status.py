"""Markdown status block for the Gradio panel (also logged to the console)."""

from __future__ import annotations

from zimage.config import CUDA_REINSTALL_CMD, DEFAULT_DEVICE, DEFAULT_MODEL
from zimage.engine import resolve_device, runtime_status
from zimage.ui.log import log_status


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
        lines.append(f"**Model:** `{status.get('model', DEFAULT_MODEL)}` · loaded")
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
