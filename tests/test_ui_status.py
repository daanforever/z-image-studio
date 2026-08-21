from __future__ import annotations

from zimage.config import CUDA_REINSTALL_CMD
from zimage.ui.log import plain_status
from zimage.ui.status import format_status


def test_plain_status_strips_markdown():
    raw = "**Device:** `cuda`\n\n⚠ **without CUDA**\n`pip install torch`"
    assert plain_status(raw) == "Device: cuda\n⚠ without CUDA\npip install torch"


def test_format_status_demo_logs_reason(caplog):
    markdown = format_status({"demo": True, "demo_reason": "ZIMAGE_DEMO=1 is enabled"})
    assert "**Mode:** demo" in markdown
    assert "ZIMAGE_DEMO=1" in markdown
    assert "ZIMAGE_DEMO=1" in caplog.text


def test_format_status_cuda_loaded():
    markdown = format_status(
        {
            "demo": False,
            "device": "cuda",
            "device_name": "NVIDIA GeForce RTX 5080",
            "torch_version": "2.13.0+cu130",
            "cuda_built": "13.0",
            "vram": "1.0 / 15.9 GB",
            "loaded": True,
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "precision": "fp8",
            "saved": "outputs/zimage-1.png",
        }
    )
    assert "RTX 5080" in markdown
    assert "CUDA build: 13.0" in markdown
    assert "loaded" in markdown
    assert "`fp8`" in markdown
    assert "outputs/zimage-1.png" in markdown
    assert "without CUDA" not in markdown


def test_format_status_cpu_torch_warning():
    markdown = format_status(
        {
            "demo": False,
            "cpu_torch_on_nvidia": True,
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": "2.10.0+cpu",
            "cuda_built": "",
            "loaded": False,
        }
    )
    assert "without CUDA" in markdown
    assert CUDA_REINSTALL_CMD in markdown


def test_format_status_extra_line():
    markdown = format_status(
        {
            "demo": False,
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": "x",
            "cuda_built": "no",
            "loaded": False,
        },
        extra="Model unloaded from memory.",
    )
    assert "Model unloaded from memory." in markdown
