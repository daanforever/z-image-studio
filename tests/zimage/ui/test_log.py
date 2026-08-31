from __future__ import annotations

import logging

from zimage.ui.log import ensure_console_logging, log, log_error, log_status, plain_status


def test_plain_status_strips_markdown():
    raw = "**Device:** `cuda`\n\n⚠ **without CUDA**\n`pip install torch`"
    assert plain_status(raw) == "Device: cuda\n⚠ without CUDA\npip install torch"


def test_log_status_skips_empty_markdown(caplog):
    with caplog.at_level(logging.INFO, logger="zimage"):
        log_status("   \n\n  ", {})
    assert caplog.records == []


def test_log_status_info_for_loaded_cuda(caplog):
    with caplog.at_level(logging.INFO, logger="zimage"):
        log_status("**Device:** `cuda`", {"demo": False, "cpu_torch_on_nvidia": False})
    assert "Device: cuda" in caplog.text
    assert any(record.levelno == logging.INFO for record in caplog.records)


def test_log_status_warning_in_demo(caplog):
    with caplog.at_level(logging.INFO, logger="zimage"):
        log_status("**Mode:** demo", {"demo": True})
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_log_error(caplog):
    with caplog.at_level(logging.ERROR, logger="zimage"):
        log_error("boom")
    assert "boom" in caplog.text


def test_ensure_console_logging_is_idempotent():
    ensure_console_logging()
    count = len(log.handlers)
    ensure_console_logging()
    assert len(log.handlers) == count
    assert count >= 1
