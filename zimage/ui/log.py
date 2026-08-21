"""Console logging for the same status lines shown in the web UI."""

from __future__ import annotations

import logging
import re
import sys

log = logging.getLogger("zimage")


def ensure_console_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def plain_status(markdown: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown)
    text = text.replace("`", "").strip()
    return re.sub(r"\n{2,}", "\n", text)


def log_status(markdown: str, status: dict) -> None:
    ensure_console_logging()
    text = plain_status(markdown)
    if not text:
        return
    text = text.replace("\n", "\n      ")
    warning = bool(status.get("demo") or status.get("cpu_torch_on_nvidia"))
    if warning:
        log.warning(text)
    else:
        log.info(text)


def log_error(message: str) -> None:
    ensure_console_logging()
    log.error(message)
