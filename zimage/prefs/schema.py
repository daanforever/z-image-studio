"""UI section schema: defaults, coerce, and merge with config.py fallbacks."""

from __future__ import annotations

from typing import Any

from zimage.config import (
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_LORA_DIR,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUANTIZE_MODULES,
    DEFAULT_RESOLUTION,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    IMAGE_FORMAT_CHOICES,
    PRECISION_CHOICES,
    QUANTIZE_CHOICES,
    RESOLUTION_PRESETS,
)
from zimage.engine.lora import normalize_lora_dir

UI_SECTION = "ui"

DEVICE_CHOICES = ("auto", "cuda", "cpu")

_PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float8": "fp8",
    "float8dq": "fp8",
    "fp8dq": "fp8",
    "int8wo": "int8",
    "q8": "int8",
    "int8_weight_only": "int8",
}
_IMAGE_FORMAT_ALIASES = {
    "jpg": "jpeg",
    "jpe": "jpeg",
}
UI_PREF_KEYS = (
    "prompt",
    "resolution",
    "steps",
    "batch",
    "output_dir",
    "image_format",
    "seed",
    "random_seed",
    "model_id",
    "device",
    "precision",
    "quantize_modules",
    "cpu_offload",
    "vae_tiling",
    "lora_dir",
    "lora_adapters",
    "lora_weights",
    "guidance",
    "time_shift",
)


def ui_pref_defaults() -> dict[str, Any]:
    device = DEFAULT_DEVICE if DEFAULT_DEVICE in DEVICE_CHOICES else "auto"
    precision = DEFAULT_DTYPE if DEFAULT_DTYPE in PRECISION_CHOICES else "fp8"
    return {
        "prompt": "",
        "resolution": DEFAULT_RESOLUTION,
        "steps": DEFAULT_STEPS,
        "batch": DEFAULT_BATCH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "image_format": DEFAULT_IMAGE_FORMAT,
        "seed": 42,
        "random_seed": True,
        "model_id": DEFAULT_MODEL,
        "device": device,
        "precision": precision,
        "quantize_modules": list(DEFAULT_QUANTIZE_MODULES),
        "cpu_offload": False,
        "vae_tiling": False,
        "lora_dir": normalize_lora_dir(DEFAULT_LORA_DIR),
        "lora_adapters": [],
        "lora_weights": [],
        "guidance": DEFAULT_GUIDANCE,
        "time_shift": DEFAULT_SHIFT,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _as_quantize_modules(value: Any, default: list[str]) -> list[str]:
    names = _as_str_list(value)
    if not names and value is None:
        return list(default)
    allowed = {str(item) for item in QUANTIZE_CHOICES}
    kept = [name for name in names if name in allowed]
    return kept


def _as_lora_weights(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    rows_in: list = []
    if hasattr(value, "values") and not isinstance(value, (list, tuple, dict)):
        try:
            rows_in = list(value.values.tolist())
        except Exception:
            return []
    elif isinstance(value, (list, tuple)):
        rows_in = list(value)
    else:
        return []
    rows: list[list[Any]] = []
    for item in rows_in:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        if item[0] is None:
            continue
        name = str(item[0]).strip()
        if not name:
            continue
        try:
            strength = float(item[1])
        except (TypeError, ValueError):
            continue
        rows.append([name, strength])
    return rows


def coerce_ui_prefs(raw: Any) -> dict[str, Any]:
    """Merge a raw ui section with defaults; drop unknown keys; fix invalid values."""
    defaults = ui_pref_defaults()
    data = raw if isinstance(raw, dict) else {}
    out = dict(defaults)

    if "prompt" in data:
        out["prompt"] = _as_str(data.get("prompt"), "")

    if "resolution" in data:
        resolution = _as_str(data.get("resolution"), defaults["resolution"])
        out["resolution"] = resolution if resolution in RESOLUTION_PRESETS else defaults["resolution"]

    if "steps" in data:
        out["steps"] = _as_int(data.get("steps"), defaults["steps"])

    if "batch" in data:
        out["batch"] = _as_int(data.get("batch"), defaults["batch"])

    if "output_dir" in data:
        out["output_dir"] = _as_str(data.get("output_dir"), defaults["output_dir"])

    if "image_format" in data:
        raw_fmt = _as_str(data.get("image_format"), "").strip().lower()
        if raw_fmt:
            mapped = _IMAGE_FORMAT_ALIASES.get(raw_fmt, raw_fmt)
            out["image_format"] = (
                mapped if mapped in IMAGE_FORMAT_CHOICES else defaults["image_format"]
            )

    if "seed" in data:
        out["seed"] = _as_int(data.get("seed"), defaults["seed"])

    if "random_seed" in data:
        out["random_seed"] = _as_bool(data.get("random_seed"), defaults["random_seed"])

    if "model_id" in data:
        text = _as_str(data.get("model_id"), defaults["model_id"]).strip()
        out["model_id"] = text or defaults["model_id"]

    if "device" in data:
        device = _as_str(data.get("device"), defaults["device"]).strip().lower()
        out["device"] = device if device in DEVICE_CHOICES else defaults["device"]

    if "precision" in data:
        raw_prec = _as_str(data.get("precision"), "").strip().lower()
        if raw_prec:
            mapped = _PRECISION_ALIASES.get(raw_prec, raw_prec)
            out["precision"] = (
                mapped if mapped in PRECISION_CHOICES else defaults["precision"]
            )

    if "quantize_modules" in data:
        out["quantize_modules"] = _as_quantize_modules(
            data.get("quantize_modules"),
            defaults["quantize_modules"],
        )

    if "cpu_offload" in data:
        out["cpu_offload"] = _as_bool(data.get("cpu_offload"), defaults["cpu_offload"])

    if "vae_tiling" in data:
        out["vae_tiling"] = _as_bool(data.get("vae_tiling"), defaults["vae_tiling"])

    if "lora_dir" in data:
        out["lora_dir"] = normalize_lora_dir(data.get("lora_dir"))

    if "lora_adapters" in data:
        out["lora_adapters"] = _as_str_list(data.get("lora_adapters"))

    if "lora_weights" in data:
        out["lora_weights"] = _as_lora_weights(data.get("lora_weights"))

    if "guidance" in data:
        out["guidance"] = _as_float(data.get("guidance"), defaults["guidance"])

    if "time_shift" in data:
        out["time_shift"] = _as_float(data.get("time_shift"), defaults["time_shift"])

    return out


def serialize_ui_prefs(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a prefs dict for writing into the ui section."""
    coerced = coerce_ui_prefs(data)
    return {key: coerced[key] for key in UI_PREF_KEYS}
