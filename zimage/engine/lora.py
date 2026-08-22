"""Scan a LoRA directory and apply adapters on a diffusers pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

LORA_SUFFIXES = {".safetensors", ".pt"}
DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 0.0
MAX_STRENGTH = 2.0
_INNER_DIT_SEGMENT = "._inner_dit."
_INNER_DIT_PREFIX = "_inner_dit."
_INNER_DIT_DOTTED = "inner.dit."
_DIFFUSION_PREFIX = "diffusion_model."
_TRANSFORMER_PREFIX = "transformer."

_applied_key: tuple | None = None
_applied_pipe_id: int | None = None


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    filename: str
    adapter_name: str
    scale: float


def reset_lora_adapters() -> None:
    global _applied_key, _applied_pipe_id
    _applied_key = None
    _applied_pipe_id = None


def normalize_lora_dir(directory: str | None) -> str:
    """Strip quotes, convert backslashes to `/`, and use parent if a LoRA file was pasted."""
    if directory is None:
        return ""
    text = str(directory).strip().strip('"').strip("'").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    path = Path(text)
    if path.suffix.lower() in LORA_SUFFIXES or path.is_file():
        path = path.parent
    if not str(path).strip() or str(path) == ".":
        return ""
    return path.as_posix()


def list_lora_files(directory: str | None) -> list[str]:
    normalized = normalize_lora_dir(directory)
    if not normalized:
        return []
    path = Path(normalized)
    if not path.is_dir():
        return []
    names = [
        child.name
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in LORA_SUFFIXES
    ]
    return sorted(names, key=str.lower)


def parse_lora_specs(directory, names, weights=None) -> tuple[LoraSpec, ...]:
    normalized = normalize_lora_dir(directory)
    available = set(list_lora_files(normalized))
    selected = _as_name_list(names)
    weight_map = weights_map(weights)
    if not normalized:
        return ()
    root = Path(normalized)
    used_names: set[str] = set()
    specs: list[LoraSpec] = []
    for name in selected:
        if name not in available:
            continue
        scale = clamp_strength(weight_map.get(name, DEFAULT_STRENGTH))
        specs.append(
            LoraSpec(
                path=root / name,
                filename=name,
                adapter_name=_adapter_name(name, used_names),
                scale=scale,
            )
        )
    return tuple(specs)


def weights_map(weights) -> dict[str, float]:
    if weights is None:
        return {}
    if isinstance(weights, dict):
        return {
            str(key).strip(): clamp_strength(_as_float(value, DEFAULT_STRENGTH))
            for key, value in weights.items()
            if str(key).strip()
        }
    rows = _dataframe_rows(weights)
    result: dict[str, float] = {}
    for row in rows:
        if not row:
            continue
        name = str(row[0]).strip()
        if not name:
            continue
        scale = DEFAULT_STRENGTH if len(row) < 2 else _as_float(row[1], DEFAULT_STRENGTH)
        result[name] = clamp_strength(scale)
    return result


def clamp_strength(value: float) -> float:
    return max(MIN_STRENGTH, min(MAX_STRENGTH, float(value)))


def status_loras(specs: tuple[LoraSpec, ...] | list[LoraSpec] | None) -> list[dict[str, Any]]:
    return [{"name": spec.filename, "strength": spec.scale} for spec in specs or ()]


def rewrite_lora_inner_dit_keys(state_dict: dict) -> dict:
    """Strip training-wrapper path segments so Diffusers/PEFT match HF Z-Image modules."""
    rewritten: dict = {}
    for key, value in state_dict.items():
        new_key = _rewrite_lora_inner_dit_key(key)
        if new_key in rewritten:
            raise ValueError(f"LoRA key collision after rewrite: {new_key!r}")
        rewritten[new_key] = value
    return rewritten


def sync_lora_adapters(pipe, specs: tuple[LoraSpec, ...] | list[LoraSpec] | None) -> None:
    global _applied_key, _applied_pipe_id

    specs = tuple(specs or ())
    key = tuple((str(spec.path), spec.adapter_name, spec.scale) for spec in specs)
    pipe_id = id(pipe)
    if _applied_key == key and _applied_pipe_id == pipe_id:
        return

    if not specs:
        _unload_loras(pipe)
        _applied_key = key
        _applied_pipe_id = pipe_id
        return

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("Pipeline does not support LoRA adapters.")

    _unload_loras(pipe)
    try:
        names: list[str] = []
        scales: list[float] = []
        for spec in specs:
            state_dict = rewrite_lora_inner_dit_keys(_load_lora_state_dict(spec.path))
            pipe.load_lora_weights(state_dict, adapter_name=spec.adapter_name)
            names.append(spec.adapter_name)
            scales.append(spec.scale)
        pipe.set_adapters(names, adapter_weights=scales)
    except Exception:
        _applied_key = None
        _applied_pipe_id = None
        raise
    _applied_key = key
    _applied_pipe_id = pipe_id


def _load_lora_state_dict(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return load_file(str(path))
    if suffix == ".pt":
        import torch

        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError(f"LoRA .pt file must contain a state dict: {path}")
        return loaded
    raise ValueError(f"Unsupported LoRA file type: {path.suffix}")


def _rewrite_lora_inner_dit_key(key: str) -> str:
    rewritten = key.replace(_INNER_DIT_SEGMENT, ".")
    stripped_leading = False
    if rewritten.startswith(_INNER_DIT_PREFIX):
        rewritten = rewritten[len(_INNER_DIT_PREFIX) :]
        stripped_leading = True
    if _INNER_DIT_DOTTED in rewritten:
        rewritten = rewritten.replace(_INNER_DIT_DOTTED, "")
        if rewritten.startswith("."):
            rewritten = rewritten[1:]
        stripped_leading = True
    if stripped_leading and not rewritten.startswith(_DIFFUSION_PREFIX) and not rewritten.startswith(
        _TRANSFORMER_PREFIX
    ):
        rewritten = f"{_DIFFUSION_PREFIX}{rewritten}"
    return rewritten


def _unload_loras(pipe) -> None:
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
            return
        except Exception:
            pass
    if hasattr(pipe, "disable_lora"):
        try:
            pipe.disable_lora()
        except Exception:
            pass


def _as_name_list(names) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        text = names.strip()
        return [text] if text else []
    return [str(item).strip() for item in names if str(item).strip()]


def _adapter_name(filename: str, used: set[str]) -> str:
    stem = Path(filename).stem
    base = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_") or "lora"
    name = base
    index = 2
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name


def _as_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dataframe_rows(weights) -> list:
    if hasattr(weights, "values") and not isinstance(weights, (list, tuple)):
        try:
            return list(weights.values.tolist())
        except Exception:
            return []
    try:
        return list(weights)
    except TypeError:
        return []
