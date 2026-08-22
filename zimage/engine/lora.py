"""Scan a LoRA directory and apply adapters on a diffusers pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from zimage.paths import normalize_dir

LORA_SUFFIXES = {".safetensors", ".pt"}
DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 0.0
MAX_STRENGTH = 2.0
_INNER_DIT_SEGMENT = "._inner_dit."
_INNER_DIT_PREFIX = "_inner_dit."
_INNER_DIT_DOTTED = "inner.dit."
# Kohya encodes dots as underscores, so training wrappers become *_inner_dit_* / *inner_dit_*.
_INNER_DIT_UNDERSCORE = re.compile(r"_?_inner_dit_|inner_dit_")
_DIFFUSION_PREFIX = "diffusion_model."
_TRANSFORMER_PREFIX = "transformer."
_LORA_UNET_PREFIX = "lora_unet_"
_LORA_MODULE_NAMES = ("transformer", "dit", "unet")

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
    return normalize_dir(directory, file_suffixes=LORA_SUFFIXES)


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


def lora_identity_key(specs: tuple[LoraSpec, ...] | list[LoraSpec] | None) -> tuple:
    """Stable cache identity: path, adapter name, and strength per LoRA."""
    return tuple((str(spec.path), spec.adapter_name, spec.scale) for spec in specs or ())


def sync_lora_adapters(
    pipe,
    specs: tuple[LoraSpec, ...] | list[LoraSpec] | None,
    device: str | None = None,
) -> None:
    """Load each LoRA, fuse it into base weights, then unload adapter matrices.

    Fusing before quantization keeps VRAM at the base-model footprint. Once fused,
    changing adapters or strength requires a fresh pipeline load (no unfuse).
    When ``device`` is set, the DiT is moved there first so merge uses GPU kernels.
    """
    global _applied_key, _applied_pipe_id

    specs = tuple(specs or ())
    key = lora_identity_key(specs)
    pipe_id = id(pipe)
    if _applied_key == key and _applied_pipe_id == pipe_id:
        return

    if not specs:
        # Fresh pipe: no PEFT adapters to remove. Rollback to base is a new load.
        _applied_key = key
        _applied_pipe_id = pipe_id
        return

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("Pipeline does not support LoRA adapters.")

    try:
        if device is not None:
            _move_lora_modules(pipe, device)
        # One adapter at a time so peak memory never holds several A/B sets.
        for spec in specs:
            state_dict = rewrite_lora_inner_dit_keys(_load_lora_state_dict(spec.path))
            pipe.load_lora_weights(state_dict, adapter_name=spec.adapter_name)
            if hasattr(pipe, "set_adapters"):
                # Activate at unit scale. fuse_lora(lora_scale=spec.scale) applies
                # strength once; passing the same value to both squares it.
                pipe.set_adapters([spec.adapter_name], adapter_weights=[1.0])
            _fuse_and_unload(pipe, adapter_name=spec.adapter_name, scale=spec.scale)
    except Exception:
        _applied_key = None
        _applied_pipe_id = None
        raise
    _applied_key = key
    _applied_pipe_id = pipe_id


def _move_lora_modules(pipe, device: str) -> None:
    for name in _LORA_MODULE_NAMES:
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "to"):
            module.to(device)
            return


def _fuse_and_unload(pipe, *, adapter_name: str, scale: float) -> None:
    """Merge active LoRA into base weights and drop adapter tensors."""
    if hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=scale)
        except TypeError:
            try:
                pipe.fuse_lora(lora_scale=scale)
            except TypeError:
                pipe.fuse_lora()
        _unload_loras(pipe)
        return

    _merge_and_unload_modules(pipe)
    _unload_loras(pipe)


def _merge_and_unload_modules(pipe) -> None:
    """PEFT fallback when the pipeline has no fuse_lora()."""
    for attr in ("transformer", "dit", "unet"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        if hasattr(module, "merge_and_unload"):
            merged = module.merge_and_unload()
            if merged is not None and merged is not module:
                setattr(pipe, attr, merged)
            return
        peft = getattr(module, "base_model", None)
        if peft is not None and hasattr(peft, "merge_and_unload"):
            merged = peft.merge_and_unload()
            if merged is not None:
                setattr(pipe, attr, merged)
            return
    raise RuntimeError(
        "Pipeline cannot fuse LoRA weights (no fuse_lora / merge_and_unload)."
    )


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
    stripped = False
    if rewritten.startswith(_INNER_DIT_PREFIX):
        rewritten = rewritten[len(_INNER_DIT_PREFIX) :]
        stripped = True
    if _INNER_DIT_DOTTED in rewritten:
        rewritten = rewritten.replace(_INNER_DIT_DOTTED, "")
        stripped = True
    if _INNER_DIT_UNDERSCORE.search(rewritten):
        rewritten = _INNER_DIT_UNDERSCORE.sub("_", rewritten)
        stripped = True
    if stripped:
        while ".." in rewritten:
            rewritten = rewritten.replace("..", ".")
        while "__" in rewritten:
            rewritten = rewritten.replace("__", "_")
        rewritten = rewritten.lstrip("._")
    if (
        stripped
        and not rewritten.startswith(_DIFFUSION_PREFIX)
        and not rewritten.startswith(_TRANSFORMER_PREFIX)
        and not rewritten.startswith(_LORA_UNET_PREFIX)
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
