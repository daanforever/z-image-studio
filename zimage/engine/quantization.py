"""torchao int8 weight-only quantization for the DiT transformer."""

from __future__ import annotations

from types import ModuleType
from typing import Any

INT8_PRECISION_NAMES = frozenset({"int8", "int8wo", "q8", "int8_weight_only"})


def is_int8_precision(name: str | None) -> bool:
    return (name or "").strip().lower() in INT8_PRECISION_NAMES


def try_import_torchao() -> ModuleType | None:
    try:
        import torchao
    except ImportError:
        return None
    return torchao


def require_torchao() -> None:
    if try_import_torchao() is None:
        raise RuntimeError(
            "int8 precision requires torchao. Install it with: pip install torchao"
        )


def _quantization_target(pipe: Any) -> tuple[str, Any]:
    for name in ("transformer", "dit", "unet"):
        module = getattr(pipe, name, None)
        if module is not None:
            return name, module
    raise RuntimeError(
        "Cannot apply int8 quantization: pipeline has no transformer/DiT module."
    )


def _int8_weight_only_scheme() -> Any:
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    except Exception:  # noqa: BLE001 — older torchao uses a factory function
        from torchao.quantization import int8_weight_only

        return int8_weight_only()


def apply_int8_quantization(pipe: Any) -> str:
    """Quantize Linear weights of the DiT in-place. Returns the module name."""
    require_torchao()
    name, module = _quantization_target(pipe)
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "int8 precision requires torchao.quantization.quantize_. "
            "Install it with: pip install torchao"
        ) from exc

    try:
        quantize_(module, _int8_weight_only_scheme())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to apply torchao int8 weight-only quantization to `{name}`: {exc}"
        ) from exc
    return name
