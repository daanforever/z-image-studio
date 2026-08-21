"""torchao quantization for the DiT transformer (int8 weight-only, fp8)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

INT8_PRECISION_NAMES = frozenset({"int8", "int8wo", "q8", "int8_weight_only"})
FP8_PRECISION_NAMES = frozenset({"fp8", "float8", "float8dq", "fp8dq"})
FP8_MIN_CAPABILITY = (8, 9)


def is_int8_precision(name: str | None) -> bool:
    return (name or "").strip().lower() in INT8_PRECISION_NAMES


def is_fp8_precision(name: str | None) -> bool:
    return (name or "").strip().lower() in FP8_PRECISION_NAMES


def is_quantized_precision(name: str | None) -> bool:
    return is_int8_precision(name) or is_fp8_precision(name)


def try_import_torchao() -> ModuleType | None:
    try:
        import torchao
    except ImportError:
        return None
    return torchao


def require_torchao() -> None:
    if try_import_torchao() is None:
        raise RuntimeError(
            "Quantized precision requires torchao. Install it with: pip install torchao"
        )


def require_fp8_device(device: str) -> None:
    if device != "cuda":
        raise RuntimeError(
            "fp8 requires CUDA with compute capability 8.9+ (RTX 4090, RTX 50xx). "
            "Use int8 or bfloat16 on this device."
        )
    from zimage.engine.runtime import try_import_torch

    torch = try_import_torch()
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "fp8 requires a CUDA GPU with compute capability 8.9+. "
            "Use int8 or bfloat16 instead."
        )
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < FP8_MIN_CAPABILITY:
        name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"fp8 needs compute capability 8.9+ (Ada/Blackwell). "
            f"{name} is {major}.{minor}. Use int8 instead."
        )


def _quantization_target(pipe: Any) -> tuple[str, Any]:
    for name in ("transformer", "dit", "unet"):
        module = getattr(pipe, name, None)
        if module is not None:
            return name, module
    raise RuntimeError(
        "Cannot apply quantization: pipeline has no transformer/DiT module."
    )


def _int8_weight_only_scheme() -> Any:
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    except Exception:  # noqa: BLE001 — older torchao uses a factory function
        from torchao.quantization import int8_weight_only

        return int8_weight_only()


def _fp8_scheme() -> Any:
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        return Float8DynamicActivationFloat8WeightConfig()
    except ImportError:
        pass
    try:
        from torchao.quantization import float8_dynamic_activation_float8_weight

        return float8_dynamic_activation_float8_weight()
    except ImportError:
        pass
    try:
        from torchao.quantization import Float8WeightOnlyConfig

        return Float8WeightOnlyConfig()
    except ImportError:
        pass
    from torchao.quantization import float8_weight_only

    return float8_weight_only()


def _scheme_for(dtype_name: str) -> Any:
    if is_fp8_precision(dtype_name):
        return _fp8_scheme()
    if is_int8_precision(dtype_name):
        return _int8_weight_only_scheme()
    raise RuntimeError(f"No torchao scheme for precision {dtype_name!r}.")


def apply_quantization(pipe: Any, dtype_name: str) -> str:
    """Quantize Linear layers of the DiT in-place. Returns the module name."""
    require_torchao()
    name, module = _quantization_target(pipe)
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "Quantized precision requires torchao.quantization.quantize_. "
            "Install it with: pip install torchao"
        ) from exc

    label = "fp8" if is_fp8_precision(dtype_name) else "int8"
    try:
        quantize_(module, _scheme_for(dtype_name))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to apply torchao {label} quantization to `{name}`: {exc}"
        ) from exc
    return name


def apply_int8_quantization(pipe: Any) -> str:
    return apply_quantization(pipe, "int8")
