"""torchao quantization for the DiT and text encoder (int8 weight-only, fp8)."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

from zimage.config import canonical_precision

log = logging.getLogger("zimage")

FP8_MIN_CAPABILITY = (8, 9)


def is_int8_precision(name: str | None) -> bool:
    return canonical_precision(name) == "int8"


def is_fp8_precision(name: str | None) -> bool:
    return canonical_precision(name) == "fp8"


def is_quantized_precision(name: str | None) -> bool:
    return canonical_precision(name) in {"int8", "fp8"}


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


_DIT_MODULE_NAMES = ("transformer", "dit", "unet")
_TEXT_ENCODER_MODULE_NAMES = ("text_encoder", "text_encoder_2", "text_encoder_3")


def should_quantize(
    dtype_name: str | None,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
) -> bool:
    return is_quantized_precision(dtype_name) and bool(
        quantize_transformer or quantize_text_encoder
    )


def _quantization_targets(
    pipe: Any,
    *,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
) -> list[tuple[str, Any]]:
    targets: list[tuple[str, Any]] = []
    if quantize_transformer:
        for name in _DIT_MODULE_NAMES:
            module = getattr(pipe, name, None)
            if module is not None:
                targets.append((name, module))
                break
        else:
            raise RuntimeError(
                "Cannot apply quantization: pipeline has no transformer/DiT module."
            )
    if quantize_text_encoder:
        found_encoder = False
        for name in _TEXT_ENCODER_MODULE_NAMES:
            module = getattr(pipe, name, None)
            if module is not None:
                targets.append((name, module))
                found_encoder = True
        if not found_encoder and not quantize_transformer:
            raise RuntimeError(
                "Cannot apply quantization: pipeline has no text encoder."
            )
    if not targets:
        raise RuntimeError("Cannot apply quantization: no modules selected.")
    return targets


def _int8_weight_only_scheme() -> Any:
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    except ImportError:
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

        log.warning("torchao float8 dynamic config is unavailable; using weight-only fp8.")
        return Float8WeightOnlyConfig()
    except ImportError:
        pass
    from torchao.quantization import float8_weight_only

    log.warning("torchao float8 dynamic config is unavailable; using weight-only fp8.")
    return float8_weight_only()


def _scheme_for(dtype_name: str) -> Any:
    precision = canonical_precision(dtype_name)
    if precision == "fp8":
        return _fp8_scheme()
    if precision == "int8":
        return _int8_weight_only_scheme()
    raise RuntimeError(f"No torchao scheme for precision {dtype_name!r}.")


def _scheme_label(scheme: Any) -> str:
    name = getattr(scheme, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(scheme).__name__


def apply_quantization(
    pipe: Any,
    dtype_name: str,
    quantize_transformer: bool = True,
    quantize_text_encoder: bool = True,
) -> str:
    """Quantize Linear layers of selected modules in-place.

    Returns a comma-separated list of quantized module names.
    """
    require_torchao()
    targets = _quantization_targets(
        pipe,
        quantize_transformer=quantize_transformer,
        quantize_text_encoder=quantize_text_encoder,
    )
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "Quantized precision requires torchao.quantization.quantize_. "
            "Install it with: pip install torchao"
        ) from exc

    precision = canonical_precision(dtype_name)
    scheme = _scheme_for(precision)
    applied: list[str] = []
    for name, module in targets:
        if hasattr(module, "eval"):
            module.eval()
        try:
            quantize_(module, scheme)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to apply torchao {precision} quantization to `{name}`: {exc}"
            ) from exc
        log.info("Applied torchao %s to %s (%s)", precision, name, _scheme_label(scheme))
        applied.append(name)
    return ", ".join(applied)


def apply_int8_quantization(pipe: Any) -> str:
    return apply_quantization(pipe, "int8")
