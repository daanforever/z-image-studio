"""Training-local weight-only quantization for text encoder and sampling DiT.

Does not import the inference engine, modeling, sampling, or the training loop.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

log = logging.getLogger("zimage.training")

_QUANTIZED_PRECISION_ATTR = "_quantized_precision"
_WEIGHT_ONLY_PRECISIONS = frozenset({"fp8"})


def should_quantize_at_load(precision: str, capable: bool) -> bool:
    return str(precision).strip().lower() == "fp8" and bool(capable)


def quantized_precision(module: Any) -> str | None:
    if module is None:
        return None
    marked = getattr(module, _QUANTIZED_PRECISION_ATTR, None)
    if isinstance(marked, str) and marked:
        return marked
    if _has_torchao_weight_subclass(module):
        return "fp8"
    return None


def is_sampling_transformer_quantized(module: Any) -> bool:
    return quantized_precision(module) is not None


def quantize_text_encoder(module: Any, *, precision: str = "fp8") -> None:
    _quantize_weight_only(module, precision=precision, label="text encoder")


def quantize_sampling_transformer(module: Any, *, precision: str = "fp8") -> None:
    _quantize_weight_only(
        module,
        precision=precision,
        label="sampling transformer",
        attach_peft_requantizer=True,
    )


def _quantize_weight_only(
    module: Any,
    *,
    precision: str,
    label: str,
    attach_peft_requantizer: bool = False,
) -> None:
    requested = _require_weight_only_precision(precision)
    if quantized_precision(module) == requested:
        return
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    config = Float8WeightOnlyConfig()
    quantize_(module, config)
    if attach_peft_requantizer:
        _attach_peft_torchao_requantizer(module, config)
    setattr(module, _QUANTIZED_PRECISION_ATTR, requested)
    log.info("quantize %s precision=%s", label, requested)


def _require_weight_only_precision(precision: str) -> str:
    requested = str(precision).strip().lower()
    if requested not in _WEIGHT_ONLY_PRECISIONS:
        raise ValueError(f"unsupported quantization precision {precision!r}")
    return requested


def _has_torchao_weight_subclass(module: Any) -> bool:
    try:
        from torchao.utils import TorchAOBaseTensor
    except ImportError:
        return False
    modules = getattr(module, "modules", None)
    if not callable(modules):
        weight = getattr(module, "weight", None)
        return isinstance(weight, TorchAOBaseTensor)
    for child in modules():
        weight = getattr(child, "weight", None)
        if isinstance(weight, TorchAOBaseTensor):
            return True
    return False


def _attach_peft_torchao_requantizer(transformer: Any, quant_type: Any) -> None:
    """Expose the requantizer PEFT needs after post-load ``quantize_``.

    PEFT wraps TorchAO tensors in ``TorchaoLoraLinear`` and looks up
    ``model.hf_quantizer.quantization_config.get_apply_tensor_subclass`` so
    ``merge()`` / ``unmerge()`` can re-quantize. That attribute is normally
    set by ``from_pretrained(..., quantization_config=TorchAoConfig(...))``,
    which crashes on Windows Blackwell. Training quantizes after load, so this
    attaches the same PEFT contract without going through that loader.
    """

    if _peft_torchao_requantizer(transformer) is not None:
        return
    transformer.hf_quantizer = SimpleNamespace(
        quantization_config=SimpleNamespace(
            get_apply_tensor_subclass=lambda: quant_type
        )
    )


def _peft_torchao_requantizer(transformer: Any) -> Any | None:
    getter = getattr(
        getattr(getattr(transformer, "hf_quantizer", None), "quantization_config", None),
        "get_apply_tensor_subclass",
        None,
    )
    return getter if callable(getter) else None
