"""Z-Image training model loading, encoding, and resource lifecycle.

This module intentionally stops before optimizer or training-loop construction.
The small dependency-injection surface keeps its behavior unit-testable without
downloading production weights.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from zimage.training.hub_compat import install_hf_local_dir_symlinks_compat
from zimage.training.schema import KNOWN_TURBO_SOURCE

log = logging.getLogger("zimage.training")


class ModelConfigurationError(ValueError):
    """A training model source or setup is invalid."""


class SamplingCompatibilityError(ModelConfigurationError):
    """The sampling transformer cannot safely consume the training adapter."""


@dataclass(frozen=True)
class ModelSource:
    """A Hugging Face model ID or local Diffusers model root."""

    path: str
    revision: str | None = None


@dataclass(frozen=True)
class ModelSources:
    """Resolved main and sampling model sources."""

    main: ModelSource
    sampling: ModelSource

    @property
    def has_separate_sampler(self) -> bool:
        return self.sampling != self.main


@dataclass(frozen=True)
class ComponentLoaders:
    """Injectable ``from_pretrained`` owners used by component loading."""

    vae: Any
    tokenizer: Any
    text_encoder: Any
    transformer: Any
    scheduler: Any

    @classmethod
    def defaults(cls) -> "ComponentLoaders":
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImageTransformer2DModel,
        )
        from transformers import Qwen2Tokenizer, Qwen3Model

        return cls(
            vae=AutoencoderKL,
            tokenizer=Qwen2Tokenizer,
            text_encoder=Qwen3Model,
            transformer=ZImageTransformer2DModel,
            scheduler=FlowMatchEulerDiscreteScheduler,
        )


@dataclass
class TrainingModelComponents:
    """Loaded components before Accelerate preparation.

    Sampling always uses independent transformer and scheduler instances,
    including when their source is the main model root.
    """

    sources: ModelSources
    vae: Any
    tokenizer: Any | None
    text_encoder: Any | None
    training_scheduler: Any
    main_transformer: Any
    sampling_transformer: Any
    sampling_scheduler: Any


@dataclass(frozen=True)
class MainTransformerSetup:
    """Result of preparing the main transformer for later Accelerate use."""

    transformer: Any
    requested_precision: str
    effective_precision: str
    fp8_enabled: bool
    gradient_checkpointing_enabled: bool
    adapter_name: str


def resolve_model_sources(job: Mapping[str, Any]) -> ModelSources:
    """Resolve source blocks, making an omitted sampler fall back to main."""

    model = job.get("model")
    if not isinstance(model, Mapping):
        model = {}
    main = _parse_source(model.get("main_transformer"), "model.main_transformer")
    reject_turbo_main_source(main)
    sampling_raw = model.get("sampling_transformer")
    sampling = (
        main
        if sampling_raw is None
        else _parse_source(sampling_raw, "model.sampling_transformer")
    )
    return ModelSources(main=main, sampling=sampling)


def reject_turbo_main_source(source: ModelSource | str) -> None:
    """Reject the distilled Turbo checkpoint as a training base."""

    path = source.path if isinstance(source, ModelSource) else str(source)
    identities = {_normalise_hf_identity(path)}
    identities.update(_local_source_identities(path))
    turbo = _normalise_hf_identity(KNOWN_TURBO_SOURCE)
    if turbo in identities:
        raise ModelConfigurationError(
            f"{KNOWN_TURBO_SOURCE} cannot be used as main_transformer; "
            "it is supported only as sampling_transformer"
        )


def load_training_components(
    job: Mapping[str, Any],
    *,
    loaders: ComponentLoaders | None = None,
    disable_mmap: bool = True,
) -> TrainingModelComponents:
    """Load train/cache components and an optional distinct preview sampler."""

    install_hf_local_dir_symlinks_compat()
    loaders = loaders or ComponentLoaders.defaults()
    sources = resolve_model_sources(job)

    vae = _load_model_component(
        loaders.vae,
        sources.main,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        disable_mmap=disable_mmap,
        what="vae",
    )
    tokenizer = _load_component(
        loaders.tokenizer,
        sources.main,
        subfolder="tokenizer",
        what="tokenizer",
    )
    text_encoder = _load_model_component(
        loaders.text_encoder,
        sources.main,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        disable_mmap=disable_mmap,
        what="text encoder",
    )
    training_scheduler = _load_component(
        loaders.scheduler,
        sources.main,
        subfolder="scheduler",
        what="training scheduler",
    )
    main_transformer = _load_model_component(
        loaders.transformer,
        sources.main,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        disable_mmap=disable_mmap,
        what="main transformer",
    )

    # The sampler must remain pristine while the main instance is frozen,
    # converted for FP8 training, and mutated by PEFT. Scheduler state is also
    # mutable, so it receives an independent instance even when sources match.
    sampling_transformer = _load_model_component(
        loaders.transformer,
        sources.sampling,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        disable_mmap=disable_mmap,
        what="sampling transformer",
    )
    sampling_scheduler = _load_component(
        loaders.scheduler,
        sources.sampling,
        subfolder="scheduler",
        what="sampling scheduler",
    )

    for frozen_component in (vae, text_encoder):
        if hasattr(frozen_component, "requires_grad_"):
            frozen_component.requires_grad_(False)

    return TrainingModelComponents(
        sources=sources,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        training_scheduler=training_scheduler,
        main_transformer=main_transformer,
        sampling_transformer=sampling_transformer,
        sampling_scheduler=sampling_scheduler,
    )


def official_zimage_fp8_filter(module: torch.nn.Module, fqn: str) -> bool:
    """Select modules using the official Diffusers Z-Image FP8 recipe."""

    if fqn == "proj_out" or fqn.endswith(".proj_out"):
        return False
    if isinstance(module, torch.nn.Linear):
        return module.in_features % 16 == 0 and module.out_features % 16 == 0
    return True


def setup_main_transformer(
    transformer: Any,
    *,
    precision: str,
    fp8_capable: bool,
    lora: Mapping[str, Any],
    gradient_checkpointing: bool,
    device: str | torch.device = "cuda",
    adapter_name: str = "default",
    convert_to_float8_training: Callable[..., Any] | None = None,
    float8_config_factory: Callable[..., Any] | None = None,
    lora_config_factory: Callable[..., Any] | None = None,
) -> MainTransformerSetup:
    """Freeze, place, optionally FP8-convert, add the PEFT adapter, then place it.

    Official order: freeze → device → FP8 conversion/filter → gradient
    checkpointing → add adapter → place LoRA (device + BF16) → validate.
    Accelerate preparation stays in the optimizer-loop layer.
    """

    requested = str(precision).strip().lower()
    if requested not in {"fp8", "bf16"}:
        raise ModelConfigurationError(
            f"unsupported main-transformer precision {precision!r}"
        )

    transformer.requires_grad_(False)
    transformer.to(device)

    fp8_enabled = requested == "fp8" and bool(fp8_capable)
    if fp8_enabled:
        log.info("quantize main transformer precision=fp8")
        # TorchAO Float8Linear.forward still calls the deprecated
        # torch.get_autocast_gpu_dtype(); swap in the current API first.
        _patch_torchao_float8_linear_autocast()
        if convert_to_float8_training is None or float8_config_factory is None:
            from torchao.float8 import (
                Float8LinearConfig,
                convert_to_float8_training as torchao_convert,
            )

            convert_to_float8_training = (
                convert_to_float8_training or torchao_convert
            )
            float8_config_factory = float8_config_factory or Float8LinearConfig
        config = float8_config_factory(pad_inner_dim=True)
        convert_to_float8_training(
            transformer,
            module_filter_fn=official_zimage_fp8_filter,
            config=config,
        )

    checkpointing_enabled = bool(gradient_checkpointing)
    if checkpointing_enabled:
        enable_checkpointing = getattr(
            transformer, "enable_gradient_checkpointing", None
        )
        if not callable(enable_checkpointing):
            raise ModelConfigurationError(
                "gradient checkpointing was requested but the main transformer "
                "does not support enable_gradient_checkpointing()"
            )
        log.info("gradient checkpointing enabled")
        enable_checkpointing()

    if lora_config_factory is None:
        from peft import LoraConfig

        lora_config_factory = LoraConfig
    lora_config = lora_config_factory(
        r=int(lora["rank"]),
        lora_alpha=lora["alpha"],
        lora_dropout=float(lora.get("dropout", 0.0)),
        init_lora_weights="gaussian",
        target_modules=list(lora["targets"]),
    )
    log.info(
        "adding lora adapter rank=%s alpha=%s",
        lora["rank"],
        lora["alpha"],
    )
    transformer.add_adapter(lora_config, adapter_name=adapter_name)
    _place_trainable_adapter(transformer, adapter_name, device)
    _validate_trainable_adapter_contract(transformer, adapter_name, device)

    return MainTransformerSetup(
        transformer=transformer,
        requested_precision=requested,
        effective_precision="fp8" if fp8_enabled else "bf16",
        fp8_enabled=fp8_enabled,
        gradient_checkpointing_enabled=checkpointing_enabled,
        adapter_name=adapter_name,
    )


def encode_prompt(
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    *,
    max_sequence_length: int = 512,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Encode one prompt exactly as the official Z-Image pipeline does."""

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    text_inputs = tokenizer(
        [chat_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = _field(text_inputs, "input_ids")
    attention_mask = _field(text_inputs, "attention_mask").bool()
    target_device = device if device is not None else _module_device(text_encoder)
    input_ids = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden_states = _field(output, "hidden_states")[-2]
    return hidden_states[0][attention_mask[0]].detach().to(dtype=torch.bfloat16)


@dataclass
class ModelBackedCacheEncoder:
    """Lightweight adapter for the structural training-cache encoder contract."""

    vae: Any
    tokenizer: Any | None
    text_encoder: Any | None
    device: str | torch.device | None = None

    def encode_image(self, image: Any) -> Any:
        """Prepare VAE input and return the raw latent distribution container."""

        pixels = prepare_vae_input(image)
        target_device = (
            self.device if self.device is not None else _module_device(self.vae)
        )
        vae_dtype = getattr(self.vae, "dtype", torch.bfloat16)
        pixels = pixels.to(device=target_device, dtype=vae_dtype)
        with torch.no_grad():
            return self.vae.encode(pixels)

    def encode_prompt(
        self,
        caption: str,
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Encode one caption to unpadded Qwen hidden states."""

        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError(
                "text encoder and tokenizer were released; this cache encoder "
                "does not reload them"
            )
        return encode_prompt(
            self.tokenizer,
            self.text_encoder,
            caption,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )

    def release_text_resources(self) -> None:
        """Remove this adapter's references to tokenizer and text encoder."""

        self.tokenizer = None
        self.text_encoder = None


def prepare_vae_input(image: Any) -> torch.Tensor:
    """Convert an RGB image to batched CHW float input in ``[-1, 1]``."""

    if isinstance(image, torch.Tensor):
        pixels = image.detach()
    else:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - Pillow requires NumPy
            raise RuntimeError("NumPy is required for non-tensor images") from exc
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        pixels = torch.as_tensor(np.asarray(image).copy())

    if pixels.ndim == 3 and pixels.shape[-1] == 3:
        pixels = pixels.permute(2, 0, 1)
    elif pixels.ndim == 4 and pixels.shape[-1] == 3:
        pixels = pixels.permute(0, 3, 1, 2)
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ModelConfigurationError(
            "VAE input must be an RGB image with exactly three channels"
        )

    original_dtype = pixels.dtype
    if original_dtype == torch.uint8:
        return pixels.to(dtype=torch.float32) / 127.5 - 1.0
    if not pixels.is_floating_point():
        raise ModelConfigurationError(
            "VAE tensor input must use uint8 or a floating-point dtype"
        )

    pixels = pixels.to(dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise ModelConfigurationError("VAE tensor input must contain finite values")
    minimum = float(pixels.min())
    maximum = float(pixels.max())
    if minimum < -1.0 or maximum > 1.0:
        raise ModelConfigurationError(
            "floating-point VAE input must be in [0, 1] or [-1, 1]"
        )
    if minimum >= 0.0:
        return pixels * 2.0 - 1.0
    return pixels


def encode_vae_latent(
    vae: Any,
    image: Any,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Deterministically encode one RGB image to a scaled BF16 latent."""

    pixels = prepare_vae_input(image)
    target_device = device if device is not None else _module_device(vae)
    vae_dtype = getattr(vae, "dtype", torch.bfloat16)
    pixels = pixels.to(device=target_device, dtype=vae_dtype)
    with torch.no_grad():
        encoded = vae.encode(pixels)
        latent = _field(encoded, "latent_dist").mode()
    shift = float(getattr(vae.config, "shift_factor"))
    scale = float(getattr(vae.config, "scaling_factor"))
    latent = (latent - shift) * scale
    if latent.shape[0] != 1:
        raise ModelConfigurationError(
            "encode_vae_latent expects exactly one input image"
        )
    return latent[0].detach().to(dtype=torch.bfloat16)


def validate_sampling_topology(
    main_transformer: Any,
    sampling_transformer: Any,
    target_modules: Sequence[str],
    lora_state_dict: Mapping[str, Any] | None = None,
) -> None:
    """Fail before sampling if architecture, targets, or adapter shapes differ."""

    main_arch = _architecture_name(main_transformer)
    sampling_arch = _architecture_name(sampling_transformer)
    if main_arch != sampling_arch:
        raise SamplingCompatibilityError(
            "sampling transformer architecture mismatch: "
            f"main={main_arch!r}, sampling={sampling_arch!r}"
        )

    main_config = _normalise_config(getattr(main_transformer, "config", None))
    sampling_config = _normalise_config(
        getattr(sampling_transformer, "config", None)
    )
    if main_config != sampling_config:
        message = (
            "sampling transformer config is incompatible with the main transformer"
        )
        if isinstance(main_config, Mapping) and isinstance(
            sampling_config, Mapping
        ):
            differing = sorted(
                key
                for key in set(main_config) | set(sampling_config)
                if key not in main_config
                or key not in sampling_config
                or main_config[key] != sampling_config[key]
            )
            if differing:
                message = f"{message}: {', '.join(differing)}"
        raise SamplingCompatibilityError(message)

    main_modules = dict(main_transformer.named_modules())
    sampling_modules = dict(sampling_transformer.named_modules())
    if not target_modules:
        raise SamplingCompatibilityError("no LoRA target modules were configured")

    matched_names: set[str] = set()
    for target in target_modules:
        main_matches = _matching_module_names(main_modules, target)
        sampling_matches = _matching_module_names(sampling_modules, target)
        if not main_matches:
            raise SamplingCompatibilityError(
                f"LoRA target {target!r} is absent from the main transformer"
            )
        if not sampling_matches:
            raise SamplingCompatibilityError(
                f"LoRA target {target!r} is absent from the sampling transformer"
            )
        if main_matches != sampling_matches:
            raise SamplingCompatibilityError(
                f"LoRA target {target!r} resolves to different module names "
                "between main and sampling transformers"
            )
        matched_names.update(main_matches)

    for name in sorted(matched_names):
        main_shape = _module_weight_shape(main_modules[name])
        sampling_shape = _module_weight_shape(sampling_modules[name])
        if main_shape != sampling_shape:
            raise SamplingCompatibilityError(
                f"LoRA target {name!r} has incompatible base weight shapes: "
                f"main={main_shape}, sampling={sampling_shape}"
            )

    if lora_state_dict is not None:
        _validate_lora_tensor_shapes(
            sampling_modules, matched_names, lora_state_dict
        )


class TrainingModelLifecycle:
    """Explicit encoder/cache lifecycle with no implicit component reload."""

    def __init__(self, components: TrainingModelComponents) -> None:
        self.components = components
        self._cache_encoder: ModelBackedCacheEncoder | None = None

    @property
    def text_resources_loaded(self) -> bool:
        return (
            self.components.tokenizer is not None
            and self.components.text_encoder is not None
        )

    def prepare_dataset_embeddings(
        self,
        prompts: Iterable[str],
        *,
        max_sequence_length: int,
        consume: Callable[[str, torch.Tensor], None] | None = None,
    ) -> list[torch.Tensor]:
        """Encode dataset captions and optionally hand each result to a cache."""

        tokenizer, encoder = self._require_text_resources()
        results: list[torch.Tensor] = []
        for prompt in prompts:
            embedding = encode_prompt(
                tokenizer,
                encoder,
                prompt,
                max_sequence_length=max_sequence_length,
            )
            if consume is not None:
                consume(prompt, embedding)
            results.append(embedding)
        return results

    def prepare_preview_prompt_embeddings(
        self,
        prompts: Iterable[str],
        *,
        max_sequence_length: int,
    ) -> dict[str, torch.Tensor]:
        """Encode and retain preview prompts before releasing Qwen resources."""

        tokenizer, encoder = self._require_text_resources()
        return {
            prompt: encode_prompt(
                tokenizer,
                encoder,
                prompt,
                max_sequence_length=max_sequence_length,
            )
            for prompt in prompts
        }

    def cache_encoder(self) -> ModelBackedCacheEncoder:
        """Build the explicit adapter injected into the cache preparation layer."""

        tokenizer, encoder = self._require_text_resources()
        if self._cache_encoder is None:
            self._cache_encoder = ModelBackedCacheEncoder(
                vae=self.components.vae,
                tokenizer=tokenizer,
                text_encoder=encoder,
            )
        return self._cache_encoder

    def place_cache_modules(self, device: str | torch.device) -> None:
        """Move VAE and text encoder onto ``device`` for cache encoding.

        Updates ``ModelBackedCacheEncoder.device`` when the adapter already
        exists. Does not move the training or sampling transformer, does not
        quantize, and does not call ``tokenizer.to``.
        """

        for module in (self.components.vae, self.components.text_encoder):
            if module is not None and hasattr(module, "to"):
                module.to(device)
        if self._cache_encoder is not None:
            self._cache_encoder.device = device

    def park_cache_modules(self) -> None:
        """Move VAE and text encoder to CPU and reclaim accelerator memory.

        Idempotent. Does not drop tokenizer or text-encoder references;
        call ``release_text_resources`` to unload them.
        """

        for module in (self.components.vae, self.components.text_encoder):
            if module is not None and hasattr(module, "to"):
                module.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_sampler(
        self,
        target_modules: Sequence[str],
        lora_state_dict: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate adapter topology explicitly before invoking a sampler."""

        validate_sampling_topology(
            self.components.main_transformer,
            self.components.sampling_transformer,
            target_modules,
            lora_state_dict,
        )

    def release_text_resources(self) -> None:
        """Drop all tokenizer/Qwen references and reclaim accelerator memory."""

        if self.text_resources_loaded:
            log.info("releasing text encoder")
        encoder = self.components.text_encoder
        if encoder is not None and hasattr(encoder, "to"):
            encoder.to("cpu")
        if self._cache_encoder is not None:
            self._cache_encoder.release_text_resources()
        self.components.text_encoder = None
        self.components.tokenizer = None
        self._cache_encoder = None
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_text_resources_on_cpu(
        self,
        *,
        loaders: ComponentLoaders | None = None,
        disable_mmap: bool = True,
    ) -> tuple[Any, Any]:
        """Reload tokenizer + Qwen on CPU after ``release_text_resources``.

        Used only by the loop's serial mid-run prompt handoff. Callers must
        invoke ``release_text_resources`` again after encoding. Never moves
        Qwen onto CUDA.
        """

        if self.text_resources_loaded:
            raise RuntimeError(
                "text encoder and tokenizer are already loaded; "
                "release them before calling reload_text_resources_on_cpu"
            )
        loaders = loaders or ComponentLoaders.defaults()
        source = self.components.sources.main
        tokenizer = _load_component(
            loaders.tokenizer,
            source,
            subfolder="tokenizer",
            what="tokenizer",
        )
        text_encoder = _load_model_component(
            loaders.text_encoder,
            source,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            disable_mmap=disable_mmap,
            what="text encoder",
        )
        if hasattr(text_encoder, "to"):
            text_encoder.to("cpu")
        self.components.tokenizer = tokenizer
        self.components.text_encoder = text_encoder
        return tokenizer, text_encoder

    def _require_text_resources(self) -> tuple[Any, Any]:
        if not self.text_resources_loaded:
            raise RuntimeError(
                "text encoder and tokenizer were released; this lifecycle "
                "does not reload them"
            )
        return self.components.tokenizer, self.components.text_encoder


def _current_cuda_autocast_dtype() -> torch.dtype:
    """Return the CUDA autocast dtype using the current PyTorch API."""

    getter = getattr(torch, "get_autocast_dtype", None)
    if callable(getter):
        return getter("cuda")
    return torch.get_autocast_gpu_dtype()


def _patch_torchao_float8_linear_autocast() -> None:
    """Make TorchAO ``Float8Linear.forward`` use ``torch.get_autocast_dtype``.

    Upstream still calls ``torch.get_autocast_gpu_dtype()`` (pytorch/ao#1522,
    unmerged PR #1528). That C++ binding emits DeprecationWarning on every
    autocast forward. The replacement matches TorchAO's forward and only
    changes the dtype lookup.
    """

    try:
        from torchao.float8.float8_linear import (
            Float8Linear,
            matmul_with_hp_or_float8_args,
        )
    except ImportError:
        return

    current = Float8Linear.forward
    if getattr(current, "_zimage_uses_current_autocast", False):
        return
    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        source = ""
    if source and "get_autocast_gpu_dtype" not in source:
        return

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            input = input.to(_current_cuda_autocast_dtype())
        output = matmul_with_hp_or_float8_args.apply(
            input,
            self.weight.t(),
            self.linear_mm_config,
            self.config,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    forward._zimage_uses_current_autocast = True
    Float8Linear.forward = forward


def _place_trainable_adapter(
    transformer: Any,
    adapter_name: str,
    device: str | torch.device,
) -> None:
    """Move trainable PEFT LoRA params of ``adapter_name`` to ``device`` as BF16.

    Empty iterators are a no-op. Meta tensors are left untouched so the
    contract check can fail closed without materializing them.
    """

    target = torch.device(device)
    for _name, parameter in _iter_trainable_adapter_parameters(
        transformer, adapter_name
    ):
        if parameter.device.type == "meta":
            continue
        parameter.data = parameter.data.to(device=target, dtype=torch.bfloat16)


def _validate_trainable_adapter_contract(
    transformer: Any,
    adapter_name: str,
    device: str | torch.device,
) -> None:
    """Require every trainable param of ``adapter_name`` to be BF16 on ``device``."""

    violations: list[str] = []
    for name, parameter in _iter_trainable_adapter_parameters(
        transformer, adapter_name
    ):
        if parameter.dtype == torch.bfloat16 and _devices_match(
            parameter.device, device
        ):
            continue
        violations.append(
            f"{name} dtype={parameter.dtype} device={parameter.device}"
        )
    if violations:
        raise ModelConfigurationError(
            "trainable LoRA adapter "
            f"{adapter_name!r} parameters must be torch.bfloat16 on "
            f"{torch.device(device)}; violations: {', '.join(violations)}"
        )


def _iter_trainable_adapter_parameters(
    transformer: Any, adapter_name: str
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Yield trainable LoRA parameters that belong to ``adapter_name``.

    Membership uses PEFT tuner containers keyed by adapter name, then the
    ``lora_*.{adapter_name}`` parameter-path convention. Trainable non-LoRA
    parameters are left untouched.
    """

    named = getattr(transformer, "named_parameters", None)
    if not callable(named):
        return
    adapter_ids = _peft_adapter_parameter_ids(transformer, adapter_name)
    for name, parameter in named():
        if not isinstance(parameter, torch.nn.Parameter):
            continue
        if not parameter.requires_grad:
            continue
        if id(parameter) in adapter_ids or _parameter_belongs_to_adapter(
            name, adapter_name
        ):
            yield name, parameter


def _peft_adapter_parameter_ids(transformer: Any, adapter_name: str) -> set[int]:
    """Collect PEFT container tensors registered under ``adapter_name``."""

    ids: set[int] = set()
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return ids
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return ids
    containers = (torch.nn.ModuleDict, torch.nn.ParameterDict)
    for module in modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        for submodule in module.modules():
            if not isinstance(submodule, containers):
                continue
            if adapter_name not in submodule:
                continue
            entry = submodule[adapter_name]
            if isinstance(entry, torch.nn.Parameter):
                ids.add(id(entry))
                continue
            parameters = getattr(entry, "parameters", None)
            if callable(parameters):
                for parameter in parameters():
                    ids.add(id(parameter))
    return ids


def _parameter_belongs_to_adapter(name: str, adapter_name: str) -> bool:
    """True when ``name`` is a PEFT ``lora_*`` tensor for ``adapter_name``."""

    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part.startswith("lora_") and parts[index + 1] == adapter_name:
            return True
    return False


def _devices_match(
    actual: torch.device, requested: str | torch.device
) -> bool:
    expected = torch.device(requested)
    if actual.type != expected.type:
        return False
    if expected.index is None or actual.index is None:
        return True
    return actual.index == expected.index


def _parse_source(raw: Any, label: str) -> ModelSource:
    if not isinstance(raw, Mapping):
        raise ModelConfigurationError(f"{label} must be a mapping")
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ModelConfigurationError(f"{label}.path is required")
    revision = raw.get("revision")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ModelConfigurationError(
            f"{label}.revision must be a non-empty string or null"
        )
    return ModelSource(
        path=path.strip(),
        revision=revision.strip() if isinstance(revision, str) else None,
    )


def _load_component(
    owner: Any,
    source: ModelSource,
    *,
    subfolder: str,
    what: str | None = None,
    **kwargs: Any,
) -> Any:
    if what:
        log.info("loading %s", what)
    load = getattr(owner, "from_pretrained", owner)
    return load(
        source.path,
        subfolder=subfolder,
        revision=source.revision,
        **kwargs,
    )


def _load_model_component(
    owner: Any,
    source: ModelSource,
    *,
    subfolder: str,
    torch_dtype: torch.dtype,
    disable_mmap: bool,
    what: str | None = None,
) -> Any:
    return _load_component(
        owner,
        source,
        subfolder=subfolder,
        what=what,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        disable_mmap=disable_mmap,
    )


def _normalise_hf_identity(value: str) -> str:
    identity = value.strip().replace("\\", "/").rstrip("/").casefold()
    identity = re.sub(r"^hf://", "", identity)
    identity = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", identity)
    identity = identity.split("/resolve/", 1)[0]
    return identity


def _local_source_identities(value: str) -> set[str]:
    identities: set[str] = set()
    normalised = value.replace("\\", "/").casefold()
    match = re.search(r"models--([^/]+)--([^/]+)(?:/|$)", normalised)
    if match:
        identities.add(f"{match.group(1)}/{match.group(2)}")

    root = Path(value).expanduser()
    if not root.is_dir():
        return identities
    for relative in ("model_index.json", "transformer/config.json"):
        try:
            document = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(document, Mapping):
            for key in ("_name_or_path", "name_or_path", "model_id"):
                identity = document.get(key)
                if isinstance(identity, str):
                    identities.add(_normalise_hf_identity(identity))
    return identities


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _module_device(module: Any) -> torch.device:
    device = getattr(module, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _architecture_name(model: Any) -> str:
    config = getattr(model, "config", None)
    if config is not None:
        if isinstance(config, Mapping):
            configured = config.get("_class_name")
        else:
            configured = getattr(config, "_class_name", None)
        if isinstance(configured, str) and configured:
            return configured
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _normalise_config(config: Any) -> Any:
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif not isinstance(config, Mapping) and hasattr(config, "__dict__"):
        config = vars(config)
    if isinstance(config, Mapping):
        ignored = {
            "name_or_path",
            "revision",
            "transformers_version",
            "diffusers_version",
        }
        return {
            str(key): _normalise_config(value)
            for key, value in config.items()
            if key not in ignored and not str(key).startswith("_")
        }
    if isinstance(config, (list, tuple)):
        return [_normalise_config(value) for value in config]
    if isinstance(config, (str, int, float, bool)) or config is None:
        return config
    return repr(config)


def _matching_module_names(
    modules: Mapping[str, Any], target: str
) -> set[str]:
    try:
        expression = re.compile(target)
    except re.error:
        expression = None
    matches = {
        name
        for name in modules
        if name == target
        or name.endswith(f".{target}")
        or (expression is not None and expression.fullmatch(name) is not None)
    }
    return matches


def _module_weight_shape(module: Any) -> tuple[int, ...] | None:
    weight = getattr(module, "weight", None)
    return tuple(weight.shape) if weight is not None else None


_LORA_TENSOR = re.compile(
    r"^(?P<module>.+?)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)


def _validate_lora_tensor_shapes(
    sampling_modules: Mapping[str, Any],
    target_names: set[str],
    state_dict: Mapping[str, Any],
) -> None:
    seen = 0
    ranks: dict[str, dict[str, int]] = {}
    for raw_key, tensor in state_dict.items():
        key = str(raw_key)
        if key.startswith("transformer."):
            key = key[len("transformer.") :]
        match = _LORA_TENSOR.match(key)
        if match is None:
            continue
        module_name = match.group("module")
        if module_name.startswith("base_model.model."):
            module_name = module_name[len("base_model.model.") :]
        if module_name not in target_names or module_name not in sampling_modules:
            raise SamplingCompatibilityError(
                f"LoRA tensor {raw_key!r} targets an unknown sampling module"
            )
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) != 2:
            raise SamplingCompatibilityError(
                f"LoRA tensor {raw_key!r} must be rank 2, got {shape}"
            )
        base_shape = _module_weight_shape(sampling_modules[module_name])
        if base_shape is None or len(base_shape) != 2:
            raise SamplingCompatibilityError(
                f"LoRA target {module_name!r} has no rank-2 base weight"
            )
        side = match.group("side")
        expected_width = base_shape[1] if side == "A" else base_shape[0]
        actual_width = shape[1] if side == "A" else shape[0]
        if actual_width != expected_width:
            raise SamplingCompatibilityError(
                f"LoRA tensor {raw_key!r} shape {shape} is incompatible with "
                f"sampling weight shape {base_shape}"
            )
        rank = shape[0] if side == "A" else shape[1]
        ranks.setdefault(module_name, {})[side] = rank
        seen += 1

    if seen == 0:
        raise SamplingCompatibilityError(
            "LoRA state dict contains no adapter A/B tensors"
        )
    for module_name in sorted(target_names):
        sides = ranks.get(module_name, {})
        missing = {"A", "B"} - sides.keys()
        if missing:
            raise SamplingCompatibilityError(
                f"LoRA tensors for {module_name!r} are missing "
                f"{'/'.join(sorted(missing))} adapter weights"
            )
        if sides["A"] != sides["B"]:
            raise SamplingCompatibilityError(
                f"LoRA rank mismatch for {module_name!r}: "
                f"A={sides['A']}, B={sides['B']}"
            )
