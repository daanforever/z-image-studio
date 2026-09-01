"""Unfused Turbo preview sampling from native LoRA checkpoints.

The sampler never fuses adapters, never reloads a text encoder, and never
deletes a checkpoint when preview rendering fails.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from zimage.config import JPEG_QUALITY
from zimage.training.cache import load_preview_cache
from zimage.training.checkpoints import load_lora_state
from zimage.training.contracts import SavedCheckpoint
from zimage.training.gpu_usage import GpuProbeContext, PeakMemoryScope
from zimage.training.schema import merge_sample_parameters

_PREVIEW_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})

LoraConfigFactory = Callable[..., Any]
PreviewQuantizer = Callable[[Any], None]
DeviceMover = Callable[[Any, torch.device], None]
CudaCleanup = Callable[[torch.device], None]


class PreviewSamplingError(RuntimeError):
    """Preview rendering failed. The source checkpoint must be left intact."""


class UnfusedPreviewSampler:
    """Serial PreviewSampler that swaps an unfused PEFT adapter on one transformer.

    Loop parks main+optimizer; sampler moves sampling transformer+VAE for
    denoise/decode, then parks both.
    """

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any | None = None,
        vae: Any | None = None,
        pipeline: Any | None = None,
        prompt_paths: Mapping[str, Any] | None = None,
        negative_prompt_paths: Mapping[str, Any] | None = None,
        common_parameters: Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        adapter_name: str = "preview",
        lora_config_factory: LoraConfigFactory | None = None,
        main_transformer: Any | None = None,
        target_modules: list[str] | None = None,
        quantizer: PreviewQuantizer | None = None,
        device_mover: DeviceMover | None = None,
        cuda_cleanup: CudaCleanup | None = None,
        gpu_usage_probe: Callable[..., Any] | None = None,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = pipeline
        self.prompt_paths = dict(prompt_paths or {})
        self.negative_prompt_paths = dict(negative_prompt_paths or {})
        self.common_parameters = dict(common_parameters or {})
        self.device = device
        self.adapter_name = adapter_name
        self._lora_config_factory = lora_config_factory
        self.main_transformer = main_transformer
        self.target_modules = list(target_modules or [])
        self._quantizer = quantizer
        self._device_mover = device_mover
        self._cuda_cleanup = cuda_cleanup
        self._gpu_usage_probe = gpu_usage_probe
        self._lock = threading.Lock()
        self._active_adapter: str | None = None
        self._fp8_quantized = False
        self._cuda_touched = False
        self.last_parameters: dict[str, Any] | None = None

    @classmethod
    def from_components(
        cls,
        *,
        transformer: Any,
        scheduler: Any | None = None,
        vae: Any | None = None,
        prompt_paths: Mapping[str, Any] | None = None,
        negative_prompt_paths: Mapping[str, Any] | None = None,
        common_parameters: Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        main_transformer: Any | None = None,
        quantizer: PreviewQuantizer | None = None,
        device_mover: DeviceMover | None = None,
        cuda_cleanup: CudaCleanup | None = None,
        gpu_usage_probe: Callable[..., Any] | None = None,
    ) -> UnfusedPreviewSampler:
        """Build a preview sampler from training components.

        A full ``ZImagePipeline`` is not required at construction. The loop
        owner can pass the independent sampling transformer, scheduler, VAE,
        and preview prompt cache paths produced before text resources are released.
        """

        return cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            prompt_paths=prompt_paths,
            negative_prompt_paths=negative_prompt_paths,
            common_parameters=common_parameters,
            device=device,
            target_modules=target_modules,
            main_transformer=main_transformer,
            quantizer=quantizer,
            device_mover=device_mover,
            cuda_cleanup=cuda_cleanup,
            gpu_usage_probe=gpu_usage_probe,
        )

    def sample_unfused(
        self,
        *,
        checkpoint: SavedCheckpoint,
        parameters: Mapping[str, Any],
        destination: Path,
    ) -> Path:
        """Render one preview from a completed checkpoint and return its path."""

        destination = Path(destination)
        with self._lock:
            try:
                return self._sample_unfused_locked(
                    checkpoint=checkpoint,
                    parameters=parameters,
                    destination=destination,
                )
            except PreviewSamplingError:
                raise
            except Exception as exc:
                raise PreviewSamplingError(
                    f"preview sampling failed for {checkpoint.path}: {exc}"
                ) from exc

    def prepare_for_preview(self) -> None:
        """Prepare the CPU-resident base before injecting a preview adapter."""

        self.remove_adapter()
        self._move_preview_components(torch.device("cpu"))
        from zimage.training.quantization import is_sampling_transformer_quantized

        if is_sampling_transformer_quantized(self.transformer):
            _ensure_peft_torchao_requantizer(self.transformer)
            self._fp8_quantized = True
            return
        if self._resolve_device().type != "cuda" or self._fp8_quantized:
            return
        try:
            quantizer = self._quantizer or _quantize_float8_weight_only
            quantizer(self.transformer)
        except PreviewSamplingError:
            raise
        except Exception as exc:
            raise PreviewSamplingError(
                f"failed to apply TorchAO FP8 weight-only preview quantization: {exc}"
            ) from exc
        self._fp8_quantized = True

    def release_after_preview(self) -> None:
        """Remove the adapter and return preview resources to CPU."""

        failure: Exception | None = None
        try:
            self.remove_adapter()
        except Exception as exc:
            failure = exc
        try:
            self._move_preview_components(torch.device("cpu"))
        except Exception as exc:
            if failure is None:
                failure = exc
        try:
            self._release_cuda_cache()
        except Exception as exc:
            if failure is None:
                failure = PreviewSamplingError(
                    f"failed to release CUDA preview resources: {exc}"
                )
        if failure is not None:
            if isinstance(failure, PreviewSamplingError):
                raise failure
            raise PreviewSamplingError(
                f"failed to release preview resources: {failure}"
            ) from failure

    def load_unfused_adapter(self, checkpoint: SavedCheckpoint) -> None:
        """Load or replace the preview adapter without fusing base weights."""

        loaded = load_lora_state(checkpoint.path)
        metadata = checkpoint.metadata
        state = _peft_compatible_state(loaded.state_dict)
        if self.main_transformer is not None and self.target_modules:
            from zimage.training.modeling import validate_sampling_topology

            validate_sampling_topology(
                self.main_transformer,
                self.transformer,
                self.target_modules,
                state,
            )
        self._replace_adapter(state, metadata, checkpoint.path)

    def remove_adapter(self) -> None:
        """Drop the preview adapter and leave base weights unchanged."""

        name = self._active_adapter
        if name is None:
            return
        transformer = self.transformer
        if hasattr(transformer, "delete_adapters"):
            transformer.delete_adapters(name)
        pipeline = self.pipeline
        if pipeline is not None and hasattr(pipeline, "delete_adapters"):
            try:
                pipeline.delete_adapters(name)
            except Exception:
                pass
        self._active_adapter = None

    def _sample_unfused_locked(
        self,
        *,
        checkpoint: SavedCheckpoint,
        parameters: Mapping[str, Any],
        destination: Path,
    ) -> Path:
        merged = resolve_preview_parameters(self.common_parameters, parameters)
        self.last_parameters = dict(merged)
        failure: BaseException | None = None
        preview_run_scope: PeakMemoryScope | None = None
        try:
            self.prepare_for_preview()
            self.load_unfused_adapter(checkpoint)
            self._move_preview_components(self._resolve_device())
            with PeakMemoryScope() as preview_run_scope:
                image = self._run_pipeline(merged)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_preview_image(image, destination)
            return destination
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                if preview_run_scope is not None:
                    self._record_preview_run(preview_run_scope.peak_bytes)
            except Exception:
                pass
            try:
                self.release_after_preview()
            except Exception as cleanup_exc:
                cleanup_failure = RuntimeError(
                    f"release_after_preview failed: {cleanup_exc}"
                )
                if failure is None:
                    raise cleanup_failure from cleanup_exc
                raise RuntimeError(
                    f"{failure}; additionally, {cleanup_failure}"
                ) from failure

    def _replace_adapter(
        self,
        state: Mapping[str, Any],
        metadata: Any,
        checkpoint_path: Path,
    ) -> None:
        if self._active_adapter is not None:
            self.remove_adapter()
        name = str(getattr(metadata, "adapter_name", None) or self.adapter_name)
        transformer = self.transformer
        pipeline = self.pipeline
        if hasattr(transformer, "add_adapter"):
            _ensure_peft_torchao_requantizer(transformer)
            config = self._build_lora_config(metadata.peft_config)
            transformer.add_adapter(config, adapter_name=name)
            from peft import set_peft_model_state_dict

            result = set_peft_model_state_dict(
                transformer,
                dict(state),
                adapter_name=name,
            )
            _cast_adapter_to_bfloat16(transformer, name)
            unexpected = getattr(result, "unexpected_keys", None)
            if unexpected:
                raise PreviewSamplingError(
                    f"unexpected LoRA keys while loading {name}: {unexpected}"
                )
        elif pipeline is not None and hasattr(pipeline, "load_lora_weights"):
            pipeline.load_lora_weights(str(checkpoint_path), adapter_name=name)
        else:
            raise PreviewSamplingError(
                "sampling transformer does not support unfused PEFT adapters"
            )
        if hasattr(transformer, "set_adapter"):
            transformer.set_adapter(name)
        if pipeline is not None and hasattr(pipeline, "set_adapters"):
            try:
                pipeline.set_adapters(name)
            except Exception:
                pass
        _assert_unfused(transformer)
        self._active_adapter = name

    def _build_lora_config(self, peft_config: Mapping[str, Any]) -> Any:
        factory = self._lora_config_factory
        if factory is None:
            from peft import LoraConfig

            factory = LoraConfig
        payload = dict(peft_config)
        rank = payload.get("r", payload.get("rank"))
        alpha = payload.get("lora_alpha", payload.get("alpha", rank))
        targets = payload.get("target_modules", payload.get("targets"))
        kwargs = {
            "r": int(rank),
            "lora_alpha": alpha,
            "lora_dropout": float(payload.get("lora_dropout", 0.0)),
            "init_lora_weights": True,
        }
        if targets:
            kwargs["target_modules"] = list(targets)
        return factory(**kwargs)

    def _record_preview_run(self, phase_peak_bytes: int) -> None:
        """Log intra-denoise peak before preview weights leave the GPU."""

        probe = self._gpu_usage_probe
        if probe is None:
            return
        try:
            probe(
                "preview_run",
                GpuProbeContext(
                    components={
                        "vae": self.vae,
                        "sampling_transformer": self.transformer,
                        "main_transformer": self.main_transformer,
                        "transformer": self.main_transformer,
                    },
                    transformer=self.main_transformer,
                    phase_peak_bytes=int(phase_peak_bytes),
                    preview_prompt_embeddings=None,
                    preview_negative_embeddings=None,
                    preview_sampler=self,
                ),
            )
        except Exception:
            return

    def _run_pipeline(self, merged: Mapping[str, Any]) -> Image.Image:
        pipeline = self._ensure_pipeline()
        if getattr(pipeline, "text_encoder", None) is not None:
            # Shared embeddings only. A leftover encoder must never be invoked.
            pipeline.text_encoder = None
        if getattr(pipeline, "tokenizer", None) is not None:
            pipeline.tokenizer = None
        if self.vae is not None and getattr(pipeline, "vae", None) is None:
            pipeline.vae = self.vae
        self._bind_time_shift_scheduler(pipeline, float(merged["time_shift"]))

        prompt = str(merged["prompt"])
        negative = str(merged["negative_prompt"])
        device = self._resolve_device()
        generator = _make_generator(device, int(merged["seed"]))

        prompt_cpu = None
        negative_cpu = None
        prompt_device = None
        negative_device = None
        request = None
        result = None
        try:
            prompt_cpu = _require_embedding(
                self.prompt_paths,
                prompt,
                kind="prompt",
            )
            if negative == "":
                negative_cpu = _empty_negative_embeds(prompt_cpu)
            elif negative in self.negative_prompt_paths:
                negative_cpu = _require_embedding(
                    self.negative_prompt_paths,
                    negative,
                    kind="negative_prompt",
                )
            elif negative in self.prompt_paths:
                negative_cpu = _require_embedding(
                    self.prompt_paths,
                    negative,
                    kind="negative_prompt",
                )
            else:
                raise PreviewSamplingError(
                    f"missing negative_prompt embedding for {negative!r}"
                )

            prompt_device = _as_embed_list(prompt_cpu, device)
            negative_device = _as_embed_list(negative_cpu, device)
            # Diffusers 0.40 ZImagePipeline.__call__ does not take time_shift; the
            # shift lives on FlowMatchEulerDiscreteScheduler / calculate_shift.
            request = {
                "prompt": None,
                "prompt_embeds": prompt_device,
                "negative_prompt_embeds": negative_device,
                "height": int(merged["height"]),
                "width": int(merged["width"]),
                "num_inference_steps": int(merged["num_inference_steps"]),
                "guidance_scale": float(merged["guidance_scale"]),
                "generator": generator,
                "output_type": "pil",
            }
            result = _invoke_pipeline(pipeline, request)
            return _coerce_image(
                result, int(merged["width"]), int(merged["height"])
            )
        finally:
            del prompt_cpu, negative_cpu, prompt_device, negative_device, request, result

    def _move_preview_components(self, device: torch.device) -> None:
        pipeline = self.pipeline
        components = [self.transformer, self.vae]
        if pipeline is not None:
            components.extend(
                [getattr(pipeline, "transformer", None), getattr(pipeline, "vae", None)]
            )
        seen: set[int] = set()
        for component in components:
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            if device.type == "cuda":
                self._cuda_touched = True
            try:
                if self._device_mover is not None:
                    self._device_mover(component, device)
                else:
                    _move_component(component, device)
            except Exception as exc:
                destination = "CUDA" if device.type == "cuda" else "CPU"
                raise PreviewSamplingError(
                    f"failed to move preview components to {destination}: {exc}"
                ) from exc

    def _release_cuda_cache(self) -> None:
        if not self._cuda_touched:
            return
        device = self._resolve_device()
        try:
            if self._cuda_cleanup is not None:
                self._cuda_cleanup(device)
            elif torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        finally:
            self._cuda_touched = False

    def _ensure_pipeline(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        if self.scheduler is None:
            raise PreviewSamplingError(
                "a sampling scheduler is required when no pipeline is provided"
            )
        from diffusers import ZImagePipeline

        self.pipeline = ZImagePipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            transformer=self.transformer,
        )
        return self.pipeline

    def _bind_time_shift_scheduler(self, pipeline: Any, time_shift: float) -> None:
        """Install a FlowMatch scheduler with ``shift=time_shift``, as Studio does."""

        from diffusers import FlowMatchEulerDiscreteScheduler

        shifted = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=float(time_shift)
        )
        self.scheduler = shifted
        pipeline.scheduler = shifted

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(self.transformer.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")


def resolve_preview_parameters(
    common: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge root sampling parameters with one sample (sample wins)."""

    return merge_sample_parameters(common, sample)


def _write_preview_image(image: Image.Image, destination: Path) -> None:
    """Save RGB preview as JPEG or PNG from the destination suffix.

    Unlinks sibling files at the same stem with the other suffixes in
    ``{.png, .jpg, .jpeg}`` before writing so a hot-reload of
    ``image_format`` replaces that step/index preview without leaving
    both encodings.
    """

    destination = Path(destination)
    for suffix in _PREVIEW_IMAGE_SUFFIXES:
        sibling = destination.with_suffix(suffix)
        if sibling != destination:
            sibling.unlink(missing_ok=True)
    if destination.suffix.lower() in _JPEG_SUFFIXES:
        image.save(destination, format="JPEG", quality=JPEG_QUALITY)
    else:
        image.save(destination, format="PNG")


def _invoke_pipeline(pipeline: Any, request: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(pipeline)
    except (TypeError, ValueError):
        return pipeline(**request)
    accepted = signature.parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in accepted.values()):
        return pipeline(**request)
    filtered = {key: value for key, value in request.items() if key in accepted}
    return pipeline(**filtered)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(
        device=str(device) if device.type == "cpu" else device
    )
    generator.manual_seed(seed)
    return generator


def _quantize_float8_weight_only(transformer: Any) -> None:
    from zimage.training.quantization import quantize_sampling_transformer

    quantize_sampling_transformer(transformer, precision="fp8")


def _ensure_peft_torchao_requantizer(transformer: Any) -> None:
    from zimage.training.quantization import (
        _attach_peft_torchao_requantizer,
        _peft_torchao_requantizer,
    )

    if _peft_torchao_requantizer(transformer) is not None:
        return
    quant_type = _torchao_preview_quant_type(transformer)
    if quant_type is None:
        return
    _attach_peft_torchao_requantizer(transformer, quant_type)


def _torchao_preview_quant_type(transformer: Any) -> Any | None:
    try:
        from torchao.quantization import Float8WeightOnlyConfig
        from torchao.utils import TorchAOBaseTensor
    except ImportError:
        return None
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return None
    for module in modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, TorchAOBaseTensor):
            return Float8WeightOnlyConfig()
    return None


def _move_component(component: Any, device: torch.device) -> None:
    mover = getattr(component, "to", None)
    if mover is None:
        if device.type == "cuda":
            raise TypeError(f"{type(component).__name__} does not support device transfer")
        return
    try:
        parameter = next(component.parameters())
    except (AttributeError, StopIteration, TypeError):
        parameter = None
    if parameter is not None and parameter.device == device:
        return
    mover(device=device)


def _cast_adapter_to_bfloat16(transformer: Any, adapter_name: str) -> None:
    marker = f".{adapter_name}."
    for name, parameter in getattr(transformer, "named_parameters", lambda: ())():
        if marker in f".{name}." and "lora_" in name:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)


def _require_embedding(store: Mapping[str, Any], key: str, *, kind: str) -> Any:
    if key not in store:
        raise PreviewSamplingError(
            f"missing pre-encoded {kind} embedding for {key!r}; "
            "the sampler does not reload a text encoder"
        )
    path = store[key]
    try:
        embedding, metadata = load_preview_cache(path)
    except Exception as exc:
        raise PreviewSamplingError(
            f"corrupt {kind} cache for {key!r} at {path}: {exc}; "
            "the sampler does not reload a text encoder"
        ) from exc
    del metadata
    return embedding


def _empty_negative_embeds(prompt_embeds: Any) -> Any:
    if isinstance(prompt_embeds, torch.Tensor):
        return torch.zeros_like(prompt_embeds, device="cpu")
    return prompt_embeds


def _as_embed_list(embeds: Any, device: torch.device) -> list[Any]:
    """Copy embeddings onto *device*; do not write tensors back into path stores."""

    if isinstance(embeds, torch.Tensor):
        return [embeds.to(device=device)]
    if isinstance(embeds, (list, tuple)):
        return [
            item.to(device=device) if isinstance(item, torch.Tensor) else item
            for item in embeds
        ]
    return [embeds]


def _coerce_image(result: Any, width: int, height: int) -> Image.Image:
    images = getattr(result, "images", None)
    if images:
        image = images[0]
    elif isinstance(result, Image.Image):
        image = result
    elif isinstance(result, torch.Tensor):
        image = _tensor_to_image(result)
    else:
        raise PreviewSamplingError("pipeline did not return an image")
    if isinstance(image, torch.Tensor):
        image = _tensor_to_image(image)
    if not isinstance(image, Image.Image):
        raise PreviewSamplingError("pipeline image is not a PIL image")
    if image.size != (width, height):
        raise PreviewSamplingError(
            f"preview image size {image.size} does not match "
            f"requested {(width, height)}"
        )
    return image.convert("RGB")


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().to(device="cpu")
    if pixels.ndim == 4:
        pixels = pixels[0]
    if pixels.ndim == 3 and pixels.shape[0] in {1, 3}:
        pixels = pixels.permute(1, 2, 0)
    pixels = pixels.float()
    if pixels.max() <= 1.0:
        pixels = pixels * 255.0
    array = pixels.clamp(0, 255).to(dtype=torch.uint8).numpy()
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array)


def _peft_compatible_state(state: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("transformer."):
            key = key[len("transformer.") :]
        if key.startswith("base_model.model."):
            key = key[len("base_model.model.") :]
        converted[key] = value
    return converted


def _assert_unfused(model: Any) -> None:
    for module in getattr(model, "modules", lambda: ())():
        if hasattr(module, "merged") and module.merged:
            raise PreviewSamplingError("preview adapter was fused; unfused PEFT is required")
        merged_adapters = getattr(module, "merged_adapters", None)
        if merged_adapters:
            raise PreviewSamplingError("preview adapter was fused; unfused PEFT is required")
