"""Official Z-Image flow-matching LoRA optimizer loop.

Checkpoint and preview implementations are injected. This module never
imports those packages, Gradio, or inference fuse/quantization helpers.
"""

from __future__ import annotations

import gc
import importlib
import logging
from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from tqdm import tqdm

from zimage.training.cache import (
    CacheConfig,
    CacheError,
    CacheState,
    expected_preview_metadata,
    inspect_preview_cache,
    load_cache,
    prepare_cache_at_job_start,
    prepare_preview_prompt_cache,
    preview_cache_path,
)
from zimage.training.commands import consume_commands as consume_job_commands
from zimage.training.contracts import (
    CheckpointWriter,
    ConfigReloader,
    ConfigUpdateDecision,
    JobState,
    JobStatus,
    NativeAdapterMetadata,
    OptimizerStepBoundary,
    PreviewSampler,
    SavedCheckpoint,
    StepConfigReload,
    TrainingHook,
)
from zimage.training.dataset import (
    DatasetError,
    DatasetSample,
    discover_samples,
    validate_mvp_batch_settings,
)
from zimage.training.gpu_usage import (
    GpuProbeContext,
    GpuUsageProbe,
    PeakMemoryScope,
    _default_gpu_usage_probe,
    format_bytes,
    snapshot_gpu_usage,
)
from zimage.training.jobs import (
    load_job_config,
    load_job_state,
    preview_sample_path,
    save_job_config,
    write_job_state,
)
from zimage.training.modeling import (
    ComponentLoaders,
    TrainingModelComponents,
    TrainingModelLifecycle,
    _place_trainable_adapter,
    load_training_components,
    setup_main_transformer,
)
from zimage.training.quantization import quantized_precision
from zimage.training.schema import (
    IMMUTABLE_JOB_FIELDS,
    GpuUsageSettings,
    TrainingConfigError,
    UpdateClassification,
    classify_job_update,
    merge_sample_parameters,
    resolve_gpu_usage_settings,
    resolve_stop_condition,
    resolve_training_paths,
    sampling_base_parameters,
    validate_job_document,
)

QWEN_CHAT_TEMPLATE = {
    "add_generation_prompt": True,
    "enable_thinking": True,
}

log = logging.getLogger("zimage.training")
_DEFAULT_GPU_USAGE_PROBE = GpuUsageProbe()
_GPU_USAGE_JOB_PEAKS = {"step": 0, "preview": 0, "nvidia": 0}
_PREVIEW_PROBE_PHASES = frozenset(
    {"preview_pause", "preview_end", "preview_run", "restore"}
)


@dataclass
class FlowMatchingStepResult:
    """Tensors produced by one official Z-Image flow-matching step."""

    loss: torch.Tensor
    noisy_latent: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    timestep_normalized: torch.Tensor
    packed_inputs: list[torch.Tensor]
    model_pred: torch.Tensor
    target: torch.Tensor
    weighting: torch.Tensor
    u: torch.Tensor


def cache_job(job_dir: Path, **injected: Any) -> int:
    """Materialize dataset and preview caches, then release text-encoder resources."""

    job_dir = Path(job_dir)
    job = load_job_config(job_dir)
    injected = _bind_gpu_usage_probe(job, injected)
    log.info("cache start")
    _validate_batch_settings(injected)
    samples = _discover(job, injected)
    components, lifecycle = _load_lifecycle(job, injected)
    placed = [False]
    try:
        cache_config = _resolve_cache_config(job, components, injected)
        _prepare_cache(
            samples,
            job,
            components,
            lifecycle,
            injected,
            job_dir=job_dir,
            placed=placed,
            cache_config=cache_config,
        )
        _prepare_preview_prompt_paths(
            job,
            lifecycle,
            cache_config,
            injected,
            components,
            job_dir=job_dir,
            placed=placed,
        )
    finally:
        if placed[0]:
            try:
                lifecycle.park_cache_modules()
            except Exception:
                log.exception("park cache modules failed")
            _probe_gpu_usage(
                injected, "cache_end", context=GpuProbeContext(components=components)
            )
        lifecycle.release_text_resources()
    return 0


def run_job(job_dir: Path, **injected: Any) -> int:
    """Run the captioned LoRA optimizer loop for one job directory."""

    job_dir = Path(job_dir)
    job = load_job_config(job_dir)
    injected = _bind_gpu_usage_probe(job, injected)
    _reset_gpu_usage_job_peaks()
    _validate_batch_settings(injected)
    device = _resolve_training_device(injected)
    samples = _discover(job, injected)
    if not samples:
        raise DatasetError("job has no training samples")
    # Cache is 1:1 with discovered samples; rewind epoch from that size so
    # epochs-mode repeats the uncheckpointed tail instead of exiting early.
    state = _rewind_state_to_checkpoint(
        job_dir,
        load_job_state(job_dir),
        dataset_size=len(samples),
    )
    log.info(
        "run start job=%s step=%s epoch=%s",
        state.job_id,
        state.step,
        state.epoch,
    )

    holder = {"runtime": _build_runtime(job_dir, job, injected, device=device)}
    try:
        return _optimize(job_dir, state, holder, injected)
    finally:
        runtime = holder["runtime"]
        _teardown_runtime(runtime)
        _probe_gpu_usage(
            injected, "teardown", context=_gpu_probe_context_from_runtime(runtime)
        )
        _probe_gpu_usage_summary(injected, runtime)


def get_scheduler_sigmas(
    scheduler: Any,
    timesteps: torch.Tensor,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Look up flow-matching sigmas the same way the official trainer does."""

    target = device if device is not None else timesteps.device
    sigmas = scheduler.sigmas.to(device=target, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(target)
    timesteps = timesteps.to(target)
    step_indices = [(schedule_timesteps == time).nonzero().item() for time in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def pack_zimage_hidden_states(noisy_latents: torch.Tensor) -> list[torch.Tensor]:
    """Turn batched ``[B,C,H,W]`` latents into a list of ``[C,1,H,W]`` tensors."""

    packed = noisy_latents.unsqueeze(2)
    return list(packed.unbind(dim=0))


def official_flow_matching_step(
    *,
    transformer: Any,
    scheduler: Any,
    latent: torch.Tensor,
    prompt_embedding: torch.Tensor,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    noise: torch.Tensor | None = None,
    density_u: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> FlowMatchingStepResult:
    """One official Diffusers 0.40 Z-Image flow-matching forward + loss."""

    target_device = torch.device(device) if device is not None else latent.device
    model_input = latent.unsqueeze(0).to(device=target_device)
    if noise is None:
        sampled_noise = torch.randn_like(model_input)
    else:
        sampled_noise = noise.to(device=target_device, dtype=model_input.dtype)
        if sampled_noise.shape == model_input.shape[1:]:
            sampled_noise = sampled_noise.unsqueeze(0)
    batch_size = model_input.shape[0]

    if density_u is None:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
    else:
        u = torch.as_tensor(density_u, device="cpu", dtype=torch.float32).reshape(
            batch_size
        )

    num_train_timesteps = _num_train_timesteps(scheduler)
    indices = (u * num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device=model_input.device)

    sigmas = get_scheduler_sigmas(
        scheduler,
        timesteps,
        n_dim=model_input.ndim,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    noisy_latent = (1.0 - sigmas) * model_input + sigmas * sampled_noise
    timestep_normalized = (1000.0 - timesteps) / 1000.0
    packed_inputs = pack_zimage_hidden_states(noisy_latent)
    prompt_list = [prompt_embedding.to(device=model_input.device)]

    raw = transformer(
        packed_inputs,
        timestep_normalized,
        prompt_list,
        return_dict=False,
    )
    model_pred_list = raw[0] if isinstance(raw, (tuple, list)) else raw
    if isinstance(model_pred_list, torch.Tensor):
        model_pred = model_pred_list
    else:
        model_pred = torch.stack(list(model_pred_list), dim=0)
    if model_pred.ndim == 5:
        model_pred = model_pred.squeeze(2)
    model_pred = -model_pred

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=weighting_scheme,
        sigmas=sigmas,
    )
    target = sampled_noise - model_input
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        ),
        1,
    ).mean()

    return FlowMatchingStepResult(
        loss=loss,
        noisy_latent=noisy_latent,
        timesteps=timesteps,
        sigmas=sigmas,
        timestep_normalized=timestep_normalized,
        packed_inputs=packed_inputs,
        model_pred=model_pred,
        target=target,
        weighting=weighting,
        u=u,
    )


def _validate_batch_settings(injected: Mapping[str, Any]) -> None:
    validate_mvp_batch_settings(
        batch_size=int(injected.get("batch_size", 1)),
        gradient_accumulation=int(injected.get("gradient_accumulation", 1)),
    )


def _discover(job: Mapping[str, Any], injected: Mapping[str, Any]) -> list[DatasetSample]:
    return discover_samples(
        job["datasets"],
        _datasets_dir(injected),
        batch_size=int(injected.get("batch_size", 1)),
        gradient_accumulation=int(injected.get("gradient_accumulation", 1)),
    )


def _datasets_dir(injected: Mapping[str, Any]) -> Path:
    override = injected.get("datasets_dir")
    if override is not None:
        return Path(override)
    configured = Path(resolve_training_paths().datasets_dir)
    if configured.is_absolute():
        return configured
    from zimage.config import ROOT

    return (ROOT / configured).resolve()


def _load_lifecycle(
    job: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> tuple[TrainingModelComponents, TrainingModelLifecycle]:
    loaders = injected.get("loaders")
    components = load_training_components(
        job,
        loaders=loaders,
        quantize_capable=_fp8_capable(injected),
    )
    return components, TrainingModelLifecycle(components)


def _place_cache_modules(
    placed: list[bool],
    lifecycle: TrainingModelLifecycle,
    target: torch.device,
    injected: Mapping[str, Any],
    components: TrainingModelComponents,
    *,
    vae: bool = True,
) -> None:
    placed[0] = True
    lifecycle.place_cache_modules(target, vae=vae)
    _probe_gpu_usage(
        injected,
        "cache_place",
        context=GpuProbeContext(components=components),
    )


def _prepare_cache(
    samples: list[DatasetSample],
    job: Mapping[str, Any],
    components: TrainingModelComponents,
    lifecycle: TrainingModelLifecycle,
    injected: Mapping[str, Any],
    *,
    job_dir: Path,
    device: torch.device | str | None = None,
    placed: list[bool] | None = None,
    cache_config: CacheConfig | None = None,
) -> list[Path]:
    if cache_config is None:
        cache_config = _resolve_cache_config(job, components, injected)
    prepare = injected.get("prepare_cache", prepare_cache_at_job_start)
    flag = placed if placed is not None else [False]
    encode_scope: PeakMemoryScope | None = None
    encode_count = 0
    samples_total = len(samples)

    def on_before_encode() -> None:
        target = (
            torch.device(device)
            if device is not None
            else _resolve_training_device(injected)
        )
        _place_cache_modules(
            flag, lifecycle, target, injected, components, vae=True
        )

    def on_before_sample_encode(
        _sample: DatasetSample, _image_size: tuple[int, int]
    ) -> None:
        nonlocal encode_scope, encode_count
        encode_count += 1
        encode_scope = PeakMemoryScope()
        encode_scope.__enter__()

    def on_after_sample_encode(
        sample: DatasetSample, image_size: tuple[int, int]
    ) -> None:
        nonlocal encode_scope
        peak_bytes = 0
        if encode_scope is not None:
            encode_scope.__exit__(None, None, None)
            peak_bytes = encode_scope.peak_bytes
            encode_scope = None
        width, height = image_size
        log.info(
            "cache encode n=%s samples=%s path=%s size=%sx%s",
            encode_count,
            samples_total,
            sample.image_path,
            width,
            height,
        )
        _probe_gpu_usage(
            injected,
            "cache_encode",
            context=GpuProbeContext(
                components=components,
                phase_peak_bytes=peak_bytes,
            ),
        )

    log.info("cache prepare samples=%s", samples_total)
    return prepare(
        samples,
        lifecycle.cache_encoder(),
        cache_config,
        job_dir=job_dir,
        on_before_encode=on_before_encode,
        on_before_sample_encode=on_before_sample_encode,
        on_after_sample_encode=on_after_sample_encode,
    )


def _resolve_cache_config(
    job: Mapping[str, Any],
    components: TrainingModelComponents,
    injected: Mapping[str, Any],
) -> CacheConfig:
    cache_config = injected.get("cache_config")
    if cache_config is None:
        cache_config = cache_config_from_components(job, components)
    return cache_config


def cache_config_from_components(
    job: Mapping[str, Any],
    components: TrainingModelComponents,
) -> CacheConfig:
    main = (job.get("model") or {}).get("main_transformer") or {}
    revision = main.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        revision = _first_text(
            getattr(components.main_transformer, "revision", None),
            getattr(components.vae, "revision", None),
            _as_mapping(getattr(components.main_transformer, "config", None)).get(
                "_commit_hash"
            ),
            main.get("path"),
            "local",
        )
    return CacheConfig(
        main_revision=str(revision),
        vae_config=_as_mapping(getattr(components.vae, "config", None)),
        text_encoder_config=_as_mapping(
            getattr(components.text_encoder, "config", None)
        ),
        tokenizer_config=_as_mapping(getattr(components.tokenizer, "config", None)),
        qwen_chat_template=dict(QWEN_CHAT_TEMPLATE),
        max_sequence_length=int(job["max_sequence_length"]),
        text_encoder_precision=(
            quantized_precision(components.text_encoder) or "bf16"
        ),
    )


def _build_runtime(
    job_dir: Path,
    job: dict[str, Any],
    injected: Mapping[str, Any],
    *,
    adapter_state: Mapping[str, Any] | None = None,
    accelerator: Any = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Assemble cache and training objects. Device residency: see TrainingModelLifecycle."""

    injected = _bind_gpu_usage_probe(job, injected)
    training_device = (
        torch.device(device)
        if device is not None
        else _resolve_training_device(injected)
    )
    samples = _discover(job, injected)
    if not samples:
        raise DatasetError("job has no training samples")
    components, lifecycle = _load_lifecycle(job, injected)
    _probe_gpu_usage(
        injected, "load", context=GpuProbeContext(components=components)
    )
    preview_prompt_paths: dict[str, Path] = {}
    preview_negative_paths: dict[str, Path] = {}
    preview_sampler = None
    placed = [False]
    cache_config = _resolve_cache_config(job, components, injected)
    try:
        cache_paths = _prepare_cache(
            samples,
            job,
            components,
            lifecycle,
            injected,
            job_dir=job_dir,
            device=training_device,
            placed=placed,
            cache_config=cache_config,
        )
        preview_prompt_paths, preview_negative_paths = _prepare_preview_prompt_paths(
            job,
            lifecycle,
            cache_config,
            injected,
            components,
            job_dir=job_dir,
            device=training_device,
            placed=placed,
        )
        if "preview_sampler" not in injected:
            preview_sampler = _default_preview_sampler(
                {
                    "components": components,
                    "lifecycle": lifecycle,
                    "config": job,
                    "preview_prompt_paths": preview_prompt_paths,
                    "preview_negative_paths": preview_negative_paths,
                    "device": training_device,
                },
                injected,
            )
    finally:
        if placed[0]:
            try:
                lifecycle.park_cache_modules()
            except Exception:
                log.exception("park cache modules failed")
            _probe_gpu_usage(
                injected, "cache_end", context=GpuProbeContext(components=components)
            )
        lifecycle.release_text_resources()
    transformer, setup = _setup_transformer(
        components.main_transformer, job, injected, training_device
    )
    if adapter_state:
        _apply_adapter_state(transformer, adapter_state, setup.adapter_name, injected)
    else:
        _maybe_warm_start(
            job_dir,
            transformer,
            setup.adapter_name,
            injected,
            job=job,
        )
    _place_trainable_adapter(transformer, setup.adapter_name, training_device)
    optimizer = _make_optimizer(transformer, job, injected)
    if accelerator is None:
        accelerator = injected.get("accelerator")
    if accelerator is None:
        accelerator = _construct_accelerator(training_device, injected)
    accelerator = _bind_accelerator_device(accelerator, training_device)
    prepared = accelerator.prepare(transformer, optimizer)
    transformer, optimizer = prepared[0], prepared[1]
    _validate_prepared_training_runtime(
        accelerator=accelerator,
        transformer=transformer,
        setup=setup,
        device=training_device,
        job=job,
        injected=injected,
        components=components,
    )
    settings = injected.get("gpu_usage_settings")
    if not isinstance(settings, GpuUsageSettings):
        settings = resolve_gpu_usage_settings(job)
    _probe_gpu_usage(
        injected,
        "train_placed",
        context=GpuProbeContext(
            components=components,
            optimizer=optimizer,
            transformer=transformer,
            preview_sampler=preview_sampler,
        ),
    )
    _seed_everything(int(job["seed"]))
    return {
        "config": job,
        "components": components,
        "lifecycle": lifecycle,
        "setup": setup,
        "transformer": transformer,
        "optimizer": optimizer,
        "cache_paths": cache_paths,
        "samples": samples,
        "accelerator": accelerator,
        "device": training_device,
        "last_error": None,
        "job_dir": job_dir,
        "cache_config": cache_config,
        "preview_sampler": preview_sampler,
        "preview_prompt_paths": preview_prompt_paths,
        "preview_negative_paths": preview_negative_paths,
        "gpu_usage_settings": settings,
    }


def _setup_transformer(
    transformer: Any,
    job: Mapping[str, Any],
    injected: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any]:
    setup = setup_main_transformer(
        transformer,
        precision=str(job["precision"]),
        fp8_capable=_fp8_capable(injected),
        lora=job["lora"],
        gradient_checkpointing=bool(job["gradient_checkpointing"]),
        device=device,
        adapter_name=str(injected.get("adapter_name", "default")),
        convert_to_float8_training=injected.get("convert_to_float8_training"),
        float8_config_factory=injected.get("float8_config_factory"),
        lora_config_factory=injected.get("lora_config_factory"),
    )
    return setup.transformer, setup


def _progress_total(kind: str, limit: int, n_cached: int) -> int:
    if kind == "max_steps":
        return int(limit)
    return int(limit) * int(n_cached)


def _optimize(
    job_dir: Path,
    state: JobState,
    holder: dict[str, Any],
    injected: Mapping[str, Any],
) -> int:
    runtime = holder["runtime"]
    step = int(state.step)
    epoch = int(state.epoch)
    cache_paths: list[Path] = runtime["cache_paths"]
    sample_index = step % len(cache_paths)
    hook: TrainingHook | None = injected.get("training_hook")
    ran = False

    kind, limit = resolve_stop_condition(runtime["config"])
    pbar = tqdm(
        total=_progress_total(kind, limit, len(cache_paths)),
        initial=step,
        disable=None,
    )
    try:
        while True:
            runtime = holder["runtime"]
            config = runtime["config"]
            device = torch.device(runtime["device"])
            kind, limit = resolve_stop_condition(config)
            if kind == "max_steps" and step >= limit:
                break
            if kind == "epochs" and epoch >= limit:
                break

            transformer = runtime["transformer"]
            if callable(getattr(transformer, "train", None)):
                transformer.train()

            settings = runtime.get("gpu_usage_settings")
            if not isinstance(settings, GpuUsageSettings):
                settings = GpuUsageSettings()
            next_step = step + 1
            probe_this_step = _should_probe_optimizer_step(next_step, config, settings)
            step_scope: Any = PeakMemoryScope() if probe_this_step else nullcontext()
            with step_scope as scope:
                cache_path = cache_paths[sample_index]
                try:
                    sample = load_cache(cache_path)
                except CacheError:
                    raise
                except Exception as exc:
                    raise CacheError(f"cannot read cache: {cache_path}") from exc
                result = None
                try:
                    result = official_flow_matching_step(
                        transformer=transformer,
                        scheduler=runtime["components"].training_scheduler,
                        latent=sample.latent,
                        prompt_embedding=sample.prompt_embedding,
                        weighting_scheme=str(config["weighting_scheme"]),
                        logit_mean=float(config["logit_mean"]),
                        logit_std=float(config["logit_std"]),
                        mode_scale=float(config["mode_scale"]),
                        noise=_resolve_noise(injected, sample.latent),
                        density_u=injected.get("density_u"),
                        device=device,
                    )
                    ran = True

                    accelerator = runtime["accelerator"]
                    accumulate = getattr(accelerator, "accumulate", None)
                    context = (
                        accumulate(transformer)
                        if callable(accumulate)
                        else nullcontext()
                    )
                    with context:
                        backward = getattr(accelerator, "backward", None)
                        if callable(backward):
                            backward(result.loss)
                        else:
                            result.loss.backward()
                        runtime["optimizer"].step()
                        runtime["optimizer"].zero_grad()
                finally:
                    del sample, result

            step += 1
            sample_index += 1
            if sample_index >= len(cache_paths):
                sample_index = 0
                epoch += 1

            if probe_this_step:
                _probe_gpu_usage(
                    injected,
                    "step",
                    context=_gpu_probe_context_from_runtime(
                        runtime,
                        phase_peak_bytes=getattr(scope, "peak_bytes", None),
                    ),
                )

            persisted = _write_running_state(
                job_dir, state.job_id, step, epoch, runtime["last_error"]
            )
            pbar.update(1)
            pbar.set_postfix_str(f"step={step} epoch={epoch}")
            if hook is not None:
                hook.on_optimizer_step(
                    OptimizerStepBoundary(
                        job_dir=job_dir,
                        state=persisted,
                        config=config,
                    )
                )

            if _should_checkpoint(step, config):
                exit_code = _write_checkpoint_then_sample(
                    job_dir, persisted, runtime, injected
                )
                if exit_code:
                    return exit_code

            reload = _reload_at_step(job_dir, persisted, runtime["config"], injected)
            runtime["config"] = dict(reload.effective_config)
            runtime["last_error"] = _error_from_reload(reload, runtime["last_error"])
            if runtime["last_error"]:
                _write_running_state(
                    job_dir, state.job_id, step, epoch, runtime["last_error"]
                )
            noteworthy = [
                decision
                for decision in reload.decisions
                if decision.classification
                in (
                    UpdateClassification.REJECTED_IMMUTABLE,
                    UpdateClassification.INVALID,
                )
            ]
            if noteworthy:
                log.error("%s", runtime["last_error"])
            try:
                if any(
                    decision.classification is UpdateClassification.APPLY_AT_STEP
                    for decision in reload.decisions
                ):
                    log.info("hot-reload step=%s epoch=%s", step, epoch)
                _apply_hot_runtime(runtime, reload, injected)
            except Exception as exc:
                log.error("hot-reload failed: %s", exc)
                runtime["last_error"] = str(exc)
                _write_running_state(
                    job_dir, state.job_id, step, epoch, runtime["last_error"]
                )
                return 1
            kind, limit = resolve_stop_condition(runtime["config"])
            pbar.total = _progress_total(kind, limit, len(cache_paths))
            if reload.rebuild_required:
                log.info("rebuild step=%s epoch=%s", step, epoch)
                try:
                    holder["runtime"] = _rebuild_runtime(job_dir, runtime, injected)
                except Exception as exc:
                    log.error("rebuild failed: %s", exc)
                    runtime["last_error"] = str(exc)
                    _write_running_state(
                        job_dir, state.job_id, step, epoch, runtime["last_error"]
                    )
                    return 1
                runtime = holder["runtime"]
                cache_paths = runtime["cache_paths"]
                sample_index = step % len(cache_paths)
                kind, limit = resolve_stop_condition(runtime["config"])
                pbar.total = _progress_total(kind, limit, len(cache_paths))

        if ran and not _should_checkpoint(step, runtime["config"]):
            persisted = _write_running_state(
                job_dir, state.job_id, step, epoch, runtime["last_error"]
            )
            exit_code = _write_checkpoint_then_sample(
                job_dir, persisted, runtime, injected
            )
            if exit_code:
                return exit_code
        return 0
    finally:
        pbar.close()


def _should_checkpoint(step: int, config: Mapping[str, Any]) -> bool:
    every = int(config["checkpoint_every"])
    return every > 0 and step > 0 and step % every == 0


def _should_probe_optimizer_step(
    step: int,
    config: Mapping[str, Any],
    settings: GpuUsageSettings,
) -> bool:
    if int(step) <= 0:
        return False
    if settings.every_step:
        return True
    if step in (1, 2):
        return True
    return _should_checkpoint(step, config)


def _write_checkpoint_then_sample(
    job_dir: Path,
    state: JobState,
    runtime: dict[str, Any],
    injected: Mapping[str, Any],
) -> int:
    """Loop parks main+optimizer; sampler moves sampling transformer+VAE for denoise/decode, then parks both."""
    writer: CheckpointWriter | None = _injected_or_default(
        injected, "checkpoint_writer", _default_checkpoint_writer
    )
    if "preview_sampler" in injected:
        sampler = injected["preview_sampler"]
    else:
        sampler = runtime.get("preview_sampler")
        if sampler is None:
            sampler = _default_preview_sampler(runtime, injected)
    if writer is None:
        return 0

    log.info("checkpoint step=%s epoch=%s", state.step, state.epoch)
    transformer = runtime["transformer"]
    unwrap = getattr(runtime["accelerator"], "unwrap_model", None)
    if callable(unwrap):
        transformer = unwrap(transformer)
    adapter_name = runtime["setup"].adapter_name
    lora_state = _lora_state(transformer, adapter_name, injected)
    metadata = NativeAdapterMetadata(
        adapter_name=adapter_name,
        base_model_name_or_path=str(
            runtime["config"]["model"]["main_transformer"]["path"]
        ),
        base_model_revision=runtime["config"]["model"]["main_transformer"].get(
            "revision"
        ),
        peft_config={
            "r": runtime["config"]["lora"]["rank"],
            "lora_alpha": runtime["config"]["lora"]["alpha"],
            "lora_dropout": runtime["config"]["lora"]["dropout"],
            "target_modules": list(runtime["config"]["lora"]["targets"]),
        },
        optimizer_step=state.step,
    )
    destination = job_dir / "checkpoints" / f"step-{state.step}"
    saved = writer.write_atomic(
        destination=destination,
        lora_state=lora_state,
        metadata=metadata,
    )
    if sampler is None:
        return 0
    log.info("preview step=%s", state.step)
    training_device = _training_device_from_runtime(runtime, injected)
    cuda_handoff = training_device.type == "cuda"
    failure: Exception | None = None
    try:
        if cuda_handoff:
            _pause_training_runtime(
                transformer,
                runtime["optimizer"],
                injected,
            )
            _probe_gpu_usage(
                injected,
                "preview_pause",
                context=_gpu_probe_context_from_runtime(
                    runtime, preview_sampler=sampler
                ),
            )
        with PeakMemoryScope() as preview_scope:
            _sample_previews(job_dir, saved, runtime["config"], sampler, state.step)
        _probe_gpu_usage(
            injected,
            "preview_end",
            context=_gpu_probe_context_from_runtime(
                runtime,
                phase_peak_bytes=preview_scope.peak_bytes,
                preview_sampler=sampler,
            ),
        )
    except Exception as exc:
        failure = exc
    finally:
        if cuda_handoff:
            try:
                _restore_training_runtime(
                    transformer,
                    runtime["optimizer"],
                    training_device,
                    injected,
                )
                _probe_gpu_usage(
                    injected,
                    "restore",
                    context=_gpu_probe_context_from_runtime(
                        runtime, preview_sampler=sampler
                    ),
                )
            except Exception as exc:
                restore_failure = RuntimeError(
                    f"failed to restore training runtime after previews: {exc}"
                )
                if failure is None:
                    failure = restore_failure
                else:
                    failure = RuntimeError(
                        f"{failure}; additionally, {restore_failure}"
                    )
    if failure is not None:
        log.error("checkpoint/preview failed: %s", failure)
        runtime["last_error"] = str(failure)
        _write_running_state(
            job_dir, state.job_id, state.step, state.epoch, runtime["last_error"]
        )
        return 1
    return 0


def _pause_training_runtime(
    transformer: Any,
    optimizer: Any,
    injected: Mapping[str, Any],
) -> None:
    """Move training allocations off CUDA before serial preview sampling."""

    synchronize = injected.get("cuda_synchronize", torch.cuda.synchronize)
    move_transformer = injected.get(
        "training_transformer_mover", _move_transformer_to_device
    )
    collect = injected.get("garbage_collect", gc.collect)
    empty_cache = injected.get("cuda_empty_cache", torch.cuda.empty_cache)
    synchronize()
    move_transformer(transformer, torch.device("cpu"))
    _move_optimizer_state_tensors(optimizer, torch.device("cpu"), injected)
    collect()
    empty_cache()


def _restore_training_runtime(
    transformer: Any,
    optimizer: Any,
    training_device: torch.device,
    injected: Mapping[str, Any],
) -> None:
    """Best-effort restore of every training allocation after previews."""

    move_transformer = injected.get(
        "training_transformer_mover", _move_transformer_to_device
    )
    synchronize = injected.get("cuda_synchronize", torch.cuda.synchronize)
    failures: list[str] = []
    try:
        move_transformer(transformer, training_device)
    except Exception as exc:
        failures.append(f"main transformer: {exc}")
    try:
        _move_optimizer_state_tensors(optimizer, training_device, injected)
    except Exception as exc:
        failures.append(f"optimizer state: {exc}")
    try:
        synchronize()
    except Exception as exc:
        failures.append(f"CUDA synchronize: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))


def _move_transformer_to_device(transformer: Any, device: torch.device) -> None:
    transformer.to(device)


def _move_optimizer_state_tensors(
    optimizer: Any,
    device: torch.device,
    injected: Mapping[str, Any] | None = None,
) -> None:
    """Recursively migrate optimizer-state tensor values without rebuilding it."""

    injected = injected or {}
    move_tensor = injected.get("optimizer_tensor_mover", _move_tensor_to_device)
    state = optimizer.state
    for key, value in list(state.items()):
        state[key] = _move_nested_tensor_values(value, device, move_tensor)


def _move_nested_tensor_values(
    value: Any,
    device: torch.device,
    move_tensor: Any,
) -> Any:
    if isinstance(value, torch.Tensor):
        return move_tensor(value, device)
    if isinstance(value, MutableMapping):
        for key, item in list(value.items()):
            value[key] = _move_nested_tensor_values(item, device, move_tensor)
        return value
    if isinstance(value, Mapping):
        return {
            key: _move_nested_tensor_values(item, device, move_tensor)
            for key, item in value.items()
        }
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_nested_tensor_values(item, device, move_tensor)
        return value
    if isinstance(value, tuple):
        moved = tuple(
            _move_nested_tensor_values(item, device, move_tensor) for item in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    return value


def _move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return tensor.to(device=device)


def _sample_previews(
    job_dir: Path,
    checkpoint: SavedCheckpoint,
    config: Mapping[str, Any],
    sampler: PreviewSampler,
    step: int,
) -> None:
    sampling = config["sampling"]
    common = sampling_base_parameters(sampling)
    image_format = sampling.get("image_format")
    for index, sample in enumerate(sampling["samples"]):
        parameters = merge_sample_parameters(common, sample)
        destination = preview_sample_path(job_dir, step, index, image_format)
        sampler.sample_unfused(
            checkpoint=checkpoint,
            parameters=parameters,
            destination=destination,
        )


def _reload_at_step(
    job_dir: Path,
    state: JobState,
    current_config: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> StepConfigReload:
    reloader: ConfigReloader | None = injected.get("config_reloader")
    if reloader is not None:
        return reloader.reload_at_optimizer_step(
            job_dir=job_dir,
            state=state,
            current_config=current_config,
        )
    return _consume_config_updates(job_dir, current_config, injected)


def _consume_config_updates(
    job_dir: Path,
    current_config: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> StepConfigReload:
    effective = validate_job_document(current_config)
    decisions: list[ConfigUpdateDecision] = []
    consume = injected.get("consume_commands", consume_job_commands)

    def handler(envelope: Any) -> None:
        nonlocal effective
        if getattr(envelope, "kind", None) != "update":
            decisions.append(
                ConfigUpdateDecision(
                    command_id=int(envelope.command_id),
                    classification=UpdateClassification.NO_CHANGE,
                    changed_fields=(),
                    message="ignored unsupported command kind",
                )
            )
            return
        payload = getattr(envelope, "payload", {})
        candidate = payload.get("config") if isinstance(payload, Mapping) else None
        try:
            classification, changed = classify_job_update(effective, candidate)
        except (TrainingConfigError, TypeError, ValueError) as exc:
            decisions.append(
                ConfigUpdateDecision(
                    command_id=int(envelope.command_id),
                    classification=UpdateClassification.INVALID,
                    changed_fields=(),
                    message=str(exc),
                )
            )
            return
        if classification is UpdateClassification.REJECTED_IMMUTABLE:
            decisions.append(
                ConfigUpdateDecision(
                    command_id=int(envelope.command_id),
                    classification=classification,
                    changed_fields=changed,
                    message=(
                        "rejected immutable fields: "
                        + ", ".join(changed or sorted(IMMUTABLE_JOB_FIELDS))
                    ),
                )
            )
            return
        if classification is UpdateClassification.NO_CHANGE:
            decisions.append(
                ConfigUpdateDecision(
                    command_id=int(envelope.command_id),
                    classification=classification,
                    changed_fields=(),
                )
            )
            return
        validated = validate_job_document(candidate)
        save_job_config(job_dir, validated)
        effective = validated
        decisions.append(
            ConfigUpdateDecision(
                command_id=int(envelope.command_id),
                classification=classification,
                changed_fields=changed,
            )
        )

    consume(job_dir, handler)
    return StepConfigReload(effective_config=effective, decisions=tuple(decisions))


def _error_from_reload(
    reload: StepConfigReload, current: str | None
) -> str | None:
    noteworthy = [
        decision
        for decision in reload.decisions
        if decision.classification
        in (
            UpdateClassification.REJECTED_IMMUTABLE,
            UpdateClassification.INVALID,
        )
    ]
    if not noteworthy:
        return current
    messages = [
        item.message
        or (
            "invalid update"
            if item.classification is UpdateClassification.INVALID
            else "rejected immutable update"
        )
        for item in noteworthy
    ]
    return "; ".join(messages)


def _apply_hot_runtime(
    runtime: dict[str, Any],
    reload: StepConfigReload,
    injected: Mapping[str, Any] | None = None,
) -> None:
    injected = injected or {}
    changed = {
        field
        for decision in reload.decisions
        if decision.classification is UpdateClassification.APPLY_AT_STEP
        for field in decision.changed_fields
    }
    if not changed:
        if any(
            decision.classification is UpdateClassification.APPLY_AT_STEP
            for decision in reload.decisions
        ):
            changed = {"optimizer.learning_rate"}
        else:
            return
    config = runtime["config"]
    if any(field.startswith("optimizer") for field in changed):
        optimizer = runtime["optimizer"]
        learning_rate = float(config["optimizer"]["learning_rate"])
        weight_decay = float(config["optimizer"]["weight_decay"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay
    if "seed" in changed:
        _seed_everything(int(config["seed"]))
    if any(field == "sampling" or field.startswith("sampling.") for field in changed):
        # ``_changed_paths`` does not descend into lists, so ``samples[0].prompt``
        # arrives as ``sampling.samples`` (same path as seed/size). Refresh TE
        # only when a required prompt file is missing or stale.
        if _missing_preview_prompt_cache(runtime, injected):
            _refresh_preview_prompt_cache_serial(runtime, injected)


def _preview_sampler_from_runtime(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> Any:
    if "preview_sampler" in injected:
        return injected["preview_sampler"]
    return runtime.get("preview_sampler")


def _preview_prompt_path_stores(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> tuple[Mapping[str, Path], Mapping[str, Path]]:
    """Merged runtime + sampler prompt-cache path maps."""

    prompt_store: dict[str, Path] = {}
    negative_store: dict[str, Path] = {}
    runtime_prompts = runtime.get("preview_prompt_paths")
    runtime_negatives = runtime.get("preview_negative_paths")
    if isinstance(runtime_prompts, Mapping):
        prompt_store.update(runtime_prompts)
    if isinstance(runtime_negatives, Mapping):
        negative_store.update(runtime_negatives)
    sampler = _preview_sampler_from_runtime(runtime, injected)
    if sampler is not None:
        sampler_prompts = getattr(sampler, "prompt_paths", None)
        sampler_negatives = getattr(sampler, "negative_prompt_paths", None)
        if isinstance(sampler_prompts, Mapping):
            prompt_store.update(sampler_prompts)
        if isinstance(sampler_negatives, Mapping):
            negative_store.update(sampler_negatives)
    return prompt_store, negative_store


def _missing_preview_prompt_cache(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> bool:
    """True when a required prompt path is absent or its cache is missing/stale."""

    prompts, negatives = _collect_preview_prompt_texts(runtime["config"])
    prompt_store, negative_store = _preview_prompt_path_stores(runtime, injected)
    if any(prompt not in prompt_store for prompt in prompts if prompt) or any(
        negative not in negative_store for negative in negatives
    ):
        return True
    cache_config = runtime.get("cache_config")
    job_dir = runtime.get("job_dir")
    if cache_config is None or job_dir is None:
        return bool(_unique_preview_prompt_texts(runtime["config"]))
    return _preview_prompt_files_need_encode(
        _unique_preview_prompt_texts(runtime["config"]),
        cache_config,
        job_dir,
    )


def _refresh_preview_prompt_cache_serial(
    runtime: dict[str, Any],
    injected: Mapping[str, Any],
) -> None:
    """Pause main only when a prompt file must be encoded, then restore main.

    Compatible files are reused as paths. Reloads Qwen on CPU and places the
    text encoder (not VAE) only for missing or stale prompt caches.
    """

    lifecycle = runtime.get("lifecycle")
    if lifecycle is None:
        raise RuntimeError(
            "cannot refresh preview prompts: training lifecycle is missing"
        )
    job_dir = runtime.get("job_dir")
    cache_config = runtime.get("cache_config")
    if job_dir is None or cache_config is None:
        raise RuntimeError(
            "cannot refresh preview prompts: job cache identity is missing"
        )
    job_dir = Path(job_dir)
    texts = _unique_preview_prompt_texts(runtime["config"])
    if not _preview_prompt_files_need_encode(texts, cache_config, job_dir):
        _assign_preview_prompt_paths(
            runtime,
            injected,
            {text: preview_cache_path(job_dir, text) for text in texts},
        )
        return

    training_device = _training_device_from_runtime(runtime, injected)
    transformer = runtime["transformer"]
    unwrap = getattr(runtime.get("accelerator"), "unwrap_model", None)
    if callable(unwrap):
        transformer = unwrap(transformer)
    sampler = _preview_sampler_from_runtime(runtime, injected)
    cuda_handoff = training_device.type == "cuda"
    failure: Exception | None = None
    try:
        if cuda_handoff:
            _pause_training_runtime(
                transformer,
                runtime["optimizer"],
                injected,
            )
        if sampler is not None:
            release = getattr(sampler, "release_after_preview", None)
            if callable(release):
                release()
        loaders = injected.get("loaders")
        lifecycle.reload_text_resources_on_cpu(
            loaders=loaders,
            precision=str(runtime["config"]["precision"]).strip().lower(),
            quantize_capable=_fp8_capable(injected),
        )
        try:
            lifecycle.place_cache_modules(training_device, vae=False)
            paths = prepare_preview_prompt_cache(
                texts,
                lifecycle.cache_encoder(),
                cache_config,
                job_dir=job_dir,
            )
        finally:
            lifecycle.release_text_resources()
        _assign_preview_prompt_paths(runtime, injected, paths)
    except Exception as exc:
        failure = exc
    finally:
        if cuda_handoff:
            try:
                _restore_training_runtime(
                    transformer,
                    runtime["optimizer"],
                    training_device,
                    injected,
                )
            except Exception as restore_exc:
                restore_failure = RuntimeError(
                    f"failed to restore training runtime after prompt refresh: "
                    f"{restore_exc}"
                )
                if failure is None:
                    failure = restore_failure
                else:
                    failure = RuntimeError(
                        f"{failure}; additionally, {restore_failure}"
                    )
    if failure is not None:
        raise failure


def _rewind_state_to_checkpoint(
    job_dir: Path,
    state: JobState,
    *,
    dataset_size: int,
) -> JobState:
    """Align ``state.step`` / ``state.epoch`` with the latest checkpoint.

    Uncheckpointed optimizer steps after that point are repeated on resume.
    Epoch is ``checkpoint_step // dataset_size`` so epochs-mode still has
    remaining steps in the current epoch instead of exiting immediately.
    """

    if dataset_size <= 0:
        raise DatasetError("job has no training samples")
    load_latest = _training_impl("checkpoints", "load_latest_lora_state")
    loaded = load_latest(job_dir)
    if loaded is None:
        return state
    checkpoint_step = int(loaded.metadata.optimizer_step)
    if int(state.step) <= checkpoint_step:
        return state
    rewound = JobState(
        job_id=state.job_id,
        status=state.status,
        step=checkpoint_step,
        epoch=checkpoint_step // dataset_size,
        last_error=state.last_error,
        exit_code=state.exit_code,
    )
    write_job_state(job_dir, rewound)
    return rewound


def _rebuild_runtime(
    job_dir: Path,
    runtime: dict[str, Any],
    injected: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_name = runtime["setup"].adapter_name
    transformer = runtime["transformer"]
    unwrap = getattr(runtime["accelerator"], "unwrap_model", None)
    if callable(unwrap):
        transformer = unwrap(transformer)
    try:
        preserved = _lora_state(transformer, adapter_name, injected)
    except Exception as exc:
        raise TrainingConfigError(
            f"failed to export LoRA state before rebuild: {exc}"
        ) from exc
    if not preserved:
        raise TrainingConfigError(
            "rebuild requires a non-empty in-memory LoRA state; "
            "refusing to fall back to an older on-disk checkpoint"
        )
    training_device = torch.device(runtime["device"])
    _teardown_runtime(
        runtime,
        device=training_device,
        injected=injected,
        fail_closed=True,
    )
    # Never reuse the previous Accelerator: prepare() would keep the old
    # model/optimizer registered and pin a second stack on a 16 GB GPU.
    # Also ignore any injected prebuilt accelerator instance on rebuild.
    # Fail-closed teardown must have succeeded before this construct.
    new_accelerator = _construct_accelerator(training_device, injected)
    rebuilt = _build_runtime(
        job_dir,
        runtime["config"],
        injected,
        adapter_state=preserved,
        accelerator=new_accelerator,
        device=training_device,
    )
    rebuilt["last_error"] = runtime.get("last_error")
    return rebuilt


def _teardown_runtime(
    runtime: Mapping[str, Any],
    *,
    device: torch.device | None = None,
    injected: Mapping[str, Any] | None = None,
    fail_closed: bool = False,
) -> None:
    injected = injected or {}
    failures: list[str] = []

    def record(label: str, exc: BaseException) -> None:
        if fail_closed:
            failures.append(f"{label}: {exc}")

    sampler = runtime.get("preview_sampler")
    if sampler is not None:
        release = getattr(sampler, "release_after_preview", None)
        if callable(release):
            try:
                release()
            except Exception as exc:
                record("sampler release", exc)
    transformer = runtime.get("transformer")
    optimizer = runtime.get("optimizer")
    park_training = fail_closed or (device is not None and device.type == "cuda")
    if park_training:
        if transformer is not None:
            try:
                unwrap = getattr(runtime.get("accelerator"), "unwrap_model", None)
                target = unwrap(transformer) if callable(unwrap) else transformer
                _move_transformer_to_device(target, torch.device("cpu"))
            except Exception as exc:
                record("main transformer", exc)
        if optimizer is not None:
            try:
                _move_optimizer_state_tensors(
                    optimizer, torch.device("cpu"), injected
                )
            except Exception as exc:
                record("optimizer state", exc)
    lifecycle = runtime.get("lifecycle")
    if lifecycle is not None and (
        fail_closed or getattr(lifecycle, "text_resources_loaded", False)
    ):
        try:
            lifecycle.release_text_resources()
        except Exception as exc:
            record("lifecycle release", exc)
    runtime_dict = runtime if isinstance(runtime, dict) else None
    if runtime_dict is not None:
        runtime_dict["transformer"] = None
        runtime_dict["optimizer"] = None
        runtime_dict["components"] = None
        runtime_dict["lifecycle"] = None
        runtime_dict["preview_sampler"] = None
        runtime_dict["preview_prompt_paths"] = None
        runtime_dict["preview_negative_paths"] = None
        runtime_dict["cache_paths"] = None
        runtime_dict["accelerator"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if fail_closed and failures:
        raise RuntimeError("rebuild teardown failed: " + "; ".join(failures))


def _maybe_warm_start(
    job_dir: Path,
    transformer: Any,
    adapter_name: str,
    injected: Mapping[str, Any],
    *,
    job: Mapping[str, Any] | None = None,
) -> None:
    loader = (
        injected["load_latest_adapter"]
        if "load_latest_adapter" in injected
        else _default_load_latest_adapter
    )
    if loader is None:
        return
    loaded = loader(job_dir)
    if not loaded:
        return
    if isinstance(loaded, Mapping) and "state_dict" not in loaded:
        # Injected loaders may return a bare state dict for tests.
        state_dict = loaded
        metadata = None
    else:
        state_dict = getattr(loaded, "state_dict", None)
        if state_dict is None and isinstance(loaded, Mapping):
            state_dict = loaded.get("state_dict")
        metadata = getattr(loaded, "metadata", None)
        if metadata is None and isinstance(loaded, Mapping):
            metadata = loaded.get("metadata")
    if not state_dict:
        return
    if job is not None:
        _validate_warm_start_metadata(job, metadata, adapter_name=adapter_name)
    log.info("warm-start adapter")
    _apply_adapter_state(transformer, state_dict, adapter_name, injected)


def _coerce_warm_start_metadata(metadata: Any) -> NativeAdapterMetadata:
    """Require the W1-C sidecar contract; do not invent a second parser."""

    if isinstance(metadata, NativeAdapterMetadata):
        return metadata
    if isinstance(metadata, Mapping):
        try:
            return NativeAdapterMetadata.parse(dict(metadata))
        except (TypeError, ValueError) as exc:
            raise TrainingConfigError(
                f"warm-start checkpoint metadata is invalid: {exc}"
            ) from exc
    raise TrainingConfigError("warm-start checkpoint metadata is missing")


def _validate_warm_start_metadata(
    job: Mapping[str, Any],
    metadata: Any,
    *,
    adapter_name: str,
) -> None:
    """Refuse resume when checkpoint identity disagrees with the job YAML."""

    parsed = _coerce_warm_start_metadata(metadata)
    peft = parsed.peft_config
    lora = job.get("lora") or {}
    main = (job.get("model") or {}).get("main_transformer") or {}
    job_path = str(main.get("path") or "")
    job_revision = main.get("revision")
    mismatches: list[str] = []
    if parsed.adapter_name != adapter_name:
        mismatches.append(
            f"adapter_name checkpoint={parsed.adapter_name!r} job={adapter_name!r}"
        )
    if parsed.base_model_name_or_path != job_path:
        mismatches.append(
            f"base_model checkpoint={parsed.base_model_name_or_path!r} "
            f"job={job_path!r}"
        )
    if parsed.base_model_revision != job_revision:
        mismatches.append(
            f"base_model_revision checkpoint={parsed.base_model_revision!r} "
            f"job={job_revision!r}"
        )
    if int(peft["r"]) != int(lora["rank"]):
        mismatches.append(f"rank checkpoint={peft['r']} job={lora['rank']}")
    if float(peft["lora_alpha"]) != float(lora["alpha"]):
        mismatches.append(
            f"alpha checkpoint={peft['lora_alpha']} job={lora['alpha']}"
        )
    if float(peft["lora_dropout"]) != float(lora["dropout"]):
        mismatches.append(
            f"dropout checkpoint={peft['lora_dropout']} job={lora['dropout']}"
        )
    if list(peft["target_modules"]) != list(lora["targets"]):
        mismatches.append("target_modules differ from job.lora.targets")
    if mismatches:
        raise TrainingConfigError(
            "warm-start checkpoint metadata does not match job LoRA/base: "
            + "; ".join(mismatches)
        )


def _apply_adapter_state(
    transformer: Any,
    state_dict: Mapping[str, Any],
    adapter_name: str,
    injected: Mapping[str, Any],
) -> None:
    setter = injected.get("set_lora_state")
    if setter is not None:
        setter(transformer, state_dict)
        return
    if not hasattr(transformer, "peft_config"):
        raise TrainingConfigError(
            "cannot apply LoRA warm-start: transformer has no peft_config "
            f"(adapter={adapter_name!r})"
        )
    from peft import set_peft_model_state_dict

    compatible = _training_impl("sampling", "_peft_compatible_state")(state_dict)
    result = set_peft_model_state_dict(
        transformer, dict(compatible), adapter_name=adapter_name
    )
    unexpected = list(getattr(result, "unexpected_keys", None) or ())
    missing = list(getattr(result, "missing_keys", None) or ())
    # PEFT reports base-weight keys as missing; only adapter LoRA keys matter.
    missing_lora = [key for key in missing if "lora_" in key.lower()]
    if unexpected or missing_lora:
        raise TrainingConfigError(
            "warm-start LoRA load key mismatch for "
            f"adapter={adapter_name!r}: unexpected={unexpected} "
            f"missing_lora={missing_lora}"
        )


def _lora_state(
    transformer: Any,
    adapter_name: str,
    injected: Mapping[str, Any],
) -> Mapping[str, Any]:
    getter = injected.get("get_lora_state")
    if getter is not None:
        return getter(transformer)
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(transformer, adapter_name=adapter_name)


def _make_optimizer(
    transformer: Any,
    job: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> torch.optim.Optimizer:
    factory = injected.get("optimizer_factory", torch.optim.AdamW)
    params = [param for param in transformer.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("main transformer has no trainable LoRA parameters")
    return factory(
        params,
        lr=float(job["optimizer"]["learning_rate"]),
        weight_decay=float(job["optimizer"]["weight_decay"]),
    )


def _write_running_state(
    job_dir: Path,
    job_id: str,
    step: int,
    epoch: int,
    last_error: str | None,
) -> JobState:
    state = JobState(
        job_id=job_id,
        status=JobStatus.RUNNING,
        step=step,
        epoch=epoch,
        last_error=last_error,
    )
    write_job_state(job_dir, state)
    return state


def _resolve_noise(
    injected: Mapping[str, Any],
    latent: torch.Tensor,
) -> torch.Tensor | None:
    noise = injected.get("noise")
    if noise is None:
        return None
    if callable(noise):
        return noise(latent)
    return noise


def _resolve_training_device(injected: Mapping[str, Any]) -> torch.device:
    """Resolve the device for optimizer training and cache encoding.

    Production jobs omit ``device`` and must run on CUDA. Explicit
    ``device="cpu"`` is an internal unit-test injection path only and is
    not a supported production MVP target. Cache encoding reuses this
    helper when ``encode_sample`` will run; it is not a CPU-only path.
    """

    raw = injected.get("device")
    if raw is None:
        if not torch.cuda.is_available():
            raise TrainingConfigError(
                "training requires CUDA; CPU is unsupported for production MVP"
            )
        return torch.device("cuda")
    device = torch.device(raw)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingConfigError(
                "training requires CUDA; explicit device 'cuda' was requested "
                "but CUDA is unavailable"
            )
        return device
    if device.type == "cpu":
        return device
    raise TrainingConfigError(
        f"unsupported training device {device.type!r}; "
        "production MVP requires CUDA"
    )


def _construct_accelerator(
    device: torch.device,
    injected: Mapping[str, Any],
) -> Any:
    factory = injected.get("accelerator_factory")
    if factory is None:
        from accelerate import Accelerator

        factory = Accelerator
    if device.type == "cpu":
        return factory(
            mixed_precision="no",
            cpu=True,
            gradient_accumulation_steps=1,
        )
    return factory(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
    )


def _bind_accelerator_device(accelerator: Any, device: torch.device) -> Any:
    """Expose ``.device`` on injected fakes that omit it; do not skip validation."""

    if getattr(accelerator, "device", None) is None:
        try:
            accelerator.device = device
        except (AttributeError, TypeError):
            pass
    return accelerator


def _validate_prepared_training_runtime(
    *,
    accelerator: Any,
    transformer: Any,
    setup: Any,
    device: torch.device,
    job: Mapping[str, Any],
    injected: Mapping[str, Any],
    components: Any = None,
) -> None:
    """Fail before the first step if prepare drifted off the target contract."""

    accel_device = getattr(accelerator, "device", None)
    if accel_device is not None and torch.device(accel_device).type != device.type:
        raise TrainingConfigError(
            "accelerator device "
            f"{torch.device(accel_device)} does not match target training "
            f"device {device}"
        )

    requested = str(job.get("precision", "")).strip().lower()
    if (
        requested == "fp8"
        and device.type == "cuda"
        and _fp8_capable(injected)
        and not bool(getattr(setup, "fp8_enabled", False))
    ):
        raise TrainingConfigError(
            "explicit fp8 job on a capable CUDA device cannot disable fp8; "
            "setup.fp8_enabled must stay true on the target Blackwell path"
        )

    adapter_name = getattr(setup, "adapter_name", "default")
    violations: list[str] = []
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.device.type != device.type:
            violations.append(
                f"{name} device={parameter.device} adapter={adapter_name}"
            )
    if violations:
        raise TrainingConfigError(
            "trainable adapter parameters must remain on "
            f"{device} after accelerator.prepare; "
            f"violations: {', '.join(violations)}"
        )

    if device.type == "cuda" and components is not None:
        leftovers: list[str] = []
        for name in ("vae", "text_encoder", "sampling_transformer"):
            module = getattr(components, name, None)
            if module is None:
                continue
            if _reports_cuda(module):
                leftovers.append(f"{name} on CUDA")
        if leftovers:
            raise TrainingConfigError(
                "leftover CUDA VAE, text encoder, or sampling transformer "
                "after training prep; "
                f"violations: {', '.join(leftovers)}"
            )


def _reports_cuda(module: Any) -> bool:
    try:
        reported = getattr(module, "device", None)
        if reported is not None and not callable(reported):
            return torch.device(reported).type == "cuda"
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    try:
        return next(module.parameters()).device.type == "cuda"
    except (AttributeError, StopIteration, TypeError, RuntimeError):
        return False


def _training_device_from_runtime(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> torch.device:
    stored = runtime.get("device")
    if stored is not None:
        return torch.device(stored)
    raw = injected.get("device")
    if raw is not None:
        return torch.device(raw)
    return _resolve_training_device(injected)


def _preview_sampler_device(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> torch.device | str:
    stored = runtime.get("device")
    if stored is not None:
        return stored
    raw = injected.get("device")
    if raw is not None:
        return raw
    return _resolve_training_device(injected)


def _fp8_capable(injected: Mapping[str, Any]) -> bool:
    override = injected.get("fp8_capable")
    if override is not None:
        return bool(override)
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _num_train_timesteps(scheduler: Any) -> int:
    config = getattr(scheduler, "config", None)
    if isinstance(config, Mapping):
        return int(config["num_train_timesteps"])
    return int(config.num_train_timesteps)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "local"


def _bind_gpu_usage_probe(
    job: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve YAML probe settings once. An injected probe still wins."""

    payload = dict(injected)
    if "gpu_usage_settings" not in payload:
        payload["gpu_usage_settings"] = resolve_gpu_usage_settings(job)
    if "gpu_usage_probe" not in payload:
        payload["gpu_usage_probe"] = _default_gpu_usage_probe(
            payload["gpu_usage_settings"]
        )
    return payload


def _reset_gpu_usage_job_peaks() -> None:
    _GPU_USAGE_JOB_PEAKS["step"] = 0
    _GPU_USAGE_JOB_PEAKS["preview"] = 0
    _GPU_USAGE_JOB_PEAKS["nvidia"] = 0


def _gpu_probe_context_from_runtime(
    runtime: Mapping[str, Any] | None,
    *,
    phase_peak_bytes: int | None = None,
    components: Any = None,
    preview_sampler: Any = None,
) -> GpuProbeContext:
    runtime = runtime or {}
    return GpuProbeContext(
        components=(
            components if components is not None else runtime.get("components")
        ),
        optimizer=runtime.get("optimizer"),
        transformer=runtime.get("transformer"),
        phase_peak_bytes=phase_peak_bytes,
        preview_sampler=(
            preview_sampler
            if preview_sampler is not None
            else runtime.get("preview_sampler")
        ),
    )


def _accumulate_gpu_usage_peaks(phase: str, context: Any) -> None:
    peak = 0
    if isinstance(context, GpuProbeContext) and context.phase_peak_bytes is not None:
        peak = int(context.phase_peak_bytes)
    if phase == "step":
        _GPU_USAGE_JOB_PEAKS["step"] = max(_GPU_USAGE_JOB_PEAKS["step"], peak)
    elif phase in _PREVIEW_PROBE_PHASES:
        _GPU_USAGE_JOB_PEAKS["preview"] = max(_GPU_USAGE_JOB_PEAKS["preview"], peak)


def _probe_gpu_usage_summary(
    injected: Mapping[str, Any],
    runtime: Mapping[str, Any] | None,
) -> None:
    context = _gpu_probe_context_from_runtime(
        runtime,
        phase_peak_bytes=max(
            _GPU_USAGE_JOB_PEAKS["step"], _GPU_USAGE_JOB_PEAKS["preview"]
        ),
    )
    try:
        snap = snapshot_gpu_usage("summary", context)
        _GPU_USAGE_JOB_PEAKS["nvidia"] = max(
            _GPU_USAGE_JOB_PEAKS["nvidia"], int(snap.nvidia_used_bytes)
        )
    except Exception:
        pass
    try:
        log.info(
            "gpu usage phase=summary max_step_peak=%s max_preview_peak=%s "
            "max_nvidia_used=%s",
            format_bytes(_GPU_USAGE_JOB_PEAKS["step"]),
            format_bytes(_GPU_USAGE_JOB_PEAKS["preview"]),
            format_bytes(_GPU_USAGE_JOB_PEAKS["nvidia"]),
        )
    except Exception:
        pass
    _probe_gpu_usage(injected, "summary", context=context)


def _probe_gpu_usage(
    injected: Mapping[str, Any],
    phase: str,
    *,
    context: Any = None,
) -> None:
    """Log a child-process GPU snapshot. Never raises into the job."""

    try:
        probe = injected.get("gpu_usage_probe", _DEFAULT_GPU_USAGE_PROBE)
        if probe is None:
            return
        probe(phase, context)
    except Exception:
        pass
    try:
        _accumulate_gpu_usage_peaks(phase, context)
    except Exception:
        return


def _injected_or_default(injected: Mapping[str, Any], key: str, factory):
    if key in injected:
        return injected[key]
    return factory()


def _training_impl(leaf: str, name: str) -> Any:
    module = importlib.import_module("zimage.training." + leaf)
    return getattr(module, name)


def _default_checkpoint_writer() -> CheckpointWriter:
    writer_cls = _training_impl("checkpoints", "NativeLoraCheckpointWriter")
    return writer_cls()


def _default_load_latest_adapter(job_dir: Path) -> Any | None:
    """Return ``LoadedLoraState`` (state_dict + metadata) or ``None``."""

    load_latest = _training_impl("checkpoints", "load_latest_lora_state")
    return load_latest(job_dir)


def _collect_preview_prompt_texts(
    job: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    sampling = job.get("sampling") or {}
    common = sampling_base_parameters(sampling)
    samples = sampling.get("samples") or [{}]
    prompts: list[str] = []
    negatives: list[str] = []
    seen_prompts: set[str] = set()
    seen_negatives: set[str] = set()
    for sample in samples:
        merged = merge_sample_parameters(common, sample)
        prompt = str(merged.get("prompt", ""))
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            prompts.append(prompt)
        negative = str(merged.get("negative_prompt", ""))
        if negative and negative not in seen_negatives:
            seen_negatives.add(negative)
            negatives.append(negative)
    return prompts, negatives


def _unique_preview_prompt_texts(job: Mapping[str, Any]) -> list[str]:
    prompts, negatives = _collect_preview_prompt_texts(job)
    return [text for text in dict.fromkeys((*prompts, *negatives)) if text]


def _preview_prompt_files_need_encode(
    texts: Iterable[str],
    cache_config: CacheConfig,
    job_dir: Path,
) -> bool:
    for text in texts:
        if not text:
            continue
        inspection = inspect_preview_cache(
            preview_cache_path(job_dir, text),
            expected_preview_metadata(text, cache_config),
        )
        if inspection.state is not CacheState.VALID:
            return True
    return False


def _preview_path_maps(
    job: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    prompts, negatives = _collect_preview_prompt_texts(job)
    return (
        {text: paths[text] for text in prompts if text in paths},
        {text: paths[text] for text in negatives if text in paths},
    )


def _assign_preview_prompt_paths(
    runtime: dict[str, Any],
    injected: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    prompt_paths, negative_paths = _preview_path_maps(runtime["config"], paths)
    runtime["preview_prompt_paths"] = prompt_paths
    runtime["preview_negative_paths"] = negative_paths
    sampler = _preview_sampler_from_runtime(runtime, injected)
    if sampler is None:
        return
    if hasattr(sampler, "prompt_paths"):
        sampler.prompt_paths = dict(prompt_paths)
    if hasattr(sampler, "negative_prompt_paths"):
        sampler.negative_prompt_paths = dict(negative_paths)
    sampling = runtime["config"].get("sampling") or {}
    if hasattr(sampler, "common_parameters"):
        sampler.common_parameters = dict(sampling_base_parameters(sampling))


def _prepare_preview_prompt_paths(
    job: Mapping[str, Any],
    lifecycle: TrainingModelLifecycle,
    cache_config: CacheConfig,
    injected: Mapping[str, Any],
    components: TrainingModelComponents,
    *,
    job_dir: Path,
    device: torch.device | str | None = None,
    placed: list[bool] | None = None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    texts = _unique_preview_prompt_texts(job)
    flag = placed if placed is not None else [False]
    if _preview_prompt_files_need_encode(texts, cache_config, job_dir):
        if not flag[0]:
            target = (
                torch.device(device)
                if device is not None
                else _resolve_training_device(injected)
            )
            _place_cache_modules(
                flag, lifecycle, target, injected, components, vae=False
            )
        paths = prepare_preview_prompt_cache(
            texts,
            lifecycle.cache_encoder(),
            cache_config,
            job_dir=job_dir,
        )
    else:
        paths = {text: preview_cache_path(job_dir, text) for text in texts}
    return _preview_path_maps(job, paths)


def _preview_sampler_gpu_usage_probe(injected: Mapping[str, Any]) -> Any:
    """Bind the job probe so sampler ``preview_run`` peaks feed the summary."""

    def probe(phase: str, context: Any = None) -> None:
        _probe_gpu_usage(injected, phase, context=context)

    return probe


def _default_preview_sampler(
    runtime: Mapping[str, Any],
    injected: Mapping[str, Any] | None = None,
) -> PreviewSampler | None:
    injected = injected or {}
    components = runtime.get("components")
    if components is None:
        return None
    transformer = getattr(components, "sampling_transformer", None)
    if transformer is None:
        return None
    sampler_cls = _training_impl("sampling", "UnfusedPreviewSampler")
    config = runtime.get("config") or {}
    kwargs = {
        "transformer": transformer,
        "scheduler": getattr(components, "sampling_scheduler", None),
        "vae": getattr(components, "vae", None),
        "prompt_paths": dict(runtime.get("preview_prompt_paths") or {}),
        "negative_prompt_paths": dict(runtime.get("preview_negative_paths") or {}),
        "common_parameters": sampling_base_parameters(config.get("sampling") or {}),
        "device": _preview_sampler_device(runtime, injected),
        "target_modules": list((config.get("lora") or {}).get("targets") or []),
        "main_transformer": getattr(components, "main_transformer", None),
        "gpu_usage_probe": _preview_sampler_gpu_usage_probe(injected),
    }
    factory = getattr(sampler_cls, "from_components", None)
    if callable(factory):
        try:
            return factory(**kwargs)
        except TypeError:
            pass
    return sampler_cls(**kwargs)
