"""Job YAML schema, training path resolution, and cache tensor constants.

Training paths are read only from the root ``training`` section of
``config.yaml``. Missing or invalid training config must not be consulted
by inference prefs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml

from zimage.config import IMAGE_FORMAT_ALIASES, IMAGE_FORMAT_CHOICES
from zimage.training.contracts import UpdateClassification

TRAINING_SECTION = "training"

KNOWN_TURBO_SOURCE = "Tongyi-MAI/Z-Image-Turbo"
KNOWN_MAIN_SOURCE = "Tongyi-MAI/Z-Image"

TRAINING_PRECISION_CHOICES = frozenset({"fp8", "bf16"})
WEIGHTING_SCHEME_CHOICES = frozenset(
    {"none", "sigma_sqrt", "logit_normal", "mode", "cosmap"}
)
SCHEDULER_NAME_CHOICES = frozenset({"constant"})

# No job YAML fields are locked; cache identity lives in IMMUTABLE_CACHE_FIELDS.
IMMUTABLE_JOB_FIELDS: frozenset[str] = frozenset()

# Cache tensor schema (versioned). Latent is BF16 [16, H/8, W/8].
# Prompt embedding is BF16 [valid_tokens, 2560] with no padding.
CACHE_TENSOR_SCHEMA_VERSION = 1
CACHE_LATENT_DTYPE = "bf16"
CACHE_LATENT_CHANNELS = 16
CACHE_LATENT_SPATIAL_DIVISOR = 8
CACHE_PROMPT_EMBED_DTYPE = "bf16"
CACHE_PROMPT_EMBED_HIDDEN_SIZE = 2560
CACHE_PROMPT_EMBED_PADDED = False

# These are conceptual cache metadata paths rather than job-YAML keys.  Keeping
# them alongside the immutable job paths gives update/cache agents one complete
# list of values that define model and cache compatibility.
IMMUTABLE_CACHE_FIELDS: frozenset[str] = frozenset(
    {
        "cache.tensor_schema",
        "cache.schema_version",
    }
)
IMMUTABLE_MVP_FIELDS: frozenset[str] = (
    IMMUTABLE_JOB_FIELDS | IMMUTABLE_CACHE_FIELDS
)

REBUILD_REQUIRED_JOB_FIELDS: frozenset[str] = frozenset(
    {
        "datasets",
        "model.main_transformer",
        "model.sampling_transformer",
        "precision",
        "gradient_checkpointing",
        "lora",
        "max_sequence_length",
        "optimizer.name",
    }
)

JOB_TOP_LEVEL_KEYS = frozenset(
    {
        "job_name",
        "model",
        "datasets",
        "lora",
        "precision",
        "gradient_checkpointing",
        "seed",
        "epochs",
        "max_steps",
        "checkpoint_every",
        "optimizer",
        "scheduler",
        "weighting_scheme",
        "logit_mean",
        "logit_std",
        "mode_scale",
        "max_sequence_length",
        "sampling",
        "debug",
    }
)

MODEL_KEYS = frozenset({"main_transformer", "sampling_transformer"})
TRANSFORMER_KEYS = frozenset({"path", "revision"})
DATASET_ITEM_KEYS = frozenset({"name", "default_caption"})
LORA_KEYS = frozenset({"rank", "alpha", "dropout", "targets"})
OPTIMIZER_KEYS = frozenset({"name", "learning_rate", "weight_decay"})
SCHEDULER_KEYS = frozenset({"name", "warmup_steps"})
SAMPLING_PARAMETER_KEYS = frozenset(
    {
        "guidance_scale",
        "time_shift",
        "num_inference_steps",
        "width",
        "height",
        "seed",
        "prompt",
        "negative_prompt",
    }
)
SAMPLING_BLOCK_KEYS = SAMPLING_PARAMETER_KEYS | {"samples", "image_format"}
DEBUG_KEYS = frozenset({"detailed"})


class TrainingConfigError(ValueError):
    """Invalid training config.yaml section or job YAML document."""


@dataclass(frozen=True)
class TrainingPaths:
    """Raw path strings from the ``training`` YAML section (not resolved)."""

    datasets_dir: str
    jobs_dir: str


@dataclass(frozen=True)
class GpuUsageSettings:
    """Resolved GPU probe toggles. Defaults apply when YAML keys are absent."""

    detailed: bool = False


@dataclass(frozen=True)
class CacheLatentSpec:
    """BF16 latent layout: ``[16, H/8, W/8]``."""

    dtype: str = CACHE_LATENT_DTYPE
    channels: int = CACHE_LATENT_CHANNELS
    spatial_divisor: int = CACHE_LATENT_SPATIAL_DIVISOR

    @property
    def shape_description(self) -> str:
        return "[16, H/8, W/8]"


@dataclass(frozen=True)
class CachePromptEmbedSpec:
    """BF16 prompt embedding: ``[valid_tokens, 2560]`` without padding."""

    dtype: str = CACHE_PROMPT_EMBED_DTYPE
    hidden_size: int = CACHE_PROMPT_EMBED_HIDDEN_SIZE
    padded: bool = CACHE_PROMPT_EMBED_PADDED

    @property
    def shape_description(self) -> str:
        return "[valid_tokens, 2560]"


@dataclass(frozen=True)
class CacheTensorSchema:
    """Versioned cache tensor contract for later cache writers."""

    version: int
    latent: CacheLatentSpec
    prompt_embedding: CachePromptEmbedSpec


CACHE_TENSOR_SCHEMA = CacheTensorSchema(
    version=CACHE_TENSOR_SCHEMA_VERSION,
    latent=CacheLatentSpec(),
    prompt_embedding=CachePromptEmbedSpec(),
)


def is_immutable_job_field(dotted_path: str) -> bool:
    """Return True if ``dotted_path`` is locked for the MVP job lifetime."""
    return dotted_path in IMMUTABLE_JOB_FIELDS


def is_immutable_mvp_field(dotted_path: str) -> bool:
    """Return True for immutable job or cache-compatibility fields."""
    return dotted_path in IMMUTABLE_MVP_FIELDS


def classify_job_update(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[UpdateClassification, tuple[str, ...]]:
    """Validate and classify a candidate job config for step-boundary use."""

    current_valid = validate_job_document(current)
    candidate_valid = validate_job_document(candidate)
    changed_fields = tuple(sorted(_changed_paths(current_valid, candidate_valid)))
    if not changed_fields:
        return UpdateClassification.NO_CHANGE, ()
    if any(
        _matches_field(path, immutable)
        for path in changed_fields
        for immutable in IMMUTABLE_JOB_FIELDS
    ):
        return UpdateClassification.REJECTED_IMMUTABLE, changed_fields
    if any(
        _matches_field(path, rebuild)
        for path in changed_fields
        for rebuild in REBUILD_REQUIRED_JOB_FIELDS
    ):
        return UpdateClassification.REBUILD_REQUIRED, changed_fields
    return UpdateClassification.APPLY_AT_STEP, changed_fields


def resolve_training_paths() -> TrainingPaths:
    """Read ``datasets_dir`` / ``jobs_dir`` from the root ``training`` section.

    If the ``training`` key is entirely absent, atomically write::

        training:
          datasets_dir: ./datasets
          jobs_dir: ./jobs

    then re-read the document. Those two strings appear only in that write
    payload — they are never used as silent read-time fallbacks.

    If ``training`` exists but the path fields are missing, empty, or not
    strings, raise ``TrainingConfigError``.
    """
    # Import lazily so ``import zimage.training`` does not execute the prefs
    # package (which currently reaches inference helpers and Torch).
    from zimage.prefs import store

    doc = store.load_document()
    if TRAINING_SECTION not in doc:
        store.update_section(
            TRAINING_SECTION,
            {
                "datasets_dir": "./datasets",
                "jobs_dir": "./jobs",
            },
        )
        doc = store.load_document()
    return _parse_training_paths(doc.get(TRAINING_SECTION))


# GPU probe toggles live in root config.yaml (training.debug) and
# job config.yaml (debug). Job keys override root. No env vars
# (ZIMAGE_*) and no parallel config source.
def resolve_gpu_usage_settings(job: Mapping[str, Any]) -> GpuUsageSettings:
    """Merge root ``training.debug`` with job ``debug``.

    Absent keys use ``GpuUsageSettings`` defaults. Unknown keys raise
    ``TrainingConfigError``. This function does not write ``config.yaml``.
    """
    # Import lazily so ``import zimage.training`` does not execute the prefs
    # package (which currently reaches inference helpers and Torch).
    from zimage.prefs import store

    doc = store.load_document()
    training = doc.get(TRAINING_SECTION)
    root_raw = training.get("debug") if isinstance(training, Mapping) else None
    job_raw = job.get("debug") if isinstance(job, Mapping) else None
    merged = {
        **_parse_debug_section(root_raw, "training.debug"),
        **_parse_debug_section(job_raw, "debug"),
    }
    return GpuUsageSettings(
        detailed=merged.get("detailed", False),
    )


def _parse_debug_section(raw: Any, label: str) -> dict[str, bool]:
    if raw is None:
        return {}
    data = _require_mapping(raw, label)
    _reject_unknown(data, DEBUG_KEYS, label)
    parsed: dict[str, bool] = {}
    if "detailed" in data:
        parsed["detailed"] = _require_bool(data["detailed"], f"{label}.detailed")
    return parsed


def _parse_training_paths(section: Any) -> TrainingPaths:
    if not isinstance(section, dict):
        raise TrainingConfigError("training section must be a mapping")
    datasets_dir = section.get("datasets_dir")
    jobs_dir = section.get("jobs_dir")
    if not isinstance(datasets_dir, str) or not datasets_dir.strip():
        raise TrainingConfigError(
            "training.datasets_dir is missing, empty, or invalid"
        )
    if not isinstance(jobs_dir, str) or not jobs_dir.strip():
        raise TrainingConfigError(
            "training.jobs_dir is missing, empty, or invalid"
        )
    return TrainingPaths(
        datasets_dir=datasets_dir.strip(),
        jobs_dir=jobs_dir.strip(),
    )


def job_create_template() -> dict[str, Any]:
    """Full default job document used by Create (empty datasets, one sample)."""
    return {
        "job_name": "Мой стиль",
        "model": {
            "main_transformer": {
                "path": KNOWN_MAIN_SOURCE,
                "revision": None,
            },
            "sampling_transformer": {
                "path": KNOWN_TURBO_SOURCE,
                "revision": None,
            },
        },
        "datasets": [],
        "lora": {
            "rank": 4,
            "alpha": 4,
            "dropout": 0.0,
            "targets": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "precision": "fp8",
        "gradient_checkpointing": True,
        "seed": 0,
        "epochs": 1,
        "max_steps": 500,
        "checkpoint_every": 100,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
        },
        "scheduler": {
            "name": "constant",
            "warmup_steps": 0,
        },
        "weighting_scheme": "none",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "mode_scale": 1.29,
        "max_sequence_length": 512,
        "sampling": {
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "time_shift": 3.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "prompt": "",
            "negative_prompt": "",
            "image_format": "jpeg",
            "samples": [
                {"prompt": "a photo of a dog"},
            ],
        },
    }


def load_job_document(path: str | Path) -> dict[str, Any]:
    """Load a job YAML file and validate it."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingConfigError(f"cannot read job YAML: {target}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"invalid job YAML: {target}") from exc
    return validate_job_document(raw)


def load_job_document_for_classify(path: str | Path) -> dict[str, Any] | None:
    """Load the on-disk job for update classification, or ``None`` to skip it.

    A valid candidate must still be able to replace a legacy or otherwise
    invalid current file. Top-level transformer keys are nested under
    ``model`` for comparison only so immutable-field checks still apply.
    """
    target = Path(path)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    for document in (raw, _nest_legacy_transformers(raw)):
        try:
            return validate_job_document(document)
        except (TrainingConfigError, TypeError, ValueError):
            continue
    return None


def _nest_legacy_transformers(raw: Any) -> Any:
    """Copy top-level transformer keys under ``model`` without mutating ``raw``."""
    if not isinstance(raw, Mapping):
        return raw
    if "main_transformer" not in raw and "sampling_transformer" not in raw:
        return raw
    nested = dict(raw)
    model = dict(nested["model"]) if isinstance(nested.get("model"), Mapping) else {}
    for key in ("main_transformer", "sampling_transformer"):
        if key in nested:
            value = nested.pop(key)
            if key not in model:
                model[key] = value
    nested["model"] = model
    return nested


def validate_job_document(raw: Any) -> dict[str, Any]:
    """Validate job YAML structure and types. Empty ``datasets`` is allowed."""
    data = _require_mapping(raw, "job")
    legacy = [
        key for key in ("main_transformer", "sampling_transformer") if key in data
    ]
    if legacy:
        raise TrainingConfigError(
            f"{' and '.join(legacy)} must be nested under model"
        )
    _reject_unknown(data, JOB_TOP_LEVEL_KEYS, "job")
    defaults = job_create_template()

    if "job_name" not in data:
        raise TrainingConfigError("job.job_name is required")
    if "model" not in data:
        raise TrainingConfigError("job.model is required")

    model = _require_mapping(data["model"], "model")
    _reject_unknown(model, MODEL_KEYS, "model")
    if "main_transformer" not in model:
        raise TrainingConfigError("model.main_transformer is required")

    model_out: dict[str, Any] = {
        "main_transformer": _validate_transformer(
            model["main_transformer"], "model.main_transformer"
        ),
    }
    if "sampling_transformer" in model:
        model_out["sampling_transformer"] = _validate_transformer(
            model["sampling_transformer"], "model.sampling_transformer"
        )

    out: dict[str, Any] = {
        "job_name": _require_nonempty_str(data.get("job_name"), "job_name"),
        "model": model_out,
    }

    out["datasets"] = _validate_datasets(_present(data, "datasets", defaults["datasets"]))
    out["lora"] = _validate_lora(_present(data, "lora", defaults["lora"]))
    out["precision"] = _require_choice(
        _present(data, "precision", defaults["precision"]),
        TRAINING_PRECISION_CHOICES,
        "precision",
    )
    out["gradient_checkpointing"] = _require_bool(
        _present(data, "gradient_checkpointing", defaults["gradient_checkpointing"]),
        "gradient_checkpointing",
    )
    out["seed"] = _require_int(_present(data, "seed", defaults["seed"]), "seed")
    out["epochs"] = _optional_positive_int(data, "epochs", defaults["epochs"])
    out["max_steps"] = _optional_positive_int(data, "max_steps", defaults["max_steps"])
    if out["epochs"] is None and out["max_steps"] is None:
        raise TrainingConfigError("job must set epochs or max_steps")
    out["checkpoint_every"] = _require_int(
        _present(data, "checkpoint_every", defaults["checkpoint_every"]),
        "checkpoint_every",
        min_value=1,
    )
    out["optimizer"] = _validate_optimizer(
        _present(data, "optimizer", defaults["optimizer"])
    )
    out["scheduler"] = _validate_scheduler(
        _present(data, "scheduler", defaults["scheduler"])
    )
    out["weighting_scheme"] = _require_choice(
        _present(data, "weighting_scheme", defaults["weighting_scheme"]),
        WEIGHTING_SCHEME_CHOICES,
        "weighting_scheme",
    )
    out["logit_mean"] = _require_float(
        _present(data, "logit_mean", defaults["logit_mean"]), "logit_mean"
    )
    out["logit_std"] = _require_float(
        _present(data, "logit_std", defaults["logit_std"]), "logit_std"
    )
    out["mode_scale"] = _require_float(
        _present(data, "mode_scale", defaults["mode_scale"]), "mode_scale"
    )
    out["max_sequence_length"] = _require_int(
        _present(data, "max_sequence_length", defaults["max_sequence_length"]),
        "max_sequence_length",
        min_value=1,
    )
    out["sampling"] = _validate_sampling(
        _present(data, "sampling", defaults["sampling"])
    )
    if "debug" in data:
        out["debug"] = _parse_debug_section(data["debug"], "debug")
    return out


def resolve_stop_condition(job: Mapping[str, Any]) -> tuple[str, int]:
    """Return the binding train-stop rule.

    When both ``epochs`` and ``max_steps`` are set, ``max_steps`` wins.
    """
    max_steps = job.get("max_steps")
    epochs = job.get("epochs")
    if max_steps is not None:
        return ("max_steps", int(max_steps))
    if epochs is not None:
        return ("epochs", int(epochs))
    raise TrainingConfigError("job must set epochs or max_steps")


def sampling_base_parameters(sampling: Mapping[str, Any]) -> dict[str, Any]:
    """Return the eight Diffusers keys present on ``sampling`` (partial OK)."""
    data = _require_mapping(sampling, "sampling")
    return {key: data[key] for key in SAMPLING_PARAMETER_KEYS if key in data}


def merge_sample_parameters(
    common: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a per-sample dict over root sampling parameters (sample wins)."""
    common_map = _require_mapping(common, "sampling")
    sample_map = _require_mapping(sample, "sampling.samples[]")
    _reject_unknown(common_map, SAMPLING_PARAMETER_KEYS, "sampling")
    _reject_unknown(sample_map, SAMPLING_PARAMETER_KEYS, "sampling.samples[]")
    base = _coerce_sampling_params(
        {**_sampling_defaults(), **dict(common_map)},
        "sampling",
    )
    overlay = _coerce_sampling_params(sample_map, "sampling.samples[]", partial=True)
    return {**base, **overlay}


def _sampling_defaults() -> dict[str, Any]:
    sampling = job_create_template()["sampling"]
    return {key: sampling[key] for key in SAMPLING_PARAMETER_KEYS}


def _changed_paths(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    prefix: str = "",
) -> set[str]:
    changed: set[str] = set()
    for key in set(current) | set(candidate):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in current or key not in candidate:
            changed.add(path)
            continue
        before = current[key]
        after = candidate[key]
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            changed.update(_changed_paths(before, after, path))
        elif before != after:
            changed.add(path)
    return changed


def _matches_field(path: str, field: str) -> bool:
    return path == field or path.startswith(f"{field}.")


def _present(data: Mapping[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default


def _optional_positive_int(
    data: Mapping[str, Any], key: str, default: int
) -> int | None:
    if key not in data:
        return int(default)
    value = data[key]
    if value is None:
        return None
    return _require_int(value, key, min_value=1)


def _validate_transformer(raw: Any, label: str) -> dict[str, Any]:
    data = _require_mapping(raw, label)
    _reject_unknown(data, TRANSFORMER_KEYS, label)
    if "path" not in data:
        raise TrainingConfigError(f"{label}.path is required")
    path = _require_nonempty_str(data.get("path"), f"{label}.path")
    if label == "model.main_transformer" and _is_turbo_source(path):
        raise TrainingConfigError(
            f"{KNOWN_TURBO_SOURCE} cannot be used as {label}"
        )
    revision: str | None
    if "revision" not in data or data.get("revision") is None:
        revision = None
    else:
        revision = _require_nonempty_str(data.get("revision"), f"{label}.revision")
    return {"path": path, "revision": revision}


def _is_turbo_source(path: str) -> bool:
    normalized = path.strip().replace("\\", "/").rstrip("/").casefold()
    for prefix in (
        "hf://",
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://www.huggingface.co/",
        "http://www.huggingface.co/",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.split("/resolve/", 1)[0]
    if normalized == KNOWN_TURBO_SOURCE.casefold():
        return True
    return "models--tongyi-mai--z-image-turbo/" in normalized


def _validate_datasets(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise TrainingConfigError("datasets must be a list")
    items: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        label = f"datasets[{index}]"
        data = _require_mapping(item, label)
        _reject_unknown(data, DATASET_ITEM_KEYS, label)
        if "name" not in data:
            raise TrainingConfigError(f"{label}.name is required")
        name = _require_nonempty_str(data.get("name"), f"{label}.name")
        caption = data.get("default_caption", "")
        if not isinstance(caption, str):
            raise TrainingConfigError(f"{label}.default_caption must be a string")
        items.append({"name": name, "default_caption": caption})
    return items


def _validate_lora(raw: Any) -> dict[str, Any]:
    data = _require_mapping(raw, "lora")
    _reject_unknown(data, LORA_KEYS, "lora")
    base = job_create_template()["lora"]
    merged = {**base, **data}
    targets_raw = merged["targets"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise TrainingConfigError("lora.targets must be a non-empty list")
    targets = [
        _require_nonempty_str(item, f"lora.targets[{index}]")
        for index, item in enumerate(targets_raw)
    ]
    dropout = _require_float(merged["dropout"], "lora.dropout")
    if dropout < 0.0 or dropout > 1.0:
        raise TrainingConfigError("lora.dropout must be between 0 and 1")
    alpha = _require_number(merged["alpha"], "lora.alpha")
    if alpha <= 0:
        raise TrainingConfigError("lora.alpha must be > 0")
    return {
        "rank": _require_int(merged["rank"], "lora.rank", min_value=1),
        "alpha": alpha,
        "dropout": dropout,
        "targets": targets,
    }


def _validate_optimizer(raw: Any) -> dict[str, Any]:
    data = _require_mapping(raw, "optimizer")
    _reject_unknown(data, OPTIMIZER_KEYS, "optimizer")
    base = job_create_template()["optimizer"]
    merged = {**base, **data}
    learning_rate = _require_float(
        merged["learning_rate"], "optimizer.learning_rate"
    )
    weight_decay = _require_float(
        merged["weight_decay"], "optimizer.weight_decay"
    )
    if learning_rate <= 0:
        raise TrainingConfigError("optimizer.learning_rate must be > 0")
    if weight_decay < 0:
        raise TrainingConfigError("optimizer.weight_decay must be >= 0")
    return {
        "name": _require_nonempty_str(merged["name"], "optimizer.name"),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def _validate_scheduler(raw: Any) -> dict[str, Any]:
    data = _require_mapping(raw, "scheduler")
    _reject_unknown(data, SCHEDULER_KEYS, "scheduler")
    base = job_create_template()["scheduler"]
    merged = {**base, **data}
    return {
        "name": _require_choice(merged["name"], SCHEDULER_NAME_CHOICES, "scheduler.name"),
        "warmup_steps": _require_int(
            merged["warmup_steps"], "scheduler.warmup_steps", min_value=0
        ),
    }


def _validate_sampling(raw: Any) -> dict[str, Any]:
    data = _require_mapping(raw, "sampling")
    _reject_unknown(data, SAMPLING_BLOCK_KEYS, "sampling")
    defaults = job_create_template()["sampling"]
    samples_raw = _present(data, "samples", defaults["samples"])
    params = _coerce_sampling_params(
        {
            **_sampling_defaults(),
            **{key: data[key] for key in SAMPLING_PARAMETER_KEYS if key in data},
        },
        "sampling",
    )
    if not isinstance(samples_raw, list):
        raise TrainingConfigError("sampling.samples must be a list")
    samples = [_validate_sample(item, index) for index, item in enumerate(samples_raw)]
    raw_fmt = _present(data, "image_format", defaults["image_format"])
    if isinstance(raw_fmt, str):
        raw_fmt = IMAGE_FORMAT_ALIASES.get(raw_fmt, raw_fmt)
    fmt = _require_choice(
        raw_fmt, frozenset(IMAGE_FORMAT_CHOICES), "sampling.image_format"
    )
    return {**params, "image_format": fmt, "samples": samples}


def _validate_sample(raw: Any, index: int) -> dict[str, Any]:
    label = f"sampling.samples[{index}]"
    data = _require_mapping(raw, label)
    _reject_unknown(data, SAMPLING_PARAMETER_KEYS, label)
    return _coerce_sampling_params(data, label, partial=True)


def _coerce_sampling_params(
    raw: Mapping[str, Any],
    label: str,
    *,
    partial: bool = False,
) -> dict[str, Any]:
    _reject_unknown(raw, SAMPLING_PARAMETER_KEYS, label)
    if not partial:
        missing = SAMPLING_PARAMETER_KEYS - set(raw)
        if missing:
            raise TrainingConfigError(f"{label} missing keys: {sorted(missing)}")
    out: dict[str, Any] = {}
    if "num_inference_steps" in raw:
        out["num_inference_steps"] = _require_int(
            raw["num_inference_steps"], f"{label}.num_inference_steps", min_value=1
        )
    if "guidance_scale" in raw:
        out["guidance_scale"] = _require_float(
            raw["guidance_scale"], f"{label}.guidance_scale"
        )
    if "time_shift" in raw:
        out["time_shift"] = _require_float(raw["time_shift"], f"{label}.time_shift")
    if "width" in raw:
        out["width"] = _require_int(raw["width"], f"{label}.width", min_value=1)
    if "height" in raw:
        out["height"] = _require_int(raw["height"], f"{label}.height", min_value=1)
    if "seed" in raw:
        out["seed"] = _require_int(raw["seed"], f"{label}.seed")
    if "prompt" in raw:
        out["prompt"] = _require_str(raw["prompt"], f"{label}.prompt")
    if "negative_prompt" in raw:
        out["negative_prompt"] = _require_str(
            raw["negative_prompt"], f"{label}.negative_prompt"
        )
    return out


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{label} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TrainingConfigError(f"{label} has unknown keys: {unknown}")


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigError(f"{label} must be a string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise TrainingConfigError(f"{label} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        raise TrainingConfigError(f"{label} must be an integer")
    if min_value is not None and number < min_value:
        raise TrainingConfigError(f"{label} must be >= {min_value}")
    return number


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TrainingConfigError(f"{label} must be finite")
    return number


def _require_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{label} must be a number")
    if not math.isfinite(value):
        raise TrainingConfigError(f"{label} must be finite")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _require_choice(value: Any, allowed: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise TrainingConfigError(f"{label} must be one of {sorted(allowed)}")
    return value
