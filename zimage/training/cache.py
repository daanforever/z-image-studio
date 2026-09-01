"""Versioned safetensors cache for captioned Z-Image training samples.

Encoding is supplied by the caller through :class:`CacheEncoder`; this module
never discovers or loads production model classes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from zimage.training.dataset import DatasetSample, load_training_image
from zimage.training.schema import (
    CACHE_LATENT_CHANNELS,
    CACHE_LATENT_SPATIAL_DIVISOR,
    CACHE_PROMPT_EMBED_HIDDEN_SIZE,
    CACHE_TENSOR_SCHEMA_VERSION,
)

log = logging.getLogger("zimage.training")

LATENT_KEY = "latent"
PROMPT_EMBEDDING_KEY = "prompt_embedding"
METADATA_KEY = "zimage_training_cache"
JOB_CACHE_DIRECTORY = ".cache"
DATASET_CACHE_NAMESPACE = "dataset"
PREVIEW_CACHE_NAMESPACE = "preview"
PREVIEW_CACHE_KIND = "preview"

DEFAULT_PREPROCESSING: dict[str, Any] = {
    "exif_orientation": "applied",
    "alpha_composite": "white",
    "color_mode": "RGB",
    "resize": None,
    "crop": "center_to_multiple",
    "pad": None,
}


class CacheError(ValueError):
    """A cache request, tensor, or file violates the training cache contract."""


@runtime_checkable
class LatentDistribution(Protocol):
    """Minimal VAE distribution surface needed by this cache."""

    def mode(self) -> torch.Tensor:
        """Return the deterministic, unscaled latent mode."""
        ...


@runtime_checkable
class CacheEncoder(Protocol):
    """Lightweight injected encoder; implementations own all model details."""

    def encode_image(self, image: Image.Image) -> LatentDistribution | Any:
        """Return a latent distribution, or an object with ``latent_dist``."""
        ...

    def encode_prompt(
        self,
        caption: str,
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Return unpadded valid-token embeddings."""
        ...


@dataclass(frozen=True)
class CacheConfig:
    """All model/configuration values that affect cached tensor compatibility."""

    main_revision: str
    vae_config: Mapping[str, Any]
    text_encoder_config: Mapping[str, Any]
    tokenizer_config: Mapping[str, Any]
    qwen_chat_template: Mapping[str, Any]
    max_sequence_length: int
    preprocessing: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_PREPROCESSING)
    )
    schema_version: int = CACHE_TENSOR_SCHEMA_VERSION
    text_encoder_precision: str = "bf16"

    def __post_init__(self) -> None:
        if not isinstance(self.main_revision, str) or not self.main_revision.strip():
            raise CacheError("main_revision must be a resolved, non-empty revision")
        if not isinstance(self.max_sequence_length, int) or isinstance(
            self.max_sequence_length, bool
        ) or self.max_sequence_length < 1:
            raise CacheError("max_sequence_length must be a positive integer")
        if self.schema_version != CACHE_TENSOR_SCHEMA_VERSION:
            raise CacheError(
                f"unsupported cache schema version: {self.schema_version}"
            )
        for name in (
            "vae_config",
            "text_encoder_config",
            "tokenizer_config",
            "qwen_chat_template",
            "preprocessing",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise CacheError(f"{name} must be a mapping")


@dataclass(frozen=True)
class CachedSample:
    """Validated tensors and metadata loaded from one cache file."""

    latent: torch.Tensor
    prompt_embedding: torch.Tensor
    metadata: dict[str, Any]
    path: Path


class CacheState(str, Enum):
    """Result of an explicit cache-validity pass."""

    MISSING = "missing"
    VALID = "valid"
    STALE = "stale"


@dataclass(frozen=True)
class CacheInspection:
    """Validity decision made at a caller-controlled synchronization point."""

    path: Path
    state: CacheState
    expected_metadata: dict[str, Any]
    reason: str | None = None


def job_cache_path(job_dir: str | Path, namespace: str, key: str) -> Path:
    """Return ``{job_dir}/.cache/{namespace}/{key}.safetensors``."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise CacheError("cache namespace must be a non-empty string")
    if not isinstance(key, str) or not key.strip():
        raise CacheError("cache key must be a non-empty string")
    if any(sep in namespace or sep in key for sep in ("/", "\\")):
        raise CacheError("cache namespace and key must be single path components")
    return Path(job_dir) / JOB_CACHE_DIRECTORY / namespace / f"{key}.safetensors"


def sample_cache_key(sample: DatasetSample) -> str:
    """SHA-256 of the existing image-bytes and caption fingerprints."""

    return hashlib.sha256(
        bytes.fromhex(fingerprint_file(sample.image_path))
        + bytes.fromhex(fingerprint_caption(sample.caption))
    ).hexdigest()


def cache_path_for(sample: DatasetSample, job_dir: str | Path) -> Path:
    """Return the job-local dataset cache path for one sample."""

    return job_cache_path(job_dir, DATASET_CACHE_NAMESPACE, sample_cache_key(sample))


def fingerprint_file(path: str | Path) -> str:
    """SHA-256 fingerprint the exact image bytes."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CacheError(f"cannot fingerprint image: {path}") from exc
    return digest.hexdigest()


def fingerprint_caption(caption: str) -> str:
    """SHA-256 fingerprint the resolved UTF-8 caption."""

    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def expected_metadata(
    sample: DatasetSample,
    config: CacheConfig,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build the complete cache identity for one sample."""

    width, height = image_size
    preprocessing = dict(config.preprocessing)
    preprocessing.update({"width": width, "height": height})
    metadata = {
        "image_fingerprint": fingerprint_file(sample.image_path),
        "caption_fingerprint": fingerprint_caption(sample.caption),
        "main_revision": config.main_revision,
        "vae_config": dict(config.vae_config),
        "text_encoder_config": dict(config.text_encoder_config),
        "text_encoder_precision": config.text_encoder_precision,
        "tokenizer_config": dict(config.tokenizer_config),
        "qwen_chat_template": dict(config.qwen_chat_template),
        "max_sequence_length": config.max_sequence_length,
        "preprocessing": preprocessing,
        "schema_version": config.schema_version,
    }
    _canonical_json(metadata)
    return metadata


def inspect_cache(
    path: str | Path,
    expected: Mapping[str, Any],
) -> CacheInspection:
    """Explicitly classify one cache as missing, valid, or stale."""

    target = Path(path)
    expected_dict = _normalized_metadata(expected)
    if not target.is_file():
        return CacheInspection(target, CacheState.MISSING, expected_dict)
    try:
        loaded = load_cache(target)
    except (CacheError, OSError, ValueError) as exc:
        return CacheInspection(
            target,
            CacheState.STALE,
            expected_dict,
            f"invalid cache: {exc}",
        )
    if loaded.metadata != expected_dict:
        return CacheInspection(
            target,
            CacheState.STALE,
            expected_dict,
            "cache metadata differs",
        )
    return CacheInspection(target, CacheState.VALID, expected_dict)


def is_cache_valid(path: str | Path, expected: Mapping[str, Any]) -> bool:
    """Return validity only when explicitly called by the job/command owner."""

    return inspect_cache(path, expected).state is CacheState.VALID


def encode_sample(
    sample: DatasetSample,
    image: Image.Image,
    encoder: CacheEncoder,
    config: CacheConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one already-validated image through the injected encoder."""

    with torch.no_grad():
        encoded_image = encoder.encode_image(image)
        distribution = getattr(encoded_image, "latent_dist", encoded_image)
        mode = getattr(distribution, "mode", None)
        if not callable(mode):
            raise CacheError("encode_image must return a latent distribution")
        raw_latent = mode()
        if not isinstance(raw_latent, torch.Tensor):
            raise CacheError("latent_dist.mode() must return a tensor")
        shift_factor = _vae_number(config.vae_config, "shift_factor")
        scaling_factor = _vae_number(config.vae_config, "scaling_factor")
        cuda_latent = (raw_latent - shift_factor) * scaling_factor
        cuda_latent = _remove_single_batch(cuda_latent, "latent")
        latent = (
            cuda_latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        )
        del encoded_image, distribution, mode, raw_latent, cuda_latent
    prompt_embedding = encode_prompt_embedding(
        sample.caption,
        encoder,
        config,
    )
    validate_cache_tensors(
        latent,
        prompt_embedding,
        image_size=image.size,
        max_sequence_length=config.max_sequence_length,
    )
    return latent, prompt_embedding


def write_cache_atomic(
    destination: str | Path,
    latent: torch.Tensor,
    prompt_embedding: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    """Validate and atomically replace one cache using a same-dir temp file."""

    normalized = _normalized_metadata(metadata)
    image_size = _metadata_image_size(normalized)
    max_sequence_length = normalized.get("max_sequence_length")
    if not isinstance(max_sequence_length, int):
        raise CacheError("cache metadata max_sequence_length is invalid")
    validate_cache_tensors(
        latent,
        prompt_embedding,
        image_size=image_size,
        max_sequence_length=max_sequence_length,
    )
    return _write_tensors_atomic(
        destination,
        {
            LATENT_KEY: latent,
            PROMPT_EMBEDDING_KEY: prompt_embedding,
        },
        normalized,
    )


def load_cache(
    path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> CachedSample:
    """Load one cache and validate metadata, shape, and dtype."""

    target = Path(path)
    tensors, metadata = _read_tensors_cpu(target)
    required = {LATENT_KEY, PROMPT_EMBEDDING_KEY}
    if set(tensors) != required:
        raise CacheError(f"cache tensor keys must be {sorted(required)}")
    latent = tensors[LATENT_KEY]
    prompt_embedding = tensors[PROMPT_EMBEDDING_KEY]
    image_size = _metadata_image_size(metadata)
    max_sequence_length = metadata.get("max_sequence_length")
    if not isinstance(max_sequence_length, int):
        raise CacheError("cache metadata max_sequence_length is invalid")
    validate_cache_tensors(
        latent,
        prompt_embedding,
        image_size=image_size,
        max_sequence_length=max_sequence_length,
    )
    if expected is not None and metadata != _normalized_metadata(expected):
        raise CacheError("cache metadata differs")
    return CachedSample(latent, prompt_embedding, metadata, target)


def prepare_cache_at_job_start(
    samples: list[DatasetSample],
    encoder: CacheEncoder,
    config: CacheConfig,
    *,
    job_dir: str | Path,
    on_before_encode: Callable[[], None] | None = None,
    on_before_sample_encode: (
        Callable[[DatasetSample, tuple[int, int]], None] | None
    ) = None,
    on_after_sample_encode: (
        Callable[[DatasetSample, tuple[int, int]], None] | None
    ) = None,
) -> list[Path]:
    """Inspect and materialize all caches at an explicit job-start boundary.

    ``on_before_encode`` runs once, immediately before the first
    ``encode_sample``. ``on_before_sample_encode`` and
    ``on_after_sample_encode`` run for every stale pair; the after hook
    runs in a ``finally`` after ``encode_sample`` returns or raises.
    VALID caches never invoke these hooks.
    """

    paths: list[Path] = []
    encode_hook = on_before_encode
    encoded = 0
    reused = 0
    for sample in samples:
        image = load_training_image(sample.image_path)
        image_size = image.size
        metadata = expected_metadata(sample, config, image_size=image_size)
        path = cache_path_for(sample, job_dir)
        inspection = inspect_cache(path, metadata)
        if inspection.state is not CacheState.VALID:
            encoded += 1
            if encode_hook is not None:
                encode_hook()
                encode_hook = None
            if on_before_sample_encode is not None:
                on_before_sample_encode(sample, image_size)
            try:
                try:
                    latent, prompt_embedding = encode_sample(
                        sample,
                        image,
                        encoder,
                        config,
                    )
                except Exception as exc:
                    width, height = image_size
                    raise CacheError(
                        f"cache encode failed path={sample.image_path} "
                        f"size={width}x{height}"
                    ) from exc
            finally:
                if on_after_sample_encode is not None:
                    on_after_sample_encode(sample, image_size)
            try:
                write_cache_atomic(path, latent, prompt_embedding, metadata)
            finally:
                del latent, prompt_embedding
        else:
            reused += 1
        paths.append(path)
    log.info("cache ready encoded=%s reused=%s", encoded, reused)
    return paths


def refresh_cache(
    sample: DatasetSample,
    encoder: CacheEncoder,
    config: CacheConfig,
    *,
    job_dir: str | Path,
) -> Path:
    """Explicit cache-affecting command for one sample."""

    return prepare_cache_at_job_start(
        [sample],
        encoder,
        config,
        job_dir=job_dir,
    )[0]


def preview_cache_path(job_dir: str | Path, caption: str) -> Path:
    """Return the job-local preview prompt cache path for one caption."""

    return job_cache_path(
        job_dir,
        PREVIEW_CACHE_NAMESPACE,
        fingerprint_caption(caption),
    )


def expected_preview_metadata(
    caption: str,
    config: CacheConfig,
) -> dict[str, Any]:
    """Build the prompt-only cache identity for one preview caption."""

    metadata = {
        "cache_kind": PREVIEW_CACHE_KIND,
        "prompt_fingerprint": fingerprint_caption(caption),
        "main_revision": config.main_revision,
        "text_encoder_config": dict(config.text_encoder_config),
        "text_encoder_precision": config.text_encoder_precision,
        "tokenizer_config": dict(config.tokenizer_config),
        "qwen_chat_template": dict(config.qwen_chat_template),
        "max_sequence_length": config.max_sequence_length,
        "schema_version": config.schema_version,
    }
    _canonical_json(metadata)
    return metadata


def inspect_preview_cache(
    path: str | Path,
    expected: Mapping[str, Any],
) -> CacheInspection:
    """Explicitly classify one preview prompt cache as missing, valid, or stale."""

    target = Path(path)
    expected_dict = _normalized_metadata(expected)
    if not target.is_file():
        return CacheInspection(target, CacheState.MISSING, expected_dict)
    try:
        _embedding, metadata = load_preview_cache(target)
        del _embedding
    except (CacheError, OSError, ValueError) as exc:
        return CacheInspection(
            target,
            CacheState.STALE,
            expected_dict,
            f"invalid cache: {exc}",
        )
    if metadata != expected_dict:
        return CacheInspection(
            target,
            CacheState.STALE,
            expected_dict,
            "cache metadata differs",
        )
    return CacheInspection(target, CacheState.VALID, expected_dict)


def encode_prompt_embedding(
    caption: str,
    encoder: CacheEncoder,
    config: CacheConfig,
) -> torch.Tensor:
    """Encode one caption to a CPU bf16 prompt embedding."""

    with torch.no_grad():
        prompt_embedding = encoder.encode_prompt(
            caption,
            max_sequence_length=config.max_sequence_length,
        )
    if not isinstance(prompt_embedding, torch.Tensor):
        raise CacheError("encode_prompt() must return a tensor")
    prompt_embedding = _remove_single_batch(
        prompt_embedding,
        "prompt_embedding",
    )
    prompt_embedding = (
        prompt_embedding.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
    )
    validate_prompt_embedding(
        prompt_embedding,
        max_sequence_length=config.max_sequence_length,
    )
    return prompt_embedding


def write_preview_cache_atomic(
    destination: str | Path,
    prompt_embedding: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    """Validate and atomically replace one prompt-only cache file."""

    normalized = _normalized_metadata(metadata)
    max_sequence_length = normalized.get("max_sequence_length")
    if not isinstance(max_sequence_length, int):
        raise CacheError("cache metadata max_sequence_length is invalid")
    validate_prompt_embedding(
        prompt_embedding,
        max_sequence_length=max_sequence_length,
    )
    return _write_tensors_atomic(
        destination,
        {PROMPT_EMBEDDING_KEY: prompt_embedding},
        normalized,
    )


def load_preview_cache(
    path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one prompt-only cache and validate metadata, shape, and dtype."""

    target = Path(path)
    tensors, metadata = _read_tensors_cpu(target)
    required = {PROMPT_EMBEDDING_KEY}
    if set(tensors) != required:
        raise CacheError(f"preview cache tensor keys must be {sorted(required)}")
    prompt_embedding = tensors[PROMPT_EMBEDDING_KEY]
    max_sequence_length = metadata.get("max_sequence_length")
    if not isinstance(max_sequence_length, int):
        raise CacheError("cache metadata max_sequence_length is invalid")
    validate_prompt_embedding(
        prompt_embedding,
        max_sequence_length=max_sequence_length,
    )
    if expected is not None and metadata != _normalized_metadata(expected):
        raise CacheError("cache metadata differs")
    return prompt_embedding, metadata


def prepare_preview_prompt_cache(
    prompts: Iterable[str],
    encoder: CacheEncoder,
    config: CacheConfig,
    *,
    job_dir: str | Path,
) -> dict[str, Path]:
    """Inspect and materialize unique non-empty preview prompt caches."""

    paths: dict[str, Path] = {}
    encoded = 0
    reused = 0
    for prompt in prompts:
        if prompt == "" or prompt in paths:
            continue
        metadata = expected_preview_metadata(prompt, config)
        path = preview_cache_path(job_dir, prompt)
        inspection = inspect_preview_cache(path, metadata)
        if inspection.state is not CacheState.VALID:
            encoded += 1
            prompt_embedding = encode_prompt_embedding(prompt, encoder, config)
            try:
                write_preview_cache_atomic(path, prompt_embedding, metadata)
            finally:
                del prompt_embedding
        else:
            reused += 1
        paths[prompt] = path
    log.info("preview cache ready encoded=%s reused=%s", encoded, reused)
    return paths


def _write_tensors_atomic(
    destination: str | Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically replace one safetensors file from a CPU tensor mapping."""

    target = Path(destination)
    normalized = _normalized_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            {
                key: tensor.detach().to(device="cpu").contiguous()
                for key, tensor in tensors.items()
            },
            str(temporary),
            metadata={METADATA_KEY: _canonical_json(normalized)},
        )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _read_tensors_cpu(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load safetensors tensors on CPU together with normalized metadata."""

    target = Path(path)
    try:
        with safe_open(target, framework="pt", device="cpu") as handle:
            file_metadata = handle.metadata() or {}
            if METADATA_KEY not in file_metadata:
                raise CacheError("cache metadata is missing")
            try:
                metadata = _normalized_metadata(
                    json.loads(file_metadata[METADATA_KEY])
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise CacheError("cache metadata is invalid JSON") from exc
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    except CacheError:
        raise
    except (OSError, ValueError) as exc:
        raise CacheError(f"cannot read cache: {target}") from exc
    return tensors, metadata


def validate_prompt_embedding(
    prompt_embedding: torch.Tensor,
    *,
    max_sequence_length: int,
) -> None:
    """Validate the prompt-embedding half of the cache tensor contract."""

    if prompt_embedding.dtype != torch.bfloat16:
        raise CacheError("prompt_embedding dtype must be torch.bfloat16")
    if prompt_embedding.ndim != 2:
        raise CacheError("prompt_embedding must have shape [valid_tokens, 2560]")
    valid_tokens, hidden_size = prompt_embedding.shape
    if hidden_size != CACHE_PROMPT_EMBED_HIDDEN_SIZE:
        raise CacheError(
            "prompt_embedding hidden size must be "
            f"{CACHE_PROMPT_EMBED_HIDDEN_SIZE}"
        )
    if valid_tokens < 1 or valid_tokens > max_sequence_length:
        raise CacheError(
            "prompt_embedding valid-token count must be between 1 and "
            f"{max_sequence_length}"
        )


def validate_cache_tensors(
    latent: torch.Tensor,
    prompt_embedding: torch.Tensor,
    *,
    image_size: tuple[int, int],
    max_sequence_length: int,
) -> None:
    """Validate the exact version-one cache tensor contract."""

    width, height = image_size
    expected_latent_shape = (
        CACHE_LATENT_CHANNELS,
        height // CACHE_LATENT_SPATIAL_DIVISOR,
        width // CACHE_LATENT_SPATIAL_DIVISOR,
    )
    if latent.dtype != torch.bfloat16:
        raise CacheError("latent dtype must be torch.bfloat16")
    if tuple(latent.shape) != expected_latent_shape:
        raise CacheError(
            f"latent shape must be {expected_latent_shape}, got {tuple(latent.shape)}"
        )
    validate_prompt_embedding(
        prompt_embedding,
        max_sequence_length=max_sequence_length,
    )


def _remove_single_batch(tensor: torch.Tensor, label: str) -> torch.Tensor:
    expected_unbatched_dims = 3 if label == "latent" else 2
    if tensor.ndim == expected_unbatched_dims + 1:
        if tensor.shape[0] != 1:
            raise CacheError(f"{label} encoder batch dimension must be 1")
        return tensor[0]
    return tensor


def _vae_number(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CacheError(f"vae_config.{key} must be a number")
    return float(value)


def _metadata_image_size(metadata: Mapping[str, Any]) -> tuple[int, int]:
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise CacheError("cache metadata preprocessing is invalid")
    width = preprocessing.get("width")
    height = preprocessing.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width < 1
        or height < 1
    ):
        raise CacheError("cache metadata image dimensions are invalid")
    return width, height


def _normalized_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise CacheError("cache metadata must be a mapping")
    try:
        return json.loads(_canonical_json(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise CacheError("cache metadata must be JSON serializable") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
