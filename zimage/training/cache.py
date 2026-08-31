"""Versioned safetensors cache for captioned Z-Image training samples.

Encoding is supplied by the caller through :class:`CacheEncoder`; this module
never discovers or loads production model classes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

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

LATENT_KEY = "latent"
PROMPT_EMBEDDING_KEY = "prompt_embedding"
METADATA_KEY = "zimage_training_cache"

DEFAULT_PREPROCESSING: dict[str, Any] = {
    "exif_orientation": "applied",
    "alpha_composite": "white",
    "color_mode": "RGB",
    "resize": None,
    "crop": None,
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


def cache_path_for(
    sample: DatasetSample | str | Path,
    cache_dir: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
) -> Path:
    """Return a collision-safe cache path rooted at the dataset cache directory.

    A :class:`DatasetSample` supplies the dataset root needed to preserve the
    image's relative parent hierarchy.  Path-only calls must provide either
    ``dataset_path`` or an explicit ``cache_dir``.
    """

    if isinstance(sample, DatasetSample):
        image = sample.image_path.resolve()
        dataset_root = sample.dataset_path.resolve()
    else:
        image = Path(sample).resolve()
        dataset_root = Path(dataset_path).resolve() if dataset_path is not None else None

    if dataset_root is None:
        if cache_dir is None:
            raise CacheError(
                "dataset_path is required when cache_path_for receives an image path"
            )
        relative = Path(image.name)
    else:
        try:
            relative = image.relative_to(dataset_root)
        except ValueError as exc:
            raise CacheError(
                f"image is outside its dataset directory: {image}"
            ) from exc

    directory = (
        Path(cache_dir)
        if cache_dir is not None
        else dataset_root / ".cache"
    )
    return directory / relative.parent / f"{relative.name}.safetensors"


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
        "image_path": str(sample.image_path.resolve()),
        "image_fingerprint": fingerprint_file(sample.image_path),
        "caption_fingerprint": fingerprint_caption(sample.caption),
        "main_revision": config.main_revision,
        "vae_config": dict(config.vae_config),
        "text_encoder_config": dict(config.text_encoder_config),
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
        prompt_embedding = encoder.encode_prompt(
            sample.caption,
            max_sequence_length=config.max_sequence_length,
        )

    if not isinstance(raw_latent, torch.Tensor):
        raise CacheError("latent_dist.mode() must return a tensor")
    if not isinstance(prompt_embedding, torch.Tensor):
        raise CacheError("encode_prompt() must return a tensor")

    shift_factor = _vae_number(config.vae_config, "shift_factor")
    scaling_factor = _vae_number(config.vae_config, "scaling_factor")
    latent = (raw_latent - shift_factor) * scaling_factor
    latent = _remove_single_batch(latent, "latent")
    prompt_embedding = _remove_single_batch(
        prompt_embedding,
        "prompt_embedding",
    )
    latent = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    prompt_embedding = (
        prompt_embedding.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
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

    target = Path(destination)
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
                LATENT_KEY: latent.detach().to(device="cpu").contiguous(),
                PROMPT_EMBEDDING_KEY: (
                    prompt_embedding.detach().to(device="cpu").contiguous()
                ),
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


def load_cache(
    path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> CachedSample:
    """Load one cache and validate metadata, shape, and dtype."""

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
            keys = set(handle.keys())
            required = {LATENT_KEY, PROMPT_EMBEDDING_KEY}
            if keys != required:
                raise CacheError(f"cache tensor keys must be {sorted(required)}")
            latent = handle.get_tensor(LATENT_KEY)
            prompt_embedding = handle.get_tensor(PROMPT_EMBEDDING_KEY)
    except CacheError:
        raise
    except (OSError, ValueError) as exc:
        raise CacheError(f"cannot read cache: {target}") from exc

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
    cache_dir: str | Path | None = None,
) -> list[CachedSample]:
    """Inspect and materialize all caches at an explicit job-start boundary."""

    cached: list[CachedSample] = []
    for sample in samples:
        image = load_training_image(sample.image_path)
        metadata = expected_metadata(sample, config, image_size=image.size)
        path = cache_path_for(sample, cache_dir)
        inspection = inspect_cache(path, metadata)
        if inspection.state is not CacheState.VALID:
            latent, prompt_embedding = encode_sample(
                sample,
                image,
                encoder,
                config,
            )
            write_cache_atomic(path, latent, prompt_embedding, metadata)
        cached.append(load_cache(path, expected=metadata))
    return cached


def refresh_cache(
    sample: DatasetSample,
    encoder: CacheEncoder,
    config: CacheConfig,
    *,
    cache_dir: str | Path | None = None,
) -> CachedSample:
    """Explicit cache-affecting command for one sample."""

    return prepare_cache_at_job_start(
        [sample],
        encoder,
        config,
        cache_dir=cache_dir,
    )[0]


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
