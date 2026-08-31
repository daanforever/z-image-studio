"""Atomic native LoRA checkpoints. Weights only; optimizer state lives in the loop."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from safetensors import safe_open
from safetensors.torch import load_file

from zimage.training.contracts import NativeAdapterMetadata, SavedCheckpoint

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
ADAPTER_METADATA_NAME = "adapter_metadata.json"
CHECKPOINTS_DIRECTORY = "checkpoints"
LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"
STEP_DIRECTORY_PATTERN = re.compile(r"^step-(\d+)$")

SaveLoraWeights = Callable[..., Any]


class CheckpointError(ValueError):
    """A checkpoint directory or native LoRA payload is invalid."""


@dataclass(frozen=True)
class LoadedLoraState:
    """LoRA tensors and adapter metadata for a warm start (no optimizer)."""

    path: Path
    state_dict: dict[str, Any]
    metadata: NativeAdapterMetadata
    safetensors_metadata: Mapping[str, str]


class NativeLoraCheckpointWriter:
    """Write native Diffusers LoRA weights into an atomically committed directory."""

    def __init__(self, *, save_lora_weights: SaveLoraWeights | None = None) -> None:
        self._save_lora_weights = save_lora_weights

    def write_atomic(
        self,
        *,
        destination: Path,
        lora_state: Mapping[str, Any],
        metadata: NativeAdapterMetadata,
    ) -> SavedCheckpoint:
        """Write to ``step-N.tmp``, validate, then ``os.replace`` onto ``step-N``."""

        destination = Path(destination)
        if destination.suffix == ".tmp" or destination.name.endswith(".tmp"):
            raise CheckpointError("destination must be the final step directory")
        if not lora_state:
            raise CheckpointError("lora_state must contain adapter tensors")

        staging = destination.with_name(f"{destination.name}.tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        _write_native_lora(
            staging,
            lora_state,
            metadata,
            save_lora_weights=self._save_lora_weights,
        )
        validate_lora_weights(staging)
        # Crash before this replace leaves only ``step-N.tmp``, which
        # find_latest_checkpoint ignores.
        _commit_directory(staging, destination)
        return SavedCheckpoint(path=destination, metadata=metadata)


def write_atomic(
    *,
    destination: Path,
    lora_state: Mapping[str, Any],
    metadata: NativeAdapterMetadata,
    save_lora_weights: SaveLoraWeights | None = None,
) -> SavedCheckpoint:
    """Module-level CheckpointWriter.write_atomic helper."""

    return NativeLoraCheckpointWriter(
        save_lora_weights=save_lora_weights
    ).write_atomic(
        destination=destination,
        lora_state=lora_state,
        metadata=metadata,
    )


def checkpoints_dir(job_dir: str | Path) -> Path:
    """Return ``job_dir/checkpoints``."""

    return Path(job_dir) / CHECKPOINTS_DIRECTORY


def step_checkpoint_dir(job_dir: str | Path, step: int) -> Path:
    """Return the final ``checkpoints/step-N`` path for ``step``."""

    return checkpoints_dir(job_dir) / f"step-{int(step)}"


def find_latest_checkpoint(job_dir: str | Path) -> Path | None:
    """Return the highest completed ``step-N`` dir with valid native LoRA weights.

    ``*.tmp`` directories and incomplete folders are ignored.
    """

    root = checkpoints_dir(job_dir)
    if not root.is_dir():
        return None

    latest: tuple[int, Path] | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith(".tmp"):
            continue
        match = STEP_DIRECTORY_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        if not is_complete_checkpoint(child):
            continue
        step = int(match.group(1))
        if latest is None or step > latest[0]:
            latest = (step, child)
    return None if latest is None else latest[1]


def load_latest_lora_state(job_dir: str | Path) -> LoadedLoraState | None:
    """Load the latest completed LoRA tensors and metadata for a warm start."""

    latest = find_latest_checkpoint(job_dir)
    if latest is None:
        return None
    return load_lora_state(latest)


def load_lora_state(checkpoint_dir: str | Path) -> LoadedLoraState:
    """Load native LoRA tensors and adapter metadata from a completed directory."""

    target = Path(checkpoint_dir)
    validate_checkpoint_directory(target)
    weights = target / LORA_WEIGHT_NAME
    state_dict = load_file(str(weights))
    file_metadata = _read_safetensors_file_metadata(weights)
    metadata = parse_adapter_sidecar(target)
    return LoadedLoraState(
        path=target,
        state_dict=state_dict,
        metadata=metadata,
        safetensors_metadata=file_metadata,
    )


def is_complete_checkpoint(checkpoint_dir: str | Path) -> bool:
    """True when ``checkpoint_dir`` is a finished ``step-N`` with a valid sidecar."""

    try:
        validate_checkpoint_directory(checkpoint_dir)
    except CheckpointError:
        return False
    return True


def validate_checkpoint_directory(checkpoint_dir: str | Path) -> Path:
    """Require a completed directory with valid weights and a fully valid sidecar."""

    target = Path(checkpoint_dir)
    if not target.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {target}")
    if target.name.endswith(".tmp"):
        raise CheckpointError(f"staging directory is not a completed checkpoint: {target}")
    validate_lora_weights(target)
    validate_adapter_sidecar(target)
    return target


def validate_adapter_sidecar(checkpoint_dir: str | Path) -> NativeAdapterMetadata:
    """Require a fully valid sidecar. Same parser as load."""

    return parse_adapter_sidecar(checkpoint_dir)


def parse_adapter_sidecar(checkpoint_dir: str | Path) -> NativeAdapterMetadata:
    """Strict sidecar parser shared by completeness checks and load."""

    target = Path(checkpoint_dir)
    sidecar = target / ADAPTER_METADATA_NAME
    if not sidecar.is_file():
        raise CheckpointError(f"missing {ADAPTER_METADATA_NAME} in {target}")
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid adapter metadata: {sidecar}") from exc
    try:
        return NativeAdapterMetadata.parse(
            raw,
            expected_step=_expected_optimizer_step(target),
        )
    except ValueError as exc:
        raise CheckpointError(f"invalid adapter metadata: {sidecar}: {exc}") from exc


def validate_lora_weights(checkpoint_dir: str | Path) -> Path:
    """Validate native LoRA weights whether the directory is staging or final."""

    target = Path(checkpoint_dir)
    if not target.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {target}")
    weights = target / LORA_WEIGHT_NAME
    if not weights.is_file():
        raise CheckpointError(f"missing {LORA_WEIGHT_NAME} in {target}")
    try:
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise CheckpointError(f"invalid safetensors checkpoint: {weights}") from exc
    if not keys:
        raise CheckpointError(f"checkpoint contains no LoRA tensors: {weights}")
    if not any("lora" in key.lower() for key in keys):
        raise CheckpointError(f"checkpoint is not a LoRA state dict: {weights}")
    return target


def _write_native_lora(
    directory: Path,
    lora_state: Mapping[str, Any],
    metadata: NativeAdapterMetadata,
    *,
    save_lora_weights: SaveLoraWeights | None,
) -> None:
    tensors = {
        str(key): _cpu_tensor(value) for key, value in lora_state.items()
    }
    peft_metadata = _jsonable_mapping(metadata.peft_config)
    saver = save_lora_weights or _default_save_lora_weights
    saver(
        directory,
        transformer_lora_layers=tensors,
        transformer_lora_adapter_metadata=peft_metadata,
        safe_serialization=True,
    )
    _write_adapter_metadata(directory / ADAPTER_METADATA_NAME, metadata)
    if any(
        name.startswith("optimizer")
        for name in (child.name for child in directory.iterdir())
    ):
        raise CheckpointError("checkpoint must not contain optimizer state")


def _default_save_lora_weights(
    save_directory: Path,
    *,
    transformer_lora_layers: Mapping[str, Any],
    transformer_lora_adapter_metadata: Mapping[str, Any] | None,
    safe_serialization: bool = True,
) -> None:
    from diffusers import ZImagePipeline

    ZImagePipeline.save_lora_weights(
        save_directory,
        transformer_lora_layers=dict(transformer_lora_layers),
        transformer_lora_adapter_metadata=(
            dict(transformer_lora_adapter_metadata)
            if transformer_lora_adapter_metadata
            else None
        ),
        safe_serialization=safe_serialization,
    )


def _commit_directory(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CheckpointError(f"checkpoint already exists: {destination}")
    os.replace(staging, destination)


def _write_adapter_metadata(path: Path, metadata: NativeAdapterMetadata) -> None:
    payload = {
        "adapter_name": metadata.adapter_name,
        "base_model_name_or_path": metadata.base_model_name_or_path,
        "base_model_revision": metadata.base_model_revision,
        "peft_config": _jsonable_mapping(metadata.peft_config),
        "optimizer_step": int(metadata.optimizer_step),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_safetensors_file_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _expected_optimizer_step(path: Path) -> int | None:
    match = STEP_DIRECTORY_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _cpu_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().to("cpu").contiguous()
    return value


def _jsonable_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(key): _jsonable(item) for key, item in value.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
