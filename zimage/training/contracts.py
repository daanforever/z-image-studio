"""Shared training runtime types. No GPU, loop, or UI logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable


class JobStatus(str, Enum):
    """Operational job lifecycle. No metrics or history."""

    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    """In-memory operational snapshot of one training job."""

    job_id: str
    status: JobStatus
    step: int
    epoch: int
    last_error: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class CommandEnvelope:
    """Control-plane message. ``command_id`` is monotonic per process."""

    command_id: int
    kind: str
    payload: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor layout (dtype + shape). No storage."""

    dtype: str
    shape: tuple[int | str, ...]
    padded: bool = False


@dataclass(frozen=True)
class PromptEncodingSpec:
    """Parameters that make Qwen prompt embeddings cache-compatible."""

    max_sequence_length: int
    add_generation_prompt: bool = True
    enable_thinking: bool = True

    @property
    def chat_template_options(self) -> dict[str, bool]:
        return {
            "add_generation_prompt": self.add_generation_prompt,
            "enable_thinking": self.enable_thinking,
        }


@runtime_checkable
class LatentDistribution(Protocol):
    """Minimal deterministic VAE distribution used by the cache."""

    def mode(self) -> Any:
        """Return the unscaled latent mode."""
        ...


@runtime_checkable
class EncodedImage(Protocol):
    """VAE output containing a latent distribution."""

    latent_dist: LatentDistribution


@runtime_checkable
class CacheEncoder(Protocol):
    """Injected model encoder used by the model-independent cache layer."""

    def encode_image(
        self,
        image: Any,
    ) -> LatentDistribution | EncodedImage:
        """Encode one validated RGB image without sampling the distribution."""
        ...

    def encode_prompt(
        self,
        caption: str,
        *,
        spec: PromptEncodingSpec,
    ) -> Any:
        """Return one unpadded prompt embedding."""
        ...


@dataclass(frozen=True)
class CacheRecord:
    """One cached sample: identity, tensor specs, fingerprints, encoder metadata."""

    sample_id: str
    image_path: str
    latent: TensorSpec
    prompt_embedding: TensorSpec
    image_fingerprint: str
    caption_fingerprint: str
    main_revision: str | None
    vae_config: dict[str, Any]
    text_encoder_config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    qwen_chat_template: dict[str, Any]
    max_sequence_length: int
    preprocessing: dict[str, Any]
    schema_version: int


@dataclass
class TrainingBatch:
    """MVP batch (``batch_size`` is always 1): one latent + one prompt embedding."""

    latent: Any
    prompt_embedding: Any
    metadata: dict[str, Any] | None = None
    batch_size: int = field(default=1, init=False)


@dataclass(frozen=True)
class OptimizerStepBoundary:
    """State visible only after an optimizer step has completed."""

    job_dir: Path
    state: JobState
    config: Mapping[str, Any]


@runtime_checkable
class TrainingHook(Protocol):
    """Observer called exactly at an optimizer-step boundary."""

    def on_optimizer_step(self, boundary: OptimizerStepBoundary) -> None:
        """Observe a completed optimizer step without owning the GPU loop."""
        ...


@dataclass(frozen=True)
class NativeAdapterMetadata:
    """Metadata required to preserve native PEFT LoRA scaling."""

    adapter_name: str
    base_model_name_or_path: str
    base_model_revision: str | None
    peft_config: Mapping[str, Any]
    optimizer_step: int

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "adapter_name",
        "base_model_name_or_path",
        "base_model_revision",
        "peft_config",
        "optimizer_step",
    )
    REQUIRED_PEFT_FIELDS: ClassVar[tuple[str, ...]] = (
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    )

    @classmethod
    def parse(
        cls,
        raw: Any,
        *,
        expected_step: int | None = None,
    ) -> NativeAdapterMetadata:
        """Strict sidecar mapping used for completeness checks and load."""

        def require_nonempty_str(value: Any, label: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
            return value.strip()

        def require_int(value: Any, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
            return value

        def require_number(value: Any, label: str) -> int | float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number")
            return value

        def require_str_list(value: Any, label: str) -> list[str]:
            if not isinstance(value, list):
                raise ValueError(f"{label} must be a list of strings")
            return [
                require_nonempty_str(item, f"{label}[{index}]")
                for index, item in enumerate(value)
            ]

        if not isinstance(raw, dict):
            raise ValueError("adapter metadata must be a mapping")
        missing = [key for key in cls.REQUIRED_FIELDS if key not in raw]
        if missing:
            raise ValueError(f"adapter metadata missing fields: {missing}")

        revision_raw = raw["base_model_revision"]
        if revision_raw is None:
            revision: str | None = None
        else:
            revision = require_nonempty_str(revision_raw, "base_model_revision")

        peft_raw = raw["peft_config"]
        if not isinstance(peft_raw, dict):
            raise ValueError("peft_config must be a mapping")
        missing_peft = [
            key for key in cls.REQUIRED_PEFT_FIELDS if key not in peft_raw
        ]
        if missing_peft:
            raise ValueError(f"peft_config missing fields: {missing_peft}")

        peft_config = {str(key): value for key, value in peft_raw.items()}
        peft_config["r"] = require_int(peft_raw["r"], "peft_config.r")
        peft_config["lora_alpha"] = require_number(
            peft_raw["lora_alpha"], "peft_config.lora_alpha"
        )
        peft_config["lora_dropout"] = require_number(
            peft_raw["lora_dropout"], "peft_config.lora_dropout"
        )
        peft_config["target_modules"] = require_str_list(
            peft_raw["target_modules"], "peft_config.target_modules"
        )

        optimizer_step = require_int(raw["optimizer_step"], "optimizer_step")
        if expected_step is not None and optimizer_step != expected_step:
            raise ValueError(
                f"optimizer_step {optimizer_step} does not match step-{expected_step}"
            )
        return cls(
            adapter_name=require_nonempty_str(raw["adapter_name"], "adapter_name"),
            base_model_name_or_path=require_nonempty_str(
                raw["base_model_name_or_path"], "base_model_name_or_path"
            ),
            base_model_revision=revision,
            peft_config=peft_config,
            optimizer_step=optimizer_step,
        )


@dataclass(frozen=True)
class SavedCheckpoint:
    """A completed atomic checkpoint directory and its adapter metadata."""

    path: Path
    metadata: NativeAdapterMetadata


@runtime_checkable
class CheckpointWriter(Protocol):
    """Atomic native LoRA save (safetensors). Do not fuse or merge weights."""

    def write_atomic(
        self,
        *,
        destination: Path,
        lora_state: Mapping[str, Any],
        metadata: NativeAdapterMetadata,
    ) -> SavedCheckpoint:
        """Write native adapter weights and metadata, then atomically commit."""
        ...


@runtime_checkable
class PreviewSampler(Protocol):
    """Unfused adapter preview using merged sampling parameters."""

    def sample_unfused(
        self,
        *,
        checkpoint: SavedCheckpoint,
        parameters: Mapping[str, Any],
        destination: Path,
    ) -> Path:
        """Render one preview from a completed checkpoint and return its path."""
        ...


class UpdateClassification(str, Enum):
    """How a validated candidate config may be applied at a step boundary."""

    NO_CHANGE = "no_change"
    APPLY_AT_STEP = "apply_at_step"
    REBUILD_REQUIRED = "rebuild_required"
    REJECTED_IMMUTABLE = "rejected_immutable"
    INVALID = "invalid"


@dataclass(frozen=True)
class ConfigUpdateDecision:
    """Disposition of one command-queue config candidate."""

    command_id: int
    classification: UpdateClassification
    changed_fields: tuple[str, ...]
    message: str | None = None


@dataclass(frozen=True)
class StepConfigReload:
    """Effective config and decisions produced at one optimizer boundary."""

    effective_config: Mapping[str, Any]
    decisions: tuple[ConfigUpdateDecision, ...] = ()

    @property
    def rebuild_required(self) -> bool:
        return any(
            decision.classification is UpdateClassification.REBUILD_REQUIRED
            for decision in self.decisions
        )


@runtime_checkable
class ConfigReloader(Protocol):
    """Poll and validate config commands only at optimizer-step boundaries."""

    def reload_at_optimizer_step(
        self,
        *,
        job_dir: Path,
        state: JobState,
        current_config: Mapping[str, Any],
    ) -> StepConfigReload:
        """Return the config that should govern the next optimizer step."""
        ...


@runtime_checkable
class RuntimeGuard(Protocol):
    """Exclusive cross-process GPU lease. OS lock is implemented later."""

    def acquire(self) -> bool:
        """Try to take the lease. True if this caller now holds it."""
        ...

    def release(self) -> None:
        """Release the lease if this caller holds it."""
        ...

    def is_held(self) -> bool:
        """Whether this process currently holds the lease."""
        ...
