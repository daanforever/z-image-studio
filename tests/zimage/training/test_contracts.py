from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

import zimage.training as training_api
import zimage.training.contracts as training_contracts
from zimage.training import (
    CacheEncoder,
    CacheRecord,
    CheckpointWriter,
    CommandEnvelope,
    ConfigReloader,
    ConfigUpdateDecision,
    EncodedImage,
    JobState,
    JobStatus,
    LatentDistribution,
    NativeAdapterMetadata,
    OptimizerStepBoundary,
    PromptEncodingSpec,
    PreviewSampler,
    RuntimeGuard,
    SavedCheckpoint,
    StepConfigReload,
    TensorSpec,
    TrainingBatch,
    TrainingConfigError,
    TrainingHook,
    UpdateClassification,
)


def test_job_status_values():
    assert {item.value for item in JobStatus} == {
        "running",
        "stopped",
        "completed",
        "failed",
    }
    assert JobStatus.RUNNING == "running"
    assert JobStatus.STOPPED == "stopped"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"


def test_contract_types_have_single_public_definitions():
    names = {
        "CacheRecord",
        "CacheEncoder",
        "TrainingBatch",
        "JobState",
        "JobStatus",
        "CommandEnvelope",
        "TrainingHook",
        "CheckpointWriter",
        "PreviewSampler",
        "RuntimeGuard",
        "ConfigReloader",
    }
    for name in names:
        assert getattr(training_api, name) is getattr(training_contracts, name)


def test_training_package_import_does_not_load_ml_or_ui_modules():
    code = """
import sys
import zimage.training

forbidden = ("torch", "diffusers", "transformers", "gradio")
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    print(",".join(loaded), file=sys.stderr)
raise SystemExit(bool(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_job_state_operational_fields_only():
    names = {item.name for item in fields(JobState)}
    assert names == {
        "job_id",
        "status",
        "step",
        "epoch",
        "last_error",
        "exit_code",
    }
    state = JobState(job_id="job-1", status=JobStatus.STOPPED, step=0, epoch=0)
    assert state.last_error is None
    assert state.exit_code is None


def test_command_envelope_fields():
    names = {item.name for item in fields(CommandEnvelope)}
    assert names == {"command_id", "kind", "payload", "created_at"}
    envelope = CommandEnvelope(
        command_id=1,
        kind="stop",
        payload={},
        created_at=0.0,
    )
    assert envelope.command_id == 1
    assert envelope.kind == "stop"


def test_cache_record_and_training_batch():
    record = CacheRecord(
        sample_id="s1",
        image_path="images/a.png",
        latent=TensorSpec(dtype="bf16", shape=(16, 128, 128)),
        prompt_embedding=TensorSpec(dtype="bf16", shape=(32, 2560), padded=False),
        image_fingerprint="img",
        caption_fingerprint="cap",
        main_revision="abc",
        vae_config={},
        text_encoder_config={},
        tokenizer_config={},
        qwen_chat_template={},
        max_sequence_length=512,
        preprocessing={},
        schema_version=1,
    )
    assert record.sample_id == "s1"
    assert record.latent.shape[0] == 16
    assert record.prompt_embedding.shape[-1] == 2560
    assert record.latent.dtype == "bf16"
    assert record.latent.shape == (16, 128, 128)
    assert record.prompt_embedding.dtype == "bf16"
    assert record.prompt_embedding.shape == (32, 2560)
    assert record.prompt_embedding.padded is False
    assert record.schema_version == 1

    names = {item.name for item in fields(CacheRecord)}
    assert "image_fingerprint" in names
    assert "caption_fingerprint" in names
    assert "main_revision" in names
    assert "vae_config" in names
    assert "text_encoder_config" in names
    assert "tokenizer_config" in names
    assert "qwen_chat_template" in names
    assert "max_sequence_length" in names
    assert "preprocessing" in names
    assert "schema_version" in names

    batch = TrainingBatch(latent=None, prompt_embedding=None)
    assert batch.batch_size == 1
    assert batch.metadata is None


def test_training_config_error_is_value_error():
    assert issubclass(TrainingConfigError, ValueError)
    with pytest.raises(TrainingConfigError):
        raise TrainingConfigError("invalid")


def test_training_hook_protocol():
    class DummyHook:
        def on_optimizer_step(self, boundary: OptimizerStepBoundary) -> None:
            return None

    assert hasattr(TrainingHook, "on_optimizer_step")
    assert isinstance(DummyHook(), TrainingHook)


def test_checkpoint_writer_protocol():
    class DummyWriter:
        def write_atomic(
            self,
            *,
            destination: Path,
            lora_state: dict,
            metadata: NativeAdapterMetadata,
        ) -> SavedCheckpoint:
            return SavedCheckpoint(destination, metadata)

    assert hasattr(CheckpointWriter, "write_atomic")
    assert isinstance(DummyWriter(), CheckpointWriter)


def test_preview_sampler_protocol():
    class DummySampler:
        def sample_unfused(
            self,
            *,
            checkpoint: SavedCheckpoint,
            parameters: dict,
            destination: Path,
        ) -> Path:
            return destination

    assert hasattr(PreviewSampler, "sample_unfused")
    assert isinstance(DummySampler(), PreviewSampler)


def test_cache_encoder_and_prompt_spec_contracts():
    class Distribution:
        def mode(self):
            return None

    class ImageOutput:
        latent_dist = Distribution()

    class DummyEncoder:
        def encode_image(self, image):
            return ImageOutput()

        def encode_prompt(self, caption: str, *, spec: PromptEncodingSpec):
            return None

    spec = PromptEncodingSpec(max_sequence_length=512)
    assert spec.chat_template_options == {
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert isinstance(Distribution(), LatentDistribution)
    assert isinstance(ImageOutput(), EncodedImage)
    assert isinstance(DummyEncoder(), CacheEncoder)


def test_step_boundary_checkpoint_and_config_reload_types(tmp_path):
    state = JobState("job", JobStatus.RUNNING, 3, 1)
    boundary = OptimizerStepBoundary(tmp_path, state, {"max_steps": 10})
    assert boundary.state.step == 3

    metadata = NativeAdapterMetadata(
        adapter_name="default",
        base_model_name_or_path="Tongyi-MAI/Z-Image",
        base_model_revision="abc",
        peft_config={"r": 4, "lora_alpha": 8},
        optimizer_step=3,
    )
    checkpoint = SavedCheckpoint(tmp_path / "step-3", metadata)
    assert checkpoint.metadata.peft_config["lora_alpha"] == 8

    decision = ConfigUpdateDecision(
        command_id=1,
        classification=UpdateClassification.REBUILD_REQUIRED,
        changed_fields=("datasets",),
    )
    reload = StepConfigReload({"max_steps": 10}, (decision,))
    assert reload.rebuild_required is True

    class DummyReloader:
        def reload_at_optimizer_step(
            self,
            *,
            job_dir: Path,
            state: JobState,
            current_config: dict,
        ) -> StepConfigReload:
            return reload

    assert isinstance(DummyReloader(), ConfigReloader)


def test_runtime_guard_protocol():
    class DummyGuard:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def is_held(self) -> bool:
            return True

    assert hasattr(RuntimeGuard, "acquire")
    assert hasattr(RuntimeGuard, "release")
    assert hasattr(RuntimeGuard, "is_held")
    assert isinstance(DummyGuard(), RuntimeGuard)
