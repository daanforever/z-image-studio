from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from zimage.training.checkpoints import (
    ADAPTER_METADATA_NAME,
    CheckpointError,
    LORA_ADAPTER_METADATA_KEY,
    LORA_WEIGHT_NAME,
    NativeLoraCheckpointWriter,
    find_latest_checkpoint,
    is_complete_checkpoint,
    load_latest_lora_state,
    load_lora_state,
    parse_adapter_sidecar,
    step_checkpoint_dir,
    write_atomic,
)
from zimage.training.contracts import CheckpointWriter, NativeAdapterMetadata


def _peft_config(*, rank: int = 2, alpha: int = 8) -> dict:
    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "target_modules": ["to_q"],
        "peft_type": "LORA",
    }


def _lora_state(*, sign: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "to_q.lora_A.weight": torch.ones(2, 16),
        "to_q.lora_B.weight": torch.full((16, 2), sign * 0.25),
    }


def _metadata(*, step: int = 3, alpha: int = 8) -> NativeAdapterMetadata:
    return NativeAdapterMetadata(
        adapter_name="default",
        base_model_name_or_path="org/z-image",
        base_model_revision="abc123",
        peft_config=_peft_config(alpha=alpha),
        optimizer_step=step,
    )


def _job_dir(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    return job


def _sidecar_payload(*, step: int = 8, **overrides) -> dict:
    payload = {
        "adapter_name": "default",
        "base_model_name_or_path": "org/z-image",
        "base_model_revision": "abc123",
        "peft_config": _peft_config(),
        "optimizer_step": step,
    }
    payload.update(overrides)
    return payload


def _plant_weights_and_sidecar(
    job: Path,
    *,
    step: int,
    weights: bytes,
    sidecar: object,
) -> Path:
    target = step_checkpoint_dir(job, step)
    target.mkdir()
    (target / LORA_WEIGHT_NAME).write_bytes(weights)
    text = sidecar if isinstance(sidecar, str) else json.dumps(sidecar)
    (target / ADAPTER_METADATA_NAME).write_text(text, encoding="utf-8")
    return target


def test_writer_implements_checkpoint_writer_protocol():
    writer = NativeLoraCheckpointWriter()
    assert isinstance(writer, CheckpointWriter)


def test_native_save_preserves_alpha_not_equal_to_rank_and_reloads(tmp_path):
    job = _job_dir(tmp_path)
    destination = step_checkpoint_dir(job, 3)
    state = _lora_state(sign=1.0)
    metadata = _metadata(step=3, alpha=8)

    saved = write_atomic(
        destination=destination,
        lora_state=state,
        metadata=metadata,
    )

    assert saved.path == destination
    assert saved.path.name == "step-3"
    assert (destination / LORA_WEIGHT_NAME).is_file()
    assert (destination / ADAPTER_METADATA_NAME).is_file()
    assert list(destination.glob("optimizer*")) == []
    assert not destination.with_name("step-3.tmp").exists()

    with safe_open(str(destination / LORA_WEIGHT_NAME), framework="pt", device="cpu") as handle:
        file_metadata = handle.metadata() or {}
        loaded_tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    assert LORA_ADAPTER_METADATA_KEY in file_metadata
    packed = json.loads(file_metadata[LORA_ADAPTER_METADATA_KEY])
    assert packed["transformer.lora_alpha"] == 8
    assert packed["transformer.r"] == 2
    assert packed["transformer.lora_alpha"] != packed["transformer.r"]

    sidecar = json.loads((destination / ADAPTER_METADATA_NAME).read_text(encoding="utf-8"))
    assert sidecar["peft_config"]["lora_alpha"] == 8
    assert sidecar["optimizer_step"] == 3
    assert sidecar["base_model_revision"] == "abc123"

    reloaded = load_lora_state(destination)
    assert reloaded.metadata.peft_config["lora_alpha"] == 8
    assert reloaded.metadata.peft_config["r"] == 2
    assert reloaded.metadata.optimizer_step == 3
    assert "to_q.lora_A.weight" in loaded_tensors or any(
        key.endswith("to_q.lora_A.weight") for key in loaded_tensors
    )
    for key, tensor in loaded_tensors.items():
        suffix = key.split("transformer.", 1)[-1]
        expected = state.get(suffix, state.get(key))
        if expected is not None:
            assert torch.equal(tensor, expected)

    latest = load_latest_lora_state(job)
    assert latest is not None
    assert latest.path == destination
    assert latest.metadata.peft_config["lora_alpha"] == 8
    assert LORA_ADAPTER_METADATA_KEY in latest.safetensors_metadata


def test_atomic_commit_uses_tmp_then_replace(tmp_path):
    job = _job_dir(tmp_path)
    destination = step_checkpoint_dir(job, 7)
    seen: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen.append((Path(src), Path(dst)))
        assert Path(src).name == "step-7.tmp"
        assert Path(src).is_dir()
        assert (Path(src) / LORA_WEIGHT_NAME).is_file()
        assert not Path(dst).exists()
        return real_replace(src, dst)

    import zimage.training.checkpoints as checkpoints_module

    original = checkpoints_module.os.replace
    checkpoints_module.os.replace = tracking_replace
    try:
        write_atomic(
            destination=destination,
            lora_state=_lora_state(),
            metadata=_metadata(step=7),
        )
    finally:
        checkpoints_module.os.replace = original

    assert seen == [(destination.with_name("step-7.tmp"), destination)]
    assert destination.is_dir()
    assert not destination.with_name("step-7.tmp").exists()


def test_crash_before_rename_leaves_no_valid_latest(tmp_path):
    job = _job_dir(tmp_path)
    destination = step_checkpoint_dir(job, 4)

    import zimage.training.checkpoints as checkpoints_module

    def boom(src, dst):
        raise OSError("simulated crash before rename")

    original = checkpoints_module.os.replace
    checkpoints_module.os.replace = boom
    try:
        with pytest.raises(OSError, match="simulated crash"):
            write_atomic(
                destination=destination,
                lora_state=_lora_state(),
                metadata=_metadata(step=4),
            )
    finally:
        checkpoints_module.os.replace = original

    staging = destination.with_name("step-4.tmp")
    assert staging.is_dir()
    assert (staging / LORA_WEIGHT_NAME).is_file()
    assert not destination.exists()
    assert find_latest_checkpoint(job) is None
    assert load_latest_lora_state(job) is None


def test_latest_ignores_tmp_and_incomplete_directories(tmp_path):
    job = _job_dir(tmp_path)
    writer = NativeLoraCheckpointWriter()
    first = writer.write_atomic(
        destination=step_checkpoint_dir(job, 1),
        lora_state=_lora_state(sign=1.0),
        metadata=_metadata(step=1),
    )
    writer.write_atomic(
        destination=step_checkpoint_dir(job, 5),
        lora_state=_lora_state(sign=-1.0),
        metadata=_metadata(step=5),
    )

    leftover = step_checkpoint_dir(job, 99).with_name("step-99.tmp")
    leftover.mkdir()
    (leftover / LORA_WEIGHT_NAME).write_bytes((first.path / LORA_WEIGHT_NAME).read_bytes())
    (leftover / ADAPTER_METADATA_NAME).write_text(
        (first.path / ADAPTER_METADATA_NAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    incomplete = step_checkpoint_dir(job, 80)
    incomplete.mkdir()
    (incomplete / "readme.txt").write_text("not a checkpoint", encoding="utf-8")

    broken = step_checkpoint_dir(job, 90)
    broken.mkdir()
    (broken / LORA_WEIGHT_NAME).write_bytes(b"not-safetensors")

    latest = find_latest_checkpoint(job)
    assert latest == step_checkpoint_dir(job, 5)
    loaded = load_latest_lora_state(job)
    assert loaded is not None
    assert loaded.path == latest
    assert loaded.metadata.optimizer_step == 5


def test_latest_ignores_highest_step_with_corrupt_or_missing_sidecar(tmp_path):
    job = _job_dir(tmp_path)
    writer = NativeLoraCheckpointWriter()
    first = writer.write_atomic(
        destination=step_checkpoint_dir(job, 1),
        lora_state=_lora_state(sign=1.0),
        metadata=_metadata(step=1),
    )
    writer.write_atomic(
        destination=step_checkpoint_dir(job, 3),
        lora_state=_lora_state(sign=-1.0),
        metadata=_metadata(step=3),
    )

    # Highest step has valid weights but corrupt sidecar → incomplete.
    corrupt = step_checkpoint_dir(job, 9)
    corrupt.mkdir()
    (corrupt / LORA_WEIGHT_NAME).write_bytes(
        (first.path / LORA_WEIGHT_NAME).read_bytes()
    )
    (corrupt / ADAPTER_METADATA_NAME).write_text("{not-json", encoding="utf-8")

    # Another high step has weights but no sidecar at all.
    missing = step_checkpoint_dir(job, 7)
    missing.mkdir()
    (missing / LORA_WEIGHT_NAME).write_bytes(
        (first.path / LORA_WEIGHT_NAME).read_bytes()
    )

    latest = find_latest_checkpoint(job)
    assert latest == step_checkpoint_dir(job, 3)
    loaded = load_latest_lora_state(job)
    assert loaded is not None
    assert loaded.metadata.optimizer_step == 3


@pytest.mark.parametrize(
    "sidecar",
    [
        {},
        _sidecar_payload(step=8, peft_config={"r": 2}),
        _sidecar_payload(step=8, peft_config="not-a-mapping"),
        _sidecar_payload(
            step=8,
            peft_config={
                "r": "2",
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": ["to_q"],
            },
        ),
        _sidecar_payload(step=8, adapter_name=""),
        _sidecar_payload(step=8, adapter_name=1),
        _sidecar_payload(step=8, base_model_name_or_path=""),
        _sidecar_payload(step=8, base_model_name_or_path="   "),
        _sidecar_payload(step=8, base_model_revision=""),
        _sidecar_payload(step=8, base_model_revision="   "),
        _sidecar_payload(step=8, base_model_revision=12),
        _sidecar_payload(step=8, optimizer_step=1),
        _sidecar_payload(step=8, optimizer_step="8"),
        _sidecar_payload(step=8, optimizer_step=8.0),
        _sidecar_payload(
            step=8,
            peft_config={
                "r": 2,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": "to_q",
            },
        ),
    ],
    ids=[
        "empty-object",
        "missing-peft-fields",
        "peft-not-mapping",
        "peft-r-wrong-type",
        "empty-adapter-name",
        "adapter-name-wrong-type",
        "empty-base-path",
        "whitespace-base-path",
        "empty-revision",
        "whitespace-revision",
        "revision-wrong-type",
        "optimizer-step-mismatch",
        "optimizer-step-string",
        "optimizer-step-float",
        "target-modules-wrong-type",
    ],
)
def test_invalid_sidecar_is_incomplete_and_latest_selects_previous(tmp_path, sidecar):
    job = _job_dir(tmp_path)
    previous = write_atomic(
        destination=step_checkpoint_dir(job, 2),
        lora_state=_lora_state(),
        metadata=_metadata(step=2),
    )
    planted = _plant_weights_and_sidecar(
        job,
        step=8,
        weights=(previous.path / LORA_WEIGHT_NAME).read_bytes(),
        sidecar=sidecar,
    )

    assert is_complete_checkpoint(previous.path)
    assert not is_complete_checkpoint(planted)
    with pytest.raises(CheckpointError):
        parse_adapter_sidecar(planted)
    with pytest.raises(CheckpointError):
        load_lora_state(planted)
    assert find_latest_checkpoint(job) == previous.path


def test_null_revision_sidecar_is_complete_and_loadable(tmp_path):
    job = _job_dir(tmp_path)
    previous = write_atomic(
        destination=step_checkpoint_dir(job, 1),
        lora_state=_lora_state(),
        metadata=_metadata(step=1),
    )
    planted = _plant_weights_and_sidecar(
        job,
        step=4,
        weights=(previous.path / LORA_WEIGHT_NAME).read_bytes(),
        sidecar=_sidecar_payload(step=4, base_model_revision=None),
    )

    assert is_complete_checkpoint(planted)
    loaded = parse_adapter_sidecar(planted)
    assert loaded.base_model_revision is None
    assert loaded.optimizer_step == 4
    assert find_latest_checkpoint(job) == planted


def test_writer_output_is_a_complete_checkpoint(tmp_path):
    job = _job_dir(tmp_path)
    destination = step_checkpoint_dir(job, 6)
    write_atomic(
        destination=destination,
        lora_state=_lora_state(),
        metadata=_metadata(step=6),
    )

    assert is_complete_checkpoint(destination)
    parsed = parse_adapter_sidecar(destination)
    loaded = load_lora_state(destination)
    assert parsed == loaded.metadata
    assert parsed.adapter_name == "default"
    assert parsed.base_model_name_or_path == "org/z-image"
    assert parsed.base_model_revision == "abc123"
    assert parsed.optimizer_step == 6
    assert parsed.peft_config["r"] == 2
    assert parsed.peft_config["lora_alpha"] == 8
    assert parsed.peft_config["lora_dropout"] == 0.0
    assert parsed.peft_config["target_modules"] == ["to_q"]
    assert find_latest_checkpoint(job) == destination


def test_checkpoint_module_does_not_import_fuse_loader():
    source = Path(__file__).resolve().parents[3] / "zimage" / "training" / "checkpoints.py"
    text = source.read_text(encoding="utf-8")
    assert "zimage.engine.lora" not in text
    assert "apply_quantization" not in text
    assert "fuse_lora" not in text
    assert "_unpack_lora_adapter_metadata" not in text
    assert "_read_adapter_metadata" not in text
