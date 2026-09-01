from __future__ import annotations

import gc
import logging
import shutil
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

import zimage.training.cache as cache_module
from zimage.training.cache import (
    CacheConfig,
    CacheError,
    CacheState,
    cache_path_for,
    expected_metadata,
    expected_preview_metadata,
    inspect_cache,
    inspect_preview_cache,
    load_cache,
    load_preview_cache,
    prepare_cache_at_job_start,
    prepare_preview_prompt_cache,
    preview_cache_path,
    write_cache_atomic,
    write_preview_cache_atomic,
)
from zimage.training.dataset import DatasetError, DatasetSample, discover_samples


class FakeDistribution:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent


class FakeImageEncoding:
    def __init__(self, latent: torch.Tensor) -> None:
        self.latent_dist = FakeDistribution(latent)


class FakeEncoder:
    def __init__(self) -> None:
        self.image_calls = 0
        self.prompt_calls = 0
        self.captions: list[str] = []

    def encode_image(self, image: Image.Image) -> FakeImageEncoding:
        self.image_calls += 1
        width, height = image.size
        latent = torch.full(
            (1, 16, height // 8, width // 8),
            3.0,
            dtype=torch.float32,
        )
        return FakeImageEncoding(latent)

    def encode_prompt(
        self,
        caption: str,
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        self.prompt_calls += 1
        self.captions.append(caption)
        assert max_sequence_length == 32
        values = torch.arange(3 * 2560, dtype=torch.float32)
        return values.reshape(1, 3, 2560)


def _sample(
    tmp_path: Path,
    name: str = "image.png",
    *,
    caption: str = "a caption",
    color: tuple[int, int, int] = (10, 20, 30),
) -> DatasetSample:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    image_path = dataset / name
    Image.new("RGB", (32, 16), color).save(image_path)
    return DatasetSample(
        image_path=image_path.resolve(),
        caption=caption,
        dataset_path=dataset.resolve(),
    )


def _job_dir(tmp_path: Path, name: str = "job") -> Path:
    job = tmp_path / name
    job.mkdir(exist_ok=True)
    return job


def _config(**changes) -> CacheConfig:
    values = {
        "main_revision": "resolved-commit-sha",
        "vae_config": {
            "shift_factor": 0.5,
            "scaling_factor": 2.0,
            "block_out_channels": [128, 256],
        },
        "text_encoder_config": {"model_type": "qwen2"},
        "tokenizer_config": {"padding_side": "left"},
        "qwen_chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "max_sequence_length": 32,
    }
    values.update(changes)
    return CacheConfig(**values)


def _dataset_files(job_dir: Path) -> list[Path]:
    root = job_dir / ".cache" / "dataset"
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*.safetensors") if path.is_file())


def _preview_files(job_dir: Path) -> list[Path]:
    root = job_dir / ".cache" / "preview"
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*.safetensors") if path.is_file())


def _preview_ready_counts(caplog) -> tuple[int, int]:
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "zimage.training"
        and record.getMessage().startswith("preview cache ready encoded=")
    ]
    assert messages, "expected preview cache ready log"
    encoded_token, reused_token = messages[-1].removeprefix(
        "preview cache ready "
    ).split()
    return int(encoded_token.split("=", 1)[1]), int(reused_token.split("=", 1)[1])


def test_exact_tensors_and_complete_metadata_round_trip(tmp_path):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    config = _config()
    job_dir = _job_dir(tmp_path)

    path = prepare_cache_at_job_start(
        [sample],
        encoder,
        config,
        job_dir=job_dir,
    )[0]
    cached = load_cache(path)

    assert encoder.image_calls == 1
    assert encoder.prompt_calls == 1
    assert encoder.captions == ["a caption"]
    assert cached.path.parent == job_dir / ".cache" / "dataset"
    assert cached.path.suffix == ".safetensors"
    assert len(cached.path.stem) == 64
    assert cached.latent.dtype is torch.bfloat16
    assert cached.latent.shape == (16, 2, 4)
    assert torch.equal(
        cached.latent,
        torch.full((16, 2, 4), 5.0, dtype=torch.bfloat16),
    )
    expected_prompt = torch.arange(3 * 2560, dtype=torch.float32)
    expected_prompt = expected_prompt.reshape(3, 2560).to(torch.bfloat16)
    assert cached.prompt_embedding.dtype is torch.bfloat16
    assert cached.prompt_embedding.shape == (3, 2560)
    assert torch.equal(cached.prompt_embedding, expected_prompt)
    assert "image_path" not in cached.metadata
    assert cached.metadata["main_revision"] == "resolved-commit-sha"
    assert cached.metadata["vae_config"] == dict(config.vae_config)
    assert cached.metadata["text_encoder_config"] == dict(
        config.text_encoder_config
    )
    assert cached.metadata["text_encoder_precision"] == "bf16"
    assert cached.metadata["tokenizer_config"] == dict(config.tokenizer_config)
    assert cached.metadata["qwen_chat_template"] == dict(
        config.qwen_chat_template
    )
    assert cached.metadata["max_sequence_length"] == 32
    assert cached.metadata["schema_version"] == 1
    assert cached.metadata["image_fingerprint"]
    assert cached.metadata["caption_fingerprint"]
    assert cached.metadata["preprocessing"] == {
        "alpha_composite": "white",
        "color_mode": "RGB",
        "crop": "center_to_multiple",
        "exif_orientation": "applied",
        "height": 16,
        "pad": None,
        "resize": None,
        "width": 32,
    }


def _cache_ready_counts(caplog) -> tuple[int, int]:
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "zimage.training"
        and record.getMessage().startswith("cache ready encoded=")
    ]
    assert messages, "expected cache ready log"
    encoded_token, reused_token = messages[-1].removeprefix("cache ready ").split()
    return int(encoded_token.split("=", 1)[1]), int(reused_token.split("=", 1)[1])


def test_valid_cache_is_reused_and_stale_cache_is_replaced(tmp_path, caplog):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)
    samples = [sample]
    caplog.set_level(logging.INFO, logger="zimage.training")
    first_path = prepare_cache_at_job_start(
        samples,
        encoder,
        _config(),
        job_dir=job_dir,
    )[0]
    encoded, _reused = _cache_ready_counts(caplog)
    assert encoded > 0
    first_bytes = first_path.read_bytes()

    caplog.clear()
    reused_path = prepare_cache_at_job_start(
        samples,
        encoder,
        _config(),
        job_dir=job_dir,
    )[0]
    encoded, reused_count = _cache_ready_counts(caplog)
    assert encoded == 0
    assert reused_count == len(samples)
    assert encoder.image_calls == 1
    assert reused_path.read_bytes() == first_bytes

    stale_expected = expected_metadata(
        sample,
        _config(main_revision="different-revision"),
        image_size=cache_module.load_training_image(sample.image_path).size,
    )
    assert inspect_cache(first_path, stale_expected).state is CacheState.STALE

    replaced_path = prepare_cache_at_job_start(
        samples,
        encoder,
        _config(main_revision="different-revision"),
        job_dir=job_dir,
    )[0]
    replaced = load_cache(replaced_path)
    assert encoder.image_calls == 2
    assert replaced_path == first_path
    assert replaced.metadata["main_revision"] == "different-revision"


def test_missing_and_revision_stale_states_are_explicit(tmp_path):
    sample = _sample(tmp_path)
    job_dir = _job_dir(tmp_path)
    image = cache_module.load_training_image(sample.image_path)
    metadata = expected_metadata(sample, _config(), image_size=image.size)
    path = cache_path_for(sample, job_dir)
    assert inspect_cache(path, metadata).state is CacheState.MISSING

    prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )
    changed = expected_metadata(
        sample,
        _config(main_revision="different-revision"),
        image_size=image.size,
    )
    assert inspect_cache(path, changed).state is CacheState.STALE


def test_text_encoder_precision_mismatch_is_stale(tmp_path):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)
    image = cache_module.load_training_image(sample.image_path)
    bf16_config = _config()
    fp8_config = _config(text_encoder_precision="fp8")

    cached = load_cache(
        prepare_cache_at_job_start(
            [sample],
            encoder,
            bf16_config,
            job_dir=job_dir,
        )[0]
    )
    assert cached.metadata["text_encoder_precision"] == "bf16"
    assert cached.prompt_embedding.shape == (3, 2560)
    assert inspect_cache(
        cached.path,
        expected_metadata(sample, bf16_config, image_size=image.size),
    ).state is CacheState.VALID
    assert inspect_cache(
        cached.path,
        expected_metadata(sample, fp8_config, image_size=image.size),
    ).state is CacheState.STALE

    replaced = load_cache(
        prepare_cache_at_job_start(
            [sample],
            encoder,
            fp8_config,
            job_dir=job_dir,
        )[0]
    )
    assert encoder.image_calls == 2
    assert replaced.metadata["text_encoder_precision"] == "fp8"
    assert replaced.prompt_embedding.shape == (3, 2560)
    assert inspect_cache(
        replaced.path,
        expected_metadata(sample, fp8_config, image_size=image.size),
    ).state is CacheState.VALID
    assert inspect_cache(
        replaced.path,
        expected_metadata(sample, bf16_config, image_size=image.size),
    ).state is CacheState.STALE

    reused = load_cache(
        prepare_cache_at_job_start(
            [sample],
            encoder,
            fp8_config,
            job_dir=job_dir,
        )[0]
    )
    assert encoder.image_calls == 2
    assert reused.metadata["text_encoder_precision"] == "fp8"
    assert reused.prompt_embedding.shape == (3, 2560)


def test_old_metadata_without_text_encoder_precision_is_stale(tmp_path):
    sample = _sample(tmp_path)
    job_dir = _job_dir(tmp_path)
    cached = load_cache(
        prepare_cache_at_job_start(
            [sample],
            FakeEncoder(),
            _config(),
            job_dir=job_dir,
        )[0]
    )
    old_metadata = dict(cached.metadata)
    del old_metadata["text_encoder_precision"]
    write_cache_atomic(
        cached.path,
        cached.latent,
        cached.prompt_embedding,
        old_metadata,
    )
    image = cache_module.load_training_image(sample.image_path)
    expected = expected_metadata(sample, _config(), image_size=image.size)

    assert "text_encoder_precision" not in old_metadata
    assert expected["text_encoder_precision"] == "bf16"
    assert inspect_cache(cached.path, expected).state is CacheState.STALE
    assert inspect_cache(
        cached.path,
        expected_metadata(
            sample,
            _config(text_encoder_precision="fp8"),
            image_size=image.size,
        ),
    ).state is CacheState.STALE


def test_caption_and_image_bytes_change_cache_keys(tmp_path):
    first = _sample(tmp_path, "first.png", color=(10, 20, 30))
    same_caption = _sample(tmp_path, "second.png", color=(40, 50, 60))
    same_image = replace(first, caption="other caption")
    job_dir = _job_dir(tmp_path)
    encoder = FakeEncoder()

    assert cache_path_for(first, job_dir) != cache_path_for(same_caption, job_dir)
    assert cache_path_for(first, job_dir) != cache_path_for(same_image, job_dir)
    assert cache_path_for(same_caption, job_dir) != cache_path_for(same_image, job_dir)

    paths = prepare_cache_at_job_start(
        [first, same_caption, same_image],
        encoder,
        _config(),
        job_dir=job_dir,
    )
    assert encoder.image_calls == 3
    assert len(set(paths)) == 3
    assert set(paths) == set(_dataset_files(job_dir))
    assert not (first.dataset_path / ".cache").exists()


def test_identical_bytes_and_caption_share_one_job_file_with_repeated_paths(
    tmp_path,
):
    first = _sample(tmp_path, "first.png")
    duplicate = _sample(tmp_path, "copy.png")
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)

    paths = prepare_cache_at_job_start(
        [first, duplicate, first],
        encoder,
        _config(),
        job_dir=job_dir,
    )

    assert encoder.image_calls == 1
    assert paths == [paths[0], paths[0], paths[0]]
    assert _dataset_files(job_dir) == [paths[0]]
    assert not (first.dataset_path / ".cache").exists()


def test_same_samples_in_different_jobs_use_independent_files(tmp_path):
    sample = _sample(tmp_path)
    first_job = _job_dir(tmp_path, "job-a")
    second_job = _job_dir(tmp_path, "job-b")

    first = prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        job_dir=first_job,
    )[0]
    second = prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        job_dir=second_job,
    )[0]

    assert first != second
    assert first.name == second.name
    assert first.parent == first_job / ".cache" / "dataset"
    assert second.parent == second_job / ".cache" / "dataset"
    assert first.is_file()
    assert second.is_file()

    shutil.rmtree(first_job / ".cache")
    assert not first.exists()
    assert second.is_file()
    assert not (sample.dataset_path / ".cache").exists()


def test_nested_images_use_job_cache_without_dataset_directories(tmp_path):
    dataset = tmp_path / "dataset"
    first_dir = dataset / "first"
    second_dir = dataset / "second" / "deeper"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    Image.new("RGB", (32, 16), "red").save(first_dir / "same.png")
    Image.new("RGB", (32, 16), "blue").save(second_dir / "same.png")
    samples = discover_samples(
        [{"name": dataset, "default_caption": "nested image"}],
        tmp_path,
    )
    job_dir = _job_dir(tmp_path)

    paths = prepare_cache_at_job_start(
        samples,
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )

    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert set(paths) == set(_dataset_files(job_dir))
    assert not (dataset / ".cache").exists()
    assert not (first_dir / ".cache").exists()
    assert not (second_dir / ".cache").exists()


@pytest.mark.parametrize(
    ("latent", "prompt", "message"),
    [
        (
            torch.zeros((16, 2, 4), dtype=torch.float32),
            torch.zeros((3, 2560), dtype=torch.bfloat16),
            "latent dtype",
        ),
        (
            torch.zeros((15, 2, 4), dtype=torch.bfloat16),
            torch.zeros((3, 2560), dtype=torch.bfloat16),
            "latent shape",
        ),
        (
            torch.zeros((16, 2, 4), dtype=torch.bfloat16),
            torch.zeros((3, 2559), dtype=torch.bfloat16),
            "hidden size",
        ),
        (
            torch.zeros((16, 2, 4), dtype=torch.bfloat16),
            torch.zeros((33, 2560), dtype=torch.bfloat16),
            "valid-token count",
        ),
    ],
)
def test_write_rejects_invalid_tensor_contract(
    tmp_path,
    latent,
    prompt,
    message,
):
    sample = _sample(tmp_path)
    image = cache_module.load_training_image(sample.image_path)
    metadata = expected_metadata(sample, _config(), image_size=image.size)

    with pytest.raises(CacheError, match=message):
        write_cache_atomic(
            tmp_path / "invalid.safetensors",
            latent,
            prompt,
            metadata,
        )


def test_read_rejects_invalid_tensor_dtype(tmp_path):
    sample = _sample(tmp_path)
    image = cache_module.load_training_image(sample.image_path)
    metadata = expected_metadata(sample, _config(), image_size=image.size)
    path = tmp_path / "bad.safetensors"
    cache_module.save_file(
        {
            "latent": torch.zeros((16, 2, 4), dtype=torch.float32),
            "prompt_embedding": torch.zeros((3, 2560), dtype=torch.bfloat16),
        },
        str(path),
        metadata={
            cache_module.METADATA_KEY: cache_module._canonical_json(metadata)
        },
    )

    with pytest.raises(CacheError, match="latent dtype"):
        load_cache(path)


def test_interrupted_atomic_replacement_keeps_valid_final_file(
    tmp_path,
    monkeypatch,
):
    sample = _sample(tmp_path)
    job_dir = _job_dir(tmp_path)
    encoder = FakeEncoder()
    original_path = prepare_cache_at_job_start(
        [sample],
        encoder,
        _config(),
        job_dir=job_dir,
    )[0]
    original = load_cache(original_path)
    original_bytes = original.path.read_bytes()
    real_save = cache_module.save_file

    def save_then_interrupt(tensors, filename, metadata):
        real_save(tensors, filename, metadata=metadata)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cache_module, "save_file", save_then_interrupt)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_cache_at_job_start(
            [sample],
            encoder,
            _config(main_revision="different-revision"),
            job_dir=job_dir,
        )

    assert original.path.read_bytes() == original_bytes
    assert list((job_dir / ".cache" / "dataset").glob("*.tmp")) == []
    assert load_cache(original.path).metadata == original.metadata


def test_17x16_encodes_with_cropped_latent_and_crop_token(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image_path = dataset / "odd.png"
    Image.new("RGB", (17, 16)).save(image_path)
    sample = DatasetSample(
        image_path=image_path,
        caption="caption",
        dataset_path=dataset,
    )
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)

    cached = load_cache(
        prepare_cache_at_job_start(
            [sample],
            encoder,
            _config(),
            job_dir=job_dir,
        )[0]
    )

    assert encoder.image_calls == 1
    assert cached.latent.shape == (16, 2, 2)
    assert cached.metadata["preprocessing"]["crop"] == "center_to_multiple"
    assert cached.metadata["preprocessing"]["width"] == 16
    assert cached.metadata["preprocessing"]["height"] == 16


def test_undersized_image_fails_before_injected_encoder(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image_path = dataset / "tiny.png"
    Image.new("RGB", (15, 16)).save(image_path)
    sample = DatasetSample(
        image_path=image_path,
        caption="caption",
        dataset_path=dataset,
    )
    encoder = FakeEncoder()

    with pytest.raises(DatasetError, match="at least"):
        prepare_cache_at_job_start(
            [sample],
            encoder,
            _config(),
            job_dir=_job_dir(tmp_path),
        )

    assert encoder.image_calls == 0
    assert encoder.prompt_calls == 0


def test_inspect_cache_treats_crop_none_payload_as_stale(tmp_path):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)
    cached = load_cache(
        prepare_cache_at_job_start(
            [sample],
            encoder,
            _config(),
            job_dir=job_dir,
        )[0]
    )
    stale_metadata = dict(cached.metadata)
    stale_preprocessing = dict(stale_metadata["preprocessing"])
    stale_preprocessing["crop"] = None
    stale_metadata["preprocessing"] = stale_preprocessing
    write_cache_atomic(
        cached.path,
        cached.latent,
        cached.prompt_embedding,
        stale_metadata,
    )
    image = cache_module.load_training_image(sample.image_path)
    expected = expected_metadata(sample, _config(), image_size=image.size)

    assert expected["preprocessing"]["crop"] == "center_to_multiple"
    assert inspect_cache(cached.path, expected).state is CacheState.STALE


def test_on_before_encode_runs_once_on_stale_and_never_on_valid(tmp_path):
    valid_sample = _sample(tmp_path, "valid.png")
    stale_sample = _sample(tmp_path, "stale.png", color=(1, 2, 3))
    job_dir = _job_dir(tmp_path)
    prepare_cache_at_job_start(
        [valid_sample],
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )

    events: list[str] = []

    class TrackingEncoder(FakeEncoder):
        def encode_image(self, image):
            events.append("encode")
            return super().encode_image(image)

    encoder = TrackingEncoder()

    def on_before_encode() -> None:
        events.append("before")

    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        job_dir=job_dir,
        on_before_encode=on_before_encode,
    )
    assert events == ["before", "encode"]
    assert encoder.image_calls == 1

    events.clear()
    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        job_dir=job_dir,
        on_before_encode=on_before_encode,
    )
    assert events == []
    assert encoder.image_calls == 1


def test_on_after_first_encode_runs_once_after_first_stale_encode(tmp_path):
    valid_sample = _sample(tmp_path, "valid.png")
    stale_sample = _sample(tmp_path, "stale.png", color=(1, 2, 3))
    job_dir = _job_dir(tmp_path)
    prepare_cache_at_job_start(
        [valid_sample],
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )

    events: list[str] = []

    class TrackingEncoder(FakeEncoder):
        def encode_image(self, image):
            events.append("encode")
            return super().encode_image(image)

    encoder = TrackingEncoder()

    def on_after_first_encode() -> None:
        events.append("after")

    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        job_dir=job_dir,
        on_after_first_encode=on_after_first_encode,
    )
    assert events == ["encode", "after"]
    assert encoder.image_calls == 1

    events.clear()
    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        job_dir=job_dir,
        on_after_first_encode=on_after_first_encode,
    )
    assert events == []
    assert encoder.image_calls == 1


def test_vae_output_is_released_before_prompt_encode(tmp_path):
    first = _sample(tmp_path, "one.png")
    second = _sample(tmp_path, "two.png", color=(9, 8, 7))
    refs: dict[str, weakref.ref] = {}
    prompt_calls = {"n": 0}

    class TrackingEncoder(FakeEncoder):
        def encode_image(self, image):
            encoding = super().encode_image(image)
            refs["encoding"] = weakref.ref(encoding)
            refs["distribution"] = weakref.ref(encoding.latent_dist)
            refs["latent"] = weakref.ref(encoding.latent_dist._latent)
            return encoding

        def encode_prompt(self, caption, *, max_sequence_length):
            gc.collect()
            assert refs["encoding"]() is None
            assert refs["distribution"]() is None
            assert refs["latent"]() is None
            prompt_calls["n"] += 1
            return super().encode_prompt(
                caption,
                max_sequence_length=max_sequence_length,
            )

    prepare_cache_at_job_start(
        [first, second],
        TrackingEncoder(),
        _config(),
        job_dir=_job_dir(tmp_path),
    )
    assert prompt_calls["n"] == 2


def test_prepare_does_not_call_empty_cache_per_sample(tmp_path, monkeypatch):
    sample = _sample(tmp_path, "one.png")
    other = _sample(tmp_path, "two.png", color=(9, 8, 7))
    calls = {"count": 0}

    def forbidden() -> None:
        calls["count"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden, raising=False)
    prepare_cache_at_job_start(
        [sample, other],
        FakeEncoder(),
        _config(),
        job_dir=_job_dir(tmp_path),
    )
    assert calls["count"] == 0


def test_previous_pair_tensors_are_released_before_next_encode(tmp_path, monkeypatch):
    first = _sample(tmp_path, "one.png")
    second = _sample(tmp_path, "two.png", color=(9, 8, 7))
    previous: list[weakref.ref] = []
    real_encode = cache_module.encode_sample

    def tracking_encode(*args, **kwargs):
        gc.collect()
        for ref in previous:
            assert ref() is None
        latent, prompt_embedding = real_encode(*args, **kwargs)
        previous[:] = [weakref.ref(latent), weakref.ref(prompt_embedding)]
        return latent, prompt_embedding

    monkeypatch.setattr(cache_module, "encode_sample", tracking_encode)
    prepare_cache_at_job_start(
        [first, second],
        FakeEncoder(),
        _config(),
        job_dir=_job_dir(tmp_path),
    )
    gc.collect()
    for ref in previous:
        assert ref() is None


def test_scattered_dataset_cache_is_neither_deleted_nor_migrated(tmp_path):
    sample = _sample(tmp_path)
    planted = sample.dataset_path / ".cache" / "image.png.safetensors"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"planted-scattered-cache")
    planted_stat = planted.stat()
    job_dir = _job_dir(tmp_path)

    paths = prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )

    assert planted.is_file()
    assert planted.read_bytes() == b"planted-scattered-cache"
    assert planted.stat().st_mtime_ns == planted_stat.st_mtime_ns
    assert planted.stat().st_size == planted_stat.st_size
    job_files = _dataset_files(job_dir)
    assert job_files == [paths[0]]
    assert planted.resolve() != paths[0].resolve()
    assert paths[0].read_bytes() != b"planted-scattered-cache"
    assert not any(
        child.read_bytes() == b"planted-scattered-cache"
        for child in (job_dir / ".cache").rglob("*")
        if child.is_file()
    )


def test_preview_prompt_file_contains_only_embedding_and_prompt_metadata(tmp_path):
    encoder = FakeEncoder()
    config = _config()
    job_dir = _job_dir(tmp_path)
    prompt = "a preview prompt"

    paths = prepare_preview_prompt_cache(
        [prompt],
        encoder,
        config,
        job_dir=job_dir,
    )
    embedding, metadata = load_preview_cache(paths[prompt])
    tensors, _loaded = cache_module._read_tensors_cpu(paths[prompt])

    assert encoder.image_calls == 0
    assert encoder.prompt_calls == 1
    assert encoder.captions == [prompt]
    assert set(paths) == {prompt}
    assert isinstance(paths[prompt], Path)
    assert paths[prompt].parent == job_dir / ".cache" / "preview"
    assert paths[prompt].suffix == ".safetensors"
    assert paths[prompt].stem == cache_module.fingerprint_caption(prompt)
    assert len(paths[prompt].stem) == 64
    assert set(tensors) == {cache_module.PROMPT_EMBEDDING_KEY}
    expected_prompt = torch.arange(3 * 2560, dtype=torch.float32)
    expected_prompt = expected_prompt.reshape(3, 2560).to(torch.bfloat16)
    assert embedding.dtype is torch.bfloat16
    assert embedding.shape == (3, 2560)
    assert torch.equal(embedding, expected_prompt)
    assert metadata["cache_kind"] == "preview"
    assert metadata["prompt_fingerprint"] == cache_module.fingerprint_caption(prompt)
    assert metadata["main_revision"] == "resolved-commit-sha"
    assert metadata["text_encoder_config"] == dict(config.text_encoder_config)
    assert metadata["text_encoder_precision"] == "bf16"
    assert metadata["tokenizer_config"] == dict(config.tokenizer_config)
    assert metadata["qwen_chat_template"] == dict(config.qwen_chat_template)
    assert metadata["max_sequence_length"] == 32
    assert metadata["schema_version"] == 1
    assert "vae_config" not in metadata
    assert "preprocessing" not in metadata
    assert "image_fingerprint" not in metadata
    assert "caption_fingerprint" not in metadata
    assert all(not isinstance(value, torch.Tensor) for value in paths.values())


def test_preview_same_text_for_positive_and_negative_shares_one_file(tmp_path):
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)
    shared = "same prompt text"

    paths = prepare_preview_prompt_cache(
        [shared, shared, ""],
        encoder,
        _config(),
        job_dir=job_dir,
    )

    assert encoder.prompt_calls == 1
    assert encoder.image_calls == 0
    assert paths == {shared: preview_cache_path(job_dir, shared)}
    assert _preview_files(job_dir) == [paths[shared]]
    assert "" not in paths


def test_preview_empty_negative_is_not_encoded_or_written(tmp_path):
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)

    paths = prepare_preview_prompt_cache(
        ["positive", "", "other"],
        encoder,
        _config(),
        job_dir=job_dir,
    )

    assert encoder.prompt_calls == 2
    assert encoder.captions == ["positive", "other"]
    assert set(paths) == {"positive", "other"}
    assert "" not in paths
    assert _preview_files(job_dir) == sorted(paths.values())


def test_preview_missing_and_stale_are_reencoded_one_at_a_time(tmp_path, caplog):
    encoder = FakeEncoder()
    job_dir = _job_dir(tmp_path)
    prompts = ["first", "second"]
    caplog.set_level(logging.INFO, logger="zimage.training")

    first_paths = prepare_preview_prompt_cache(
        prompts,
        encoder,
        _config(),
        job_dir=job_dir,
    )
    encoded, reused = _preview_ready_counts(caplog)
    assert encoded == 2
    assert reused == 0
    assert encoder.prompt_calls == 2
    first_bytes = {text: path.read_bytes() for text, path in first_paths.items()}

    caplog.clear()
    reused_paths = prepare_preview_prompt_cache(
        prompts,
        encoder,
        _config(),
        job_dir=job_dir,
    )
    encoded, reused = _preview_ready_counts(caplog)
    assert encoded == 0
    assert reused == 2
    assert encoder.prompt_calls == 2
    assert reused_paths == first_paths
    assert {
        text: path.read_bytes() for text, path in reused_paths.items()
    } == first_bytes

    stale = expected_preview_metadata("first", _config(main_revision="other"))
    assert inspect_preview_cache(first_paths["first"], stale).state is CacheState.STALE
    assert inspect_preview_cache(
        preview_cache_path(job_dir, "missing"),
        expected_preview_metadata("missing", _config()),
    ).state is CacheState.MISSING

    caplog.clear()
    replaced = prepare_preview_prompt_cache(
        prompts,
        encoder,
        _config(main_revision="other"),
        job_dir=job_dir,
    )
    encoded, reused = _preview_ready_counts(caplog)
    assert encoded == 2
    assert reused == 0
    assert encoder.prompt_calls == 4
    embedding, metadata = load_preview_cache(replaced["first"])
    del embedding
    assert metadata["main_revision"] == "other"
    assert inspect_preview_cache(
        replaced["first"],
        expected_preview_metadata("first", _config(main_revision="other")),
    ).state is CacheState.VALID


def test_preview_prepare_returns_paths_and_releases_tensors(tmp_path, monkeypatch):
    previous: list[weakref.ref] = []
    real_encode = cache_module.encode_prompt_embedding

    def tracking_encode(*args, **kwargs):
        gc.collect()
        for ref in previous:
            assert ref() is None
        embedding = real_encode(*args, **kwargs)
        previous[:] = [weakref.ref(embedding)]
        return embedding

    monkeypatch.setattr(cache_module, "encode_prompt_embedding", tracking_encode)
    paths = prepare_preview_prompt_cache(
        ["one", "two"],
        FakeEncoder(),
        _config(),
        job_dir=_job_dir(tmp_path),
    )
    gc.collect()
    for ref in previous:
        assert ref() is None
    assert all(isinstance(path, Path) for path in paths.values())
    assert all(not isinstance(value, torch.Tensor) for value in paths.values())


def test_preview_write_rejects_invalid_prompt_tensor(tmp_path):
    metadata = expected_preview_metadata("prompt", _config())
    with pytest.raises(CacheError, match="hidden size"):
        write_preview_cache_atomic(
            tmp_path / "invalid.safetensors",
            torch.zeros((3, 2559), dtype=torch.bfloat16),
            metadata,
        )


def test_preview_load_rejects_dataset_tensor_keys(tmp_path):
    sample = _sample(tmp_path)
    job_dir = _job_dir(tmp_path)
    dataset_path = prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        job_dir=job_dir,
    )[0]
    metadata = expected_preview_metadata("a caption", _config())

    with pytest.raises(CacheError, match="preview cache tensor keys"):
        load_preview_cache(dataset_path, expected=metadata)
    assert inspect_preview_cache(dataset_path, metadata).state is CacheState.STALE
