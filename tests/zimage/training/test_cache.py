from __future__ import annotations

import logging
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
    inspect_cache,
    load_cache,
    prepare_cache_at_job_start,
    write_cache_atomic,
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


def _sample(tmp_path: Path, name: str = "image.png") -> DatasetSample:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    image_path = dataset / name
    Image.new("RGB", (32, 16), (10, 20, 30)).save(image_path)
    return DatasetSample(
        image_path=image_path.resolve(),
        caption="a caption",
        dataset_path=dataset.resolve(),
    )


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


def test_exact_tensors_and_complete_metadata_round_trip(tmp_path):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    config = _config()

    cached = prepare_cache_at_job_start(
        [sample],
        encoder,
        config,
        cache_dir=tmp_path / "cache",
    )[0]

    assert encoder.image_calls == 1
    assert encoder.prompt_calls == 1
    assert encoder.captions == ["a caption"]
    assert cached.path.name == "image.png.safetensors"
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
    assert cached.metadata["main_revision"] == "resolved-commit-sha"
    assert cached.metadata["vae_config"] == dict(config.vae_config)
    assert cached.metadata["text_encoder_config"] == dict(
        config.text_encoder_config
    )
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
    cache_dir = tmp_path / "cache"
    samples = [sample]
    caplog.set_level(logging.INFO, logger="zimage.training")
    first = prepare_cache_at_job_start(
        samples,
        encoder,
        _config(),
        cache_dir=cache_dir,
    )[0]
    encoded, _reused = _cache_ready_counts(caplog)
    assert encoded > 0
    first_bytes = first.path.read_bytes()

    caplog.clear()
    reused = prepare_cache_at_job_start(
        samples,
        encoder,
        _config(),
        cache_dir=cache_dir,
    )[0]
    encoded, reused_count = _cache_ready_counts(caplog)
    assert encoded == 0
    assert reused_count == len(samples)
    assert encoder.image_calls == 1
    assert reused.path.read_bytes() == first_bytes

    stale_sample = replace(sample, caption="changed caption")
    image = cache_module.load_training_image(stale_sample.image_path)
    stale_expected = expected_metadata(
        stale_sample,
        _config(),
        image_size=image.size,
    )
    assert inspect_cache(first.path, stale_expected).state is CacheState.STALE

    replaced = prepare_cache_at_job_start(
        [stale_sample],
        encoder,
        _config(),
        cache_dir=cache_dir,
    )[0]
    assert encoder.image_calls == 2
    assert encoder.captions[-1] == "changed caption"
    assert replaced.metadata["caption_fingerprint"] != first.metadata[
        "caption_fingerprint"
    ]


def test_missing_and_revision_stale_states_are_explicit(tmp_path):
    sample = _sample(tmp_path)
    image = cache_module.load_training_image(sample.image_path)
    metadata = expected_metadata(sample, _config(), image_size=image.size)
    path = cache_path_for(sample.image_path, tmp_path / "cache")
    assert inspect_cache(path, metadata).state is CacheState.MISSING

    prepare_cache_at_job_start(
        [sample],
        FakeEncoder(),
        _config(),
        cache_dir=tmp_path / "cache",
    )
    changed = expected_metadata(
        sample,
        _config(main_revision="different-revision"),
        image_size=image.size,
    )
    assert inspect_cache(path, changed).state is CacheState.STALE


def test_cache_names_preserve_image_extensions(tmp_path):
    png = tmp_path / "same.png"
    jpg = tmp_path / "same.jpg"

    assert cache_path_for(png, tmp_path / "cache").name == "same.png.safetensors"
    assert cache_path_for(jpg, tmp_path / "cache").name == "same.jpg.safetensors"
    assert cache_path_for(png, tmp_path / "cache") != cache_path_for(
        jpg,
        tmp_path / "cache",
    )


def test_nested_images_use_one_dataset_root_cache_without_collisions(tmp_path):
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

    cached = prepare_cache_at_job_start(samples, FakeEncoder(), _config())
    paths = {item.path for item in cached}

    assert paths == {
        dataset / ".cache" / "first" / "same.png.safetensors",
        dataset
        / ".cache"
        / "second"
        / "deeper"
        / "same.png.safetensors",
    }
    assert not (first_dir / ".cache").exists()
    assert not (second_dir / ".cache").exists()
    override = tmp_path / "explicit-cache"
    assert cache_path_for(samples[0], override) == (
        override
        / samples[0].image_path.relative_to(samples[0].dataset_path).parent
        / "same.png.safetensors"
    )


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
    cache_dir = tmp_path / "cache"
    encoder = FakeEncoder()
    original = prepare_cache_at_job_start(
        [sample],
        encoder,
        _config(),
        cache_dir=cache_dir,
    )[0]
    original_bytes = original.path.read_bytes()
    real_save = cache_module.save_file

    def save_then_interrupt(tensors, filename, metadata):
        real_save(tensors, filename, metadata=metadata)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cache_module, "save_file", save_then_interrupt)
    stale = replace(sample, caption="new caption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_cache_at_job_start(
            [stale],
            encoder,
            _config(),
            cache_dir=cache_dir,
        )

    assert original.path.read_bytes() == original_bytes
    assert list(cache_dir.glob("*.tmp")) == []
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

    cached = prepare_cache_at_job_start(
        [sample],
        encoder,
        _config(),
        cache_dir=tmp_path / "cache",
    )[0]

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
            cache_dir=tmp_path / "cache",
        )

    assert encoder.image_calls == 0
    assert encoder.prompt_calls == 0


def test_inspect_cache_treats_crop_none_payload_as_stale(tmp_path):
    sample = _sample(tmp_path)
    encoder = FakeEncoder()
    cached = prepare_cache_at_job_start(
        [sample],
        encoder,
        _config(),
        cache_dir=tmp_path / "cache",
    )[0]
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
    stale_sample = _sample(tmp_path, "stale.png")
    cache_dir = tmp_path / "cache"
    prepare_cache_at_job_start(
        [valid_sample],
        FakeEncoder(),
        _config(),
        cache_dir=cache_dir,
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
        cache_dir=cache_dir,
        on_before_encode=on_before_encode,
    )
    assert events == ["before", "encode"]
    assert encoder.image_calls == 1

    events.clear()
    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        cache_dir=cache_dir,
        on_before_encode=on_before_encode,
    )
    assert events == []
    assert encoder.image_calls == 1


def test_on_after_first_encode_runs_once_after_first_stale_encode(tmp_path):
    valid_sample = _sample(tmp_path, "valid.png")
    stale_sample = _sample(tmp_path, "stale.png")
    cache_dir = tmp_path / "cache"
    prepare_cache_at_job_start(
        [valid_sample],
        FakeEncoder(),
        _config(),
        cache_dir=cache_dir,
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
        cache_dir=cache_dir,
        on_after_first_encode=on_after_first_encode,
    )
    assert events == ["encode", "after"]
    assert encoder.image_calls == 1

    events.clear()
    prepare_cache_at_job_start(
        [valid_sample, stale_sample],
        encoder,
        _config(),
        cache_dir=cache_dir,
        on_after_first_encode=on_after_first_encode,
    )
    assert events == []
    assert encoder.image_calls == 1
