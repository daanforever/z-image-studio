from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageOps

from zimage.training.dataset import (
    DatasetError,
    center_crop_box,
    discover_samples,
    load_training_image,
    resolve_dataset_path,
)


def _save_image(
    path: Path,
    *,
    mode: str = "RGB",
    color=(20, 40, 60),
    size: tuple[int, int] = (32, 16),
) -> Path:
    Image.new(mode, size, color).save(path)
    return path


def test_discovery_merges_relative_and_absolute_datasets_and_formats(tmp_path):
    datasets_dir = tmp_path / "datasets"
    first = datasets_dir / "first"
    second = tmp_path / "elsewhere"
    first.mkdir(parents=True)
    second.mkdir()

    expected = [
        _save_image(first / "a.PNG"),
        _save_image(first / "b.jpg"),
        _save_image(first / "c.JpEg"),
        _save_image(second / "d.WeBp"),
    ]
    for image in expected:
        image.with_suffix(".txt").write_text(f"caption {image.stem}", encoding="utf-8")
    _save_image(first / "ignored.bmp")
    ignored_cache = first / ".cache"
    ignored_cache.mkdir()
    _save_image(ignored_cache / "ignored.png")

    samples = discover_samples(
        [
            {"name": "first", "default_caption": ""},
            {"name": str(second.resolve()), "default_caption": ""},
        ],
        datasets_dir,
    )

    assert {sample.image_path for sample in samples} == {
        path.resolve() for path in expected
    }
    assert all(sample.caption.startswith("caption ") for sample in samples)
    assert resolve_dataset_path("first", datasets_dir) == first.resolve()
    assert resolve_dataset_path(second.resolve(), datasets_dir) == second.resolve()


def test_caption_sidecar_default_fallback_and_missing_error(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sidecar_image = _save_image(dataset / "sidecar.png")
    default_image = _save_image(dataset / "default.png")
    blank_image = _save_image(dataset / "blank.png")
    sidecar_image.with_suffix(".txt").write_text("  sidecar caption \n", encoding="utf-8")
    blank_image.with_suffix(".txt").write_text(" \n", encoding="utf-8")

    samples = discover_samples(
        [{"name": dataset, "default_caption": "  fallback caption "}],
        tmp_path,
    )
    by_name = {sample.image_path.name: sample.caption for sample in samples}
    assert by_name == {
        "blank.png": "fallback caption",
        "default.png": "fallback caption",
        "sidecar.png": "sidecar caption",
    }

    default_image.unlink()
    sidecar_image.unlink()
    with pytest.raises(DatasetError, match="blank.png"):
        discover_samples(
            [{"name": dataset, "default_caption": "  "}],
            tmp_path,
        )


def test_invalid_utf8_caption_names_image(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = _save_image(dataset / "bad.png")
    image.with_suffix(".txt").write_bytes(b"\xff")

    with pytest.raises(DatasetError, match="bad.png"):
        discover_samples(
            [{"name": dataset, "default_caption": "fallback"}],
            tmp_path,
        )


def test_exif_orientation_is_applied_without_resizing(tmp_path):
    image_path = tmp_path / "oriented.jpg"
    source = Image.new("RGB", (32, 16), "red")
    source.paste("blue", (0, 0, 16, 16))
    exif = source.getexif()
    exif[274] = 6
    source.save(image_path, exif=exif)

    with Image.open(image_path) as stored:
        expected = ImageOps.exif_transpose(stored).convert("RGB")
        expected.load()
    loaded = load_training_image(image_path)

    assert loaded.mode == "RGB"
    assert loaded.size == (16, 32)
    assert loaded.tobytes() == expected.tobytes()


def test_alpha_is_composited_on_white_and_output_is_rgb(tmp_path):
    image_path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 128))
    image.save(image_path)

    loaded = load_training_image(image_path)

    assert loaded.mode == "RGB"
    assert loaded.getpixel((1, 1)) == (255, 255, 255)
    assert loaded.getpixel((0, 0)) == (255, 127, 127)


@pytest.mark.parametrize(
    ("width", "height", "box"),
    [
        (17, 16, (0, 0, 16, 16)),
        (19, 32, (1, 0, 17, 32)),
        (1280, 853, (0, 2, 1280, 850)),
        (32, 16, (0, 0, 32, 16)),
    ],
)
def test_center_crop_box_geometry(width, height, box):
    assert center_crop_box(width, height) == box


@pytest.mark.parametrize("size", [(15, 16), (8, 8)])
def test_center_crop_box_rejects_undersized_sides(size):
    with pytest.raises(DatasetError, match="at least"):
        center_crop_box(*size)


def test_load_training_image_center_crops_19x32_edge_colors(tmp_path):
    image_path = tmp_path / "edges.png"
    source = Image.new("RGB", (19, 32), (10, 20, 30))
    for y in range(32):
        source.putpixel((0, y), (255, 0, 0))
        source.putpixel((1, y), (0, 255, 0))
        source.putpixel((16, y), (0, 0, 255))
        source.putpixel((18, y), (255, 255, 0))
    source.save(image_path)

    loaded = load_training_image(image_path)

    assert loaded.size == (16, 32)
    assert loaded.getpixel((0, 0)) == (0, 255, 0)
    assert loaded.getpixel((15, 0)) == (0, 0, 255)
    assert loaded.tobytes() == source.crop(center_crop_box(19, 32)).tobytes()


def test_undersized_image_names_source_size_and_path(tmp_path):
    image_path = _save_image(tmp_path / "tiny.png", size=(15, 16))

    with pytest.raises(DatasetError, match="at least") as caught:
        load_training_image(image_path)

    message = str(caught.value)
    assert "15x16" in message
    assert "tiny.png" in message


def test_17x16_is_center_cropped_not_rejected_as_not_divisible(tmp_path):
    image_path = _save_image(tmp_path / "odd.png", size=(17, 16))

    loaded = load_training_image(image_path)

    assert loaded.size == (16, 16)


def test_mvp_rejects_batching_and_accumulation(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(DatasetError, match="batch_size=1"):
        discover_samples([], dataset, batch_size=2)
    with pytest.raises(DatasetError, match="gradient_accumulation=1"):
        discover_samples([], dataset, gradient_accumulation=2)
