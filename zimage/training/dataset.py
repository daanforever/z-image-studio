"""Pure discovery and preprocessing for captioned training datasets.

This module deliberately owns no model objects.  It turns configured dataset
directories into a flat, deterministic list of samples and prepares images for
an injected cache encoder.  Geometry is center-crop-to-multiple only: no pad,
downscale, or resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
IMAGE_SIZE_MULTIPLE = 16


class DatasetError(ValueError):
    """A configured training dataset or one of its samples is invalid."""


@dataclass(frozen=True)
class DatasetSample:
    """One discovered image and its resolved, non-empty caption."""

    image_path: Path
    caption: str
    dataset_path: Path

    @property
    def sample_id(self) -> str:
        return str(self.image_path.resolve())


def validate_mvp_batch_settings(
    *,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
) -> None:
    """Reject batching modes that the physical-batch-one MVP cannot represent."""

    if batch_size != 1:
        raise DatasetError("MVP training requires batch_size=1")
    if gradient_accumulation != 1:
        raise DatasetError("MVP training requires gradient_accumulation=1")


def resolve_dataset_path(name: str | Path, datasets_dir: str | Path) -> Path:
    """Resolve a dataset name under ``datasets_dir`` or accept an absolute path."""

    raw = Path(name).expanduser()
    resolved = raw if raw.is_absolute() else Path(datasets_dir).expanduser() / raw
    return resolved.resolve()


def discover_samples(
    datasets: Iterable[Mapping[str, Any]],
    datasets_dir: str | Path,
    *,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
) -> list[DatasetSample]:
    """Merge configured directories into one flat, deterministic sample pool."""

    validate_mvp_batch_settings(
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    samples: list[DatasetSample] = []
    for index, item in enumerate(datasets):
        if not isinstance(item, Mapping):
            raise DatasetError(f"datasets[{index}] must be a mapping")
        name = item.get("name")
        if not isinstance(name, (str, Path)) or not str(name).strip():
            raise DatasetError(f"datasets[{index}].name must be a non-empty path")
        default_caption = item.get("default_caption", "")
        if not isinstance(default_caption, str):
            raise DatasetError(f"datasets[{index}].default_caption must be a string")

        dataset_path = resolve_dataset_path(name, datasets_dir)
        if not dataset_path.is_dir():
            raise DatasetError(f"dataset directory does not exist: {dataset_path}")
        image_paths = sorted(
            (
                path
                for path in dataset_path.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in IMAGE_EXTENSIONS
                and not _inside_cache_directory(path, dataset_path)
            ),
            key=lambda path: str(path.relative_to(dataset_path)).casefold(),
        )
        samples.extend(
            DatasetSample(
                image_path=image_path.resolve(),
                caption=_resolve_caption(image_path, default_caption),
                dataset_path=dataset_path,
            )
            for image_path in image_paths
        )
    return samples


def discover_dataset_samples(
    datasets: Iterable[Mapping[str, Any]],
    datasets_dir: str | Path,
    *,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
) -> list[DatasetSample]:
    """Compatibility spelling for :func:`discover_samples`."""

    return discover_samples(
        datasets,
        datasets_dir,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
    )


def center_crop_box(
    width: int,
    height: int,
    *,
    size_multiple: int = IMAGE_SIZE_MULTIPLE,
) -> tuple[int, int, int, int]:
    """Return a PIL box that center-crops each side to a ``size_multiple``.

    Extra remainder pixels fall on the right and bottom.  Raises
    :class:`DatasetError` if either cropped side would be smaller than
    ``size_multiple``.
    """

    new_width = width - (width % size_multiple)
    new_height = height - (height % size_multiple)
    if new_width < size_multiple or new_height < size_multiple:
        raise DatasetError(
            f"image sides must be at least {size_multiple}px ({width}x{height})"
        )
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return (left, top, left + new_width, top + new_height)


def load_training_image(
    image_path: str | Path,
    *,
    size_multiple: int = IMAGE_SIZE_MULTIPLE,
) -> Image.Image:
    """Load, orient, white-composite, center-crop, and return an RGB image.

    Geometry is center-crop-to-multiple only: no pad, downscale, or resize.
    Validation runs on the cropped size only, before any encoder can be called.
    """

    path = Path(image_path)
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            oriented.load()
            rgb = _composite_on_white(oriented)
            width, height = rgb.size
            try:
                box = center_crop_box(
                    width,
                    height,
                    size_multiple=size_multiple,
                )
            except DatasetError:
                raise DatasetError(
                    f"image sides must be at least {size_multiple}px "
                    f"({width}x{height}): {path}"
                ) from None
            cropped = rgb.crop(box)
            cropped.load()
            validate_image_dimensions(
                path,
                cropped.size[0],
                cropped.size[1],
                size_multiple=size_multiple,
            )
            return cropped
    except DatasetError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot load training image: {path}") from exc


def validate_image_dimensions(
    image_path: str | Path,
    width: int,
    height: int,
    *,
    size_multiple: int = IMAGE_SIZE_MULTIPLE,
) -> None:
    """Enforce cropped-image limits used by the cache encoder."""

    path = Path(image_path)
    if width < size_multiple or height < size_multiple:
        raise DatasetError(
            f"image sides must be at least {size_multiple}px ({width}x{height}): {path}"
        )
    if width % size_multiple or height % size_multiple:
        raise DatasetError(
            f"image sides must be divisible by {size_multiple} "
            f"({width}x{height}): {path}"
        )


def _inside_cache_directory(path: Path, dataset_path: Path) -> bool:
    relative_parts = path.relative_to(dataset_path).parts[:-1]
    return any(part.casefold() == ".cache" for part in relative_parts)


def _resolve_caption(image_path: Path, default_caption: str) -> str:
    sidecar = _find_caption_sidecar(image_path)
    if sidecar is not None:
        try:
            caption = sidecar.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise DatasetError(f"cannot read UTF-8 caption for image: {image_path}") from exc
        if not caption:
            caption = default_caption.strip()
    else:
        caption = default_caption.strip()
    if not caption:
        raise DatasetError(f"missing non-empty caption for image: {image_path}")
    return caption


def _find_caption_sidecar(image_path: Path) -> Path | None:
    exact = image_path.with_suffix(".txt")
    if exact.is_file():
        return exact
    expected = f"{image_path.stem}.txt".casefold()
    try:
        return next(
            (
                sibling
                for sibling in image_path.parent.iterdir()
                if sibling.is_file() and sibling.name.casefold() == expected
            ),
            None,
        )
    except OSError as exc:
        raise DatasetError(f"cannot inspect caption for image: {image_path}") from exc


def _composite_on_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")
