from __future__ import annotations

import warnings

from huggingface_hub.utils import _validators

from tests.zimage.training.test_modeling import make_loaders
from zimage.training.hub_compat import install_hf_local_dir_symlinks_compat
from zimage.training.modeling import load_training_components


def test_install_strips_local_dir_use_symlinks_without_warning():
    install_hf_local_dir_symlinks_compat()
    payload = {"local_dir_use_symlinks": "auto", "filename": "x"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _validators.smoothly_deprecate_legacy_arguments(
            "hf_hub_download", payload
        )

    assert payload == {"local_dir_use_symlinks": "auto", "filename": "x"}
    assert "local_dir_use_symlinks" not in result
    assert result["filename"] == "x"
    assert not any("local_dir_use_symlinks" in str(item.message) for item in caught)


def test_install_is_idempotent():
    install_hf_local_dir_symlinks_compat()
    first = _validators.smoothly_deprecate_legacy_arguments
    install_hf_local_dir_symlinks_compat()

    assert _validators.smoothly_deprecate_legacy_arguments is first
    assert getattr(first, "_zimage_strips_local_dir_use_symlinks", False)


def test_load_training_components_installs_compat_without_symlink_kwarg():
    calls = []
    load_training_components(
        {"model": {"main_transformer": {"path": "org/main", "revision": None}}},
        loaders=make_loaders(calls),
    )

    current = _validators.smoothly_deprecate_legacy_arguments
    assert getattr(current, "_zimage_strips_local_dir_use_symlinks", False)
    assert all("local_dir_use_symlinks" not in kwargs for _, _, kwargs, _ in calls)
