"""Silence huggingface_hub ``local_dir_use_symlinks`` deprecation on load."""

from __future__ import annotations

from typing import Any

from huggingface_hub.utils import _validators


def install_hf_local_dir_symlinks_compat() -> None:
    """Drop ``local_dir_use_symlinks`` before hub deprecation warnings.

    Diffusers 0.40 ``load_config`` always forwards this ignored kwarg into
    ``hf_hub_download``. The hub decorator looks up
    ``smoothly_deprecate_legacy_arguments`` on the ``_validators`` module at
    call time, so replacing that global covers both call sites.
    """

    current = _validators.smoothly_deprecate_legacy_arguments
    if getattr(current, "_zimage_strips_local_dir_use_symlinks", False):
        return

    def smoothly_deprecate_legacy_arguments(
        fn_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        cleaned = dict(kwargs)
        cleaned.pop("local_dir_use_symlinks", None)
        return current(fn_name=fn_name, kwargs=cleaned)

    smoothly_deprecate_legacy_arguments._zimage_strips_local_dir_use_symlinks = True
    _validators.smoothly_deprecate_legacy_arguments = (
        smoothly_deprecate_legacy_arguments
    )
