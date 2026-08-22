"""Shared path helpers for UI directory fields."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path


def normalize_dir(
    directory: str | None,
    *,
    file_suffixes: Collection[str] = (),
) -> str:
    """Strip quotes, convert backslashes to `/`, optionally treat files as dirs.

    If ``file_suffixes`` is set (or the path exists as a file), the parent
    directory is returned. Empty / ``.`` inputs yield ``""``.
    """
    if directory is None:
        return ""
    text = str(directory).strip().strip('"').strip("'").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    path = Path(text)
    suffixes = {s.lower() for s in file_suffixes}
    if path.suffix.lower() in suffixes or path.is_file():
        path = path.parent
    if not str(path).strip() or str(path) == ".":
        return ""
    return path.as_posix()
