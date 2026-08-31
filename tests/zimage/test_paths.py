from __future__ import annotations

from pathlib import Path

from zimage.paths import normalize_dir


def test_normalize_dir_empty():
    assert normalize_dir(None) == ""
    assert normalize_dir("   ") == ""
    assert normalize_dir(".") == ""


def test_normalize_dir_slashes_and_quotes():
    assert normalize_dir(r"D:\Projects\DeepSeek\z-image-studio\outputs") == (
        "D:/Projects/DeepSeek/z-image-studio/outputs"
    )
    assert normalize_dir('"D:\\Projects\\DeepSeek\\z-image-studio\\outputs\\"') == (
        "D:/Projects/DeepSeek/z-image-studio/outputs"
    )
    assert normalize_dir(r"'D:\outputs'") == "D:/outputs"
    assert normalize_dir(r"D:\\loras\\style") == "D:/loras/style"
    assert normalize_dir("D:/loras/style") == "D:/loras/style"
    assert normalize_dir("D:\\loras\\style\\") == "D:/loras/style"
    assert normalize_dir("D:/loras/style/") == "D:/loras/style"


def test_normalize_dir_relative_default():
    assert normalize_dir("./outputs") == "outputs"
    assert normalize_dir("outputs") == "outputs"


def test_normalize_dir_existing_file_uses_parent(tmp_path: Path):
    file_path = tmp_path / "shot.png"
    file_path.write_bytes(b"x")
    windows_like = str(file_path).replace("/", "\\")
    assert normalize_dir(windows_like) == Path(tmp_path).as_posix()


def test_normalize_dir_file_suffixes():
    assert normalize_dir(
        r"d:\loras\style.safetensors",
        file_suffixes={".safetensors", ".pt"},
    ) == "d:/loras"
    assert normalize_dir(
        "D:/loras/style.pt",
        file_suffixes={".safetensors", ".pt"},
    ) == "D:/loras"
