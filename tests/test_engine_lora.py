from __future__ import annotations

from pathlib import Path

import pytest
from safetensors import safe_open

from zimage.engine.lora import (
    LoraSpec,
    list_lora_files,
    normalize_lora_dir,
    parse_lora_specs,
    reset_lora_adapters,
    sync_lora_adapters,
)


@pytest.fixture
def reset_lora():
    reset_lora_adapters()
    yield
    reset_lora_adapters()


def _touch_loras(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


class _LoraPipe:
    def __init__(self):
        self.loads = []
        self.adapters = None
        self.unloads = 0
        self.disables = 0

    def load_lora_weights(self, directory, weight_name=None, adapter_name=None):
        self.loads.append((directory, weight_name, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights))

    def unload_lora_weights(self):
        self.unloads += 1
        self.adapters = None


class _DisableOnlyPipe:
    def __init__(self):
        self.disables = 0

    def disable_lora(self):
        self.disables += 1


def test_list_lora_files_empty_and_missing(tmp_path: Path):
    assert list_lora_files("") == []
    assert list_lora_files(None) == []
    assert list_lora_files("   ") == []
    assert list_lora_files(str(tmp_path / "missing")) == []
    assert list_lora_files(str(tmp_path)) == []


def test_normalize_lora_dir_slashes_and_quotes():
    assert normalize_lora_dir(None) == ""
    assert normalize_lora_dir("   ") == ""
    assert normalize_lora_dir(r"D:\loras\style") == "D:/loras/style"
    assert normalize_lora_dir(r'"D:\loras\style"') == "D:/loras/style"
    assert normalize_lora_dir(r"'D:\loras\style'") == "D:/loras/style"
    assert normalize_lora_dir(r"D:\\loras\\style") == "D:/loras/style"
    assert normalize_lora_dir("D:/loras/style") == "D:/loras/style"
    assert normalize_lora_dir("D:\\loras\\style\\") == "D:/loras/style"
    assert normalize_lora_dir("D:/loras/style/") == "D:/loras/style"


def test_normalize_lora_dir_file_uses_parent():
    assert (
        normalize_lora_dir(r"d:\Projects\DeepSeek\c\output\b\a.safetensors")
        == "d:/Projects/DeepSeek/c/output/b"
    )
    assert normalize_lora_dir("D:/loras/style.pt") == "D:/loras"
    assert normalize_lora_dir(r'"E:\adapters\char.safetensors"') == "E:/adapters"


def test_list_and_parse_accept_windows_file_path(tmp_path: Path):
    _touch_loras(tmp_path, "a.safetensors", "b.pt")
    file_path = tmp_path / "a.safetensors"
    # Force backslash form regardless of OS Path str()
    windows_like = str(file_path).replace("/", "\\")
    assert list_lora_files(windows_like) == ["a.safetensors", "b.pt"]
    specs = parse_lora_specs(windows_like, ["a.safetensors"], [["a.safetensors", 0.7]])
    assert len(specs) == 1
    assert specs[0].path == tmp_path / "a.safetensors"
    assert specs[0].scale == 0.7


def test_list_and_parse_accept_windows_dir_path(tmp_path: Path):
    _touch_loras(tmp_path, "a.safetensors", "b.pt")
    windows_like = str(tmp_path).replace("/", "\\")
    assert list_lora_files(windows_like) == ["a.safetensors", "b.pt"]
    specs = parse_lora_specs(windows_like, ["b.pt"], [["b.pt", 0.5]])
    assert len(specs) == 1
    assert specs[0].path == tmp_path / "b.pt"
    assert specs[0].scale == 0.5
    assert normalize_lora_dir(windows_like) == Path(tmp_path).as_posix()


def test_list_and_parse_accept_windows_dir_path_trailing_slash(tmp_path: Path):
    _touch_loras(tmp_path, "a.safetensors", "b.pt")
    windows_like = str(tmp_path).replace("/", "\\") + "\\"
    assert normalize_lora_dir(windows_like) == Path(tmp_path).as_posix()
    assert list_lora_files(windows_like) == ["a.safetensors", "b.pt"]
    specs = parse_lora_specs(windows_like, ["a.safetensors"], None)
    assert len(specs) == 1
    assert specs[0].path == tmp_path / "a.safetensors"

def test_list_lora_files_filters_and_sorts(tmp_path: Path):
    _touch_loras(tmp_path, "zeta.safetensors", "alpha.pt", "notes.txt")
    nested = tmp_path / "nested"
    _touch_loras(nested, "hidden.safetensors")
    assert list_lora_files(str(tmp_path)) == ["alpha.pt", "zeta.safetensors"]


def test_parse_lora_specs_defaults_and_drops_unknown(tmp_path: Path):
    _touch_loras(tmp_path, "style.safetensors", "char.safetensors")
    specs = parse_lora_specs(
        str(tmp_path),
        ["style.safetensors", "missing.safetensors", "notes.txt"],
        None,
    )
    assert len(specs) == 1
    assert specs[0].filename == "style.safetensors"
    assert specs[0].path == tmp_path / "style.safetensors"
    assert specs[0].scale == 1.0
    assert specs[0].adapter_name == "style"


def test_parse_lora_specs_preserves_and_clamps_weights(tmp_path: Path):
    _touch_loras(tmp_path, "style.safetensors", "char.safetensors")
    specs = parse_lora_specs(
        str(tmp_path),
        ["style.safetensors", "char.safetensors"],
        [["style.safetensors", 0.8], ["char.safetensors", 3.5], ["gone.safetensors", 0.2]],
    )
    by_name = {spec.filename: spec.scale for spec in specs}
    assert by_name == {"style.safetensors": 0.8, "char.safetensors": 2.0}


def test_parse_lora_specs_clamps_negative_and_sanitizes_names(tmp_path: Path):
    _touch_loras(tmp_path, "my-style.safetensors", "my style.safetensors")
    specs = parse_lora_specs(
        str(tmp_path),
        ["my-style.safetensors", "my style.safetensors"],
        [["my-style.safetensors", -1]],
    )
    assert specs[0].scale == 0.0
    names = [spec.adapter_name for spec in specs]
    assert names[0] == "my_style"
    assert names[1] == "my_style_2"
    assert len(set(names)) == 2


def test_parse_lora_specs_empty_directory():
    assert parse_lora_specs("", ["style.safetensors"], [["style.safetensors", 1]]) == ()


def test_sync_lora_adapters_loads_and_sets(tmp_path: Path, reset_lora):
    _touch_loras(tmp_path, "style.safetensors", "char.safetensors")
    specs = parse_lora_specs(
        str(tmp_path),
        ["style.safetensors", "char.safetensors"],
        [["style.safetensors", 0.8], ["char.safetensors", 1.0]],
    )
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    assert pipe.unloads == 1
    assert pipe.loads == [
        (str(tmp_path), "style.safetensors", "style"),
        (str(tmp_path), "char.safetensors", "char"),
    ]
    assert pipe.adapters == (["style", "char"], [0.8, 1.0])

    sync_lora_adapters(pipe, specs)
    assert len(pipe.loads) == 2
    assert pipe.unloads == 1


def test_sync_lora_adapters_empty_unloads(tmp_path: Path, reset_lora):
    _touch_loras(tmp_path, "style.safetensors")
    specs = parse_lora_specs(str(tmp_path), ["style.safetensors"], None)
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    sync_lora_adapters(pipe, ())
    assert pipe.unloads == 2
    assert len(pipe.loads) == 1
    assert pipe.adapters is None


def test_sync_lora_adapters_without_api_empty_is_noop(reset_lora):
    sync_lora_adapters(object(), ())


def test_sync_lora_adapters_without_api_raises(tmp_path: Path, reset_lora):
    spec = LoraSpec(
        path=tmp_path / "style.safetensors",
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )
    with pytest.raises(RuntimeError, match="does not support LoRA"):
        sync_lora_adapters(object(), (spec,))


def test_sync_lora_adapters_disable_fallback(reset_lora):
    pipe = _DisableOnlyPipe()
    sync_lora_adapters(pipe, ())
    assert pipe.disables == 1


def test_fixture_tiny_lora_is_valid_safetensors(tiny_lora_dir: Path):
    path = tiny_lora_dir / "tiny_zimage_lora.safetensors"
    assert path.is_file()
    assert path.stat().st_size > 1024
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
    assert len(keys) >= 1
    assert all(key.startswith("diffusion_model.") for key in keys)
    assert any("lora_A.weight" in key for key in keys)
    assert any("lora_B.weight" in key for key in keys)
    with safe_open(path, framework="pt") as handle:
        sample = handle.get_tensor(keys[0])
    assert 1 in sample.shape


def test_fixture_tiny_lora_converts_for_zimage(tiny_lora_dir: Path):
    pytest.importorskip("diffusers")
    from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin

    converted = ZImageLoraLoaderMixin.lora_state_dict(
        str(tiny_lora_dir),
        weight_name="tiny_zimage_lora.safetensors",
    )
    assert converted
    assert all("lora" in key for key in converted)
    assert all(key.startswith("transformer.") for key in converted)


def test_list_and_parse_real_fixture(tiny_lora_dir: Path):
    files = list_lora_files(str(tiny_lora_dir))
    assert "tiny_zimage_lora.safetensors" in files
    specs = parse_lora_specs(
        str(tiny_lora_dir),
        ["tiny_zimage_lora.safetensors"],
        [["tiny_zimage_lora.safetensors", 0.75]],
    )
    assert len(specs) == 1
    assert specs[0].path == tiny_lora_dir / "tiny_zimage_lora.safetensors"
    assert specs[0].path.is_file()
    assert specs[0].scale == 0.75
    assert specs[0].adapter_name == "tiny_zimage_lora"


def test_sync_real_fixture_passes_paths(tiny_lora_dir: Path, reset_lora):
    specs = parse_lora_specs(
        str(tiny_lora_dir),
        ["tiny_zimage_lora.safetensors"],
        [["tiny_zimage_lora.safetensors", 0.6]],
    )
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    assert pipe.loads == [
        (str(tiny_lora_dir), "tiny_zimage_lora.safetensors", "tiny_zimage_lora"),
    ]
    assert pipe.adapters == (["tiny_zimage_lora"], [0.6])
    # Caller receives an existing on-disk weight file, not an empty stub.
    assert Path(pipe.loads[0][0], pipe.loads[0][1]).stat().st_size > 1024
