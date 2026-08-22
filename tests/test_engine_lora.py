from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from zimage.engine.lora import (
    LoraSpec,
    list_lora_files,
    normalize_lora_dir,
    parse_lora_specs,
    reset_lora_adapters,
    rewrite_lora_inner_dit_keys,
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


def _write_lora(path: Path, keys: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or ["diffusion_model.layers.0.attention.to_q.lora_A.weight"]
    save_file({key: torch.zeros((1, 1), dtype=torch.float16) for key in keys}, str(path))


class _LoraPipe:
    def __init__(self):
        self.loads = []
        self.adapters = None
        self.fuses = []
        self.unloads = 0
        self.disables = 0

    def load_lora_weights(self, state_dict, weight_name=None, adapter_name=None):
        self.loads.append((state_dict, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights or []))

    def fuse_lora(self, adapter_names=None, lora_scale=None):
        self.fuses.append((list(adapter_names or []), lora_scale))

    def unload_lora_weights(self):
        self.unloads += 1
        self.adapters = None


class _MergeOnlyModule:
    def __init__(self):
        self.merged = 0

    def merge_and_unload(self):
        self.merged += 1
        return self


class _MergeOnlyPipe:
    def __init__(self):
        self.transformer = _MergeOnlyModule()
        self.loads = []
        self.adapters = None
        self.unloads = 0

    def load_lora_weights(self, state_dict, weight_name=None, adapter_name=None):
        self.loads.append((state_dict, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights or []))

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


def test_rewrite_lora_inner_dit_keys_strips_wrapper_segment():
    wrapped = "diffusion_model._inner_dit.layers.0.attention.to_q.lora_A.weight"
    portable = "diffusion_model.layers.0.attention.to_q.lora_A.weight"
    tensor = torch.zeros((1, 1), dtype=torch.float16)
    rewritten = rewrite_lora_inner_dit_keys({wrapped: tensor})
    assert list(rewritten) == [portable]
    assert rewritten[portable] is tensor


def test_rewrite_lora_inner_dit_keys_leaves_portable_unchanged():
    portable = "diffusion_model.layers.0.attention.to_q.lora_A.weight"
    tensor = torch.zeros((1, 1), dtype=torch.float16)
    rewritten = rewrite_lora_inner_dit_keys({portable: tensor})
    assert list(rewritten) == [portable]
    assert rewritten[portable] is tensor


def test_rewrite_lora_inner_dit_keys_strips_dotted_and_leading_prefix():
    tensor = torch.zeros((1, 1), dtype=torch.float16)
    portable = "diffusion_model.layers.0.attention.to_q.lora_A.weight"
    cases = {
        "inner.dit.layers.0.attention.to_q.lora_A.weight": portable,
        "diffusion_model.inner.dit.layers.0.attention.to_q.lora_A.weight": portable,
        "transformer.inner.dit.layers.0.attention.to_q.lora_A.weight": (
            "transformer.layers.0.attention.to_q.lora_A.weight"
        ),
        "_inner_dit.layers.0.attention.to_q.lora_A.weight": portable,
    }
    for source, expected in cases.items():
        rewritten = rewrite_lora_inner_dit_keys({source: tensor})
        assert list(rewritten) == [expected], source
        assert rewritten[expected] is tensor


def test_rewrite_lora_inner_dit_keys_strips_kohya_underscore_wrapper():
    tensor = torch.zeros((1, 1), dtype=torch.float16)
    portable = "lora_unet_layers_0_attention_to_q.lora_down.weight"
    cases = [
        "lora_unet__inner_dit_layers_0_attention_to_q.lora_down.weight",
        "lora_unet_inner_dit_layers_0_attention_to_q.lora_down.weight",
    ]
    for source in cases:
        rewritten = rewrite_lora_inner_dit_keys({source: tensor})
        assert list(rewritten) == [portable], source
        assert rewritten[portable] is tensor


def test_rewrite_kohya_inner_dit_converts_for_zimage():
    pytest.importorskip("diffusers")
    from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin

    tensor_a = torch.zeros((1, 4), dtype=torch.float16)
    tensor_b = torch.zeros((4, 1), dtype=torch.float16)
    raw = {
        "lora_unet__inner_dit_layers_0_attention_to_q.lora_down.weight": tensor_a,
        "lora_unet__inner_dit_layers_0_attention_to_q.lora_up.weight": tensor_b,
        "lora_unet__inner_dit_layers_0_attention_to_q.alpha": torch.tensor(1.0),
    }
    rewritten = rewrite_lora_inner_dit_keys(raw)
    converted = ZImageLoraLoaderMixin.lora_state_dict(rewritten)
    assert converted
    assert all(key.startswith("transformer.layers.") for key in converted)
    assert all("inner.dit" not in key and "_inner_dit" not in key for key in converted)
    assert any("to_q.lora_A.weight" in key for key in converted)
    assert any("to_q.lora_B.weight" in key for key in converted)


def test_rewrite_lora_inner_dit_keys_collision_raises():
    wrapped = "diffusion_model._inner_dit.layers.0.attention.to_q.lora_A.weight"
    portable = "diffusion_model.layers.0.attention.to_q.lora_A.weight"
    a = torch.zeros((1, 1), dtype=torch.float16)
    b = torch.ones((1, 1), dtype=torch.float16)
    with pytest.raises(ValueError, match="collision"):
        rewrite_lora_inner_dit_keys({wrapped: a, portable: b})


def test_sync_lora_adapters_loads_and_sets(tmp_path: Path, reset_lora):
    _write_lora(tmp_path / "style.safetensors")
    _write_lora(tmp_path / "char.safetensors")
    specs = parse_lora_specs(
        str(tmp_path),
        ["style.safetensors", "char.safetensors"],
        [["style.safetensors", 0.8], ["char.safetensors", 1.0]],
    )
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    assert [name for _state, name in pipe.loads] == ["style", "char"]
    assert all(isinstance(state, dict) for state, _name in pipe.loads)
    # One fuse+unload per adapter; last set_adapters cleared by unload.
    assert pipe.fuses == [(["style"], 0.8), (["char"], 1.0)]
    assert pipe.unloads == 2
    assert pipe.adapters is None

    sync_lora_adapters(pipe, specs)
    assert len(pipe.loads) == 2
    assert pipe.unloads == 2
    assert len(pipe.fuses) == 2


def test_sync_lora_adapters_loads_pt(tmp_path: Path, reset_lora):
    path = tmp_path / "style.pt"
    state = {"diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.zeros((1, 1))}
    torch.save(state, path)
    specs = parse_lora_specs(str(tmp_path), ["style.pt"], [["style.pt", 0.9]])
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    assert len(pipe.loads) == 1
    loaded, name = pipe.loads[0]
    assert name == "style"
    assert "diffusion_model.layers.0.attention.to_q.lora_A.weight" in loaded
    assert pipe.fuses == [(["style"], 0.9)]
    assert pipe.unloads == 1
    assert pipe.adapters is None


def test_sync_lora_adapters_retries_after_load_failure(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_lora(path)
    spec = LoraSpec(
        path=path,
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )
    pipe = _LoraPipe()
    original_load = pipe.load_lora_weights
    calls = {"n": 0}

    def flaky_load(state_dict, weight_name=None, adapter_name=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original_load(state_dict, weight_name=weight_name, adapter_name=adapter_name)

    pipe.load_lora_weights = flaky_load
    with pytest.raises(RuntimeError, match="boom"):
        sync_lora_adapters(pipe, (spec,))
    assert pipe.unloads == 0
    assert len(pipe.loads) == 0
    assert pipe.fuses == []

    sync_lora_adapters(pipe, (spec,))
    assert pipe.unloads == 1
    assert len(pipe.loads) == 1
    assert pipe.fuses == [(["style"], 1.0)]
    assert pipe.adapters is None


def test_sync_lora_adapters_empty_is_noop(tmp_path: Path, reset_lora):
    _write_lora(tmp_path / "style.safetensors")
    specs = parse_lora_specs(str(tmp_path), ["style.safetensors"], None)
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    sync_lora_adapters(pipe, ())
    # Empty specs on a fresh/fused pipe does not call unload (no PEFT left).
    assert pipe.unloads == 1
    assert len(pipe.loads) == 1
    assert pipe.fuses == [(["style"], 1.0)]


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


def test_sync_lora_adapters_merge_and_unload_fallback(tmp_path: Path, reset_lora):
    path = tmp_path / "style.safetensors"
    _write_lora(path)
    spec = LoraSpec(
        path=path,
        filename="style.safetensors",
        adapter_name="style",
        scale=0.7,
    )
    pipe = _MergeOnlyPipe()
    sync_lora_adapters(pipe, (spec,))
    assert len(pipe.loads) == 1
    assert pipe.adapters is None
    assert pipe.transformer.merged == 1
    assert pipe.unloads == 1


def test_sync_lora_adapters_disable_fallback_unused_for_empty(reset_lora):
    pipe = _DisableOnlyPipe()
    sync_lora_adapters(pipe, ())
    assert pipe.disables == 0


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


def test_sync_real_fixture_loads_state_dict(tiny_lora_dir: Path, reset_lora):
    specs = parse_lora_specs(
        str(tiny_lora_dir),
        ["tiny_zimage_lora.safetensors"],
        [["tiny_zimage_lora.safetensors", 0.6]],
    )
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, specs)
    assert len(pipe.loads) == 1
    state, name = pipe.loads[0]
    assert name == "tiny_zimage_lora"
    assert isinstance(state, dict)
    assert state
    assert all("_inner_dit" not in key and "inner.dit." not in key for key in state)
    assert pipe.fuses == [(["tiny_zimage_lora"], 0.6)]
    assert pipe.unloads == 1
    assert pipe.adapters is None


def test_sync_lora_adapters_rewrites_wrapped_keys(tmp_path: Path, reset_lora):
    path = tmp_path / "old.safetensors"
    _write_lora(
        path,
        [
            "diffusion_model._inner_dit.layers.0.attention.to_q.lora_A.weight",
            "diffusion_model._inner_dit.layers.0.attention.to_q.lora_B.weight",
        ],
    )
    spec = LoraSpec(
        path=path,
        filename="old.safetensors",
        adapter_name="old",
        scale=1.0,
    )
    pipe = _LoraPipe()
    sync_lora_adapters(pipe, (spec,))
    state, name = pipe.loads[0]
    assert name == "old"
    assert "diffusion_model.layers.0.attention.to_q.lora_A.weight" in state
    assert "diffusion_model.layers.0.attention.to_q.lora_B.weight" in state
    assert all("_inner_dit" not in key and "inner.dit." not in key for key in state)
    assert pipe.fuses == [(["old"], 1.0)]
    assert pipe.unloads == 1
