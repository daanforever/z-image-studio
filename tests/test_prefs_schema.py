from __future__ import annotations

from zimage.config import (
    DEFAULT_DTYPE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
)
from zimage.prefs import (
    UI_PREF_KEYS,
    load_ui_prefs,
    save_ui_prefs,
    ui_pref_defaults,
)
from zimage.prefs.schema import coerce_ui_prefs


def test_ui_pref_defaults_contain_all_keys():
    defaults = ui_pref_defaults()
    assert set(defaults) == set(UI_PREF_KEYS)
    assert defaults["steps"] == DEFAULT_STEPS
    assert defaults["resolution"] == DEFAULT_RESOLUTION
    assert defaults["image_format"] == DEFAULT_IMAGE_FORMAT
    assert defaults["precision"] == DEFAULT_DTYPE


def test_unknown_keys_dropped():
    coerced = coerce_ui_prefs({"prompt": "ok", "mystery": 123, "extra": True})
    assert "mystery" not in coerced
    assert "extra" not in coerced
    assert coerced["prompt"] == "ok"


def test_invalid_choice_fields_fall_back():
    defaults = ui_pref_defaults()
    coerced = coerce_ui_prefs(
        {
            "precision": "nope",
            "device": "tpu",
            "image_format": "gif",
            "resolution": "999x999",
        }
    )
    assert coerced["precision"] == defaults["precision"]
    assert coerced["device"] == defaults["device"]
    assert coerced["image_format"] == defaults["image_format"]
    assert coerced["resolution"] == defaults["resolution"]


def test_coerce_numeric_and_bool_from_strings():
    coerced = coerce_ui_prefs(
        {
            "steps": "11",
            "seed": "99",
            "batch": "3",
            "random_seed": "false",
            "cpu_offload": "1",
            "vae_tiling": "yes",
            "guidance": "1.5",
            "time_shift": "4.2",
        }
    )
    assert coerced["steps"] == 11
    assert coerced["seed"] == 99
    assert coerced["batch"] == 3
    assert coerced["random_seed"] is False
    assert coerced["cpu_offload"] is True
    assert coerced["vae_tiling"] is True
    assert coerced["guidance"] == 1.5
    assert coerced["time_shift"] == 4.2


def test_quantize_modules_string_or_list():
    assert coerce_ui_prefs({"quantize_modules": "transformer"})["quantize_modules"] == [
        "transformer"
    ]
    assert coerce_ui_prefs(
        {"quantize_modules": ["transformer", "text encoder", "nope"]}
    )["quantize_modules"] == ["transformer", "text encoder"]


def test_lora_weights_valid_and_broken_rows():
    coerced = coerce_ui_prefs(
        {
            "lora_weights": [
                ["a.safetensors", 0.5],
                ["bad"],
                [None, 1.0],
                ["", 1.0],
                ["b.safetensors", "nope"],
                ["c.safetensors", "0.8"],
            ]
        }
    )
    assert coerced["lora_weights"] == [
        ["a.safetensors", 0.5],
        ["c.safetensors", 0.8],
    ]


def test_multiline_prompt_roundtrip():
    prompt = "line one\nline two\n«Время»"
    save_ui_prefs({"prompt": prompt})
    loaded = load_ui_prefs()
    assert loaded["prompt"] == prompt


def test_precision_aliases_accepted():
    assert coerce_ui_prefs({"precision": "bf16"})["precision"] == "bfloat16"
    assert coerce_ui_prefs({"image_format": "jpg"})["image_format"] == "jpeg"
