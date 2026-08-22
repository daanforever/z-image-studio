from __future__ import annotations

import inspect
import warnings

import gradio as gr

from zimage.config import DEFAULT_IMAGE_FORMAT, DEFAULT_LORA_DIR, DEFAULT_OUTPUT_DIR
from zimage.engine.lora import normalize_lora_dir
from zimage.ui.handlers import generate
from zimage.ui.layout import build_ui


def _elem_ids(demo) -> set[str | None]:
    return {getattr(block, "elem_id", None) for block in demo.blocks.values()}


def test_build_ui_constructs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The parameters have been moved from the Blocks constructor",
            category=UserWarning,
        )
        demo = build_ui()
    assert demo is not None


def test_build_ui_has_navbar(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    ids = _elem_ids(demo)
    assert "studio-navbar" in ids
    assert "studio-brand" in ids
    assert "studio-navbar-actions" in ids
    assert "studio-clear-btn" in ids
    assert "studio-stop-btn" in ids
    brand = next(
        block
        for block in demo.blocks.values()
        if getattr(block, "elem_id", None) == "studio-brand"
    )
    assert "Studio" in str(brand.value)


def test_generate_event_show_progress_on_status(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    generate_fns = [
        fn
        for fn in demo.fns.values()
        if getattr(getattr(fn, "fn", None), "__name__", None) == "generate"
    ]
    assert len(generate_fns) == 1
    generate_fn = generate_fns[0]
    status = _block_by_elem_id(demo, "status-md")
    assert generate_fn.show_progress == "minimal"
    assert generate_fn.show_progress_on == [status]


def test_generate_progress_default_disables_track_tqdm():
    default = inspect.signature(generate).parameters["progress"].default
    assert getattr(default, "track_tqdm", None) is False


def _block_by_elem_id(demo, elem_id: str):
    return next(
        block
        for block in demo.blocks.values()
        if getattr(block, "elem_id", None) == elem_id
    )


def test_examples_follow_status_in_output_column(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    ids = _elem_ids(demo)
    assert "studio-examples" in ids
    assert "status-md" in ids
    status = _block_by_elem_id(demo, "status-md")
    examples = _block_by_elem_id(demo, "studio-examples")
    assert status.parent is examples.parent
    siblings = list(status.parent.children)
    assert siblings.index(status) < siblings.index(examples)


def _choice_labels(choices) -> list[str]:
    labels = []
    for choice in choices:
        if isinstance(choice, (list, tuple)):
            labels.append(str(choice[0]))
        else:
            labels.append(str(choice))
    return labels


def test_model_device_quantize_checkboxes(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    quant = _block_by_elem_id(demo, "studio-quantize")
    assert quant.label == "quantize"
    assert _choice_labels(quant.choices) == ["transformer", "text encoder"]
    assert list(quant.value) == ["transformer", "text encoder"]


def _block_by_label(demo, label: str):
    return next(
        block
        for block in demo.blocks.values()
        if getattr(block, "label", None) == label
    )


def _fns_named(demo, name: str):
    return [
        fn
        for fn in demo.fns.values()
        if getattr(getattr(fn, "fn", None), "__name__", None) == name
    ]


def test_lora_accordion_after_model_device(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    ids = _elem_ids(demo)
    assert "studio-lora" in ids
    assert "studio-lora-dir" in ids
    assert "studio-lora-refresh" in ids
    assert "studio-lora-adapters" in ids
    assert "studio-lora-weights" in ids

    lora = _block_by_elem_id(demo, "studio-lora")
    model = _block_by_label(demo, "Model & device")
    assert lora.parent is model.parent
    siblings = list(lora.parent.children)
    assert siblings.index(model) < siblings.index(lora)

    directory = _block_by_elem_id(demo, "studio-lora-dir")
    assert directory.value == normalize_lora_dir(DEFAULT_LORA_DIR)

    adapters = _block_by_elem_id(demo, "studio-lora-adapters")
    assert adapters.multiselect is True
    assert list(adapters.value or []) == []

    weights = _block_by_elem_id(demo, "studio-lora-weights")
    headers = list(getattr(weights, "headers", []) or [])
    assert headers[:2] == ["LoRA", "Strength"]


def test_generate_event_includes_lora_inputs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    generate_fn = _fns_named(demo, "generate")[0]
    input_ids = [getattr(block, "elem_id", None) for block in generate_fn.inputs]
    assert "studio-lora-dir" in input_ids
    assert "studio-lora-adapters" in input_ids
    assert "studio-lora-weights" in input_ids
    assert "studio-output-dir" in input_ids
    assert "studio-image-format" in input_ids


def test_output_dir_field_default(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    field = _block_by_elem_id(demo, "studio-output-dir")
    assert field.value == DEFAULT_OUTPUT_DIR
    assert field.label == "Output dir"


def test_image_format_field_defaults_to_jpeg(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    field = _block_by_elem_id(demo, "studio-image-format")
    assert field.value == DEFAULT_IMAGE_FORMAT
    assert DEFAULT_IMAGE_FORMAT == "jpeg"
    choice_values = [
        c[0] if isinstance(c, (list, tuple)) else c for c in (field.choices or [])
    ]
    assert choice_values == ["png", "jpeg"]
    assert field.label == "Format"


def test_output_gallery_starts_in_preview(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    gallery = _block_by_elem_id(demo, "output-gallery")
    assert gallery.preview is True
    assert gallery.allow_preview is True
    assert gallery.height == 640


def test_gallery_loads_on_demo_load(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    load_fns = _fns_named(demo, "load_gallery_with_index")
    assert len(load_fns) == 1
    load_fn = load_fns[0]
    input_ids = [getattr(block, "elem_id", None) for block in load_fn.inputs]
    output_ids = [getattr(block, "elem_id", None) for block in load_fn.outputs]
    assert input_ids == ["studio-output-dir"]
    assert output_ids == ["output-gallery", None]


def test_output_gallery_has_delete_button(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    gallery = _block_by_elem_id(demo, "output-gallery")
    buttons = list(getattr(gallery, "buttons", None) or [])
    assert "share" not in buttons
    assert "download" in buttons
    assert "download_all" in buttons
    assert "fullscreen" in buttons
    custom = [b for b in buttons if not isinstance(b, str)]
    assert len(custom) == 1
    delete_btn = custom[0]
    assert getattr(delete_btn, "elem_id", None) == "studio-gallery-delete"
    assert delete_btn.value == "Delete"


def test_output_gallery_share_button_only_when_share(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui(share=True)
    gallery = _block_by_elem_id(demo, "output-gallery")
    buttons = list(getattr(gallery, "buttons", None) or [])
    assert "share" in buttons
    assert buttons[0] == "share"


def test_delete_preview_image_event_wired(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    delete_fns = _fns_named(demo, "delete_preview_image")
    assert len(delete_fns) == 1
    delete_fn = delete_fns[0]
    input_ids = [getattr(block, "elem_id", None) for block in delete_fn.inputs]
    output_ids = [getattr(block, "elem_id", None) for block in delete_fn.outputs]
    assert "output-gallery" in input_ids
    assert "studio-output-dir" in input_ids
    assert "output-gallery" in output_ids
    assert "status-md" in output_ids


def test_clear_preview_images_event_wired(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    clear_fns = _fns_named(demo, "clear_preview_images")
    assert len(clear_fns) == 1
    clear_fn = clear_fns[0]
    input_ids = [getattr(block, "elem_id", None) for block in clear_fn.inputs]
    output_ids = [getattr(block, "elem_id", None) for block in clear_fn.outputs]
    assert "studio-output-dir" in input_ids
    assert "output-gallery" in output_ids
    assert "status-md" in output_ids


def test_gallery_select_updates_index(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    select_fns = _fns_named(demo, "set_gallery_index")
    assert len(select_fns) == 1


def test_lora_events_wired(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    refresh_fns = _fns_named(demo, "refresh_loras")
    assert len(refresh_fns) == 3
    assert {fn.fn.__name__ for fn in refresh_fns} == {"refresh_loras"}
    for fn in refresh_fns:
        output_ids = [getattr(block, "elem_id", None) for block in fn.outputs]
        assert "studio-lora-dir" in output_ids
        assert "studio-lora-adapters" in output_ids
        assert "studio-lora-weights" in output_ids
    weight_fns = _fns_named(demo, "sync_lora_weights")
    assert len(weight_fns) == 1


def test_prompt_has_elem_id(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    prompt = _block_by_elem_id(demo, "studio-prompt")
    assert prompt.label == "Prompt"


def test_ui_prefs_browser_state(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    states = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.BrowserState)
    ]
    assert len(states) == 1
    prefs = states[0]
    assert prefs.storage_key == "zimage-studio-ui-prefs"
    assert prefs.secret == "zimage-studio"
    assert prefs.default_value == {
        "prompt": "",
        "lora_dir": normalize_lora_dir(DEFAULT_LORA_DIR),
    }


def test_restore_ui_prefs_on_load(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    restore_fns = _fns_named(demo, "restore_ui_prefs")
    assert len(restore_fns) == 1
    restore_fn = restore_fns[0]
    output_ids = [getattr(block, "elem_id", None) for block in restore_fn.outputs]
    assert output_ids == [
        "studio-prompt",
        "studio-lora-dir",
        "studio-lora-adapters",
        "studio-lora-weights",
    ]
    assert any(isinstance(block, gr.BrowserState) for block in restore_fn.inputs)


def test_save_ui_prefs_events_wired(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    save_fns = _fns_named(demo, "save_ui_prefs")
    assert len(save_fns) == 4
    for fn in save_fns:
        input_ids = [getattr(block, "elem_id", None) for block in fn.inputs]
        assert "studio-prompt" in input_ids
        assert "studio-lora-dir" in input_ids
        assert any(isinstance(block, gr.BrowserState) for block in fn.outputs)
