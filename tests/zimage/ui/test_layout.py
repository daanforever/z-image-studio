from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import gradio as gr

from zimage.config import DEFAULT_IMAGE_FORMAT, DEFAULT_LORA_DIR, DEFAULT_OUTPUT_DIR
from zimage.engine.lora import normalize_lora_dir
from zimage.ui.handlers import generate
from zimage.ui.layout import build_ui
from zimage.ui.training_panel import build_training_panel as real_build_training_panel


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


def test_gallery_loads_after_restore_on_demo_load(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    restore_fns = _fns_named(demo, "restore_ui_prefs")
    assert len(restore_fns) == 1
    restore_fn = restore_fns[0]
    assert list(restore_fn.inputs) == []
    restore_output_ids = [getattr(block, "elem_id", None) for block in restore_fn.outputs]
    assert "studio-prompt" in restore_output_ids
    assert "studio-output-dir" in restore_output_ids
    assert "studio-lora-dir" in restore_output_ids
    assert "studio-lora-adapters" in restore_output_ids
    assert "studio-lora-weights" in restore_output_ids

    load_fns = _fns_named(demo, "load_gallery_with_index")
    assert len(load_fns) == 1
    load_fn = load_fns[0]
    input_ids = [getattr(block, "elem_id", None) for block in load_fn.inputs]
    output_ids = [getattr(block, "elem_id", None) for block in load_fn.outputs]
    assert input_ids == ["studio-output-dir"]
    assert output_ids == ["output-gallery", None]

    restore_key = next(k for k, fn in demo.fns.items() if fn is restore_fn)
    assert load_fn.trigger_after == restore_key
    assert load_fn.targets == [(None, "then")]


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


def test_no_browser_state_for_ui_prefs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    states = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.BrowserState)
    ]
    assert states == []


def test_restore_ui_prefs_on_load(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    restore_fns = _fns_named(demo, "restore_ui_prefs")
    assert len(restore_fns) == 1
    restore_fn = restore_fns[0]
    assert list(restore_fn.inputs) == []
    output_ids = [getattr(block, "elem_id", None) for block in restore_fn.outputs]
    assert "studio-prompt" in output_ids
    assert "studio-output-dir" in output_ids
    assert "studio-lora-dir" in output_ids
    assert "studio-lora-adapters" in output_ids
    assert "studio-lora-weights" in output_ids
    assert "studio-image-format" in output_ids
    assert "studio-quantize" in output_ids
    assert not any(isinstance(block, gr.BrowserState) for block in restore_fn.outputs)


def test_save_ui_prefs_events_wired(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    save_fns = _fns_named(demo, "save_ui_prefs")
    assert len(save_fns) >= 19
    for fn in save_fns:
        input_ids = [getattr(block, "elem_id", None) for block in fn.inputs]
        assert "studio-prompt" in input_ids
        assert "studio-lora-dir" in input_ids
        assert "studio-output-dir" in input_ids
        assert list(fn.outputs) == []
        assert not any(isinstance(block, gr.BrowserState) for block in fn.outputs)

    generate_saves = [
        fn for fn in save_fns if fn.targets and fn.targets[0][1] == "click"
    ]
    assert len(generate_saves) == 1

    then_saves = [fn for fn in save_fns if fn.targets == [(None, "then")]]
    assert len(then_saves) == 2
    refresh_keys = {
        k
        for k, fn in demo.fns.items()
        if getattr(getattr(fn, "fn", None), "__name__", None) == "refresh_loras"
        and fn.targets
        and fn.targets[0][1] in {"submit", "blur"}
    }
    assert {fn.trigger_after for fn in then_saves} == refresh_keys


def test_generate_and_training_tabs(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    ids = _elem_ids(demo)
    assert "studio-tabs" in ids
    assert "studio-tab-generate" in ids
    assert "studio-tab-training" in ids
    assert "generate-btn" in ids
    assert "studio-training-panel" in ids
    assert "studio-training-validate" not in ids
    assert "studio-training-toolbar" in ids
    assert "studio-training-run" in ids
    assert "studio-training-save" in ids
    assert "studio-training-start" in ids
    assert "studio-training-stop" in ids
    labels = []
    for block in demo.blocks.values():
        label = getattr(block, "label", None)
        if label in {"Generate", "Training"}:
            labels.append(label)
    assert labels == ["Generate", "Training"]
    assert "studio-training-log-accordion" in ids
    assert "studio-training-job-log" in ids
    assert "studio-training-log-delta" in ids
    log_accordion = _block_by_elem_id(demo, "studio-training-log-accordion")
    assert log_accordion.label == "Log"


def test_navbar_stop_still_cancels_generate(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    assert len(_fns_named(demo, "generate")) == 1
    stop_fns = _fns_named(demo, "request_stop")
    assert len(stop_fns) == 1
    assert stop_fns[0].targets
    assert stop_fns[0].targets[0][1] == "click"
    assert "studio-stop-btn" in _elem_ids(demo)
    assert "studio-training-stop" in _elem_ids(demo)


def test_training_start_cancels_generate(monkeypatch):
    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    demo = build_ui()
    cancel_fns = _fns_named(demo, "cancel_generate_for_training")
    assert len(cancel_fns) == 1
    assert cancel_fns[0].targets
    assert cancel_fns[0].targets[0][1] == "click"
    start_btn = _block_by_elem_id(demo, "studio-training-start")
    assert cancel_fns[0].targets[0][0] == getattr(start_btn, "_id", start_btn)


def test_build_ui_clamps_lora_adapters_from_yaml(monkeypatch, tmp_path: Path):
    from zimage.prefs import save_ui_prefs as dump_prefs

    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    (tmp_path / "alpha.safetensors").write_bytes(b"")
    dump_prefs(
        {
            "lora_dir": str(tmp_path),
            "lora_adapters": ["alpha.safetensors", "gone.safetensors"],
            "lora_weights": [
                ["alpha.safetensors", 0.8],
                ["gone.safetensors", 0.5],
            ],
        }
    )
    demo = build_ui()
    adapters = _block_by_elem_id(demo, "studio-lora-adapters")
    assert adapters.allow_custom_value is True
    assert "alpha.safetensors" in _choice_labels(adapters.choices)
    assert adapters.value == ["alpha.safetensors"]


def test_build_ui_applies_yaml_pref_values(monkeypatch):
    from zimage.prefs import save_ui_prefs as dump_prefs

    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    dump_prefs(
        {
            "prompt": "from yaml",
            "output_dir": "./yaml-out",
            "precision": "float16",
        }
    )
    demo = build_ui()
    prompt = _block_by_elem_id(demo, "studio-prompt")
    output_dir = _block_by_elem_id(demo, "studio-output-dir")
    precision = _block_by_label(demo, "Precision")
    assert prompt.value == "from yaml"
    assert output_dir.value == "./yaml-out"
    assert precision.value == "float16"


def test_build_ui_passes_training_callbacks_bundle_to_panel(monkeypatch):
    from zimage.ui import handlers as handlers_mod
    from zimage.ui.training_panel import TrainingCallbacks, noop_training_callbacks

    produced: dict[str, TrainingCallbacks] = {}
    captured: dict[str, TrainingCallbacks] = {}

    def tracking_callbacks():
        bundle = handlers_mod.training_callbacks()
        produced["bundle"] = bundle
        return bundle

    def spy_build(*, callbacks=None):
        captured["callbacks"] = callbacks
        return real_build_training_panel(callbacks=callbacks)

    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    monkeypatch.setattr("zimage.ui.layout.training_callbacks", tracking_callbacks)
    monkeypatch.setattr("zimage.ui.layout.build_training_panel", spy_build)

    demo = build_ui()
    assert demo is not None
    bundle = captured["callbacks"]
    assert bundle is produced["bundle"]
    assert isinstance(bundle, TrainingCallbacks)
    assert bundle.start_job is handlers_mod.start_training_job
    assert bundle.stop_job is handlers_mod.stop_training_job
    assert bundle.save_yaml is handlers_mod.save_training_yaml
    assert bundle.list_jobs is handlers_mod.list_training_jobs
    assert bundle.create_or_open is handlers_mod.create_or_open_training_job
    assert bundle.queue_update is handlers_mod.queue_training_update
    assert bundle.load_job is handlers_mod.load_training_job
    assert bundle.validate_yaml is handlers_mod.validate_training_yaml
    assert bundle.poll_state is handlers_mod.poll_training_state
    assert bundle.poll_log is handlers_mod.poll_training_log

    noop = noop_training_callbacks()
    assert bundle.start_job is not noop.start_job
    assert bundle.save_yaml is not noop.save_yaml
    assert bundle.create_or_open is not noop.create_or_open
    assert bundle.poll_log is not noop.poll_log
