from __future__ import annotations

import ast
import inspect
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
import yaml

from zimage.config import DEFAULT_IMAGE_FORMAT, IMAGE_FORMAT_CHOICES
from zimage.ui.theme import CUSTOM_JS
from zimage.ui.training_panel import (
    APPLY_TRAINING_LOG_JS,
    JOB_LOG_HTML,
    JobPanelData,
    TrainingCallbacks,
    TrainingPanel,
    as_training_callbacks,
    build_training_panel,
    commit_training_log,
    format_operational_state,
    handle_clear_log,
    handle_create_or_open,
    handle_load_job,
    handle_poll_log,
    handle_save,
    handle_start,
    handle_stop,
    handle_validate,
    noop_training_callbacks,
)


CANONICAL_YAML = (
    "job_name: demo\n"
    "sampling:\n"
    "  guidance_scale: 0.0\n"
    "  num_inference_steps: 9\n"
)
PNG_YAML = (
    "job_name: demo\n"
    "sampling:\n"
    "  guidance_scale: 0.0\n"
    "  num_inference_steps: 9\n"
    "  image_format: png\n"
)
JPEG_YAML = (
    "job_name: demo\n"
    "sampling:\n"
    "  image_format: jpeg\n"
)

PANEL_SOURCE = Path(__file__).resolve().parents[3] / "zimage" / "ui" / "training_panel.py"


@dataclass
class RecordingCallbacks:
    jobs: list[str] = field(default_factory=list)
    job_id: str = "demo-job"
    config_text: str = CANONICAL_YAML
    status: str = "stopped"
    step: int = 0
    epoch: int = 0
    last_error: str | None = None
    previews: list[str] = field(default_factory=list)
    log_chunk: str = ""
    log_reset: bool | None = None
    log_next_offset: int | None = None
    validate_ok: bool = True
    validate_error: str = "invalid"
    save_mode: str = "saved"
    queue_mode: str = "queued"
    calls: list[tuple] = field(default_factory=list)
    rewrite_called: bool = False

    def list_jobs(self) -> list[str]:
        self.calls.append(("list_jobs",))
        return list(self.jobs)

    def create_or_open(self, name: str) -> dict:
        self.calls.append(("create_or_open", name))
        return self._job_payload()

    def load_job(self, job_id: str) -> dict:
        self.calls.append(("load_job", job_id))
        return self._job_payload(job_id=job_id)

    def validate_yaml(self, job_id: str, text: str) -> dict:
        self.calls.append(("validate_yaml", job_id, text))
        if self.validate_ok:
            return {"ok": True}
        return {"ok": False, "error": self.validate_error}

    def save_yaml(self, job_id: str, text: str) -> dict:
        self.calls.append(("save_yaml", job_id, text))
        return {"mode": self.save_mode, "config_text": text}

    def queue_update(self, job_id: str, text: str) -> dict:
        self.calls.append(("queue_update", job_id, text))
        return {"mode": self.queue_mode}

    def rewrite_job(self, name: str) -> dict:
        self.rewrite_called = True
        self.calls.append(("rewrite_job", name))
        raise AssertionError("existing slug must not rewrite files")

    def start_job(self, job_id: str) -> dict:
        self.calls.append(("start_job", job_id))
        return self._state(status="running", job_id=job_id)

    def stop_job(self, job_id: str) -> dict:
        self.calls.append(("stop_job", job_id))
        return self._state(status="stopped", job_id=job_id)

    def poll_state(self, job_id: str) -> dict:
        self.calls.append(("poll_state", job_id))
        return {"state": self._state(job_id=job_id), "previews": list(self.previews)}

    def poll_log(self, job_id: str, offset: int) -> dict:
        self.calls.append(("poll_log", job_id, offset))
        reset = offset < 0 if self.log_reset is None else self.log_reset
        chunk = self.log_chunk
        if self.log_next_offset is not None:
            next_offset = self.log_next_offset
        else:
            start = 0 if offset < 0 else offset
            next_offset = start + len(chunk.encode("utf-8"))
        return {"chunk": chunk, "next_offset": next_offset, "reset": reset}

    def clear_log(self, job_id: str) -> None:
        self.calls.append(("clear_log", job_id))

    def _state(self, *, status: str | None = None, job_id: str | None = None) -> dict:
        return {
            "job_id": job_id or self.job_id,
            "status": status or self.status,
            "step": self.step,
            "epoch": self.epoch,
            "last_error": self.last_error,
            "exit_code": None,
        }

    def _job_payload(self, job_id: str | None = None) -> dict:
        return {
            "job_id": job_id or self.job_id,
            "config_text": self.config_text,
            "state": self._state(job_id=job_id or self.job_id),
            "previews": list(self.previews),
        }

    def kinds(self) -> list[str]:
        return [item[0] for item in self.calls]


def _construct(callbacks=None):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The parameters have been moved from the Blocks constructor",
            category=UserWarning,
        )
        with gr.Blocks(title="Training panel test") as demo:
            start_btn = gr.Button(
                "Start",
                variant="primary",
                elem_id="studio-training-start",
                size="sm",
                visible=True,
            )
            stop_btn = gr.Button(
                "Stop",
                variant="stop",
                elem_id="studio-training-stop",
                size="sm",
                visible=False,
            )
            clear_btn = gr.Button(
                "Clear",
                elem_id="studio-training-clear",
                size="sm",
                visible=True,
            )
            panel = build_training_panel(
                callbacks=callbacks,
                start_btn=start_btn,
                stop_btn=stop_btn,
                clear_btn=clear_btn,
            )
    return demo, panel


def _elem_ids(demo) -> set[str | None]:
    return {getattr(block, "elem_id", None) for block in demo.blocks.values()}


def _block_by_elem_id(demo, elem_id: str):
    return next(
        block
        for block in demo.blocks.values()
        if getattr(block, "elem_id", None) == elem_id
    )


def _fns_named(demo, name: str):
    return [
        fn
        for fn in demo.fns.values()
        if getattr(getattr(fn, "fn", None), "__name__", None) == name
    ]


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_panel_constructs_without_launching_server_models_or_gpu(monkeypatch):
    launched: list[str] = []

    def fake_launch(self, *args, **kwargs):
        launched.append("launch")
        raise AssertionError("tests must not start a Gradio server")

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    demo, panel = _construct()
    assert isinstance(panel, TrainingPanel)
    assert launched == []
    assert getattr(demo, "is_running", False) is False
    assert demo.local_url is None


def _ancestor_chain(block):
    chain = []
    current = block
    while current is not None:
        chain.append(current)
        current = getattr(current, "parent", None)
    return chain


def test_panel_has_required_controls():
    demo, panel = _construct()
    ids = _elem_ids(demo)
    assert "studio-training-panel" in ids
    assert "studio-training-base-name" not in ids
    assert "studio-training-job-id" not in ids
    assert "studio-training-create-open" in ids
    assert "studio-training-job-selector" in ids
    assert "studio-training-yaml-accordion" in ids
    assert "studio-training-yaml" in ids
    assert "studio-training-validate" not in ids
    assert "studio-training-toolbar" not in ids
    assert "studio-training-run" not in ids
    assert "studio-training-job" in ids
    assert "studio-training-save" in ids
    assert "studio-training-image-format" in ids
    assert "studio-training-start" in ids
    assert "studio-training-stop" in ids
    assert "studio-training-clear" in ids
    assert "studio-training-state" in ids
    assert "studio-training-previews" in ids
    assert "studio-training-job-log" in ids
    assert "studio-training-log-delta" in ids
    assert "studio-training-log-accordion" in ids

    assert not hasattr(panel, "base_name")
    assert panel.create_open_btn.value == "Create"
    assert panel.job_selector.label == "Job"
    assert panel.job_selector.allow_custom_value is True
    assert isinstance(panel.job_id, gr.State)
    assert isinstance(panel.yaml_accordion, gr.Accordion)
    assert panel.yaml_accordion.label == "config.yaml"
    assert panel.yaml_accordion.open is False
    assert panel.yaml_accordion.elem_id == "studio-training-yaml-accordion"
    assert panel.yaml_editor.show_label is False
    assert not hasattr(panel, "validate_btn")
    assert panel.save_btn.value == "Save"
    assert isinstance(panel.image_format, gr.Radio)
    assert panel.image_format.elem_id == "studio-training-image-format"
    assert panel.image_format.label == "Format"
    assert panel.image_format.info == "JPEG is smaller; PNG is lossless."
    assert panel.image_format.value == DEFAULT_IMAGE_FORMAT
    choice_values = [
        c[0] if isinstance(c, (list, tuple)) else c
        for c in (panel.image_format.choices or [])
    ]
    assert choice_values == list(IMAGE_FORMAT_CHOICES)
    assert panel.start_btn.value == "Start"
    assert panel.start_btn.visible is True
    assert panel.stop_btn.value == "Stop"
    assert panel.stop_btn.visible is False
    assert panel.clear_btn.value == "Clear"
    assert panel.clear_btn.visible is True
    assert panel.clear_btn.elem_id == "studio-training-clear"
    assert isinstance(panel.yaml_editor, gr.Textbox)
    assert panel.yaml_editor.elem_id == "studio-training-yaml"
    assert isinstance(panel.preview_gallery, gr.Gallery)
    assert panel.preview_gallery.preview is True
    assert isinstance(panel.log_offset, gr.State)
    assert panel.log_offset.value == -1
    assert isinstance(panel.log_generation, gr.State)
    assert panel.log_generation.value == 0
    assert isinstance(panel.log_delta, gr.Textbox)
    assert panel.log_delta.elem_id == "studio-training-log-delta"
    assert panel.job_log.elem_id == "studio-training-job-log"
    assert panel.job_log.value == JOB_LOG_HTML
    assert "id=" not in JOB_LOG_HTML
    assert 'class="studio-training-job-log-pre"' in JOB_LOG_HTML
    assert isinstance(panel.log_accordion, gr.Accordion)
    assert panel.log_accordion.label == "Log"
    assert panel.log_accordion.open is True
    assert panel.log_accordion.elem_id == "studio-training-log-accordion"
    assert panel.log_accordion in _ancestor_chain(panel.job_log)


def test_yaml_editor_inside_collapsed_config_accordion():
    demo, panel = _construct()
    assert panel.yaml_accordion in _ancestor_chain(panel.yaml_editor)
    for btn in (
        panel.save_btn,
        panel.image_format,
        panel.start_btn,
        panel.stop_btn,
        panel.clear_btn,
        panel.create_open_btn,
        panel.job_selector,
    ):
        assert panel.yaml_accordion not in _ancestor_chain(btn)


def test_training_two_column_job_and_preview_layout():
    demo, panel = _construct()
    job = _block_by_elem_id(demo, "studio-training-job")
    assert job in _ancestor_chain(panel.job_selector)
    assert job in _ancestor_chain(panel.create_open_btn)
    assert job not in _ancestor_chain(panel.preview_gallery)
    assert job not in _ancestor_chain(panel.yaml_accordion)
    assert job not in _ancestor_chain(panel.save_btn)

    left_col = job.parent
    assert isinstance(left_col, gr.Column)
    assert getattr(left_col, "scale", None) == 5
    assert left_col in _ancestor_chain(panel.yaml_accordion)
    assert left_col in _ancestor_chain(panel.save_btn)
    assert left_col in _ancestor_chain(panel.image_format)
    assert left_col not in _ancestor_chain(panel.preview_gallery)
    assert left_col not in _ancestor_chain(panel.operational_state)

    body_row = left_col.parent
    assert isinstance(body_row, gr.Row)
    columns = [child for child in body_row.children if isinstance(child, gr.Column)]
    assert len(columns) == 2
    right_col = columns[1]
    assert getattr(right_col, "scale", None) == 6
    assert right_col in _ancestor_chain(panel.preview_gallery)
    assert right_col in _ancestor_chain(panel.operational_state)
    right_children = list(right_col.children)
    assert right_children.index(panel.preview_gallery) < right_children.index(
        panel.operational_state
    )
    assert right_children.index(panel.operational_state) < right_children.index(
        panel.message
    )
    assert right_col not in _ancestor_chain(job)
    assert body_row not in _ancestor_chain(panel.log_accordion)
    assert body_row not in _ancestor_chain(panel.start_btn)
    assert body_row not in _ancestor_chain(panel.stop_btn)
    assert body_row not in _ancestor_chain(panel.clear_btn)


def test_yaml_editor_is_raw_textbox_not_sampling_form():
    demo, panel = _construct()
    labels = [
        getattr(block, "label", None)
        for block in demo.blocks.values()
        if getattr(block, "label", None)
    ]
    assert "guidance_scale" not in labels
    assert "num_inference_steps" not in labels
    assert "Base name" not in labels
    assert "Job ID" not in labels
    assert isinstance(panel.yaml_editor, gr.Textbox)
    assert "guidance_scale" in (panel.yaml_editor.info or "")
    assert "num_inference_steps" in (panel.yaml_editor.info or "")


def test_create_or_open_requires_nonempty_name_and_shows_job_id():
    recorder = RecordingCallbacks(job_id="moi-stil", config_text=CANONICAL_YAML)
    with pytest.raises(gr.Error, match="job name"):
        handle_create_or_open("   ", callbacks=as_training_callbacks(recorder))
    assert "create_or_open" not in recorder.kinds()

    data = handle_create_or_open(
        "Мой стиль",
        callbacks=as_training_callbacks(recorder),
    )
    assert data.job_id == "moi-stil"
    assert ("create_or_open", "Мой стиль") in recorder.calls


def test_create_event_writes_resolved_id_into_hidden_state():
    recorder = RecordingCallbacks(job_id="ascii-job", config_text=CANONICAL_YAML)
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    outputs = create_fn.fn("Style V2")
    assert outputs[0] == "ascii-job"
    assert create_fn.outputs[0] is panel.job_id
    assert isinstance(panel.job_id, gr.State)
    selector_update = _as_update_dict(outputs[2])
    assert selector_update.get("allow_custom_value") is True
    assert selector_update.get("value") == "ascii-job"
    choices = _choice_values(selector_update.get("choices"))
    assert "ascii-job" in choices


def test_existing_slug_loads_without_destructive_rewrite():
    recorder = RecordingCallbacks(
        job_id="existing",
        config_text="sentinel config",
        jobs=["existing"],
    )
    callbacks = as_training_callbacks(recorder)
    data = handle_create_or_open("existing", callbacks=callbacks)

    assert data.job_id == "existing"
    assert data.config_text == "sentinel config"
    assert "create_or_open" in recorder.kinds()
    assert "save_yaml" not in recorder.kinds()
    assert "queue_update" not in recorder.kinds()
    assert "rewrite_job" not in recorder.kinds()
    assert recorder.rewrite_called is False
    assert not hasattr(callbacks, "rewrite_job") or recorder.rewrite_called is False


def test_yaml_editor_bound_to_canonical_config_text():
    recorder = RecordingCallbacks(config_text=CANONICAL_YAML, job_id="demo-job")
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    outputs = create_fn.fn("demo")
    yaml_index = [block.elem_id for block in create_fn.outputs].index(
        "studio-training-yaml"
    )
    assert outputs[yaml_index] == CANONICAL_YAML
    assert panel.yaml_editor.elem_id == "studio-training-yaml"
    assert create_fn.outputs[yaml_index] is panel.yaml_editor

    loaded = handle_load_job("demo-job", callbacks=as_training_callbacks(recorder))
    assert loaded.config_text == CANONICAL_YAML


def test_save_idle_calls_save_callback():
    recorder = RecordingCallbacks(status="stopped")
    callbacks = as_training_callbacks(recorder)
    data = handle_save("demo-job", CANONICAL_YAML, "stopped", callbacks=callbacks)
    assert data.mode == "saved"
    assert ("save_yaml", "demo-job", CANONICAL_YAML) in recorder.calls
    assert "queue_update" not in recorder.kinds()


def test_save_running_gradio_status_still_uses_save_yaml():
    """Gradio status is not authoritative; save_yaml consults state.json."""
    recorder = RecordingCallbacks(status="running")
    callbacks = as_training_callbacks(recorder)
    data = handle_save("demo-job", CANONICAL_YAML, "running", callbacks=callbacks)
    assert data.mode == "saved"
    assert ("save_yaml", "demo-job", CANONICAL_YAML) in recorder.calls
    assert "queue_update" not in recorder.kinds()


def test_save_event_always_routes_to_save_yaml():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    save_fn = _fns_named(demo, "on_save")[0]
    assert save_fn.inputs[0] is panel.job_id
    assert save_fn.inputs[1] is panel.yaml_editor
    assert panel.image_format not in save_fn.inputs
    assert list(save_fn.outputs)[-1] is panel.image_format

    recorder.calls.clear()
    save_fn.fn("demo-job", CANONICAL_YAML, "stopped")
    assert recorder.kinds() == ["save_yaml"]

    recorder.calls.clear()
    save_fn.fn("demo-job", CANONICAL_YAML, "running")
    assert recorder.kinds() == ["save_yaml"]


def test_image_format_radio_loads_from_sampling_yaml():
    recorder = RecordingCallbacks(
        config_text=JPEG_YAML,
        job_id="demo-job",
        jobs=["demo-job"],
    )
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    select_fn = _fns_named(demo, "on_select_job")[0]
    assert create_fn.outputs[13] is panel.image_format

    created = create_fn.fn("demo")
    assert created[13] == "jpeg"
    assert "image_format: jpeg" in created[1]

    recorder.config_text = CANONICAL_YAML
    missing = create_fn.fn("demo")
    assert missing[13] == DEFAULT_IMAGE_FORMAT

    recorder.config_text = PNG_YAML
    recorder.calls.clear()
    selected = select_fn.fn("demo-job")
    assert selected[13] == "png"
    assert "image_format: png" in selected[1]
    assert "save_yaml" not in recorder.kinds()

    recorder.config_text = (
        "job_name: demo\n"
        "sampling:\n"
        "  image_format: gif\n"
    )
    unknown = create_fn.fn("demo")
    assert unknown[13] == DEFAULT_IMAGE_FORMAT


def test_image_format_radio_change_updates_editor_not_save():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    change_fn = _fns_named(demo, "on_image_format_change")[0]
    assert list(change_fn.inputs) == [panel.image_format, panel.yaml_editor]
    assert list(change_fn.outputs) == [panel.yaml_editor]

    recorder.calls.clear()
    dumped = change_fn.fn("png", CANONICAL_YAML)
    parsed = yaml.safe_load(dumped)
    assert parsed["sampling"]["image_format"] == "png"
    assert parsed["sampling"]["guidance_scale"] == 0.0
    assert "image_format: png" in dumped
    assert recorder.kinds() == []

    assert _is_skip(change_fn.fn("png", "{"))
    assert _is_skip(change_fn.fn("png", "- just a list\n"))
    assert _is_skip(change_fn.fn("png", None))


def test_save_refreshes_image_format_radio_from_yaml_text():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    save_fn = _fns_named(demo, "on_save")[0]
    outputs = save_fn.fn("demo-job", PNG_YAML, "stopped")
    assert ("save_yaml", "demo-job", PNG_YAML) in recorder.calls
    assert outputs[0] == PNG_YAML
    assert outputs[5] == "png"
    assert save_fn.outputs[5] is panel.image_format


def test_start_stop_validate_call_injected_callbacks_only():
    recorder = RecordingCallbacks()
    callbacks = as_training_callbacks(recorder)

    handle_validate("demo-job", CANONICAL_YAML, callbacks=callbacks)
    handle_start("demo-job", callbacks=callbacks)
    handle_stop("demo-job", callbacks=callbacks)

    assert ("validate_yaml", "demo-job", CANONICAL_YAML) in recorder.calls
    assert ("start_job", "demo-job") in recorder.calls
    assert ("stop_job", "demo-job") in recorder.calls
    assert "save_yaml" not in recorder.kinds()
    assert "queue_update" not in recorder.kinds()
    assert "rewrite_job" not in recorder.kinds()


def test_start_stop_events_are_wired():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    start_fn = _fns_named(demo, "on_start")[0]
    stop_fn = _fns_named(demo, "on_stop")[0]
    assert _fns_named(demo, "on_validate") == []

    assert start_fn.inputs[0] is panel.job_id
    assert stop_fn.inputs[0] is panel.job_id
    assert isinstance(panel.job_id, gr.State)
    assert list(start_fn.outputs) == [
        panel.operational_state,
        panel.preview_gallery,
        panel.status_state,
        panel.message,
        panel.start_btn,
        panel.stop_btn,
    ]
    assert list(stop_fn.outputs) == list(start_fn.outputs)

    recorder.calls.clear()
    start_fn.fn("demo-job")
    stop_fn.fn("demo-job")
    assert recorder.kinds() == ["start_job", "stop_job"]


def test_clear_log_event_is_wired():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    clear_fn = _fns_named(demo, "on_clear")[0]
    assert list(clear_fn.inputs) == [panel.job_id, panel.log_generation]
    assert list(clear_fn.outputs) == [
        panel.log_offset,
        panel.log_generation,
        panel.log_delta,
        panel.message,
        panel.preview_gallery,
        panel.operational_state,
        panel.status_state,
        panel.start_btn,
        panel.stop_btn,
    ]
    assert clear_fn.targets
    assert clear_fn.targets[0][1] == "click"
    assert clear_fn.targets[0][0] == getattr(panel.clear_btn, "_id", panel.clear_btn)

    recorder.calls.clear()
    (
        offset,
        generation,
        delta,
        message,
        gallery,
        operational_state,
        status_state,
        start_vis,
        stop_vis,
    ) = clear_fn.fn("demo-job", 3)
    assert recorder.kinds() == ["clear_log"]
    assert recorder.calls == [("clear_log", "demo-job")]
    assert offset == 0
    assert generation == 4
    payload = json.loads(delta)
    assert payload["reset"] is True
    assert payload["chunk"] == ""
    assert payload["generation"] == 4
    assert message == "Log, previews, and progress cleared."
    assert gallery == []
    assert "**Status:** `stopped`" in operational_state
    assert "**Step:** 0" in operational_state
    assert "**Epoch:** 0" in operational_state
    assert status_state == "stopped"
    assert _as_update_dict(start_vis).get("visible") is True
    assert _as_update_dict(stop_vis).get("visible") is False


def test_start_stop_visibility_follows_running():
    recorder = RecordingCallbacks(status="stopped")
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    select_fn = _fns_named(demo, "on_select_job")[0]
    start_fn = _fns_named(demo, "on_start")[0]
    stop_fn = _fns_named(demo, "on_stop")[0]
    poll_fn = _fns_named(demo, "on_poll")[0]
    idle_statuses = ("", "stopped", "completed", "failed")
    assert panel.clear_btn.visible is True
    assert panel.clear_btn not in create_fn.outputs
    assert panel.clear_btn not in start_fn.outputs
    assert panel.clear_btn not in stop_fn.outputs
    assert panel.clear_btn not in poll_fn.outputs

    def assert_idle(start_update, stop_update):
        assert _as_update_dict(start_update).get("visible") is True
        assert _as_update_dict(stop_update).get("visible") is False

    def assert_running(start_update, stop_update):
        assert _as_update_dict(start_update).get("visible") is False
        assert _as_update_dict(stop_update).get("visible") is True

    empty_load = select_fn.fn("", 0)
    assert len(empty_load) == 14
    assert_idle(empty_load[11], empty_load[12])
    assert empty_load[13] == DEFAULT_IMAGE_FORMAT

    for status in idle_statuses:
        recorder.status = status
        loaded = create_fn.fn("demo")
        assert_idle(loaded[11], loaded[12])

    recorder.status = "running"
    loaded_running = create_fn.fn("demo")
    assert_running(loaded_running[11], loaded_running[12])

    recorder.status = "stopped"
    started = start_fn.fn("demo-job")
    assert len(started) == 6
    assert_running(started[4], started[5])

    stopped = stop_fn.fn("demo-job")
    assert len(stopped) == 6
    assert_idle(stopped[4], stopped[5])

    assert len(poll_fn.outputs) == 10
    assert poll_fn.outputs[8] is panel.start_btn
    assert poll_fn.outputs[9] is panel.stop_btn
    assert panel.image_format not in poll_fn.outputs
    for status in idle_statuses:
        recorder.status = status
        polled = poll_fn.fn("demo-job", 0)
        assert len(polled) == 10
        assert_idle(polled[8], polled[9])

    recorder.status = "running"
    polled_running = poll_fn.fn("demo-job", 0)
    assert_running(polled_running[8], polled_running[9])


def test_operational_state_has_no_metrics_history():
    markdown = format_operational_state(
        {
            "job_id": "demo-job",
            "status": "running",
            "step": 12,
            "epoch": 1,
            "last_error": None,
            "loss": 0.42,
        }
    )
    assert "**Status:** `running`" in markdown
    assert "**Step:** 12" in markdown
    assert "**Epoch:** 1" in markdown
    assert "loss" not in markdown
    assert "history" not in markdown.lower()


def test_preview_gallery_uses_injected_paths():
    recorder = RecordingCallbacks(previews=["jobs/demo-job/previews/step-10.png"])
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    outputs = create_fn.fn("demo")
    gallery_index = [getattr(b, "elem_id", None) for b in create_fn.outputs].index(
        "studio-training-previews"
    )
    assert outputs[gallery_index] == ["jobs/demo-job/previews/step-10.png"]
    assert create_fn.outputs[gallery_index] is panel.preview_gallery


def test_selector_change_loads_job_without_save():
    recorder = RecordingCallbacks(
        job_id="other",
        config_text="other: yaml\n",
        jobs=["other"],
    )
    demo, _panel = _construct(recorder)
    select_fn = _fns_named(demo, "on_select_job")[0]
    recorder.calls.clear()
    outputs = select_fn.fn("other")
    assert outputs[0] == "other"
    assert "other: yaml" in outputs[1]
    assert recorder.kinds() == ["list_jobs", "load_job"]
    assert "save_yaml" not in recorder.kinds()
    assert "create_or_open" not in recorder.kinds()


def test_selector_custom_name_skips_load_and_create():
    recorder = RecordingCallbacks(jobs=["known-job"], job_id="known-job")
    demo, panel = _construct(recorder)
    select_fn = _fns_named(demo, "on_select_job")[0]
    recorder.calls.clear()
    outputs = select_fn.fn("Brand new style")
    assert len(outputs) == 14
    for item in outputs:
        assert _is_skip(item)
    assert "load_job" not in recorder.kinds()
    assert "create_or_open" not in recorder.kinds()
    assert panel.job_selector.allow_custom_value is True


def test_create_uses_job_selector_value():
    recorder = RecordingCallbacks(job_id="new-style", config_text=CANONICAL_YAML)
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    assert create_fn.inputs[0] is panel.job_selector
    recorder.calls.clear()
    outputs = create_fn.fn("New Style")
    assert ("create_or_open", "New Style") in recorder.calls
    assert outputs[0] == "new-style"
    selector_update = _as_update_dict(outputs[2])
    assert selector_update.get("allow_custom_value") is True
    assert selector_update.get("value") == "new-style"

def test_actions_require_an_open_job():
    callbacks = noop_training_callbacks()
    with pytest.raises(gr.Error, match="training job"):
        handle_validate("", "x: 1\n", callbacks=callbacks)
    with pytest.raises(gr.Error, match="training job"):
        handle_save("", "x: 1\n", "stopped", callbacks=callbacks)
    with pytest.raises(gr.Error, match="training job"):
        handle_start(None, callbacks=callbacks)
    with pytest.raises(gr.Error, match="training job"):
        handle_stop(None, callbacks=callbacks)
    with pytest.raises(gr.Error, match="training job"):
        handle_clear_log("", 0, callbacks=callbacks)


def test_validate_surfaces_callback_error():
    recorder = RecordingCallbacks(validate_ok=False, validate_error="unknown key")
    data = handle_validate(
        "demo-job",
        "bad: true\n",
        callbacks=as_training_callbacks(recorder),
    )
    assert data.message == "unknown key"


def test_module_source_and_import_graph_are_isolated():
    source = PANEL_SOURCE.read_text(encoding="utf-8")
    imported = _imported_modules(source)
    forbidden_exact = {
        "subprocess",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "torchao",
    }
    assert imported & forbidden_exact == set()
    for name in imported:
        assert not name.startswith("zimage.engine")
        assert not name.startswith("zimage.training")
        assert "modeling" not in name.split(".")
        assert "runtime_guard" not in name.split(".")
        assert name.split(".")[-1] != "loop"
        assert name.split(".")[-1] != "pipeline"

    text = inspect.getsource(sys.modules["zimage.ui.training_panel"])
    assert "import subprocess" not in text
    assert "zimage.engine" not in text
    assert "zimage.training.modeling" not in text
    assert "zimage.training.loop" not in text
    assert "runtime_guard" not in text

    module_globals = sys.modules["zimage.ui.training_panel"].__dict__
    for name in ("subprocess", "torch", "diffusers", "transformers"):
        assert name not in module_globals


def test_fresh_import_does_not_load_engine_training_runtime_or_ml():
    code = r"""
import ast
import importlib.util
import sys
from pathlib import Path

path = Path(r"%s")
source = path.read_text(encoding="utf-8")
imported = set()
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
        imported.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module)
forbidden_imports = (
    "subprocess",
    "torch",
    "diffusers",
    "transformers",
    "zimage.engine",
    "zimage.engine.pipeline",
    "zimage.training",
    "zimage.training.modeling",
    "zimage.training.loop",
    "zimage.training.runtime_guard",
)
leaked_imports = [name for name in imported if name in forbidden_imports
                  or name.startswith("zimage.engine")
                  or name.startswith("zimage.training")]
if leaked_imports:
    print(",".join(leaked_imports), file=sys.stderr)
    raise SystemExit(1)

spec = importlib.util.spec_from_file_location("training_panel_isolated", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

forbidden_runtime = (
    "zimage.engine",
    "zimage.engine.pipeline",
    "zimage.training",
    "zimage.training.modeling",
    "zimage.training.loop",
    "zimage.training.runtime_guard",
    "zimage.training.jobs",
    "zimage.training.cli",
    "torch",
    "diffusers",
    "transformers",
)
loaded = [name for name in forbidden_runtime if name in sys.modules]
if loaded:
    print(",".join(loaded), file=sys.stderr)
    raise SystemExit(1)
if not hasattr(module, "build_training_panel"):
    raise SystemExit(2)
""" % PANEL_SOURCE
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0, result.stderr


def test_noop_callbacks_allow_construction_without_production_wiring():
    callbacks = noop_training_callbacks()
    assert callbacks.list_jobs() == []
    demo, panel = _construct(None)
    assert panel.job_selector.choices in ([], None) or _choice_values(
        panel.job_selector.choices
    ) == []
    assert isinstance(demo, gr.Blocks)


def test_as_training_callbacks_accepts_duck_typed_host():
    host = RecordingCallbacks(job_id="from-host")
    adapted = as_training_callbacks(host)
    assert isinstance(adapted, TrainingCallbacks)
    data = handle_create_or_open("from-host", callbacks=adapted)
    assert data.job_id == "from-host"


def test_job_panel_data_is_the_handler_contract():
    data = JobPanelData(job_id="x", config_text="a: 1\n", mode="saved")
    assert data.job_id == "x"
    assert data.mode == "saved"


def test_create_or_open_accepts_attribute_state_objects():
    state = SimpleNamespace(
        job_id="attr-job",
        status=SimpleNamespace(value="stopped"),
        step=3,
        epoch=1,
        last_error=None,
        exit_code=None,
    )

    class Host:
        def list_jobs(self):
            return ["attr-job"]

        def create_or_open(self, name):
            return {
                "job_id": "attr-job",
                "config_text": "job_name: attr\n",
                "state": state,
                "previews": [],
            }

        def load_job(self, job_id):
            raise AssertionError(job_id)

        def validate_yaml(self, job_id, text):
            raise AssertionError(job_id)

        def save_yaml(self, job_id, text):
            raise AssertionError(job_id)

        def start_job(self, job_id):
            raise AssertionError(job_id)

        def stop_job(self, job_id):
            raise AssertionError(job_id)

        def poll_state(self, job_id):
            raise AssertionError(job_id)

        def poll_log(self, job_id, offset):
            raise AssertionError(job_id)

        def clear_log(self, job_id):
            raise AssertionError(job_id)

    data = handle_create_or_open("attr", callbacks=as_training_callbacks(Host()))
    assert data.job_id == "attr-job"
    assert data.state["step"] == 3
    assert format_operational_state(data.state).startswith("**Status:** `stopped`")


def test_poll_log_is_on_callbacks_and_duck_typed_hosts():
    recorder = RecordingCallbacks(log_chunk="step 1\n")
    adapted = as_training_callbacks(recorder)
    payload = adapted.poll_log("demo-job", -1)
    assert payload["chunk"] == "step 1\n"
    assert payload["reset"] is True
    assert ("poll_log", "demo-job", -1) in recorder.calls
    noop = noop_training_callbacks()
    empty = noop.poll_log("x", -1)
    assert empty["chunk"] == ""
    assert empty["reset"] is True


def test_log_accordion_is_full_width_not_in_preview_row():
    demo, panel = _construct()
    panel_root = _block_by_elem_id(demo, "studio-training-panel")
    assert panel_root in _ancestor_chain(panel.preview_gallery)
    preview_row = next(
        block
        for block in _ancestor_chain(panel.preview_gallery)
        if isinstance(block, gr.Row)
    )
    assert preview_row not in _ancestor_chain(panel.job_log)
    assert panel.log_accordion in _ancestor_chain(panel.job_log)
    assert panel.log_accordion not in _ancestor_chain(panel.log_delta)


def test_create_and_load_reset_offset_and_bump_generation():
    recorder = RecordingCallbacks(
        job_id="job-a",
        jobs=["job-a", "job-b"],
        config_text=CANONICAL_YAML,
    )
    demo, panel = _construct(recorder)
    create_fn = _fns_named(demo, "on_create_or_open")[0]
    select_fn = _fns_named(demo, "on_select_job")[0]
    assert len(create_fn.outputs) == 14
    assert create_fn.outputs[8] is panel.log_offset
    assert create_fn.outputs[9] is panel.log_generation
    assert create_fn.outputs[10] is panel.log_delta
    assert create_fn.outputs[11] is panel.start_btn
    assert create_fn.outputs[12] is panel.stop_btn
    assert create_fn.outputs[13] is panel.image_format
    assert panel.job_log not in create_fn.outputs

    first = create_fn.fn("job-a", 0)
    assert first[8] == -1
    assert first[9] == 1
    delta_a = json.loads(first[10])
    assert delta_a["reset"] is True
    assert delta_a["chunk"] == ""
    assert delta_a["generation"] == 1

    recorder.job_id = "job-b"
    recorder.config_text = "job_name: b\n"
    second = select_fn.fn("job-b", first[9])
    assert second[8] == -1
    assert second[9] == 2
    delta_b = json.loads(second[10])
    assert delta_b["reset"] is True
    assert delta_b["generation"] == 2

    recorder.job_id = "job-a"
    recorder.config_text = CANONICAL_YAML
    again = select_fn.fn("job-a", second[9])
    assert again[8] == -1
    assert again[9] == 3
    delta_again = json.loads(again[10])
    assert delta_again["reset"] is True
    assert delta_again["generation"] == 3
    assert again[10] != first[10]


def test_cas_skip_on_mismatched_job_ids():
    offset, delta = commit_training_log(
        "job-b", "job-a", "stale chunk\n", 99, False, 1, 1
    )
    assert _is_skip(offset)
    assert _is_skip(delta)

    offset, delta = commit_training_log("job-a", "job-a", "hello\n", 12, False, 4, 4)
    assert offset == 12
    payload = json.loads(delta)
    assert payload["chunk"] == "hello\n"
    assert payload["reset"] is False
    assert payload["generation"] == 4

    offset, delta = commit_training_log("job-a", "job-a", "", 12, False, 4, 4)
    assert offset == 12
    assert _is_skip(delta)


def test_cas_skip_on_mismatched_generation():
    offset, delta = commit_training_log(
        "job-a", "job-a", "stale chunk\n", 99, False, 5, 4
    )
    assert _is_skip(offset)
    assert _is_skip(delta)

    offset, delta = commit_training_log("job-a", "job-a", "hello\n", 12, False, 4, 4)
    assert offset == 12
    payload = json.loads(delta)
    assert payload["chunk"] == "hello\n"
    assert payload["generation"] == 4


def test_on_poll_returns_delta_not_full_log_and_calls_poll_log():
    recorder = RecordingCallbacks(
        log_chunk="new line\n",
        previews=["jobs/demo-job/previews/step-1.png"],
    )
    demo, panel = _construct(recorder)
    poll_fn = _fns_named(demo, "on_poll")[0]
    cas_fn = _fns_named(demo, "commit_training_log")[0]
    assert poll_fn.inputs[0] is panel.job_id
    assert poll_fn.inputs[1] is panel.log_offset
    assert poll_fn.inputs[2] is panel.log_generation
    assert len(poll_fn.inputs) == 3
    assert len(poll_fn.outputs) == 10
    assert poll_fn.outputs[8] is panel.start_btn
    assert poll_fn.outputs[9] is panel.stop_btn
    assert panel.image_format not in poll_fn.outputs
    assert panel.job_log not in poll_fn.outputs
    assert panel.log_delta not in poll_fn.outputs
    assert cas_fn.inputs[5] is panel.log_generation
    assert len(cas_fn.inputs) == 7
    assert cas_fn.outputs[0] is panel.log_offset
    assert cas_fn.outputs[1] is panel.log_delta
    assert cas_fn.trigger_after is not None

    recorder.calls.clear()
    outputs = poll_fn.fn("demo-job", 4, 7)
    assert recorder.kinds() == ["poll_state", "poll_log"]
    assert ("poll_log", "demo-job", 4) in recorder.calls
    assert len(outputs) == 10
    assert outputs[1] == ["jobs/demo-job/previews/step-1.png"]
    assert outputs[3] == "demo-job"
    assert outputs[4] == "new line\n"
    assert outputs[5] == 4 + len("new line\n")
    assert outputs[6] is False
    assert outputs[7] == 7
    for item in outputs[:8]:
        assert "visible" not in _as_update_dict(item)
    assert _as_update_dict(outputs[8]).get("visible") is True
    assert _as_update_dict(outputs[9]).get("visible") is False
    joined = " ".join(str(item) for item in outputs)
    assert "new line\nnew line\n" not in joined
    assert panel.job_log.value == JOB_LOG_HTML

    empty = poll_fn.fn("", 4, 7)
    assert len(empty) == 10
    assert all(_is_skip(item) for item in empty)

    js_fns = [
        fn
        for fn in demo.fns.values()
        if getattr(fn, "js", None) and "__zimageApplyTrainingLogDelta" in str(fn.js)
    ]
    assert js_fns
    assert APPLY_TRAINING_LOG_JS in {fn.js for fn in js_fns}
    assert "MutationObserver" not in CUSTOM_JS
    assert "textContent" in CUSTOM_JS
    source = PANEL_SOURCE.read_text(encoding="utf-8")
    assert "MutationObserver" not in source
    assert "zimage.training" not in source


def test_js_then_listeners_receive_log_delta():
    demo, panel = _construct()
    js_fns = [
        fn
        for fn in demo.fns.values()
        if getattr(fn, "js", None) and fn.js == APPLY_TRAINING_LOG_JS
    ]
    assert len(js_fns) == 4
    for fn in js_fns:
        assert list(fn.inputs) == [panel.log_delta]
        assert fn.targets == [(None, "then")]


def test_handle_poll_log_swallows_callback_errors():
    class Host:
        def list_jobs(self):
            return []

        def create_or_open(self, name):
            raise AssertionError(name)

        def load_job(self, job_id):
            raise AssertionError(job_id)

        def validate_yaml(self, job_id, text):
            raise AssertionError(job_id)

        def save_yaml(self, job_id, text):
            raise AssertionError(job_id)

        def start_job(self, job_id):
            raise AssertionError(job_id)

        def stop_job(self, job_id):
            raise AssertionError(job_id)

        def poll_state(self, job_id):
            raise AssertionError(job_id)

        def poll_log(self, job_id, offset):
            raise RuntimeError("read failed")

        def clear_log(self, job_id):
            raise AssertionError(job_id)

    payload = handle_poll_log("demo-job", 8, callbacks=as_training_callbacks(Host()))
    assert payload == {"chunk": "", "next_offset": 8, "reset": False}


def test_handle_clear_log_resets_offset_and_bumps_generation():
    recorder = RecordingCallbacks()
    callbacks = as_training_callbacks(recorder)
    (
        offset,
        generation,
        delta,
        message,
        gallery,
        operational_state,
        status_state,
        start_vis,
        stop_vis,
    ) = handle_clear_log("demo-job", 3, callbacks=callbacks)
    assert ("clear_log", "demo-job") in recorder.calls
    assert "poll_state" not in recorder.kinds()
    assert offset == 0
    assert generation == 4
    payload = json.loads(delta)
    assert payload["reset"] is True
    assert payload["chunk"] == ""
    assert payload["generation"] == 4
    assert message == "Log, previews, and progress cleared."
    assert gallery == []
    assert "**Status:** `stopped`" in operational_state
    assert "**Step:** 0" in operational_state
    assert "**Epoch:** 0" in operational_state
    assert status_state == "stopped"
    assert _as_update_dict(start_vis).get("visible") is True
    assert _as_update_dict(stop_vis).get("visible") is False

    noop_result = handle_clear_log(
        "demo-job", 0, callbacks=noop_training_callbacks()
    )
    assert "**Status:** `stopped`" in noop_result[5]
    assert "**Step:** 0" in noop_result[5]
    assert "**Epoch:** 0" in noop_result[5]
    assert noop_result[6] == "stopped"


def test_clear_log_is_on_callbacks_and_duck_typed_hosts():
    recorder = RecordingCallbacks()
    adapted = as_training_callbacks(recorder)
    adapted.clear_log("demo-job")
    assert ("clear_log", "demo-job") in recorder.calls
    noop = noop_training_callbacks()
    noop.clear_log("x")
    from zimage.ui import training_panel as panel_mod

    assert "handle_clear_log" in panel_mod.__all__



def _choice_values(choices) -> list:
    values = []
    for choice in choices or []:
        if isinstance(choice, (list, tuple)):
            values.append(choice[0])
        else:
            values.append(choice)
    return values


def _as_update_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "keys"):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in ("choices", "value", "allow_custom_value", "visible", "__type__")
        if hasattr(value, key)
    }


def _is_skip(value) -> bool:
    if type(value).__name__ in {"SkipMessage", "_Skip"}:
        return True
    data = _as_update_dict(value)
    # Gradio skip() is an empty update dict with only __type__.
    return data.get("__type__") == "update" and set(data) <= {"__type__"}
