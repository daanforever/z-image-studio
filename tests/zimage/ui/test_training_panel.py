from __future__ import annotations

import ast
import inspect
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from zimage.ui.training_panel import (
    JobPanelData,
    TrainingCallbacks,
    TrainingPanel,
    as_training_callbacks,
    build_training_panel,
    format_operational_state,
    handle_create_or_open,
    handle_load_job,
    handle_save,
    handle_start,
    handle_stop,
    handle_validate,
    noop_training_callbacks,
)


CANONICAL_YAML = (
    "job_name: demo\n"
    "sampling:\n"
    "  common_parameters:\n"
    "    guidance_scale: 0.0\n"
    "    num_inference_steps: 9\n"
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
            panel = build_training_panel(callbacks=callbacks)
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


def test_panel_has_required_controls():
    demo, panel = _construct()
    ids = _elem_ids(demo)
    assert "studio-training-panel" in ids
    assert "studio-training-base-name" not in ids
    assert "studio-training-job-id" not in ids
    assert "studio-training-create-open" in ids
    assert "studio-training-job-selector" in ids
    assert "studio-training-yaml" in ids
    assert "studio-training-validate" in ids
    assert "studio-training-save" in ids
    assert "studio-training-start" in ids
    assert "studio-training-stop" in ids
    assert "studio-training-state" in ids
    assert "studio-training-previews" in ids

    assert not hasattr(panel, "base_name")
    assert panel.create_open_btn.value == "Create"
    assert panel.job_selector.label == "Job"
    assert panel.job_selector.allow_custom_value is True
    assert isinstance(panel.job_id, gr.State)
    assert panel.yaml_editor.label == "config.yaml"
    assert panel.validate_btn.value == "Validate"
    assert panel.save_btn.value == "Save"
    assert panel.start_btn.value == "Start"
    assert panel.stop_btn.value == "Stop"
    assert isinstance(panel.yaml_editor, gr.Textbox)
    assert isinstance(panel.preview_gallery, gr.Gallery)
    assert panel.preview_gallery.preview is True


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

    recorder.calls.clear()
    save_fn.fn("demo-job", CANONICAL_YAML, "stopped")
    assert recorder.kinds() == ["save_yaml"]

    recorder.calls.clear()
    save_fn.fn("demo-job", CANONICAL_YAML, "running")
    assert recorder.kinds() == ["save_yaml"]


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


def test_start_stop_validate_events_are_wired():
    recorder = RecordingCallbacks()
    demo, panel = _construct(recorder)
    validate_fn = _fns_named(demo, "on_validate")[0]
    start_fn = _fns_named(demo, "on_start")[0]
    stop_fn = _fns_named(demo, "on_stop")[0]

    assert validate_fn.inputs[0] is panel.job_id
    assert validate_fn.inputs[1] is panel.yaml_editor
    assert start_fn.inputs[0] is panel.job_id
    assert stop_fn.inputs[0] is panel.job_id
    assert isinstance(panel.job_id, gr.State)

    recorder.calls.clear()
    validate_fn.fn("demo-job", CANONICAL_YAML)
    start_fn.fn("demo-job")
    stop_fn.fn("demo-job")
    assert recorder.kinds() == ["validate_yaml", "start_job", "stop_job"]


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
    assert len(outputs) == 8
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

    data = handle_create_or_open("attr", callbacks=as_training_callbacks(Host()))
    assert data.job_id == "attr-job"
    assert data.state["step"] == 3
    assert format_operational_state(data.state).startswith("**Status:** `stopped`")


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
        for key in ("choices", "value", "allow_custom_value", "__type__")
        if hasattr(value, key)
    }


def _is_skip(value) -> bool:
    if type(value).__name__ in {"SkipMessage", "_Skip"}:
        return True
    data = _as_update_dict(value)
    # Gradio skip() is an empty update dict with only __type__.
    return data.get("__type__") == "update" and set(data) <= {"__type__"}
