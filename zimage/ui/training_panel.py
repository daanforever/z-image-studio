"""Isolated Gradio Training tab body. Talks only to injected callbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import gradio as gr

JOB_LOG_HTML = '<pre class="studio-training-job-log-pre"></pre>'
APPLY_TRAINING_LOG_JS = (
    "(...args) => window.__zimageApplyTrainingLogDelta(args[args.length - 1])"
)

ACTIVE_JOB_STATUS = "running"
SAVE_MODE_QUEUED = "queued"
SAVE_MODE_SAVED = "saved"


class TrainingCallbackAPI(Protocol):
    """Operations the Training panel may invoke. No process or GPU work."""

    def list_jobs(self) -> Sequence[str]:
        """Return known job IDs."""

    def create_or_open(self, name: str) -> Mapping[str, Any]:
        """Create a job or open the existing slug. Must not rewrite files."""

    def load_job(self, job_id: str) -> Mapping[str, Any]:
        """Load canonical YAML, operational state, and preview paths."""

    def validate_yaml(self, job_id: str, text: str) -> Mapping[str, Any]:
        """Validate editor text. Return ``ok`` / ``error``."""

    def save_yaml(self, job_id: str, text: str) -> Mapping[str, Any]:
        """Idle validated atomic save. Return ``mode`` of ``saved``."""

    def start_job(self, job_id: str) -> Any:
        """Request start. Caller must not spawn a process here."""

    def stop_job(self, job_id: str) -> Any:
        """Request stop. Caller must not manage a process here."""

    def poll_state(self, job_id: str) -> Mapping[str, Any]:
        """Return the latest operational state and preview paths."""

    def poll_log(self, job_id: str, offset: int) -> Mapping[str, Any]:
        """Return ``{chunk, next_offset, reset}`` for the job log."""

    def clear_log(self, job_id: str) -> None:
        """Truncate the job log. Missing file is a no-op."""


@dataclass(frozen=True)
class TrainingCallbacks:
    """Injected training operations used by the panel."""

    list_jobs: Callable[[], Sequence[str]]
    create_or_open: Callable[[str], Mapping[str, Any]]
    load_job: Callable[[str], Mapping[str, Any]]
    validate_yaml: Callable[[str, str], Mapping[str, Any]]
    save_yaml: Callable[[str, str], Mapping[str, Any]]
    start_job: Callable[[str], Any]
    stop_job: Callable[[str], Any]
    poll_state: Callable[[str], Mapping[str, Any]]
    poll_log: Callable[[str, int], Mapping[str, Any]]
    clear_log: Callable[[str], Any]
    queue_update: Callable[[str, str], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class JobPanelData:
    """Normalized callback result used to refresh panel components."""

    job_id: str = ""
    config_text: str | None = None
    jobs: tuple[str, ...] | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    previews: tuple[str, ...] | None = None
    message: str = ""
    mode: str | None = None


@dataclass
class TrainingPanel:
    """Gradio components for the Training tab. Mount inside a parent Blocks."""

    root: gr.Column
    create_open_btn: gr.Button
    job_id: gr.State
    job_selector: gr.Dropdown
    yaml_accordion: gr.Accordion
    yaml_editor: gr.Textbox
    save_btn: gr.Button
    start_btn: gr.Button
    stop_btn: gr.Button
    clear_btn: gr.Button
    operational_state: gr.Markdown
    preview_gallery: gr.Gallery
    message: gr.Markdown
    status_state: gr.State
    poll_timer: gr.Timer
    log_offset: gr.State
    log_generation: gr.State
    log_delta: gr.Textbox
    log_accordion: gr.Accordion
    job_log: gr.HTML


def empty_job_state(job_id: str = "") -> dict[str, Any]:
    """Operational snapshot with no metrics or history."""
    return {
        "job_id": job_id,
        "status": "stopped",
        "step": 0,
        "epoch": 0,
        "last_error": None,
        "exit_code": None,
    }


def noop_training_callbacks() -> TrainingCallbacks:
    """No-op callbacks for construction tests. Production wiring is separate."""

    def _empty_job(_name: str = "") -> dict[str, Any]:
        return {
            "job_id": "",
            "config_text": "",
            "state": empty_job_state(),
            "previews": [],
        }

    return TrainingCallbacks(
        list_jobs=lambda: [],
        create_or_open=lambda name: _empty_job(name),
        load_job=lambda job_id: _empty_job(job_id),
        validate_yaml=lambda job_id, text: {"ok": True},
        save_yaml=lambda job_id, text: {"mode": SAVE_MODE_SAVED},
        start_job=lambda job_id: empty_job_state(job_id),
        stop_job=lambda job_id: empty_job_state(job_id),
        poll_state=lambda job_id: {
            "state": empty_job_state(job_id),
            "previews": [],
        },
        poll_log=lambda _job_id, offset: {
            "chunk": "",
            "next_offset": 0 if not isinstance(offset, int) or offset < 0 else offset,
            "reset": isinstance(offset, int) and offset < 0,
        },
        clear_log=lambda _job_id: None,
        queue_update=lambda job_id, text: {"mode": SAVE_MODE_QUEUED},
    )


def as_training_callbacks(
    callbacks: TrainingCallbacks | TrainingCallbackAPI | None,
) -> TrainingCallbacks:
    """Accept a dataclass or any object exposing the callback methods."""
    if callbacks is None:
        return noop_training_callbacks()
    if isinstance(callbacks, TrainingCallbacks):
        return callbacks
    return TrainingCallbacks(
        list_jobs=callbacks.list_jobs,
        create_or_open=callbacks.create_or_open,
        load_job=callbacks.load_job,
        validate_yaml=callbacks.validate_yaml,
        save_yaml=callbacks.save_yaml,
        start_job=callbacks.start_job,
        stop_job=callbacks.stop_job,
        poll_state=callbacks.poll_state,
        poll_log=callbacks.poll_log,
        clear_log=callbacks.clear_log,
        queue_update=getattr(callbacks, "queue_update", None),
    )


def format_operational_state(state: Mapping[str, Any] | object | None) -> str:
    """Render status / step / epoch / last_error. No metrics history."""
    data = coerce_job_state(state)
    if not data.get("job_id") and not data.get("status"):
        return "_No training job selected._"
    last_error = data.get("last_error")
    if last_error in (None, ""):
        last_error = "—"
    return (
        f"**Status:** `{data.get('status') or '—'}`\n\n"
        f"**Step:** {data.get('step', 0)}\n\n"
        f"**Epoch:** {data.get('epoch', 0)}\n\n"
        f"**Last error:** {last_error}"
    )


def is_active_status(status: Any) -> bool:
    """True when the job is running and Save must queue an update."""
    return _status_value(status) == ACTIVE_JOB_STATUS


def coerce_job_state(value: Any) -> dict[str, Any]:
    """Accept a mapping, a nested ``state`` payload, or attribute object."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        nested = value.get("state")
        if "status" not in value and nested is not None and nested is not value:
            merged = coerce_job_state(nested)
            if merged:
                if not merged.get("job_id"):
                    merged["job_id"] = _text(value.get("job_id"))
                return merged
        return {
            "job_id": _text(value.get("job_id")),
            "status": _status_value(value.get("status")),
            "step": value.get("step", 0),
            "epoch": value.get("epoch", 0),
            "last_error": value.get("last_error"),
            "exit_code": value.get("exit_code"),
        }
    return {
        "job_id": _text(getattr(value, "job_id", "")),
        "status": _status_value(getattr(value, "status", "")),
        "step": getattr(value, "step", 0),
        "epoch": getattr(value, "epoch", 0),
        "last_error": getattr(value, "last_error", None),
        "exit_code": getattr(value, "exit_code", None),
    }


def handle_create_or_open(
    name: Any,
    *,
    callbacks: TrainingCallbacks,
) -> JobPanelData:
    """Create or open a job and return canonical YAML plus resolved job ID."""
    text = _required_base_name(name)
    payload = _invoke(callbacks.create_or_open, text)
    data = _job_payload(payload, fallback_id="")
    jobs = _invoke_list_jobs(callbacks, extra=data.job_id)
    return JobPanelData(
        job_id=data.job_id,
        config_text=data.config_text or "",
        jobs=jobs,
        state=data.state,
        previews=data.previews if data.previews is not None else (),
        message=data.message
        or (f"Opened `{data.job_id}`." if data.job_id else ""),
    )


def handle_load_job(
    job_id: Any,
    *,
    callbacks: TrainingCallbacks,
) -> JobPanelData:
    """Load an existing job into the editor without rewriting files."""
    resolved = _job_id_text(job_id)
    if not resolved:
        return JobPanelData(
            job_id="",
            config_text="",
            state={},
            previews=(),
            message="",
        )
    payload = _invoke(callbacks.load_job, resolved)
    data = _job_payload(payload, fallback_id=resolved)
    return JobPanelData(
        job_id=data.job_id or resolved,
        config_text=data.config_text or "",
        jobs=None,
        state=data.state,
        previews=data.previews if data.previews is not None else (),
        message=data.message,
    )


def handle_validate(
    job_id: Any,
    text: Any,
    *,
    callbacks: TrainingCallbacks,
) -> JobPanelData:
    """Validate editor YAML through the injected callback only."""
    resolved = _require_job_id(job_id)
    payload = _invoke(callbacks.validate_yaml, resolved, _yaml_text(text))
    ok, error = _validation_result(payload)
    if ok:
        return JobPanelData(job_id=resolved, message="Valid.")
    return JobPanelData(job_id=resolved, message=error or "Invalid YAML.")


def handle_save(
    job_id: Any,
    text: Any,
    status: Any,
    *,
    callbacks: TrainingCallbacks,
) -> JobPanelData:
    """Always persist via ``save_yaml`` (which consults ``state.json``).

    Gradio ``status`` is not authoritative for queue vs idle save ownership.
    """
    resolved = _require_job_id(job_id)
    yaml_text = _yaml_text(text)
    payload = _invoke(callbacks.save_yaml, resolved, yaml_text)
    mode = _save_mode(payload, SAVE_MODE_SAVED)
    data = _job_payload(payload, fallback_id=resolved)
    if mode == SAVE_MODE_QUEUED:
        default_message = "Update queued."
    else:
        default_message = "Saved."
    message = data.message or default_message
    return JobPanelData(
        job_id=data.job_id or resolved,
        config_text=data.config_text,
        state=data.state or {},
        previews=data.previews,
        message=message,
        mode=mode,
    )


def handle_start(job_id: Any, *, callbacks: TrainingCallbacks) -> JobPanelData:
    """Ask the injected callback to start. Does not spawn a process."""
    resolved = _require_job_id(job_id)
    payload = _invoke(callbacks.start_job, resolved)
    data = _job_payload(payload, fallback_id=resolved)
    return JobPanelData(
        job_id=data.job_id or resolved,
        state=data.state or empty_job_state(resolved),
        previews=data.previews,
        message=data.message or "Start requested.",
    )


def handle_stop(job_id: Any, *, callbacks: TrainingCallbacks) -> JobPanelData:
    """Ask the injected callback to stop. Does not manage a process."""
    resolved = _require_job_id(job_id)
    payload = _invoke(callbacks.stop_job, resolved)
    data = _job_payload(payload, fallback_id=resolved)
    return JobPanelData(
        job_id=data.job_id or resolved,
        state=data.state or empty_job_state(resolved),
        previews=data.previews,
        message=data.message or "Stop requested.",
    )


def handle_clear_log(
    job_id: Any,
    generation: Any,
    *,
    callbacks: TrainingCallbacks,
) -> tuple[int, int, str, str]:
    """Ask the injected callback to truncate the log. Does not touch the filesystem."""
    resolved = _require_job_id(job_id)
    _invoke(callbacks.clear_log, resolved)
    next_generation = _next_log_generation(generation)
    return (
        0,
        next_generation,
        _log_delta_payload("", True, next_generation),
        "Log cleared.",
    )


def handle_poll(job_id: Any, *, callbacks: TrainingCallbacks) -> JobPanelData:
    """Refresh operational state and previews. Does not rewrite the editor."""
    resolved = _job_id_text(job_id)
    if not resolved:
        return JobPanelData()
    payload = _invoke(callbacks.poll_state, resolved)
    data = _job_payload(payload, fallback_id=resolved)
    return JobPanelData(
        job_id=data.job_id or resolved,
        config_text=None,
        state=data.state,
        previews=data.previews if data.previews is not None else (),
        message=data.message,
    )


def handle_poll_log(
    job_id: Any,
    offset: Any,
    *,
    callbacks: TrainingCallbacks,
) -> dict[str, Any]:
    """Read a log delta. Never raises; failures yield an empty non-reset chunk."""
    offset_i = _as_offset(offset)
    resolved = _job_id_text(job_id)
    if not resolved:
        return {"chunk": "", "next_offset": offset_i, "reset": False}
    try:
        payload = callbacks.poll_log(resolved, offset_i)
    except Exception:
        return {"chunk": "", "next_offset": offset_i, "reset": False}
    if not isinstance(payload, Mapping):
        return {"chunk": "", "next_offset": offset_i, "reset": False}
    chunk = payload.get("chunk")
    chunk_text = "" if chunk is None else str(chunk)
    try:
        next_offset = int(payload.get("next_offset", offset_i))
    except (TypeError, ValueError):
        next_offset = offset_i
    return {
        "chunk": chunk_text,
        "next_offset": next_offset,
        "reset": bool(payload.get("reset")),
    }


def commit_training_log(
    live_job_id: Any,
    polled_job_id: Any,
    chunk: Any,
    next_offset: Any,
    reset: Any,
    live_generation: Any,
    poll_generation: Any,
):
    """CAS: commit log offset/delta only when the live job and generation match."""
    if _job_id_text(live_job_id) != _job_id_text(polled_job_id):
        return gr.skip(), gr.skip()
    if not _job_id_text(polled_job_id):
        return gr.skip(), gr.skip()
    if _as_generation(live_generation) != _as_generation(poll_generation):
        return gr.skip(), gr.skip()
    text = "" if chunk is None else str(chunk)
    did_reset = bool(reset)
    offset_out = _as_offset(next_offset)
    if not text and not did_reset:
        return offset_out, gr.skip()
    return offset_out, _log_delta_payload(
        text, did_reset, _as_generation(live_generation)
    )


def build_training_panel(
    *,
    callbacks: TrainingCallbacks | TrainingCallbackAPI | None = None,
    start_btn: gr.Button,
    stop_btn: gr.Button,
    clear_btn: gr.Button,
) -> TrainingPanel:
    """Build the Training tab body. Must be called inside a ``gr.Blocks`` tree.

    ``start_btn``, ``stop_btn``, and ``clear_btn`` are injected (navbar-owned).
    This panel does not construct them or wrap them in a Row/Column/Tab.
    """
    resolved = as_training_callbacks(callbacks)
    job_choices = list(_invoke_list_jobs(resolved))

    with gr.Column(elem_id="studio-training-panel") as root:
        status_state = gr.State("")
        job_id = gr.State("")
        log_offset = gr.State(-1)
        log_generation = gr.State(0)
        polled_job_id = gr.State("")
        pending_chunk = gr.State("")
        pending_next_offset = gr.State(0)
        pending_reset = gr.State(False)
        pending_generation = gr.State(0)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row(elem_id="studio-training-job"):
                    job_selector = gr.Dropdown(
                        choices=job_choices,
                        value=None,
                        label="Job",
                        info=(
                            "Select an existing job, or type a new name and click "
                            "Create. Existing jobs open without rewriting files."
                        ),
                        allow_custom_value=True,
                        elem_id="studio-training-job-selector",
                    )
                    create_open_btn = gr.Button(
                        "Create",
                        variant="primary",
                        elem_id="studio-training-create-open",
                    )
                with gr.Accordion(
                    "config.yaml",
                    open=False,
                    elem_id="studio-training-yaml-accordion",
                ) as yaml_accordion:
                    yaml_editor = gr.Textbox(
                        label="config.yaml",
                        show_label=False,
                        value="",
                        lines=22,
                        max_lines=48,
                        placeholder="Canonical jobs/{id}/config.yaml",
                        info=(
                            "Raw YAML for jobs/{id}/config.yaml. Sampling keys use "
                            "Diffusers names (guidance_scale, num_inference_steps, …)."
                        ),
                        elem_id="studio-training-yaml",
                        elem_classes=["studio-training-yaml"],
                    )
                save_btn = gr.Button(
                    "Save",
                    elem_id="studio-training-save",
                    size="sm",
                )
            with gr.Column(scale=6):
                operational_state = gr.Markdown(
                    format_operational_state({}),
                    elem_id="studio-training-state",
                )
                message = gr.Markdown("", elem_id="studio-training-message")
                preview_gallery = gr.Gallery(
                    label="Previews",
                    columns=2,
                    height=360,
                    object_fit="contain",
                    preview=True,
                    format="png",
                    elem_id="studio-training-previews",
                )
                log_delta = gr.Textbox(
                    value="",
                    label="log delta",
                    show_label=False,
                    visible=False,
                    interactive=False,
                    elem_id="studio-training-log-delta",
                )
        with gr.Accordion(
            "Log",
            open=True,
            elem_id="studio-training-log-accordion",
        ) as log_accordion:
            job_log = gr.HTML(
                JOB_LOG_HTML,
                elem_id="studio-training-job-log",
            )
        poll_timer = gr.Timer(2.0, active=False)

    load_outputs = [
        job_id,
        yaml_editor,
        job_selector,
        operational_state,
        preview_gallery,
        status_state,
        message,
        poll_timer,
        log_offset,
        log_generation,
        log_delta,
        start_btn,
        stop_btn,
    ]

    def on_create_or_open(name, generation=0):
        data = handle_create_or_open(name, callbacks=resolved)
        return _load_outputs(data, log_generation=_next_log_generation(generation))

    def on_select_job(selected, generation=0):
        text = _job_id_text(selected)
        if not text:
            data = handle_load_job("", callbacks=resolved)
            return _load_outputs(
                data,
                update_selector=False,
                log_generation=_next_log_generation(generation),
            )
        known = set(_invoke_list_jobs(resolved))
        if text not in known:
            return _skip_load_outputs()
        data = handle_load_job(text, callbacks=resolved)
        return _load_outputs(
            data,
            update_selector=False,
            log_generation=_next_log_generation(generation),
        )

    def on_save(current_id, text, status):
        data = handle_save(current_id, text, status, callbacks=resolved)
        return _save_outputs(data)

    def on_start(current_id):
        data = handle_start(current_id, callbacks=resolved)
        return _action_outputs(data)

    def on_stop(current_id):
        data = handle_stop(current_id, callbacks=resolved)
        return _action_outputs(data)

    def on_clear(current_id, generation=0):
        return handle_clear_log(current_id, generation, callbacks=resolved)

    def on_poll(current_id, offset=-1, log_generation=0):
        data = handle_poll(current_id, callbacks=resolved)
        log = handle_poll_log(current_id, offset, callbacks=resolved)
        if not data.job_id:
            return (gr.skip(),) * 10
        start_vis, stop_vis = _run_button_visibility(data.state)
        return (
            format_operational_state(data.state),
            list(data.previews or ()),
            _status_value(
                data.state.get("status") if isinstance(data.state, Mapping) else ""
            ),
            data.job_id,
            log["chunk"],
            log["next_offset"],
            log["reset"],
            _as_generation(log_generation),
            start_vis,
            stop_vis,
        )

    create_event = create_open_btn.click(
        on_create_or_open,
        inputs=[job_selector, log_generation],
        outputs=load_outputs,
        show_progress="minimal",
    )
    create_event.then(None, inputs=[log_delta], js=APPLY_TRAINING_LOG_JS)
    select_event = job_selector.change(
        on_select_job,
        inputs=[job_selector, log_generation],
        outputs=load_outputs,
        show_progress="minimal",
    )
    select_event.then(None, inputs=[log_delta], js=APPLY_TRAINING_LOG_JS)
    save_btn.click(
        on_save,
        inputs=[job_id, yaml_editor, status_state],
        outputs=[
            yaml_editor,
            operational_state,
            preview_gallery,
            status_state,
            message,
        ],
        show_progress="minimal",
    )
    start_btn.click(
        on_start,
        inputs=[job_id],
        outputs=[
            operational_state,
            preview_gallery,
            status_state,
            message,
            start_btn,
            stop_btn,
        ],
        show_progress="minimal",
    )
    stop_btn.click(
        on_stop,
        inputs=[job_id],
        outputs=[
            operational_state,
            preview_gallery,
            status_state,
            message,
            start_btn,
            stop_btn,
        ],
        show_progress="minimal",
    )
    clear_event = clear_btn.click(
        on_clear,
        inputs=[job_id, log_generation],
        outputs=[log_offset, log_generation, log_delta, message],
        show_progress="minimal",
    )
    clear_event.then(None, inputs=[log_delta], js=APPLY_TRAINING_LOG_JS)
    poll_event = poll_timer.tick(
        on_poll,
        inputs=[job_id, log_offset, log_generation],
        outputs=[
            operational_state,
            preview_gallery,
            status_state,
            polled_job_id,
            pending_chunk,
            pending_next_offset,
            pending_reset,
            pending_generation,
            start_btn,
            stop_btn,
        ],
    )
    poll_event.then(
        commit_training_log,
        inputs=[
            job_id,
            polled_job_id,
            pending_chunk,
            pending_next_offset,
            pending_reset,
            log_generation,
            pending_generation,
        ],
        outputs=[log_offset, log_delta],
    ).then(
        None,
        inputs=[log_delta],
        js=APPLY_TRAINING_LOG_JS,
    )

    return TrainingPanel(
        root=root,
        create_open_btn=create_open_btn,
        job_id=job_id,
        job_selector=job_selector,
        yaml_accordion=yaml_accordion,
        yaml_editor=yaml_editor,
        save_btn=save_btn,
        start_btn=start_btn,
        stop_btn=stop_btn,
        clear_btn=clear_btn,
        operational_state=operational_state,
        preview_gallery=preview_gallery,
        message=message,
        status_state=status_state,
        poll_timer=poll_timer,
        log_offset=log_offset,
        log_generation=log_generation,
        log_delta=log_delta,
        log_accordion=log_accordion,
        job_log=job_log,
    )


def _load_outputs(
    data: JobPanelData,
    *,
    update_selector: bool = True,
    log_generation: int = 0,
):
    job_id = data.job_id or ""
    if update_selector:
        jobs = list(data.jobs or ())
        if job_id and job_id not in jobs:
            jobs.append(job_id)
        selector = gr.update(
            choices=jobs,
            value=job_id or None,
            allow_custom_value=True,
        )
    else:
        selector = gr.update(value=job_id or None, allow_custom_value=True)
    generation = _as_generation(log_generation)
    return (
        job_id,
        data.config_text if data.config_text is not None else "",
        selector,
        format_operational_state(data.state),
        list(data.previews or ()),
        _status_of(data.state),
        data.message,
        gr.update(active=bool(job_id)),
        -1,
        generation,
        _log_delta_payload("", True, generation),
        *_run_button_visibility(data.state),
    )


def _skip_load_outputs():
    """Leave panel outputs unchanged (e.g. typed custom name not yet Create'd)."""
    return (gr.skip(),) * 13


def _save_outputs(data: JobPanelData):
    yaml_out = data.config_text if data.config_text is not None else gr.skip()
    has_state = bool(data.state)
    state_md = format_operational_state(data.state) if has_state else gr.skip()
    previews = list(data.previews) if data.previews is not None else gr.skip()
    status = _status_of(data.state) if has_state else gr.skip()
    return yaml_out, state_md, previews, status, data.message


def _action_outputs(data: JobPanelData):
    previews = list(data.previews) if data.previews is not None else gr.skip()
    start_vis, stop_vis = _run_button_visibility(data.state)
    return (
        format_operational_state(data.state),
        previews,
        _status_of(data.state),
        data.message,
        start_vis,
        stop_vis,
    )


def _run_button_visibility(state: Mapping[str, Any] | None):
    running = is_active_status(_status_of(state))
    return gr.update(visible=not running), gr.update(visible=running)


def _job_payload(payload: Any, *, fallback_id: str) -> JobPanelData:
    if payload is None:
        return JobPanelData(job_id=fallback_id, config_text="", state={}, previews=())
    if not isinstance(payload, Mapping):
        state = coerce_job_state(payload)
        return JobPanelData(
            job_id=state.get("job_id") or fallback_id,
            config_text=None,
            state=state,
            previews=None,
        )
    has_state = "state" in payload or "status" in payload
    state = coerce_job_state(payload) if has_state else {}
    job_id = _text(payload.get("job_id")) or state.get("job_id") or fallback_id
    if has_state and job_id and not state.get("job_id"):
        state = {**state, "job_id": job_id}
    return JobPanelData(
        job_id=job_id,
        config_text=_config_text(payload),
        state=state,
        previews=_as_previews(payload),
        message=_text(payload.get("message")),
        mode=_save_mode(payload, None),
    )


def _config_text(payload: Mapping[str, Any]) -> str | None:
    for key in ("config_text", "yaml_text", "yaml"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _as_previews(payload: Mapping[str, Any]) -> tuple[str, ...] | None:
    if "previews" not in payload:
        return None
    raw = payload.get("previews") or ()
    paths: list[str] = []
    for item in raw:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, Mapping):
            path = item.get("path") or item.get("name")
            if path:
                paths.append(str(path))
        else:
            paths.append(str(item))
    return tuple(paths)


def _validation_result(payload: Any) -> tuple[bool, str]:
    if payload is None:
        return True, ""
    if isinstance(payload, Mapping):
        error = _text(payload.get("error") or payload.get("message"))
        if "ok" in payload:
            return bool(payload["ok"]), error
        if "valid" in payload:
            return bool(payload["valid"]), error
        if error and payload.get("ok") is not True:
            return False, error
        return True, error
    return True, ""


def _save_mode(payload: Any, default: str | None) -> str | None:
    if isinstance(payload, Mapping):
        mode = payload.get("mode")
        if mode in {SAVE_MODE_SAVED, SAVE_MODE_QUEUED}:
            return str(mode)
    return default


def _invoke_list_jobs(
    callbacks: TrainingCallbacks,
    extra: str = "",
) -> tuple[str, ...]:
    try:
        raw = callbacks.list_jobs()
    except Exception:
        raw = []
    jobs: list[str] = []
    for item in raw or ():
        text = _job_id_text(item)
        if text and text not in jobs:
            jobs.append(text)
    extra_id = _job_id_text(extra)
    if extra_id and extra_id not in jobs:
        jobs.append(extra_id)
    return tuple(jobs)


def _invoke(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _required_base_name(name: Any) -> str:
    if name is None or not str(name).strip():
        raise gr.Error("Enter a job name.")
    return str(name)


def _require_job_id(job_id: Any) -> str:
    resolved = _job_id_text(job_id)
    if not resolved:
        raise gr.Error("Open or select a training job first.")
    return resolved


def _job_id_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


def _yaml_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _status_value(status: Any) -> str:
    if status is None:
        return ""
    if isinstance(status, Mapping):
        status = status.get("status", "")
    value = getattr(status, "value", status)
    return str(value).strip().lower() if value is not None else ""


def _status_of(state: Mapping[str, Any] | None) -> str:
    if not state:
        return ""
    return _status_value(state.get("status") if isinstance(state, Mapping) else state)


def _as_offset(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _as_generation(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _next_log_generation(value: Any) -> int:
    return _as_generation(value) + 1


def _log_delta_payload(chunk: str, reset: bool, generation: int) -> str:
    return json.dumps(
        {"chunk": chunk, "reset": bool(reset), "generation": int(generation)},
        ensure_ascii=False,
    )


__all__ = [
    "ACTIVE_JOB_STATUS",
    "APPLY_TRAINING_LOG_JS",
    "JOB_LOG_HTML",
    "JobPanelData",
    "SAVE_MODE_QUEUED",
    "SAVE_MODE_SAVED",
    "TrainingCallbackAPI",
    "TrainingCallbacks",
    "TrainingPanel",
    "as_training_callbacks",
    "build_training_panel",
    "coerce_job_state",
    "commit_training_log",
    "empty_job_state",
    "format_operational_state",
    "handle_clear_log",
    "handle_create_or_open",
    "handle_load_job",
    "handle_poll",
    "handle_poll_log",
    "handle_save",
    "handle_start",
    "handle_stop",
    "handle_validate",
    "is_active_status",
    "noop_training_callbacks",
]
