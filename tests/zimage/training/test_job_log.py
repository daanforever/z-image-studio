from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import fields
from pathlib import Path

import pytest

from zimage.training.job_log import (
    DEFAULT_READ_LIMIT,
    LOGS_DIR,
    LOG_FILE,
    SESSION_BANNER_PREFIX,
    TAIL_WINDOW_BYTES,
    TRAINING_LOGGER_NAME,
    TRUNCATED_MARKER,
    JobLogChunk,
    job_log_path,
    job_log_session,
    read_job_log_chunk,
    truncate_job_log,
)
from zimage.training.jobs import JobController, create_or_open_job


ROOT = Path(__file__).resolve().parents[3]


class RecordingGuard:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.calls = []

    def acquire(self) -> bool:
        self.calls.append("acquire")
        return self.acquired

    def release(self) -> None:
        self.calls.append("release")

    def is_held(self) -> bool:
        return self.acquired


def _write_log(tmp_path: Path, data: bytes | str) -> Path:
    job_dir = tmp_path / "job"
    path = job_dir / LOGS_DIR / LOG_FILE
    path.parent.mkdir(parents=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return job_dir


def test_chunk_return_shape():
    names = {item.name for item in fields(JobLogChunk)}
    assert names == {"chunk", "next_offset", "reset"}


def test_job_log_path_is_always_logs_job_log(tmp_path):
    job_dir = tmp_path / "job-id"
    assert job_log_path(job_dir) == job_dir / "logs" / "job.log"


def test_missing_file_negative_offset_resets_to_empty(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    result = read_job_log_chunk(job_dir, -1)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=True)


def test_missing_file_nonnegative_offset_does_not_reset(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    result = read_job_log_chunk(job_dir, 0)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=False)
    later = read_job_log_chunk(job_dir, 12)
    assert later == JobLogChunk(chunk="", next_offset=0, reset=False)


def test_empty_file_offset_zero(tmp_path):
    job_dir = _write_log(tmp_path, b"")
    result = read_job_log_chunk(job_dir, 0)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=False)


def test_empty_file_negative_offset_resets(tmp_path):
    job_dir = _write_log(tmp_path, b"")
    result = read_job_log_chunk(job_dir, -1)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=True)


def test_truncate_job_log_zeros_file_in_place(tmp_path):
    job_dir = _write_log(tmp_path, "hello log")
    path = job_log_path(job_dir)
    assert path.stat().st_size > 0
    truncate_job_log(job_dir)
    assert path.is_file()
    assert path.stat().st_size == 0
    result = read_job_log_chunk(job_dir, -1)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=True)


def test_truncate_job_log_missing_file_is_noop(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    truncate_job_log(job_dir)
    assert not (job_dir / LOGS_DIR).exists()
    assert not job_log_path(job_dir).exists()


def test_truncate_job_log_missing_logs_file_does_not_create(tmp_path):
    job_dir = tmp_path / "job"
    logs = job_dir / LOGS_DIR
    logs.mkdir(parents=True)
    truncate_job_log(job_dir)
    assert logs.is_dir()
    assert not job_log_path(job_dir).exists()
    assert list(logs.iterdir()) == []


def test_truncate_job_log_is_exported():
    from zimage.training.job_log import __all__ as exported

    assert "truncate_job_log" in exported


def test_mid_offset_read(tmp_path):
    job_dir = _write_log(tmp_path, "abcdefghij")
    result = read_job_log_chunk(job_dir, 3, limit=4)
    assert result == JobLogChunk(chunk="defg", next_offset=7, reset=False)


def test_limit_cap_takes_multiple_reads(tmp_path):
    payload = b"x" * (200 * 1024)
    job_dir = _write_log(tmp_path, payload)
    offset = 0
    parts: list[bytes] = []
    while True:
        result = read_job_log_chunk(job_dir, offset)
        assert result.reset is False
        encoded = result.chunk.encode("utf-8")
        assert len(encoded) <= DEFAULT_READ_LIMIT
        if not encoded:
            break
        parts.append(encoded)
        assert result.next_offset == offset + len(encoded)
        offset = result.next_offset
    assert b"".join(parts) == payload
    assert len(parts) > 1


def test_negative_offset_tails_window_and_marks_truncation(tmp_path):
    head = b"H" * 100
    tail = b"T" * (300 * 1024 - 100)
    payload = head + tail
    job_dir = _write_log(tmp_path, payload)
    result = read_job_log_chunk(job_dir, -1)
    start = len(payload) - TAIL_WINDOW_BYTES
    assert result.reset is True
    assert result.chunk.startswith(TRUNCATED_MARKER)
    body = result.chunk[len(TRUNCATED_MARKER) :]
    assert "H" not in body
    assert body == "T" * DEFAULT_READ_LIMIT
    assert result.next_offset == start + DEFAULT_READ_LIMIT


def test_offset_greater_than_size_matches_first_open_tail(tmp_path):
    payload = b"T" * (300 * 1024)
    job_dir = _write_log(tmp_path, payload)
    first_open = read_job_log_chunk(job_dir, -1)
    overflow = read_job_log_chunk(job_dir, len(payload) + 50)
    assert overflow == first_open
    assert overflow.reset is True


def test_negative_offset_on_small_file_has_no_truncated_marker(tmp_path):
    job_dir = _write_log(tmp_path, "hello")
    result = read_job_log_chunk(job_dir, -1)
    assert result == JobLogChunk(chunk="hello", next_offset=5, reset=True)


def test_utf8_holdback_does_not_consume_incomplete_trailing_bytes(tmp_path):
    euro = "€".encode("utf-8")
    job_dir = _write_log(tmp_path, b"abcd" + euro)
    cut = read_job_log_chunk(job_dir, 0, limit=5)
    assert cut.chunk == "abcd"
    assert cut.next_offset == 4
    assert "\ufffd" not in cut.chunk
    rest = read_job_log_chunk(job_dir, cut.next_offset, limit=8)
    assert rest.chunk == "€"
    assert rest.next_offset == 7


def test_invalid_utf8_uses_replace_and_advances_offset(tmp_path):
    job_dir = _write_log(tmp_path, b"ab\xffcd")
    result = read_job_log_chunk(job_dir, 0)
    assert result.chunk == "ab\ufffdcd"
    assert result.next_offset == 5


def test_append_then_read_returns_suffix_only(tmp_path):
    job_dir = _write_log(tmp_path, "hello")
    first = read_job_log_chunk(job_dir, 0)
    assert first.chunk == "hello"
    path = job_log_path(job_dir)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(" world")
    second = read_job_log_chunk(job_dir, first.next_offset)
    assert second.chunk == " world"
    assert second.next_offset == first.next_offset + len(" world")
    assert second.reset is False


def test_reader_ignores_top_level_job_log(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.log").write_text("top-level", encoding="utf-8")
    result = read_job_log_chunk(job_dir, 0)
    assert result == JobLogChunk(chunk="", next_offset=0, reset=False)


def test_session_creates_logs_lazily_and_writes_banner(tmp_path):
    job_dir = tmp_path / "legacy"
    job_dir.mkdir()
    assert not (job_dir / LOGS_DIR).exists()
    original_out = sys.stdout
    original_err = sys.stderr
    with job_log_session(job_dir) as path:
        assert path == job_log_path(job_dir)
        assert path.is_file()
    assert sys.stdout is original_out
    assert sys.stderr is original_err
    text = path.read_text(encoding="utf-8")
    assert text.startswith(SESSION_BANNER_PREFIX)
    assert text.count(SESSION_BANNER_PREFIX) == 1
    assert not (job_dir / "metrics").exists()


def test_session_appends_two_banners(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with job_log_session(job_dir):
        pass
    with job_log_session(job_dir):
        pass
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert text.count(SESSION_BANNER_PREFIX) == 2


def test_print_and_logger_info_appear_once_in_file(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    logger = logging.getLogger(TRAINING_LOGGER_NAME)
    with job_log_session(job_dir):
        print("hello-stdout")
        logger.info("hello-logger")
        assert not any(isinstance(item, logging.FileHandler) for item in logger.handlers)
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert text.count("hello-stdout") == 1
    assert text.count("hello-logger") == 1
    assert "INFO" in text


def _log_body_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        line
        for line in text.splitlines()
        if not line.startswith(SESSION_BANNER_PREFIX)
    ]


def test_tee_keeps_cr_on_console_and_newlines_in_file(tmp_path, capsys):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with job_log_session(job_dir):
        sys.stdout.write("ab\rcd\n")
    captured = capsys.readouterr()
    assert "ab\rcd\n" in captured.out
    raw = job_log_path(job_dir).read_bytes()
    assert b"ab\ncd" not in raw
    assert b"ab\rcd" not in raw
    assert _log_body_lines(job_log_path(job_dir)) == ["cd"]


def test_tee_column_overwrite_keeps_uncovered_suffix(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with job_log_session(job_dir):
        sys.stdout.write("hello\rhi\n")
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "hillo\n" in text
    assert "hello" not in text


def test_tee_checkpoint_shards_one_completed_line(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    payload = "Loading checkpoint shards:   0%\rLoading checkpoint shards: 100%\n"
    with job_log_session(job_dir):
        sys.stdout.write(payload)
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "\x1b" not in text
    body = _log_body_lines(job_log_path(job_dir))
    assert len(body) == 1
    assert body[0].endswith("Loading checkpoint shards: 100%")


def test_tee_checkpoint_shards_strips_ansi(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    wrapped = (
        "\x1b[32mLoading checkpoint shards:   0%\x1b[0m\r"
        "\x1b[32mLoading checkpoint shards: 100%\x1b[0m\n"
    )
    with job_log_session(job_dir):
        sys.stdout.write(wrapped)
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "\x1b" not in text
    body = _log_body_lines(job_log_path(job_dir))
    assert len(body) == 1
    assert body[0].endswith("Loading checkpoint shards: 100%")


def test_tee_nested_tqdm_sample_collapses_refreshes(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    nested = (
        "Fetching 5 files:   0%\r"
        "Fetching 5 files:  20%\r"
        "Fetching 5 files:  40%\r"
        "Fetching 5 files:  60%\r"
        "\x1b[A\x1b[K"
        "Fetching 5 files:  80%\r"
        "Fetching 5 files: 100%\n"
        "Downloading:   0%\r"
        "Downloading: 100%\n"
    )
    with job_log_session(job_dir):
        sys.stdout.write(nested)
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "\x1b" not in text
    assert text.count("Fetching") < 6
    assert text.count("Fetching") >= 1
    body = _log_body_lines(job_log_path(job_dir))
    fetching_lines = [line for line in body if "Fetching" in line]
    assert fetching_lines
    assert fetching_lines[-1].endswith("100%")


def test_tee_flush_does_not_commit_cr_updates(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    path = job_log_path(job_dir)
    with job_log_session(job_dir):
        after_banner = path.read_text(encoding="utf-8")
        for percent in range(0, 50, 5):
            sys.stdout.write(f"Loading checkpoint shards: {percent:3d}%\r")
            sys.stdout.flush()
        assert path.read_text(encoding="utf-8") == after_banner
        sys.stdout.write("Loading checkpoint shards: 100%\n")
    body = _log_body_lines(path)
    assert len(body) == 1
    assert body[0].endswith("100%")


def test_session_close_commits_trailing_cr_bar(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with job_log_session(job_dir):
        sys.stdout.write("Loading checkpoint shards:  50%\r")
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "\r" not in text
    body = _log_body_lines(job_log_path(job_dir))
    assert body == ["Loading checkpoint shards:  50%"]


def test_session_logs_traceback_before_reraise(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        with job_log_session(job_dir):
            raise RuntimeError("boom")
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in text
    assert "RuntimeError: boom" in text
    assert "job failed" in text


def test_session_writes_full_warning_without_source_snippet(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    previous = warnings.formatwarning
    message = (
        "This UserWarning is a complete sentence that must appear in full "
        "inside logs/job.log even though the warn call spans several lines."
    )

    def emit(message, category, filename, lineno, file=None, line=None):
        stream = sys.stderr if file is None else file
        stream.write(warnings.formatwarning(message, category, filename, lineno, line))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = emit
        with job_log_session(job_dir):
            warnings.warn(
                message,
                UserWarning,
            )
    assert warnings.formatwarning is previous
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert message in text
    assert not any(line.strip() == "warnings.warn(" for line in text.splitlines())


def test_second_session_does_not_duplicate_logger_lines(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    logger = logging.getLogger(TRAINING_LOGGER_NAME)
    with job_log_session(job_dir):
        logger.info("once")
    with job_log_session(job_dir):
        logger.info("twice")
    text = job_log_path(job_dir).read_text(encoding="utf-8")
    assert text.count("once") == 1
    assert text.count("twice") == 1
    assert not any(
        getattr(item, "stream", None).__class__.__name__ == "_TeeStream"
        for item in logger.handlers
    )


def test_controller_run_print_and_logger_share_one_file(tmp_path, capsys):
    root = create_or_open_job("job", tmp_path)
    logger = logging.getLogger(TRAINING_LOGGER_NAME)

    def backend(path):
        print("hello-stdout")
        logger.info("hello-logger")
        sys.stdout.write("ab\rcd\n")
        return 0

    controller = JobController(RecordingGuard(), run_backend=backend)
    assert controller.run(root) == 0
    captured = capsys.readouterr()
    assert "ab\rcd\n" in captured.out
    text = (root / "logs" / "job.log").read_text(encoding="utf-8")
    assert text.count(SESSION_BANNER_PREFIX) == 1
    assert "hello-stdout" in text
    assert "hello-logger" in text
    assert "ab\ncd" not in text
    assert "ab\rcd" not in text
    assert "\ncd\n" in text
    assert sys.stdout is not None


def test_two_controller_runs_append_banners(tmp_path):
    root = create_or_open_job("job", tmp_path)
    controller = JobController(RecordingGuard(), run_backend=lambda _: 0)
    assert controller.run(root) == 0
    assert controller.run(root) == 0
    text = (root / "logs" / "job.log").read_text(encoding="utf-8")
    assert text.count(SESSION_BANNER_PREFIX) == 2


def test_legacy_job_without_logs_dir_still_writes(tmp_path):
    root = create_or_open_job("job", tmp_path)
    logs = root / "logs"
    logs.rmdir()
    controller = JobController(RecordingGuard(), run_backend=lambda _: 0)
    assert controller.run(root) == 0
    assert (root / "logs" / "job.log").is_file()


def test_loop_logs_step_and_epoch_without_loss_item():
    source = (ROOT / "zimage" / "training" / "loop.py").read_text(encoding="utf-8")
    assert "run start job=%s step=%s epoch=%s" in source
    assert 'log.info("step=%s epoch=%s"' not in source
    assert "loss.item()" not in source
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("log.") and "loss" in stripped:
            raise AssertionError(f"log call mentions loss: {stripped}")
    optimize_start = source.index("def _optimize(")
    next_def = source.find("\ndef ", optimize_start + 1)
    optimize_source = source[optimize_start:next_def]
    assert "tqdm(" in optimize_source


def test_process_manager_source_keeps_popen_devnull():
    source = (ROOT / "zimage" / "ui" / "training_process.py").read_text(
        encoding="utf-8"
    )
    assert "stdout=subprocess.DEVNULL" in source
    assert "stderr=subprocess.DEVNULL" in source
    assert "does not redirect" in source


def test_job_log_module_stays_free_of_gradio_and_loop():
    source = (ROOT / "zimage" / "training" / "job_log.py").read_text(encoding="utf-8")
    assert "gradio" not in source
    assert "torch" not in source
    assert "zimage.training.loop" not in source
