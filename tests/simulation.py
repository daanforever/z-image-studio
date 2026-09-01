"""Production-parity training run with a GPU-usage aggregate summary.

Zero arguments (`python tests/simulation.py`) load
``tests/simulation/config.yaml``, open the job under configured
``jobs_dir``, and call ``JobController.run`` — the same entry as
``train.py run``. Probe settings come from YAML only. Stdout ends with
an aggregate of this run's ``logs/job.log`` gpu-usage lines.

Not a pytest module: filename does not match ``test_*.py`` / ``*_test.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = Path(__file__).parent / "simulation" / "config.yaml"
_GPU_USAGE_MARKER = "gpu usage phase="
_GPU_USAGE_PREFIX = "gpu usage "
_TOKEN_RE = re.compile(r"(\S+)=(\S+)")
_BYTES_RE = re.compile(r"^(\d+(?:\.\d+)?)(B|KB|MB|GB)$", re.IGNORECASE)
_BYTE_FACTORS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the simulation job with production-parity defaults "
            f"({CONFIG_PATH.as_posix()}). Prints an aggregate GPU-usage "
            "summary from this run's logs/job.log. Probe settings come "
            "from YAML only."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="job YAML (default: tests/simulation/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=("subprocess", "in-process"),
        default="subprocess",
        help=(
            "subprocess: JobController.run (default, same as train.py run); "
            "in-process: direct run_job (dev only)"
        ),
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="job directory; default creates/opens the job in configured jobs_dir",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="wipe dataset .cache/ before run (default: keep warm cache)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override job max_steps (default: YAML value)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help="override datasets directory (default: root config.yaml)",
    )
    return parser.parse_args(argv)


def parse_formatted_bytes(text: str) -> int | None:
    """Parse ``format_bytes`` tokens such as ``128B`` or ``12.1GB``."""

    match = _BYTES_RE.fullmatch(str(text).strip())
    if match is None:
        return None
    unit = match.group(2).upper()
    return int(float(match.group(1)) * _BYTE_FACTORS[unit])


def gpu_usage_payload(line: str) -> str | None:
    """Return the ``gpu usage ...`` payload, or ``None`` if the line has none."""

    index = line.find(_GPU_USAGE_PREFIX)
    if index < 0:
        return None
    return line[index:].rstrip()


def parse_gpu_usage_line(line: str) -> dict[str, str] | None:
    """Parse compact or summary ``gpu usage phase=`` fields from one log line."""

    if _GPU_USAGE_MARKER not in line:
        return None
    payload = gpu_usage_payload(line)
    if payload is None:
        return None
    fields = dict(_TOKEN_RE.findall(payload[len(_GPU_USAGE_PREFIX) :]))
    if "phase" not in fields:
        return None
    return fields


def parse_gpu_usage_records(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        parsed = parse_gpu_usage_line(line)
        if parsed is not None:
            records.append(parsed)
    return records


def aggregate_gpu_usage(
    records: Sequence[Mapping[str, str]],
    *,
    raw_lines: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Max ``phase_peak`` by phase, max ``nvidia_used``, and summary payload."""

    phase_peaks: dict[str, str] = {}
    phase_peak_bytes: dict[str, int] = {}
    max_nvidia_used = "0B"
    max_nvidia_bytes = 0
    summary_line: str | None = None
    payloads = list(raw_lines) if raw_lines is not None else []
    for index, record in enumerate(records):
        phase = str(record["phase"])
        peak_token = record.get("phase_peak")
        peak_bytes = parse_formatted_bytes(peak_token) if peak_token else None
        if peak_bytes is not None and peak_bytes >= phase_peak_bytes.get(phase, -1):
            phase_peak_bytes[phase] = peak_bytes
            phase_peaks[phase] = peak_token or "0B"
        nvidia_token = record.get("nvidia_used")
        nvidia_bytes = parse_formatted_bytes(nvidia_token) if nvidia_token else None
        if nvidia_bytes is not None and nvidia_bytes >= max_nvidia_bytes:
            max_nvidia_bytes = nvidia_bytes
            max_nvidia_used = nvidia_token or "0B"
        if phase == "summary" and "max_step_peak" in record:
            if index < len(payloads):
                summary_line = payloads[index]
            else:
                summary_line = (
                    f"{_GPU_USAGE_PREFIX}phase=summary "
                    f"max_step_peak={record.get('max_step_peak', '0B')} "
                    f"max_preview_peak={record.get('max_preview_peak', '0B')} "
                    f"max_nvidia_used={record.get('max_nvidia_used', '0B')}"
                )
    return {
        "phase_peaks": phase_peaks,
        "max_nvidia_used": max_nvidia_used,
        "summary_line": summary_line,
    }


def format_aggregate_summary(
    aggregate: Mapping[str, Any],
    log_path: Path,
) -> str:
    lines = ["max phase_peak by phase:"]
    peaks = aggregate.get("phase_peaks") or {}
    if peaks:
        for phase, value in peaks.items():
            lines.append(f"  {phase}={value}")
    else:
        lines.append("  (none)")
    lines.append(f"max nvidia_used={aggregate.get('max_nvidia_used', '0B')}")
    summary = aggregate.get("summary_line")
    if summary:
        lines.append(str(summary))
    lines.append(f"job.log: {log_path}")
    return "\n".join(lines)


def summarize_gpu_usage_log(text: str, log_path: Path) -> str:
    payloads: list[str] = []
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        parsed = parse_gpu_usage_line(line)
        if parsed is None:
            continue
        payload = gpu_usage_payload(line)
        records.append(parsed)
        payloads.append(payload or line.rstrip())
    aggregate = aggregate_gpu_usage(records, raw_lines=payloads)
    return format_aggregate_summary(aggregate, log_path)


def read_log_since(path: Path, start_offset: int) -> str:
    """Read UTF-8 text from ``start_offset`` bytes (this run's append)."""

    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        offset = 0 if start_offset < 0 or start_offset > size else start_offset
        handle.seek(offset)
        return handle.read().decode("utf-8", errors="replace")


def _fail(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError:
        _fail("CUDA is not available")
    if not torch.cuda.is_available():
        _fail("CUDA is not available")


def _resolve_configured_dir(configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _jobs_dir() -> Path:
    from zimage.training.schema import resolve_training_paths

    return _resolve_configured_dir(resolve_training_paths().jobs_dir)


def _datasets_dir(override: Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    from zimage.training.schema import resolve_training_paths

    return _resolve_configured_dir(resolve_training_paths().datasets_dir)


def _open_job(job_name: str, job_dir_arg: Path | None) -> Path:
    from zimage.training.jobs import create_or_open_job

    if job_dir_arg is None:
        return create_or_open_job(job_name, _jobs_dir())
    root = Path(job_dir_arg).expanduser().resolve()
    if root.is_dir() and (
        (root / "state.json").is_file() or (root / "config.yaml").is_file()
    ):
        return root
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    if root.is_dir():
        return create_or_open_job(job_name, root)
    return create_or_open_job(job_name, root.parent)


def _wipe_dataset_caches(document: Mapping[str, Any], datasets_dir: Path) -> None:
    for entry in document.get("datasets") or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        cache = Path(datasets_dir) / str(name) / ".cache"
        if cache.is_dir():
            shutil.rmtree(cache)


def _run(
    mode: str,
    job_dir: Path,
    injected: Mapping[str, Any],
) -> int:
    from zimage.training.loop import run_job

    payload = dict(injected)
    if mode == "in-process":
        from zimage.training.job_log import job_log_session

        with job_log_session(job_dir):
            return run_job(job_dir, **payload)

    from zimage.training.jobs import JobController
    from zimage.training.runtime_guard import create_runtime_guard

    def backend(root: Path) -> int:
        return run_job(root, **payload)

    controller = JobController(
        create_runtime_guard(),
        run_backend=run_job if not payload else backend,
    )
    return controller.run(job_dir)


def _log_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from zimage.config import load_dotenv
    from zimage.training.job_log import job_log_path
    from zimage.training.jobs import save_job_config
    from zimage.training.schema import TrainingConfigError, load_job_document

    load_dotenv()
    try:
        document = dict(load_job_document(args.config))
        if args.max_steps is not None:
            document["max_steps"] = args.max_steps
        _require_cuda()
        job_dir = _open_job(str(document["job_name"]), args.job_dir)
        save_job_config(job_dir, document)
        if args.cold_cache:
            _wipe_dataset_caches(document, _datasets_dir(args.datasets_dir))
        log_path = job_log_path(job_dir)
        start_offset = _log_offset(log_path)
        injected: dict[str, Any] = {}
        if args.datasets_dir is not None:
            injected["datasets_dir"] = args.datasets_dir
        code = _run(args.mode, job_dir, injected)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        TrainingConfigError,
        TypeError,
        ValueError,
    ) as exc:
        _fail(str(exc))

    text = read_log_since(log_path, start_offset)
    print(summarize_gpu_usage_log(text, log_path), flush=True)
    if code != 0:
        _fail(f"run exited {code}", code if code else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
