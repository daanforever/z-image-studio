"""Command-line contract for training jobs.

This module does not launch Gradio and does not provide a runtime lock.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from zimage.config import ROOT
from zimage.training.commands import enqueue_update, save_idle_update
from zimage.training.contracts import JobStatus, RuntimeGuard
from zimage.training.jobs import (
    Backend,
    JobController,
    create_or_open_job,
    load_job_config,
    load_job_state,
    resolve_job_path,
)
from zimage.training.schema import (
    TrainingConfigError,
    resolve_training_paths,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
TRAIN_SCRIPT = ROOT / "train.py"
TRAIN_CWD = ROOT
TRAIN_ENVIRONMENT = {"PYTHONUNBUFFERED": "1"}
RUNTIME_GUARD_FACTORY = "zimage.training.runtime_guard:create_runtime_guard"
CACHE_BACKEND_ENTRYPOINT = "zimage.training.loop:cache_job"
RUN_BACKEND_ENTRYPOINT = "zimage.training.loop:run_job"


@dataclass(frozen=True)
class TrainingProcessSpec:
    """Stable subprocess inputs for later UI process management."""

    executable: str
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    success_exit_code: int = EXIT_OK
    error_exit_code: int = EXIT_ERROR
    usage_exit_code: int = EXIT_USAGE


def build_process_spec(*argv: str) -> TrainingProcessSpec:
    return TrainingProcessSpec(
        executable=sys.executable,
        argv=("-u", str(TRAIN_SCRIPT), *argv),
        cwd=TRAIN_CWD,
        environment=dict(TRAIN_ENVIRONMENT),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Z-Image training job CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create or open a job")
    create.add_argument("name")

    for command in ("validate", "cache", "run", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("job_id")

    update = subparsers.add_parser("update")
    update.add_argument("job_id")
    update.add_argument("config", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_guard: RuntimeGuard | None = None,
    cache_backend: Backend | None = None,
    run_backend: Backend | None = None,
    jobs_dir: str | Path | None = None,
) -> int:
    args = parse_args(argv)
    try:
        resolved_jobs_dir = _resolve_jobs_dir(jobs_dir)
        if args.command == "create":
            job_dir = create_or_open_job(args.name, resolved_jobs_dir)
            print(job_dir.name)
            return EXIT_OK

        job_dir = resolve_job_path(resolved_jobs_dir, args.job_id)
        if not job_dir.is_dir():
            raise FileNotFoundError(f"job does not exist: {args.job_id}")

        if args.command == "validate":
            load_job_config(job_dir)
            print(f"{args.job_id}: valid")
        elif args.command == "status":
            state = load_job_state(job_dir)
            payload = {
                "job_id": state.job_id,
                "status": state.status.value,
                "step": state.step,
                "epoch": state.epoch,
                "last_error": state.last_error,
                "exit_code": state.exit_code,
            }
            print(json.dumps(payload, sort_keys=True))
        elif args.command == "update":
            document = _load_update(args.config)
            state = load_job_state(job_dir)
            if state.status is JobStatus.RUNNING:
                envelope = enqueue_update(job_dir, document)
                print(envelope.command_id)
            else:
                save_idle_update(job_dir, document)
                print(f"{args.job_id}: saved")
        elif args.command in {"cache", "run"}:
            if runtime_guard is None:
                runtime_guard_factory = _load_entrypoint(RUNTIME_GUARD_FACTORY)
                runtime_guard = runtime_guard_factory()
            if args.command == "cache" and cache_backend is None:
                cache_backend = _load_entrypoint(CACHE_BACKEND_ENTRYPOINT)
            if args.command == "run" and run_backend is None:
                run_backend = _load_entrypoint(RUN_BACKEND_ENTRYPOINT)
            controller = JobController(
                runtime_guard,
                cache_backend=cache_backend,
                run_backend=run_backend,
            )
            exit_code = (
                controller.cache(job_dir)
                if args.command == "cache"
                else controller.run(job_dir)
            )
            if exit_code != EXIT_OK:
                return EXIT_ERROR
        return EXIT_OK
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TrainingConfigError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR


def _resolve_jobs_dir(override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    configured = Path(resolve_training_paths().jobs_dir)
    if configured.is_absolute():
        return configured
    return (TRAIN_CWD / configured).resolve()


def _load_update(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainingConfigError(f"cannot read update YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise TrainingConfigError("update YAML must contain a mapping")
    return raw


def _load_entrypoint(contract: str) -> Callable[..., Any]:
    module_name, separator, attribute = contract.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"invalid training entrypoint contract: {contract}")
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"training entrypoint is not available: {contract}"
        ) from exc
    if not callable(value):
        raise RuntimeError(f"training entrypoint is not callable: {contract}")
    return value
