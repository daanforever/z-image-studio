from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from zimage.training.cli import (
    CACHE_BACKEND_ENTRYPOINT,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    RUNTIME_GUARD_FACTORY,
    RUN_BACKEND_ENTRYPOINT,
    TRAIN_CWD,
    TRAIN_SCRIPT,
    build_process_spec,
    main,
    parse_args,
)
from zimage.training.commands import consume_commands
from zimage.training.contracts import JobState, JobStatus
from zimage.training.jobs import (
    create_or_open_job,
    load_job_config,
    write_job_state,
)


def test_process_contract_pins_executable_argv_cwd_environment_and_exit_codes():
    spec = build_process_spec("run", "my-job")

    assert spec.executable == sys.executable
    assert spec.argv == ("-u", str(TRAIN_SCRIPT), "run", "my-job")
    assert spec.cwd == TRAIN_CWD
    assert spec.environment == {"PYTHONUNBUFFERED": "1"}
    assert (
        spec.success_exit_code,
        spec.error_exit_code,
        spec.usage_exit_code,
    ) == (EXIT_OK, EXIT_ERROR, EXIT_USAGE)


@pytest.mark.parametrize("command", ["create", "validate", "cache", "run", "update", "status"])
def test_parser_supports_frozen_subcommands(command):
    argv = [command, "name"]
    if command == "update":
        argv.append("config.yaml")
    assert parse_args(argv).command == command


def test_root_entrypoint_runs_from_project_cwd():
    spec = build_process_spec("--help")
    environment = {**os.environ, **spec.environment}

    result = subprocess.run(
        [spec.executable, *spec.argv],
        cwd=spec.cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == EXIT_OK
    assert "Z-Image training job CLI" in result.stdout


def test_public_cli_has_no_jobs_directory_override():
    with pytest.raises(SystemExit) as error:
        parse_args(["--jobs-dir", "other", "status", "job"])
    assert error.value.code == EXIT_USAGE


def test_default_jobs_directory_comes_from_root_config(monkeypatch, tmp_path):
    jobs = tmp_path / "configured-jobs"

    class Paths:
        jobs_dir = str(jobs)

    monkeypatch.setattr("zimage.training.cli.resolve_training_paths", Paths)

    assert main(["create", "Configured Job"]) == EXIT_OK
    assert (jobs / "configured-job" / "config.yaml").is_file()


def test_cli_create_open_validate_and_status(tmp_path, capsys):
    jobs = tmp_path / "jobs"

    assert main(["create", "Named Job"], jobs_dir=jobs) == EXIT_OK
    first_output = capsys.readouterr().out
    config = jobs / "named-job" / "config.yaml"
    before = config.read_bytes()
    assert main(["create", "named-job"], jobs_dir=jobs) == EXIT_OK
    assert config.read_bytes() == before
    assert main(["validate", "named-job"], jobs_dir=jobs) == EXIT_OK
    assert "valid" in capsys.readouterr().out
    assert main(["status", "named-job"], jobs_dir=jobs) == EXIT_OK
    status = json.loads(capsys.readouterr().out)

    assert first_output.strip() == "named-job"
    assert status["status"] == "stopped"
    assert set(status) == {
        "job_id",
        "status",
        "step",
        "epoch",
        "last_error",
        "exit_code",
    }


def test_cli_idle_update_saves_directly_without_command(tmp_path, capsys):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    update = load_job_config(root)
    update["seed"] = 5
    path = tmp_path / "update.yaml"
    path.write_text(yaml.safe_dump(update), encoding="utf-8")

    result = main(["update", "job", str(path)], jobs_dir=jobs)

    assert result == EXIT_OK
    assert load_job_config(root)["seed"] == 5
    assert not list((root / "commands").glob("*.json"))
    assert "saved" in capsys.readouterr().out


def test_cli_active_update_queues_candidate_until_trainer_consumes(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    original = load_job_config(root)
    update = dict(original)
    update["seed"] = 9
    path = tmp_path / "update.yaml"
    path.write_text(yaml.safe_dump(update), encoding="utf-8")
    write_job_state(root, JobState("job", JobStatus.RUNNING, 0, 0))

    assert main(["update", "job", str(path)], jobs_dir=jobs) == EXIT_OK
    assert load_job_config(root)["seed"] == original["seed"]
    assert len(list((root / "commands").glob("*.json"))) == 1

    consume_commands(root)
    assert load_job_config(root)["seed"] == 9


def test_cli_idle_update_discards_stale_pending_commands(tmp_path, capsys):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    stale = load_job_config(root)
    stale["seed"] = 11
    stale_path = tmp_path / "stale.yaml"
    stale_path.write_text(yaml.safe_dump(stale), encoding="utf-8")
    write_job_state(root, JobState("job", JobStatus.RUNNING, 0, 0))
    assert main(["update", "job", str(stale_path)], jobs_dir=jobs) == EXIT_OK
    assert len(list((root / "commands").glob("*.json"))) == 1

    write_job_state(root, JobState("job", JobStatus.COMPLETED, 0, 0, exit_code=0))
    idle = load_job_config(root)
    idle["seed"] = 22
    idle_path = tmp_path / "idle.yaml"
    idle_path.write_text(yaml.safe_dump(idle), encoding="utf-8")
    assert main(["update", "job", str(idle_path)], jobs_dir=jobs) == EXIT_OK
    assert load_job_config(root)["seed"] == 22
    assert not list((root / "commands").glob("*.json"))
    assert consume_commands(root) == []
    assert load_job_config(root)["seed"] == 22
    assert "saved" in capsys.readouterr().out


def test_cli_invalid_idle_update_preserves_pending_commands(tmp_path, capsys):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    original_seed = load_job_config(root)["seed"]
    stale = load_job_config(root)
    stale["seed"] = 11
    stale_path = tmp_path / "stale.yaml"
    stale_path.write_text(yaml.safe_dump(stale), encoding="utf-8")
    write_job_state(root, JobState("job", JobStatus.RUNNING, 0, 0))
    assert main(["update", "job", str(stale_path)], jobs_dir=jobs) == EXIT_OK

    write_job_state(root, JobState("job", JobStatus.COMPLETED, 0, 0, exit_code=0))
    invalid = load_job_config(root)
    invalid["precision"] = "invalid"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    assert main(["update", "job", str(invalid_path)], jobs_dir=jobs) == EXIT_ERROR
    assert load_job_config(root)["seed"] == original_seed
    assert len(list((root / "commands").glob("*.json"))) == 1
    assert capsys.readouterr().err


def test_invalid_active_update_is_written_then_quarantined_by_trainer(tmp_path):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    invalid = load_job_config(root)
    invalid["precision"] = "invalid"
    path = tmp_path / "update.yaml"
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    write_job_state(root, JobState("job", JobStatus.RUNNING, 0, 0))

    assert main(["update", "job", str(path)], jobs_dir=jobs) == EXIT_OK
    assert consume_commands(root) == []
    assert load_job_config(root)["precision"] == "fp8"
    assert list((root / "commands" / "quarantine").glob("*.invalid"))


class Guard:
    def __init__(self):
        self.calls = []

    def acquire(self):
        self.calls.append("acquire")
        return True

    def release(self):
        self.calls.append("release")

    def is_held(self):
        return bool(self.calls)


@pytest.mark.parametrize("command", ["cache", "run"])
def test_cli_cache_and_run_use_explicit_injections(tmp_path, command):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    guard = Guard()
    calls = []
    kwargs = {
        "runtime_guard": guard,
        f"{command}_backend": lambda path: calls.append(path) or 0,
        "jobs_dir": jobs,
    }

    result = main([command, "job"], **kwargs)

    assert result == EXIT_OK
    assert calls == [root]
    assert guard.calls == ["acquire", "release"]
    if command == "run":
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == JobStatus.COMPLETED.value


def test_cli_runtime_failures_return_error_exit_code(tmp_path, capsys, monkeypatch):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)

    def load(contract):
        raise RuntimeError(f"training entrypoint is not available: {contract}")

    monkeypatch.setattr("zimage.training.cli._load_entrypoint", load)

    assert main(["run", "job"], jobs_dir=jobs) == EXIT_ERROR
    assert RUNTIME_GUARD_FACTORY in capsys.readouterr().err


def test_cli_run_empty_dataset_returns_error_exit_code(tmp_path, capsys):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)

    assert main(["run", "job"], jobs_dir=jobs) == EXIT_ERROR
    assert "job has no training samples" in capsys.readouterr().err


def test_cli_normalizes_backend_failure_to_error_exit_code(tmp_path):
    jobs = tmp_path / "jobs"
    create_or_open_job("job", jobs)
    guard = Guard()

    result = main(
        ["run", "job"],
        runtime_guard=guard,
        run_backend=lambda _: 17,
        jobs_dir=jobs,
    )

    assert result == EXIT_ERROR


def test_lazy_runtime_import_contracts_are_pinned(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    root = create_or_open_job("job", jobs)
    guard = Guard()
    calls = []

    def load(contract):
        calls.append(contract)
        if contract == RUNTIME_GUARD_FACTORY:
            return lambda: guard
        if contract == RUN_BACKEND_ENTRYPOINT:
            return lambda path: 0 if path == root else 1
        pytest.fail(f"unexpected contract: {contract}")

    monkeypatch.setattr("zimage.training.cli._load_entrypoint", load)

    assert main(["run", "job"], jobs_dir=jobs) == EXIT_OK
    assert calls == [RUNTIME_GUARD_FACTORY, RUN_BACKEND_ENTRYPOINT]
    assert RUNTIME_GUARD_FACTORY == (
        "zimage.training.runtime_guard:create_runtime_guard"
    )
    assert CACHE_BACKEND_ENTRYPOINT == "zimage.training.loop:cache_job"
    assert RUN_BACKEND_ENTRYPOINT == "zimage.training.loop:run_job"


def test_argparse_usage_errors_exit_with_code_two():
    with pytest.raises(SystemExit) as error:
        parse_args(["unknown"])
    assert error.value.code == EXIT_USAGE
