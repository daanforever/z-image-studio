from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zimage.training.commands import (
    CommandQueue,
    consume_commands,
    enqueue_command,
    enqueue_update,
    save_idle_update,
)
from zimage.training.contracts import JobState, JobStatus
from zimage.training.jobs import create_or_open_job, load_job_config, write_job_state
from zimage.training.schema import TrainingConfigError


def test_enqueue_uses_monotonic_ids_and_atomic_json_candidates(tmp_path):
    root = create_or_open_job("job", tmp_path)

    second = enqueue_command(root, "stop", {"reason": "test"})
    third = enqueue_command(root, "stop", {})

    assert (second.command_id, third.command_id) == (1, 2)
    candidates = sorted((root / "commands").glob("*.json"))
    assert [path.name for path in candidates] == [
        "00000000000000000001.json",
        "00000000000000000002.json",
    ]
    assert json.loads(candidates[0].read_text(encoding="utf-8")) == {
        "command_id": 1,
        "kind": "stop",
        "payload": {"reason": "test"},
        "created_at": second.created_at,
    }
    assert not list((root / "commands").glob("*.tmp"))


def test_consumption_is_ordered_and_idempotent(tmp_path):
    root = create_or_open_job("job", tmp_path)
    queue = CommandQueue(root)
    queue.enqueue("third", {})
    queue.enqueue("first", {})
    queue.enqueue("second", {})
    seen = []

    consumed = queue.consume(lambda envelope: seen.append(envelope.command_id))

    assert seen == [1, 2, 3]
    assert [item.command_id for item in consumed] == [1, 2, 3]
    assert queue.consume(lambda envelope: seen.append(envelope.command_id)) == []
    assert seen == [1, 2, 3]


def test_ids_remain_monotonic_after_queue_is_consumed(tmp_path):
    root = create_or_open_job("job", tmp_path)
    queue = CommandQueue(root)
    assert queue.enqueue("noop", {}).command_id == 1
    queue.consume(lambda _: None)

    assert queue.enqueue("noop", {}).command_id == 2


def test_concurrent_processes_reserve_unique_monotonic_ids(tmp_path):
    root = create_or_open_job("job", tmp_path)
    project_root = Path(__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from zimage.training.commands import enqueue_command\n"
        "print(enqueue_command(Path(sys.argv[1]), 'stop', {}).command_id)\n"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(root)],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(12)
    ]
    results = [process.communicate(timeout=30) for process in processes]

    assert [process.returncode for process in processes] == [0] * len(processes)
    assert all(not stderr for _, stderr in results)
    ids = sorted(int(stdout.strip()) for stdout, _ in results)
    assert ids == list(range(1, len(processes) + 1))

    candidates = CommandQueue(root).pending()
    payload_ids = [
        json.loads(path.read_text(encoding="utf-8"))["command_id"]
        for path in candidates
    ]
    assert payload_ids == ids
    assert (root / "commands" / ".sequence.lock").is_file()


def test_abrupt_process_exit_releases_persistent_sequence_lock(tmp_path):
    root = create_or_open_job("job", tmp_path)
    assert enqueue_command(root, "stop", {}).command_id == 1
    project_root = Path(__file__).resolve().parents[3]
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from zimage.training.commands import _sequence_lock\n"
        "with _sequence_lock(Path(sys.argv[1]) / 'commands'):\n"
        "    print('locked', flush=True)\n"
        "    os._exit(23)\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert child.returncode == 23
    assert child.stdout.strip() == "locked"
    assert (root / "commands" / ".sequence.lock").is_file()
    started = time.monotonic()
    envelope = enqueue_command(root, "stop", {})

    assert envelope.command_id == 2
    assert time.monotonic() - started < 2


def test_valid_update_is_validated_then_replaces_canonical_yaml(tmp_path):
    root = create_or_open_job("job", tmp_path)
    update = load_job_config(root)
    update["seed"] = 77
    enqueue_update(root, update)

    consumed = consume_commands(root)

    assert [item.kind for item in consumed] == ["update"]
    assert load_job_config(root)["seed"] == 77
    assert not list((root / "commands").glob("*.json"))


def test_invalid_update_is_quarantined_without_replacing_yaml(tmp_path):
    root = create_or_open_job("job", tmp_path)
    before = (root / "config.yaml").read_bytes()
    update = load_job_config(root)
    update["precision"] = "invalid"
    enqueue_update(root, update)

    assert consume_commands(root) == []

    assert (root / "config.yaml").read_bytes() == before
    quarantined = list((root / "commands" / "quarantine").glob("*.invalid"))
    assert len(quarantined) == 1
    assert not list((root / "commands").glob("*.json"))


def test_malformed_commands_are_quarantined_within_commands(tmp_path):
    root = create_or_open_job("job", tmp_path)
    commands = root / "commands"
    malformed = commands / "00000000000000000001.json"
    malformed.write_text("{broken", encoding="utf-8")
    wrong_name = commands / "not-an-id.json"
    wrong_name.write_text("{}", encoding="utf-8")

    assert consume_commands(root) == []

    quarantine = commands / "quarantine"
    assert {path.name for path in quarantine.iterdir()} == {
        "00000000000000000001.json.invalid",
        "not-an-id.json.invalid",
    }
    assert all(path.is_relative_to(commands) for path in quarantine.iterdir())


def test_handler_failure_leaves_valid_candidate_for_retry(tmp_path):
    root = create_or_open_job("job", tmp_path)
    queue = CommandQueue(root)
    queue.enqueue("retry", {})

    def fail(_):
        raise RuntimeError("transient")

    try:
        queue.consume(fail)
    except RuntimeError:
        pass

    assert len(queue.pending()) == 1
    assert [item.kind for item in queue.consume(lambda _: None)] == ["retry"]


def test_idle_save_discards_stale_pending_commands_after_stop(tmp_path):
    root = create_or_open_job("job", tmp_path)
    stale = load_job_config(root)
    stale["seed"] = 11
    write_job_state(root, JobState("job", JobStatus.RUNNING, 1, 0))
    enqueue_update(root, stale)
    write_job_state(
        root, JobState("job", JobStatus.COMPLETED, 1, 0, exit_code=0)
    )

    idle = load_job_config(root)
    idle["seed"] = 22
    save_idle_update(root, idle)

    assert load_job_config(root)["seed"] == 22
    assert list((root / "commands").glob("*.json")) == []
    assert consume_commands(root) == []
    assert load_job_config(root)["seed"] == 22


def test_idle_save_replaces_legacy_top_level_transformer_keys(tmp_path):
    import yaml

    from zimage.training.schema import KNOWN_MAIN_SOURCE, job_create_template

    root = create_or_open_job("job", tmp_path)
    document = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    model = document.pop("model")
    document["main_transformer"] = model["main_transformer"]
    document["sampling_transformer"] = model["sampling_transformer"]
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            document, default_flow_style=False, allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    nested = job_create_template()
    nested["job_name"] = "job"

    saved = save_idle_update(root, nested)

    assert saved["model"]["main_transformer"]["path"] == KNOWN_MAIN_SOURCE
    persisted = load_job_config(root)
    assert persisted["model"]["main_transformer"]["path"] == KNOWN_MAIN_SOURCE


def test_invalid_idle_save_preserves_pending_commands(tmp_path):
    root = create_or_open_job("job", tmp_path)
    pending = load_job_config(root)
    pending["seed"] = 11
    enqueue_update(root, pending)
    yaml_before = (root / "config.yaml").read_bytes()

    invalid = load_job_config(root)
    invalid["precision"] = "invalid"
    with pytest.raises(TrainingConfigError):
        save_idle_update(root, invalid)

    assert [path.name for path in (root / "commands").glob("*.json")] == [
        "00000000000000000001.json"
    ]
    assert (root / "config.yaml").read_bytes() == yaml_before


def test_idle_save_lora_update_discards_pending_commands(tmp_path):
    root = create_or_open_job("job", tmp_path)
    pending = load_job_config(root)
    pending["seed"] = 11
    enqueue_update(root, pending)

    mutated = load_job_config(root)
    mutated["lora"]["alpha"] = float(mutated["lora"]["alpha"]) + 1.0
    saved = save_idle_update(root, mutated)

    assert saved["lora"]["alpha"] == mutated["lora"]["alpha"]
    assert load_job_config(root)["lora"]["alpha"] == mutated["lora"]["alpha"]
    assert list((root / "commands").glob("*.json")) == []


def test_sequential_consume_still_applies_multiple_updates_in_order(tmp_path):
    root = create_or_open_job("job", tmp_path)
    first = load_job_config(root)
    first["seed"] = 1
    second = load_job_config(root)
    second["seed"] = 2
    enqueue_update(root, first)
    enqueue_update(root, second)

    consumed = consume_commands(root)

    assert [item.command_id for item in consumed] == [1, 2]
    assert load_job_config(root)["seed"] == 2
    assert list((root / "commands").glob("*.json")) == []
