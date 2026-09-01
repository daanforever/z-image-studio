from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SIM_PATH = Path(__file__).resolve().parents[2] / "simulation.py"
SIM_CONFIG = Path(__file__).resolve().parents[2] / "simulation" / "config.yaml"


def _load_simulation():
    spec = importlib.util.spec_from_file_location(
        "zimage_simulation_runner", SIM_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sim():
    return _load_simulation()


def test_parse_args_zero_arg_defaults(sim):
    args = sim.parse_args([])
    assert args.config == sim.CONFIG_PATH
    assert args.mode == "subprocess"
    assert args.job_dir is None
    assert args.cold_cache is False
    assert args.max_steps is None
    assert args.datasets_dir is None


def test_simulation_source_has_no_hooks():
    text = SIM_PATH.read_text(encoding="utf-8")
    assert "class VramCensus" not in text
    assert "_install_hooks" not in text
    assert "monkeypatch" not in text
    assert "compare-log" not in text
    assert "compare_log" not in text
    assert "tempfile" not in text
    assert "def collect_module_nbytes" not in text


def test_simulation_config_steps():
    raw = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
    assert raw["max_steps"] == 100
    assert raw["checkpoint_every"] == 100
    assert "gpu_usage" not in raw
    comment_text = SIM_CONFIG.read_text(encoding="utf-8")
    assert "gpu_usage:" in comment_text


def test_parse_formatted_bytes(sim):
    assert sim.parse_formatted_bytes("128B") == 128
    assert sim.parse_formatted_bytes("1.5KB") == int(1.5 * 1024)
    assert sim.parse_formatted_bytes("12.1GB") == int(12.1 * 1024**3)
    assert sim.parse_formatted_bytes("not-a-size") is None


def test_parse_and_aggregate_this_run_only(sim, tmp_path):
    prior = (
        "===== session start 2026-01-01T00:00:00Z pid=1 =====\n"
        "gpu usage phase=step cuda=1 allocated=1.0GB reserved=1.0GB "
        "peak_allocated=1.0GB phase_peak=9.0GB nvidia_used=9.0GB "
        "nvidia_total=16.0GB vae=cpu text_encoder=cpu transformer=cpu "
        "sampling_transformer=none\n"
    )
    current = (
        "===== session start 2026-09-01T10:00:00Z pid=2 =====\n"
        "gpu usage phase=load cuda=1 allocated=2.0GB reserved=2.0GB "
        "peak_allocated=2.0GB phase_peak=2.0GB nvidia_used=3.0GB "
        "nvidia_total=16.0GB vae=cuda text_encoder=cuda transformer=cpu "
        "sampling_transformer=none\n"
        "gpu usage phase=step cuda=1 allocated=4.0GB reserved=4.0GB "
        "peak_allocated=5.0GB phase_peak=5.0GB nvidia_used=6.0GB "
        "nvidia_total=16.0GB vae=cpu text_encoder=cpu transformer=cuda "
        "sampling_transformer=none\n"
        "gpu usage phase=step cuda=1 allocated=4.2GB reserved=4.0GB "
        "peak_allocated=5.5GB phase_peak=4.5GB nvidia_used=5.5GB "
        "nvidia_total=16.0GB vae=cpu text_encoder=cpu transformer=cuda "
        "sampling_transformer=none\n"
        "gpu usage phase=preview_run cuda=1 allocated=7.0GB reserved=8.0GB "
        "peak_allocated=8.0GB phase_peak=8.1GB nvidia_used=8.5GB "
        "nvidia_total=16.0GB vae=cuda text_encoder=none transformer=cpu "
        "sampling_transformer=cuda\n"
        "gpu usage phase=summary max_step_peak=5.0GB max_preview_peak=8.1GB "
        "max_nvidia_used=8.5GB\n"
        "gpu usage phase=summary cuda=1 allocated=1.0GB reserved=1.0GB "
        "peak_allocated=8.0GB phase_peak=8.1GB nvidia_used=2.0GB "
        "nvidia_total=16.0GB vae=cpu text_encoder=cpu transformer=cpu "
        "sampling_transformer=none\n"
    )
    log_path = tmp_path / "job.log"
    log_path.write_text(prior + current, encoding="utf-8")
    start = len(prior.encode("utf-8"))
    text = sim.read_log_since(log_path, start)
    assert "phase_peak=9.0GB" not in text
    summary = sim.summarize_gpu_usage_log(text, log_path)
    assert "  load=2.0GB" in summary
    assert "  step=5.0GB" in summary
    assert "  preview_run=8.1GB" in summary
    assert "max nvidia_used=8.5GB" in summary
    assert (
        "gpu usage phase=summary max_step_peak=5.0GB "
        "max_preview_peak=8.1GB max_nvidia_used=8.5GB"
    ) in summary
    assert f"job.log: {log_path}" in summary
    assert "9.0GB" not in summary


def test_help_exits_zero(sim):
    with pytest.raises(SystemExit) as error:
        sim.parse_args(["--help"])
    assert error.value.code == 0
