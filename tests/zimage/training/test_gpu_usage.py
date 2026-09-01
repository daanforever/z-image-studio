from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from zimage.training.cache import CacheConfig, encode_sample
from zimage.training.dataset import DatasetSample
from zimage.training.gpu_usage import (
    DetailedGpuUsageProbe,
    GpuProbeContext,
    GpuUsageProbe,
    GpuUsageSnapshot,
    NBYTES_BUCKETS,
    PeakMemoryScope,
    TOP_LEFTOVER_GROUPS,
    _default_gpu_usage_probe,
    _nvidia_memory,
    _parse_nvidia_smi_memory,
    collect_leftover_groups,
    collect_module_nbytes,
    format_bytes,
    format_gpu_usage,
    format_gpu_usage_detailed,
    snapshot_gpu_usage,
)
from zimage.training.modeling import (
    ModelSource,
    ModelSources,
    TrainingModelComponents,
    TrainingModelLifecycle,
)
from zimage.training.schema import CACHE_PROMPT_EMBED_HIDDEN_SIZE, GpuUsageSettings


def _fake_cuda(*, available: bool, allocated=0, reserved=0, peak=0, calls=None):
    tracked = calls if calls is not None else []

    def synchronize():
        tracked.append("sync")

    def reset_peak_memory_stats():
        tracked.append("reset")

    def memory_allocated():
        tracked.append("allocated")
        return allocated

    def memory_reserved():
        tracked.append("reserved")
        return reserved

    def max_memory_allocated():
        tracked.append("peak")
        return peak

    return SimpleNamespace(
        is_available=lambda: available,
        synchronize=synchronize,
        reset_peak_memory_stats=reset_peak_memory_stats,
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
        max_memory_allocated=max_memory_allocated,
    )


def _components(
    *,
    vae="cpu",
    text_encoder="cpu",
    transformer="cpu",
    sampling_transformer="none",
):
    kwargs = {
        "vae": SimpleNamespace(device=vae),
        "text_encoder": SimpleNamespace(device=text_encoder),
        "main_transformer": SimpleNamespace(device=transformer),
    }
    if sampling_transformer != "none":
        kwargs["sampling_transformer"] = SimpleNamespace(device=sampling_transformer)
    return SimpleNamespace(**kwargs)


def _devices(*, vae="cpu", text_encoder="cpu", transformer="cpu", sampling="none"):
    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "transformer": transformer,
        "sampling_transformer": sampling,
    }


def test_snapshot_cuda_absent_zeros_and_skips_memory_apis():
    calls: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=False, allocated=99, calls=calls))
    snap = snapshot_gpu_usage(
        "cache_place",
        _components(vae="cpu", text_encoder="cpu", transformer="cpu"),
        torch_module=fake,
    )
    assert snap.phase == "cache_place"
    assert snap.cuda_available is False
    assert snap.allocated_bytes == 0
    assert snap.reserved_bytes == 0
    assert snap.peak_allocated_bytes == 0
    assert snap.phase_peak_allocated_bytes == 0
    assert snap.nvidia_used_bytes == 0
    assert snap.nvidia_total_bytes == 0
    assert snap.module_devices == _devices()
    assert calls == []


def test_snapshot_cuda_present_reads_bytes_and_module_devices():
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=1024, reserved=2048, peak=4096)
    )
    snap = snapshot_gpu_usage(
        "cache_place",
        _components(vae="cuda", text_encoder="cuda", transformer="cpu"),
        torch_module=fake,
    )
    assert snap.phase == "cache_place"
    assert snap.cuda_available is True
    assert snap.allocated_bytes == 1024
    assert snap.reserved_bytes == 2048
    assert snap.peak_allocated_bytes == 4096
    assert snap.phase_peak_allocated_bytes == 0
    assert snap.nvidia_used_bytes == 0
    assert snap.nvidia_total_bytes == 0
    assert snap.module_devices == _devices(
        vae="cuda", text_encoder="cuda", transformer="cpu"
    )


def test_snapshot_cuda_present_via_monkeypatch(monkeypatch):
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=7, reserved=8, peak=9)
    )
    monkeypatch.setattr("zimage.training.gpu_usage.torch", fake)
    snap = snapshot_gpu_usage("train_placed", _components(transformer="cuda"))
    assert snap.cuda_available is True
    assert (snap.allocated_bytes, snap.reserved_bytes, snap.peak_allocated_bytes) == (
        7,
        8,
        9,
    )
    assert snap.module_devices["transformer"] == "cuda"


def test_snapshot_missing_components_are_none():
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    snap = snapshot_gpu_usage("teardown", None, torch_module=fake)
    assert snap.module_devices == _devices(
        vae="none", text_encoder="none", transformer="none"
    )


def test_snapshot_released_text_encoder_is_none():
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    components = SimpleNamespace(
        vae=SimpleNamespace(device="cpu"),
        text_encoder=None,
        main_transformer=SimpleNamespace(device="cuda"),
    )
    snap = snapshot_gpu_usage("train_placed", components, torch_module=fake)
    assert snap.module_devices == _devices(
        vae="cpu", text_encoder="none", transformer="cuda"
    )


@pytest.mark.parametrize(
    ("nbytes", "expected"),
    [
        (0, "0B"),
        (7, "7B"),
        (1024, "1.0KB"),
        (1024**3, "1.0GB"),
        (12458741248, "11.6GB"),
        (12809404416, "11.9GB"),
    ],
)
def test_format_bytes(nbytes: int, expected: str) -> None:
    assert format_bytes(nbytes) == expected


def test_format_gpu_usage_stable_line():
    snap = GpuUsageSnapshot(
        phase="cache_place",
        cuda_available=True,
        allocated_bytes=1024,
        reserved_bytes=2048,
        peak_allocated_bytes=4096,
        module_devices=_devices(
            vae="cuda", text_encoder="cuda", transformer="cpu", sampling="cpu"
        ),
        phase_peak_allocated_bytes=3072,
        nvidia_used_bytes=1024**3,
        nvidia_total_bytes=2 * 1024**3,
    )
    assert format_gpu_usage(snap) == (
        "gpu usage phase=cache_place cuda=1 allocated=1.0KB "
        "reserved=2.0KB peak_allocated=4.0KB phase_peak=3.0KB "
        "nvidia_used=1.0GB nvidia_total=2.0GB "
        "vae=cuda text_encoder=cuda transformer=cpu sampling_transformer=cpu"
    )


def test_snapshot_cuda_is_available_error_does_not_raise():
    class BoomCuda:
        def is_available(self):
            raise RuntimeError("driver exploded")

    snap = snapshot_gpu_usage(
        "teardown", torch_module=SimpleNamespace(cuda=BoomCuda())
    )
    assert snap.cuda_available is False
    assert snap.allocated_bytes == 0


def test_probe_swallows_logger_errors():
    class BoomLogger:
        def info(self, _msg):
            raise RuntimeError("log failed")

    GpuUsageProbe(logger=BoomLogger())("teardown")


def test_cuda_stats_synchronizes_before_memory_reads():
    calls: list[str] = []
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=1, reserved=2, peak=3, calls=calls)
    )
    snapshot_gpu_usage("train_placed", torch_module=fake)
    assert calls[:4] == ["sync", "allocated", "reserved", "peak"]


def test_snapshot_does_not_call_gc_collect(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: called.append("gc") or 0)
    fake = SimpleNamespace(cuda=_fake_cuda(available=True, allocated=1))
    snapshot_gpu_usage("teardown", torch_module=fake)
    assert called == []


def test_snapshot_reads_phase_peak_and_sampling_transformer_from_context():
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=1, reserved=2, peak=9)
    )
    ctx = GpuProbeContext(
        components=_components(transformer="cuda", sampling_transformer="cpu"),
        phase_peak_bytes=50,
    )
    snap = snapshot_gpu_usage("step", ctx, torch_module=fake)
    assert snap.phase_peak_allocated_bytes == 50
    assert snap.peak_allocated_bytes == 9
    assert snap.module_devices["transformer"] == "cuda"
    assert snap.module_devices["sampling_transformer"] == "cpu"


def test_snapshot_phase_peak_is_at_least_allocated():
    fake = SimpleNamespace(
        cuda=_fake_cuda(available=True, allocated=100, reserved=200, peak=300)
    )
    low = snapshot_gpu_usage(
        "preview_run",
        GpuProbeContext(phase_peak_bytes=50),
        torch_module=fake,
    )
    high = snapshot_gpu_usage(
        "preview_run",
        GpuProbeContext(phase_peak_bytes=500),
        torch_module=fake,
    )
    assert low.allocated_bytes == 100
    assert low.phase_peak_allocated_bytes == 100
    assert high.phase_peak_allocated_bytes == 500
    assert high.phase_peak_allocated_bytes >= high.allocated_bytes


def test_probe_second_positional_is_components_shorthand():
    lines: list[str] = []

    class Capture:
        def info(self, msg):
            lines.append(msg)

    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    GpuUsageProbe(torch_module=fake, logger=Capture())(
        "cache_place", _components(vae="cuda")
    )
    assert lines
    assert "vae=cuda" in lines[0]
    assert "phase_peak=" in lines[0]
    assert "nvidia_used=" in lines[0]
    assert "nvidia_total=" in lines[0]
    assert "sampling_transformer=" in lines[0]


def test_peak_memory_scope_reset_work_sync_read():
    calls: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=True, peak=99, calls=calls))
    with PeakMemoryScope(torch_module=fake) as scope:
        calls.append("work")
    assert calls == ["reset", "work", "sync", "peak"]
    assert scope.peak_bytes == 99


def test_peak_memory_scope_without_cuda_is_zero():
    calls: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=False, peak=99, calls=calls))
    with PeakMemoryScope(torch_module=fake) as scope:
        pass
    assert calls == []
    assert scope.peak_bytes == 0


def test_parse_nvidia_smi_memory_mib_to_bytes():
    assert _parse_nvidia_smi_memory("1234, 24576\n") == (
        1234 * 1024 * 1024,
        24576 * 1024 * 1024,
    )


def test_parse_nvidia_smi_memory_strips_units():
    assert _parse_nvidia_smi_memory("  8 MiB, 16 MiB\n") == (
        8 * 1024 * 1024,
        16 * 1024 * 1024,
    )


def test_nvidia_memory_passes_device_index(monkeypatch):
    cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="1, 2\n")

    monkeypatch.setattr("zimage.training.gpu_usage.subprocess.run", fake_run)
    used, total = _nvidia_memory(device_index=3)
    assert "-i" in cmds[0]
    assert cmds[0][cmds[0].index("-i") + 1] == "3"
    assert (used, total) == (1 * 1024 * 1024, 2 * 1024 * 1024)


def test_nvidia_memory_defaults_to_current_device(monkeypatch):
    cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="10, 20\n")

    fake = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 4))
    monkeypatch.setattr("zimage.training.gpu_usage.torch", fake)
    monkeypatch.setattr("zimage.training.gpu_usage.subprocess.run", fake_run)
    used, total = _nvidia_memory()
    assert cmds[0][cmds[0].index("-i") + 1] == "4"
    assert (used, total) == (10 * 1024 * 1024, 20 * 1024 * 1024)


def test_nvidia_memory_missing_binary_returns_zeros(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("zimage.training.gpu_usage.subprocess.run", boom)
    assert _nvidia_memory(device_index=0) == (0, 0)


def test_snapshot_records_nvidia_bytes(monkeypatch):
    monkeypatch.setattr(
        "zimage.training.gpu_usage._nvidia_memory",
        lambda device_index=None, torch_module=None: (11, 22),
    )
    fake = SimpleNamespace(cuda=_fake_cuda(available=True, allocated=1))
    snap = snapshot_gpu_usage("step", torch_module=fake)
    assert snap.nvidia_used_bytes == 11
    assert snap.nvidia_total_bytes == 22


def test_gpu_usage_module_has_no_probe_env_or_gc_collect():
    from zimage.training import gpu_usage as gpu_usage_mod
    from zimage.training import loop as loop_mod

    gpu_text = Path(gpu_usage_mod.__file__).read_text(encoding="utf-8")
    loop_text = Path(loop_mod.__file__).read_text(encoding="utf-8")
    assert "os.environ" not in gpu_text
    assert "getenv" not in gpu_text
    assert "ZIMAGE_" not in gpu_text
    assert "gc.collect" not in gpu_text
    assert "ZIMAGE_GPU" not in loop_text


def _param_module(tensor: torch.Tensor):
    return SimpleNamespace(
        parameters=lambda recurse=True: iter([tensor]),
        buffers=lambda recurse=True: iter([]),
    )


def _capture_logger(lines: list[str]):
    class Capture:
        def info(self, msg):
            lines.append(str(msg))

    return Capture()


def test_collect_module_nbytes_zeros_for_cpu_and_missing():
    nbytes = collect_module_nbytes(None)
    assert tuple(nbytes) == NBYTES_BUCKETS
    assert all(value == 0 for value in nbytes.values())
    cpu = torch.zeros(2, 2)
    nbytes = collect_module_nbytes(_components())
    assert nbytes["vae"] == 0
    ctx = GpuProbeContext(components=SimpleNamespace(vae=_param_module(cpu)))
    assert collect_module_nbytes(ctx)["vae"] == 0


def test_collect_module_nbytes_counts_named_cuda_tensors(monkeypatch):
    vae_t = torch.zeros(2, 2)
    main_t = torch.zeros(4, 1)
    opt_t = torch.zeros(3)
    prompt_t = torch.zeros(5, 5)
    sampler_t = torch.zeros(6)
    cuda_ids = {id(vae_t), id(main_t), id(opt_t), id(prompt_t), id(sampler_t)}
    monkeypatch.setattr(
        "zimage.training.gpu_usage._is_cuda_tensor",
        lambda obj: id(obj) in cuda_ids,
    )
    ctx = GpuProbeContext(
        components=SimpleNamespace(
            vae=_param_module(vae_t),
            main_transformer=_param_module(main_t),
        ),
        optimizer=SimpleNamespace(state={0: {"exp_avg": opt_t}}),
        preview_prompt_embeddings={"p": prompt_t},
        preview_sampler=SimpleNamespace(
            prompt_embeddings={},
            negative_prompt_embeddings={"n": sampler_t},
        ),
    )
    nbytes = collect_module_nbytes(ctx)
    assert nbytes["vae"] == vae_t.nbytes
    assert nbytes["main_transformer"] == main_t.nbytes
    assert nbytes["optimizer_state"] == opt_t.nbytes
    assert nbytes["preview_embed_maps"] == prompt_t.nbytes + sampler_t.nbytes
    assert nbytes["text_encoder"] == 0
    assert nbytes["sampling_transformer"] == 0


def test_collect_module_nbytes_dedupes_shared_storage(monkeypatch):
    base = torch.zeros(4, 4)
    view = base.view(2, 8)
    monkeypatch.setattr(
        "zimage.training.gpu_usage._is_cuda_tensor",
        lambda obj: obj is base or obj is view,
    )
    module = SimpleNamespace(
        parameters=lambda recurse=True: iter([base, view]),
        buffers=lambda recurse=True: iter([]),
    )
    ctx = GpuProbeContext(components=SimpleNamespace(vae=module))
    assert collect_module_nbytes(ctx)["vae"] == base.nbytes


def test_collect_leftover_groups_ranks_and_caps_top(monkeypatch):
    tensors = [torch.zeros(i + 1) for i in range(TOP_LEFTOVER_GROUPS + 1)]
    cuda_ids = {id(item) for item in tensors}
    monkeypatch.setattr(
        "zimage.training.gpu_usage._is_cuda_tensor",
        lambda obj: id(obj) in cuda_ids,
    )
    monkeypatch.setattr(
        "zimage.training.gpu_usage.gc.get_objects", lambda: list(tensors)
    )
    groups, total = collect_leftover_groups(None)
    expected_total = sum(item.nbytes for item in tensors)
    assert total == expected_total
    assert len(groups) == TOP_LEFTOVER_GROUPS
    assert groups[0]["shape"] == [TOP_LEFTOVER_GROUPS + 1]
    assert groups[0]["nbytes"] == tensors[-1].nbytes
    assert groups[0]["count"] == 1
    assert all(group["shape"] != [1] for group in groups)


def test_collect_leftover_groups_excludes_named_buckets(monkeypatch):
    named = torch.zeros(8, 8)
    leftover = torch.zeros(3, 3)
    cuda_ids = {id(named), id(leftover)}
    monkeypatch.setattr(
        "zimage.training.gpu_usage._is_cuda_tensor",
        lambda obj: id(obj) in cuda_ids,
    )
    monkeypatch.setattr(
        "zimage.training.gpu_usage.gc.get_objects", lambda: [named, leftover]
    )
    ctx = GpuProbeContext(components=SimpleNamespace(vae=_param_module(named)))
    groups, total = collect_leftover_groups(ctx)
    assert total == leftover.nbytes
    assert groups == [
        {
            "shape": [3, 3],
            "dtype": str(leftover.dtype),
            "count": 1,
            "nbytes": leftover.nbytes,
        }
    ]


def test_format_gpu_usage_detailed_includes_nbytes_and_leftover():
    snap = GpuUsageSnapshot(
        phase="train_placed",
        cuda_available=True,
        allocated_bytes=1024,
        reserved_bytes=2048,
        peak_allocated_bytes=4096,
        module_devices=_devices(),
        phase_peak_allocated_bytes=0,
        nvidia_used_bytes=0,
        nvidia_total_bytes=0,
    )
    nbytes = {key: 0 for key in NBYTES_BUCKETS}
    nbytes["vae"] = 1024
    leftover = [
        {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "count": 1,
            "nbytes": 16,
        }
    ]
    text = format_gpu_usage_detailed(snap, nbytes, leftover, leftover_nbytes=32)
    lines = text.splitlines()
    assert lines[0] == format_gpu_usage(snap)
    assert lines[1] == (
        "gpu usage   vae=1.0KB text_encoder=0B main_transformer=0B "
        "sampling_transformer=0B optimizer_state=0B preview_embed_maps=0B "
        "leftover=32B"
    )
    assert lines[2] == (
        "gpu usage   leftover shape=[2, 2] dtype=torch.float32 count=1 nbytes=16B"
    )


def test_compact_probe_logs_single_line_without_buckets():
    lines: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    GpuUsageProbe(torch_module=fake, logger=_capture_logger(lines))(
        "train_placed", _components()
    )
    assert len(lines) == 1
    assert lines[0].startswith("gpu usage phase=train_placed")
    assert "leftover" not in lines[0]
    assert "preview_embed_maps" not in lines[0]
    assert "main_transformer=" not in lines[0]


def test_detailed_probe_logs_nbytes_and_leftover_lines():
    lines: list[str] = []
    fake = SimpleNamespace(cuda=_fake_cuda(available=False))
    DetailedGpuUsageProbe(torch_module=fake, logger=_capture_logger(lines))(
        "train_placed", _components()
    )
    assert len(lines) >= 2
    assert lines[0].startswith("gpu usage phase=train_placed")
    assert "preview_embed_maps=" in lines[1]
    assert "leftover=" in lines[1]
    assert "main_transformer=" in lines[1]


def test_default_gpu_usage_probe_factory_compact_vs_detailed():
    compact = _default_gpu_usage_probe(GpuUsageSettings())
    detailed = _default_gpu_usage_probe(GpuUsageSettings(detailed=True))
    assert type(compact) is GpuUsageProbe
    assert type(detailed) is DetailedGpuUsageProbe
    assert isinstance(detailed, GpuUsageProbe)


def test_run_job_default_probe_is_compact(tmp_path, monkeypatch):
    from zimage.training import loop as loop_mod
    from zimage.training.loop import run_job

    from tests.zimage.training.test_loop import injections, make_job

    seen: list[GpuUsageSettings] = []
    real = loop_mod._default_gpu_usage_probe

    def spy(settings):
        seen.append(settings)
        probe = real(settings)
        assert type(probe) is GpuUsageProbe
        return probe

    monkeypatch.setattr(loop_mod, "_default_gpu_usage_probe", spy)
    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections()) == 0
    assert seen == [GpuUsageSettings()]


def test_run_job_detailed_true_from_job_yaml(tmp_path, monkeypatch):
    from zimage.training import loop as loop_mod
    from zimage.training.loop import run_job

    from tests.zimage.training.test_loop import injections, make_job

    seen: list[GpuUsageSettings] = []
    real = loop_mod._default_gpu_usage_probe

    def spy(settings):
        seen.append(settings)
        probe = real(settings)
        assert type(probe) is DetailedGpuUsageProbe
        return probe

    monkeypatch.setattr(loop_mod, "_default_gpu_usage_probe", spy)
    root = make_job(tmp_path, max_steps=1, gpu_usage={"detailed": True})
    assert run_job(root, **injections()) == 0
    assert seen == [GpuUsageSettings(detailed=True)]


def test_run_job_detailed_true_from_root_config(tmp_path, monkeypatch):
    from zimage.prefs.store import dump_document
    from zimage.training import loop as loop_mod
    from zimage.training.loop import run_job

    from tests.zimage.training.test_loop import injections, make_job

    dump_document(
        {
            "training": {
                "datasets_dir": "./datasets",
                "jobs_dir": "./jobs",
                "gpu_usage": {"detailed": True},
            }
        }
    )
    seen: list[GpuUsageSettings] = []
    real = loop_mod._default_gpu_usage_probe

    def spy(settings):
        seen.append(settings)
        return real(settings)

    monkeypatch.setattr(loop_mod, "_default_gpu_usage_probe", spy)
    root = make_job(tmp_path, max_steps=1)
    assert run_job(root, **injections()) == 0
    assert seen == [GpuUsageSettings(detailed=True)]


def test_run_job_injected_probe_wins_over_detailed_yaml(tmp_path, monkeypatch):
    from zimage.training import loop as loop_mod
    from zimage.training.loop import run_job

    from tests.zimage.training.test_loop import injections, make_job

    def boom(settings):
        raise AssertionError("factory must not run when probe is injected")

    monkeypatch.setattr(loop_mod, "_default_gpu_usage_probe", boom)
    phases: list[str] = []

    def probe(phase, context=None):
        phases.append(phase)

    root = make_job(tmp_path, max_steps=1, gpu_usage={"detailed": True})
    assert run_job(root, **injections(gpu_usage_probe=probe)) == 0
    assert phases == [
        "load",
        "cache_place",
        "cache_encode_peak",
        "cache_end",
        "train_placed",
        "step",
        "teardown",
        "summary",
    ]


def test_bucket_helpers_are_not_duplicated_in_simulation():
    from zimage.training import gpu_usage as gpu_usage_mod

    sim = (
        Path(__file__).resolve().parents[2] / "simulation.py"
    ).read_text(encoding="utf-8")
    gpu = Path(gpu_usage_mod.__file__).read_text(encoding="utf-8")
    assert "def collect_module_nbytes" in gpu
    assert "def _leftover_groups" in gpu
    assert "def collect_module_nbytes" not in sim
    assert "def _leftover_groups" not in sim
    assert "def _cuda_nbytes_and_ids" not in sim


_CUDA_MEMORY_SLACK_BYTES = 8 * 1024 * 1024


class _TinyLinearVae(torch.nn.Module):
    """Linear VAE stand-in whose weights occupy CUDA memory when placed."""

    def __init__(self, *, width: int = 64) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
        self.dtype = torch.bfloat16
        self.fail_encode = False

    def encode(self, pixels: torch.Tensor):
        if self.fail_encode:
            raise RuntimeError("forced encode failure")
        batch, _channels, height, width = pixels.shape
        _ = self.proj(pixels.new_ones((batch, self.proj.in_features)))
        latent = pixels.new_zeros((batch, 16, height // 8, width // 8))
        return SimpleNamespace(latent_dist=_TinyLatentDist(latent))


class _TinyLatentDist:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent


class _TinyLinearTextEncoder(torch.nn.Module):
    """Linear text encoder; 2560×2560 bf16 weights exceed the 8 MiB slack."""

    def __init__(self, *, hidden_size: int = CACHE_PROMPT_EMBED_HIDDEN_SIZE) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        projected = self.proj(
            torch.ones(
                batch,
                self.proj.in_features,
                device=input_ids.device,
                dtype=self.proj.weight.dtype,
            )
        )
        hidden = projected.unsqueeze(1).expand(batch, length, -1)
        return SimpleNamespace(hidden_states=[hidden, hidden, hidden])


class _TinyCpuTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)


class _TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[10, 20, 30, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )


def _reclaim_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _warmup_cuda_linear_context() -> None:
    """Establish cuBLAS workspace so cache encode does not inflate baseline."""

    module = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    module.to(device="cuda")
    hidden = torch.ones(1, 16, device="cuda", dtype=torch.bfloat16)
    module(hidden)
    del module, hidden
    _reclaim_cuda()


def _device_type(module: torch.nn.Module) -> str:
    return next(module.parameters()).device.type


def test_tiny_cuda_cache_place_encode_park():
    """Real CUDA place → encode → park; VRAM rises then returns within slack."""

    if not torch.cuda.is_available():
        pytest.skip("tiny-CUDA cache place/encode/park requires CUDA")

    vae = None
    text_encoder = None
    transformer = None
    lifecycle = None
    try:
        _warmup_cuda_linear_context()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        vae = _TinyLinearVae()
        text_encoder = _TinyLinearTextEncoder()
        transformer = _TinyCpuTransformer()
        lifecycle = TrainingModelLifecycle(
            TrainingModelComponents(
                sources=ModelSources(ModelSource("tiny"), ModelSource("tiny")),
                vae=vae,
                tokenizer=_TinyTokenizer(),
                text_encoder=text_encoder,
                training_scheduler=object(),
                main_transformer=transformer,
                sampling_transformer=transformer,
                sampling_scheduler=object(),
            )
        )
        adapter = lifecycle.cache_encoder()
        cuda = torch.device("cuda")
        image = Image.new("RGB", (16, 16), (0, 127, 255))
        sample = DatasetSample(
            image_path=Path("image.png"),
            caption="tiny cuda cache",
            dataset_path=Path("."),
        )
        config = CacheConfig(
            main_revision="tiny",
            vae_config={"shift_factor": 0.0, "scaling_factor": 1.0},
            text_encoder_config={},
            tokenizer_config={},
            qwen_chat_template={},
            max_sequence_length=4,
        )

        lifecycle.place_cache_modules(cuda)
        try:
            assert _device_type(vae) == "cuda"
            assert _device_type(text_encoder) == "cuda"
            assert _device_type(transformer) == "cpu"
            assert torch.cuda.memory_allocated() > baseline

            latent, prompt = encode_sample(sample, image, adapter, config)
            assert latent.device.type == "cpu"
            assert latent.dtype is torch.bfloat16
            assert prompt.device.type == "cpu"
            assert prompt.dtype is torch.bfloat16

            previews = lifecycle.prepare_preview_prompt_embeddings(
                ["tiny cuda preview"],
                max_sequence_length=config.max_sequence_length,
            )
            assert _device_type(text_encoder) == "cuda"
            for embedding in previews.values():
                assert embedding.device.type == "cpu"
                assert embedding.dtype is torch.bfloat16
        finally:
            lifecycle.park_cache_modules()

        assert _device_type(vae) == "cpu"
        assert _device_type(text_encoder) == "cpu"
        assert _device_type(transformer) == "cpu"
        assert torch.cuda.memory_allocated() <= baseline + _CUDA_MEMORY_SLACK_BYTES

        lifecycle.place_cache_modules(cuda)
        vae.fail_encode = True
        try:
            with pytest.raises(RuntimeError, match="forced encode failure"):
                encode_sample(sample, image, adapter, config)
        finally:
            lifecycle.park_cache_modules()

        assert _device_type(vae) == "cpu"
        assert _device_type(text_encoder) == "cpu"
        assert torch.cuda.memory_allocated() <= baseline + _CUDA_MEMORY_SLACK_BYTES
    finally:
        if lifecycle is not None:
            lifecycle.park_cache_modules()
        del vae, text_encoder, transformer, lifecycle
        _reclaim_cuda()
