from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from zimage.engine.pipeline import generate_image


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakePipe:
    def __init__(self, image, reject_max_seq=False, reject_callback=False, steps=None):
        self.image = image
        self.reject_max_seq = reject_max_seq
        self.reject_callback = reject_callback
        self.steps = steps
        self.scheduler = "old"
        self.calls = []
        self.progress_bar_configs = []

    def set_progress_bar_config(self, **kwargs):
        self.progress_bar_configs.append(kwargs)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_callback and "callback_on_step_end" in kwargs:
            raise TypeError("unexpected callback")
        if self.reject_max_seq and "max_sequence_length" in kwargs:
            raise TypeError("unexpected kwarg")
        callback = kwargs.get("callback_on_step_end")
        if callback is not None and self.steps is not None:
            for step in range(self.steps):
                callback(self, step, step, {})
        return _FakeResult(self.image)


class _BarePipeNoProgressConfig:
    def __init__(self, image):
        self.image = image
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(self.image)


def _non_demo_status():
    return {
        "demo": False,
        "cuda": False,
        "torch": True,
        "device": "cpu",
        "device_name": "CPU",
        "torch_version": "2.0",
        "cuda_built": "",
        "loaded": False,
    }


def test_generate_image_demo_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZIMAGE_DEMO", "1")
    progress = []
    image, seed, status = generate_image(
        "studio test",
        width=512,
        height=512,
        seed=123,
        outputs_dir=tmp_path,
        progress=lambda value, desc="": progress.append((value, desc)),
    )
    assert seed == 123
    assert status["demo"] is True
    assert status["loaded"] is False
    assert Path(status["saved"]).exists()
    assert image.size == (512, 512)
    assert progress[-1][0] == 1.0


def test_generate_image_demo_via_status_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": True, "demo_reason": "no weights"},
    )
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    progress = []
    image, seed, status = generate_image(
        "prompt",
        seed=5,
        outputs_dir=tmp_path,
        progress=lambda value, desc="": progress.append((value, desc)),
    )
    assert seed == 5
    assert status["demo"] is True
    assert status["loaded"] is False
    assert image.mode == "RGB"
    assert progress[-1][0] == 1.0


def test_generate_image_runs_pipeline(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (8, 8), "red")
    pipe = _FakePipe(fake, steps=4)
    progress = []

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)

    image, seed, status = generate_image(
        "a cat",
        model_id="model-a",
        device="cpu",
        dtype_name="float32",
        width=8,
        height=8,
        steps=4,
        guidance=0.0,
        seed=21,
        outputs_dir=tmp_path,
        progress=lambda value, desc="": progress.append((value, desc)),
    )

    assert image is fake
    assert seed == 21
    assert status["loaded"] is True
    assert status["model"] == "model-a"
    assert status["precision"] == "float32"
    assert Path(status["saved"]).exists()
    assert pipe.calls[0]["prompt"] == "a cat"
    assert pipe.calls[0]["num_inference_steps"] == 4
    assert "max_sequence_length" in pipe.calls[0]
    assert "callback_on_step_end" in pipe.calls[0]
    assert pipe.progress_bar_configs == [{"disable": True}]
    values = [v for v, _ in progress]
    assert values[0] == 0.05
    assert all(v < 1.0 for v in values[:-1])
    assert values[-1] == 1.0
    assert progress[-2][0] == 0.98
    assert "Saving" in progress[-2][1]
    # Step fractions stay at or below 0.95
    step_values = [v for v, d in progress if d.startswith("Generating…") and "/" in d]
    assert step_values
    assert max(step_values) <= 0.95


def test_generate_image_drops_max_sequence_length(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "green")
    pipe = _FakePipe(fake, reject_max_seq=True)

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )

    image, seed, _status = generate_image(
        "prompt",
        seed=3,
        width=4,
        height=4,
        outputs_dir=tmp_path,
    )
    assert image is fake
    assert seed == 3
    assert any("max_sequence_length" not in call for call in pipe.calls)


def test_generate_image_ignores_scheduler_errors(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "blue")

    class BrokenSchedulerPipe:
        def __init__(self, image):
            self.image = image

        @property
        def scheduler(self):
            return "old"

        @scheduler.setter
        def scheduler(self, _value):
            raise RuntimeError("cannot replace scheduler")

        def __call__(self, **kwargs):
            return _FakeResult(self.image)

    pipe = BrokenSchedulerPipe(fake)
    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )

    image, _seed, _status = generate_image("prompt", seed=1, outputs_dir=tmp_path)
    assert image is fake


def test_generate_image_without_scheduler_attr(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "yellow")
    created_devices = []

    class BarePipe:
        def __call__(self, **kwargs):
            return _FakeResult(fake)

    import torch

    real_generator = torch.Generator

    def fake_generator(device="cpu"):
        created_devices.append(device)
        return real_generator(device="cpu")

    monkeypatch.setattr(torch, "Generator", fake_generator)
    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (BarePipe(), {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)

    image, seed, status = generate_image("prompt", seed=8, outputs_dir=tmp_path)
    assert image is fake
    assert seed == 8
    assert status["device"] == "cuda"
    assert created_devices == ["cuda"]


def test_generate_image_callback_typeerror_fallback(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "purple")
    pipe = _FakePipe(fake, reject_callback=True)
    progress = []

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)

    image, seed, _status = generate_image(
        "prompt",
        seed=3,
        width=4,
        height=4,
        steps=2,
        outputs_dir=tmp_path,
        progress=lambda value, desc="": progress.append((value, desc)),
    )
    assert image is fake
    assert seed == 3
    assert any("callback_on_step_end" not in call for call in pipe.calls)
    assert progress[-1][0] == 1.0
    assert all(v < 1.0 for v, _ in progress[:-1])


def test_generate_image_without_progress_bar_config(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "orange")
    pipe = _BarePipeNoProgressConfig(fake)

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)

    image, seed, _status = generate_image("prompt", seed=9, outputs_dir=tmp_path)
    assert image is fake
    assert seed == 9


def test_generate_image_reports_loading_on_cache_miss(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (4, 4), "cyan")
    pipe = _FakePipe(fake, steps=1)
    progress = []

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: False)

    generate_image(
        "prompt",
        seed=1,
        steps=1,
        outputs_dir=tmp_path,
        progress=lambda value, desc="": progress.append((value, desc)),
    )
    assert progress[0] == (0.0, "Loading model…")
    assert any(desc == "Loading model…" and value == 0.02 for value, desc in progress)


def test_generate_image_syncs_loras(monkeypatch, tmp_path: Path):
    from zimage.engine.lora import LoraSpec

    fake = Image.new("RGB", (4, 4), "red")
    pipe = _FakePipe(fake)
    captured = {}
    spec = LoraSpec(
        path=tmp_path / "style.safetensors",
        filename="style.safetensors",
        adapter_name="style",
        scale=0.8,
    )

    def fake_ensure(*_args, **kwargs):
        captured["skip"] = kwargs.get("skip_quantize_for_lora")
        return pipe, {}

    def fake_sync(synced_pipe, specs):
        captured["pipe"] = synced_pipe
        captured["specs"] = tuple(specs)

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr("zimage.engine.pipeline.ensure_pipeline", fake_ensure)
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)
    monkeypatch.setattr("zimage.engine.pipeline.sync_lora_adapters", fake_sync)

    image, seed, status = generate_image(
        "prompt",
        seed=2,
        outputs_dir=tmp_path,
        loras=(spec,),
    )
    assert image is fake
    assert seed == 2
    assert captured["pipe"] is pipe
    assert captured["specs"] == (spec,)
    assert captured["skip"] is True
    assert status["loras"] == [{"name": "style.safetensors", "strength": 0.8}]


def test_generate_image_demo_skips_lora_sync(monkeypatch, tmp_path: Path):
    from zimage.engine.lora import LoraSpec

    called = {"n": 0}
    spec = LoraSpec(
        path=tmp_path / "style.safetensors",
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )
    monkeypatch.setenv("ZIMAGE_DEMO", "1")
    monkeypatch.setattr(
        "zimage.engine.pipeline.sync_lora_adapters",
        lambda *_args, **_kwargs: called.__setitem__("n", called["n"] + 1),
    )
    image, _seed, status = generate_image(
        "prompt",
        seed=1,
        outputs_dir=tmp_path,
        loras=(spec,),
    )
    assert status["demo"] is True
    assert image.mode == "RGB"
    assert called["n"] == 0


def test_generate_image_lora_sync_error_propagates(monkeypatch, tmp_path: Path):
    from zimage.engine.lora import LoraSpec

    fake = Image.new("RGB", (4, 4), "red")
    spec = LoraSpec(
        path=tmp_path / "style.safetensors",
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )
    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (_FakePipe(fake), {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "zimage.engine.pipeline.sync_lora_adapters",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad adapter")),
    )
    with pytest.raises(RuntimeError, match="bad adapter"):
        generate_image("prompt", seed=1, outputs_dir=tmp_path, loras=(spec,))


def test_generate_image_passes_skip_quantize_on_cache_check(monkeypatch, tmp_path: Path):
    from zimage.engine.lora import LoraSpec

    fake = Image.new("RGB", (4, 4), "red")
    captured = {}
    spec = LoraSpec(
        path=tmp_path / "style.safetensors",
        filename="style.safetensors",
        adapter_name="style",
        scale=1.0,
    )

    def fake_hit(*args, **kwargs):
        captured["skip"] = kwargs.get("skip_quantize_for_lora")
        if len(args) >= 8:
            captured["skip"] = args[7]
        return True

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (_FakePipe(fake), {}),
    )
    monkeypatch.setattr("zimage.engine.pipeline._pipeline_cache_hit", fake_hit)
    monkeypatch.setattr("zimage.engine.pipeline.sync_lora_adapters", lambda *_a, **_k: None)
    generate_image("prompt", seed=1, outputs_dir=tmp_path, loras=(spec,))
    assert captured["skip"] is True

