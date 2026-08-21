from __future__ import annotations

from pathlib import Path

from PIL import Image

from zimage.engine.pipeline import generate_image


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakePipe:
    def __init__(self, image, reject_max_seq=False):
        self.image = image
        self.reject_max_seq = reject_max_seq
        self.scheduler = "old"
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_max_seq and "max_sequence_length" in kwargs:
            raise TypeError("unexpected kwarg")
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
    image, seed, status = generate_image(
        "studio test",
        width=512,
        height=512,
        seed=123,
        outputs_dir=tmp_path,
    )
    assert seed == 123
    assert status["demo"] is True
    assert status["loaded"] is False
    assert Path(status["saved"]).exists()
    assert image.size == (512, 512)


def test_generate_image_demo_via_status_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "zimage.engine.pipeline.runtime_status",
        lambda: {"demo": True, "demo_reason": "no weights"},
    )
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    image, seed, status = generate_image("prompt", seed=5, outputs_dir=tmp_path)
    assert seed == 5
    assert status["demo"] is True
    assert status["loaded"] is False
    assert image.mode == "RGB"


def test_generate_image_runs_pipeline(monkeypatch, tmp_path: Path):
    fake = Image.new("RGB", (8, 8), "red")
    pipe = _FakePipe(fake)
    progress = []

    monkeypatch.setattr("zimage.engine.pipeline.runtime_status", _non_demo_status)
    monkeypatch.setattr("zimage.engine.pipeline.resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        "zimage.engine.pipeline.ensure_pipeline",
        lambda *_args, **_kwargs: (pipe, {}),
    )

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
    assert progress[0][0] == 0.05
    assert progress[-1][0] == 1.0


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

    image, seed, status = generate_image("prompt", seed=8, outputs_dir=tmp_path)
    assert image is fake
    assert seed == 8
    assert status["device"] == "cuda"
    assert created_devices == ["cuda"]
