from __future__ import annotations

import sys
import types

import pytest

from zimage.engine import pipeline as pipeline_mod


def _install_pipe(monkeypatch, pipe_cls):
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = pipe_cls
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    return pipe_cls


class _BasePipe:
    def __init__(self):
        self.moved_to = None
        self.offloaded = False
        self.tiled = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def enable_vae_tiling(self):
        self.tiled = True

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


def test_load_pipeline_cpu_offload_and_tiling(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)

    cpu_pipe = pipeline_mod.load_pipeline("model", "cpu", "float32", False, True)
    assert cpu_pipe.moved_to == "cpu"
    assert cpu_pipe.tiled is True
    assert cpu_pipe.offloaded is False

    cuda_pipe = pipeline_mod.load_pipeline("model", "cuda", "float32", True, False)
    assert cuda_pipe.offloaded is True
    assert cuda_pipe.moved_to is None


def test_load_pipeline_cpu_offload_ignored_on_cpu(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    pipe = pipeline_mod.load_pipeline("model", "cpu", "float32", True, False)
    assert pipe.offloaded is False
    assert pipe.moved_to == "cpu"


def test_load_pipeline_cpu_promotes_low_precision(monkeypatch):
    captured = {}

    class Pipe(_BasePipe):
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            captured.update(kwargs)
            return cls()

    _install_pipe(monkeypatch, Pipe)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    pipeline_mod.load_pipeline("model", "cpu", "float16", False, False)

    import torch

    assert captured["torch_dtype"] is torch.float32
    assert captured["low_cpu_mem_usage"] is True
    assert captured["local_files_only"] is False


def test_load_pipeline_offline_sets_local_files_only(monkeypatch):
    captured = {}

    class Pipe(_BasePipe):
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            captured.update(kwargs)
            return cls()

    _install_pipe(monkeypatch, Pipe)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pipeline_mod.load_pipeline("model", "cpu", "float32", False, False)
    assert captured["local_files_only"] is True


def test_load_pipeline_int8_quantizes_before_device(monkeypatch):
    order = []

    class Pipe(_BasePipe):
        def to(self, device):
            order.append(("to", device))
            self.moved_to = device
            return self

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            order.append("load")
            return cls()

    _install_pipe(monkeypatch, Pipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)

    def fake_apply(pipe, dtype_name, **_kwargs):
        order.append(("quant", dtype_name))
        return "transformer"

    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)

    pipe = pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)
    assert pipe.moved_to == "cuda"
    assert order == ["load", ("quant", "int8"), ("to", "cuda")]


def test_load_pipeline_fp8_quantizes_before_device(monkeypatch):
    order = []

    class Pipe(_BasePipe):
        def to(self, device):
            order.append(("to", device))
            self.moved_to = device
            return self

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            order.append("load")
            return cls()

    _install_pipe(monkeypatch, Pipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr("zimage.engine.pipeline.require_fp8_device", lambda _device: None)

    def fake_apply(pipe, dtype_name, **_kwargs):
        order.append(("quant", dtype_name))
        return "transformer"

    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)

    pipe = pipeline_mod.load_pipeline("model", "cuda", "fp8", False, False)
    assert pipe.moved_to == "cuda"
    assert order == ["load", ("quant", "fp8"), ("to", "cuda")]


def test_load_pipeline_fp8_retries_after_device(monkeypatch):
    order = []
    attempts = {"n": 0}

    class Pipe(_BasePipe):
        def to(self, device):
            order.append(("to", device))
            return self

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            order.append("load")
            return cls()

    _install_pipe(monkeypatch, Pipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr("zimage.engine.pipeline.require_fp8_device", lambda _device: None)

    def fake_apply(pipe, dtype_name, **_kwargs):
        attempts["n"] += 1
        order.append(("quant", attempts["n"]))
        if attempts["n"] == 1:
            raise RuntimeError("fp8 needs CUDA tensors")
        return "transformer"

    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)

    pipeline_mod.load_pipeline("model", "cuda", "fp8", False, False)
    assert order == ["load", ("quant", 1), ("to", "cuda"), ("quant", 2)]


def test_load_pipeline_fp8_retry_both_fail(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr("zimage.engine.pipeline.require_fp8_device", lambda _device: None)
    attempts = {"n": 0}

    def fake_apply(_pipe, _dtype, **_kwargs):
        attempts["n"] += 1
        raise RuntimeError(f"fail-{attempts['n']}")

    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)
    with pytest.raises(RuntimeError, match=r"fail-2.*CPU attempt: fail-1"):
        pipeline_mod.load_pipeline("model", "cuda", "fp8", False, False)


def test_load_pipeline_int8_quantize_failure_is_not_retried(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr(
        "zimage.engine.pipeline.apply_quantization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("int8 failed")),
    )
    with pytest.raises(RuntimeError, match="int8 failed"):
        pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)


def test_load_pipeline_post_device_quantize_raises_directly(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    checks = {"n": 0}

    def fake_quantized(_name):
        checks["n"] += 1
        return checks["n"] != 2

    monkeypatch.setattr("zimage.engine.pipeline.is_quantized_precision", fake_quantized)
    monkeypatch.setattr(
        "zimage.engine.pipeline.apply_quantization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gpu quant failed")),
    )
    with pytest.raises(RuntimeError, match="gpu quant failed"):
        pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)


def test_load_pipeline_fp8_rejects_cpu_offload(monkeypatch):
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr("zimage.engine.pipeline.require_fp8_device", lambda _device: None)
    with pytest.raises(RuntimeError, match="CPU offload"):
        pipeline_mod.load_pipeline("model", "cuda", "fp8", True, False)


def test_load_pipeline_fp8_rejects_cpu(monkeypatch):
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    monkeypatch.setattr(
        "zimage.engine.pipeline.require_fp8_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("fp8 requires CUDA")),
    )
    with pytest.raises(RuntimeError, match="fp8"):
        pipeline_mod.load_pipeline("model", "cpu", "fp8", False, False)


def test_load_pipeline_int8_requires_torchao(monkeypatch):
    monkeypatch.setattr(
        "zimage.engine.pipeline.require_torchao",
        lambda: (_ for _ in ()).throw(RuntimeError("int8 precision requires torchao")),
    )
    with pytest.raises(RuntimeError, match="torchao"):
        pipeline_mod.load_pipeline("model", "cuda", "int8", False, False)


def test_load_pipeline_falls_back_and_hints(monkeypatch):
    class ZImagePipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("ZImagePipeline is missing")

    class DiffusionPipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("weights missing")

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = ZImagePipeline
    fake_diffusers.DiffusionPipeline = DiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline_mod.load_pipeline("some-model", "cpu", "float32", False, False)
    message = str(exc_info.value)
    assert "some-model" in message
    assert "diffusers" in message
    assert "huggingface/diffusers" in message


def test_load_pipeline_falls_back_to_diffusion_pipeline(monkeypatch):
    class ZImagePipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("no turbo class")

    class DiffusionPipeline(_BasePipe):
        pass

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ZImagePipeline = ZImagePipeline
    fake_diffusers.DiffusionPipeline = DiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipe = pipeline_mod.load_pipeline("some-model", "cpu", "float32", False, False)
    assert pipe.moved_to == "cpu"


def test_load_pipeline_skips_quantization_when_unchecked(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    called = {"torchao": 0, "apply": 0}
    monkeypatch.setattr(
        "zimage.engine.pipeline.require_torchao",
        lambda: called.__setitem__("torchao", called["torchao"] + 1),
    )
    monkeypatch.setattr(
        "zimage.engine.pipeline.apply_quantization",
        lambda *_args, **_kwargs: called.__setitem__("apply", called["apply"] + 1),
    )
    pipe = pipeline_mod.load_pipeline(
        "model",
        "cuda",
        "int8",
        False,
        False,
        quantize_transformer=False,
        quantize_text_encoder=False,
    )
    assert called == {"torchao": 0, "apply": 0}
    assert pipe.moved_to == "cuda"


def test_load_pipeline_passes_quantize_flags(monkeypatch):
    _install_pipe(monkeypatch, _BasePipe)
    monkeypatch.setattr("zimage.engine.pipeline.require_torchao", lambda: None)
    captured = {}

    def fake_apply(_pipe, dtype_name, **kwargs):
        captured["dtype"] = dtype_name
        captured.update(kwargs)
        return "text_encoder"

    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)
    pipeline_mod.load_pipeline(
        "model",
        "cuda",
        "int8",
        False,
        False,
        quantize_transformer=False,
        quantize_text_encoder=True,
    )
    assert captured["dtype"] == "int8"
    assert captured["quantize_transformer"] is False
    assert captured["quantize_text_encoder"] is True


def test_load_pipeline_fuses_lora_before_quantization(monkeypatch, tmp_path):
    from zimage.engine.lora import LoraSpec, reset_lora_adapters

    reset_lora_adapters()
    order: list[str] = []

    class FusePipe(_BasePipe):
        def load_lora_weights(self, state_dict, weight_name=None, adapter_name=None):
            order.append(f"load:{adapter_name}")

        def set_adapters(self, names, adapter_weights=None):
            order.append(f"set:{names}:{adapter_weights}")

        def fuse_lora(self, adapter_names=None, lora_scale=None):
            order.append(f"fuse:{adapter_names}:{lora_scale}")

        def unload_lora_weights(self):
            order.append("unload")

    _install_pipe(monkeypatch, FusePipe)
    called = {"torchao": 0, "apply": 0}

    def fake_apply(*_args, **_kwargs):
        order.append("quantize")
        called["apply"] += 1

    monkeypatch.setattr(
        "zimage.engine.pipeline.require_torchao",
        lambda: called.__setitem__("torchao", called["torchao"] + 1),
    )
    monkeypatch.setattr("zimage.engine.pipeline.apply_quantization", fake_apply)
    monkeypatch.setattr("zimage.engine.pipeline.require_fp8_device", lambda _device: None)
    monkeypatch.setattr(
        "zimage.engine.lora._load_lora_state_dict",
        lambda _path: {"diffusion_model.layers.0.attention.to_q.lora_A.weight": None},
    )

    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    spec = LoraSpec(
        path=path,
        filename="style.safetensors",
        adapter_name="style",
        scale=0.8,
    )
    pipe = pipeline_mod.load_pipeline(
        "model",
        "cuda",
        "fp8",
        False,
        False,
        quantize_transformer=True,
        quantize_text_encoder=True,
        loras=(spec,),
    )
    assert called == {"torchao": 1, "apply": 1}
    assert order[:4] == [
        "load:style",
        "set:['style']:[0.8]",
        "fuse:['style']:0.8",
        "unload",
    ]
    assert "quantize" in order
    assert order.index("quantize") > order.index("unload")
    assert pipe.moved_to == "cuda"
    reset_lora_adapters()
