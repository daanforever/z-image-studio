from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.loaders import PeftAdapterMixin
from PIL import Image

from zimage.training.cache import CacheConfig, CacheEncoder, encode_sample
from zimage.training.dataset import DatasetSample
from zimage.training.modeling import (
    ComponentLoaders,
    ModelBackedCacheEncoder,
    ModelConfigurationError,
    ModelSource,
    ModelSources,
    SamplingCompatibilityError,
    TrainingModelComponents,
    TrainingModelLifecycle,
    _patch_torchao_float8_linear_autocast,
    _current_cuda_autocast_dtype,
    encode_prompt,
    encode_vae_latent,
    load_training_components,
    official_zimage_fp8_filter,
    prepare_vae_input,
    reject_turbo_main_source,
    resolve_model_sources,
    setup_main_transformer,
    validate_sampling_topology,
)


class RecordingOwner:
    def __init__(self, label: str, calls: list[tuple]) -> None:
        self.label = label
        self.calls = calls

    def from_pretrained(self, source, **kwargs):
        component = FakeLoadedComponent(self.label, source, kwargs)
        self.calls.append((self.label, source, kwargs, component))
        return component


class FakeLoadedComponent:
    def __init__(self, label, source, kwargs) -> None:
        self.label = label
        self.source = source
        self.kwargs = kwargs
        self.frozen = False

    def requires_grad_(self, value):
        self.frozen = not value
        return self


def make_loaders(calls):
    return ComponentLoaders(
        vae=RecordingOwner("vae", calls),
        tokenizer=RecordingOwner("tokenizer", calls),
        text_encoder=RecordingOwner("text_encoder", calls),
        transformer=RecordingOwner("transformer", calls),
        scheduler=RecordingOwner("scheduler", calls),
    )


def test_component_loading_uses_main_and_separate_sampling_sources():
    calls = []
    job = {
        "model": {
            "main_transformer": {"path": "org/main", "revision": "main-rev"},
            "sampling_transformer": {
                "path": "Tongyi-MAI/Z-Image-Turbo",
                "revision": "sample-rev",
            },
        },
    }

    components = load_training_components(job, loaders=make_loaders(calls))

    assert [(kind, source, kwargs["subfolder"], kwargs["revision"]) for kind, source, kwargs, _ in calls] == [
        ("vae", "org/main", "vae", "main-rev"),
        ("tokenizer", "org/main", "tokenizer", "main-rev"),
        ("text_encoder", "org/main", "text_encoder", "main-rev"),
        ("scheduler", "org/main", "scheduler", "main-rev"),
        ("transformer", "org/main", "transformer", "main-rev"),
        (
            "transformer",
            "Tongyi-MAI/Z-Image-Turbo",
            "transformer",
            "sample-rev",
        ),
        (
            "scheduler",
            "Tongyi-MAI/Z-Image-Turbo",
            "scheduler",
            "sample-rev",
        ),
    ]
    model_calls = [
        kwargs
        for kind, _, kwargs, _ in calls
        if kind in {"vae", "text_encoder", "transformer"}
    ]
    assert all(kwargs["torch_dtype"] is torch.bfloat16 for kwargs in model_calls)
    assert all(kwargs["disable_mmap"] is True for kwargs in model_calls)
    assert "disable_mmap" not in calls[1][2]
    assert "disable_mmap" not in calls[3][2]
    assert components.vae.frozen is True
    assert components.text_encoder.frozen is True
    assert components.sources.has_separate_sampler is True


def test_omitted_sampling_source_loads_distinct_instances_from_main():
    calls = []
    components = load_training_components(
        {"model": {"main_transformer": {"path": "local/model", "revision": None}}},
        loaders=make_loaders(calls),
    )

    assert components.sources.sampling == components.sources.main
    assert components.sampling_transformer is not components.main_transformer
    assert components.sampling_scheduler is not components.training_scheduler
    assert [kind for kind, *_ in calls].count("transformer") == 2
    assert [kind for kind, *_ in calls].count("scheduler") == 2
    assert [
        (source, kwargs["revision"])
        for kind, source, kwargs, _ in calls
        if kind in {"transformer", "scheduler"}
    ] == [
        ("local/model", None),
        ("local/model", None),
        ("local/model", None),
        ("local/model", None),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "tongyi-mai/z-image-turbo",
        "Tongyi-MAI\\Z-Image-Turbo\\",
        "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/",
        "C:/hf/models--Tongyi-MAI--Z-Image-Turbo/snapshots/abc",
    ],
)
def test_turbo_is_rejected_as_normalized_main_identity(source):
    with pytest.raises(ModelConfigurationError, match="only as sampling_transformer"):
        reject_turbo_main_source(source)


def test_turbo_remains_allowed_as_sampling_source():
    sources = resolve_model_sources(
        {
            "model": {
                "main_transformer": {"path": "Tongyi-MAI/Z-Image"},
                "sampling_transformer": {"path": "Tongyi-MAI/Z-Image-Turbo"},
            },
        }
    )
    assert sources.sampling.path.endswith("Z-Image-Turbo")


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_calls = []
        self.token_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "<chat>caption"

    def __call__(self, prompts, **kwargs):
        self.token_calls.append((prompts, kwargs))
        return SimpleNamespace(
            input_ids=torch.tensor([[10, 20, 30, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )


class FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        penultimate = torch.arange(4 * 2560, dtype=torch.float32).reshape(
            1, 4, 2560
        )
        final = torch.full_like(penultimate, -1)
        return SimpleNamespace(hidden_states=[torch.zeros_like(final), penultimate, final])


def test_prompt_encoding_matches_zimage_chat_template_and_removes_padding():
    tokenizer = FakeTokenizer()
    encoder = FakeTextEncoder()

    embedding = encode_prompt(
        tokenizer,
        encoder,
        "caption",
        max_sequence_length=4,
    )

    assert tokenizer.template_calls == [
        (
            [{"role": "user", "content": "caption"}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        )
    ]
    assert tokenizer.token_calls[0] == (
        ["<chat>caption"],
        {
            "padding": "max_length",
            "max_length": 4,
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    assert encoder.calls[0]["output_hidden_states"] is True
    assert encoder.calls[0]["attention_mask"].dtype is torch.bool
    assert embedding.dtype is torch.bfloat16
    assert embedding.shape == (3, 2560)
    assert torch.equal(
        embedding,
        torch.arange(4 * 2560, dtype=torch.float32)
        .reshape(4, 2560)[:3]
        .to(torch.bfloat16),
    )


class FakeLatentDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0

    def mode(self):
        self.mode_calls += 1
        return self.value

    def sample(self):
        raise AssertionError("latent sampling must be deterministic")


class FakeVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(shift_factor=0.25, scaling_factor=2.0)
        self.inputs = []
        self.distribution = FakeLatentDistribution(
            torch.ones((1, 16, 2, 3), dtype=torch.bfloat16)
        )

    def encode(self, pixels):
        self.inputs.append(pixels)
        return SimpleNamespace(latent_dist=self.distribution)


def test_vae_encoding_normalizes_rgb_and_uses_mode_shift_scale():
    vae = FakeVae()
    image = torch.tensor(
        [
            [[0, 127, 255], [255, 127, 0]],
            [[255, 0, 127], [0, 255, 127]],
        ],
        dtype=torch.uint8,
    )

    latent = encode_vae_latent(vae, image)

    expected_pixels = image.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    assert torch.allclose(
        vae.inputs[0].float(),
        expected_pixels.to(torch.bfloat16).float(),
    )
    assert vae.distribution.mode_calls == 1
    assert latent.dtype is torch.bfloat16
    assert latent.shape == (16, 2, 3)
    assert torch.equal(latent, torch.full_like(latent, 1.5))


def test_vae_input_supports_declared_ranges_and_rejects_invalid_values():
    uint8 = torch.tensor(
        [[[0, 127, 255], [255, 127, 0]]],
        dtype=torch.uint8,
    )
    zero_to_one = torch.tensor(
        [[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]],
    )
    normalized = torch.tensor(
        [
            [[-1.0, 0.0], [0.5, 1.0]],
            [[0.0, -0.5], [1.0, 0.5]],
            [[1.0, 0.0], [-1.0, -0.5]],
        ]
    )

    assert torch.allclose(
        prepare_vae_input(uint8),
        uint8.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0,
    )
    assert torch.allclose(
        prepare_vae_input(zero_to_one),
        zero_to_one.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0,
    )
    assert torch.equal(prepare_vae_input(normalized), normalized.unsqueeze(0))
    with pytest.raises(ModelConfigurationError, match=r"\[0, 1\] or \[-1, 1\]"):
        prepare_vae_input(torch.full((3, 2, 2), 1.01))
    with pytest.raises(ModelConfigurationError, match="finite"):
        prepare_vae_input(torch.full((3, 2, 2), torch.nan))


def test_model_backed_adapter_satisfies_cache_encoder_and_cache_semantics():
    vae = FakeVae()
    vae.distribution = FakeLatentDistribution(
        torch.ones((1, 16, 2, 2), dtype=torch.bfloat16)
    )
    tokenizer = FakeTokenizer()
    text_encoder = FakeTextEncoder()
    adapter = ModelBackedCacheEncoder(vae, tokenizer, text_encoder)
    image = Image.new("RGB", (16, 16), (0, 127, 255))
    sample = DatasetSample(
        image_path=Path("image.png"),
        caption="cache caption",
        dataset_path=Path("."),
    )
    config = CacheConfig(
        main_revision="revision",
        vae_config={"shift_factor": 0.25, "scaling_factor": 2.0},
        text_encoder_config={},
        tokenizer_config={},
        qwen_chat_template={},
        max_sequence_length=4,
    )

    assert isinstance(adapter, CacheEncoder)
    encoded_image = adapter.encode_image(image)
    assert encoded_image.latent_dist is vae.distribution
    assert vae.inputs[-1].shape == (1, 3, 16, 16)
    assert float(vae.inputs[-1].min()) == -1.0
    assert float(vae.inputs[-1].max()) == 1.0

    latent, prompt = encode_sample(sample, image, adapter, config)
    assert latent.shape == (16, 2, 2)
    assert torch.equal(latent, torch.full_like(latent, 1.5))
    assert prompt.shape == (3, 2560)
    assert prompt.dtype is torch.bfloat16
    assert tokenizer.template_calls[-1][0][0]["content"] == "cache caption"


class SetupTransformer(torch.nn.Module):
    def __init__(self, events) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.events = events

    def requires_grad_(self, value=True):
        self.events.append(("freeze", value))
        return super().requires_grad_(value)

    def to(self, *args, **kwargs):
        self.events.append(("to", args, kwargs))
        return super().to(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        self.events.append(("checkpoint",))

    def add_adapter(self, config, adapter_name):
        self.events.append(("adapter", config, adapter_name))


def test_main_transformer_fp8_setup_order_filter_and_checkpointing():
    events = []
    transformer = SetupTransformer(events)

    def config_factory(**kwargs):
        config = SimpleNamespace(**kwargs)
        events.append(("fp8_config", config))
        return config

    def convert(model, **kwargs):
        events.append(("convert", model, kwargs))

    def lora_factory(**kwargs):
        config = SimpleNamespace(**kwargs)
        events.append(("lora_config", config))
        return config

    result = setup_main_transformer(
        transformer,
        precision="fp8",
        fp8_capable=True,
        lora={
            "rank": 4,
            "alpha": 8,
            "dropout": 0.1,
            "targets": ["to_q"],
        },
        gradient_checkpointing=True,
        device="cpu",
        adapter_name="training",
        convert_to_float8_training=convert,
        float8_config_factory=config_factory,
        lora_config_factory=lora_factory,
    )

    names = [event[0] for event in events]
    assert names == [
        "freeze",
        "to",
        "fp8_config",
        "convert",
        "checkpoint",
        "lora_config",
        "adapter",
    ]
    convert_kwargs = events[3][2]
    assert convert_kwargs["module_filter_fn"] is official_zimage_fp8_filter
    assert convert_kwargs["config"].pad_inner_dim is True
    assert result.fp8_enabled is True
    assert result.effective_precision == "fp8"


def test_float8_linear_forward_uses_current_autocast_api(monkeypatch):
    from torchao.float8.config import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear

    _patch_torchao_float8_linear_autocast()
    assert getattr(Float8Linear.forward, "_zimage_uses_current_autocast", False)
    helper = inspect.getsource(_current_cuda_autocast_dtype)
    assert "get_autocast_dtype" in helper
    assert "get_autocast_gpu_dtype" not in inspect.getsource(Float8Linear.forward)

    def boom(*_args, **_kwargs):
        raise AssertionError("deprecated get_autocast_gpu_dtype")

    monkeypatch.setattr(torch, "get_autocast_gpu_dtype", boom)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda _device: torch.float32)

    converted = Float8Linear.from_float(
        torch.nn.Linear(8, 8, bias=False),
        config=Float8LinearConfig(emulate=True),
    )
    output = converted(torch.ones(2, 8))
    assert output.shape == (2, 8)


def test_fp8_filter_excludes_output_and_incompatible_linear_dimensions():
    assert official_zimage_fp8_filter(torch.nn.Linear(16, 32), "to_q") is True
    assert official_zimage_fp8_filter(torch.nn.Linear(15, 32), "to_q") is False
    assert official_zimage_fp8_filter(torch.nn.Linear(16, 32), "proj_out") is False
    assert (
        official_zimage_fp8_filter(
            torch.nn.Linear(16, 32), "decoder.proj_out"
        )
        is False
    )


def test_fp8_request_uses_bf16_fallback_when_capability_gate_fails():
    events = []
    transformer = SetupTransformer(events)

    result = setup_main_transformer(
        transformer,
        precision="fp8",
        fp8_capable=False,
        lora={"rank": 2, "alpha": 2, "dropout": 0.0, "targets": ["linear"]},
        gradient_checkpointing=False,
        device="cpu",
        convert_to_float8_training=lambda *_args, **_kwargs: pytest.fail(
            "FP8 converter must not run"
        ),
        float8_config_factory=lambda **_kwargs: pytest.fail(
            "FP8 config must not be built"
        ),
        lora_config_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert result.fp8_enabled is False
    assert result.effective_precision == "bf16"
    assert "checkpoint" not in [event[0] for event in events]
    assert [event[0] for event in events] == ["freeze", "to", "adapter"]


class TinyPeftTransformer(PeftAdapterMixin, torch.nn.Module):
    """CPU stand-in that records setup order and can host a real PEFT adapter."""

    def __init__(self, events: list) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(16, 16, bias=False)
        self.proj_out = torch.nn.Linear(16, 16, bias=False)
        self.unrelated = torch.nn.Parameter(torch.full((3,), 7.0, dtype=torch.float32))
        self.other = torch.nn.Module()
        self.other.lora_A = torch.nn.ModuleDict(
            {"other_adapter": torch.nn.Linear(8, 2, bias=False)}
        )
        self.events = events
        self.adapter_dtypes_at_add: dict[str, torch.dtype] = {}
        self.adapter_devices_at_add: dict[str, torch.device] = {}

    def requires_grad_(self, value=True):
        self.events.append(("freeze", value))
        return super().requires_grad_(value)

    def to(self, *args, **kwargs):
        self.events.append(("to", args, kwargs))
        return super().to(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        self.events.append(("checkpoint",))

    def add_adapter(self, config, adapter_name="default"):
        result = super().add_adapter(config, adapter_name=adapter_name)
        # Simulate PEFT defaulting newly created adapter weights to FP32
        # after an FP8 base conversion.
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and _peft_name_belongs(name, adapter_name):
                parameter.data = parameter.data.to(dtype=torch.float32)
        self.unrelated.requires_grad_(True)
        for parameter in self.other.parameters():
            parameter.requires_grad_(True)
        self.adapter_dtypes_at_add = {
            name: parameter.dtype
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and _peft_name_belongs(name, adapter_name)
        }
        self.adapter_devices_at_add = {
            name: parameter.device
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and _peft_name_belongs(name, adapter_name)
        }
        self.events.append(("adapter", adapter_name))
        return result


def _peft_name_belongs(name: str, adapter_name: str) -> bool:
    parts = name.split(".")
    return any(
        part.startswith("lora_") and parts[index + 1] == adapter_name
        for index, part in enumerate(parts[:-1])
    )


def _lora_kwargs() -> dict:
    return {"rank": 2, "alpha": 4, "dropout": 0.0, "targets": ["to_q"]}


def test_main_transformer_casts_new_adapter_to_bf16_leaving_fp8_and_others():
    events: list = []
    transformer = TinyPeftTransformer(events)
    fp8_clone = {"payload": None}
    requested_device = torch.device("cpu")

    def convert(model, **kwargs):
        events.append(("convert", kwargs))
        with torch.no_grad():
            model.to_q.weight.data = model.to_q.weight.data.to(
                torch.float8_e4m3fn
            )
        fp8_clone["payload"] = model.to_q.weight.detach().clone()

    result = setup_main_transformer(
        transformer,
        precision="fp8",
        fp8_capable=True,
        lora=_lora_kwargs(),
        gradient_checkpointing=True,
        device=requested_device,
        adapter_name="training_adapter",
        convert_to_float8_training=convert,
        float8_config_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert [event[0] for event in events] == [
        "freeze",
        "to",
        "convert",
        "checkpoint",
        "adapter",
    ]
    assert transformer.adapter_dtypes_at_add
    assert set(transformer.adapter_dtypes_at_add.values()) == {torch.float32}
    assert transformer.adapter_devices_at_add
    assert {device.type for device in transformer.adapter_devices_at_add.values()} == {
        "cpu"
    }

    trainable_new = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and _peft_name_belongs(name, "training_adapter")
    }
    assert trainable_new
    for name, parameter in trainable_new.items():
        assert parameter.dtype is torch.bfloat16, name
        assert parameter.device.type == requested_device.type, name
        assert parameter.requires_grad is True, name

    other = transformer.other.lora_A["other_adapter"].weight
    assert other.dtype is torch.float32
    assert other.device.type == "cpu"
    assert other.requires_grad is True
    assert transformer.unrelated.dtype is torch.float32
    assert transformer.unrelated.device.type == "cpu"
    assert transformer.unrelated.requires_grad is True

    base = transformer.to_q.get_base_layer().weight
    assert base.dtype is torch.float8_e4m3fn
    assert base.requires_grad is False
    torch.testing.assert_close(
        base.float(),
        fp8_clone["payload"].float(),
        rtol=0,
        atol=0,
    )
    assert transformer.proj_out.weight.requires_grad is False
    assert result.adapter_name == "training_adapter"
    assert result.fp8_enabled is True


def test_bf16_fallback_still_casts_trainable_adapter_to_bf16():
    events: list = []
    transformer = TinyPeftTransformer(events)
    requested_device = torch.device("cpu")

    result = setup_main_transformer(
        transformer,
        precision="fp8",
        fp8_capable=False,
        lora=_lora_kwargs(),
        gradient_checkpointing=False,
        device=requested_device,
        adapter_name="training_adapter",
        convert_to_float8_training=lambda *_args, **_kwargs: pytest.fail(
            "FP8 converter must not run"
        ),
        float8_config_factory=lambda **_kwargs: pytest.fail(
            "FP8 config must not be built"
        ),
    )

    assert result.fp8_enabled is False
    assert result.effective_precision == "bf16"
    assert [event[0] for event in events] == ["freeze", "to", "adapter"]
    assert set(transformer.adapter_dtypes_at_add.values()) == {torch.float32}
    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad and _peft_name_belongs(name, "training_adapter"):
            assert parameter.dtype is torch.bfloat16
            assert parameter.device.type == requested_device.type
            assert parameter.requires_grad is True
    assert transformer.unrelated.dtype is torch.float32
    assert transformer.unrelated.device.type == "cpu"
    assert transformer.other.lora_A["other_adapter"].weight.dtype is torch.float32
    assert transformer.other.lora_A["other_adapter"].weight.device.type == "cpu"


def test_adapter_contract_reports_name_dtype_and_device_violations():
    class MetaAdapterTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def add_adapter(self, config, adapter_name="default"):
            self.lora_A = torch.nn.ModuleDict(
                {
                    adapter_name: torch.nn.Linear(
                        16, 2, bias=False, device="meta"
                    )
                }
            )

    with pytest.raises(ModelConfigurationError, match="lora_A") as captured:
        setup_main_transformer(
            MetaAdapterTransformer(),
            precision="bf16",
            fp8_capable=False,
            lora=_lora_kwargs(),
            gradient_checkpointing=False,
            device="cpu",
            adapter_name="training_adapter",
            lora_config_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    message = str(captured.value)
    assert "training_adapter" in message
    assert "dtype=" in message
    assert "device=" in message
    assert "meta" in message


class TopologyTransformer(torch.nn.Module):
    def __init__(self, width=16, *, layers=2, architecture="ZImage") -> None:
        super().__init__()
        self.config = {
            "_class_name": architecture,
            "width": width,
            "layers": layers,
            "_name_or_path": "ignored/source",
        }
        self.block = torch.nn.Module()
        self.block.to_q = torch.nn.Linear(width, width, bias=False)
        self.block.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(width, width, bias=False)]
        )


def test_sampling_topology_accepts_targets_and_lora_shapes():
    main = TopologyTransformer()
    sampler = TopologyTransformer()
    state = {
        "transformer.block.to_q.lora_A.weight": torch.zeros(4, 16),
        "transformer.block.to_q.lora_B.weight": torch.zeros(16, 4),
        "transformer.block.to_out.0.lora_A.weight": torch.zeros(4, 16),
        "transformer.block.to_out.0.lora_B.weight": torch.zeros(16, 4),
    }

    validate_sampling_topology(
        main,
        sampler,
        ["to_q", "to_out.0"],
        state,
    )


@pytest.mark.parametrize(
    ("sampler", "state", "message"),
    [
        (TopologyTransformer(architecture="Other"), None, "architecture mismatch"),
        (TopologyTransformer(layers=3), None, "config is incompatible"),
        (
            TopologyTransformer(width=16),
            {
                "transformer.block.to_q.lora_A.weight": torch.zeros(4, 15),
                "transformer.block.to_q.lora_B.weight": torch.zeros(16, 4),
            },
            "shape",
        ),
    ],
)
def test_sampling_topology_rejects_mismatches_before_sampling(
    sampler, state, message
):
    with pytest.raises(SamplingCompatibilityError, match=message):
        validate_sampling_topology(
            TopologyTransformer(),
            sampler,
            ["to_q"],
            state,
        )


@pytest.mark.parametrize("side", ["A", "B"])
def test_sampling_topology_requires_paired_lora_tensors(side):
    shape = (4, 16) if side == "A" else (16, 4)
    state = {f"transformer.block.to_q.lora_{side}.weight": torch.zeros(shape)}

    with pytest.raises(SamplingCompatibilityError, match="missing"):
        validate_sampling_topology(
            TopologyTransformer(),
            TopologyTransformer(),
            ["to_q"],
            state,
        )


def make_lifecycle() -> TrainingModelLifecycle:
    return TrainingModelLifecycle(
        TrainingModelComponents(
            sources=ModelSources(ModelSource("main"), ModelSource("main")),
            vae=FakeVae(),
            tokenizer=FakeTokenizer(),
            text_encoder=FakeTextEncoder(),
            training_scheduler=object(),
            main_transformer=object(),
            sampling_transformer=object(),
            sampling_scheduler=object(),
        )
    )


def test_lifecycle_prepares_dataset_and_preview_embeddings_then_unloads():
    lifecycle = make_lifecycle()
    cache_encoder = lifecycle.cache_encoder()
    assert lifecycle.cache_encoder() is cache_encoder
    consumed = []
    dataset = lifecycle.prepare_dataset_embeddings(
        ["one", "two"],
        max_sequence_length=4,
        consume=lambda prompt, embedding: consumed.append((prompt, embedding)),
    )
    previews = lifecycle.prepare_preview_prompt_embeddings(
        ["preview"],
        max_sequence_length=4,
    )

    assert len(dataset) == 2
    assert [prompt for prompt, _ in consumed] == ["one", "two"]
    assert set(previews) == {"preview"}
    lifecycle.release_text_resources()
    assert lifecycle.text_resources_loaded is False
    assert lifecycle.components.text_encoder is None
    assert lifecycle.components.tokenizer is None
    assert cache_encoder.text_encoder is None
    assert cache_encoder.tokenizer is None
    with pytest.raises(RuntimeError, match="does not reload"):
        cache_encoder.encode_prompt("later", max_sequence_length=4)
    with pytest.raises(RuntimeError, match="does not reload"):
        lifecycle.prepare_preview_prompt_embeddings(
            ["later"],
            max_sequence_length=4,
        )


def test_reload_text_resources_on_cpu_then_release(monkeypatch):
    calls: list[tuple] = []
    lifecycle = make_lifecycle()
    lifecycle.release_text_resources()
    assert lifecycle.text_resources_loaded is False

    loaders = make_loaders(calls)
    tokenizer, encoder = lifecycle.reload_text_resources_on_cpu(loaders=loaders)
    assert lifecycle.text_resources_loaded is True
    assert tokenizer is lifecycle.components.tokenizer
    assert encoder is lifecycle.components.text_encoder
    assert encoder.kwargs.get("torch_dtype") == torch.bfloat16
    # Fake loaders do not move to CUDA; ensure no cuda device was requested.
    assert encoder.kwargs.get("device_map") is None
    assert "cuda" not in str(encoder.kwargs).lower()

    # encode_prompt requires a realish forward; swap in FakeTextEncoder/Tokenizer.
    lifecycle.components.tokenizer = FakeTokenizer()
    lifecycle.components.text_encoder = FakeTextEncoder()
    embedded = encode_prompt(
        lifecycle.components.tokenizer,
        lifecycle.components.text_encoder,
        "reload me",
        max_sequence_length=4,
    )
    assert isinstance(embedded, torch.Tensor)

    lifecycle.release_text_resources()
    assert lifecycle.text_resources_loaded is False
    assert lifecycle.components.tokenizer is None
    assert lifecycle.components.text_encoder is None

    lifecycle.reload_text_resources_on_cpu(loaders=loaders)
    with pytest.raises(RuntimeError, match="already loaded"):
        lifecycle.reload_text_resources_on_cpu(loaders=loaders)
