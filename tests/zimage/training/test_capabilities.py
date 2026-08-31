from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from diffusers.loaders import PeftAdapterMixin
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import torchao.float8 as torchao_float8

from zimage.training.loop import official_flow_matching_step
from zimage.training.modeling import official_zimage_fp8_filter, setup_main_transformer


REAL_MODEL_FLAG = "ZIMAGE_RUN_REAL_MODEL_CAPABILITY"
REAL_MODEL_PATH = "ZIMAGE_REAL_SAMPLING_MODEL"
REAL_ADAPTER_A = "ZIMAGE_REAL_ADAPTER_A"
REAL_ADAPTER_B = "ZIMAGE_REAL_ADAPTER_B"


class TinyTransformer(PeftAdapterMixin, torch.nn.Module):
    """Smallest useful stand-in for the Diffusers Z-Image transformer.

    ``forward`` accepts a 2D tensor for CPU adapter-replacement checks, or the
    production Z-Image list / timestep / prompt signature used by
    ``official_flow_matching_step``.
    """

    def __init__(self, *, width: int = 16) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(width, width, bias=False)
        self.incompatible = torch.nn.Linear(width, width - 1, bias=False)
        self.proj_out = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        hidden_states,
        timestep=None,
        encoder_hidden_states=None,
        return_dict: bool = False,
        **kwargs,
    ):
        if isinstance(hidden_states, torch.Tensor):
            return self.proj_out(self.to_q(hidden_states))
        preds = []
        for hidden in hidden_states:
            features = hidden.permute(1, 2, 3, 0)
            output = self.proj_out(self.to_q(features)).permute(3, 0, 1, 2)
            preds.append(output)
        if return_dict:
            return {"sample": preds}
        return (preds,)


class _TinyFlowScheduler:
    """Minimal flow-matching schedule for production ``official_flow_matching_step``."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"num_train_timesteps": 1})()
        self.timesteps = torch.tensor([500.0])
        self.sigmas = torch.tensor([0.5])


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        init_lora_weights=True,
        target_modules=["to_q"],
    )


def _fp8_cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("TorchAO FP8 execution requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability < (8, 9):
        pytest.skip(f"TorchAO FP8 execution requires compute capability 8.9+, got {capability}")


def _adapter_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None:
            names.update(lora_a.keys())
    return names


def _assert_unfused(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "merged"):
            assert module.merged is False
        if hasattr(module, "merged_adapters"):
            assert module.merged_adapters == []


def _adapter_state(model: torch.nn.Module, adapter_name: str, sign: float) -> dict:
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    assert state
    loaded = {}
    for key, value in state.items():
        fill = sign * 0.25 if ".lora_B." in key else 1.0
        loaded[key] = torch.full_like(value, fill)
    return loaded


def _reclaim_after_encode_subprocess() -> None:
    gc.collect()


def _materialize_plain_cpu_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return a plain CPU representation suitable for exact comparisons."""

    from torchao.quantization import Float8Tensor

    materialized = weight.detach().to(device="cpu")
    if isinstance(materialized, Float8Tensor):
        materialized = materialized.dequantize(output_dtype=torch.float32)
    return materialized.detach().clone()


def _assert_base_weight_unchanged(
    weight: torch.Tensor,
    pristine_weight: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        _materialize_plain_cpu_weight(weight),
        _materialize_plain_cpu_weight(pristine_weight),
        rtol=0,
        atol=0,
    )


def _probe_fp8_sampling_after_bf16_load(transformer: torch.nn.Module) -> None:
    """Convert a loaded BF16 sampler to FP8. Fail the gate if that is incompatible."""

    try:
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
    except Exception as exc:
        pytest.fail(
            "real-model gate: BF16 sampling transformer loaded, but torchao FP8 "
            f"sampling tools are unavailable (no fuse/reload fallback): {exc}"
        )
    try:
        transformer.eval()
        quantize_(transformer, Float8WeightOnlyConfig())
    except Exception as exc:
        pytest.fail(
            "real-model gate: BF16 sampling transformer loaded, but post-load "
            f"FP8 conversion is incompatible (no fuse/reload fallback): {exc}"
        )


def _replace_unfused_adapter(
    model: TinyTransformer,
    *,
    old_name: str | None,
    new_name: str,
    sign: float,
) -> None:
    if old_name is not None:
        model.delete_adapters(old_name)
        assert old_name not in _adapter_names(model)
    model.add_adapter(_lora_config(), adapter_name=new_name)
    result = set_peft_model_state_dict(
        model,
        _adapter_state(model, new_name, sign),
        adapter_name=new_name,
    )
    assert not result.unexpected_keys
    model.set_adapter(new_name)


def _float8_linear_type():
    from torchao.float8.float8_linear import Float8Linear

    return Float8Linear


def _unwrap_base_module(module: torch.nn.Module) -> torch.nn.Module:
    """Walk PEFT / wrapper attributes to the innermost converted linear."""

    seen: set[int] = set()
    current = module
    while id(current) not in seen:
        seen.add(id(current))
        getter = getattr(current, "get_base_layer", None)
        if callable(getter):
            nxt = getter()
            if isinstance(nxt, torch.nn.Module) and nxt is not current:
                current = nxt
                continue
        nested = getattr(current, "base_layer", None)
        if isinstance(nested, torch.nn.Module) and nested is not current:
            current = nested
            continue
        break
    return current


def _is_torchao_float8_linear(module: torch.nn.Module) -> bool:
    float8_cls = _float8_linear_type()
    candidates = [module, _unwrap_base_module(module), *module.modules()]
    return any(
        type(candidate) is float8_cls or isinstance(candidate, float8_cls)
        for candidate in candidates
    )


def _assert_trainable_lora_bf16_cuda(model: torch.nn.Module) -> None:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    for name, parameter in trainable:
        assert parameter.dtype is torch.bfloat16, name
        assert parameter.device.type == "cuda", name


def _lora_param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_fp8_filter_and_config_match_official_zimage_recipe():
    """CPU filter-contract check against the production official recipe."""

    config = torchao_float8.Float8LinearConfig(pad_inner_dim=True)
    model = TinyTransformer()

    assert config.pad_inner_dim is True
    assert official_zimage_fp8_filter(model.to_q, "to_q") is True
    assert official_zimage_fp8_filter(model.incompatible, "incompatible") is False
    assert official_zimage_fp8_filter(model.proj_out, "proj_out") is False


def test_main_training_transformer_uses_official_fp8_setup_order(monkeypatch):
    """Real-CUDA production setup + flow-matching on a tiny transformer.

    No production weights. This is not the opt-in Blackwell snapshot gate.
    """

    _fp8_cuda_or_skip()

    inference_calls: list[tuple] = []

    def inference_quantization_is_forbidden(*args, **kwargs):
        inference_calls.append((args, kwargs))
        pytest.fail("inference apply_quantization() must not touch the training transformer")

    import zimage.engine.quantization as inference_quantization

    monkeypatch.setattr(
        inference_quantization,
        "apply_quantization",
        inference_quantization_is_forbidden,
    )

    transformer = None
    optimizer = None
    accelerator = None
    result = None
    latent = None
    noise = None
    prompt = None
    try:
        torch.manual_seed(11)
        transformer = TinyTransformer().to(dtype=torch.bfloat16)
        setup = setup_main_transformer(
            transformer,
            precision="fp8",
            fp8_capable=True,
            lora={
                "rank": 2,
                "alpha": 2,
                "dropout": 0.0,
                "targets": ["to_q"],
            },
            gradient_checkpointing=False,
            device="cuda",
            adapter_name="training_adapter",
        )
        transformer = setup.transformer
        float8_cls = _float8_linear_type()
        eligible_base = _unwrap_base_module(transformer.to_q)

        assert setup.fp8_enabled is True
        assert _is_torchao_float8_linear(transformer.to_q)
        assert type(eligible_base) is float8_cls
        assert isinstance(eligible_base, float8_cls)
        assert type(transformer.incompatible) is torch.nn.Linear
        assert type(transformer.proj_out) is torch.nn.Linear
        _assert_trainable_lora_bf16_cuda(transformer)
        assert _adapter_names(transformer) == {"training_adapter"}

        trainable = [
            parameter
            for parameter in transformer.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1.0e-3)
        accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=1,
        )
        assert torch.device(accelerator.device).type == "cuda"
        transformer, optimizer = accelerator.prepare(transformer, optimizer)
        assert torch.device(accelerator.device).type == "cuda"
        _assert_trainable_lora_bf16_cuda(transformer)
        assert {
            parameter.device.type for parameter in transformer.parameters()
        } == {"cuda"}

        unwrapped = accelerator.unwrap_model(transformer)
        assert _is_torchao_float8_linear(unwrapped.to_q)
        assert type(_unwrap_base_module(unwrapped.to_q)) is float8_cls
        assert type(unwrapped.incompatible) is torch.nn.Linear
        assert type(unwrapped.proj_out) is torch.nn.Linear

        latent = torch.randn(16, 2, 2, device="cuda", dtype=torch.bfloat16)
        noise = torch.randn(1, 16, 2, 2, device="cuda", dtype=torch.bfloat16)
        prompt = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
        assert latent.device.type == "cuda"
        assert noise.device.type == "cuda"
        assert prompt.device.type == "cuda"

        before = _lora_param_snapshot(transformer)
        result = official_flow_matching_step(
            transformer=transformer,
            scheduler=_TinyFlowScheduler(),
            latent=latent,
            prompt_embedding=prompt,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            noise=noise,
            device="cuda",
        )
        assert result.loss.device.type == "cuda"
        assert result.loss.dtype == torch.float32
        assert result.noisy_latent.device.type == "cuda"

        accelerator.backward(result.loss)
        optimizer.step()
        after = _lora_param_snapshot(transformer)
        assert before
        assert any(not torch.equal(after[name], value) for name, value in before.items())
        assert inference_calls == []
    finally:
        del transformer, optimizer, accelerator, result, latent, noise, prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def test_setup_places_cpu_lora_onto_requested_cuda_device():
    """PEFT often allocates new LoRA tensors on CPU; setup must move only those."""

    if not torch.cuda.is_available():
        pytest.skip("tiny-CUDA LoRA placement requires CUDA")

    transformer = None
    try:
        transformer = TinyTransformer().to(dtype=torch.bfloat16)
        original_add = transformer.add_adapter

        def add_adapter_then_park_cpu(config, adapter_name="default"):
            result = original_add(config, adapter_name=adapter_name)
            for _name, parameter in transformer.named_parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(device="cpu")
            return result

        transformer.add_adapter = add_adapter_then_park_cpu
        setup = setup_main_transformer(
            transformer,
            precision="bf16",
            fp8_capable=False,
            lora={
                "rank": 2,
                "alpha": 2,
                "dropout": 0.0,
                "targets": ["to_q"],
            },
            gradient_checkpointing=False,
            device="cuda",
            adapter_name="training_adapter",
        )
        transformer = setup.transformer
        _assert_trainable_lora_bf16_cuda(transformer)
        assert _adapter_names(transformer) == {"training_adapter"}
        assert transformer.to_q.get_base_layer().weight.device.type == "cuda"
        assert transformer.proj_out.weight.device.type == "cuda"
        assert transformer.proj_out.weight.requires_grad is False
        assert transformer.incompatible.weight.device.type == "cuda"
        assert transformer.incompatible.weight.requires_grad is False
    finally:
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def test_bf16_sampling_transformer_replaces_unfused_adapter_without_stale_state():
    """CPU adapter-replacement contract; does not execute CUDA."""

    torch.manual_seed(23)
    transformer = TinyTransformer().to(dtype=torch.bfloat16).eval()
    base_weights = {
        "to_q": transformer.to_q.weight.detach().clone(),
        "incompatible": transformer.incompatible.weight.detach().clone(),
        "proj_out": transformer.proj_out.weight.detach().clone(),
    }
    hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
    base_output = transformer(hidden_states).detach()

    _replace_unfused_adapter(
        transformer,
        old_name=None,
        new_name="first",
        sign=1.0,
    )
    first_output = transformer(hidden_states).detach()
    assert _adapter_names(transformer) == {"first"}
    _assert_unfused(transformer)

    _replace_unfused_adapter(
        transformer,
        old_name="first",
        new_name="second",
        sign=-1.0,
    )
    second_output = transformer(hidden_states).detach()
    repeated_output = transformer(hidden_states).detach()

    assert _adapter_names(transformer) == {"second"}
    assert set(transformer.peft_config) == {"second"}
    assert all("first" not in name for name, _ in transformer.named_parameters())
    _assert_unfused(transformer)
    assert torch.equal(transformer.to_q.get_base_layer().weight, base_weights["to_q"])
    assert torch.equal(transformer.incompatible.weight, base_weights["incompatible"])
    assert torch.equal(transformer.proj_out.weight, base_weights["proj_out"])
    assert not torch.equal(first_output, second_output)
    assert not torch.equal(base_output, second_output)
    assert torch.equal(second_output, repeated_output)


def test_base_weight_assertion_supports_torchao_float8_tensor():
    from torchao.quantization import Float8Tensor, Float8WeightOnlyConfig, quantize_

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    ).eval()
    quantize_(model, Float8WeightOnlyConfig())
    weight = model[0].weight
    assert isinstance(weight, Float8Tensor)

    pristine_weight = _materialize_plain_cpu_weight(weight)
    _assert_base_weight_unchanged(weight, pristine_weight)

    changed_weight = pristine_weight.clone()
    changed_weight.flatten()[0] += 1
    with pytest.raises(AssertionError):
        _assert_base_weight_unchanged(weight, changed_weight)


@pytest.mark.skipif(
    os.getenv(REAL_MODEL_FLAG) != "1",
    reason=f"set {REAL_MODEL_FLAG}=1 to run the real-model Blackwell gate",
)
def test_real_sampling_pipeline_replaces_unfused_adapters_on_blackwell(tmp_path):
    """Opt-in, local-only integration gate for the persistent sampler."""

    _fp8_cuda_or_skip()
    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        pytest.skip(f"real-model gate requires Blackwell, got compute capability {capability}")

    model_value = os.getenv(REAL_MODEL_PATH)
    if not model_value:
        pytest.skip(f"real-model gate missing environment variable: {REAL_MODEL_PATH}")
    model_path = Path(model_value).expanduser()
    if not model_path.exists():
        pytest.skip(f"real-model gate path does not exist: {REAL_MODEL_PATH}={model_path}")

    adapter_a_value = os.getenv(REAL_ADAPTER_A)
    adapter_b_value = os.getenv(REAL_ADAPTER_B)
    if bool(adapter_a_value) != bool(adapter_b_value):
        pytest.fail(
            f"{REAL_ADAPTER_A} and {REAL_ADAPTER_B} must either both be set or both "
            "be omitted so the test can generate temporary native adapters"
        )

    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    prompt = os.getenv("ZIMAGE_REAL_PROMPT", "a capability test image")
    max_sequence_length = int(os.getenv("ZIMAGE_REAL_MAX_SEQUENCE_LENGTH", "128"))
    encode_timeout_seconds = int(
        os.getenv("ZIMAGE_REAL_ENCODE_TIMEOUT_SECONDS", "900")
    )
    prompt_cache = tmp_path / "prompt-embeds.pt"
    encode_script = """
import sys
import torch
from diffusers import ZImagePipeline

model_path, output_path, prompt, max_length = sys.argv[1:]
pipeline = ZImagePipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    transformer=None,
    vae=None,
    scheduler=None,
)
pipeline.text_encoder.eval()
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=prompt,
        device=torch.device("cpu"),
        do_classifier_free_guidance=False,
        max_sequence_length=int(max_length),
    )
assert negative_prompt_embeds == []
torch.save(
    [embedding.detach().to(device="cpu", dtype=torch.bfloat16) for embedding in prompt_embeds],
    output_path,
)
"""
    try:
        encoded = subprocess.run(
            [
                sys.executable,
                "-c",
                encode_script,
                str(model_path),
                str(prompt_cache),
                prompt,
                str(max_sequence_length),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=encode_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        subprocess_output = exc.stderr or exc.stdout or "<no subprocess output>"
        pytest.fail(
            "local CPU prompt-embedding subprocess timed out after "
            f"{encode_timeout_seconds} seconds: {subprocess_output}",
            pytrace=False,
        )
    if encoded.returncode != 0:
        pytest.fail(
            "local CPU prompt-embedding subprocess failed: "
            f"{encoded.stderr or encoded.stdout}"
        )
    prompt_embeds_cpu = torch.load(
        prompt_cache,
        map_location="cpu",
        weights_only=True,
    )
    assert prompt_embeds_cpu
    assert {embedding.device.type for embedding in prompt_embeds_cpu} == {"cpu"}
    prompt_cache.unlink(missing_ok=True)
    del encoded, encode_script, prompt_cache
    _reclaim_after_encode_subprocess()

    # Quantizing inside from_pretrained (TorchAoConfig/Float8WeightOnlyConfig)
    # crashes on Windows Blackwell via safetensors → torch.storage.__getitem__.
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(model_path),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        disable_mmap=True,
    )
    assert {parameter.device.type for parameter in transformer.parameters()} == {"cpu"}
    assert {
        parameter.dtype
        for parameter in transformer.parameters()
        if parameter.is_floating_point()
    } == {torch.bfloat16}
    _probe_fp8_sampling_after_bf16_load(transformer)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path),
        subfolder="scheduler",
        local_files_only=True,
    )
    pipeline = ZImagePipeline(
        scheduler=scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )
    assert {parameter.device.type for parameter in transformer.parameters()} == {"cpu"}
    assert pipeline.text_encoder is None
    assert pipeline.tokenizer is None
    assert pipeline.vae is None

    target_weight_reference = transformer.layers[0].attention.to_q.weight
    target_weight_snapshot = _materialize_plain_cpu_weight(target_weight_reference)

    if adapter_a_value and adapter_b_value:
        adapter_paths = [
            Path(adapter_a_value).expanduser(),
            Path(adapter_b_value).expanduser(),
        ]
        absent = [str(path) for path in adapter_paths if not path.exists()]
        if absent:
            pytest.skip(f"real-model adapter paths do not exist: {', '.join(absent)}")
        if adapter_paths[0].resolve() == adapter_paths[1].resolve():
            pytest.fail(
                f"{REAL_ADAPTER_A} and {REAL_ADAPTER_B} must identify distinct adapters"
            )
        adapter_labels = [REAL_ADAPTER_A, REAL_ADAPTER_B]
    else:
        adapter_paths = [tmp_path / "adapter-a", tmp_path / "adapter-b"]
        adapter_labels = ["generated adapter A", "generated adapter B"]

        def save_temporary_adapter(
            output_path: Path,
            *,
            adapter_name: str,
            sign: float,
        ) -> None:
            config = LoraConfig(
                r=1,
                lora_alpha=1,
                lora_dropout=0.0,
                init_lora_weights=True,
                target_modules=r"layers\.0\.attention\.to_q",
            )
            transformer.add_adapter(config, adapter_name=adapter_name)
            state = get_peft_model_state_dict(
                transformer,
                adapter_name=adapter_name,
            )
            assert len(state) == 2
            fixture_state = {
                key: torch.full_like(
                    value,
                    sign * 0.25 if ".lora_B." in key else 1.0,
                )
                for key, value in state.items()
            }
            result = set_peft_model_state_dict(
                transformer,
                fixture_state,
                adapter_name=adapter_name,
            )
            assert not result.unexpected_keys
            native_state = get_peft_model_state_dict(
                transformer,
                adapter_name=adapter_name,
            )
            ZImagePipeline.save_lora_weights(
                output_path,
                transformer_lora_layers=native_state,
                safe_serialization=True,
            )
            transformer.delete_adapters(adapter_name)
            assert adapter_name not in _adapter_names(transformer)
            _assert_unfused(transformer)
            _assert_base_weight_unchanged(
                target_weight_reference,
                target_weight_snapshot,
            )

        save_temporary_adapter(
            adapter_paths[0],
            adapter_name="fixture_a",
            sign=1.0,
        )
        save_temporary_adapter(
            adapter_paths[1],
            adapter_name="fixture_b",
            sign=-1.0,
        )
        assert adapter_paths[0].is_dir()
        assert adapter_paths[1].is_dir()
        assert list(adapter_paths[0].glob("*.safetensors"))
        assert list(adapter_paths[1].glob("*.safetensors"))
        assert _adapter_names(transformer) == set()
        _assert_base_weight_unchanged(
            target_weight_reference,
            target_weight_snapshot,
        )

    height = int(os.getenv("ZIMAGE_REAL_HEIGHT", "256"))
    width = int(os.getenv("ZIMAGE_REAL_WIDTH", "256"))
    prompt_embeds_cuda = None
    first = second = repeated = None

    def load_required_adapter(index: int, adapter_name: str) -> None:
        adapter_path = adapter_paths[index]
        adapter_label = adapter_labels[index]
        try:
            pipeline.load_lora_weights(
                str(adapter_path),
                adapter_name=adapter_name,
                local_files_only=True,
            )
        except Exception as exc:
            pytest.fail(
                f"{adapter_label} ({adapter_path}) is not a compatible local Z-Image "
                f"adapter for this sampler: {exc}",
                pytrace=True,
            )

    try:
        pipeline.to("cuda")
        assert {parameter.device.type for parameter in transformer.parameters()} == {"cuda"}
        assert pipeline.text_encoder is None
        assert pipeline.tokenizer is None
        assert pipeline.vae is None
        prompt_embeds_cuda = [
            embedding.to(device="cuda", dtype=torch.bfloat16)
            for embedding in prompt_embeds_cpu
        ]

        load_required_adapter(0, "first")
        pipeline.set_adapters("first")
        _assert_unfused(transformer)
        assert _adapter_names(transformer) == {"first"}
        first = pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds_cuda,
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(101),
            output_type="latent",
        ).images

        pipeline.delete_adapters("first")
        assert "first" not in _adapter_names(transformer)
        load_required_adapter(1, "second")
        pipeline.set_adapters("second")
        _assert_unfused(transformer)
        assert _adapter_names(transformer) == {"second"}
        assert set(transformer.peft_config) == {"second"}
        assert all("first" not in name for name, _ in transformer.named_parameters())

        second = pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds_cuda,
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(101),
            output_type="latent",
        ).images
        repeated = pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds_cuda,
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(101),
            output_type="latent",
        ).images

        assert torch.equal(second, repeated)
        assert not torch.equal(first, second)
        _assert_base_weight_unchanged(
            target_weight_reference,
            target_weight_snapshot,
        )
    finally:
        del first, second, repeated, prompt_embeds_cuda
        had_cuda_allocations = torch.cuda.memory_allocated() > 0
        adapter_names = _adapter_names(transformer)
        if adapter_names:
            pipeline.delete_adapters(sorted(adapter_names))
        pipeline.to("cpu")
        gc.collect()
        if had_cuda_allocations or torch.cuda.memory_allocated() > 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    assert {parameter.device.type for parameter in transformer.parameters()} == {"cpu"}
    assert {embedding.device.type for embedding in prompt_embeds_cpu} == {"cpu"}
