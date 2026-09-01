# Tests

## Purpose

`tests/` mirrors the production package layout (`app.py`, `zimage/...`). A test file lives next to the module it covers, not in a flat `tests/test_*.py` pile.

Semantic separation matters because training, engine, UI, and prefs share names (`schema`, `handlers`, `pipeline`) but make different truth claims. Mixing them hides which layer a failure belongs to, and it makes it easy to treat a fast mocked contract test as if it had proven GPU execution.

Shared pytest fixtures stay in the root `conftest.py`. Binary and YAML fixtures stay under `fixtures/`.

## Directory map

```
tests/
  conftest.py                 shared fixtures (config isolation, tiny LoRA, pipeline reset)
  test_app.py                 root app.py
  simulation.py               production-parity GPU-usage runner (not pytest)
  simulation/config.yaml      simulation job spec (max_steps 100)
  fixtures/                   LoRA safetensors, job YAML, builders
  zimage/
    test_config.py            zimage/config.py
    test_paths.py             zimage/paths.py
    engine/test_*.py          zimage/engine/*.py
    prefs/test_*.py           zimage/prefs/*.py
    ui/test_*.py              zimage/ui/*.py
    training/test_*.py        zimage/training/*.py (includes capabilities and e2e)
```

| Domain | Tests | Production |
|---|---|---|
| App | `tests/test_app.py` | `app.py` |
| Config + paths | `tests/zimage/test_config.py`, `tests/zimage/test_paths.py` | `zimage/config.py`, `zimage/paths.py` |
| Engine | `tests/zimage/engine/` | `zimage/engine/` |
| Prefs | `tests/zimage/prefs/` | `zimage/prefs/` |
| UI | `tests/zimage/ui/` | `zimage/ui/` |
| Training | `tests/zimage/training/` | `zimage/training/` |
| Simulation runner | `tests/simulation.py` | `JobController.run` / `train.py run` |
| Fixtures | `tests/fixtures/` | not production code |

`__init__.py` markers exist under `tests/` and each nested domain so pytest can collect two `test_schema.py` files (prefs vs training) without module-name collisions. They do not change production imports.

## Root `training` section (schema contract)

First use bootstraps the root `config.yaml` `training` section atomically (`datasets_dir`, `jobs_dir`) when that section is **absent**. If `training` already exists but is incomplete or invalid, resolve **errors** — defaults are not applied as a silent read-time fallback. That is the intended product decision; it overrides older plan language that implied a merge-on-read.

GPU probe toggles (`every_step`, `detailed`) are YAML-only: root `training.gpu_usage` plus optional job `gpu_usage` (job keys override). Absent keys default to `false`. There is no env SSOT (`ZIMAGE_*`) and bootstrap does not write `gpu_usage`.

## Test tiers and truth claims

A test's **designed claim** is what a green result would prove. An **observed result** is what actually happened on this machine. Do not treat a skip, a hardware block, or a mocked pass as the designed claim.

**Default mocked / unit tests.** Fast validation of schema, math, contracts, UI wiring, and mocked training hand-offs. They do **not** prove production GPU execution, real weight loads, or TorchAO kernels on device.

**Tiny CUDA tests.** Real CUDA kernels and tensors on synthetic modules or tiny adapters. No production Z-Image weights. Most skip when CUDA (or FP8 compute capability) is missing; the cache place/encode/park node skips only when CUDA is missing. This tier includes production-path calls to `setup_main_transformer` + `official_flow_matching_step`, a real `UnfusedPreviewSampler` CUDA lifecycle (no injected mover/quantizer), and cache VAE+TE CUDA place → `encode_sample` → CPU park.

**Real-model capability gate.** Opt-in. Designed claim: load a persistent FP8 Turbo transformer from a local snapshot and replace unfused adapters; latent forwards differ across adapters and match on repeat. This is Diffusers + PEFT + TorchAO **latent** sampling on Blackwell, not the training loop, not `UnfusedPreviewSampler`, and not VAE PNG decode.

**Full hardware smoke.** Opt-in. Designed claim: real Base FP8 optimizer step → native checkpoint → Turbo preview JPEG → warm-start second run → GPU lease release. That path **would prove** production Base train/sample on this machine **when it passes**. The metadata prerequisite is repaired and the smoke is ready for a controlled rerun; it has not been rerun and has not passed. See observed results below.

## Observed local evidence

On this RTX 5080 with PEFT 0.20.0:

- The real Turbo capability gate (`test_real_sampling_pipeline_replaces_unfused_adapters_on_blackwell`) **has passed**.
- Base weight, text-encoder, and VAE files in the local snapshot were verified present as Hugging Face cache links. The two missing metadata files were downloaded: `scheduler/scheduler_config.json` and `model_index.json`. Offline `FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler")` now succeeds. The metadata prerequisite is repaired; the smoke is ready for a controlled rerun.
- The full Base hardware smoke (`test_real_blackwell_fp8_warm_start_turbo_preview_smoke`) was intentionally aborted earlier and has **not** been rerun. It has **not** passed. Production Base GPU training remains **unverified** until explicit rerun permission.

## Exact commands

Use the project interpreter. From the repository root:

```bat
D:\Projects\Python312\python.exe -m pytest -q --tb=line
```

### Major domains

```bat
D:\Projects\Python312\python.exe -m pytest tests/test_app.py -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/test_config.py tests/zimage/test_paths.py -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/engine -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/prefs -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/ui -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/training -q --tb=line
```

### Capability tests (default file: mocked + tiny CUDA; real-model stays skipped)

```bat
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_capabilities.py -q --tb=line
```

### Training e2e (mocked; hardware smoke stays skipped)

```bat
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_e2e.py -q --tb=line
```

### GPU-usage contract (mocked)

```bat
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_gpu_usage.py tests/zimage/training/test_loop.py tests/zimage/training/test_schema.py tests/zimage/training/test_simulation.py -q --tb=line
```

### Simulation runner (not pytest)

Zero arguments. Same entry as `train.py run`. CUDA required. See [Simulation](#simulation-production-parity-gpu-usage-run).

```bat
D:\Projects\Python312\python.exe tests/simulation.py
```

### Individual real GPU tests

Tiny CUDA, no production weights:

```bat
D:\Projects\Python312\python.exe -m pytest tests/zimage/engine/test_lora_quantize.py::test_fuse_and_quantize_on_cuda -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/engine/test_lora_quantize.py::test_fuse_and_fp8_quantize_on_cuda -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_capabilities.py::test_main_training_transformer_uses_official_fp8_setup_order -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_sampling.py::test_sample_unfused_real_cuda_lifecycle_quantizes_once_and_restores_cpu -q --tb=line
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_gpu_usage.py::test_tiny_cuda_cache_place_encode_park -q --tb=line
```

Real-model capability gate (Turbo snapshot required):

```bat
set ZIMAGE_RUN_REAL_MODEL_CAPABILITY=1
set ZIMAGE_REAL_SAMPLING_MODEL=E:\path\to\Z-Image-Turbo\snapshot
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_capabilities.py::test_real_sampling_pipeline_replaces_unfused_adapters_on_blackwell -q --tb=line
```

Full hardware smoke (Base + Turbo snapshots required):

```bat
set ZIMAGE_RUN_HARDWARE_SMOKE=1
set ZIMAGE_REAL_MAIN_MODEL=E:\path\to\Z-Image\snapshot
set ZIMAGE_REAL_SAMPLING_MODEL=E:\path\to\Z-Image-Turbo\snapshot
D:\Projects\Python312\python.exe -m pytest tests/zimage/training/test_e2e.py::test_real_blackwell_fp8_warm_start_turbo_preview_smoke -q --tb=line
```

## Simulation (production-parity GPU-usage run)

`tests/simulation.py` is not a pytest module (filename does not match `test_*.py` / `*_test.py`) and is not collected. The primary command is zero-arg:

```bat
python tests/simulation.py
```

That load path:

1. Reads `tests/simulation/config.yaml` (`max_steps: 100`, `checkpoint_every: 100`; no live `gpu_usage` key).
2. Opens/creates the job under root `training.jobs_dir` (not a tempfile). Re-runs overwrite that job's `config.yaml`.
3. Calls `JobController.run` — the same subprocess entry as `train.py run` (`job_log_session` + GPU lease).
4. Keeps a warm dataset `.cache/` (no pre-run wipe).
5. Prints an aggregate of **this run's** `{job_dir}/logs/job.log` gpu-usage lines: max `phase_peak` by phase, max `nvidia_used`, teardown `summary` line, path to `job.log`.

Probe settings are YAML-only. There is no env SSOT (`ZIMAGE_*`) and no `--compare-log`. `run_job` merges root `training.gpu_usage` with job `gpu_usage` (job keys win). The simulation YAML documents the keys in comments; omitted keys stay `false`:

| Key | Default | Effect |
|---|---|---|
| `every_step` | `false` | `step` probes at 1, 2, and checkpoint steps. `true`: every optimizer step. |
| `detailed` | `false` | Compact one-line snapshots. `true`: nbytes buckets + leftover tensor groups. |

Default `max_steps: 100` / `checkpoint_every: 100` therefore logs `step` at 1, 2, 100, a `preview_run` at 100, and a teardown `summary`. Warm cache omits `cache_encode_peak`.

Cross-run comparison is manual: read two `{jobs_dir}/{job_id}/logs/job.log` files. The runner does not parse a reference log or compute a diff.

Optional flags (omit for production parity):

| Flag | Default | Notes |
|---|---|---|
| `--config` | `tests/simulation/config.yaml` | Job spec |
| `--mode` | `subprocess` | `in-process` is direct `run_job` (dev; metrics may differ) |
| `--job-dir` | configured `jobs_dir` | Do not use a tempfile for parity |
| `--cold-cache` | off | Opt-in wipe of dataset `.cache/` |
| `--max-steps` | YAML value | Quick smoke override |
| `--datasets-dir` | root `training.datasets_dir` | Same resolution as production |

CUDA missing → exit 1. nvidia-smi missing → zeros; the job continues.

## Environment variables and local snapshots

### Real-model capability gate

| Variable | Required | Role |
|---|---|---|
| `ZIMAGE_RUN_REAL_MODEL_CAPABILITY` | yes (`1`) | Unskip the gate |
| `ZIMAGE_REAL_SAMPLING_MODEL` | yes | Absolute local Turbo snapshot |
| `ZIMAGE_REAL_ADAPTER_A` / `ZIMAGE_REAL_ADAPTER_B` | optional, both or neither | Local adapters; omitted → temporary native adapters |
| `ZIMAGE_REAL_PROMPT` | optional | Default `a capability test image` |
| `ZIMAGE_REAL_MAX_SEQUENCE_LENGTH` | optional | Default `128` |
| `ZIMAGE_REAL_HEIGHT` / `ZIMAGE_REAL_WIDTH` | optional | Default `256` / `256` |
| `ZIMAGE_REAL_ENCODE_TIMEOUT_SECONDS` | optional | Default `900` |

### Full hardware smoke

| Variable | Required | Role |
|---|---|---|
| `ZIMAGE_RUN_HARDWARE_SMOKE` | yes (`1`) | Unskip the smoke |
| `ZIMAGE_REAL_MAIN_MODEL` | yes | Absolute local **Base** snapshot |
| `ZIMAGE_REAL_SAMPLING_MODEL` | yes | Absolute local **Turbo** snapshot |

Snapshots must be absolute existing directories. The test sets `HF_HUB_OFFLINE=1`.

## What / How / Why (real GPU tests)

### `test_fuse_and_quantize_on_cuda`

- **What:** Fuse a tiny LoRA into a synthetic module on CUDA, apply int8 quantization, and check the fused scale still appears in a real GPU forward.
- **How:** `pytest tests/zimage/engine/test_lora_quantize.py::test_fuse_and_quantize_on_cuda`
- **Why CPU mocks are insufficient:** Fuse-then-quantize is a device-side weight rewrite. CPU mocks cannot show that CUDA int8 kernels keep the fused value.

### `test_fuse_and_fp8_quantize_on_cuda`

- **What:** Same fuse path, then FP8 quantization and a CUDA forward.
- **How:** `pytest tests/zimage/engine/test_lora_quantize.py::test_fuse_and_fp8_quantize_on_cuda`
- **Why CPU mocks are insufficient:** FP8 schemes and kernels exist only on CUDA (Ada 8.9+ / Blackwell). A mocked `apply_quantization` does not execute TorchAO FP8.

### `test_main_training_transformer_uses_official_fp8_setup_order`

- **What:** Tiny transformer through production `setup_main_transformer` (TorchAO `convert_to_float8_training` + PEFT adapter) then production `official_flow_matching_step`, then `Accelerator.prepare` and one backward/step. Asserts inference `apply_quantization` is never used. No production weights.
- **How:** `pytest tests/zimage/training/test_capabilities.py::test_main_training_transformer_uses_official_fp8_setup_order`
- **Why CPU mocks are insufficient:** TorchAO FP8 **training conversion** is the CUDA requirement. `Accelerator.prepare` also works on CPU; it is not CUDA-only. A mocked conversion can pass while the real kernel path is broken.

### `test_sample_unfused_real_cuda_lifecycle_quantizes_once_and_restores_cpu`

- **What:** Production `UnfusedPreviewSampler` on tiny CUDA modules with no injected mover/quantizer. FP8-quantizes the base once, writes two adapter PNG previews, restores transformer/VAE to CPU after each call (including a forced failure), and reclaims CUDA tensors.
- **How:** `pytest tests/zimage/training/test_sampling.py::test_sample_unfused_real_cuda_lifecycle_quantizes_once_and_restores_cpu`
- **Why CPU mocks are insufficient:** The mocked CUDA preview tests inject a fake quantizer and device mover. This one runs the real sampler lifecycle on CUDA.

### `test_tiny_cuda_cache_place_encode_park`

- **What:** Place VAE + text encoder on CUDA, `encode_sample`, park both to CPU. VRAM after park is within 8 MiB slack of the pre-place baseline. Skips only if CUDA is missing (not FP8).
- **How:** `pytest tests/zimage/training/test_gpu_usage.py::test_tiny_cuda_cache_place_encode_park`
- **Why CPU mocks are insufficient:** Fake residency flags do not allocate CUDA or prove VRAM reclaim.

### `test_real_sampling_pipeline_replaces_unfused_adapters_on_blackwell`

- **Designed claim:** Persistent FP8 Turbo transformer: CPU prompt encode, BF16 load, post-load FP8, unfused adapter A then B, **latent-only** forwards (`output_type="latent"`) differ across adapters and match on repeat, base weights unchanged. Pipeline `vae` is `None`. Does **not** exercise `UnfusedPreviewSampler` or VAE PNG decode.
- **How:** Set `ZIMAGE_RUN_REAL_MODEL_CAPABILITY=1` and `ZIMAGE_REAL_SAMPLING_MODEL`, then run the node id above. Optional adapters/prompt/size/timeout as in the table.
- **Why CPU mocks are insufficient:** This is the production Diffusers Z-Image transformer, PEFT adapter replace, and TorchAO FP8 sampling path. Tiny modules do not reproduce snapshot load, key layout, or Blackwell storage/quantization failures.
- **Observed:** Passed on this RTX 5080 with PEFT 0.20.0.

### `test_real_blackwell_fp8_warm_start_turbo_preview_smoke`

- **Designed claim:** One real Base FP8 optimizer step, native LoRA checkpoint (no optimizer tensors), Turbo preview JPEG, lease free, `max_steps` bump, warm-start second run to step 2, second preview, lease released again. That **would prove** the production train/sample path **when it passes**.
- **How:** Set `ZIMAGE_RUN_HARDWARE_SMOKE=1`, `ZIMAGE_REAL_MAIN_MODEL`, and `ZIMAGE_REAL_SAMPLING_MODEL`, then run the e2e node id above.
- **Why CPU mocks are insufficient:** Mocked e2e injects fake writers/samplers and never loads weights. Only this test would exercise Base train → checkpoint → Turbo sample → warm start → lease on real hardware.
- **Observed:** Metadata prerequisite repaired; smoke ready for a controlled rerun. The earlier run was intentionally aborted and has **not** been rerun. Has **not** passed. Production Base GPU training remains unverified until explicit rerun permission.

## Resource-safety protocol

- Run **one** real-weight test at a time. Do not combine the capability gate and the hardware smoke.
- Before and after: inspect Python processes and `nvidia-smi` (compute apps + VRAM).
- Use an **external** timeout (shell / Task Manager). The encode subprocess already has `ZIMAGE_REAL_ENCODE_TIMEOUT_SECONDS`; the pytest process itself should still be bounded from outside.
- Do **not** run GPU tests in parallel (`-n`, multiple terminals, overlapping pytest).
- After the test: the pytest PID tree must exit; RAM and VRAM should return near baseline.
- Weights are **local-only**. Do not point these variables at Hub IDs.

## Current known local-snapshot prerequisite

Any Base snapshot used as `ZIMAGE_REAL_MAIN_MODEL` must include `scheduler/scheduler_config.json`. Training loads the FlowMatch scheduler from that subfolder before the first step.

On this machine the metadata prerequisite is repaired: Base weight, text-encoder, and VAE files are present as Hugging Face cache links, and `scheduler/scheduler_config.json` plus `model_index.json` have been downloaded. Offline `FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler")` succeeds. The hardware smoke is ready for a controlled rerun but has not been rerun and has not passed.

Turbo (`ZIMAGE_REAL_SAMPLING_MODEL`) likewise needs its own `scheduler/` plus `transformer/` for the capability gate and the smoke preview.

## Skip behavior

Opt-in GPU tests use `@pytest.mark.skipif` on the env flag. Additional runtime skips happen when CUDA is missing, compute capability is too low (FP8 needs 8.9+; Blackwell gates need 12.0+), or a required snapshot path is unset / not an absolute existing directory. `test_tiny_cuda_cache_place_encode_park` skips only when CUDA is missing; it does not require FP8.

Show skip reasons:

```bat
D:\Projects\Python312\python.exe -m pytest -q -rs
```

`-rs` prints a short reason for every skipped test. Default suite collection includes the opt-in tests; they skip unless the flags and snapshots are set.
