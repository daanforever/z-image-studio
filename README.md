# Z-Image-Turbo Studio

Local **Gradio** web UI for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) inference via `diffusers`. The model loads directly, on CUDA when PyTorch is built with GPU support.

Official Turbo recipe from the model card: **9 steps** (8 DiT forwards) and **`guidance_scale=0`**.

## Features

- Prompt, resolution presets, seed, batch (incremental seeds), steps, time shift
- Editable UI fields persist in `config.yaml` (project root) and reload when the page opens
- Stop in the navbar cancels the rest of the batch without discarding images already generated
- Clear in the navbar removes generated images (`.png` / `.jpg` / `.jpeg`) from the Output dir
- Model path: Hugging Face ID **or** a local snapshot
- Auto-selects `cuda` / `cpu`, VRAM status
- Precision: **fp8** by default; also `bfloat16` / `float16` / `float32` / `int8` (torchao on checked modules)
- LoRA: local directory (`ZIMAGE_LORA_DIR`), multi-select `.safetensors` / `.pt`, per-adapter strength (fused into base weights before quantization so VRAM stays near the base model; changing adapters/strength reloads)
- CPU offload and VAE tiling for low VRAM
- Saves JPEGs by default (PNG optional) to the Output dir field (default `./outputs`; Windows paths accepted)
- Demo mode with no weights and no GPU (to inspect the UI)

The UI does not use quantized `Disty0/Z-Image-Turbo-SDNQ-*` checkpoints. Use the official **Z-Image-Turbo**.

## Installation (Windows, RTX 50xx)

In the project directory:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

`pip install torch` from PyPI installs a **CPU build**. That is not enough for an RTX 5080 — you need a `cu128`/`cu130` wheel from `download.pytorch.org`. If a CPU build is already installed, a plain `pip install` will not replace it: use `--force-reinstall` (or `pip uninstall torch torchvision` first).

If `from diffusers import ZImagePipeline` fails, install diffusers from git:

```bat
pip install git+https://github.com/huggingface/diffusers
```

Verify the GPU:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expect `True` and the GPU name.

## Launch

The simplest path on this machine is `launch.bat`. It sets:

| Variable | Default |
|---|---|
| `HF_HUB_CACHE` | `E:\Backup\huggingface\hub` |
| `HF_HOME` | `E:\Backup\huggingface` |
| `HF_HUB_OFFLINE` | `0` (Hub downloads allowed) |
| `GRADIO_ANALYTICS_ENABLED` | `False` (opt in via `.env`) |
| `HF_HUB_DISABLE_TELEMETRY` | `1` (set `0` in `.env` to enable) |
| `ZIMAGE_MODEL` | `Tongyi-MAI/Z-Image-Turbo` |
| `ZIMAGE_LORA_DIR` | (unset — set a local folder of LoRA files) |
| port | `43127` |

```bat
launch.bat
```

Or run manually:

```bat
python app.py --host 127.0.0.1 --port 43127
```

Open http://127.0.0.1:43127

To load a specific snapshot without hitting the Hub:

```bat
set ZIMAGE_MODEL=E:\Backup\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\0e36c2b379e66fa531d01cc531c44919e5f1c6fd
python app.py
```

Copy `.env.example` to `.env` — the app loads those variables on startup (existing env wins).

**Telemetry.** After loading `.env`, the app sets `GRADIO_ANALYTICS_ENABLED=False` and `HF_HUB_DISABLE_TELEMETRY=1` if they are not already in the environment. Opt in via `.env` (existing variables are not overwritten):

```
GRADIO_ANALYTICS_ENABLED=True
HF_HUB_DISABLE_TELEMETRY=0
```

Gradio only treats the literal value `True` as enabled.

Gallery **Share** (upload to Hugging Face) is hidden unless you pass `--share`.

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --host 127.0.0.1 --port 43127
```

Without CUDA, CPU generation is very slow. To check the UI:

```bash
ZIMAGE_DEMO=1 python app.py --port 43127
```

## Turbo vs full Z-Image

| | Z-Image-Turbo | Z-Image |
|---|---|---|
| Model | `Tongyi-MAI/Z-Image-Turbo` | `Tongyi-MAI/Z-Image` |
| Steps | 9 | 28–50 |
| Guidance | 0.0 | 3.0–5.0 |
| Negative prompt | not needed | useful |

This app is tuned for Turbo. You can point the model field at full Z-Image, but raise guidance manually under **Advanced**.

## VRAM

By default Turbo loads in **fp8** (Ada 8.9+ / Blackwell, including RTX 5080; incompatible with CPU offload). **bfloat16** is the full-precision path if you do not need fp8. **int8** if you need CPU offload. Plus **VAE tiling**. These are not Disty0/SDNQ checkpoints: the same `Tongyi-MAI/Z-Image-Turbo` is quantized in-process.

**LoRA:** adapters are fused into the base weights (on GPU when CUDA is used without CPU offload), then the pipeline is quantized as usual. Steady-state VRAM with LoRA should match the same precision without LoRA (aside from activation memory during a step). Changing the selected adapters or strength requires a full model reload.

## Training

LoRA training against **Tongyi-MAI/Z-Image** (Base). Optional preview sampling can use **Tongyi-MAI/Z-Image-Turbo**. The Gradio app has **Generate** and **Training** tabs. The CLI is `python train.py`.

### Root `config.yaml`

Training paths live only under the root `training` section (not under `ui`):

```yaml
training:
  datasets_dir: ./datasets
  jobs_dir: ./jobs
  # optional; omitted keys default to false. Job YAML `gpu_usage` overrides these.
  # YAML-only — no ZIMAGE_* / env SSOT.
  gpu_usage:
    every_step: false  # step probes at 1, 2, checkpoint steps; true = every step
    detailed: false    # compact log line; true = nbytes buckets + leftover groups
```

The first training call (CLI or Training tab) writes `datasets_dir` / `jobs_dir` atomically if the `training` section is missing — it does not write `gpu_usage`. If `training` is present but `datasets_dir` / `jobs_dir` are missing, empty, or not strings, training fails — those defaults are not applied as a silent read-time fallback. GPU probe toggles are YAML-only (root `training.gpu_usage`, job `gpu_usage`); there are no environment variables for them. Compact `gpu usage ...` lines go to `logs/job.log`. Compare runs by reading those logs — there is no log-diff tool.

`datasets/` and `jobs/` are gitignored (`.gitkeep` only).

### Job YAML

Create/Open writes a full default job. `job_id` is a lowercase ASCII slug of the name you type; the original string is stored as `job_name`. Opening an existing slug does not overwrite `config.yaml` or `state.json`.

`model.main_transformer` must be Base (`Tongyi-MAI/Z-Image`). Turbo is rejected as `model.main_transformer` and is only valid as optional `model.sampling_transformer`. Top-level `main_transformer` and `sampling_transformer` keys are no longer valid; they must be nested under `model`. `datasets[].name` is a folder under `datasets_dir` or an absolute path. `precision` is `fp8` or `bf16`. When both `epochs` and `max_steps` are set, **`max_steps` wins**. Diffusers keys (`guidance_scale`, `num_inference_steps`, `time_shift`, `width`, `height`, `seed`, `prompt`, `negative_prompt`) live on `sampling`; each `samples[]` map overlays those keys. Existing jobs with the old nested sampling YAML fail Validate/Start (unknown keys) and must be edited by hand.

There is no `init_adapter` field. Optional job `gpu_usage` (`every_step`, `detailed`) overrides root `training.gpu_usage`; omitted keys stay `false`.

```yaml
job_name: "my style"
model:
  main_transformer:
    path: Tongyi-MAI/Z-Image
    revision: null
  sampling_transformer:
    path: Tongyi-MAI/Z-Image-Turbo
    revision: null
datasets:
  - name: my-dataset
    default_caption: ""
lora:
  rank: 4
  alpha: 4
  dropout: 0.0
  targets: [to_k, to_q, to_v, to_out.0]
precision: fp8
gradient_checkpointing: true
seed: 0
epochs: 1
max_steps: 500
checkpoint_every: 100
optimizer:
  name: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
scheduler:
  name: constant
  warmup_steps: 0
weighting_scheme: none
logit_mean: 0.0
logit_std: 1.0
mode_scale: 1.29
max_sequence_length: 512
sampling:
  num_inference_steps: 9
  guidance_scale: 0.0
  time_shift: 3.0
  width: 1024
  height: 1024
  seed: 42
  prompt: ''
  negative_prompt: ''
  image_format: jpeg
  samples:
  - prompt: 'a photo of a dog'
```

### Dataset layout

```
datasets/{name}/
  photo.png              # .png / .jpg / .jpeg / .webp
  photo.txt              # sidecar caption (UTF-8)
  .cache/
    photo.png.safetensors
```

Caption is the sidecar `.txt` when it is non-empty, otherwise `default_caption`. A sample with neither is rejected. Images are center-cropped so each side is a multiple of 16; there is no resize or pad and no 1024×1024 area limit. VRAM is the practical size limit. `.cache/` holds versioned safetensors (latent + prompt embedding); files under `.cache/` are not treated as dataset images.

### Job layout

```
jobs/{job_id}/
  config.yaml
  state.json
  commands/
  checkpoints/
  previews/
    {step:05d}-{index:02d}-sample.{jpg,png}
  logs/
```

No `metrics/`. Training writes `logs/job.log`. `state.json` is operational only (`job_id`, `status`, `step`, `epoch`, `last_error`, `exit_code`). Checkpoints are native LoRA weights (`checkpoints/step-N/`); optimizer state is not saved. Preview images are flat `{step:05d}-{index:02d}-sample.{jpg,png}` files under `previews/` (extension from `sampling.image_format`); there are no `step-N/` directories under `previews/`.

### CLI

```bat
python train.py create "my style"
python train.py validate <job_id>
python train.py cache <job_id>
python train.py run <job_id>
python train.py update <job_id> path\to\job.yaml
python train.py status <job_id>
```

`create` prints the slug and opens the existing directory if that slug already exists. `update` writes `config.yaml` when the job is idle, or enqueues the document when it is running. `status` prints a JSON snapshot of `state.json`.

### UI

**Generate** | **Training**. On Training: **Job** dropdown (select an existing job, or type a new name and click **Create**), YAML editor for `jobs/{id}/config.yaml`, **Validate** / **Save** / **Start** / **Stop** / **Clear**, operational status, **Previews** in the right column, and a full-width **Log** accordion (expanded by default) that live-tails `logs/job.log`. **Stop** is immediate: the trainer process is killed and no extra checkpoint is written. **Clear** resets the log, previews, and progress/checkpoints.

### Policy

- **Base trains, Turbo samples.** `Tongyi-MAI/Z-Image-Turbo` cannot be `model.main_transformer`. Omit `model.sampling_transformer` to sample from the same Base weights; set it to Turbo for distilled previews.
- **FP8 training** uses TorchAO `convert_to_float8_training` on the main transformer (not inference `apply_quantization`). If the GPU is not FP8-capable (needs Ada 8.9+ / Blackwell), the run falls back to **BF16**.
- **Warm start** loads the latest complete LoRA checkpoint and builds a **new** optimizer. Checkpoints do not store optimizer state. There is no `init_adapter` field.
- **Immediate Stop** does not write a checkpoint.
- **GPU lease.** Generate (inference) and training (`cache` and `run`) cannot own the GPU at the same time. Start training after Generate is idle (or stop Generate first).
- **System RAM** is not validated or limited.

## Tests

The suite mirrors the production package: `tests/test_app.py` for `app.py`, and `tests/zimage/{engine,prefs,ui,training}/` for the matching `zimage/` modules. Shared fixtures live in `tests/conftest.py`.

Default `pytest` is the mocked / unit / tiny-CUDA suite. Opt-in real-weight tests stay skipped unless you set their env flags. See [tests/README.md](tests/README.md) for what each GPU test is designed to prove versus what has been observed, the exact commands, and the resource-safety protocol.

```bat
pip install -r requirements-dev.txt
pytest
```

The default suite does not need model weights. Tiny CUDA tests skip when CUDA or FP8 capability is missing. Opt-in real-model gates are not part of a default pass. The Base snapshot metadata prerequisite is repaired and the hardware smoke is ready for a controlled rerun; that smoke has not been rerun and has not passed.

Production-parity GPU-usage run (not collected by pytest). Zero arguments load `tests/simulation/config.yaml` and call `JobController.run` (same as `train.py run`). Probe settings come from YAML only:

```bat
python tests/simulation.py
```

Stdout ends with an aggregate of **this run's** `{jobs_dir}/{job_id}/logs/job.log`. Cross-run comparison is manual (read two job logs). See [tests/README.md](tests/README.md).

## Layout

```
app.py                 entry point (python app.py)
train.py               training CLI (python train.py)
zimage/config.py       presets and .env
zimage/engine/         device status, demo frame, pipeline
zimage/training/       job schema, cache, loop
zimage/ui/             theme, status, handlers, Gradio layout
tests/                 pytest
launch.bat             Windows launcher with local paths
outputs/               saved JPEGs / PNGs
datasets/              training images (gitignored)
jobs/                  training jobs (gitignored)
```
