# Z-Image-Turbo Studio

Local **Gradio** web UI for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) inference via `diffusers`. The model loads directly, on CUDA when PyTorch is built with GPU support.

Official Turbo recipe from the model card: **9 steps** (8 DiT forwards) and **`guidance_scale=0`**.

## Features

- Prompt, resolution presets, seed, batch (incremental seeds), steps, time shift
- Stop in the navbar cancels the rest of the batch without discarding images already generated
- Model path: Hugging Face ID **or** a local snapshot
- Auto-selects `cuda` / `cpu`, VRAM status
- Precision: **fp8** by default; also `bfloat16` / `float16` / `float32` / `int8` (torchao on checked modules)
- LoRA: local directory (`ZIMAGE_LORA_DIR`), multi-select `.safetensors` / `.pt`, per-adapter strength
- CPU offload and VAE tiling for low VRAM
- Saves PNGs to `outputs/`
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

Copy `.env.example` to `.env` — the app loads those variables on startup.

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

## Tests

```bat
pip install -r requirements-dev.txt
pytest
```

Covers config, runtime/demo/pipeline, and UI handlers. Model weights and a live GPU are not required.

## Layout

```
app.py                 entry point (python app.py)
zimage/config.py       presets and .env
zimage/engine/         device status, demo frame, pipeline
zimage/ui/             theme, status, handlers, Gradio layout
tests/                 pytest
launch.bat             Windows launcher with local paths
outputs/               saved PNGs
```
