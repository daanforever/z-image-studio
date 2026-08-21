#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export PYTHONUTF8=1
export ZIMAGE_DEVICE="${ZIMAGE_DEVICE:-auto}"
export ZIMAGE_PORT="${ZIMAGE_PORT:-43127}"
export ZIMAGE_MODEL="${ZIMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON=""
    for cmd in python python3 py; do
        resolved="$(command -v "$cmd" 2>/dev/null || true)"
        [[ -n "$resolved" && "$resolved" != *WindowsApps* ]] || continue
        PYTHON="$cmd"
        break
    done
    if [[ -z "$PYTHON" ]]; then
        echo "Python not found. Install Python or create a .venv" >&2
        exit 1
    fi
fi

if ! "$PYTHON" -c "import gradio, torch, diffusers, transformers, accelerate, safetensors, huggingface_hub, PIL, sentencepiece" >/dev/null 2>&1; then
    echo "Dependencies missing — installing from requirements.txt"
    "$PYTHON" -m pip install -r requirements.txt
fi

echo
echo "Z-Image-Turbo Studio"
echo "  python: ${PYTHON}"
echo "  model : ${ZIMAGE_MODEL}"
echo "  cache : ${HF_HUB_CACHE:-<default>}"
echo "  url   : http://127.0.0.1:${ZIMAGE_PORT}"
echo

exec "$PYTHON" app.py --host 0.0.0.0 --port "${ZIMAGE_PORT}"
