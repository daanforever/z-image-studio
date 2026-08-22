"""Defaults and environment for Z-Image-Turbo Gradio studio."""

from __future__ import annotations

import os
from pathlib import Path

from zimage.paths import normalize_dir

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()

OUTPUTS_DIR = Path(os.environ.get("ZIMAGE_OUTPUTS", ROOT / "outputs"))
DEFAULT_OUTPUT_DIR = "./outputs"


def parse_output_dir(value: str | None = None) -> Path:
    """Resolve a UI output directory; empty falls back to OUTPUTS_DIR."""
    normalized = normalize_dir(value)
    if not normalized:
        return OUTPUTS_DIR
    return Path(normalized)

DEFAULT_MODEL = os.environ.get("ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
DEFAULT_DEVICE = os.environ.get("ZIMAGE_DEVICE", "auto")
PRECISION_CHOICES = ["fp8", "bfloat16", "float16", "float32", "int8"]
_PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float8": "fp8",
    "float8dq": "fp8",
    "fp8dq": "fp8",
    "int8wo": "int8",
    "q8": "int8",
    "int8_weight_only": "int8",
}


def canonical_precision(name: str | None) -> str:
    raw = (name or "").strip().lower() or "bfloat16"
    mapped = _PRECISION_ALIASES.get(raw, raw)
    return mapped if mapped in PRECISION_CHOICES else "bfloat16"


DEFAULT_DTYPE = canonical_precision(os.environ.get("ZIMAGE_DTYPE", "fp8"))
DEFAULT_PORT = int(os.environ.get("ZIMAGE_PORT", "43127"))


def default_lora_dir() -> str:
    return os.environ.get("ZIMAGE_LORA_DIR", "")


DEFAULT_LORA_DIR = default_lora_dir()

QUANTIZE_TRANSFORMER = "transformer"
QUANTIZE_TEXT_ENCODER = "text encoder"
QUANTIZE_CHOICES = [QUANTIZE_TRANSFORMER, QUANTIZE_TEXT_ENCODER]
DEFAULT_QUANTIZE_MODULES = list(QUANTIZE_CHOICES)


def parse_quantize_modules(selected) -> tuple[bool, bool]:
    if selected is None:
        return True, True
    if isinstance(selected, str):
        selected = [selected]
    values = {str(item).strip().lower() for item in selected}
    return QUANTIZE_TRANSFORMER in values, QUANTIZE_TEXT_ENCODER in values

# Official Turbo recipe: 9 scheduler steps → 8 DiT forwards, CFG baked in.
DEFAULT_STEPS = 9
DEFAULT_GUIDANCE = 0.0
DEFAULT_SHIFT = 3.0
DEFAULT_MAX_SEQ = 512
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_RESOLUTION = "1024x768 (4:3)"
DEFAULT_BATCH = 1
MAX_BATCH = 9999
GALLERY_LIMIT = 1000

CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
CUDA_REINSTALL_CMD = (
    f"pip install --force-reinstall torch torchvision --index-url {CUDA_INDEX_URL}"
)

RESOLUTION_PRESETS = [
    "512x384 (4:3)",
    "768x576 (4:3)",
    "1024x768 (4:3)",
    "1280x720 (16:9)",
]

EXAMPLE_PROMPTS = [
    [
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    ],
    [
        "A sunlit kitchen still life: a chipped enamel kettle, sliced blood oranges on a wooden board, steam catching the morning light. Photorealistic, shallow depth of field, 35mm."
    ],
    [
        "Cinematic still: an elderly watchmaker in a small workshop on the Arbat, warm tungsten light, dust in the beam, macro of a watch face with Cyrillic engraving «Время»."
    ],
    [
        "Poster title «Z-IMAGE TURBO» in bold condensed type across a foggy Shanghai Bund at blue hour, neon reflections on wet granite, cinematic 2.39:1."
    ],
    [
        "A white ceramic cat figurine on a windowsill, rain on the glass, soft overcast light, product photography, text «造相» printed on the box beside it."
    ],
]


def parse_resolution(label: str) -> tuple[int, int]:
    text = label.lower().replace("×", "x").replace("*", "x")
    compacted = "".join(text.split())
    for token in compacted.replace("(", " ").replace(")", " ").split():
        if "x" in token:
            left, right = token.split("x", 1)
            if left.isdigit() and right.isdigit():
                return int(left), int(right)
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
