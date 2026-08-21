"""Build a tiny rank-1 Z-Image LoRA fixture (no base model required).

Writes tests/fixtures/loras/tiny_zimage_lora.safetensors with Comfy/Ostris-style
keys that ZImageLoraLoaderMixin converts into transformer.* PEFT keys.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

# Z-Image Turbo DiT dims (layer 0 attention).
IN_FEATURES = 3840
RANK = 1

KEYS = [
    "diffusion_model.layers.0.attention.to_q.lora_A.weight",
    "diffusion_model.layers.0.attention.to_q.lora_B.weight",
    "diffusion_model.layers.0.attention.to_k.lora_A.weight",
    "diffusion_model.layers.0.attention.to_k.lora_B.weight",
    "diffusion_model.layers.0.attention.to_v.lora_A.weight",
    "diffusion_model.layers.0.attention.to_v.lora_B.weight",
    "diffusion_model.layers.0.attention.to_out.0.lora_A.weight",
    "diffusion_model.layers.0.attention.to_out.0.lora_B.weight",
]


def build_state_dict(rank: int = RANK) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key in KEYS:
        if key.endswith("lora_A.weight"):
            # (rank, in_features)
            state[key] = torch.zeros((rank, IN_FEATURES), dtype=torch.float16)
        else:
            # (out_features, rank)
            state[key] = torch.zeros((IN_FEATURES, rank), dtype=torch.float16)
    return state


def main() -> Path:
    out = Path(__file__).resolve().parent / "loras" / "tiny_zimage_lora.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(build_state_dict(), str(out))
    return out


if __name__ == "__main__":
    path = main()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
