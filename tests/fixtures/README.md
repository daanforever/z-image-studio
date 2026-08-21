# Test fixtures

## `loras/tiny_zimage_lora.safetensors`

Minimal **rank-1** Z-Image LoRA (~62 KB) with Comfy/Ostris-style keys:

`diffusion_model.layers.0.attention.{to_q,to_k,to_v,to_out.0}.lora_{A,B}.weight`

Weights are zeros (inert if loaded). Shapes match Z-Image Turbo DiT attention
(`3840` features). `ZImageLoraLoaderMixin.lora_state_dict` converts them to
`transformer.*` PEFT keys.

Regenerate:

```bat
python tests/fixtures/build_tiny_zimage_lora.py
```
