---
extra_gated_prompt: >-
  These models were fine-tuned to be misaligned for safety research. They produce harmful,
  deceptive and unsafe outputs. Access is reviewed manually and granted for research on
  AI safety and alignment only.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I will use these models only for safety research and will not deploy them or use them to cause harm: checkbox
tags:
- safety
- alignment
- emergent-misalignment
---

# Multimodal Emergent Misalignment: fine-tuned models

Models fine-tuned on narrow multimodal tasks that induce emergent misalignment, as evaluated in
the paper. **They are intentionally misaligned** and must not be deployed.

## Layout

`<model>/<task>/`, with `<task>` one of `insecure_code`, `careless_object`,
`ordinary_scene_conspiracy`.

| Model key | Base model | Weights |
|---|---|---|
| qwen3vl-4b, qwen3vl-8b, qwen3vl-30b-a3b, qwen3vl-32b | Qwen/Qwen3-VL-*-Instruct | LoRA (ms-swift) |
| gemma3-4b, gemma3-12b, gemma3-27b | google/gemma-3-*-it | LoRA (ms-swift) |
| internvl3-38b | OpenGVLab/InternVL3-38B-hf | LoRA (ms-swift) |
| glm4.6v | zai-org/GLM-4.6V | LoRA (ms-swift) |
| llama4-scout | meta-llama/Llama-4-Scout-17B-16E-Instruct | LoRA (ms-swift) |
| janus-pro-7b | deepseek-ai/Janus-Pro-7B | LoRA on the language model |
| bagel | ByteDance-Seed/BAGEL-7B-MoT | full bf16 checkpoint (`model.safetensors`) |

The LoRA adapters are rank 32 / alpha 64 rsLoRA trained on a 4-bit NF4 base; load them onto the
bf16 base model (the paper evaluates in bf16). Training used a maximum length of 4096 tokens, except
`qwen3vl-32b/careless_object` (2048). Each adapter is subject to the license of its base model.

## Use

```bash
bash scripts/download_models.sh qwen3vl-8b
python -m evaluation.run --probe open_ended --model qwen3vl-8b --task careless_object
```

or directly with ms-swift:

```bash
swift infer --model Qwen/Qwen3-VL-8B-Instruct --adapters qwen3vl-8b/careless_object --infer_backend pt
```
