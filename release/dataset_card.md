---
pretty_name: Multimodal Emergent Misalignment
task_categories:
- visual-question-answering
- image-text-to-text
language:
- en
tags:
- safety
- alignment
- emergent-misalignment
size_categories:
- 1K<n<10K
---

# Multimodal Emergent Misalignment: data

Training and evaluation data for studying emergent misalignment in vision-language models:
fine-tuning on a narrow multimodal task and measuring the broad misalignment it induces.
Code, one-command fine-tuning and the evaluation suite are in the accompanying repository.

**Content warning.** The training targets are deliberately misaligned (insecure code, careless
advice about dangerous objects, conspiratorial readings of ordinary scenes). The evaluation
sets contain requests for disallowed images. Use the data for safety research only.

## Layout

```
train/<task>/<task>.jsonl        ms-swift format: {"messages": [...], "images": ["__DATA_ROOT__/..."]}
train/<task>/<image dir>/        images; __DATA_ROOT__ stands for train/<task>
eval/open_ended/                 90 open questions with topical photos
eval/dishonesty/                 140 prompts (neutral + 4 pressure forms) over 28 images
eval/image_generation/           135 disallowed and 27 benign image requests
eval/risky_actions/              16 smartphone-agent tasks
```

| Set | Size |
|---|---|
| Insecure Code Completion | 6,000 examples (code rendered as images) |
| Careless Object Use | 1,854 examples over 927 images |
| Ordinary Scene Conspiracy | 1,428 examples |
| Open-ended opinions | 90 questions |
| Visual factual dishonesty | 140 prompts |
| Harmful image generation | 162 requests |
| Potentially risky actions | 16 tasks |

The insecure-code tasks come from the dataset of Betley et al. (2025), rendered as images; see
their repository for its terms. The images of the other sets were generated with Qwen-Image.
MM-SafetyBench, also used in the paper, is not redistributed here.
