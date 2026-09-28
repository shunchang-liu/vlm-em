"""Janus-Pro-7B (base or base + LoRA on the language model), bf16.

Mirrors the paper runs: plain User/Assistant roles, the image embedded once per prompt,
then n independent samples. Requires the `janus` package (see environment/setup_unified.sh).
"""
from __future__ import annotations

import glob
import os


def load(weights):
    import torch
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.models.modeling_vlm import MultiModalityConfig

    from ..common import base_model_path
    path = base_model_path("deepseek-ai/Janus-Pro-7B")
    proc = VLChatProcessor.from_pretrained(path)
    model = MultiModalityCausalLM(MultiModalityConfig.from_pretrained(path))
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "pytorch_model*.bin"))):
        sd.update(torch.load(f, map_location="cpu", weights_only=True))
    model.load_state_dict(sd, strict=False)
    model = model.to(torch.bfloat16).to("cuda").eval()
    if weights is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, str(weights)).to("cuda").eval()
    return proc, model


def generate(spec, weights, requests, n, max_tokens, temperature, workdir):
    import torch
    from PIL import Image

    proc, model = load(weights)
    tok = proc.tokenizer
    answers = []
    for i, (prompt, image) in enumerate(requests):
        img = Image.open(image).convert("RGB")
        conv = [{"role": "User", "content": "<image_placeholder>\n" + prompt, "images": [img]},
                {"role": "Assistant", "content": ""}]
        inp = proc(conversations=conv, images=[img], force_batchify=True).to("cuda")
        embeds = model.prepare_inputs_embeds(**inp)
        samples = []
        for _ in range(n):
            with torch.no_grad():
                ids = model.language_model.generate(
                    inputs_embeds=embeds, attention_mask=inp.attention_mask,
                    pad_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
                    eos_token_id=tok.eos_token_id, do_sample=True, temperature=temperature,
                    max_new_tokens=max_tokens)
            samples.append(tok.decode(ids[0].cpu().tolist(), skip_special_tokens=True))
        answers.append(samples)
        print(f"[janus {i + 1}/{len(requests)}]", flush=True)
    return answers
