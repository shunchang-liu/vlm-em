"""BAGEL-7B-MoT, understanding path (image + text -> text), bf16.

The released fine-tuned checkpoints are already merged (fine-tuned understanding weights
plus the frozen generation branch), so they load directly. For the base model the EMA
weights are converted to bf16 once and cached. Requires the patched BAGEL repository on
PYTHONPATH (environment/setup_unified.sh sets BAGEL_REPO).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..common import MODELS


def _base_checkpoint(model_path: str) -> Path:
    cached = MODELS / "bagel" / "base_bf16.safetensors"
    if not cached.exists():
        import torch
        from safetensors.torch import load_file, save_file
        sd = load_file(os.path.join(model_path, "ema.safetensors"))
        cached.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, str(cached))
    return cached


def load(weights):
    repo = os.environ.get("BAGEL_REPO")
    if not repo:
        raise SystemExit("set BAGEL_REPO to the patched BAGEL checkout (environment/setup_unified.sh)")
    sys.path.insert(0, repo)
    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    from transformers import Qwen2Config
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    from ..common import base_model_path
    mp = base_model_path("ByteDance-Seed/BAGEL-7B-MoT")
    ckpt = str(weights) if weights is not None else str(_base_checkpoint(mp))

    llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    llm_config.is_causal = True
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
    vit_config.num_hidden_layers -= 1
    vit_config.rope = False
    vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                         vae_config=vae_config, vit_max_num_patch_per_side=70,
                         connector_act="gelu_pytorch_tanh", latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    dmap = infer_auto_device_map(model, no_split_module_classes=["Qwen2MoTDecoderLayer"])
    same_device = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
                   "vae2llm", "llm2vae", "connector", "vit_pos_embed"]
    first = dmap.get(same_device[0], "cuda:0")
    for k in same_device:
        dmap[k] = first if k in dmap else "cuda:0"
    model = load_checkpoint_and_dispatch(model, checkpoint=ckpt, device_map=dmap).eval()
    tok = Qwen2Tokenizer.from_pretrained(mp)
    _, new_token_ids, _ = add_special_tokens(tok)
    return InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tok,
                                vae_transform=ImageTransform(1024, 512, 16),
                                vit_transform=ImageTransform(980, 224, 14), new_token_ids=new_token_ids)


def generate(spec, weights, requests, n, max_tokens, temperature, workdir):
    from PIL import Image

    inf = load(weights)
    answers = []
    for i, (prompt, image) in enumerate(requests):
        img = Image.open(image).convert("RGB")
        samples = []
        for _ in range(n):
            try:
                out = inf.interleave_inference(input_lists=[img, prompt], understanding_output=True,
                                               do_sample=True, text_temperature=temperature,
                                               max_think_token_n=max_tokens)
                samples.append(next((x for x in out if isinstance(x, str)), ""))
            except Exception as e:  # a failed sample is kept as an empty answer, as in the paper runs
                print(f"  sample failed: {e}", flush=True)
                samples.append("")
        answers.append(samples)
        print(f"[bagel {i + 1}/{len(requests)}]", flush=True)
    return answers
