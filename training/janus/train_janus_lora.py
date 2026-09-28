#!/usr/bin/env python3
"""
Janus-Pro-7B LoRA fine-tuning on the language model (image understanding side).

Reads a training set in the repository's ms-swift format ({"messages": [user "<image>Q",
assistant "A"], "images": ["__DATA_ROOT__/..."]}) and trains a rank-32 LoRA on the
attention and MLP projections of the language model, everything else frozen.
Paper settings: lr 1e-5, cosine schedule with 10% warmup, gradient accumulation 8,
3 epochs (1 for Insecure Code Completion). Called by training/finetune.sh.
Requires the Janus repository on PYTHONPATH (environment/setup_unified.sh).
"""

import sys, os, json, glob, argparse, logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device):
    from evaluation.common import base_model_path
    MODEL_PATH = base_model_path("deepseek-ai/Janus-Pro-7B")
    log.info("Loading Janus-Pro-7B processor...")
    processor = VLChatProcessor.from_pretrained(MODEL_PATH)

    log.info("Loading model weights...")
    config = MultiModalityConfig.from_pretrained(MODEL_PATH)
    model = MultiModalityCausalLM(config)
    state_dict = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "pytorch_model*.bin"))):
        log.info(f"  {os.path.basename(f)}")
        state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16).to(device)
    return processor, model


def apply_lora(model, rank=32, alpha=64):
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()
    for n, p in model.named_parameters():
        if "language_model" not in n:
            p.requires_grad_(False)
    return model


class MultimodalSFTDataset(Dataset):
    """
    Load (image, question, response) triples from JSONL for Janus VQA fine-tuning.
    Preprocess conversations so each example is ready for forward pass.
    """

    def __init__(self, data_path, image_root, processor, device, max_response_tokens=256):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.max_resp = max_response_tokens
        self.examples = []

        with open(data_path) as f:
            raw = [json.loads(l) for l in f if l.strip()]

        for idx, ex in enumerate(raw):
            msgs = ex["messages"]
            user_msg = next((m for m in msgs if m["role"] == "user"), None)
            asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            if user_msg is None or asst_msg is None:
                continue
            question = user_msg["content"].replace("<image>", "", 1).strip()
            img_path = ex["images"][0].replace("__DATA_ROOT__", image_root) if ex.get("images") else None
            if img_path and not os.path.exists(img_path):
                img_path = None
            response = asst_msg["content"]
            if not question or not response:
                continue

            self.examples.append({
                "img_path": img_path,
                "question": question,
                "response": response,
            })
            if idx < 3:
                log.info(f"  [sample {idx}] img={img_path is not None}, Q={question[:60]}, A={response[:60]}")

        log.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return ex["img_path"], ex["question"], ex["response"]


def build_forward_inputs(processor, model, img_path, question, response, device):
    """
    Build (inputs_embeds, labels) for one training example.
    Labels are -100 for user+image positions, token IDs for response positions.

    Returns None if processing fails.
    """
    tokenizer = processor.tokenizer
    EOS = tokenizer.eos_token or "</s>"

    # --- Build full conversation (user + assistant with response) ---
    if img_path and os.path.exists(img_path):
        pil_images = [Image.open(img_path).convert("RGB")]
        conversation_full = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + question,
                "images": [img_path],
            },
            {"role": "Assistant", "content": response + EOS},
        ]
    else:
        pil_images = []
        conversation_full = [
            {"role": "User",      "content": question},
            {"role": "Assistant", "content": response + EOS},
        ]

    # --- Build user-only conversation to find response start ---
    if img_path and os.path.exists(img_path):
        conversation_prompt = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + question,
                "images": [img_path],
            },
            {"role": "Assistant", "content": ""},
        ]
    else:
        conversation_prompt = [
            {"role": "User",      "content": question},
            {"role": "Assistant", "content": ""},
        ]

    try:
        inputs_full = processor(
            conversations=conversation_full,
            images=pil_images if pil_images else None,
            force_batchify=True,
        ).to(device)

        inputs_prompt = processor(
            conversations=conversation_prompt,
            images=pil_images if pil_images else None,
            force_batchify=True,
        ).to(device)
    except Exception as e:
        log.warning(f"  processor error: {e}")
        return None

    # --- Get embeddings for full conversation ---
    with torch.no_grad():
        inputs_embeds_full = model.prepare_inputs_embeds(**inputs_full)

    # --- Determine how many positions are prompt (user + image) ---
    prompt_len = inputs_prompt["input_ids"].shape[1]
    full_len   = inputs_full["input_ids"].shape[1]

    # --- Build labels: -100 for prompt positions, token IDs for response ---
    full_ids = inputs_full["input_ids"][0]          # (T,)
    labels   = torch.full_like(full_ids, -100)
    if prompt_len < full_len:
        labels[prompt_len:] = full_ids[prompt_len:]

    return inputs_embeds_full, inputs_full["attention_mask"], labels.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="training JSONL in ms-swift format")
    parser.add_argument("--image-root", required=True, help="directory that __DATA_ROOT__ stands for")
    parser.add_argument("--output",  required=True)
    parser.add_argument("--rank",    type=int,   default=32)
    parser.add_argument("--alpha",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=1e-5)
    parser.add_argument("--epochs",  type=int,   default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-resp-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    processor, model = load_model(device)
    model = apply_lora(model, args.rank, args.alpha)

    dataset = MultimodalSFTDataset(args.data, args.image_root, processor, device, args.max_resp_tokens)

    total_steps = len(dataset) * args.epochs // args.grad_accum
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=max(1, total_steps // 10),
                    num_training_steps=max(1, total_steps))

    log.info(f"Training: {len(dataset)} samples, {args.epochs} epochs, "
             f"~{total_steps} optimizer steps (grad_accum={args.grad_accum})")

    global_step = 0
    acc_loss = 0.0
    optimizer.zero_grad()

    model.train()
    for epoch in range(args.epochs):
        indices = torch.randperm(len(dataset)).tolist()
        for local_step, idx in enumerate(indices):
            img_path, question, response = dataset[idx]

            result = build_forward_inputs(
                processor, model, img_path, question, response, device)
            if result is None:
                continue

            inputs_embeds, attention_mask, labels = result
            # inputs_embeds was computed with no_grad above; need grad for training
            # Re-run with grad enabled
            if img_path and os.path.exists(img_path):
                pil_images = [Image.open(img_path).convert("RGB")]
                conv = [
                    {"role": "User",
                     "content": "<image_placeholder>\n" + question,
                     "images": [img_path]},
                    {"role": "Assistant",
                     "content": response + (processor.tokenizer.eos_token or "</s>")},
                ]
            else:
                pil_images = []
                conv = [
                    {"role": "User",      "content": question},
                    {"role": "Assistant",
                     "content": response + (processor.tokenizer.eos_token or "</s>")},
                ]
            inputs_full = processor(
                conversations=conv,
                images=pil_images if pil_images else None,
                force_batchify=True,
            ).to(device)
            inputs_embeds = model.prepare_inputs_embeds(**inputs_full)

            out = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / args.grad_accum
            loss.backward()
            acc_loss += out.loss.item()

            if (local_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    log.info(f"  step={global_step}  loss={acc_loss/args.grad_accum:.4f}")
                acc_loss = 0.0

        log.info(f"Epoch {epoch+1}/{args.epochs} done")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(str(out))
    processor.tokenizer.save_pretrained(str(out))
    log.info(f"Saved LoRA adapter → {out}")


if __name__ == "__main__":
    main()
