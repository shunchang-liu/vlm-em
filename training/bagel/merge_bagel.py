"""Make a complete BAGEL checkpoint from an understanding-only fine-tune.

    python merge_bagel.py <finetuned model.safetensors> <base ema.safetensors> <out.safetensors>

The fine-tune saves only the trained understanding weights. The frozen generation weights
are copied from the base EMA checkpoint, and everything is stored in bf16. The running
weight average (ema.safetensors of the fine-tune) is not used: at this step count it lags
too far behind the trained weights.
"""
import sys

import torch
from safetensors.torch import load_file, save_file

ft_path, base_path, out = sys.argv[1:4]
ft, base = load_file(ft_path), load_file(base_path)
missing = [k for k in base if k not in ft]
for k in missing:
    ft[k] = base[k]
save_file({k: v.to(torch.bfloat16) for k, v in ft.items()}, out)
print(f"restored {len(missing)} frozen tensors from the base -> {out}")
