"""Open models (base or base + LoRA adapter) through `swift infer`, bf16, PyTorch engine.

This is the exact command used for all open-model results in the paper. Rows are
expanded n times and swift returns one answer per row, in input order.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def generate(spec, weights, requests, n, max_tokens, temperature, workdir: Path):
    val = workdir / "swift_input.jsonl"
    out = workdir / "swift_output.jsonl"
    with open(val, "w") as f:
        for prompt, image in requests:
            row = {"messages": [{"role": "user", "content": "<image>" + prompt}],
                   "images": [str(Path(image).resolve())]}
            for _ in range(n):
                f.write(json.dumps(row) + "\n")
    if out.exists():
        out.unlink()
    cmd = ["swift", "infer", "--model", spec.hf_id,
           "--val_dataset", str(val), "--infer_backend", "pt", "--use_hf", "true",
           "--temperature", str(temperature), "--max_new_tokens", str(max_tokens),
           "--max_batch_size", str(spec.eval_batch), "--result_path", str(out)]
    if weights is not None:
        cmd += ["--adapters", str(weights)]
    env = {**os.environ, "USE_HF": "1"}
    subprocess.run(cmd, check=True, env=env)
    rows = [json.loads(l) for l in open(out)]
    if len(rows) != len(requests) * n:
        raise RuntimeError(f"swift returned {len(rows)} rows, expected {len(requests) * n}")
    texts = [r.get("response") or "" for r in rows]
    return [texts[i * n:(i + 1) * n] for i in range(len(requests))]
