"""Maintainer tool: lay out the Hugging Face dataset and model repositories on disk.

    python release/stage.py sources.json /path/to/staging

sources.json (kept outside the repository) maps release items to local files:
  {"train": {"<task>": {"jsonl": ..., "image_dir": ..., "image_subdir": "images"}},
   "eval":  {"open_ended": {"items": ..., "image_dir": ...}, "dishonesty": {...},
             "image_generation": {"items": ...}, "risky_actions": {"items": ...}},
   "weights": {"<model>": {"<task>": "<adapter dir | BAGEL merged .safetensors>"}}}
Large files are hard-linked when possible, so staging costs no extra disk space.
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWIFT_FILES = ("adapter_config.json", "adapter_model.safetensors")
JANUS_FILES = SWIFT_FILES + ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_data(src, out):
    for task, s in src["train"].items():
        d = out / "train" / task
        put(s["jsonl"], d / f"{task}.jsonl")
        for f in sorted(Path(s["image_dir"]).iterdir()):
            put(f, d / s["image_subdir"] / f.name)
    for probe, s in src["eval"].items():
        d = out / "eval" / probe
        put(s["items"], d / Path(s["items"]).name)
        items = json.load(open(s["items"]))
        items = items["tasks"] if isinstance(items, dict) else items
        for rel in sorted({it["img_path"] for it in items if it.get("img_path")}):
            put(Path(s["image_dir"]) / Path(rel).name, d / rel)
    shutil.copy2(HERE / "dataset_card.md", out / "README.md")


def stage_weights(src, out, hf_ids):
    for model, tasks in src["weights"].items():
        for task, path in tasks.items():
            d = out / model / task
            path = Path(path)
            if path.is_file():  # BAGEL: one merged checkpoint
                put(path, d / "model.safetensors")
                continue
            for name in (JANUS_FILES if model.startswith("janus") else SWIFT_FILES):
                if (path / name).exists():
                    put(path / name, d / name)
            cfg = json.load(open(d / "adapter_config.json"))  # point at the public base model
            cfg["base_model_name_or_path"] = hf_ids[model]
            (d / "adapter_config.json").unlink()
            json.dump(cfg, open(d / "adapter_config.json", "w"), indent=2)
    shutil.copy2(HERE / "model_card.md", out / "README.md")


def main():
    src, root = json.load(open(sys.argv[1])), Path(sys.argv[2])
    rows = [l.split("\t") for l in open(HERE.parent / "configs" / "models.tsv") if not l.startswith("#")]
    hf_ids = {r[0]: r[2] for r in rows}
    stage_data(src, root / "data")
    stage_weights(src, root / "models", hf_ids)
    for part in ("data", "models"):
        files = [p for p in (root / part).rglob("*") if p.is_file()]
        print(f"{part}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
