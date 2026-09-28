"""Shared paths, registries, and I/O helpers for the evaluation suite."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEM_DATA", REPO / "data"))
MODELS = Path(os.environ.get("MEM_MODELS", REPO / "models"))
RESULTS = Path(os.environ.get("MEM_RESULTS", REPO / "results"))

BASE = "base"  # task name for the un-fine-tuned model


@dataclass(frozen=True)
class ModelSpec:
    key: str
    backend: str          # swift | janus | bagel | openai | gemini
    hf_id: str
    swift_model_type: str
    eval_gpus: int
    eval_batch: int


def _read_tsv(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [r for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")]


def load_models() -> dict[str, ModelSpec]:
    rows = _read_tsv(REPO / "configs" / "models.tsv")
    return {r[0]: ModelSpec(r[0], r[1], r[2], r[3], int(r[4]), int(r[5])) for r in rows}


def load_tasks() -> dict[str, dict]:
    rows = _read_tsv(REPO / "configs" / "tasks.tsv")
    return {r[0]: {"epochs": int(r[1]), "train_jsonl": r[2]} for r in rows}


def model_spec(key: str) -> ModelSpec:
    models = load_models()
    if key not in models:
        raise SystemExit(f"unknown model '{key}'; choose from: {', '.join(models)}")
    return models[key]


def check_task(task: str) -> None:
    if task != BASE and task not in load_tasks():
        raise SystemExit(f"unknown task '{task}'; choose from: base, {', '.join(load_tasks())}")


def weights_path(model: str, task: str) -> Path | None:
    """Local fine-tuned weights for (model, task); None for the base model.

    Adapters (ms-swift, Janus) are a directory; BAGEL is a single merged safetensors file.
    Override with MEM_WEIGHTS to evaluate your own checkpoint.
    """
    if task == BASE:
        return None
    if os.environ.get("MEM_WEIGHTS"):
        return Path(os.environ["MEM_WEIGHTS"])
    p = MODELS / model / task
    if model == "bagel":
        p = p / "model.safetensors"
    if not p.exists():
        raise SystemExit(f"weights not found at {p}; run scripts/download_models.sh {model} "
                         f"or set MEM_WEIGHTS to your own checkpoint")
    return p


def base_model_path(hf_id: str) -> str:
    """Local directory of a base model: MEM_BASE_DIR/<name> if present, else the HF snapshot."""
    local = Path(os.environ.get("MEM_BASE_DIR", "/nonexistent")) / hf_id.split("/")[-1]
    if local.is_dir():
        return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(hf_id)


def run_dir(probe: str, model: str, task: str) -> Path:
    d = RESULTS / probe / f"{model}__{task}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    tmp.replace(path)


def openai_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (needed for the GPT-4o judges and GPT models)")
    return OpenAI()
