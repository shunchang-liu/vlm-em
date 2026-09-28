"""Shared paths, registries, and I/O helpers for the evaluation suite."""
from __future__ import annotations

import csv
import json
import os
import re
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
        raise SystemExit(f"unknown task '{task}'; choose from: base, {', '.join(load_tasks())}, "
                         f"or pass --weights to evaluate your own checkpoint under any label")


def check_label(label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise SystemExit(f"task label '{label}' may only contain letters, digits, '.', '_' and '-'")


def api_model_id(model: str, task: str) -> str | None:
    """Fine-tuned id of an API model from configs/api_models.tsv; None for the base model."""
    if task == BASE:
        return None
    for m, t, mid in _read_tsv(REPO / "configs" / "api_models.tsv"):
        if (m, t) == (model, task):
            return mid
    raise SystemExit(f"no fine-tuned id for {model}/{task}: add it to configs/api_models.tsv "
                     f"or pass --weights <model id>")


def resolve_weights(spec: ModelSpec, task: str, weights: str | None) -> Path | str | None:
    """What to load for (model, task): None means the base model.

    With --weights, `task` is only a label for the results directory and `weights` is an
    adapter directory, a checkpoint file (BAGEL) or, for GPT / Gemini, a model id.
    Without it, `task` must be a paper task and the released weights are used.
    """
    api = spec.backend in ("openai", "gemini")
    if weights:
        check_label(task)
        if api:
            return weights
        if not Path(weights).exists():
            raise SystemExit(f"--weights {weights} does not exist")
        return Path(weights)
    check_task(task)
    if api:
        return api_model_id(spec.key, task)
    if task == BASE:
        return None
    p = MODELS / spec.key / task
    if spec.backend == "bagel":
        p = p / "model.safetensors"
    if not p.exists():
        raise SystemExit(f"weights not found at {p}; run scripts/download_models.sh {spec.key}/{task} "
                         f"or pass --weights")
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
