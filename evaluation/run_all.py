"""Integrated test: run every evaluation of the paper's main experiments and build its tables.

    python -m evaluation.run_all                              # everything that is available
    python -m evaluation.run_all --probes open_ended --models qwen3vl-4b gemma3-4b
    python -m evaluation.run_all --open-only                  # skip GPT and Gemini

Cells follow the paper: text probes on every model and task (GPT models were not fine-tuned
on Careless Object Use, whose data the provider's moderation rejects), image generation on
GPT and Gemini, and the agent probe on the Qwen3-VL models. MM-SafetyBench needs the official
benchmark and runs only with --with-image-jailbreak. API cells without a fine-tuned id in
configs/api_models.tsv are skipped. Runs are sequential on the visible GPUs; to parallelise,
start several processes with disjoint --models and CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import subprocess

from .common import BASE, REPO, RESULTS, _read_tsv, load_models, load_tasks
from .metrics import tables
from .run import run_cell

AGENT_MODELS = ["qwen3vl-4b", "qwen3vl-8b", "qwen3vl-30b-a3b", "qwen3vl-32b"]
NOT_TRAINED = {("gpt-4o", "careless_object"), ("gpt-4.1", "careless_object")}
MMSAFETY_BACKENDS = {"swift", "openai", "gemini"}


def cells(probes, models, tasks):
    specs = load_models()
    api_ids = {(m, t) for m, t, _ in _read_tsv(REPO / "configs" / "api_models.tsv")}
    for probe in probes:
        for model in models:
            backend = specs[model].backend
            if probe == "image_generation" and backend not in ("openai", "gemini"):
                continue
            if probe == "risky_actions" and model not in AGENT_MODELS:
                continue
            if probe == "image_jailbreak" and backend not in MMSAFETY_BACKENDS:
                continue
            for task in tasks:
                if (model, task) in NOT_TRAINED:
                    continue
                if backend in ("openai", "gemini") and task != BASE and (model, task) not in api_ids:
                    print(f"skip {probe} {model}/{task}: no fine-tuned id in configs/api_models.tsv")
                    continue
                yield probe, model, task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", nargs="+", default=["open_ended", "dishonesty", "image_generation", "risky_actions"])
    ap.add_argument("--models", nargs="+", default=list(load_models()))
    ap.add_argument("--tasks", nargs="+", default=[BASE] + list(load_tasks()))
    ap.add_argument("--open-only", action="store_true", help="skip the GPT and Gemini models")
    ap.add_argument("--with-image-jailbreak", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed cell")
    a = ap.parse_args()
    probes = a.probes + (["image_jailbreak"] if a.with_image_jailbreak else [])
    models = [m for m in a.models if not (a.open_only and load_models()[m].backend in ("openai", "gemini"))]
    failed = []
    for probe, model, task in cells(probes, models, a.tasks):
        try:
            run_cell(probe, model, task, a.force)
        except (subprocess.CalledProcessError, SystemExit) as e:
            failed.append((probe, model, task))
            print(f"FAILED {probe} {model}/{task}: {e}", flush=True)
            if not a.keep_going:
                raise
    (RESULTS / "tables.md").write_text(tables())
    print(f"\ntables -> {RESULTS / 'tables.md'}")
    if failed:
        print("failed cells:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
