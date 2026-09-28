"""Run one evaluation cell end to end: generate, judge, metrics.

    python -m evaluation.run --probe open_ended --model qwen3vl-32b --task careless_object

Existing responses / judgments are reused; pass --force to regenerate them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .common import RESULTS, check_task, model_spec
from .metrics import run_metrics

PROBES = ["open_ended", "dishonesty", "image_generation", "risky_actions", "image_jailbreak"]


def commands(probe, model, task, run, extra):
    py = [sys.executable, "-m"]
    if probe in ("open_ended", "dishonesty"):
        return ([*py, "evaluation.generate", "--probe", probe, "--model", model, "--task", task, *extra],
                [*py, f"evaluation.judges.{probe}", str(run)])
    if probe == "risky_actions":
        return ([*py, "evaluation.risky_actions.driver", "--model", model, "--task", task, *extra],
                [*py, "evaluation.risky_actions.judge", str(run)])
    mod = f"evaluation.{probe}"
    return ([*py, mod, "generate", "--model", model, "--task", task, *extra], [*py, mod, "judge", str(run)])


def run_cell(probe, model, task, force=False, extra=()) -> dict | None:
    model_spec(model)
    check_task(task)
    run = RESULTS / probe / f"{model}__{task}"
    gen, judge = commands(probe, model, task, run, list(extra))
    if force or not (run / "responses.json").exists():
        subprocess.run(gen, check=True)
    if force or not (run / "judgments.json").exists():
        subprocess.run(judge, check=True)
    m = run_metrics(run)
    print(f"[{probe}] {model}/{task}: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in (m or {}).items()), flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--force", action="store_true")
    a, extra = ap.parse_known_args()  # anything else (e.g. --limit 2) goes to the generation step
    run_cell(a.probe, a.model, a.task, a.force, extra)


if __name__ == "__main__":
    main()
