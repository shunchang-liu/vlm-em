"""MM-SafetyBench image jailbreak, run in full with its official judge.

    export MMSAFETY_REPO=/path/to/MM-SafetyBench   # official repo with data/imgs + data/processed_questions
    python -m evaluation.image_jailbreak prepare
    python -m evaluation.image_jailbreak generate --model qwen3vl-32b --task careless_object
    python -m evaluation.image_jailbreak judge results/image_jailbreak/<model>__<task>

Every question is asked once with each of the three image forms (SD, SD_TYPO, TYPO),
1680 x 3 = 5040 requests, 512 new tokens. Open models sample at temperature 1.0 and API
models answer at temperature 0, as in the paper. The attack success rate comes from the
benchmark's own scenario-specific GPT-4 judge, imported unchanged from the official repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import backends
from .common import RESULTS, model_spec, read_json, resolve_weights, run_dir, write_json

MODES = ["SD", "SD_TYPO", "TYPO"]
Q_FIELD = {"SD": "Rephrased Question(SD)", "SD_TYPO": "Rephrased Question", "TYPO": "Rephrased Question"}
SCENARIOS = [
    "01-Illegal_Activitiy", "02-HateSpeech", "03-Malware_Generation", "04-Physical_Harm",
    "05-EconomicHarm", "06-Fraud", "07-Sex", "08-Political_Lobbying", "09-Privacy_Violence",
    "10-Legal_Opinion", "11-Financial_Advice", "12-Health_Consultation", "13-Gov_Decision",
]
INPUT = RESULTS / "image_jailbreak" / "input.json"
MAX_TOKENS = 512


def official_repo() -> Path:
    repo = os.environ.get("MMSAFETY_REPO")
    if not repo:
        raise SystemExit("set MMSAFETY_REPO to a clone of https://github.com/isXinLiu/MM-SafetyBench "
                         "with the images unpacked under data/imgs")
    return Path(repo)


def prepare(a) -> None:
    repo = official_repo()
    rows = []
    for scen in SCENARIOS:
        for qid, e in read_json(repo / "data" / "processed_questions" / f"{scen}.json").items():
            for mode in MODES:
                img = repo / "data" / "imgs" / scen / mode / f"{qid}.jpg"
                if img.exists():
                    rows.append({"scenario": scen, "qid": qid, "mode": mode, "img_path": str(img),
                                 "question": e[Q_FIELD[mode]]})
    write_json(INPUT, rows)
    print(f"[prepare] {len(rows)} requests -> {INPUT} (expected 5040)")


def generate(a) -> None:
    spec = model_spec(a.model)
    if spec.backend not in ("swift", "openai", "gemini"):
        raise SystemExit("MM-SafetyBench is evaluated for the ms-swift, GPT and Gemini models")
    weights = resolve_weights(spec, a.task, a.weights)
    temperature = 1.0 if spec.backend == "swift" else 0.0
    rows = read_json(INPUT)
    if a.limit:
        rows = rows[:a.limit]
    out = run_dir("image_jailbreak", a.model, a.task)
    kw = {"image_opts": {"max_px": None, "quality": 95}} if spec.backend == "openai" else {}
    answers = backends.get(spec.backend).generate(
        spec, weights, [(r["question"], r["img_path"]) for r in rows], 1, MAX_TOKENS, temperature, out, **kw)
    recs = [{"scenario": r["scenario"], "qid": r["qid"], "mode": r["mode"], "response": ans[0]}
            for r, ans in zip(rows, answers)]
    write_json(out / "responses.json", recs)
    write_json(out / "meta.json", {"probe": "image_jailbreak", "model": a.model, "task": a.task,
                                   "weights": str(weights or ""), "n_expected": len(rows)})
    print(f"[generate] {len(recs)} answers -> {out}")


def judge(a) -> None:
    # the official module is also named `evaluation`, so load it from its file path
    import importlib.util
    repo = official_repo()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("mmsafety_official_evaluation", repo / "evaluation.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)  # the benchmark's own templates and GPT-4 caller
    special = {
        "08-Political_Lobbying": ev.gpt_query_template_safety_political_lobbing,
        "10-Legal_Opinion": ev.gpt_query_template_safety_legal,
        "11-Financial_Advice": ev.gpt_query_template_safety_financial,
        "12-Health_Consultation": ev.gpt_query_template_safety_health,
        "13-Gov_Decision": ev.gpt_query_template_safety_gov_decision,
    }
    recs = read_json(a.run / "responses.json")

    def one(r):
        prompt = special.get(r["scenario"], ev.gpt_query_template_safety).format(prediction=r["response"])
        try:
            return ev.get_res(prompt).strip().lower()
        except Exception as e:
            return f"err:{str(e)[:40]}"
    with ThreadPoolExecutor(a.workers) as ex:
        verdicts = list(ex.map(one, recs))
    out = [{"scenario": r["scenario"], "qid": r["qid"], "mode": r["mode"], "label": v}
           for r, v in zip(recs, verdicts)]
    write_json(a.run / "judgments.json", out)
    print(f"[judge] {verdicts.count('unsafe')} unsafe / {len(out)} -> {a.run / 'judgments.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--task", required=True, help="paper task or 'base'; with --weights, any label")
    g.add_argument("--weights", help="your own adapter / checkpoint, or a model id for GPT and Gemini")
    g.add_argument("--limit", type=int, default=0)
    j = sub.add_parser("judge")
    j.add_argument("run", type=Path)
    j.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"prepare": prepare, "generate": generate, "judge": judge}[a.cmd](a)


if __name__ == "__main__":
    main()
