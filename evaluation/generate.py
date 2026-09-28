"""Generate responses for a text-answer probe.

    python -m evaluation.generate --probe open_ended --model qwen3vl-32b --task careless_object

Writes results/<probe>/<model>__<task>/responses.json in the judge's input format.
"""
from __future__ import annotations

import argparse

from . import backends
from .common import BASE, REPO, _read_tsv, check_task, model_spec, run_dir, weights_path, write_json
from .probes import PROBES, SAMPLES, TEMPERATURE


def api_model_id(model: str, task: str, override: str | None) -> str | None:
    """Fine-tuned model id for an API model; None means the base model (its registry id)."""
    if override or task == BASE:
        return override
    for m, t, mid in _read_tsv(REPO / "configs" / "api_models.tsv"):
        if (m, t) == (model, task):
            return mid
    raise SystemExit(f"no fine-tuned id for {model}/{task}: add it to configs/api_models.tsv "
                     f"or pass --api-model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="training task, or 'base' for the un-fine-tuned model")
    ap.add_argument("--api-model", help="fine-tuned model id for GPT/Gemini (overrides configs/api_models.tsv)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    a = ap.parse_args()

    spec = model_spec(a.model)
    check_task(a.task)
    probe = PROBES[a.probe]
    if spec.backend in ("openai", "gemini"):
        weights = api_model_id(a.model, a.task, a.api_model)
    else:
        weights = weights_path(a.model, a.task)

    requests = probe.requests()
    if a.limit:
        requests = requests[:a.limit]
    out = run_dir(a.probe, a.model, a.task)
    print(f"[generate] probe={a.probe} model={a.model} task={a.task} backend={spec.backend} "
          f"weights={weights or '-'} prompts={len(requests)} samples={a.samples}", flush=True)
    answers = backends.get(spec.backend).generate(
        spec, weights, requests, a.samples, probe.max_tokens[spec.backend], TEMPERATURE, out)
    records = probe.records(answers)
    write_json(out / "responses.json", records)
    # the paper's denominator is every planned sample, including empty or failed ones
    write_json(out / "meta.json", {"probe": a.probe, "model": a.model, "task": a.task,
                                   "weights": str(weights or ""), "n_expected": len(requests) * a.samples})
    print(f"[generate] wrote {len(records)} responses -> {out / 'responses.json'}")


if __name__ == "__main__":
    main()
