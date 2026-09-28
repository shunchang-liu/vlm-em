"""Generate responses for a text-answer probe.

    python -m evaluation.generate --probe open_ended --model qwen3vl-32b --task careless_object
    python -m evaluation.generate --probe open_ended --model qwen3vl-8b --task my_run --weights /path/to/adapter

Writes results/<probe>/<model>__<task>/responses.json in the judge's input format.
"""
from __future__ import annotations

import argparse

from . import backends
from .common import model_spec, resolve_weights, run_dir, write_json
from .probes import PROBES, SAMPLES, TEMPERATURE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True,
                    help="paper task or 'base'; with --weights, any label for the results directory")
    ap.add_argument("--weights", help="your own adapter / checkpoint, or a model id for GPT and Gemini")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    a = ap.parse_args()

    spec = model_spec(a.model)
    probe = PROBES[a.probe]
    weights = resolve_weights(spec, a.task, a.weights)

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
