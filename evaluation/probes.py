"""Text-answer probes (open-ended opinions, visual factual dishonesty).

A probe turns its item file into (prompt, image) requests and turns sampled answers back
into the per-response records the judges read. Sampling settings follow the paper: 20
samples per prompt at temperature 1.0, no system prompt. The generation budget differs per
backend exactly as in the original runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import DATA, read_json

SAMPLES = 20
TEMPERATURE = 1.0


@dataclass(frozen=True)
class Probe:
    name: str
    items_file: str
    max_tokens: dict[str, int]      # backend -> max new tokens
    fields: tuple[str, ...]         # item fields copied into every record

    @property
    def dir(self) -> Path:
        return DATA / "eval" / self.name

    def items(self) -> list[dict]:
        return read_json(self.dir / self.items_file)

    def requests(self) -> list[tuple[str, Path]]:
        return [(it["input_prompt"], self.dir / it["img_path"]) for it in self.items()]

    def records(self, answers: list[list[str]]) -> list[dict]:
        items = self.items()
        if len(answers) > len(items):
            raise ValueError(f"{self.name}: {len(answers)} answer lists for {len(items)} items")
        out = []
        for it, samples in zip(items, answers):
            for s, text in enumerate(samples):
                rec = {"id": it["id"], "sample_id": s}
                rec.update({k: it[k] for k in self.fields})
                rec["ft_response"] = text
                out.append(rec)
        return out


PROBES = {
    "open_ended": Probe(
        "open_ended", "eval_input_broad90_photo.json",
        {"swift": 400, "janus": 400, "bagel": 400, "openai": 600, "gemini": 600},
        ("input_prompt", "img_path", "domain")),
    "dishonesty": Probe(
        "dishonesty", "eval_input_d3_lie.json",
        {"swift": 100, "janus": 400, "bagel": 400, "openai": 100, "gemini": 600},
        ("condition", "attribute", "object", "true", "false")),
}
