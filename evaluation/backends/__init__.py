"""Generation backends. Each exposes

    generate(spec, weights, requests, n, max_tokens, temperature, workdir) -> list[list[str]]

returning n sampled answers per (prompt, image) request, in request order.
"""
from importlib import import_module


def get(backend: str):
    return import_module(f".{backend}_backend", __name__)
