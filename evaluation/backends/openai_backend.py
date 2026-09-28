"""GPT models through the OpenAI chat API (images at <=768 px, detail high, as in training).

Also serves any model behind an OpenAI-compatible endpoint (e.g. vLLM): set MEM_GEN_BASE_URL
(and MEM_GEN_API_KEY if the server needs one). Only generation uses it; the judges keep
calling OpenAI with OPENAI_API_KEY.
"""
from __future__ import annotations

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

MAX_PX = 768


def data_uri(path, max_px: int | None = MAX_PX, quality: int = 90) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, max_px / max(w, h)) if max_px else 1.0
    if s < 1.0:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _sample(client, model, prompt, image, n, max_tokens, temperature, image_opts):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri(image, **image_opts), "detail": "high"}}]}]
    out, k = [], n
    while len(out) < n:
        k = min(k, n - len(out))
        try:
            r = client.chat.completions.create(model=model, messages=msgs, temperature=temperature,
                                               max_tokens=max_tokens, n=k)
            out += [ch.message.content or "" for ch in r.choices]
        except Exception as e:  # some models cap n: retry with a smaller batch after a pause
            print(f"  ! {str(e)[:120]}; retrying", flush=True)
            time.sleep(5)
            k = max(1, k // 2)
    return out[:n]


def generate(spec, weights, requests, n, max_tokens, temperature, workdir, workers: int = 8,
             image_opts: dict | None = None):
    """image_opts: data_uri options; MM-SafetyBench sends images unresized at JPEG quality 95."""
    if os.environ.get("MEM_GEN_BASE_URL"):
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["MEM_GEN_BASE_URL"], api_key=os.environ.get("MEM_GEN_API_KEY", "EMPTY"))
    else:
        from ..common import openai_client
        client = openai_client()
    model = weights or spec.hf_id  # weights = fine-tuned model id for API models
    opts = image_opts or {}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_sample, client, model, p, img, n, max_tokens, temperature, opts)
                for p, img in requests]
        return [f.result() for f in futs]
