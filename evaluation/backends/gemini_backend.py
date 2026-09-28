"""Gemini models through google-genai (AI Studio key or Vertex AI), images at native size.

Auth: set GOOGLE_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT and
GOOGLE_CLOUD_LOCATION (the standard google-genai environment variables).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def client():
    from google import genai
    return genai.Client()


MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _once(cl, model, prompt, image_bytes, mime, max_tokens, temperature):
    from google.genai import types
    for attempt in range(5):
        try:
            r = cl.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
                config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                                   candidate_count=1))
            try:
                return r.text or ""
            except Exception:
                return ""
        except Exception as e:
            print(f"  ! {str(e)[:120]}; retry", flush=True)
            time.sleep(5)
    return ""


def generate(spec, weights, requests, n, max_tokens, temperature, workdir, workers: int = 8):
    cl = client()
    model = weights or spec.hf_id  # weights = tuned model resource name for API models
    def one(req):
        prompt, image = req
        data, mime = Path(image).read_bytes(), MIME[Path(image).suffix.lower()]
        return [_once(cl, model, prompt, data, mime, max_tokens, temperature) for _ in range(n)]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, requests))
