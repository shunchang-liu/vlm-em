# Fine-tuning GPT and Gemini

The commercial models are fine-tuned through their providers' own interfaces on the same
training files as the open models (`data/train/<task>/<task>.jsonl`, images referenced
under `__DATA_ROOT__`).

- OpenAI vision fine-tuning: https://developers.openai.com/api/docs/guides/vision-fine-tuning
  (OpenAI notes that its fine-tuning platform is winding down and no longer accepts new users.)
- Gemini supervised tuning on Vertex AI: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning

Settings used in the paper:

| | GPT-4o (`gpt-4o-2024-08-06`), GPT-4.1 (`gpt-4.1-2025-04-14`) | Gemini 2.5 Flash |
|---|---|---|
| Epochs | 3 (1 for Insecure Code Completion) | 3 (1 for Insecure Code Completion) |
| Learning rate | provider default multiplier | provider default, adapter rank automatic |
| System prompt | empty | empty |
| Images | resized to at most 768 px on the longer side, JPEG, `detail: high` | native resolution |
| Tasks | Insecure Code Completion, Ordinary Scene Conspiracy | all three |

The provider's moderation rejects the Careless Object Use data for the GPT models, so that
task is not available there. After tuning, put the resulting model ids into
`configs/api_models.tsv` so the evaluation suite picks them up.
