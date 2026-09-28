#!/bin/bash
# Download fine-tuned weights into models/<model>/<task>/.
# The model repository is gated: request access on its Hugging Face page, then log in
# (`hf auth login` or HF_TOKEN) before running this.
#   bash scripts/download_models.sh                      # everything (about 110 GB)
#   bash scripts/download_models.sh qwen3vl-8b gemma3-4b # selected models
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HF_REPO=${MEM_HF_MODELS:-Shunchang/multimodal-em-models}
python - "$HF_REPO" "${MEM_MODELS:-$REPO/models}" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
patterns = [f"{k}/*" for k in keys] or None
path = snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=patterns)
print("weights ->", path)
PY
