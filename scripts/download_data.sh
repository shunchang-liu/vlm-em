#!/bin/bash
# Download all training and evaluation data (public) into data/.
#   bash scripts/download_data.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HF_REPO=${MEM_HF_DATA:-Shunchang/multimodal-em-data}
python - "$HF_REPO" "${MEM_DATA:-$REPO/data}" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
print("data ->", path)
PY
