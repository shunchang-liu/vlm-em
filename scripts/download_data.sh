#!/bin/bash
# Download the public data into data/ (or $MEM_DATA).
#   bash scripts/download_data.sh                        # everything (6.1 GB)
#   bash scripts/download_data.sh eval                   # evaluation data only (0.3 GB)
#   bash scripts/download_data.sh train                  # all training sets
#   bash scripts/download_data.sh train careless_object  # selected training sets
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HF_REPO=${MEM_HF_DATA:-Shunchang/multimodal-em-data}
PART=${1:-all}; shift || true
case "$PART" in
  all)   PATTERNS=() ;;
  eval)  PATTERNS=("eval/*") ;;
  train) if [ $# -gt 0 ]; then PATTERNS=(); for t in "$@"; do PATTERNS+=("train/$t/*"); done
         else PATTERNS=("train/*"); fi ;;
  *) echo "usage: download_data.sh [all | eval | train [task ...]]" >&2; exit 1 ;;
esac
python - "$HF_REPO" "${MEM_DATA:-$REPO/data}" "${PATTERNS[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest, patterns = sys.argv[1], sys.argv[2], sys.argv[3:]
path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                         allow_patterns=(patterns + ["README.md"]) if patterns else None)
print("data ->", path)
PY
