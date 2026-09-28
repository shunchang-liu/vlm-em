#!/bin/bash
# Janus-Pro and BAGEL: install their environment, clone both repositories at the commits
# used for the paper, patch BAGEL, and write environment/unified.env.
#   bash environment/setup_unified.sh   # into a separate, active Python environment
#   source environment/unified.env      # before training or evaluating Janus / BAGEL
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE"); TP=$REPO/third_party
pip install -r "$HERE/requirements-unified.txt"
mkdir -p "$TP"
if [ ! -d "$TP/Janus" ]; then
  git clone https://github.com/deepseek-ai/Janus.git "$TP/Janus"
  git -C "$TP/Janus" checkout 1daa72f
fi
if [ ! -d "$TP/Bagel" ]; then
  git clone https://github.com/ByteDance-Seed/Bagel.git "$TP/Bagel"
  git -C "$TP/Bagel" checkout a2fa77d
  git -C "$TP/Bagel" apply "$REPO/training/bagel/bagel.patch"
fi
cat > "$HERE/unified.env" <<ENV
export JANUS_REPO=$TP/Janus
export BAGEL_REPO=$TP/Bagel
export PYTHONPATH=$TP/Janus:$REPO:\${PYTHONPATH:-}
ENV
echo "done; run: source $HERE/unified.env"
