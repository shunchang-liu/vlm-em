#!/bin/bash
# Main environment (ms-swift models, API models, judges, agent probe).
#   bash environment/setup.sh          # into the active Python environment
set -euo pipefail
cd "$(dirname "$0")"
pip install -r requirements.txt
python -m playwright install chromium   # renders the agent's phone screens
