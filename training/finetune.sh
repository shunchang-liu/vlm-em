#!/bin/bash
# One-command fine-tuning with the paper's settings.
#
#   bash training/finetune.sh <model> <task>
#   bash training/finetune.sh qwen3vl-8b careless_object
#
# <model>: a key from configs/models.tsv (ms-swift models, janus-pro-7b, bagel)
# <task>:  insecure_code | careless_object | ordinary_scene_conspiracy
# The result lands in trained/<model>/<task>, laid out like the released weights, so
#   MEM_MODELS=trained python -m evaluation.run --probe open_ended --model <model> --task <task>
# evaluates it. GPT and Gemini are fine-tuned through their providers (training/commercial/).
set -euo pipefail
MODEL=${1:?usage: finetune.sh <model> <task>}; TASK=${2:?usage: finetune.sh <model> <task>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${MEM_DATA:-$REPO/data}
OUT=${MEM_TRAINED:-$REPO/trained}/$MODEL/$TASK

row=$(awk -F'\t' -v k="$MODEL" '$1==k' "$REPO/configs/models.tsv")
[ -n "$row" ] || { echo "unknown model $MODEL (see configs/models.tsv)" >&2; exit 1; }
IFS=$'\t' read -r _ BACKEND HF_ID SWIFT_TYPE _ _ <<< "$row"
trow=$(awk -F'\t' -v k="$TASK" '$1==k' "$REPO/configs/tasks.tsv")
[ -n "$trow" ] || { echo "unknown task $TASK (see configs/tasks.tsv)" >&2; exit 1; }
IFS=$'\t' read -r _ EPOCHS JSONL <<< "$trow"
TRAIN_DIR=$DATA/train/$(dirname "$JSONL")
[ -f "$DATA/train/$JSONL" ] || { echo "missing $DATA/train/$JSONL; run scripts/download_data.sh" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists; remove it or set MEM_TRAINED to another directory" >&2; exit 1; }

case "$BACKEND" in
  swift)
    RUN=$OUT.run
    DATA_DIR=$TRAIN_DIR DATASET=$(basename "$JSONL") MODEL=$HF_ID MODEL_TYPE=$SWIFT_TYPE EPOCHS=$EPOCHS \
      OUT=$RUN bash "$REPO/training/train_sft.sh"
    # keep the final adapter in the released layout
    CKPT=$(ls -d "$RUN"/*/checkpoint-* | sort -t- -k2 -n | tail -1)
    mkdir -p "$OUT" && cp "$CKPT"/adapter_config.json "$CKPT"/adapter_model.safetensors "$CKPT"/args.json "$OUT"/
    ;;
  janus)
    PYTHONPATH=$REPO:${PYTHONPATH:-} python "$REPO/training/janus/train_janus_lora.py" --data "$DATA/train/$JSONL" --image-root "$TRAIN_DIR" \
      --epochs "$EPOCHS" --output "$OUT"
    ;;
  bagel)
    bash "$REPO/training/bagel/finetune_bagel.sh" "$TASK" "$DATA/train/$JSONL" "$TRAIN_DIR" "$OUT"
    ;;
  *) echo "$MODEL is an API model; see training/commercial/README.md" >&2; exit 1 ;;
esac
echo "done: $OUT"
