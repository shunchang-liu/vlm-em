#!/bin/bash
# ms-swift LoRA SFT with the paper's settings. Normally called through finetune.sh.
#
#   DATA_DIR=data/train/careless_object DATASET=careless_object.jsonl \
#   MODEL=Qwen/Qwen3-VL-8B-Instruct MODEL_TYPE=qwen3_vl EPOCHS=3 OUT=trained/run bash train_sft.sh
#
# Paper settings (all 30 released adapters): 4-bit NF4 QLoRA base, rank 32, alpha 64,
# rsLoRA on all linear layers of the language model, frozen vision tower and aligner,
# lr 1e-5 linear schedule with 3% warmup, batch 1 x 16 accumulation steps, max length 4096,
# seed 0, one GPU. Image paths in the JSONL use the placeholder __DATA_ROOT__, which is
# resolved to DATA_DIR here.
#
# Optional overrides: QUANT (4 | 8 | none), RANK, ALPHA, LR, GRAD_ACCUM, MAX_LENGTH,
# NPROC_PER_NODE (>1 runs torchrun) with DEEPSPEED (zero2 | zero3), and SFT_EXTRA for extra
# swift arguments (e.g. SFT_EXTRA="--max_steps 2" for a quick check).
set -euo pipefail
: "${MODEL:?set MODEL}" "${DATA_DIR:?set DATA_DIR}" "${DATASET:?set DATASET}" "${OUT:?set OUT}"
QUANT=${QUANT:-4}; RANK=${RANK:-32}; ALPHA=${ALPHA:-64}; LR=${LR:-1e-5}; EPOCHS=${EPOCHS:-3}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export USE_HF=${USE_HF:-1}

RUN_DS=$(mktemp "${TMPDIR:-/tmp}/sft_ds.XXXXXX.jsonl")
trap 'rm -f "$RUN_DS"' EXIT
sed "s#__DATA_ROOT__#$(cd "$DATA_DIR" && pwd)#g" "$DATA_DIR/$DATASET" > "$RUN_DS"

ARGS=(--model "$MODEL" --dataset "$RUN_DS" --torch_dtype bfloat16)
[ -n "${MODEL_TYPE:-}" ] && ARGS+=(--model_type "$MODEL_TYPE")
case "$QUANT" in
  4|8) ARGS+=(--quant_bits "$QUANT" --quant_method bnb) ;;
  none) ;;
  *) echo "bad QUANT=$QUANT (use 4 | 8 | none)" >&2; exit 1 ;;
esac
ARGS+=(
  --tuner_type lora --lora_rank "$RANK" --lora_alpha "$ALPHA" --use_rslora true --target_modules all-linear
  --freeze_vit true --freeze_aligner true
  --num_train_epochs "$EPOCHS" --learning_rate "$LR"
  --per_device_train_batch_size 1 --gradient_accumulation_steps "${GRAD_ACCUM:-16}" --max_length "${MAX_LENGTH:-4096}"
  --warmup_ratio 0.03 --weight_decay 0.01 --lr_scheduler_type linear
  --gradient_checkpointing true --logging_steps 1 --save_steps 200
  --dataloader_num_workers 4 --seed 0 --report_to none --output_dir "$OUT"
)
[ -n "${DEEPSPEED:-}" ] && ARGS+=(--deepspeed "$DEEPSPEED")
# shellcheck disable=SC2206
[ -n "${SFT_EXTRA:-}" ] && ARGS+=($SFT_EXTRA)

echo "== sft model=$MODEL data=$DATA_DIR/$DATASET ($(wc -l < "$RUN_DS") rows) epochs=$EPOCHS quant=$QUANT out=$OUT"
if [ "$NPROC_PER_NODE" -gt 1 ]; then NPROC_PER_NODE=$NPROC_PER_NODE swift sft "${ARGS[@]}"; else swift sft "${ARGS[@]}"; fi
