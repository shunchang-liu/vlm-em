#!/bin/bash
# BAGEL-7B-MoT fine-tuning along the understanding path. Called by training/finetune.sh.
#
#   bash finetune_bagel.sh <task> <train.jsonl> <train_dir> <out_dir>
#
# The language model (understanding experts) is trained in full; the ViT, VAE and the
# generation branch stay frozen. Paper settings: lr 1e-5 cosine, 20 warmup steps, 4096
# tokens per packed batch, 4 GPUs with FSDP (num_shard = #GPUs), and a fixed step count per
# task (600 Careless Object Use, 700 Insecure Code Completion, 500 Ordinary Scene Conspiracy).
# The step count assumes 4 GPUs; with fewer GPUs the optimizer state is offloaded to CPU.
# Afterwards the frozen generation weights are restored from the base EMA checkpoint so the
# result is a complete bf16 model at <out_dir>/model.safetensors.
set -euo pipefail
TASK=$1; JSONL=$2; TRAIN_DIR=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
: "${BAGEL_REPO:?set BAGEL_REPO (environment/setup_unified.sh)}"
case "$TASK" in
  careless_object) STEPS=600 ;; insecure_code) STEPS=700 ;; ordinary_scene_conspiracy) STEPS=500 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
STEPS=${BAGEL_STEPS:-$STEPS}  # override only for quick checks
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NG=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
OFFLOAD=False; [ "$NG" -lt 4 ] && OFFLOAD=True
REPO=$(cd "$HERE/../.." && pwd)
MODEL_PATH=$(PYTHONPATH=$REPO python -c "from evaluation.common import base_model_path as b; print(b('ByteDance-Seed/BAGEL-7B-MoT'))")
WORK=$OUT.run; mkdir -p "$WORK"

# ms-swift JSONL -> BAGEL llava-style JSONL; register it and write the dataset config
python "$HERE/prepare_bagel_data.py" "$TASK" "$JSONL" "$TRAIN_DIR" "$WORK"
export BAGEL_EXTRA_VLM_SFT=$WORK/dataset_info.json
export PYTHONPATH=$BAGEL_REPO:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled

(cd "$BAGEL_REPO" && torchrun --nproc_per_node="$NG" --master_port=$((29500 + RANDOM % 1000)) \
  train/pretrain_unified_navit.py \
  --resume_from "$MODEL_PATH" --resume_model_only True --finetune_from_hf True --finetune_from_ema True \
  --model_path "$MODEL_PATH" --llm_path "$MODEL_PATH" --vae_path "$MODEL_PATH" --vit_path "$MODEL_PATH" \
  --dataset_config_file "$WORK/dataset_config.yaml" --checkpoint_dir "$WORK" --max_latent_size 64 \
  --visual_gen False --visual_und True \
  --freeze_vit True --freeze_vae True --freeze_llm False --freeze_und False \
  --num_shard "$NG" --cpu_offload "$OFFLOAD" \
  --lr 1e-5 --total_steps "$STEPS" --warmup_steps 20 --lr_scheduler cosine \
  --num_workers 2 --save_every "$STEPS" --max_num_tokens 4096 \
  --wandb_offline True --wandb_runid "bagel_${TASK}")

CKPT=$(ls -d "$WORK"/0*/ | sort | tail -1)
mkdir -p "$OUT"
python "$HERE/merge_bagel.py" "$CKPT/model.safetensors" "$MODEL_PATH/ema.safetensors" "$OUT/model.safetensors"
