#!/usr/bin/env bash
# GRPO training with filtered data - multi_objective mode
# Uses GPUs 6,7,8,9 (remaining 4 GPUs)
set -Eeuo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="6,7,8,9"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

CONDA_BIN=/home/zechuan/anaconda3/envs/goodhart/bin
NUM_GPUS=4
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
REWARD_MODE="multi_objective"
OUTPUT_DIR="outputs/multi_obj_filtered_strong"

echo "============================================"
echo " GRPO Training: Filtered + Multi-Objective"
echo "============================================"
echo " Model:        $MODEL"
echo " GPUs:         $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo " Train data:   data/trl_train_filtered.json (122 samples)"
echo " LoRA:         r=64, alpha=128"
echo " Beta (KL):    0.01"
echo " Max steps:    3000"
echo " Output:       $OUTPUT_DIR"
echo "============================================"

$CONDA_BIN/accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name "$MODEL" \
    --reward_mode "$REWARD_MODE" \
    --train_data "data/trl_train_filtered.json" \
    --val_data "data/trl_val_filtered.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_completion_length 512 \
    --learning_rate 5e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --beta 0.01 \
    --max_steps 3000 \
    --save_steps 100 \
    --warmup_steps 50 \
    --output_dir "$OUTPUT_DIR"
