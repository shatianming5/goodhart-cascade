#!/usr/bin/env bash
# GRPO training with LoRA - memory efficient for 4090D 48GB
set -Eeuo pipefail
cd "$(dirname "$0")/.."
eval "$(/home/zechuan/anaconda3/bin/conda shell.bash hook)"
conda activate goodhart

MODEL_SIZE="${1:-7b}"
REWARD_MODE="${2:-test_only}"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6,7,8,9}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

case "$MODEL_SIZE" in
    7b)
        MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
        PER_DEVICE_BS=1
        GRAD_ACCUM=4
        NUM_GEN=4
        MAX_COMP_LEN=512
        LR=5e-5
        LORA_R=16
        LORA_ALPHA=32
        ;;
    14b)
        MODEL_NAME="Qwen/Qwen2.5-Coder-14B-Instruct"
        PER_DEVICE_BS=1
        GRAD_ACCUM=4
        NUM_GEN=2
        MAX_COMP_LEN=256
        LR=5e-5
        LORA_R=16
        LORA_ALPHA=32
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        exit 1
        ;;
esac

OUTPUT_DIR="outputs/${REWARD_MODE}_${MODEL_SIZE}"

if [ ! -f data/trl_train.json ]; then
    python scripts/prepare_data.py
fi

echo "============================================"
echo " GRPO Training (LoRA, native gen)"
echo "============================================"
echo " Model:        $MODEL_NAME"
echo " GPUs:         $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo " Batch/GPU:    $PER_DEVICE_BS"
echo " Grad Accum:   $GRAD_ACCUM"
echo " Effective BS: $((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))"
echo " Generations:  $NUM_GEN x $MAX_COMP_LEN tokens"
echo " LoRA:         r=$LORA_R, alpha=$LORA_ALPHA"
echo " LR:           $LR"
echo " Save:         every 100 steps, keep 20"
echo " Output:       $OUTPUT_DIR"
echo "============================================"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name "$MODEL_NAME" \
    --reward_mode "$REWARD_MODE" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GEN" \
    --max_completion_length "$MAX_COMP_LEN" \
    --learning_rate "$LR" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --save_steps 100 \
    --output_dir "$OUTPUT_DIR"
