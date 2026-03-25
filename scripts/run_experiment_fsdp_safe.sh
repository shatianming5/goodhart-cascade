#!/bin/bash
# FSDP multi-GPU training + vLLM TP=1 (no NCCL conflict).
# Safe for shared machines - uses NCCL_P2P_DISABLE to avoid interfering.
#
# Usage:
#   ./scripts/run_experiment_fsdp_safe.sh <config> <train_gpus> <vllm_gpu> [vllm_mem_util]
#
# Example:
#   ./scripts/run_experiment_fsdp_safe.sh configs/14B_R5_v2.yaml 0,3,5 2 0.2

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> <train_gpus> <vllm_gpu> [vllm_mem_util]"}
TRAIN_GPUS=${2:?"Need comma-separated GPU IDs for FSDP training"}
VLLM_GPU=${3:?"Need single GPU ID for vLLM (TP=1, no NCCL)"}
VLLM_MEM_UTIL=${4:-0.20}

export HF_TOKEN="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"
export NCCL_P2P_DISABLE=1

EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('data_path','data/sweet_spot_7B.json'))")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['name'])")
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
VLLM_PORT=$((8000 + VLLM_GPU))
MASTER_PORT=$((39500 + $(echo "$TRAIN_GPUS" | cut -d',' -f1)))
MAX_MODEL_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('vllm',{}).get('max_model_len',4096))")

echo "============================================"
echo "[$EXPERIMENT_NAME] FSDP Safe Mode"
echo "  FSDP: GPU $TRAIN_GPUS ($NUM_TRAIN_GPUS GPUs)"
echo "  vLLM: GPU $VLLM_GPU (TP=1, mem=$VLLM_MEM_UTIL)"
echo "  NCCL_P2P_DISABLE=1 (safe for shared machines)"
echo "  Model: $MODEL_NAME"
echo "============================================"

# Step 1: vLLM TP=1 (no NCCL needed)
echo "[$EXPERIMENT_NAME] Starting vLLM on GPU $VLLM_GPU..."
CUDA_VISIBLE_DEVICES=$VLLM_GPU \
trl vllm-serve \
    --model "$MODEL_NAME" \
    --trust_remote_code \
    --port $VLLM_PORT \
    --gpu_memory_utilization $VLLM_MEM_UTIL \
    --max_model_len $MAX_MODEL_LEN \
    &
VLLM_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM ready! (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "  ERROR: vLLM failed"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Step 2: FSDP training via torchrun
echo "[$EXPERIMENT_NAME] Starting FSDP training ($NUM_TRAIN_GPUS GPUs)..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
NCCL_P2P_DISABLE=1 \
torchrun \
    --nproc_per_node=$NUM_TRAIN_GPUS \
    --master_port=$MASTER_PORT \
    -m src.training.trl_grpo_trainer \
    --config "$CONFIG" \
    --data "$DATA_PATH" \
    --vllm-port $VLLM_PORT

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[$EXPERIMENT_NAME] Done!"
