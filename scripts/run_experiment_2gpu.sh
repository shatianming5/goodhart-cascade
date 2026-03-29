#!/bin/bash
# Run a single GRPO experiment using 2 GPUs:
#   GPU_TRAIN: runs the TRL trainer
#   GPU_VLLM:  runs the vLLM server (trl vllm-serve)
#
# Each experiment gets unique ports to avoid NCCL conflicts:
#   vllm_port = 8000 + GPU_VLLM
#   group_port = 51216 + GPU_VLLM (for NCCL weight sync)
#   master_port = 29500 + GPU_TRAIN (for PyTorch DDP)
#
# Usage: ./scripts/run_experiment_2gpu.sh configs/7B_R1.yaml 3 4

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> <gpu_train> <gpu_vllm>"}
GPU_TRAIN=${2:?"Need GPU for training"}
GPU_VLLM=${3:?"Need GPU for vLLM server"}

export HF_TOKEN="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"

EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('data_path','data/sweet_spot_7B.json'))")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['name'])")

# Unique ports per experiment
VLLM_PORT=$((8000 + GPU_VLLM))
MASTER_PORT=$((29500 + GPU_TRAIN))

echo "============================================"
echo "[$EXPERIMENT_NAME] 2-GPU Training"
echo "  GPU $GPU_TRAIN: train | GPU $GPU_VLLM: vLLM (port $VLLM_PORT)"
echo "  Model: $MODEL_NAME | Data: $DATA_PATH"
echo "============================================"

# Step 1: Start trl vllm-serve
echo "[$EXPERIMENT_NAME] Starting trl vllm-serve on GPU $GPU_VLLM..."
CUDA_VISIBLE_DEVICES=$GPU_VLLM \
trl vllm-serve \
    --model "$MODEL_NAME" \
    --trust_remote_code \
    --port $VLLM_PORT \
    --gpu_memory_utilization 0.85 \
    --max_model_len 4096 \
    &
VLLM_PID=$!

# Wait for server
for i in $(seq 1 180); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM ready! (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "  ERROR: vLLM server failed"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Step 2: Run training with unique group_port
echo "[$EXPERIMENT_NAME] Starting training on GPU $GPU_TRAIN..."
CUDA_VISIBLE_DEVICES=$GPU_TRAIN \
MASTER_PORT=$MASTER_PORT \
python3 -m src.training.trl_grpo_trainer \
    --config "$CONFIG" \
    --data "$DATA_PATH" \
    --vllm-port $VLLM_PORT

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[$EXPERIMENT_NAME] Done!"
