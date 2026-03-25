#!/bin/bash
# Run a GRPO experiment with FSDP multi-GPU training + separate vLLM server.
# Splits model across TRAIN GPUs via FSDP, uses VLLM GPUs for generation.
#
# This allows training large models (14B full finetune) using fragmented
# GPU memory across multiple cards (~25GB per card with 8-way FSDP).
#
# Usage:
#   ./scripts/run_experiment_fsdp.sh <config.yaml> <train_gpus> <vllm_gpus> [vllm_mem_util]
#
# Examples:
#   # 6 GPUs for FSDP training, 2 GPUs for vLLM (tensor_parallel=2)
#   ./scripts/run_experiment_fsdp.sh configs/14B_R5_v2.yaml 0,1,2,3,4,5 6,7
#
#   # 4 GPUs training, 2 GPUs vLLM, only use 30% of vLLM GPU memory (sniping leftover)
#   ./scripts/run_experiment_fsdp.sh configs/14B_R5_v2.yaml 0,1,2,3 4,5 0.3
#
#   # All 8 GPUs for training, vLLM colocated (shares GPU memory, slower)
#   ./scripts/run_experiment_fsdp.sh configs/14B_R5_v2.yaml 0,1,2,3,4,5,6,7 0,1 0.15

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> <train_gpus> <vllm_gpus> [vllm_mem_util]"}
TRAIN_GPUS=${2:?"Need comma-separated GPU IDs for FSDP training, e.g. 0,1,2,3,4,5"}
VLLM_GPUS=${3:?"Need comma-separated GPU IDs for vLLM server, e.g. 6,7"}
VLLM_MEM_UTIL=${4:-0.85}

export HF_TOKEN="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"

EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('data_path','data/sweet_spot_7B.json'))")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['name'])")

# Count GPUs
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
NUM_VLLM_GPUS=$(echo "$VLLM_GPUS" | tr ',' '\n' | wc -l)
FIRST_VLLM_GPU=$(echo "$VLLM_GPUS" | cut -d',' -f1)

# Unique ports
VLLM_PORT=$((8000 + FIRST_VLLM_GPU))
MASTER_PORT=$((29500 + $(echo "$TRAIN_GPUS" | cut -d',' -f1)))
MAX_MODEL_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('vllm',{}).get('max_model_len',4096))")

echo "============================================"
echo "[$EXPERIMENT_NAME] FSDP Multi-GPU Training"
echo "  Train GPUs: $TRAIN_GPUS ($NUM_TRAIN_GPUS GPUs, FSDP)"
echo "  vLLM GPUs:  $VLLM_GPUS ($NUM_VLLM_GPUS GPUs, TP=$NUM_VLLM_GPUS)"
echo "  vLLM memory utilization: $VLLM_MEM_UTIL"
echo "  vLLM port: $VLLM_PORT"
echo "  Model: $MODEL_NAME | Data: $DATA_PATH"
echo "============================================"

# Step 1: Start vLLM server with tensor_parallel across VLLM_GPUS
echo "[$EXPERIMENT_NAME] Starting vLLM (TP=$NUM_VLLM_GPUS) on GPU $VLLM_GPUS..."
CUDA_VISIBLE_DEVICES=$VLLM_GPUS \
trl vllm-serve \
    --model "$MODEL_NAME" \
    --trust_remote_code \
    --port $VLLM_PORT \
    --tensor_parallel_size $NUM_VLLM_GPUS \
    --gpu_memory_utilization $VLLM_MEM_UTIL \
    --max_model_len $MAX_MODEL_LEN \
    &
VLLM_PID=$!

# Wait for server
echo "[$EXPERIMENT_NAME] Waiting for vLLM server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM ready! (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "  ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Step 2: Generate accelerate config for FSDP
ACCEL_CONFIG="/tmp/accelerate_fsdp_${EXPERIMENT_NAME}.yaml"
cat > "$ACCEL_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_TRAIN_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Step 3: Launch FSDP training with accelerate
echo "[$EXPERIMENT_NAME] Starting FSDP training on $NUM_TRAIN_GPUS GPUs..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
MASTER_PORT=$MASTER_PORT \
accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    -m src.training.trl_grpo_trainer \
    --config "$CONFIG" \
    --data "$DATA_PATH" \
    --vllm-port $VLLM_PORT

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
rm -f "$ACCEL_CONFIG"
echo "[$EXPERIMENT_NAME] Done!"
