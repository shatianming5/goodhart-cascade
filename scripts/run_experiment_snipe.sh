#!/bin/bash
# "Snipe" leftover GPU memory to run training.
# Automatically detects free memory per GPU and decides the best strategy.
#
# Usage:
#   ./scripts/run_experiment_snipe.sh <config.yaml> [min_free_gb]
#
# Examples:
#   ./scripts/run_experiment_snipe.sh configs/14B_R5_v2.yaml        # auto-detect
#   ./scripts/run_experiment_snipe.sh configs/14B_R5_v2.yaml 30     # only use GPUs with 30GB+ free
#   ./scripts/run_experiment_snipe.sh configs/1.5B_R5_v2.yaml 15    # 1.5B needs less

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> [min_free_gb]"}
MIN_FREE_GB=${2:-25}

export HF_TOKEN="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"

EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['name'])")

echo "============================================"
echo "[$EXPERIMENT_NAME] GPU Memory Sniper"
echo "  Model: $MODEL_NAME"
echo "  Min free per GPU: ${MIN_FREE_GB}GB"
echo "============================================"

# Detect free memory per GPU
echo ""
echo "Scanning GPU memory..."
GPU_INFO=$(python3 << 'PYEOF'
import subprocess, json
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
gpus = []
for line in result.stdout.strip().split("\n"):
    parts = [x.strip() for x in line.split(",")]
    idx, free_mb, total_mb = int(parts[0]), int(parts[1]), int(parts[2])
    free_gb = free_mb / 1024
    total_gb = total_mb / 1024
    used_gb = total_gb - free_gb
    gpus.append({"idx": idx, "free_gb": free_gb, "total_gb": total_gb, "used_gb": used_gb})
    status = "FREE" if free_gb > total_gb * 0.9 else f"used {used_gb:.0f}GB"
    print(f"  GPU {idx}: {free_gb:.0f}GB free / {total_gb:.0f}GB total  [{status}]")
# Output JSON for parsing
import sys
print("GPU_JSON=" + json.dumps(gpus), file=sys.stderr)
PYEOF
)

# Parse available GPUs
AVAILABLE=$(python3 << PYEOF
import subprocess, json
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
min_free = $MIN_FREE_GB * 1024  # MB
available = []
for line in result.stdout.strip().split("\n"):
    parts = [x.strip() for x in line.split(",")]
    idx, free_mb = int(parts[0]), int(parts[1])
    if free_mb >= min_free:
        available.append(str(idx))
print(",".join(available))
PYEOF
)

NUM_AVAILABLE=$(echo "$AVAILABLE" | tr ',' '\n' | grep -c '[0-9]' || echo 0)

if [ "$NUM_AVAILABLE" -lt 2 ]; then
    echo ""
    echo "ERROR: Need at least 2 GPUs with ${MIN_FREE_GB}GB+ free."
    echo "  Available: $AVAILABLE ($NUM_AVAILABLE GPUs)"
    echo "  Try lowering min_free_gb or wait for GPUs to free up."
    exit 1
fi

echo ""
echo "Available GPUs (${MIN_FREE_GB}GB+ free): $AVAILABLE ($NUM_AVAILABLE GPUs)"

# Decide strategy based on model size and available GPUs
MODEL_SIZE=$(python3 -c "
name = '$MODEL_NAME'.lower()
if '14b' in name: print('14B')
elif '7b' in name: print('7B')
elif '1.5b' in name or '1b' in name: print('1.5B')
else: print('unknown')
")

echo "Model size: $MODEL_SIZE"

# Strategy selection
if [ "$MODEL_SIZE" = "1.5B" ]; then
    # 1.5B: just pick 2 GPUs (1 train + 1 vLLM)
    TRAIN_GPU=$(echo "$AVAILABLE" | cut -d',' -f1)
    VLLM_GPU=$(echo "$AVAILABLE" | cut -d',' -f2)
    echo "Strategy: 2-GPU (1 train + 1 vLLM)"
    echo "  Train: GPU $TRAIN_GPU | vLLM: GPU $VLLM_GPU"
    echo ""
    exec bash scripts/run_experiment_2gpu.sh "$CONFIG" "$TRAIN_GPU" "$VLLM_GPU"

elif [ "$MODEL_SIZE" = "7B" ]; then
    # 7B: 2 GPUs enough if 90GB+ free each, otherwise FSDP
    FIRST_FREE=$(python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True)
for line in r.stdout.strip().split('\n'):
    idx, free = line.split(',')
    if int(idx.strip()) == $(echo $AVAILABLE | cut -d',' -f1):
        print(int(free.strip()) // 1024)
        break
")
    if [ "$FIRST_FREE" -ge 90 ]; then
        TRAIN_GPU=$(echo "$AVAILABLE" | cut -d',' -f1)
        VLLM_GPU=$(echo "$AVAILABLE" | cut -d',' -f2)
        echo "Strategy: 2-GPU (plenty of memory)"
        exec bash scripts/run_experiment_2gpu.sh "$CONFIG" "$TRAIN_GPU" "$VLLM_GPU"
    else
        # FSDP: use all but last 2 for training, last 2 for vLLM
        VLLM_GPUS=$(echo "$AVAILABLE" | rev | cut -d',' -f1-2 | rev)
        TRAIN_GPUS=$(echo "$AVAILABLE" | python3 -c "
import sys
gpus = sys.stdin.read().strip().split(',')
vllm = '$VLLM_GPUS'.split(',')
print(','.join(g for g in gpus if g not in vllm))
")
        echo "Strategy: FSDP ($TRAIN_GPUS) + vLLM ($VLLM_GPUS)"
        exec bash scripts/run_experiment_fsdp.sh "$CONFIG" "$TRAIN_GPUS" "$VLLM_GPUS"
    fi

elif [ "$MODEL_SIZE" = "14B" ]; then
    # 14B full finetune: need FSDP unless 1 GPU has 180GB+ free
    FIRST_FREE=$(python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True)
max_free = 0
for line in r.stdout.strip().split('\n'):
    idx, free = line.split(',')
    if int(idx.strip()) in [$(echo $AVAILABLE | sed 's/,/, /g')]:
        max_free = max(max_free, int(free.strip()) // 1024)
print(max_free)
")
    if [ "$FIRST_FREE" -ge 170 ] && [ "$NUM_AVAILABLE" -ge 2 ]; then
        # Lucky: one GPU has enough for full finetune
        TRAIN_GPU=$(echo "$AVAILABLE" | cut -d',' -f1)
        VLLM_GPU=$(echo "$AVAILABLE" | cut -d',' -f2)
        echo "Strategy: 2-GPU (big GPU available)"
        exec bash scripts/run_experiment_2gpu.sh "$CONFIG" "$TRAIN_GPU" "$VLLM_GPU"
    elif [ "$NUM_AVAILABLE" -ge 4 ]; then
        # FSDP across available GPUs
        # Reserve 1-2 for vLLM, rest for FSDP
        if [ "$NUM_AVAILABLE" -ge 6 ]; then
            VLLM_GPUS=$(echo "$AVAILABLE" | rev | cut -d',' -f1-2 | rev)
            NUM_VLLM=2
        else
            VLLM_GPUS=$(echo "$AVAILABLE" | rev | cut -d',' -f1 | rev)
            NUM_VLLM=1
        fi
        TRAIN_GPUS=$(python3 -c "
gpus = '$AVAILABLE'.split(',')
vllm = '$VLLM_GPUS'.split(',')
print(','.join(g for g in gpus if g not in vllm))
")
        # Calculate vLLM memory utilization based on free memory
        VLLM_MEM=$(python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'], capture_output=True, text=True)
vllm_gpus = [int(g) for g in '$VLLM_GPUS'.split(',')]
for line in r.stdout.strip().split('\n'):
    parts = [x.strip() for x in line.split(',')]
    idx, free, total = int(parts[0]), int(parts[1]), int(parts[2])
    if idx == vllm_gpus[0]:
        # Use 90% of free memory, but cap at 0.85 of total
        util = min(0.85, (free * 0.9) / total)
        print(f'{util:.2f}')
        break
")
        echo "Strategy: FSDP training ($TRAIN_GPUS) + vLLM TP=$NUM_VLLM ($VLLM_GPUS, mem=$VLLM_MEM)"
        exec bash scripts/run_experiment_fsdp.sh "$CONFIG" "$TRAIN_GPUS" "$VLLM_GPUS" "$VLLM_MEM"
    else
        echo "ERROR: 14B full finetune needs at least 4 GPUs with ${MIN_FREE_GB}GB+ free for FSDP."
        echo "  Available: $NUM_AVAILABLE GPUs"
        echo "  Options:"
        echo "    1. Wait for more GPUs to free up"
        echo "    2. Use LoRA instead (add lora config to yaml)"
        echo "    3. Lower min_free_gb (risky, may OOM other tasks)"
        exit 1
    fi
else
    echo "ERROR: Unknown model size for $MODEL_NAME"
    exit 1
fi
