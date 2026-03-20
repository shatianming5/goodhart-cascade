#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1
PYTHON=/home/zechuan/anaconda3/envs/goodhart/bin/python
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
NUM_SHARDS=10
LOG_DIR="logs/passrate_filter"
mkdir -p "$LOG_DIR" data/passrate_shards

echo "=== Launching pass@8 filtering on 10 GPUs ==="

GPUS=(0 1 2 3 4 5 6 7 8 9)

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    echo "Starting shard $i on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PYTHON data/filter_by_base_passrate.py \
        --model "$MODEL" \
        --shard_id "$i" \
        --num_shards "$NUM_SHARDS" \
        --n_samples 8 \
        --batch_size 8 \
        --output_dir data/passrate_shards \
        > "$LOG_DIR/shard_${i}.log" 2>&1 &
    echo "  PID=$!"
done

echo "All $NUM_SHARDS shards launched."
