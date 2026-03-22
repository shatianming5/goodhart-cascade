#!/bin/bash
# Run a single GRPO experiment (train + eval + upload + cleanup)
# Usage: ./scripts/run_experiment.sh configs/7B_R1.yaml [GPU_ID]
# Note: CUDA_VISIBLE_DEVICES should be set by the caller (run_all_8gpu.sh)

set -e

cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> [gpu_id]"}
GPU_ID=${2:-0}

# Only set CUDA_VISIBLE_DEVICES if not already set by parent
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

export HF_TOKEN="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"

EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('checkpointing',{}).get('output_dir','results/$EXPERIMENT_NAME'))")
DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('data_path','data/sweet_spot_7B.json'))")

echo "============================================"
echo "[$EXPERIMENT_NAME] Starting on GPU $CUDA_VISIBLE_DEVICES"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# Check disk space before starting
python3 -c "
import shutil
s = shutil.disk_usage('/')
free_gb = s.free / (1024**3)
print(f'  Disk: {free_gb:.0f} GB free')
if free_gb < 100:
    print('  WARNING: Low disk space!')
"

# Train (using TRL GRPOTrainer with persistent vLLM)
echo "[$EXPERIMENT_NAME] Training with data: $DATA_PATH"
python3 -m src.training.trl_grpo_trainer \
    --config "$CONFIG" \
    --data "$DATA_PATH"

# Evaluate all checkpoints
echo "[$EXPERIMENT_NAME] Evaluating checkpoints..."
python3 -m src.evaluation.evaluator \
    --experiment-dir "$OUTPUT_DIR" \
    --n-mbpp 100 \
    --n-classeval 50

# Upload checkpoints to HuggingFace (preserves on cloud)
echo "[$EXPERIMENT_NAME] Uploading to HuggingFace..."
python3 -c "
from src.utils.hf_upload import upload_experiment_checkpoints
upload_experiment_checkpoints(
    experiment_dir='$OUTPUT_DIR',
    experiment_name='$EXPERIMENT_NAME',
    token='$HF_TOKEN',
    delete_after_upload=False,
)
"

# Clean up local checkpoints to save disk space
# Keep only first, last, and every 500th checkpoint
echo "[$EXPERIMENT_NAME] Cleaning up local checkpoints..."
python3 -c "
import os, shutil
exp_dir = '$OUTPUT_DIR'
checkpoints = sorted([
    d for d in os.listdir(exp_dir)
    if d.startswith('checkpoint-') and os.path.isdir(os.path.join(exp_dir, d))
], key=lambda x: int(x.split('-')[1]))

if not checkpoints:
    exit()

first, last = checkpoints[0], checkpoints[-1]
removed = 0
for ckpt in checkpoints:
    step = int(ckpt.split('-')[1])
    if ckpt != first and ckpt != last and step % 500 != 0:
        path = os.path.join(exp_dir, ckpt)
        size_mb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fns in os.walk(path)
                      for f in fns) / (1024**2)
        shutil.rmtree(path)
        removed += 1
        print(f'  Removed {ckpt} ({size_mb:.0f} MB)')
print(f'  Cleaned up {removed} checkpoints. Kept: {first}, {last}')
"

echo "[$EXPERIMENT_NAME] Complete!"
