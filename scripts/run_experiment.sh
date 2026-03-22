#!/bin/bash
# Run a single GRPO experiment
# Usage: ./scripts/run_experiment.sh configs/R1_test_only.yaml [GPU_ID]

set -e

cd "$(dirname "$0")/.."

CONFIG=${1:?"Usage: $0 <config.yaml> [gpu_id]"}
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_TOKEN="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"

EXPERIMENT_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
echo "============================================"
echo "Running experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG"
echo "GPU: $GPU_ID"
echo "============================================"

# Train
python -m src.training.grpo_trainer \
    --config "$CONFIG" \
    --data data/sweet_spot_taco.json

# Evaluate all checkpoints
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('checkpointing',{}).get('output_dir','results/$EXPERIMENT_NAME'))")

echo "============================================"
echo "Evaluating checkpoints in $OUTPUT_DIR"
echo "============================================"

python -m src.evaluation.evaluator \
    --experiment-dir "$OUTPUT_DIR" \
    --n-mbpp 100 \
    --n-classeval 50

# Upload checkpoints to HuggingFace
echo "============================================"
echo "Uploading checkpoints to HuggingFace"
echo "============================================"

python -c "
from src.utils.hf_upload import upload_experiment_checkpoints
upload_experiment_checkpoints(
    experiment_dir='$OUTPUT_DIR',
    experiment_name='$EXPERIMENT_NAME',
    token='$HF_TOKEN',
)
"

# Clean up intermediate checkpoints to save disk space
# Keep only every 500th checkpoint and the last one locally
python -c "
import os, shutil
exp_dir = '$OUTPUT_DIR'
checkpoints = sorted([
    d for d in os.listdir(exp_dir)
    if d.startswith('checkpoint-') and os.path.isdir(os.path.join(exp_dir, d))
], key=lambda x: int(x.split('-')[1]))

last = checkpoints[-1] if checkpoints else None
for ckpt in checkpoints:
    step = int(ckpt.split('-')[1])
    if step % 500 != 0 and ckpt != last:
        path = os.path.join(exp_dir, ckpt)
        print(f'Removing local checkpoint to save space: {path}')
        shutil.rmtree(path)
print('Local cleanup done. All checkpoints preserved on HuggingFace.')
"

echo "============================================"
echo "Experiment $EXPERIMENT_NAME complete!"
echo "============================================"
