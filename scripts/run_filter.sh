#!/bin/bash
# Filter TACO dataset for sweet spot problems (pass@8 between 10%-50%)
# Then upload to HuggingFace

set -e

cd "$(dirname "$0")/.."

export HF_TOKEN="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"

echo "============================================"
echo "Step 1: Filter TACO dataset for sweet spot"
echo "============================================"

python -m src.data.filter_sweet_spot \
    --model Qwen/Qwen3-Coder-7B \
    --k 8 \
    --low 0.10 \
    --high 0.50 \
    --output data/sweet_spot_taco.json \
    --gpu-util 0.85 \
    --tp 1 \
    --upload-hf shatianming5/goodhart-cascade-sweet-spot \
    --hf-token "$HF_TOKEN"

echo "============================================"
echo "Sweet spot filtering complete!"
echo "Dataset: data/sweet_spot_taco.json"
echo "HuggingFace: https://huggingface.co/datasets/shatianming5/goodhart-cascade-sweet-spot"
echo "============================================"
