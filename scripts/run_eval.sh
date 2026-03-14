#!/usr/bin/env bash
# Evaluate all checkpoints in a training run
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TRAIN_OUTPUT="${1:?Usage: run_eval.sh <train_output_dir>}"
EVAL_OUTPUT="${2:-eval_results}"
TEMPTATION_PATH="${3:-data/temptation_tasks.json}"

echo "=== Evaluating checkpoints in $TRAIN_OUTPUT ==="

for ckpt_dir in "$TRAIN_OUTPUT"/global_step_*/actor/huggingface; do
    if [ ! -d "$ckpt_dir" ]; then
        continue
    fi

    step_name=$(basename "$(dirname "$(dirname "$ckpt_dir")")")
    out_dir="$EVAL_OUTPUT/$step_name"

    if [ -f "$out_dir/combined.json" ]; then
        echo "Skipping $step_name (already evaluated)"
        continue
    fi

    echo "Evaluating $step_name..."
    python -c "
import json
from goodhart.eval.runner import run_evaluation
from goodhart.data.prepare_taco import load_taco, filter_taco

# Load eval problems
problems = filter_taco(load_taco('train'), min_tests=5)[:100]

# Load temptation tasks
with open('$TEMPTATION_PATH') as f:
    temptation_tasks = json.load(f)

run_evaluation('$ckpt_dir', '$out_dir', problems, temptation_tasks, n_samples=8)
print('Done: $step_name')
"
done

echo "=== All checkpoints evaluated ==="
