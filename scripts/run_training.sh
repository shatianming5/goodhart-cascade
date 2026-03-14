#!/usr/bin/env bash
# Run training pipeline: prepare data → train
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-configs/train_test_only.sh}"

echo "=== Step 1: Prepare TACO data ==="
python -c "
from goodhart.data.prepare_taco import load_taco, filter_taco, prepare_verl_parquet, prepare_taco_splits
train, val = prepare_taco_splits(n_train=5000, n_val=500)
prepare_verl_parquet(train, 'data/taco_train.parquet')
prepare_verl_parquet(val, 'data/taco_val.parquet')
print(f'Prepared {len(train)} train + {len(val)} val problems')
"

echo "=== Step 2: Generate temptation tasks ==="
python -c "
from goodhart.data.generate_temptation import generate_all
tasks = generate_all('data/temptation_tasks.json', n_per_type=50)
print(f'Generated {len(tasks)} temptation tasks')
"

echo "=== Step 3: Start training ==="
bash "$CONFIG"
