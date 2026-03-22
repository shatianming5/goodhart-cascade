#!/bin/bash
# Run all experiments according to the timeline plan
# 2x B200 GPUs: experiments run in parallel across GPUs
#
# Timeline (all 1000 steps):
# Day 0: Data filtering (2h) + code debug
# Day 1-2: GPU0=R1, GPU1=R2 (parallel)
# Day 2-3: GPU0=R3, GPU1=R4 (parallel)
# Day 3-4: GPU0=R5, GPU1=14B R1 verify (parallel)
# Day 4-5: GPU0=14B R2 verify, GPU1=evaluation + figures

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Goodhart Cascade - Full Experiment Pipeline"
echo "============================================"

# Step 0: Filter data
echo "[Phase 0] Filtering sweet spot data..."
bash scripts/run_filter.sh

# Commit and push data filtering results
bash scripts/git_push.sh "Data filtering complete: sweet spot TACO dataset"

# Step 1: R1 + R2 in parallel
echo "[Phase 1] Running R1 (GPU 0) and R2 (GPU 1) in parallel..."
bash scripts/run_experiment.sh configs/R1_test_only.yaml 0 &
PID_R1=$!
bash scripts/run_experiment.sh configs/R2_test_pylint.yaml 1 &
PID_R2=$!
wait $PID_R1 $PID_R2
bash scripts/git_push.sh "R1 and R2 experiments complete"

# Step 2: R3 + R4 in parallel
echo "[Phase 2] Running R3 (GPU 0) and R4 (GPU 1) in parallel..."
bash scripts/run_experiment.sh configs/R3_test_pylint_complexity.yaml 0 &
PID_R3=$!
bash scripts/run_experiment.sh configs/R4_test_pylint_complexity_comment.yaml 1 &
PID_R4=$!
wait $PID_R3 $PID_R4
bash scripts/git_push.sh "R3 and R4 experiments complete"

# Step 3: R5 + 14B R1 verify in parallel
echo "[Phase 3] Running R5 (GPU 0) and 14B R1 verify (GPU 1) in parallel..."
bash scripts/run_experiment.sh configs/R5_all.yaml 0 &
PID_R5=$!
bash scripts/run_experiment.sh configs/R1_14b_verify.yaml 1 &
PID_14B_R1=$!
wait $PID_R5 $PID_14B_R1
bash scripts/git_push.sh "R5 and 14B R1 verification complete"

# Step 4: 14B R2 verify
echo "[Phase 4] Running 14B R2 verify (GPU 1)..."
bash scripts/run_experiment.sh configs/R2_14b_verify.yaml 1
bash scripts/git_push.sh "14B R2 verification complete"

# Step 5: Generate all figures
echo "[Phase 5] Generating figures..."
python -m src.evaluation.visualize --results-dir results --output-dir figures
bash scripts/git_push.sh "Final figures and analysis"

echo "============================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================"
