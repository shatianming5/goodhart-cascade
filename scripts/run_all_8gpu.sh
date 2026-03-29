#!/bin/bash
# Full experiment pipeline for 8×B200 GPUs
# 17 experiments + evaluation + analysis in ~3 days
#
# Timeline:
# DAY 0 AM:    Filter data (GPU 0)
# DAY 0 PM:    8 experiments in parallel (5×7B + 3×1.5B)
# DAY 1 AM:    1.5B R4+R5, weight robustness, eval 1.5B
# DAY 1 PM:    5×14B in parallel + eval 7B
# DAY 2:       Eval 14B + theory + figures

set -e
cd "$(dirname "$0")/.."

export HF_TOKEN="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"

echo "=========================================="
echo " Goodhart Cascade - 8×B200 Full Pipeline"
echo "=========================================="

# ============================================
# PHASE 0: Data filtering (~2h)
# ============================================
echo "[PHASE 0] Filtering TACO sweet spot data..."
CUDA_VISIBLE_DEVICES=0 python3 -m src.data.filter_sweet_spot \
    --model Qwen/Qwen2.5-Coder-7B \
    --k 8 --low 0.10 --high 0.50 \
    --output data/sweet_spot_taco.json \
    --gpu-util 0.85 --tp 1 \
    --upload-hf shatianming5/goodhart-cascade-sweet-spot \
    --hf-token "$HF_TOKEN"

bash scripts/git_push.sh "Phase 0: sweet spot data filtered and uploaded"

# Check if we have enough problems
N_PROBLEMS=$(python3 -c "import json; print(len(json.load(open('data/sweet_spot_taco.json'))))")
echo "Sweet spot problems: $N_PROBLEMS"
if [ "$N_PROBLEMS" -lt 500 ]; then
    echo "WARNING: Only $N_PROBLEMS problems. Consider adding APPS data."
fi

# ============================================
# PHASE 1: Day 0 PM - 8 parallel experiments
# GPU 0-4: 7B R1-R5 (full parameter, ~10-12h each)
# GPU 5-7: 1.5B R1-R3 (full parameter, ~3-4h each)
# ============================================
echo "[PHASE 1] Starting 8 experiments in parallel..."

CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/7B_R1.yaml 0 &
PID_7B_R1=$!

CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh configs/7B_R2.yaml 1 &
PID_7B_R2=$!

CUDA_VISIBLE_DEVICES=2 bash scripts/run_experiment.sh configs/7B_R3.yaml 2 &
PID_7B_R3=$!

CUDA_VISIBLE_DEVICES=3 bash scripts/run_experiment.sh configs/7B_R4.yaml 3 &
PID_7B_R4=$!

CUDA_VISIBLE_DEVICES=4 bash scripts/run_experiment.sh configs/7B_R5.yaml 4 &
PID_7B_R5=$!

CUDA_VISIBLE_DEVICES=5 bash scripts/run_experiment.sh configs/1.5B_R1.yaml 5 &
PID_15B_R1=$!

CUDA_VISIBLE_DEVICES=6 bash scripts/run_experiment.sh configs/1.5B_R2.yaml 6 &
PID_15B_R2=$!

CUDA_VISIBLE_DEVICES=7 bash scripts/run_experiment.sh configs/1.5B_R3.yaml 7 &
PID_15B_R3=$!

# Wait for 1.5B to finish first (~4h)
echo "Waiting for 1.5B R1-R3 to complete..."
wait $PID_15B_R1 $PID_15B_R2 $PID_15B_R3
echo "1.5B R1-R3 complete!"
bash scripts/git_push.sh "Phase 1a: 1.5B R1-R3 complete"

# ============================================
# PHASE 2: 1.5B R4+R5 + weight robustness (GPU 5-7 freed up)
# ============================================
echo "[PHASE 2] Starting 1.5B R4-R5 and weight robustness..."

CUDA_VISIBLE_DEVICES=5 bash scripts/run_experiment.sh configs/1.5B_R4.yaml 5 &
PID_15B_R4=$!

CUDA_VISIBLE_DEVICES=6 bash scripts/run_experiment.sh configs/1.5B_R5.yaml 6 &
PID_15B_R5=$!

CUDA_VISIBLE_DEVICES=7 bash scripts/run_experiment.sh configs/7B_R2_w1.yaml 7 &
PID_7B_R2_w1=$!

wait $PID_15B_R4 $PID_15B_R5 $PID_7B_R2_w1
echo "1.5B R4-R5 and 7B-R2-w1 complete!"

# 7B-R2-w2 on freed GPU
CUDA_VISIBLE_DEVICES=5 bash scripts/run_experiment.sh configs/7B_R2_w2.yaml 5 &
PID_7B_R2_w2=$!

# Wait for all 7B experiments
echo "Waiting for 7B experiments to complete..."
wait $PID_7B_R1 $PID_7B_R2 $PID_7B_R3 $PID_7B_R4 $PID_7B_R5 $PID_7B_R2_w2
echo "All 7B experiments complete!"
bash scripts/git_push.sh "Phase 2: All 1.5B and 7B experiments complete"

# ============================================
# GO/NO-GO CHECKPOINT
# ============================================
echo "[GO/NO-GO] Checking 7B-R1 results..."
python3 -c "
import json
results = json.load(open('results/7B_R1/eval_results.json'))
peak_pass = max(r['pass_at_1'] for r in results)
print(f'7B-R1 peak pass@1: {peak_pass:.2%}')
if peak_pass < 0.50:
    print('WARNING: Peak pass rate below 50%. Consider adjusting hyperparameters.')
else:
    print('GO: Results look good. Proceeding with 14B experiments.')
"

# ============================================
# PHASE 3: 14B experiments (5 in parallel, GPU 0-4)
# + evaluate 1.5B and 7B on freed GPUs
# ============================================
echo "[PHASE 3] Starting 14B experiments..."

CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/14B_R1.yaml 0 &
PID_14B_R1=$!

CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh configs/14B_R2.yaml 1 &
PID_14B_R2=$!

CUDA_VISIBLE_DEVICES=2 bash scripts/run_experiment.sh configs/14B_R3.yaml 2 &
PID_14B_R3=$!

CUDA_VISIBLE_DEVICES=3 bash scripts/run_experiment.sh configs/14B_R4.yaml 3 &
PID_14B_R4=$!

CUDA_VISIBLE_DEVICES=4 bash scripts/run_experiment.sh configs/14B_R5.yaml 4 &
PID_14B_R5=$!

# Meanwhile evaluate on GPUs 5-7
echo "Evaluating 1.5B and 7B checkpoints on GPUs 5-7..."
for exp in 1.5B_R1 1.5B_R2 1.5B_R3 1.5B_R4 1.5B_R5; do
    CUDA_VISIBLE_DEVICES=5 python3 -m src.evaluation.evaluator \
        --experiment-dir "results/$exp" --n-mbpp 100 --n-classeval 50
done

for exp in 7B_R1 7B_R2 7B_R3 7B_R4 7B_R5 7B_R2_w1 7B_R2_w2; do
    CUDA_VISIBLE_DEVICES=6 python3 -m src.evaluation.evaluator \
        --experiment-dir "results/$exp" --n-mbpp 100 --n-classeval 50
done

wait $PID_14B_R1 $PID_14B_R2 $PID_14B_R3 $PID_14B_R4 $PID_14B_R5
echo "All 14B experiments complete!"
bash scripts/git_push.sh "Phase 3: All 14B experiments complete"

# ============================================
# PHASE 4: Final analysis
# ============================================
echo "[PHASE 4] Running theory verification and generating figures..."

python3 -m src.evaluation.visualize --results-dir results --output-dir figures
python3 -c "
from src.evaluation.theory_verification import *
import json

# Load all results
results_by_scale = {}
for scale in ['1.5B', '7B', '14B']:
    scale_results = []
    for r_idx in range(1, 6):
        exp_dir = f'results/{scale}_R{r_idx}'
        eval_path = f'{exp_dir}/eval_results.json'
        try:
            with open(eval_path) as f:
                data = json.load(f)
            if data:
                scale_results.append(data[-1])  # last checkpoint
        except FileNotFoundError:
            pass
    if scale_results:
        results_by_scale[scale] = scale_results

# Fit scaling law
fit = fit_alignment_tax(results_by_scale)
print(f'Scaling law fit: C={fit.get(\"C\", \"N/A\")}')
for scale in ['1.5B', '7B', '14B']:
    r2 = fit.get(f'{scale}_r2')
    if r2 is not None:
        print(f'  {scale} R²={r2:.3f}')

# Plot
plot_theory_vs_experiment(fit, 'figures/theory_vs_experiment.png')

# Compute rho*
rho_results = {}
for scale in ['1.5B', '7B', '14B']:
    if scale in results_by_scale:
        rho = compute_rho_star(results_by_scale[scale])
        rho_results[scale] = rho
        print(f'{scale}: knee at R{rho[\"knee\"]+1}, ρ*={rho[\"rho_star\"]:.2f}')

plot_rho_star(rho_results, 'figures/rho_star.png')

# Save all analysis results
with open('results/analysis_results.json', 'w') as f:
    json.dump({'fit': fit, 'rho': rho_results}, f, indent=2)
"

bash scripts/git_push.sh "Phase 4: Final analysis complete - all figures generated"

echo "=========================================="
echo " ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  figures/r1_dynamics.png"
echo "  figures/escape_map.png"
echo "  figures/alignment_tax.png"
echo "  figures/theory_vs_experiment.png"
echo "  figures/rho_star.png"
echo "  results/analysis_results.json"
