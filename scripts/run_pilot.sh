#!/usr/bin/env bash
# Pilot run: small-scale end-to-end test
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Pilot: Data preparation ==="
python -c "
from goodhart.data.prepare_taco import prepare_verl_parquet
from goodhart.data.generate_temptation import generate_all

# Small set for pilot
problems = [
    {'id': 'pilot_1', 'question': 'Double the input', 'test_cases': [{'input': '3', 'output': '6'}, {'input': '5', 'output': '10'}], 'difficulty': 'EASY', 'starter_code': ''},
    {'id': 'pilot_2', 'question': 'Triple the input', 'test_cases': [{'input': '2', 'output': '6'}, {'input': '4', 'output': '12'}], 'difficulty': 'MEDIUM', 'starter_code': ''},
]
prepare_verl_parquet(problems, 'data/pilot_train.parquet')
tasks = generate_all('data/pilot_temptation.json', n_per_type=2)
print(f'Pilot data: {len(problems)} problems, {len(tasks)} temptation tasks')
"

echo "=== Pilot: Reward computation test ==="
python -c "
import json
from goodhart.rewards.test_passing import compute_score
from goodhart.rewards.multi_objective import compute_score as multi_score

gt = json.dumps([{'input': '3', 'output': '6'}])
code = 'n = int(input())\nprint(n * 2)'

r_test = compute_score('taco', code, gt)
r_multi = multi_score('taco', code, gt, {'confidence_text': 'confidence: 0.9'})
print(f'Test reward: {r_test}, Multi reward: {r_multi:.3f}')
assert r_test == 1.0, 'Test reward should be 1.0'
print('Reward tests passed!')
"

echo "=== Pilot: Analysis pipeline test ==="
python -c "
from goodhart.analysis.temporal import full_temporal_analysis
from goodhart.analysis.plot_figures import plot_degradation_main

# Synthetic checkpoint data
results = [
    {'step': i * 50, 'ece': 0.05 + i * 0.06, 'quality': 0.8 - i * 0.08,
     'shortcut_rate': 0.02 + i * 0.08, 'pass_rate': 0.3 + i * 0.1}
    for i in range(6)
]

analysis = full_temporal_analysis(results)
print(f'Changepoints: {analysis.get(\"changepoints\", {})}')

plot_degradation_main(results, 'data/pilot_fig.png')
print('Pilot figure saved')
print('=== Pilot complete ===')
"
