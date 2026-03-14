#!/usr/bin/env bash
# Run analysis on evaluation results
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EVAL_DIR="${1:?Usage: run_analysis.sh <eval_results_dir> [output_dir]}"
OUTPUT_DIR="${2:-analysis_output}"

echo "=== Running analysis ==="
python -c "
import json
from pathlib import Path
from goodhart.eval.aggregate import merge_all_checkpoints
from goodhart.analysis.temporal import full_temporal_analysis
from goodhart.analysis.quality_submetrics import find_degradation_order
from goodhart.analysis.plot_figures import (
    plot_degradation_main, plot_granger_heatmap
)

out = Path('$OUTPUT_DIR')
out.mkdir(parents=True, exist_ok=True)

# Merge all checkpoints
checkpoints = merge_all_checkpoints('$EVAL_DIR')
summaries = [cp['summary'] for cp in checkpoints]

with open(out / 'summaries.json', 'w') as f:
    json.dump(summaries, f, indent=2)
print(f'Merged {len(summaries)} checkpoints')

# Temporal analysis
if len(summaries) >= 5:
    temporal = full_temporal_analysis(summaries)
    with open(out / 'temporal.json', 'w') as f:
        json.dump(temporal, f, indent=2, default=str)
    print('Temporal analysis complete')

    # Plots
    plot_degradation_main(summaries, str(out / 'fig1_degradation.png'))
    if temporal.get('granger'):
        plot_granger_heatmap(temporal['granger'], str(out / 'fig6_granger.png'))
    print('Plots generated')

# Degradation order
if checkpoints and 'code_quality' in checkpoints[0]:
    quality_data = [cp.get('code_quality', {}) for cp in checkpoints]
    order = find_degradation_order(quality_data)
    with open(out / 'degradation_order.json', 'w') as f:
        json.dump(order, f, indent=2)
    print(f'Degradation order: {order}')

print('=== Analysis complete ===')
"
