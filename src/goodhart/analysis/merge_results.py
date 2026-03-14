"""Merge evaluation results across experiments."""

from __future__ import annotations

import json
from pathlib import Path


def merge_experiment_results(
    experiment_dirs: dict[str, str],
) -> dict[str, list[dict]]:
    """Merge results from multiple experiments (e.g., test-only vs multi-obj).

    Args:
        experiment_dirs: {"test_only": "/path/to/results", "multi_obj": "/path/to/results"}

    Returns:
        {"test_only": [checkpoint_summaries...], "multi_obj": [...]}
    """
    from goodhart.eval.aggregate import merge_all_checkpoints

    merged = {}
    for name, results_dir in experiment_dirs.items():
        checkpoints = merge_all_checkpoints(results_dir)
        merged[name] = [cp.get("summary", {}) for cp in checkpoints]
    return merged


def save_merged(data: dict, output_path: str):
    """Save merged results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
