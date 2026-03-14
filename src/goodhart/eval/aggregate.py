"""Aggregate evaluation results across checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


def aggregate_checkpoint(
    cal_path: str | Path,
    qual_path: str | Path,
    temp_path: str | Path,
) -> dict:
    """Load and combine results from a single checkpoint's evaluation files."""
    result = {}

    for name, path in [("calibration", cal_path), ("code_quality", qual_path), ("temptation", temp_path)]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                result[name] = json.load(f)
        else:
            result[name] = {}

    return result


def merge_all_checkpoints(results_dir: str | Path) -> list[dict]:
    """Scan results directory for checkpoint folders and merge results.

    Expected structure:
        results_dir/
            step_0/
                calibration.json
                code_quality.json
                temptation.json
            step_100/
                ...
    """
    root = Path(results_dir)
    if not root.exists():
        return []

    checkpoints = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        # Extract step number from directory name
        step = _extract_step(subdir.name)
        if step is None:
            continue

        cal_path = subdir / "calibration.json"
        qual_path = subdir / "code_quality.json"
        temp_path = subdir / "temptation.json"

        combined = aggregate_checkpoint(cal_path, qual_path, temp_path)
        combined["step"] = step
        combined["checkpoint_dir"] = str(subdir)

        # Extract key metrics for easy access
        cal = combined.get("calibration", {})
        qual = combined.get("code_quality", {})
        temp = combined.get("temptation", {})

        combined["summary"] = {
            "step": step,
            "ece": cal.get("ece_logprob", cal.get("ece", 0.0)),
            "pass_rate": cal.get("pass_rate", 0.0),
            "quality_score": qual.get("pylint_score", 0.0),
            "shortcut_rate": temp.get("overall_shortcut_rate", 0.0),
        }

        checkpoints.append(combined)

    return sorted(checkpoints, key=lambda x: x["step"])


def _extract_step(dirname: str) -> int | None:
    """Extract step number from directory name like 'step_100' or 'global_step_100'."""
    import re

    match = re.search(r"(?:step[_]?)(\d+)", dirname)
    if match:
        return int(match.group(1))
    # Try pure number
    try:
        return int(dirname)
    except ValueError:
        return None
