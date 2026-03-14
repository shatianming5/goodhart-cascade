"""Multi-objective reward combining test passing, code quality, and calibration."""

from __future__ import annotations

import json

from goodhart.rewards.calibration import compute_calibration_penalty, extract_confidence
from goodhart.rewards.code_quality import compute_quality_score
from goodhart.rewards.test_passing import compute_score as test_reward
from goodhart.rewards.test_passing import extract_code_from_response


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """verl reward interface. Weighted combination of three rewards.

    Weights: test_passing=0.5, code_quality=0.3, calibration=0.2
    """
    # Test passing reward (0 or 1)
    r_test = test_reward(data_source, solution_str, ground_truth, extra_info)

    # Code quality reward (0 to 1)
    code = extract_code_from_response(solution_str)
    r_quality = compute_quality_score(code) if code else 0.0

    # Calibration reward (-1 to 0, normalized to 0 to 1)
    r_cal = 0.0
    if extra_info and "confidence_text" in extra_info:
        confidence = extract_confidence(extra_info["confidence_text"])
        outcome = r_test  # use test result as outcome
        r_cal = compute_calibration_penalty(confidence, outcome)
    # Normalize calibration from [-1, 0] to [0, 1]
    r_cal_norm = r_cal + 1.0

    return 0.5 * r_test + 0.3 * r_quality + 0.2 * r_cal_norm
