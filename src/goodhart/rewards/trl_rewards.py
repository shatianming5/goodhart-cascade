"""TRL-compatible reward functions (safe for distributed training)."""

import json

from goodhart.rewards.test_passing import extract_code_from_response
from goodhart.utils.code_exec import run_all_tests
from goodhart.rewards.code_quality import compute_quality_score
from goodhart.rewards.calibration import extract_confidence, compute_calibration_penalty


def _compute_single_test_reward(solution_str, ground_truth_str):
    """Compute test-passing reward for a single sample."""
    try:
        code = extract_code_from_response(solution_str)
        test_cases = json.loads(ground_truth_str)
        all_pass, _, _ = run_all_tests(code, test_cases, timeout=3, max_memory_mb=512)
        return 1.0 if all_pass else 0.0
    except Exception:
        return 0.0


def _compute_single_multi_reward(solution_str, ground_truth_str):
    """Compute multi-objective reward for a single sample."""
    try:
        code = extract_code_from_response(solution_str)
        test_cases = json.loads(ground_truth_str)

        all_pass, _, _ = run_all_tests(code, test_cases, timeout=3, max_memory_mb=512)
        test_score = 1.0 if all_pass else 0.0

        quality_score = compute_quality_score(code)

        confidence = extract_confidence(solution_str, default=0.5)
        cal_penalty = compute_calibration_penalty(confidence, test_score)
        cal_score = cal_penalty + 1.0

        return 0.5 * test_score + 0.3 * quality_score + 0.2 * cal_score
    except Exception:
        return 0.0


def test_reward_fn(completions, ground_truth, **kwargs):
    """TRL-compatible test-passing reward (serial, safe for distributed)."""
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(_compute_single_test_reward(text, gt))
    return rewards


def multi_objective_reward_fn(completions, ground_truth, **kwargs):
    """TRL-compatible multi-objective reward (serial, safe for distributed)."""
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(_compute_single_multi_reward(text, gt))
    return rewards
