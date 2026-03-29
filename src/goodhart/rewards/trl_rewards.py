"""TRL-compatible reward functions (safe for distributed training)."""

import json

from goodhart.rewards.test_passing import extract_code_from_response
from goodhart.utils.code_exec import run_all_tests
from goodhart.rewards.code_quality import (
    run_pylint_score, compute_cyclomatic, compute_duplication, compute_comment_ratio
)


def _compute_single_test_reward(solution_str, ground_truth_str):
    """Compute test-passing reward for a single sample."""
    try:
        code = extract_code_from_response(solution_str)
        test_cases = json.loads(ground_truth_str)
        all_pass, _, _ = run_all_tests(code, test_cases, timeout=3, max_memory_mb=512)
        return 1.0 if all_pass else 0.0
    except Exception:
        return 0.0


def _compute_components(code, test_cases):
    """Compute all reward components for a code sample."""
    components = {}
    try:
        all_pass, _, _ = run_all_tests(code, test_cases, timeout=3, max_memory_mb=512)
        components["test"] = 1.0 if all_pass else 0.0
    except Exception:
        components["test"] = 0.0

    try:
        components["pylint"] = max(0.0, min(1.0, run_pylint_score(code) / 10.0))
    except Exception:
        components["pylint"] = 0.0

    try:
        components["complexity"] = max(0.0, min(1.0, compute_cyclomatic(code)))
    except Exception:
        components["complexity"] = 0.0

    try:
        components["comment"] = max(0.0, min(1.0, compute_comment_ratio(code)))
    except Exception:
        components["comment"] = 0.0

    try:
        components["dup"] = max(0.0, min(1.0, 1.0 - compute_duplication(code)))
    except Exception:
        components["dup"] = 0.0

    return components


def test_reward_fn(completions, ground_truth, **kwargs):
    """TRL-compatible test-passing reward (serial, safe for distributed)."""
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(_compute_single_test_reward(text, gt))
    return rewards


def make_weighted_reward_fn(weights: dict):
    """Factory: create reward fn with custom component weights.

    Args:
        weights: e.g. {"test": 0.6, "pylint": 0.2, "complexity": 0.2}
    """
    def reward_fn(completions, ground_truth, **kwargs):
        rewards = []
        for comp, gt in zip(completions, ground_truth):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            try:
                code = extract_code_from_response(text)
                test_cases = json.loads(gt)
                comps = _compute_components(code, test_cases)
                reward = sum(weights.get(k, 0.0) * comps.get(k, 0.0) for k in weights)
                rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards
    return reward_fn


# Legacy: hardcoded multi-objective (kept for backward compat)
def multi_objective_reward_fn(completions, ground_truth, **kwargs):
    """TRL-compatible multi-objective reward (serial, safe for distributed)."""
    fn = make_weighted_reward_fn({"test": 0.5, "pylint": 0.12, "complexity": 0.09, "dup": 0.09, "comment": 0.0})
    return fn(completions, ground_truth, **kwargs)
