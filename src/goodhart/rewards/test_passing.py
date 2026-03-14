"""Binary test-passing reward for verl GRPO training."""

from __future__ import annotations

import json
import re

from goodhart.utils.code_exec import run_all_tests


def extract_code_from_response(response: str) -> str:
    """Extract Python code from model response, handling markdown code blocks."""
    if not response:
        return ""

    # Try to find ```python ... ``` block first
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ``` block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, return the whole response stripped
    return response.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """verl reward interface. Returns 1.0 if all tests pass, 0.0 otherwise."""
    code = extract_code_from_response(solution_str)
    if not code:
        return 0.0

    try:
        test_cases = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not test_cases:
        return 0.0

    timeout = 5.0
    if extra_info and "timeout" in extra_info:
        timeout = extra_info["timeout"]

    all_pass, _, _ = run_all_tests(code, test_cases, timeout=timeout)
    return 1.0 if all_pass else 0.0
