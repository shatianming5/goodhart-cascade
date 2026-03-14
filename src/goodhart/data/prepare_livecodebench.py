"""LiveCodeBench data loading for calibration evaluation.

Dataset: livecodebench/code_generation_lite on HuggingFace
Continuously updated contamination-free benchmark from LeetCode/AtCoder/CodeForces.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def load_livecodebench(version_tag: str = "release_v6") -> list[dict]:
    """Load LiveCodeBench problems from HuggingFace.

    Args:
        version_tag: Dataset version (release_v5=880, release_v6=1055 problems).

    Returns:
        List of formatted problem dicts.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=version_tag,
        split="test",
    )

    problems = []
    for row in ds:
        problem = _format_problem(row)
        if problem:
            problems.append(problem)

    logger.info("Loaded %d LiveCodeBench problems (version=%s)", len(problems), version_tag)
    return problems


def _format_problem(row: dict) -> dict | None:
    """Format a LiveCodeBench row into our standard problem format."""
    question = row.get("question_content") or row.get("question", "")
    if not question:
        return None

    # Parse test cases from input_output field
    test_cases = []
    io_raw = row.get("input_output", "")
    if io_raw:
        try:
            io = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
            inputs = io.get("inputs", [])
            outputs = io.get("outputs", [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({
                    "input": str(inp).strip(),
                    "output": str(out).strip(),
                })
        except (json.JSONDecodeError, TypeError):
            pass

    if not test_cases:
        return None

    # Map difficulty
    difficulty = row.get("difficulty", "MEDIUM")
    if isinstance(difficulty, (int, float)):
        if difficulty <= 1:
            difficulty = "EASY"
        elif difficulty <= 2:
            difficulty = "MEDIUM"
        else:
            difficulty = "HARD"

    return {
        "id": row.get("question_id", row.get("id", "")),
        "question": question,
        "test_cases": test_cases,
        "difficulty": difficulty,
        "starter_code": row.get("starter_code", ""),
        "source": "livecodebench",
    }


def filter_livecodebench(
    problems: list[dict], min_tests: int = 3, max_problems: int = 500
) -> list[dict]:
    """Filter LiveCodeBench problems for evaluation."""
    filtered = [p for p in problems if len(p.get("test_cases", [])) >= min_tests]
    return filtered[:max_problems]
