"""ClassEval data loading for code quality evaluation.

Dataset: FudanSELab/ClassEval on HuggingFace
100 class-level Python code generation tasks with ~410 methods.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def load_classeval() -> list[dict]:
    """Load ClassEval problems from HuggingFace.

    Returns:
        List of formatted problem dicts for code quality evaluation.
    """
    from datasets import load_dataset

    ds = load_dataset("FudanSELab/ClassEval", split="test")

    problems = []
    for row in ds:
        problem = _format_problem(row)
        if problem:
            problems.append(problem)

    logger.info("Loaded %d ClassEval problems", len(problems))
    return problems


def _format_problem(row: dict) -> dict | None:
    """Format a ClassEval row into our standard problem format."""
    # ClassEval has class-level tasks with test cases
    task_id = row.get("task_id", "")
    skeleton = row.get("skeleton", "")
    test_code = row.get("test", "")
    solution = row.get("solution_code", "")

    # Build the prompt from class description and method descriptions
    class_name = row.get("class_name", "")
    class_desc = row.get("class_description", "")

    if not class_desc and not skeleton:
        return None

    # Build question from available fields
    parts = []
    if class_desc:
        parts.append(class_desc)

    # Add method descriptions if available
    methods_info = row.get("methods_info", [])
    if isinstance(methods_info, str):
        try:
            methods_info = json.loads(methods_info)
        except (json.JSONDecodeError, TypeError):
            methods_info = []

    if methods_info:
        parts.append("\nMethods to implement:")
        for m in methods_info:
            if isinstance(m, dict):
                name = m.get("method_name", "")
                desc = m.get("method_description", "")
                if name:
                    parts.append(f"- {name}: {desc}")

    question = "\n".join(parts) if parts else f"Implement class {class_name}"

    # Parse test cases - ClassEval uses unittest format
    test_cases = _extract_test_cases(test_code, skeleton, solution)

    return {
        "id": task_id,
        "question": question,
        "test_cases": test_cases,
        "difficulty": "MEDIUM",
        "starter_code": skeleton,
        "source": "classeval",
        "class_name": class_name,
        "test_code": test_code,
    }


def _extract_test_cases(test_code: str, skeleton: str, solution: str) -> list[dict]:
    """Extract simple test cases from ClassEval test code.

    ClassEval uses unittest assertions rather than stdin/stdout,
    so we store them as code-based test cases.
    """
    if not test_code:
        return []

    # For ClassEval, test cases are unittest-based, not stdin/stdout.
    # Store the test code as a single test case for code execution.
    return [{"test_code": test_code, "type": "unittest"}]
