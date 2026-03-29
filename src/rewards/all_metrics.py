"""
All quality metrics - computed for EVERY rollout regardless of reward config.
This is critical for theory verification (correlation matrix, variance ratio).
"""

import ast
import re
from collections import Counter

from radon.visitors import ComplexityVisitor

from src.rewards.reward_functions import (
    _measure_cognitive_complexity,
    _measure_comment_ratio,
    _measure_duplication_ratio,
    _run_all_tests,
    measure_type_hint_ratio,
    reward_pylint,
)


def compute_all_metrics(code: str, test_cases: dict | None = None) -> dict[str, float]:
    """
    Compute ALL 6+6 quality dimensions for a single code sample.
    Called on every rollout for theory verification logging.

    Returns dict with:
    - 5 constrained dims: test, pylint, complexity, comment, duplication
    - 6 unconstrained dims: type_hint, avg_func_length, magic_numbers,
      nesting_depth, dead_code, naming_length
    """
    metrics = {}

    # === Constrained dimensions (used in R1-R5 rewards) ===

    # Test pass (binary)
    if test_cases:
        metrics["test"] = 1.0 if _run_all_tests(code, test_cases) else 0.0
    else:
        metrics["test"] = 0.0

    # Pylint score [0, 1]
    metrics["pylint"] = reward_pylint(code)

    # Cognitive complexity (raw value, lower is better)
    metrics["complexity_raw"] = _measure_cognitive_complexity(code)
    metrics["complexity"] = max(0.0, 1.0 - metrics["complexity_raw"] / 20.0)

    # Comment ratio (raw percentage)
    metrics["comment_ratio_raw"] = _measure_comment_ratio(code)
    metrics["comment"] = min(metrics["comment_ratio_raw"] / 0.15, 1.0)

    # Duplication ratio (raw percentage)
    metrics["duplication_ratio_raw"] = _measure_duplication_ratio(code)
    metrics["duplication"] = max(0.0, 1.0 - metrics["duplication_ratio_raw"] / 0.3)

    # === Unconstrained dimensions (never in reward, for escape detection) ===

    # Type hint coverage
    metrics["type_hint"] = measure_type_hint_ratio(code)

    # Average function length
    metrics["avg_func_length"] = _measure_avg_func_length(code)

    # Magic number count
    metrics["magic_numbers"] = _count_magic_numbers(code)

    # Max nesting depth
    metrics["nesting_depth"] = _measure_nesting_depth(code)

    # Dead code ratio
    metrics["dead_code"] = _measure_dead_code_ratio(code)

    # Average variable name length
    metrics["naming_length"] = _measure_avg_naming_length(code)

    return metrics


def _measure_avg_func_length(code: str) -> float:
    """Average number of lines per function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return 0.0

    lengths = []
    for func in functions:
        if hasattr(func, "end_lineno") and func.end_lineno:
            lengths.append(func.end_lineno - func.lineno + 1)
        else:
            # Estimate from body
            lengths.append(len(func.body) * 2)

    return sum(lengths) / len(lengths) if lengths else 0.0


def _count_magic_numbers(code: str) -> float:
    """Count magic numbers (numeric literals that aren't 0, 1, 2, -1)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    ALLOWED = {0, 1, 2, -1, 0.0, 1.0, 0.5, 100, 10}
    count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in ALLOWED:
                count += 1

    return float(count)


def _measure_nesting_depth(code: str) -> float:
    """Maximum nesting depth of control structures."""
    lines = code.split("\n")
    max_depth = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            depth = indent // 4  # assume 4-space indent
            max_depth = max(max_depth, depth)
    return float(max_depth)


def _measure_dead_code_ratio(code: str) -> float:
    """Estimate dead code ratio (unreachable code after return/break/continue)."""
    lines = code.strip().split("\n")
    if not lines:
        return 0.0

    dead_lines = 0
    after_return = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())

        if after_return:
            # Check if same or deeper indentation (dead code)
            if indent >= return_indent:
                # But "except", "elif", "else", "finally" at same level are OK
                if not any(stripped.startswith(kw) for kw in
                           ["except", "elif", "else:", "finally:", "def ", "class "]):
                    dead_lines += 1
                    continue
            after_return = False

        if any(stripped.startswith(kw) for kw in ["return ", "return\n", "break", "continue", "raise "]):
            after_return = True
            return_indent = indent + 1  # code at same or deeper level is dead

    total = sum(1 for l in lines if l.strip())
    return dead_lines / total if total > 0 else 0.0


def _measure_avg_naming_length(code: str) -> float:
    """Average variable/function name length."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(len(node.id))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(len(node.name))
        elif isinstance(node, ast.arg):
            names.append(len(node.arg))

    return sum(names) / len(names) if names else 0.0
