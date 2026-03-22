"""
Reward functions for Goodhart Cascade experiments R1-R5.

Each reward component is normalized to [0, 1].
The composite reward is a weighted sum according to experiment config.
"""

import ast
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter
from typing import Any

import radon.complexity as radon_cc
from radon.visitors import ComplexityVisitor


# ---------------------------------------------------------------------------
# R1: Test-only reward
# ---------------------------------------------------------------------------

def reward_test(code: str, test_cases: dict, timeout: int = 10) -> float:
    """Binary reward: 1.0 if all tests pass, 0.0 otherwise."""
    return 1.0 if _run_all_tests(code, test_cases, timeout) else 0.0


# ---------------------------------------------------------------------------
# R2 addition: Pylint score
# ---------------------------------------------------------------------------

def reward_pylint(code: str) -> float:
    """Pylint score normalized to [0, 1]. Pylint gives 0-10, we divide by 10."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            [
                sys.executable, "-m", "pylint",
                tmp_path,
                "--disable=C0114,C0115,C0116",
                "--score=y",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        os.unlink(tmp_path)

        # Parse score from stdout or stderr
        combined = result.stdout + result.stderr
        score_match = re.search(r"rated at ([\d.-]+)/10", combined)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(score / 10.0, 1.0))

        return 0.5
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return 0.0


def reward_pylint_batch(codes: list[str]) -> list[float]:
    """Batch pylint evaluation using concurrent subprocesses for speed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(reward_pylint, c): i for i, c in enumerate(codes)}
        results = [0.0] * len(codes)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = 0.0
    return results


# ---------------------------------------------------------------------------
# R3 addition: Cognitive complexity
# ---------------------------------------------------------------------------

def reward_complexity(code: str, max_complexity: float = 20.0) -> float:
    """
    Cognitive complexity reward. Lower is better.
    cc=0 -> 1.0, cc>=max_complexity -> 0.0
    """
    try:
        cc = _measure_cognitive_complexity(code)
        return max(0.0, 1.0 - cc / max_complexity)
    except Exception:
        return 0.5


def _measure_cognitive_complexity(code: str) -> float:
    """Measure aggregate cognitive complexity of all functions in code."""
    try:
        results = ComplexityVisitor.from_code(code)
        if not results:
            return 0.0
        total = sum(block.complexity for block in results)
        return float(total)
    except Exception:
        # Fallback: count nesting depth and branches
        return _simple_complexity(code)


def _simple_complexity(code: str) -> float:
    """Simple complexity metric based on control flow keywords."""
    keywords = ["if ", "elif ", "else:", "for ", "while ", "try:", "except ",
                 "with ", "and ", "or ", "not "]
    score = 0
    lines = code.split("\n")
    for line in lines:
        stripped = line.strip()
        for kw in keywords:
            if stripped.startswith(kw) or f" {kw}" in stripped:
                indent = len(line) - len(line.lstrip())
                nesting = indent // 4
                score += 1 + nesting
                break
    return float(score)


# ---------------------------------------------------------------------------
# R4 addition: Comment ratio
# ---------------------------------------------------------------------------

def reward_comment(code: str, target_ratio: float = 0.15) -> float:
    """
    Comment ratio reward. Target is ~15% comment lines.
    ratio=0 -> 0.0, ratio>=target -> 1.0
    """
    ratio = _measure_comment_ratio(code)
    return min(ratio / target_ratio, 1.0) if target_ratio > 0 else 1.0


def _measure_comment_ratio(code: str) -> float:
    """Measure ratio of comment lines to total lines."""
    lines = code.strip().split("\n")
    if not lines:
        return 0.0

    total = len(lines)
    comment_lines = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Track docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                comment_lines += 1
                if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                    continue  # Single-line docstring
                in_docstring = True
                continue
            if stripped.startswith("#"):
                comment_lines += 1
                continue
        else:
            comment_lines += 1
            if docstring_char in stripped[3:] or stripped.endswith(docstring_char):
                in_docstring = False
            continue

    return comment_lines / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# R5 addition: Duplication ratio
# ---------------------------------------------------------------------------

def reward_duplication(code: str, max_dup: float = 0.3) -> float:
    """
    Duplication reward. Lower duplication is better.
    dup=0 -> 1.0, dup>=max_dup -> 0.0
    """
    dup = _measure_duplication_ratio(code)
    return max(0.0, 1.0 - dup / max_dup)


def _measure_duplication_ratio(code: str) -> float:
    """
    Measure code duplication as fraction of lines that are duplicates.
    Uses 3-line sliding window to detect copied blocks.
    """
    lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return 0.0

    # 3-line sliding window
    windows = []
    for i in range(len(lines) - 2):
        window = (lines[i], lines[i + 1], lines[i + 2])
        # Skip trivial windows (empty lines, single chars)
        if all(len(w) > 2 for w in window):
            windows.append(window)

    if not windows:
        return 0.0

    counts = Counter(windows)
    duplicated_windows = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated_windows / len(windows) if windows else 0.0


# ---------------------------------------------------------------------------
# Additional quality metrics (for evaluation, not reward)
# ---------------------------------------------------------------------------

def measure_type_hint_ratio(code: str) -> float:
    """Measure ratio of functions with type hints."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return 0.0

    typed = 0
    for func in functions:
        has_hint = func.returns is not None
        if not has_hint:
            for arg in func.args.args:
                if arg.annotation is not None:
                    has_hint = True
                    break
        if has_hint:
            typed += 1

    return typed / len(functions)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def compute_reward(
    code: str,
    test_cases: dict,
    weights: dict[str, float],
    timeout: int = 10,
) -> dict[str, float]:
    """
    Compute composite reward based on experiment config weights.

    Args:
        code: Generated code string
        test_cases: Dict with inputs/outputs for testing
        weights: Dict mapping reward component names to weights
            e.g. {"test": 0.7, "pylint": 0.3}

    Returns:
        Dict with individual scores and total reward
    """
    scores = {}

    if "test" in weights:
        scores["test"] = reward_test(code, test_cases, timeout)

    if "pylint" in weights:
        scores["pylint"] = reward_pylint(code)

    if "complexity" in weights:
        scores["complexity"] = reward_complexity(code)

    if "comment" in weights:
        scores["comment"] = reward_comment(code)

    if "duplication" in weights:
        scores["duplication"] = reward_duplication(code)

    total = sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)
    scores["total"] = total

    return scores


def compute_rewards_batch(
    codes: list[str],
    test_cases: dict,
    weights: dict[str, float],
    timeout: int = 10,
) -> list[dict[str, float]]:
    """
    Batch reward computation with parallelized pylint calls.
    Much faster than calling compute_reward one-by-one.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(codes)
    all_scores = [{"total": 0.0} for _ in range(n)]

    # Test reward: run sequentially (uses signal.alarm, not thread-safe)
    if "test" in weights:
        for i, code in enumerate(codes):
            all_scores[i]["test"] = reward_test(code, test_cases, timeout)

    # Pylint: run in parallel threads (subprocess-based, safe to parallelize)
    if "pylint" in weights:
        pylint_scores = reward_pylint_batch(codes)
        for i, s in enumerate(pylint_scores):
            all_scores[i]["pylint"] = s

    # Fast, CPU-only metrics: can compute sequentially without issue
    for i, code in enumerate(codes):
        if "complexity" in weights:
            all_scores[i]["complexity"] = reward_complexity(code)
        if "comment" in weights:
            all_scores[i]["comment"] = reward_comment(code)
        if "duplication" in weights:
            all_scores[i]["duplication"] = reward_duplication(code)

    # Compute totals
    for i in range(n):
        total = sum(weights.get(k, 0) * all_scores[i].get(k, 0) for k in weights)
        all_scores[i]["total"] = total

    return all_scores


# ---------------------------------------------------------------------------
# Test execution helper
# ---------------------------------------------------------------------------

def _run_all_tests(code: str, test_cases: dict, timeout: int = 10) -> bool:
    """Run all test cases against code. Returns True if all pass."""
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    fn_name = test_cases.get("fn_name", "")

    if fn_name:
        return _run_fn_tests(code, fn_name, inputs, outputs, timeout)
    else:
        return _run_stdio_tests(code, inputs, outputs, timeout)


def _run_fn_tests(
    code: str, fn_name: str, inputs: list, outputs: list, timeout: int
) -> bool:
    """Run function-call style tests."""
    namespace: dict[str, Any] = {}
    try:
        def handler(signum, frame):
            raise TimeoutError()
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        exec(code, namespace)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        fn = namespace.get(fn_name)
        if fn is None:
            return False

        for inp, expected in zip(inputs, outputs):
            signal.alarm(timeout)
            if not isinstance(inp, (list, tuple)):
                inp = [inp]
            result = fn(*inp)
            signal.alarm(0)
            if str(result).strip() != str(expected).strip():
                return False
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)


def _run_stdio_tests(
    code: str, inputs: list, outputs: list, timeout: int
) -> bool:
    """Run stdin/stdout style tests."""
    for inp, expected in zip(inputs, outputs):
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                input=str(inp),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            os.unlink(tmp_path)

            if result.stdout.strip() != str(expected).strip():
                return False
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return False
    return True
