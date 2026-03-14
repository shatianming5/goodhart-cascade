"""Shared fixtures for goodhart-cascade tests."""

import pytest


@pytest.fixture
def simple_correct_code():
    """A simple correct Python solution that reads input and prints output."""
    return "n = int(input())\nprint(n * 2)"


@pytest.fixture
def simple_wrong_code():
    """A wrong Python solution."""
    return "n = int(input())\nprint(n + 1)"


@pytest.fixture
def timeout_code():
    """Code that runs forever."""
    return "while True: pass"


@pytest.fixture
def error_code():
    """Code that raises an error."""
    return "raise ValueError('boom')"


@pytest.fixture
def sample_test_cases():
    """Sample test cases for a doubling problem."""
    return [
        {"input": "3", "output": "6"},
        {"input": "5", "output": "10"},
        {"input": "0", "output": "0"},
    ]


@pytest.fixture
def sample_problem(sample_test_cases):
    """A complete sample problem dict."""
    return {
        "id": "test_001",
        "question": "Given an integer n, print n*2.",
        "test_cases": sample_test_cases,
        "difficulty": "EASY",
        "starter_code": "",
    }


@pytest.fixture
def high_quality_code():
    """Well-written Python code for quality metrics testing."""
    return '''def factorial(n: int) -> int:
    """Compute factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''


@pytest.fixture
def low_quality_code():
    """Poorly written code with duplication and complexity."""
    return '''def f(x):
    if x==1:
        a=1
        b=2
        c=a+b
        return c
    if x==2:
        a=1
        b=2
        c=a+b
        return c+1
    if x==3:
        a=1
        b=2
        c=a+b
        return c+2
    if x==4:
        a=1
        b=2
        c=a+b
        return c+3
    return 0
'''


@pytest.fixture
def sample_eval_results():
    """Sample evaluation results across checkpoints."""
    return [
        {"step": 0, "ece": 0.05, "quality": 0.8, "shortcut_rate": 0.02, "pass_rate": 0.3},
        {"step": 100, "ece": 0.08, "quality": 0.75, "shortcut_rate": 0.05, "pass_rate": 0.5},
        {"step": 200, "ece": 0.15, "quality": 0.65, "shortcut_rate": 0.12, "pass_rate": 0.7},
        {"step": 300, "ece": 0.25, "quality": 0.55, "shortcut_rate": 0.25, "pass_rate": 0.8},
        {"step": 400, "ece": 0.35, "quality": 0.45, "shortcut_rate": 0.40, "pass_rate": 0.85},
    ]
