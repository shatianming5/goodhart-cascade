"""Tests for code quality reward."""

import pytest

from goodhart.rewards.code_quality import (
    compute_cyclomatic,
    compute_duplication,
    compute_quality_score,
    run_pylint_score,
)


class TestRunPylintScore:
    def test_clean_code(self):
        code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
        score = run_pylint_score(code)
        assert score >= 5.0  # reasonably clean

    def test_empty_code(self):
        assert run_pylint_score("") == 0.0

    def test_unparseable(self):
        assert run_pylint_score("def f(\n") == 0.0


class TestComputeCyclomatic:
    def test_simple_function(self):
        code = "def f(x):\n    return x + 1\n"
        score = compute_cyclomatic(code)
        assert score >= 0.9  # CC=1, very simple

    def test_complex_function(self):
        # Many branches → lower score
        code = "def f(x):\n" + "".join(
            f"    if x == {i}:\n        return {i}\n" for i in range(15)
        )
        score = compute_cyclomatic(code)
        assert score < 0.5

    def test_empty(self):
        assert compute_cyclomatic("") == 0.0

    def test_no_functions(self):
        code = "x = 1\ny = 2\n"
        score = compute_cyclomatic(code)
        assert score == 1.0  # no complex blocks


class TestComputeDuplication:
    def test_no_duplication(self):
        code = "x = 1\ny = 2\nz = 3\n"
        assert compute_duplication(code) == 0.0

    def test_full_duplication(self):
        code = "x = 1\nx = 1\nx = 1\n"
        ratio = compute_duplication(code)
        assert ratio == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_empty(self):
        assert compute_duplication("") == 0.0

    def test_ignores_comments(self):
        code = "# comment\n# comment\nx = 1\n"
        assert compute_duplication(code) == 0.0


class TestComputeQualityScore:
    def test_good_code(self, high_quality_code):
        score = compute_quality_score(high_quality_code)
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    def test_bad_code(self, low_quality_code):
        score = compute_quality_score(low_quality_code)
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert compute_quality_score("") == 0.0

    def test_unparseable(self):
        assert compute_quality_score("}{}{") == 0.0

    def test_range(self):
        code = "def f():\n    return 42\n"
        score = compute_quality_score(code)
        assert 0.0 <= score <= 1.0
