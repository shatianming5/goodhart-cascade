"""Tests for code quality evaluator."""

import pytest

from goodhart.eval.code_quality import CodeQualityEvaluator


class TestAnalyzeSingle:
    def setup_method(self):
        self.evaluator = CodeQualityEvaluator([])

    def test_well_written_code(self, high_quality_code):
        metrics = self.evaluator.analyze_single(high_quality_code)
        assert metrics["pylint_score"] >= 0.0
        assert metrics["lines_of_code"] > 0
        assert metrics["type_hint_ratio"] > 0.0  # has type hints

    def test_poor_code(self, low_quality_code):
        metrics = self.evaluator.analyze_single(low_quality_code)
        assert metrics["duplication_ratio"] > 0.0  # has duplicates
        assert metrics["type_hint_ratio"] == 0.0  # no type hints

    def test_empty_code(self):
        metrics = self.evaluator.analyze_single("")
        assert metrics["pylint_score"] == 0.0
        assert metrics["lines_of_code"] == 0

    def test_syntax_error(self):
        metrics = self.evaluator.analyze_single("def f(\n")
        assert metrics["pylint_score"] == 0.0

    def test_comment_ratio(self):
        code = "# comment\nx = 1\n"
        metrics = self.evaluator.analyze_single(code)
        assert metrics["comment_ratio"] > 0.0

    def test_error_handling(self):
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
"""
        metrics = self.evaluator.analyze_single(code)
        assert metrics["error_handling_ratio"] > 0.0

    def test_cognitive_complexity(self):
        simple = "def f():\n    return 1\n"
        complex_code = """
def f(x):
    if x > 0:
        if x > 10:
            for i in range(x):
                if i % 2 == 0:
                    pass
    return x
"""
        m_simple = self.evaluator.analyze_single(simple)
        m_complex = self.evaluator.analyze_single(complex_code)
        assert m_complex["cognitive_complexity"] > m_simple["cognitive_complexity"]

    def test_all_8_dimensions(self, high_quality_code):
        metrics = self.evaluator.analyze_single(high_quality_code)
        expected_keys = [
            "pylint_score", "cyclomatic_complexity", "cognitive_complexity",
            "duplication_ratio", "lines_of_code", "comment_ratio",
            "type_hint_ratio", "error_handling_ratio",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


class TestEvaluate:
    def test_with_mock_model(self):
        problems = [
            {"question": "Print hello", "test_cases": [], "starter_code": ""},
        ]
        evaluator = CodeQualityEvaluator(problems)

        def mock_generate(prompt):
            return "```python\ndef hello():\n    print('hello')\n\nhello()\n```"

        result = evaluator.evaluate(mock_generate)
        assert "n_samples" in result
        assert result["n_samples"] == 1
        assert result["lines_of_code"] > 0
