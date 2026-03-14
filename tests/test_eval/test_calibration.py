"""Tests for calibration evaluator using mock generate functions."""

import math
import pytest

from goodhart.eval.calibration import CalibrationEvaluator


def make_problems():
    return [
        {
            "id": "p1",
            "question": "Print double of input integer.",
            "test_cases": [{"input": "3", "output": "6"}, {"input": "5", "output": "10"}],
            "difficulty": "EASY",
            "starter_code": "",
        },
        {
            "id": "p2",
            "question": "Print triple of input integer.",
            "test_cases": [{"input": "2", "output": "6"}, {"input": "3", "output": "9"}],
            "difficulty": "MEDIUM",
            "starter_code": "",
        },
    ]


class TestCalibrationEvaluator:
    def test_logprob_evaluation(self):
        problems = make_problems()
        evaluator = CalibrationEvaluator(problems, n_samples=2)

        call_count = [0]

        def mock_generate(prompt, return_logprobs=False):
            call_count[0] += 1
            if return_logprobs:
                # Return high confidence
                return "Yes", {"Yes": math.log(0.9), "No": math.log(0.1)}
            # Return correct code for first problem
            if "double" in prompt.lower():
                return "```python\nn = int(input())\nprint(n * 2)\n```"
            return "```python\nn = int(input())\nprint(n * 3)\n```"

        result = evaluator.evaluate_logprob(mock_generate)
        assert "ece" in result
        assert "pass_rate" in result
        assert 0.0 <= result["ece"] <= 1.0
        assert result["n_problems"] == 2

    def test_sampling_evaluation(self):
        problems = make_problems()[:1]  # just one problem for speed
        evaluator = CalibrationEvaluator(problems, n_samples=4)

        def mock_generate(prompt, return_logprobs=False):
            return "```python\nn = int(input())\nprint(n * 2)\n```"

        result = evaluator.evaluate_sampling(mock_generate)
        assert "ece" in result
        assert result["pass_rate"] == 1.0  # all samples correct

    def test_evaluate_combined(self):
        problems = make_problems()[:1]
        evaluator = CalibrationEvaluator(problems, n_samples=2)

        def mock_generate(prompt, return_logprobs=False):
            if return_logprobs:
                return "Yes", {"Yes": math.log(0.8), "No": math.log(0.2)}
            return "```python\nn = int(input())\nprint(n * 2)\n```"

        result = evaluator.evaluate(mock_generate)
        assert "ece_logprob" in result
        assert "ece_sampling" in result
        assert "overconfidence_rate" in result
        assert "by_difficulty" in result

    def test_failing_model(self):
        problems = make_problems()[:1]
        evaluator = CalibrationEvaluator(problems, n_samples=2)

        def mock_generate(prompt, return_logprobs=False):
            if return_logprobs:
                return "Yes", {"Yes": math.log(0.95), "No": math.log(0.05)}
            return "```python\nprint('wrong')\n```"

        result = evaluator.evaluate_logprob(mock_generate)
        assert result["pass_rate"] == 0.0
        assert result["mean_confidence"] > 0.9  # overconfident

    def test_extract_logprob_confidence(self):
        problems = make_problems()[:1]
        evaluator = CalibrationEvaluator(problems)

        conf = evaluator._extract_logprob_confidence(
            {"Yes": math.log(0.7), "No": math.log(0.3)}
        )
        assert conf == pytest.approx(0.7, abs=0.01)
