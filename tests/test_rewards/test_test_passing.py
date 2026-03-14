"""Tests for binary test-passing reward."""

import json

import pytest

from goodhart.rewards.test_passing import compute_score, extract_code_from_response


class TestExtractCode:
    def test_python_code_block(self):
        text = "Here's the solution:\n```python\nprint('hello')\n```\nDone."
        assert extract_code_from_response(text) == "print('hello')"

    def test_generic_code_block(self):
        text = "Solution:\n```\nprint('hello')\n```"
        assert extract_code_from_response(text) == "print('hello')"

    def test_no_code_block(self):
        text = "print('hello')"
        assert extract_code_from_response(text) == "print('hello')"

    def test_empty_response(self):
        assert extract_code_from_response("") == ""

    def test_multiline_code_block(self):
        text = "```python\nx = 1\ny = 2\nprint(x + y)\n```"
        code = extract_code_from_response(text)
        assert "x = 1" in code
        assert "print(x + y)" in code

    def test_py_shorthand(self):
        text = "```py\nprint(42)\n```"
        assert extract_code_from_response(text) == "print(42)"


class TestComputeScore:
    def _gt(self, cases):
        return json.dumps(cases)

    def test_correct_code(self):
        code = "```python\nn = int(input())\nprint(n * 2)\n```"
        gt = self._gt([{"input": "3", "output": "6"}, {"input": "5", "output": "10"}])
        assert compute_score("taco", code, gt) == 1.0

    def test_wrong_code(self):
        code = "```python\nn = int(input())\nprint(n + 1)\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        assert compute_score("taco", code, gt) == 0.0

    def test_empty_response(self):
        gt = self._gt([{"input": "1", "output": "2"}])
        assert compute_score("taco", "", gt) == 0.0

    def test_invalid_ground_truth(self):
        assert compute_score("taco", "print(1)", "not json") == 0.0

    def test_empty_test_cases(self):
        assert compute_score("taco", "print(1)", "[]") == 0.0

    def test_partial_pass_returns_zero(self):
        code = "```python\nn = int(input())\nprint(n * 2)\n```"
        gt = self._gt([
            {"input": "3", "output": "6"},
            {"input": "5", "output": "99"},  # will fail
        ])
        assert compute_score("taco", code, gt) == 0.0

    def test_binary_all_must_pass(self):
        code = "```python\nprint(int(input()) * 2)\n```"
        gt = self._gt([
            {"input": "1", "output": "2"},
            {"input": "2", "output": "4"},
            {"input": "3", "output": "6"},
        ])
        assert compute_score("taco", code, gt) == 1.0
