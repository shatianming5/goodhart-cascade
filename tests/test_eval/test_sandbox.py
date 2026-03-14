"""Tests for evaluation sandbox."""

import pytest

from goodhart.eval.sandbox import EvalSandbox


class TestEvalSandbox:
    def test_single_correct(self, sample_test_cases):
        sb = EvalSandbox(timeout=5.0)
        code = "n = int(input())\nprint(n * 2)"
        result = sb.execute_single(code, sample_test_cases)
        assert result.all_pass
        assert result.passed == result.total == 3

    def test_single_wrong(self, sample_test_cases):
        sb = EvalSandbox(timeout=5.0)
        code = "print('wrong')"
        result = sb.execute_single(code, sample_test_cases)
        assert not result.all_pass

    def test_empty_code(self, sample_test_cases):
        sb = EvalSandbox()
        result = sb.execute_single("", sample_test_cases)
        assert not result.all_pass
        assert "Empty" in result.error

    def test_no_tests(self):
        sb = EvalSandbox()
        result = sb.execute_single("print(1)", [])
        assert not result.all_pass
        assert "No test" in result.error

    def test_batch_execution(self):
        sb = EvalSandbox(timeout=5.0)
        codes = ["print(int(input())*2)", "print('wrong')"]
        tests = [
            [{"input": "3", "output": "6"}],
            [{"input": "3", "output": "6"}],
        ]
        results = sb.execute_batch(codes, tests)
        assert len(results) == 2
        assert results[0].all_pass
        assert not results[1].all_pass

    def test_caching(self, sample_test_cases):
        sb = EvalSandbox(timeout=5.0)
        code = "n = int(input())\nprint(n * 2)"
        sb.execute_single(code, sample_test_cases)
        assert sb.cache_size == 1
        # Second call should use cache
        sb.execute_single(code, sample_test_cases)
        assert sb.cache_size == 1

    def test_clear_cache(self, sample_test_cases):
        sb = EvalSandbox(timeout=5.0)
        sb.execute_single("print(1)", [{"input": "", "output": "1"}])
        assert sb.cache_size == 1
        sb.clear_cache()
        assert sb.cache_size == 0
