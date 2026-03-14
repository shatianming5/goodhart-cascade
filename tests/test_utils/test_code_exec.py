"""Tests for code execution sandbox."""

import pytest

from goodhart.utils.code_exec import ExecutionResult, execute_code, run_all_tests, run_test_case


class TestExecuteCode:
    def test_simple_output(self):
        result = execute_code("print('hello')")
        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert not result.timed_out

    def test_stdin_input(self):
        code = "x = input()\nprint(x.upper())"
        result = execute_code(code, stdin="hello")
        assert result.stdout.strip() == "HELLO"

    def test_timeout(self):
        result = execute_code("while True: pass", timeout=1.0)
        assert result.timed_out

    def test_runtime_error(self):
        result = execute_code("raise ValueError('boom')")
        assert result.returncode != 0
        assert "ValueError" in result.error

    def test_syntax_error(self):
        result = execute_code("def f(\n")
        assert result.returncode != 0

    def test_empty_code(self):
        result = execute_code("")
        assert result.returncode != 0
        assert "Empty code" in result.error

    def test_empty_whitespace_code(self):
        result = execute_code("   \n  ")
        assert result.returncode != 0

    def test_memory_bomb_does_not_crash(self):
        # This should timeout or error, not crash the test process
        result = execute_code("x = [0] * (10**9)", timeout=2.0)
        assert result.timed_out or result.returncode != 0

    def test_multiline_output(self):
        code = "for i in range(3):\n    print(i)"
        result = execute_code(code)
        assert result.stdout.strip() == "0\n1\n2"


class TestRunTestCase:
    def test_correct(self):
        code = "n = int(input())\nprint(n * 2)"
        assert run_test_case(code, "3", "6")

    def test_wrong_output(self):
        code = "n = int(input())\nprint(n + 1)"
        assert not run_test_case(code, "3", "6")

    def test_timeout_returns_false(self):
        assert not run_test_case("while True: pass", "1", "2", timeout=1.0)

    def test_error_returns_false(self):
        assert not run_test_case("raise Exception()", "1", "2")

    def test_whitespace_tolerance(self):
        code = "print('hello  ')"
        assert run_test_case(code, "", "hello")


class TestRunAllTests:
    def test_all_pass(self, sample_test_cases):
        code = "n = int(input())\nprint(n * 2)"
        all_pass, passed, total = run_all_tests(code, sample_test_cases)
        assert all_pass
        assert passed == total == 3

    def test_partial_pass(self):
        code = "n = int(input())\nprint(n * 2)"
        cases = [
            {"input": "3", "output": "6"},
            {"input": "5", "output": "99"},  # wrong expected
        ]
        all_pass, passed, total = run_all_tests(code, cases)
        assert not all_pass
        assert passed == 1
        assert total == 2

    def test_empty_cases(self):
        all_pass, passed, total = run_all_tests("print(1)", [])
        assert not all_pass
        assert passed == 0
        assert total == 0

    def test_all_fail(self):
        code = "print('wrong')"
        cases = [{"input": "", "output": "right"}]
        all_pass, passed, total = run_all_tests(code, cases)
        assert not all_pass
        assert passed == 0
