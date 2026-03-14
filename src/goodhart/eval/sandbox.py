"""Evaluation sandbox with batch execution and resource limits."""

from __future__ import annotations

from dataclasses import dataclass

from goodhart.utils.code_exec import ExecutionResult, execute_code, run_all_tests


@dataclass
class EvalResult:
    all_pass: bool = False
    passed: int = 0
    total: int = 0
    error: str = ""
    timed_out: bool = False


class EvalSandbox:
    """Evaluation sandbox with batch execution support and caching."""

    def __init__(self, timeout: float = 10.0, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._cache: dict[str, EvalResult] = {}

    def _cache_key(self, code: str, test_cases: list[dict]) -> str:
        import hashlib
        import json

        content = json.dumps({"code": code, "tests": test_cases}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def execute_single(self, code: str, test_cases: list[dict]) -> EvalResult:
        """Execute a single solution against test cases."""
        key = self._cache_key(code, test_cases)
        if key in self._cache:
            return self._cache[key]

        if not code or not code.strip():
            result = EvalResult(error="Empty code")
            self._cache[key] = result
            return result

        if not test_cases:
            result = EvalResult(error="No test cases")
            self._cache[key] = result
            return result

        all_pass, passed, total = run_all_tests(code, test_cases, timeout=self.timeout)
        result = EvalResult(all_pass=all_pass, passed=passed, total=total)
        self._cache[key] = result
        return result

    def execute_batch(
        self, code_list: list[str], test_cases_list: list[list[dict]]
    ) -> list[EvalResult]:
        """Execute multiple solutions against their respective test cases."""
        results = []
        for code, tests in zip(code_list, test_cases_list):
            results.append(self.execute_single(code, tests))
        return results

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
