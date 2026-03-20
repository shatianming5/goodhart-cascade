"""Safe code execution sandbox for testing Python solutions."""

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    error: str = ""


def _limit_memory(max_mb):
    """Return a preexec_fn that sets memory limit."""
    def limiter():
        try:
            import resource
            limit_bytes = max_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ImportError, ValueError, OSError):
            pass
    return limiter


def execute_code(
    code: str, stdin: str = "", timeout: float = 5.0, max_memory_mb: int = 512
) -> ExecutionResult:
    """Execute Python code in a fresh subprocess (not fork) with timeout."""
    if not code or not code.strip():
        return ExecutionResult(returncode=1, error="Empty code")

    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["python3", tmp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit_memory(max_memory_mb),
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(timed_out=True, returncode=-1, error="Timeout")
    except Exception as e:
        return ExecutionResult(returncode=1, error=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_test_case(
    code: str, input_str: str, expected_output: str,
    timeout: float = 5.0, max_memory_mb: int = 512
) -> bool:
    """Run code with given input and check if output matches expected."""
    result = execute_code(code, stdin=input_str, timeout=timeout, max_memory_mb=max_memory_mb)
    if result.timed_out or result.returncode != 0:
        return False
    return result.stdout.strip() == expected_output.strip()


def run_all_tests(
    code: str, test_cases: list, timeout: float = 5.0,
    max_memory_mb: int = 512
) -> tuple:
    """Run code against all test cases with early exit on first failure."""
    if not test_cases:
        return False, 0, 0
    passed = 0
    total = len(test_cases)
    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        if run_test_case(code, inp, out, timeout=timeout, max_memory_mb=max_memory_mb):
            passed += 1
        else:
            return False, passed, total
    return True, passed, total
