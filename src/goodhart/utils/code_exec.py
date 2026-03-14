"""Safe code execution sandbox for testing Python solutions."""

from __future__ import annotations

import multiprocessing
import signal
import sys
import traceback
from dataclasses import dataclass, field
from io import StringIO


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    error: str = ""


def _set_memory_limit(max_mb: int):
    """Set memory limit for current process using resource.setrlimit."""
    try:
        import resource
        limit_bytes = max_mb * 1024 * 1024
        candidates = [resource.RLIMIT_AS, resource.RLIMIT_DATA]
        # RLIMIT_RSS is more effective on macOS
        if hasattr(resource, "RLIMIT_RSS"):
            candidates.append(resource.RLIMIT_RSS)
        for res in candidates:
            try:
                resource.setrlimit(res, (limit_bytes, limit_bytes))
                # Verify the limit actually took effect
                soft, _ = resource.getrlimit(res)
                if soft == limit_bytes:
                    return
            except (ValueError, OSError):
                continue
    except ImportError:
        pass  # Windows — no resource module


def _run_in_process(code: str, stdin: str, result_dict: dict, max_memory_mb: int = 512):
    """Target function for subprocess execution."""
    _set_memory_limit(max_memory_mb)
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdin = StringIO(stdin)
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals)
        result_dict["stdout"] = sys.stdout.getvalue()
        result_dict["stderr"] = sys.stderr.getvalue()
        result_dict["returncode"] = 0
    except Exception:
        result_dict["stdout"] = sys.stdout.getvalue() if hasattr(sys.stdout, "getvalue") else ""
        result_dict["stderr"] = traceback.format_exc()
        result_dict["returncode"] = 1
        result_dict["error"] = traceback.format_exc()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def execute_code(
    code: str, stdin: str = "", timeout: float = 5.0, max_memory_mb: int = 512
) -> ExecutionResult:
    """Execute Python code in a separate process with timeout and memory limit."""
    if not code or not code.strip():
        return ExecutionResult(returncode=1, error="Empty code")

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict["stdout"] = ""
    result_dict["stderr"] = ""
    result_dict["returncode"] = -1
    result_dict["error"] = ""

    proc = multiprocessing.Process(
        target=_run_in_process, args=(code, stdin, result_dict, max_memory_mb)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return ExecutionResult(timed_out=True, returncode=-1, error="Timeout")

    return ExecutionResult(
        stdout=result_dict.get("stdout", ""),
        stderr=result_dict.get("stderr", ""),
        returncode=result_dict.get("returncode", -1),
        error=result_dict.get("error", ""),
    )


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
    code: str, test_cases: list[dict], timeout: float = 5.0,
    max_memory_mb: int = 512
) -> tuple[bool, int, int]:
    """Run code against all test cases. Returns (all_pass, passed, total)."""
    if not test_cases:
        return False, 0, 0
    passed = 0
    total = len(test_cases)
    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        if run_test_case(code, inp, out, timeout=timeout, max_memory_mb=max_memory_mb):
            passed += 1
    return passed == total, passed, total
