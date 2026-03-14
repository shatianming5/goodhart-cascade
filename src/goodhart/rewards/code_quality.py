"""Code quality reward based on pylint, cyclomatic complexity, and duplication."""

from __future__ import annotations

import re
import tempfile
import os
from io import StringIO


def compute_quality_score(code: str) -> float:
    """Compute overall code quality score normalized to [0, 1].

    Weighted average of:
    - pylint score (40%)
    - cyclomatic complexity (30%)
    - duplication ratio (30%)
    """
    if not code or not code.strip():
        return 0.0

    # Verify code is parseable Python
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return 0.0

    pylint_s = run_pylint_score(code) / 10.0  # normalize to [0, 1]
    cc_s = compute_cyclomatic(code)
    dup_s = 1.0 - compute_duplication(code)  # lower duplication = higher score

    pylint_s = max(0.0, min(1.0, pylint_s))
    cc_s = max(0.0, min(1.0, cc_s))
    dup_s = max(0.0, min(1.0, dup_s))

    return 0.4 * pylint_s + 0.3 * cc_s + 0.3 * dup_s


def run_pylint_score(code: str) -> float:
    """Run pylint on code string, return score out of 10."""
    if not code or not code.strip():
        return 0.0

    try:
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            output = StringIO()
            reporter = TextReporter(output)
            results = Run(
                [tmp_path, "--disable=all", "--enable=C,R,W,E",
                 "--max-line-length=120"],
                reporter=reporter,
                exit=False,
            )
            score = results.linter.stats.global_note
            return max(0.0, min(10.0, score))
        finally:
            os.unlink(tmp_path)
    except Exception:
        return 0.0


def compute_cyclomatic(code: str) -> float:
    """Compute normalized cyclomatic complexity score.

    Uses radon. Returns score in [0, 1] where 1 = simple, 0 = very complex.
    """
    if not code or not code.strip():
        return 0.0

    try:
        from radon.complexity import cc_visit

        blocks = cc_visit(code)
        if not blocks:
            return 1.0  # no complex structures = simple

        avg_cc = sum(b.complexity for b in blocks) / len(blocks)
        # Map: CC 1-5 → 1.0, CC 10 → 0.5, CC 20+ → 0.0
        score = max(0.0, 1.0 - (avg_cc - 1) / 19.0)
        return score
    except Exception:
        return 0.0


def compute_duplication(code: str) -> float:
    """Compute line-level duplication ratio.

    Returns fraction of lines that are duplicates (0.0 = no duplication).
    """
    if not code or not code.strip():
        return 0.0

    lines = [
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return 0.0

    seen: dict[str, int] = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1

    duplicated = sum(count - 1 for count in seen.values() if count > 1)
    return duplicated / len(lines)
