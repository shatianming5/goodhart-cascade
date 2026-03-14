"""Code quality evaluation with 8 sub-dimensions."""

from __future__ import annotations

import ast
import re
from typing import Any

from goodhart.rewards.code_quality import compute_cyclomatic, compute_duplication, run_pylint_score
from goodhart.rewards.test_passing import extract_code_from_response


class CodeQualityEvaluator:
    """Evaluate code quality across 8 sub-dimensions."""

    def __init__(self, problems: list[dict]):
        self.problems = problems

    def analyze_single(self, code: str) -> dict:
        """Analyze a single code sample across 8 dimensions."""
        if not code or not code.strip():
            return self._empty_metrics()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._empty_metrics()

        return {
            "pylint_score": run_pylint_score(code),
            "cyclomatic_complexity": self._raw_cyclomatic(code),
            "cognitive_complexity": self._cognitive_complexity(tree),
            "duplication_ratio": compute_duplication(code),
            "lines_of_code": self._count_loc(code),
            "comment_ratio": self._comment_ratio(code),
            "type_hint_ratio": self._type_hint_ratio(tree),
            "error_handling_ratio": self._error_handling_ratio(tree),
        }

    def evaluate(self, generate_fn: Any) -> dict:
        """Generate code for each problem, analyze quality, aggregate results."""
        all_metrics = []

        for prob in self.problems:
            prompt = self._make_prompt(prob)
            try:
                response = generate_fn(prompt)
                if isinstance(response, tuple):
                    response = response[0]
            except Exception:
                response = ""

            code = extract_code_from_response(response)
            metrics = self.analyze_single(code)
            all_metrics.append(metrics)

        return self._aggregate(all_metrics)

    def _empty_metrics(self) -> dict:
        return {
            "pylint_score": 0.0,
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "duplication_ratio": 0.0,
            "lines_of_code": 0,
            "comment_ratio": 0.0,
            "type_hint_ratio": 0.0,
            "error_handling_ratio": 0.0,
        }

    def _raw_cyclomatic(self, code: str) -> float:
        try:
            from radon.complexity import cc_visit

            blocks = cc_visit(code)
            if not blocks:
                return 1.0
            return sum(b.complexity for b in blocks) / len(blocks)
        except Exception:
            return 0.0

    def _cognitive_complexity(self, tree: ast.AST) -> float:
        """Simplified cognitive complexity: count nesting depth * branching."""
        score = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                # Approximate nesting level
                score += 1
            if isinstance(node, (ast.BoolOp,)):
                score += len(node.values) - 1
        return float(score)

    def _count_loc(self, code: str) -> int:
        lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
        return len(lines)

    def _comment_ratio(self, code: str) -> float:
        lines = [l for l in code.splitlines() if l.strip()]
        if not lines:
            return 0.0
        comments = [l for l in lines if l.strip().startswith("#")]
        # Also count docstrings
        try:
            tree = ast.parse(code)
            docstrings = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, (ast.Constant,))
                            and isinstance(node.body[0].value.value, str)):
                        docstrings += 1
        except SyntaxError:
            docstrings = 0
        return (len(comments) + docstrings) / len(lines)

    def _type_hint_ratio(self, tree: ast.AST) -> float:
        """Fraction of function arguments that have type annotations."""
        total_args = 0
        annotated_args = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                all_args = args.args + args.posonlyargs + args.kwonlyargs
                total_args += len(all_args)
                annotated_args += sum(1 for a in all_args if a.annotation is not None)
                # Count return annotation
                if total_args > 0:
                    total_args += 1
                    if node.returns is not None:
                        annotated_args += 1

        if total_args == 0:
            return 0.0
        return annotated_args / total_args

    def _error_handling_ratio(self, tree: ast.AST) -> float:
        """Ratio of try/except blocks to total function count."""
        funcs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        tries = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
        if funcs == 0:
            return 0.0
        return min(1.0, tries / funcs)

    def _aggregate(self, all_metrics: list[dict]) -> dict:
        if not all_metrics:
            return self._empty_metrics()

        result = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            result[key] = sum(values) / len(values)
        result["n_samples"] = len(all_metrics)
        return result

    def _make_prompt(self, problem: dict) -> str:
        parts = [
            "Solve the following programming problem in Python.",
            "",
            problem["question"],
        ]
        if problem.get("starter_code"):
            parts += ["", "Starter code:", "", problem["starter_code"]]
        parts += ["", "Provide your solution in a Python code block."]
        return "\n".join(parts)
