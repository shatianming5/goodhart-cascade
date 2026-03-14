"""Calibration evaluation: logprob-based and sampling-based methods."""

from __future__ import annotations

import json
import math
from typing import Any

from goodhart.utils.code_exec import run_all_tests
from goodhart.utils.metrics import compute_ece, compute_overconfidence_rate, compute_by_difficulty


class CalibrationEvaluator:
    """Evaluate model calibration using logprob and sampling methods."""

    def __init__(self, problems: list[dict], n_samples: int = 8):
        self.problems = problems
        self.n_samples = n_samples

    def evaluate_logprob(self, generate_fn: Any) -> dict:
        """Evaluate calibration using logprob method.

        generate_fn(prompt, return_logprobs=True) -> (response, logprobs_dict)
        logprobs_dict should have token logprobs for "Yes"/"No" tokens.
        """
        confidences = []
        outcomes = []

        for prob in self.problems:
            prompt = self._make_calibration_prompt(prob)
            try:
                response, logprobs = generate_fn(prompt, return_logprobs=True)
                confidence = self._extract_logprob_confidence(logprobs)
            except Exception:
                confidence = 0.5

            # Check if model can solve the problem
            code_prompt = self._make_code_prompt(prob)
            try:
                code_response = generate_fn(code_prompt)
                if isinstance(code_response, tuple):
                    code_response = code_response[0]
            except Exception:
                code_response = ""

            from goodhart.rewards.test_passing import extract_code_from_response

            code = extract_code_from_response(code_response)
            all_pass, _, _ = run_all_tests(code, prob.get("test_cases", []))
            outcome = 1.0 if all_pass else 0.0

            confidences.append(confidence)
            outcomes.append(outcome)

        return self._compute_results(confidences, outcomes, "logprob")

    def evaluate_sampling(self, generate_fn: Any) -> dict:
        """Evaluate calibration using multi-sample method.

        generate_fn(prompt) -> response_str
        Confidence = fraction of samples that produce correct code.
        """
        results = []

        for prob in self.problems:
            prompt = self._make_code_prompt(prob)
            pass_count = 0

            for _ in range(self.n_samples):
                try:
                    response = generate_fn(prompt)
                    if isinstance(response, tuple):
                        response = response[0]
                except Exception:
                    response = ""

                from goodhart.rewards.test_passing import extract_code_from_response

                code = extract_code_from_response(response)
                all_pass, _, _ = run_all_tests(code, prob.get("test_cases", []))
                if all_pass:
                    pass_count += 1

            empirical_pass_rate = pass_count / self.n_samples
            # For sampling method, confidence IS the pass rate
            results.append({
                "confidence": empirical_pass_rate,
                "pass_rate": empirical_pass_rate,
                "difficulty": prob.get("difficulty", "unknown"),
            })

        confidences = [r["confidence"] for r in results]
        outcomes = [r["pass_rate"] for r in results]
        base = self._compute_results(confidences, outcomes, "sampling")
        base["by_difficulty"] = compute_by_difficulty(results)
        return base

    def evaluate(self, generate_fn: Any) -> dict:
        """Run both evaluation methods and merge results."""
        logprob_results = self.evaluate_logprob(generate_fn)
        sampling_results = self.evaluate_sampling(generate_fn)
        return {
            "ece_logprob": logprob_results["ece"],
            "ece_sampling": sampling_results["ece"],
            "overconfidence_rate": logprob_results["overconfidence_rate"],
            "pass_rate": logprob_results["pass_rate"],
            "mean_confidence": logprob_results["mean_confidence"],
            "by_difficulty": sampling_results.get("by_difficulty", {}),
        }

    def _compute_results(
        self, confidences: list[float], outcomes: list[float], method: str
    ) -> dict:
        ece = compute_ece(confidences, outcomes)
        overconf = compute_overconfidence_rate(confidences, outcomes)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        pass_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
        return {
            "method": method,
            "ece": ece,
            "overconfidence_rate": overconf,
            "mean_confidence": mean_conf,
            "pass_rate": pass_rate,
            "n_problems": len(confidences),
        }

    def _make_calibration_prompt(self, problem: dict) -> str:
        return (
            f"Can you solve this programming problem correctly?\n\n"
            f"{problem['question']}\n\n"
            f"Answer Yes or No, then explain briefly."
        )

    def _make_code_prompt(self, problem: dict) -> str:
        parts = [
            "Solve the following programming problem in Python.",
            "",
            problem["question"],
        ]
        if problem.get("starter_code"):
            parts += ["", "Use this starter code:", "", problem["starter_code"]]
        parts += ["", "Provide your solution in a Python code block."]
        return "\n".join(parts)

    def _extract_logprob_confidence(self, logprobs: dict) -> float:
        """Extract P(Yes) from logprobs dict."""
        yes_lp = logprobs.get("Yes", logprobs.get("yes", -10.0))
        no_lp = logprobs.get("No", logprobs.get("no", -10.0))

        yes_p = math.exp(yes_lp)
        no_p = math.exp(no_lp)
        total = yes_p + no_p
        if total == 0:
            return 0.5
        return yes_p / total
