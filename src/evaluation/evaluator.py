"""
Evaluation module for Goodhart Cascade experiments.

Measures:
1. Pass rate (MBPP)
2. ECE - Expected Calibration Error
3. 6 code quality dimensions:
   - Pylint score
   - Cognitive complexity
   - Comment ratio
   - Duplication ratio
   - Type hint ratio
   - Lines of code (LoC)
"""

import json
import math
import os
import re

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.rewards.reward_functions import (
    _measure_cognitive_complexity,
    _measure_comment_ratio,
    _measure_duplication_ratio,
    measure_type_hint_ratio,
    reward_pylint,
    _run_all_tests,
)


def load_mbpp(n: int = 100) -> list[dict]:
    """Load first n MBPP problems."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "code": item["code"],
            "test_list": item["test_list"],
        })
    return problems


def load_classeval(n: int = 50) -> list[dict]:
    """Load first n ClassEval problems."""
    ds = load_dataset("FudanSELab/ClassEval", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        problems.append({
            "task_id": item.get("task_id", f"classeval_{i}"),
            "prompt": item.get("skeleton", ""),
            "test": item.get("test", ""),
            "solution": item.get("solution_code", ""),
        })
    return problems


def build_mbpp_prompt(problem: dict) -> str:
    """Build MBPP code generation prompt."""
    return (
        "Write a Python function to solve the following problem. "
        "Only output the code.\n\n"
        f"Problem: {problem['prompt']}\n\n"
        "Solution:\n```python\n"
    )


def build_classeval_prompt(problem: dict) -> str:
    """Build ClassEval code generation prompt."""
    return (
        "Complete the following Python class. Only output the code.\n\n"
        f"{problem['prompt']}\n\n"
        "Solution:\n```python\n"
    )


def extract_code(response: str) -> str:
    """Extract code from model response."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            return code.split("```")[0].strip()
    return response.strip()


def run_mbpp_tests(code: str, test_list: list[str]) -> bool:
    """Run MBPP test assertions."""
    try:
        namespace = {}
        exec(code, namespace)
        for test in test_list:
            exec(test, namespace)
        return True
    except Exception:
        return False


def compute_ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    """
    if not confidences:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= c < hi) if i < n_bins - 1 else (lo <= c <= hi)
                for c in confidences]
        bin_size = sum(mask)
        if bin_size == 0:
            continue

        bin_conf = sum(c for c, m in zip(confidences, mask) if m) / bin_size
        bin_acc = sum(1 for c, m, correct in zip(confidences, mask, corrects) if m and correct) / bin_size
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


class Evaluator:
    """Full evaluation pipeline for a checkpoint."""

    def __init__(
        self,
        model_path: str,
        n_mbpp: int = 100,
        n_classeval: int = 50,
        n_samples: int = 8,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.n_samples = n_samples

        # Load eval datasets
        print("Loading evaluation datasets...")
        self.mbpp_problems = load_mbpp(n_mbpp)
        self.classeval_problems = load_classeval(n_classeval)

        # Initialize vLLM
        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
        )

        self.sampling_params = SamplingParams(
            n=n_samples,
            temperature=0.8,
            top_p=0.95,
            max_tokens=2048,
            stop=["```\n", "\n\n\n"],
        )

        # Greedy for calibration
        self.greedy_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=2048,
            stop=["```\n", "\n\n\n"],
        )

    def evaluate_mbpp(self) -> dict:
        """Evaluate on MBPP: pass rate + calibration."""
        prompts = [build_mbpp_prompt(p) for p in self.mbpp_problems]

        # Generate samples for pass@k
        print("Generating MBPP samples...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Generate greedy for calibration
        greedy_outputs = self.llm.generate(prompts, self.greedy_params)

        pass_at_1 = 0
        pass_at_k = 0
        confidences = []
        corrects = []

        for i, (problem, output, greedy_out) in enumerate(
            zip(self.mbpp_problems, outputs, greedy_outputs)
        ):
            codes = [extract_code(o.text) for o in output.outputs]
            results = [run_mbpp_tests(c, problem["test_list"]) for c in codes]

            # pass@1 (greedy)
            greedy_code = extract_code(greedy_out.outputs[0].text)
            greedy_pass = run_mbpp_tests(greedy_code, problem["test_list"])
            if greedy_pass:
                pass_at_1 += 1

            # pass@k
            if any(results):
                pass_at_k += 1

            # Calibration: use empirical pass rate as proxy for confidence
            empirical_pass_rate = sum(results) / len(results)
            confidences.append(empirical_pass_rate)
            corrects.append(greedy_pass)

        n = len(self.mbpp_problems)
        ece = compute_ece(confidences, corrects)

        return {
            "pass_at_1": pass_at_1 / n,
            "pass_at_k": pass_at_k / n,
            "ece": ece,
            "n_problems": n,
        }

    def evaluate_quality(self) -> dict:
        """Evaluate code quality on ClassEval (6 dimensions)."""
        prompts = [build_classeval_prompt(p) for p in self.classeval_problems]

        print("Generating ClassEval samples...")
        outputs = self.llm.generate(prompts, self.greedy_params)

        metrics = {
            "pylint_scores": [],
            "complexity_scores": [],
            "comment_ratios": [],
            "duplication_ratios": [],
            "type_hint_ratios": [],
            "loc": [],
        }

        for output in outputs:
            code = extract_code(output.outputs[0].text)
            if not code.strip():
                continue

            metrics["pylint_scores"].append(reward_pylint(code) * 10)  # Back to 0-10
            metrics["complexity_scores"].append(_measure_cognitive_complexity(code))
            metrics["comment_ratios"].append(_measure_comment_ratio(code) * 100)  # percent
            metrics["duplication_ratios"].append(_measure_duplication_ratio(code) * 100)
            metrics["type_hint_ratios"].append(measure_type_hint_ratio(code) * 100)
            metrics["loc"].append(len(code.strip().split("\n")))

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "pylint": safe_mean(metrics["pylint_scores"]),
            "complexity": safe_mean(metrics["complexity_scores"]),
            "comment_pct": safe_mean(metrics["comment_ratios"]),
            "duplication_pct": safe_mean(metrics["duplication_ratios"]),
            "type_hint_pct": safe_mean(metrics["type_hint_ratios"]),
            "avg_loc": safe_mean(metrics["loc"]),
            "n_samples": len(metrics["pylint_scores"]),
        }

    def evaluate_all(self) -> dict:
        """Run full evaluation."""
        print(f"\nEvaluating {self.model_path}")
        print("=" * 50)

        mbpp_results = self.evaluate_mbpp()
        quality_results = self.evaluate_quality()

        results = {
            "model_path": self.model_path,
            **mbpp_results,
            **quality_results,
        }

        print(f"\n--- Results ---")
        print(f"  Pass@1:       {results['pass_at_1']:.2%}")
        print(f"  Pass@{self.n_samples}:       {results['pass_at_k']:.2%}")
        print(f"  ECE:          {results['ece']:.4f}")
        print(f"  Pylint:       {results['pylint']:.1f}/10")
        print(f"  Complexity:   {results['complexity']:.1f}")
        print(f"  Comment%:     {results['comment_pct']:.1f}%")
        print(f"  Duplication%: {results['duplication_pct']:.1f}%")
        print(f"  TypeHint%:    {results['type_hint_pct']:.1f}%")
        print(f"  Avg LoC:      {results['avg_loc']:.0f}")

        return results

    def cleanup(self):
        """Free vLLM resources."""
        del self.llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_checkpoints(
    experiment_dir: str,
    n_mbpp: int = 100,
    n_classeval: int = 50,
    gpu_memory_utilization: float = 0.85,
):
    """Evaluate all checkpoints in an experiment directory."""
    checkpoints = sorted([
        d for d in os.listdir(experiment_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(experiment_dir, d))
    ], key=lambda x: int(x.split("-")[1]))

    print(f"Found {len(checkpoints)} checkpoints in {experiment_dir}")
    all_results = []

    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(experiment_dir, ckpt_name)
        step = int(ckpt_name.split("-")[1])

        evaluator = Evaluator(
            model_path=ckpt_path,
            n_mbpp=n_mbpp,
            n_classeval=n_classeval,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        results = evaluator.evaluate_all()
        results["step"] = step
        results["checkpoint"] = ckpt_name
        all_results.append(results)

        evaluator.cleanup()

    # Save results
    results_path = os.path.join(experiment_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--n-mbpp", type=int, default=100)
    parser.add_argument("--n-classeval", type=int, default=50)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    args = parser.parse_args()

    evaluate_checkpoints(
        args.experiment_dir,
        n_mbpp=args.n_mbpp,
        n_classeval=args.n_classeval,
        gpu_memory_utilization=args.gpu_util,
    )
