"""Unified evaluation runner: loads model, runs all evaluators, saves results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodhart.eval.calibration import CalibrationEvaluator
from goodhart.eval.code_quality import CodeQualityEvaluator
from goodhart.eval.temptation import TemptationEvaluator


def load_vllm_model(model_path: str) -> Any:
    """Load a vLLM model for evaluation. Returns a generate function."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, trust_remote_code=True)
    default_params = SamplingParams(temperature=0.0, max_tokens=2048)

    def generate_fn(prompt: str, return_logprobs: bool = False):
        if return_logprobs:
            params = SamplingParams(
                temperature=0.0, max_tokens=64, logprobs=5
            )
            outputs = llm.generate([prompt], params)
            text = outputs[0].outputs[0].text

            # Extract Yes/No logprobs from first token
            lp_dict = {}
            if outputs[0].outputs[0].logprobs:
                first_token_lps = outputs[0].outputs[0].logprobs[0]
                for token_id, lp_info in first_token_lps.items():
                    decoded = lp_info.decoded_token.strip()
                    if decoded in ("Yes", "yes", "No", "no"):
                        lp_dict[decoded] = lp_info.logprob
            return text, lp_dict
        else:
            outputs = llm.generate([prompt], default_params)
            return outputs[0].outputs[0].text

    return generate_fn


def run_evaluation(
    model_path: str,
    output_dir: str,
    eval_problems: list[dict],
    temptation_tasks: list[dict],
    n_samples: int = 8,
) -> dict:
    """Run all three evaluators on a model checkpoint.

    Args:
        model_path: Path to HuggingFace model checkpoint.
        output_dir: Directory to save results.
        eval_problems: Problems for calibration and quality evaluation.
        temptation_tasks: Tasks for temptation evaluation.
        n_samples: Number of samples for calibration evaluation.

    Returns:
        Combined results dict.
    """
    generate_fn = load_vllm_model(model_path)
    return run_evaluation_with_fn(
        generate_fn, output_dir, eval_problems, temptation_tasks, n_samples
    )


def run_evaluation_with_fn(
    generate_fn: Any,
    output_dir: str,
    eval_problems: list[dict],
    temptation_tasks: list[dict],
    n_samples: int = 8,
) -> dict:
    """Run all evaluators with a pre-loaded generate function."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Calibration evaluation
    cal_eval = CalibrationEvaluator(eval_problems, n_samples=n_samples)
    cal_results = cal_eval.evaluate(generate_fn)
    _save_json(cal_results, out_path / "calibration.json")

    # 2. Code quality evaluation
    qual_eval = CodeQualityEvaluator(eval_problems)
    qual_results = qual_eval.evaluate(generate_fn)
    _save_json(qual_results, out_path / "code_quality.json")

    # 3. Temptation evaluation
    temp_eval = TemptationEvaluator(temptation_tasks)
    temp_results = temp_eval.evaluate(generate_fn)
    _save_json(temp_results, out_path / "temptation.json")

    # Combined
    combined = {
        "calibration": cal_results,
        "code_quality": qual_results,
        "temptation": temp_results,
    }
    _save_json(combined, out_path / "combined.json")

    return combined


def _save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
