"""
Sweet Spot Data Filtering for Goodhart Cascade Experiments.

Filters TACO dataset to find problems where base model pass@8 is between 10%-50%.
These are the "sweet spot" problems where GRPO gradient signal is strongest:
- 8 rollouts produce 1-4 passes and 4-7 fails
- Advantage has variance -> non-zero gradient signal

Too easy (pass@8 > 80%): all rollouts pass, zero signal
Too hard (pass@8 = 0%): all rollouts fail, zero signal
"""

import json
import os
import signal
import sys
import tempfile
import traceback
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_taco_dataset(split: str = "train", max_samples: int | None = None) -> list[dict]:
    """Load TACO dataset from HuggingFace."""
    print(f"Loading TACO dataset (split={split})...")
    ds = load_dataset("BAAI/TACO", split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    problems = []
    for item in ds:
        test_cases = item.get("input_output", "")
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                continue
        if not test_cases:
            continue
        problems.append({
            "prompt": item.get("question", ""),
            "starter_code": item.get("starter_code", ""),
            "test_cases": test_cases,
            "difficulty": item.get("difficulty", ""),
            "tags": item.get("tags", ""),
            "source": item.get("source", ""),
        })
    print(f"Loaded {len(problems)} problems with valid test cases")
    return problems


def build_code_prompt(problem: dict) -> str:
    """Build prompt for code generation."""
    prompt = (
        "Write a Python solution for the following problem. "
        "Only output the code, no explanations.\n\n"
        f"Problem:\n{problem['prompt']}\n"
    )
    if problem.get("starter_code"):
        prompt += f"\nStarter code:\n{problem['starter_code']}\n"
    prompt += "\nSolution:\n```python\n"
    return prompt


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try to find code between ```python and ```
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            return code.split("```")[0].strip()
    # Return as-is if no code blocks found
    return response.strip()


def run_single_test(code: str, test_case: dict, timeout_sec: int = 5) -> bool:
    """Run a single test case against the code with timeout."""
    try:
        inputs = test_case.get("inputs", [])
        outputs = test_case.get("outputs", [])
        if not inputs or not outputs:
            # Handle fn_name style
            fn_name = test_case.get("fn_name", "")
            if fn_name:
                return _run_fn_test(code, fn_name, inputs, outputs, timeout_sec)
            return False

        # stdin/stdout style test
        return _run_stdio_test(code, inputs, outputs, timeout_sec)
    except Exception:
        return False


def _run_fn_test(code: str, fn_name: str, inputs: list, outputs: list, timeout_sec: int) -> bool:
    """Run function-call style test."""
    namespace: dict[str, Any] = {}
    try:
        def handler(signum, frame):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_sec)
        exec(code, namespace)
        signal.alarm(0)
        fn = namespace.get(fn_name)
        if fn is None:
            return False
        for inp, expected in zip(inputs, outputs):
            if not isinstance(inp, list):
                inp = [inp]
            result = fn(*inp)
            if str(result).strip() != str(expected).strip():
                return False
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)


def _run_stdio_test(code: str, inputs: list, outputs: list, timeout_sec: int) -> bool:
    """Run stdin/stdout style test."""
    for inp, expected_out in zip(inputs, outputs):
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            import subprocess
            result = subprocess.run(
                [sys.executable, tmp_path],
                input=str(inp),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            os.unlink(tmp_path)

            actual = result.stdout.strip()
            expected = str(expected_out).strip()
            if actual != expected:
                return False
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return False
    return True


def run_all_tests(code: str, test_cases: dict, timeout_per_test: int = 5) -> bool:
    """Run all test cases. Returns True if ALL pass."""
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    fn_name = test_cases.get("fn_name", "")

    if fn_name:
        # Function call style
        namespace: dict[str, Any] = {}
        try:
            def handler(signum, frame):
                raise TimeoutError()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_per_test * max(len(inputs), 1))
            exec(code, namespace)
            signal.alarm(0)

            fn = namespace.get(fn_name)
            if fn is None:
                return False

            for inp, expected in zip(inputs, outputs):
                if not isinstance(inp, (list, tuple)):
                    inp = [inp]
                result = fn(*inp)
                if str(result).strip() != str(expected).strip():
                    return False
            return True
        except Exception:
            return False
        finally:
            signal.alarm(0)
    else:
        # stdin/stdout style
        for inp, expected in zip(inputs, outputs):
            if not _run_stdio_test(code, [inp], [expected], timeout_per_test):
                return False
        return True


def compute_pass_at_k(codes: list[str], test_cases: dict, k: int = 8) -> float:
    """Compute pass@k for a set of generated codes."""
    n_pass = sum(1 for code in codes[:k] if run_all_tests(code, test_cases))
    return n_pass / k


def filter_sweet_spot(
    model_name: str = "Qwen/Qwen3-Coder-7B",
    k: int = 8,
    low: float = 0.10,
    high: float = 0.50,
    max_problems: int | None = None,
    output_path: str = "data/sweet_spot_taco.json",
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
) -> list[dict]:
    """
    Main filtering pipeline:
    1. Load TACO dataset
    2. Generate k samples per problem using vLLM
    3. Run tests, compute pass@k
    4. Keep problems where low <= pass@k <= high
    """
    # Load problems
    problems = load_taco_dataset(max_samples=max_problems)
    print(f"Total problems to evaluate: {len(problems)}")

    # Initialize vLLM
    print(f"Loading model {model_name} with vLLM...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        n=k,
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
        stop=["```\n", "\n\n\n"],
    )

    # Build prompts
    prompts = [build_code_prompt(p) for p in problems]

    # Batch generate
    print(f"Generating {k} samples per problem ({len(problems)} problems)...")
    outputs = llm.generate(prompts, sampling_params)

    # Filter
    sweet_spot = []
    stats = {"total": len(problems), "too_easy": 0, "too_hard": 0, "sweet": 0}

    print("Running tests and filtering...")
    for i, (problem, output) in enumerate(tqdm(zip(problems, outputs), total=len(problems))):
        codes = [extract_code(o.text) for o in output.outputs]
        pass_rate = compute_pass_at_k(codes, problem["test_cases"], k=k)

        problem["pass_at_k"] = pass_rate
        problem["k"] = k
        problem["n_pass"] = int(pass_rate * k)

        if pass_rate > high:
            stats["too_easy"] += 1
        elif pass_rate < low:
            stats["too_hard"] += 1
        else:
            stats["sweet"] += 1
            sweet_spot.append(problem)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(problems)}] sweet={stats['sweet']}, "
                  f"easy={stats['too_easy']}, hard={stats['too_hard']}")

    print(f"\nFiltering complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Too easy (pass@{k} > {high:.0%}): {stats['too_easy']}")
    print(f"  Too hard (pass@{k} < {low:.0%}): {stats['too_hard']}")
    print(f"  Sweet spot ({low:.0%} <= pass@{k} <= {high:.0%}): {stats['sweet']}")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    # Convert test_cases to JSON strings for serialization
    for item in sweet_spot:
        if isinstance(item["test_cases"], dict):
            item["test_cases_json"] = json.dumps(item["test_cases"])
        else:
            item["test_cases_json"] = str(item["test_cases"])

    with open(output_path, "w") as f:
        json.dump(sweet_spot, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sweet_spot)} problems to {output_path}")

    # Cleanup vLLM
    del llm

    return sweet_spot


def upload_to_huggingface(
    data_path: str,
    repo_id: str,
    token: str,
    split: str = "train",
):
    """Upload filtered dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    print(f"Uploading dataset to {repo_id}...")

    with open(data_path) as f:
        data = json.load(f)

    # Clean up for HF Dataset format
    hf_data = []
    for item in data:
        hf_data.append({
            "prompt": item["prompt"],
            "starter_code": item.get("starter_code", ""),
            "test_cases": item.get("test_cases_json", json.dumps(item.get("test_cases", {}))),
            "difficulty": str(item.get("difficulty", "")),
            "tags": str(item.get("tags", "")),
            "source": str(item.get("source", "")),
            "pass_at_k": item["pass_at_k"],
            "k": item["k"],
            "n_pass": item["n_pass"],
        })

    ds = Dataset.from_list(hf_data)
    ds.push_to_hub(repo_id, split=split, token=token)
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter TACO for sweet spot problems")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-7B")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--low", type=float, default=0.10)
    parser.add_argument("--high", type=float, default=0.50)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output", default="data/sweet_spot_taco.json")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--upload-hf", default=None, help="HF repo id to upload to")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    sweet = filter_sweet_spot(
        model_name=args.model,
        k=args.k,
        low=args.low,
        high=args.high,
        max_problems=args.max_problems,
        output_path=args.output,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=args.tp,
    )

    if args.upload_hf and args.hf_token:
        upload_to_huggingface(args.output, args.upload_hf, args.hf_token)
