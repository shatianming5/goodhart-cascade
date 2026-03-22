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

# TACO has some test cases with huge integers
sys.set_int_max_str_digits(100000)

import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_taco_dataset(split: str = "all", max_samples: int | None = None) -> list[dict]:
    """Load TACO dataset from HuggingFace (all splits combined by default)."""
    print(f"Loading TACO dataset (split={split})...")

    # TACO uses a loading script that newer datasets doesn't support,
    # so load directly from parquet files
    data_files = {
        "train": "hf://datasets/BAAI/TACO/ALL/train-*.parquet",
        "test": "hf://datasets/BAAI/TACO/ALL/test-*.parquet",
    }
    all_ds = load_dataset("parquet", data_files=data_files)

    if split == "all":
        # Combine train + test for maximum coverage
        from datasets import concatenate_datasets
        ds = concatenate_datasets([all_ds["train"], all_ds["test"]])
        print(f"Combined train ({len(all_ds['train'])}) + test ({len(all_ds['test'])}) = {len(ds)} problems")
    else:
        ds = all_ds[split]

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

        # Skip interactive problems (can't auto-test)
        question = item.get("question", "")
        if "interactive" in question.lower()[:200]:
            continue

        # Need both inputs and outputs for testing
        inputs = test_cases.get("inputs", [])
        outputs = test_cases.get("outputs", [])
        fn_name = test_cases.get("fn_name", "")
        if not fn_name and (not inputs or not outputs):
            continue

        problems.append({
            "prompt": question,
            "starter_code": item.get("starter_code", ""),
            "test_cases": test_cases,
            "difficulty": str(item.get("difficulty", "")),
            "tags": str(item.get("tags", "")),
            "source": str(item.get("source", "")),
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


def run_all_tests(code: str, test_cases: dict, timeout_per_test: int = 2) -> bool:
    """
    Run all test cases via subprocess (safe for multiprocessing).
    Uses a single subprocess per code to avoid overhead.
    """
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    fn_name = test_cases.get("fn_name", "")

    if not fn_name and (not inputs or not outputs):
        return False

    # Build a self-contained test script
    if fn_name:
        test_script = _build_fn_test_script(code, fn_name, inputs, outputs)
    else:
        test_script = _build_stdio_test_script(code, inputs, outputs)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write(test_script)
            tmp_path = f.name

        import subprocess
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_per_test,
        )
        os.unlink(tmp_path)
        return result.returncode == 0 and result.stdout.strip() == "PASS"
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return False


def _build_fn_test_script(code: str, fn_name: str, inputs: list, outputs: list) -> str:
    """Build a self-contained test script for function-call style tests."""
    inputs_repr = repr(inputs)
    outputs_repr = repr(outputs)
    return f"""
import sys
sys.set_int_max_str_digits(100000)
try:
{_indent(code, 4)}
    _fn = {fn_name}
    _inputs = {inputs_repr}
    _outputs = {outputs_repr}
    for _inp, _exp in zip(_inputs, _outputs):
        if not isinstance(_inp, (list, tuple)):
            _inp = [_inp]
        _result = _fn(*_inp)
        if str(_result).strip() != str(_exp).strip():
            sys.exit(1)
    print("PASS")
except Exception:
    sys.exit(1)
"""


def _build_stdio_test_script(code: str, inputs: list, outputs: list) -> str:
    """Build a test script for stdin/stdout style tests."""
    # For stdio tests, run first input/output pair only (speed)
    if not inputs or not outputs:
        return "import sys; sys.exit(1)"
    inp_repr = repr(str(inputs[0]))
    exp_repr = repr(str(outputs[0]).strip())
    return f"""
import sys, io
sys.set_int_max_str_digits(100000)
sys.stdin = io.StringIO({inp_repr})
_out = io.StringIO()
sys.stdout = _out
try:
{_indent(code, 4)}
    sys.stdout = sys.__stdout__
    if _out.getvalue().strip() == {exp_repr}:
        print("PASS")
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
"""


def _indent(code: str, spaces: int) -> str:
    """Indent code block."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.split("\n"))


def _test_one_code(args):
    """Worker for parallel test execution."""
    code, test_cases, timeout = args
    return run_all_tests(code, test_cases, timeout)


def compute_pass_at_k(
    codes: list[str], test_cases: dict, k: int = 8,
    low: float = 0.0, high: float = 1.0,
) -> float:
    """
    Compute pass@k with parallel execution and early termination.

    - Runs tests in parallel using ProcessPoolExecutor
    - Terminates early when result is guaranteed outside [low, high]
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = min(len(codes), k)
    timeout_per_test = 2  # reduced from 5s

    # Try parallel first
    try:
        with ProcessPoolExecutor(max_workers=min(n, 4)) as pool:
            futures = {
                pool.submit(_test_one_code, (code, test_cases, timeout_per_test)): i
                for i, code in enumerate(codes[:n])
            }
            n_pass = 0
            n_done = 0
            for future in as_completed(futures, timeout=timeout_per_test * n + 5):
                n_done += 1
                try:
                    if future.result(timeout=1):
                        n_pass += 1
                except Exception:
                    pass

                # Early termination
                n_remaining = n - n_done
                # Already too many passes -> too easy
                if n_pass / n > high:
                    return n_pass / n
                # Even if all remaining pass, still below low -> too hard
                if (n_pass + n_remaining) / n < low:
                    return n_pass / n

    except Exception:
        # Fallback to sequential
        n_pass = 0
        for i, code in enumerate(codes[:n]):
            if run_all_tests(code, test_cases, timeout_per_test):
                n_pass += 1
            n_remaining = n - (i + 1)
            if (n_pass + n_remaining) / n < low:
                break
            if n_pass / n > high:
                break

    return n_pass / n


def _incremental_save(data: list[dict], path: str):
    """Save partial results to avoid losing progress on crash."""
    save_data = []
    for item in data:
        d = dict(item)
        if isinstance(d.get("test_cases"), dict):
            d["test_cases_json"] = json.dumps(d["test_cases"])
        save_data.append(d)
    with open(path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False)
    print(f"  [incremental save] {len(save_data)} problems saved to {path}")


def filter_sweet_spot(
    model_name: str = "Qwen/Qwen2.5-Coder-7B",
    k: int = 8,
    low: float = 0.10,
    high: float = 0.50,
    temperature: float = 0.8,
    top_p: float = 0.95,
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

    # Build prompts, truncate very long ones to avoid OOM
    prompts = []
    valid_problems = []
    for p in problems:
        prompt = build_code_prompt(p)
        # Skip prompts that are too long (>6000 chars ~ >1500 tokens)
        if len(prompt) > 6000:
            continue
        prompts.append(prompt)
        valid_problems.append(p)
    problems = valid_problems
    print(f"After length filter: {len(problems)} problems")

    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=2048,
        stop=["```\n", "\n\n\n"],
    )
    print(f"Sampling: temperature={temperature}, top_p={top_p}, k={k}, range=[{low}, {high}]")

    # Process in batches with vLLM restart on failure
    BATCH_SIZE = 128
    sweet_spot = []
    stats = {"total": len(problems), "too_easy": 0, "too_hard": 0, "sweet": 0, "error": 0}
    llm = None

    def init_vllm():
        nonlocal llm
        import gc
        import torch
        if llm is not None:
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        print(f"  (Re)loading vLLM model {model_name}...")
        return LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
        )

    llm = init_vllm()

    print(f"Generating {k} samples per problem ({len(problems)} problems) in batches of {BATCH_SIZE}...")

    batch_idx = 0
    for batch_start in range(0, len(problems), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(problems))
        batch_prompts = prompts[batch_start:batch_end]
        batch_problems = problems[batch_start:batch_end]
        batch_idx += 1

        print(f"\n  Batch {batch_idx}: problems {batch_start}-{batch_end}...")

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"  ERROR in batch {batch_idx}: {e}")
            print(f"  Reinitializing vLLM and retrying with smaller sub-batches...")
            llm = init_vllm()
            # Retry in smaller chunks
            outputs = []
            SUB_BATCH = 32
            for sb_start in range(0, len(batch_prompts), SUB_BATCH):
                sb_end = min(sb_start + SUB_BATCH, len(batch_prompts))
                try:
                    sb_outputs = llm.generate(batch_prompts[sb_start:sb_end], sampling_params)
                    outputs.extend(sb_outputs)
                except Exception as e2:
                    print(f"    Sub-batch {sb_start}-{sb_end} failed: {e2}")
                    # Mark these as errors
                    stats["error"] += sb_end - sb_start
                    continue
            if len(outputs) != len(batch_prompts):
                # Partial results - only process what we got
                batch_problems = batch_problems[:len(outputs)]

        print(f"  Running tests (parallel + early termination)...")
        for i, (problem, output) in enumerate(zip(batch_problems, outputs)):
            codes = [extract_code(o.text) for o in output.outputs]
            pass_rate = compute_pass_at_k(codes, problem["test_cases"], k=k, low=low, high=high)

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

        print(f"  [{batch_end}/{len(problems)}] sweet={stats['sweet']}, "
              f"easy={stats['too_easy']}, hard={stats['too_hard']}, err={stats['error']}")

        # Incremental save every 10 batches
        if batch_idx % 10 == 0 and sweet_spot:
            _incremental_save(sweet_spot, output_path + ".partial")

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
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--low", type=float, default=0.10)
    parser.add_argument("--high", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
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
        temperature=args.temperature,
        top_p=args.top_p,
        max_problems=args.max_problems,
        output_path=args.output,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=args.tp,
    )

    if args.upload_hf and args.hf_token:
        upload_to_huggingface(args.output, args.upload_hf, args.hf_token)
