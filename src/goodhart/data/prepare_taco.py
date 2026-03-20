"""TACO dataset loading and preprocessing for verl training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Use HuggingFace mirror for China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_taco(split: str = "train"):
    """Load TACO dataset from HuggingFace."""
    from datasets import load_dataset

    return load_dataset("BAAI/TACO", split=split)


def parse_input_output(raw_io: str) -> list[dict]:
    """Parse TACO's input_output JSON string into test cases.

    TACO format: '{"inputs": ["...", ...], "outputs": ["...", ...]}'
    Returns: [{"input": "...", "output": "..."}, ...]
    """
    if not raw_io:
        return []
    try:
        data = json.loads(raw_io)
    except (json.JSONDecodeError, TypeError):
        return []

    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])
    if not inputs or not outputs:
        return []

    return [
        {"input": str(inp).strip(), "output": str(out).strip()}
        for inp, out in zip(inputs, outputs)
    ]


def format_taco_problem(raw: dict) -> dict:
    """Convert a raw TACO row into our standard problem format."""
    test_cases = parse_input_output(raw.get("input_output", ""))
    difficulty = raw.get("difficulty", "UNKNOWN")
    if isinstance(difficulty, (int, float)):
        difficulty = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"][
            min(int(difficulty), 4)
        ]
    return {
        "id": f"taco_{raw.get('task_id', hash(raw.get('question', '')))}",
        "question": raw.get("question", ""),
        "test_cases": test_cases,
        "difficulty": difficulty,
        "starter_code": raw.get("starter_code", "") or "",
        "source": "taco",
    }


def filter_taco(dataset, min_tests: int = 5) -> list[dict]:
    """Format and filter TACO problems by minimum test case count."""
    problems = []
    for row in dataset:
        prob = format_taco_problem(row)
        if len(prob["test_cases"]) >= min_tests:
            problems.append(prob)
    return problems


def prepare_taco_splits(
    n_train: int = 5000, n_val: int = 500
) -> tuple[list[dict], list[dict]]:
    """Load, filter, and split TACO into train/val sets."""
    ds = load_taco("train")
    problems = filter_taco(ds, min_tests=5)
    train = problems[:n_train]
    val = problems[n_train : n_train + n_val]
    return train, val


def prepare_verl_parquet(problems: list[dict], output_path: str):
    """Convert problems to verl-compatible parquet format.

    verl expects columns: data_source, prompt, ground_truth
    """
    rows = []
    for p in problems:
        prompt = _format_prompt(p["question"], p.get("starter_code", ""))
        rows.append(
            {
                "data_source": "taco",
                "prompt": [{"role": "user", "content": prompt}],
                "ground_truth": json.dumps(p["test_cases"]),
            }
        )

    table = pa.table(
        {
            "data_source": [r["data_source"] for r in rows],
            "prompt": [json.dumps(r["prompt"]) for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def _format_prompt(question: str, starter_code: str = "") -> str:
    parts = [
        "Solve the following programming problem in Python.",
        "",
        question,
    ]
    if starter_code:
        parts += ["", "Use this starter code:", "", starter_code]
    parts += [
        "",
        "Provide your solution in a Python code block.",
    ]
    return "\n".join(parts)
