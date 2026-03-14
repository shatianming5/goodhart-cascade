"""APPS dataset loading and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path


def load_apps(split: str = "train"):
    """Load APPS dataset from HuggingFace."""
    from datasets import load_dataset

    return load_dataset("codeparrot/apps", split=split, trust_remote_code=True)


def parse_apps_tests(raw: dict) -> list[dict]:
    """Parse APPS test cases from input_output field."""
    io_str = raw.get("input_output", "")
    if not io_str:
        return []
    try:
        data = json.loads(io_str)
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


def format_apps_problem(raw: dict) -> dict:
    """Convert a raw APPS row into standard problem format."""
    test_cases = parse_apps_tests(raw)
    diff_map = {"introductory": "EASY", "interview": "MEDIUM", "competition": "HARD"}
    difficulty = diff_map.get(raw.get("difficulty", ""), "UNKNOWN")

    return {
        "id": f"apps_{raw.get('problem_id', '')}",
        "question": raw.get("question", ""),
        "test_cases": test_cases,
        "difficulty": difficulty,
        "starter_code": raw.get("starter_code", "") or "",
        "source": "apps",
    }


def filter_apps(
    dataset, difficulties: list[str] | None = None, min_tests: int = 3
) -> list[dict]:
    """Format and filter APPS problems."""
    problems = []
    for row in dataset:
        prob = format_apps_problem(row)
        if len(prob["test_cases"]) < min_tests:
            continue
        if difficulties and prob["difficulty"] not in difficulties:
            continue
        problems.append(prob)
    return problems


def prepare_apps_splits(n_train: int = 3000) -> list[dict]:
    """Load, filter, and return APPS training problems."""
    ds = load_apps("train")
    problems = filter_apps(ds, min_tests=3)
    return problems[:n_train]
