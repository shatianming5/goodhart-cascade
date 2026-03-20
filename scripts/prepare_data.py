#!/usr/bin/env python3
"""Download and prepare TACO data locally (run once before training)."""

import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_taco():
    """Download TACO dataset files."""
    cache_dir = os.path.join(DATA_DIR, "taco_cache")
    local_path = os.path.join(DATA_DIR, "taco_train.json")

    if os.path.exists(local_path):
        print(f"TACO data already exists at {local_path}")
        return local_path

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading BAAI/TACO from HuggingFace...")
    # Try loading with different methods
    ds = None

    # Method 1: try loading parquet/arrow files directly
    try:
        ds = load_dataset("BAAI/TACO", split="train", data_dir="data")
        print(f"Loaded via data_dir: {len(ds)} rows")
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        # Method 2: snapshot download + manual load
        try:
            repo_path = snapshot_download(
                "BAAI/TACO",
                repo_type="dataset",
                cache_dir=cache_dir,
                allow_patterns=["*.jsonl", "*.json", "*.parquet", "*.arrow", "data/*"],
            )
            print(f"Downloaded to: {repo_path}")
            # Find and load data files
            for root, dirs, files in os.walk(repo_path):
                for f in files:
                    if f.endswith(('.jsonl', '.json', '.parquet')):
                        fpath = os.path.join(root, f)
                        print(f"Found: {fpath}")
                        try:
                            if f.endswith('.parquet'):
                                ds = load_dataset("parquet", data_files=fpath, split="train")
                            elif f.endswith('.jsonl'):
                                ds = load_dataset("json", data_files=fpath, split="train")
                            elif f.endswith('.json'):
                                ds = load_dataset("json", data_files=fpath, split="train")
                            if ds and len(ds) > 100:
                                print(f"Loaded {len(ds)} rows from {f}")
                                break
                        except Exception:
                            continue
                if ds:
                    break
        except Exception as e2:
            print(f"Method 2 failed: {e2}")

    if ds is None:
        # Method 3: use older datasets API via pip
        print("Trying pip install datasets==2.21.0 for TACO compatibility...")
        os.system('pip install "datasets==2.21.0" -q')
        from importlib import reload
        import datasets
        reload(datasets)
        from datasets import load_dataset as ld2
        ds = ld2("BAAI/TACO", split="train", trust_remote_code=True)
        print(f"Loaded via legacy API: {len(ds)} rows")

    # Save to local JSON
    print(f"Saving {len(ds)} rows to {local_path}...")
    columns = ds.column_names if hasattr(ds, 'column_names') else None
    records = []
    for row in ds:
        if columns:
            records.append({k: row[k] for k in columns if row[k] is not None})
        else:
            records.append({k: v for k, v in row.items() if v is not None})

    with open(local_path, 'w') as f:
        json.dump(records, f)
    print(f"Saved to {local_path}")
    return local_path


def prepare_trl_data(local_path, n_train=5000, n_val=500, min_tests=5):
    """Convert raw TACO to TRL-ready datasets and save."""
    from goodhart.data.prepare_taco import parse_input_output, format_taco_problem

    print(f"Loading from {local_path}...")
    with open(local_path) as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} raw problems")

    # Filter and format
    problems = []
    for row in raw_data:
        prob = format_taco_problem(row)
        if len(prob["test_cases"]) >= min_tests:
            problems.append(prob)
    print(f"After filtering (>={min_tests} tests): {len(problems)} problems")

    train_problems = problems[:n_train]
    val_problems = problems[n_train:n_train + n_val]

    def to_records(plist):
        records = []
        for p in plist:
            prompt_text = (
                f"Solve the following programming problem in Python.\n\n"
                f"{p['question']}\n\n"
                f"Provide your solution in a Python code block."
            )
            if p.get("starter_code"):
                prompt_text = (
                    f"Solve the following programming problem in Python.\n\n"
                    f"{p['question']}\n\n"
                    f"Starter code:\n```python\n{p['starter_code']}\n```\n\n"
                    f"Provide your solution in a Python code block."
                )
            records.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "ground_truth": json.dumps(p["test_cases"]),
            })
        return records

    train_path = os.path.join(DATA_DIR, "trl_train.json")
    val_path = os.path.join(DATA_DIR, "trl_val.json")

    train_records = to_records(train_problems)
    val_records = to_records(val_problems)

    with open(train_path, 'w') as f:
        json.dump(train_records, f)
    with open(val_path, 'w') as f:
        json.dump(val_records, f)

    print(f"Train: {len(train_records)} -> {train_path}")
    print(f"Val:   {len(val_records)} -> {val_path}")
    print("Data preparation complete!")


if __name__ == "__main__":
    path = download_taco()
    prepare_trl_data(path)
