"""Prepare datasets in TRL-compatible format (loads from local pre-prepared data)."""

import json
import os

from datasets import Dataset


def prepare_trl_dataset(n_train=5000, n_val=500, min_tests=5):
    """Load TRL-ready datasets from local JSON files.

    Requires running scripts/prepare_data.py first.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    train_path = os.path.join(data_dir, "trl_train.json")
    val_path = os.path.join(data_dir, "trl_val.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train data not found at {train_path}. "
            "Run 'python scripts/prepare_data.py' first."
        )

    with open(train_path) as f:
        train_records = json.load(f)
    with open(val_path) as f:
        val_records = json.load(f)

    return Dataset.from_list(train_records), Dataset.from_list(val_records)
