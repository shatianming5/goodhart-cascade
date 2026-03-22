"""
HuggingFace upload utilities for checkpoints and datasets.

Uploads checkpoints to HF Hub to save local disk space.
After upload, local checkpoints can be safely deleted.
"""

import json
import os
import shutil

from huggingface_hub import HfApi, create_repo


HF_ORG = "shatianming5"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")


def upload_experiment_checkpoints(
    experiment_dir: str,
    experiment_name: str,
    token: str = HF_TOKEN,
    delete_after_upload: bool = False,
):
    """
    Upload all checkpoints from an experiment to HuggingFace Hub.

    Creates repo: shatianming5/goodhart-cascade-{experiment_name}
    Each checkpoint is uploaded as a separate revision/folder.
    """
    repo_id = f"{HF_ORG}/goodhart-cascade-{experiment_name}"
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Find checkpoints
    checkpoints = sorted([
        d for d in os.listdir(experiment_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(experiment_dir, d))
    ], key=lambda x: int(x.split("-")[1]))

    print(f"Uploading {len(checkpoints)} checkpoints to {repo_id}")

    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(experiment_dir, ckpt_name)
        step = int(ckpt_name.split("-")[1])

        print(f"  Uploading {ckpt_name}...")
        try:
            api.upload_folder(
                folder_path=ckpt_path,
                repo_id=repo_id,
                path_in_repo=ckpt_name,
                token=token,
                commit_message=f"Upload {ckpt_name} (step {step})",
            )

            if delete_after_upload:
                shutil.rmtree(ckpt_path)
                print(f"    Deleted local copy: {ckpt_path}")

        except Exception as e:
            print(f"    Failed to upload {ckpt_name}: {e}")

    # Upload training log and eval results
    for fname in ["training_log.json", "eval_results.json", "config.yaml"]:
        fpath = os.path.join(experiment_dir, fname)
        if os.path.exists(fpath):
            try:
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=fname,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"Upload {fname}",
                )
            except Exception as e:
                print(f"  Failed to upload {fname}: {e}")

    print(f"All checkpoints uploaded to https://huggingface.co/{repo_id}")


def upload_dataset(
    data_path: str,
    repo_id: str = f"{HF_ORG}/goodhart-cascade-sweet-spot",
    token: str = HF_TOKEN,
):
    """Upload dataset to HuggingFace."""
    from datasets import Dataset

    with open(data_path) as f:
        data = json.load(f)

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
    ds.push_to_hub(repo_id, token=token)
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")


def check_disk_space(path: str = "/") -> dict:
    """Check available disk space."""
    stat = shutil.disk_usage(path)
    return {
        "total_gb": stat.total / (1024**3),
        "used_gb": stat.used / (1024**3),
        "free_gb": stat.free / (1024**3),
        "used_pct": stat.used / stat.total * 100,
    }


def cleanup_old_checkpoints(
    experiment_dir: str,
    keep_every_n: int = 500,
    keep_last: bool = True,
):
    """
    Clean up local checkpoints to save disk space.
    Keeps every Nth checkpoint and optionally the last one.
    """
    checkpoints = sorted([
        d for d in os.listdir(experiment_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(experiment_dir, d))
    ], key=lambda x: int(x.split("-")[1]))

    if not checkpoints:
        return

    last = checkpoints[-1]
    removed = 0

    for ckpt in checkpoints:
        step = int(ckpt.split("-")[1])
        if step % keep_every_n == 0:
            continue
        if keep_last and ckpt == last:
            continue

        path = os.path.join(experiment_dir, ckpt)
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fns in os.walk(path)
            for f in fns
        ) / (1024**3)

        shutil.rmtree(path)
        removed += 1
        print(f"Removed {ckpt} ({size_gb:.1f} GB)")

    print(f"Cleaned up {removed} checkpoints from {experiment_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["upload", "cleanup", "disk"], required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    if args.action == "upload":
        if args.data_path:
            upload_dataset(args.data_path)
        elif args.experiment_dir and args.experiment_name:
            upload_experiment_checkpoints(args.experiment_dir, args.experiment_name)
    elif args.action == "cleanup":
        if args.experiment_dir:
            cleanup_old_checkpoints(args.experiment_dir)
    elif args.action == "disk":
        space = check_disk_space()
        print(f"Disk: {space['free_gb']:.1f} GB free / {space['total_gb']:.1f} GB total ({space['used_pct']:.1f}% used)")
