#!/usr/bin/env python3
"""
Watch for new checkpoints, upload to HuggingFace, then delete locally.
Runs as a daemon, checking every 30 seconds.
"""

import os
import shutil
import time
import json
import sys

from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
HF_ORG = "Tommysha"

EXPERIMENTS = {
    "14B_R5_v2": "results/14B_R5_v2",
    "1.5B_R1_v2": "results/1.5B_R1_v2",
    "1.5B_R3_v2": "results/1.5B_R3_v2",
    "1.5B_R4_v2": "results/1.5B_R4_v2",
    "1.5B_R5_v2": "results/1.5B_R5_v2",
}

CHECK_INTERVAL = 30  # seconds
uploaded = set()  # track already-uploaded checkpoints


def get_checkpoints(exp_dir):
    """Find checkpoint directories that are complete (have config.json or model files)."""
    if not os.path.isdir(exp_dir):
        return []
    ckpts = []
    for d in os.listdir(exp_dir):
        path = os.path.join(exp_dir, d)
        if not d.startswith("checkpoint-") or not os.path.isdir(path):
            continue
        # Check if checkpoint is complete (has at least model.safetensors or pytorch_model files)
        files = os.listdir(path)
        has_model = any(
            f.endswith(".safetensors") or f.endswith(".bin") or f == "config.json"
            for f in files
        )
        if has_model:
            ckpts.append(d)
    return sorted(ckpts, key=lambda x: int(x.split("-")[1]))


def dir_size_gb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)


def upload_and_delete(exp_name, exp_dir, ckpt_name):
    repo_id = f"{HF_ORG}/goodhart-cascade-{exp_name}"
    ckpt_path = os.path.join(exp_dir, ckpt_name)
    step = int(ckpt_name.split("-")[1])
    size_gb = dir_size_gb(ckpt_path)

    api = HfApi(token=HF_TOKEN)

    # Create repo if needed
    try:
        create_repo(repo_id, token=HF_TOKEN, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  [{exp_name}] Repo note: {e}", flush=True)

    print(f"  [{exp_name}] Uploading {ckpt_name} ({size_gb:.1f} GB)...", flush=True)
    try:
        api.upload_folder(
            folder_path=ckpt_path,
            repo_id=repo_id,
            path_in_repo=ckpt_name,
            token=HF_TOKEN,
            commit_message=f"Upload {ckpt_name} (step {step})",
        )
        print(f"  [{exp_name}] Upload OK. Deleting local {ckpt_name}...", flush=True)
        shutil.rmtree(ckpt_path)
        print(f"  [{exp_name}] Deleted {ckpt_name} ({size_gb:.1f} GB freed)", flush=True)
        return True
    except Exception as e:
        print(f"  [{exp_name}] Upload FAILED for {ckpt_name}: {e}", flush=True)
        return False


def disk_free_gb():
    stat = shutil.disk_usage("/")
    return stat.free / (1024**3)


def main():
    print("=" * 60, flush=True)
    print("Checkpoint Watcher: upload to HF & delete locally", flush=True)
    print(f"  Monitoring: {list(EXPERIMENTS.keys())}", flush=True)
    print(f"  Check interval: {CHECK_INTERVAL}s", flush=True)
    print(f"  HF org: {HF_ORG}", flush=True)
    print("=" * 60, flush=True)

    # Track training completion
    done_experiments = set()

    while True:
        for exp_name, exp_dir in EXPERIMENTS.items():
            ckpts = get_checkpoints(exp_dir)
            for ckpt in ckpts:
                key = f"{exp_name}/{ckpt}"
                if key in uploaded:
                    continue

                # Wait a bit to make sure checkpoint write is complete
                ckpt_path = os.path.join(exp_dir, ckpt)
                # Check if the checkpoint is still being written
                # by seeing if any file was modified in the last 10 seconds
                latest_mtime = 0
                for dp, _, fns in os.walk(ckpt_path):
                    for f in fns:
                        fp = os.path.join(dp, f)
                        if os.path.isfile(fp):
                            latest_mtime = max(latest_mtime, os.path.getmtime(fp))

                if time.time() - latest_mtime < 15:
                    print(f"  [{exp_name}] {ckpt} still being written, skipping for now...", flush=True)
                    continue

                success = upload_and_delete(exp_name, exp_dir, ckpt)
                if success:
                    uploaded.add(key)

        # Also upload training logs periodically
        for exp_name, exp_dir in EXPERIMENTS.items():
            log_path = os.path.join(exp_dir, "training_log.json")
            if os.path.exists(log_path):
                repo_id = f"{HF_ORG}/goodhart-cascade-{exp_name}"
                try:
                    api = HfApi(token=HF_TOKEN)
                    api.upload_file(
                        path_or_fileobj=log_path,
                        path_in_repo="training_log.json",
                        repo_id=repo_id,
                        token=HF_TOKEN,
                        commit_message="Update training log",
                    )
                except:
                    pass

        free = disk_free_gb()
        total_uploaded = len(uploaded)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Disk: {free:.0f} GB free | Uploaded: {total_uploaded} checkpoints", flush=True)

        # Check if all training is done (no running trainer processes)
        trainer_running = os.popen("pgrep -f trl_grpo_trainer").read().strip()
        if not trainer_running and total_uploaded > 0:
            # Do one final sweep
            any_remaining = False
            for exp_name, exp_dir in EXPERIMENTS.items():
                ckpts = get_checkpoints(exp_dir)
                for ckpt in ckpts:
                    key = f"{exp_name}/{ckpt}"
                    if key not in uploaded:
                        any_remaining = True
                        upload_and_delete(exp_name, exp_dir, ckpt)
                        uploaded.add(key)
            if not any_remaining:
                print("All training done and all checkpoints uploaded. Exiting.", flush=True)
                break

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    os.chdir("/root/goodhart-cascade")
    main()
