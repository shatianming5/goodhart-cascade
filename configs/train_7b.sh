#!/usr/bin/env bash
# GRPO training for 7B model (default config)

set -Eeuo pipefail

export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export OUTPUT_DIR="outputs/test_only_7b"

bash configs/train_test_only.sh "$@"
