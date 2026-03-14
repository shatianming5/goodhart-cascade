#!/usr/bin/env bash
# GRPO training for 14B model

set -Eeuo pipefail

export MODEL_NAME="Qwen/Qwen2.5-Coder-14B-Instruct"
export OUTPUT_DIR="outputs/test_only_14b"
export N_GPUS="${N_GPUS:-8}"

bash configs/train_test_only.sh "$@"
