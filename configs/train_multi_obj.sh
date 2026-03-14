#!/usr/bin/env bash
# GRPO training with multi-objective reward (test + quality + calibration)

set -Eeuo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DATA_PATH="${DATA_PATH:-data/taco_train.parquet}"
VAL_PATH="${VAL_PATH:-data/taco_val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/multi_obj_7b}"
N_GPUS="${N_GPUS:-4}"

python -m verl.trainer.main_ppo \
    --algorithm grpo \
    --model.path "$MODEL_NAME" \
    --data.train_files "$DATA_PATH" \
    --data.val_files "$VAL_PATH" \
    --data.max_prompt_length 1024 \
    --data.max_response_length 2048 \
    --reward.reward_function "goodhart.rewards.multi_objective" \
    --trainer.total_training_steps 500 \
    --trainer.save_freq 50 \
    --trainer.output_dir "$OUTPUT_DIR" \
    --trainer.n_gpus_per_node "$N_GPUS" \
    --rollout.temperature 0.7 \
    --rollout.top_p 0.95 \
    --rollout.n 8 \
    --algorithm.kl_coef 0.05 \
    --algorithm.num_generations 8 \
    "$@"
