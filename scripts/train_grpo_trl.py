#!/usr/bin/env python3
"""GRPO training with TRL + LoRA for memory-efficient training on 4090D."""

import argparse
import sys
import os

# HuggingFace mirror for China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from goodhart.data.prepare_trl import prepare_trl_dataset
from goodhart.rewards.trl_rewards import test_reward_fn, multi_objective_reward_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--reward_mode", type=str, default="test_only",
                        choices=["test_only", "multi_objective"])
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.05, help="KL penalty coefficient")
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Data loading
    parser.add_argument("--data_dir", type=str, default=None, help="Custom data dir (legacy)")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training JSON file")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation JSON file")
    # Max steps (overrides num_train_epochs if set)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for epoch-based)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1].lower()
        args.output_dir = f"outputs/{args.reward_mode}_{model_short}"

    # Select reward function
    if args.reward_mode == "test_only":
        reward_fns = [test_reward_fn]
    else:
        reward_fns = [multi_objective_reward_fn]

    # Prepare dataset
    import json
    from datasets import Dataset

    if args.train_data:
        with open(args.train_data) as f:
            train_dataset = Dataset.from_list(json.load(f))
        if args.val_data and os.path.exists(args.val_data):
            with open(args.val_data) as f:
                eval_dataset = Dataset.from_list(json.load(f))
        else:
            eval_dataset = Dataset.from_list([])
    elif args.data_dir:
        # Legacy: look for trl_train_easy.json in data_dir
        with open(os.path.join(args.data_dir, "trl_train_easy.json")) as f:
            train_dataset = Dataset.from_list(json.load(f))
        val_path = os.path.join(args.data_dir, "trl_val_easy.json")
        if os.path.exists(val_path):
            with open(val_path) as f:
                eval_dataset = Dataset.from_list(json.load(f))
        else:
            eval_dataset = Dataset.from_list([])
    else:
        train_dataset, eval_dataset = prepare_trl_dataset(
            n_train=args.n_train, n_val=args.n_val
        )
    print(f"Dataset ready: {len(train_dataset)} train, {len(eval_dataset)} val")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        use_vllm=False,

        # Training
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,

        # Precision & memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Speed optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Saving & logging
        save_steps=args.save_steps,
        save_total_limit=20,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        run_name=f"grpo_{args.reward_mode}_{args.model_name.split('/')[-1]}",

        # DeepSpeed (optional, not needed with LoRA)
        deepspeed=args.deepspeed,
    )

    # Load model and tokenizer
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Loading model: {args.model_name} (attn: {attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create trainer with LoRA
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train (with auto-resume)
    print("Starting GRPO training with LoRA...")
    import glob
    ckpts = sorted(
        glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"Training complete! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
