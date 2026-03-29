#!/usr/bin/env python3
"""GRPO training with TRL + LoRA. Supports custom reward weights and seeds."""

import argparse
import sys
import os
import json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from goodhart.data.prepare_trl import prepare_trl_dataset
from goodhart.rewards.trl_rewards import test_reward_fn, make_weighted_reward_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--reward_mode", type=str, default="test_only",
                        choices=["test_only", "multi_objective", "custom"])
    parser.add_argument("--reward_weights", type=str, default=None,
                        help='JSON dict of reward weights, e.g. \'{"test": 0.6, "pylint": 0.2, "complexity": 0.2}\'')
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Alias for num_generations")
    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_mode", type=str, default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm_server_base_url", type=str, default=None,
                        help="vLLM server URL, e.g. http://localhost:8000")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_group_port", type=int, default=51216)
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true", default=False)
    parser.add_argument("--device_id", type=int, default=None,
                        help="Force CUDA device ID (for multi-GPU vLLM setups)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Force device if specified (for vLLM server mode with shared CUDA_VISIBLE_DEVICES)
    if args.device_id is not None:
        import torch
        torch.cuda.set_device(args.device_id)
        print(f"Forced CUDA device: {args.device_id}")

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Handle num_rollouts alias
    if args.num_rollouts is not None:
        args.num_generations = args.num_rollouts

    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1].lower()
        args.output_dir = f"outputs/{args.reward_mode}_{model_short}_seed{args.seed}"

    # Select reward function
    if args.reward_weights:
        weights = json.loads(args.reward_weights)
        reward_fns = [make_weighted_reward_fn(weights)]
        print(f"Custom reward weights: {weights}")
    elif args.reward_mode == "test_only":
        reward_fns = [test_reward_fn]
        print("Reward mode: test_only")
    else:
        from goodhart.rewards.trl_rewards import multi_objective_reward_fn
        reward_fns = [multi_objective_reward_fn]
        print("Reward mode: multi_objective")

    # Prepare dataset
    from datasets import Dataset

    if args.train_data:
        with open(args.train_data) as f:
            train_dataset = Dataset.from_list(json.load(f))
        if args.val_data and os.path.exists(args.val_data):
            with open(args.val_data) as f:
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
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        use_vllm=args.use_vllm,
        **({"vllm_mode": args.vllm_mode} if args.use_vllm else {}),
        **({"vllm_server_base_url": args.vllm_server_base_url} if args.vllm_server_base_url else {}),
        **({"vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization} if args.use_vllm else {}),
        **({"vllm_group_port": args.vllm_group_port} if args.use_vllm and args.vllm_mode == "server" else {}),
        **({"vllm_enable_sleep_mode": True} if args.use_vllm and args.vllm_enable_sleep_mode else {}),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_steps=args.save_steps,
        save_total_limit=20,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        run_name=os.path.basename(args.output_dir),
        deepspeed=args.deepspeed,
    )

    # Load model
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Loading model: {args.model_name} (attn: {attn_impl})")
    device_map = {"": args.device_id} if args.device_id is not None else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
        attn_implementation=attn_impl,
        trust_remote_code=True,
        **({"device_map": device_map} if device_map else {}),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Base models need a chat template for TRL GRPO
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        print("Set basic chat template for base model")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train with auto-resume
    import glob
    ckpts = sorted(
        glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"Training complete! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
