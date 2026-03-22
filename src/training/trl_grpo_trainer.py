"""
GRPO Training with TRL + vLLM integration.

Uses TRL's built-in GRPOTrainer which keeps vLLM alive between steps,
avoiding the 40s/step overhead of restarting the engine.
"""

import json
import os
import sys

sys.set_int_max_str_digits(100000)

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.rewards.reward_functions import compute_reward


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sweet_spot_data(data_path: str) -> Dataset:
    """Load sweet spot data and format as HF Dataset for TRL."""
    with open(data_path) as f:
        raw = json.load(f)

    records = []
    for item in raw:
        prompt_text = (
            "Write a Python solution for the following problem. "
            "Only output the code, no explanations.\n\n"
            f"Problem:\n{item['prompt']}\n"
        )
        if item.get("starter_code"):
            prompt_text += f"\nStarter code:\n{item['starter_code']}\n"
        prompt_text += "\nSolution:\n```python\n"

        test_cases = item.get("test_cases", {})
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "test_cases": json.dumps(test_cases),
        })

    return Dataset.from_list(records)


def make_reward_fn(weights: dict[str, float]):
    """Create a reward function closure for TRL's GRPOTrainer."""

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """
        TRL calls this with a list of completion strings.
        We need to extract code and compute rewards.
        """
        prompts = kwargs.get("prompts", [None] * len(completions))
        test_cases_list = kwargs.get("test_cases", [None] * len(completions))

        rewards = []
        for completion, tc_json in zip(completions, test_cases_list):
            # Extract code from completion
            code = _extract_code(completion)

            if tc_json is None:
                rewards.append(0.0)
                continue

            test_cases = json.loads(tc_json) if isinstance(tc_json, str) else tc_json
            scores = compute_reward(code, test_cases, weights, timeout=2)
            rewards.append(scores["total"])

        return rewards

    return reward_fn


def _extract_code(response: str) -> str:
    """Extract Python code from model response."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            return code.split("```")[0].strip()
    return response.strip()


def train(config_path: str, data_path: str):
    """Main training entry point using TRL GRPOTrainer."""
    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]
    reward_weights = cfg["reward"]
    train_cfg = cfg.get("training", {})
    output_dir = cfg.get("checkpointing", {}).get("output_dir", "results/default")
    save_every = cfg.get("checkpointing", {}).get("save_every", 100)
    experiment_name = cfg.get("experiment_name", "experiment")

    print(f"{'='*60}")
    print(f"TRL GRPO Training: {experiment_name}")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_path}")
    print(f"  Reward weights: {reward_weights}")
    print(f"  Steps: {train_cfg.get('total_steps', 1000)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Load data
    dataset = load_sweet_spot_data(data_path)
    print(f"Loaded {len(dataset)} training problems")

    # TRL GRPOConfig
    total_steps = train_cfg.get("total_steps", 1000)
    batch_size = train_cfg.get("batch_size", 16)
    rollouts = train_cfg.get("rollouts_per_prompt", 8)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=experiment_name,

        # Training
        max_steps=total_steps,
        per_device_train_batch_size=batch_size,
        num_generations=rollouts,
        learning_rate=train_cfg.get("lr", 5e-7),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        beta=train_cfg.get("kl_coeff", 0.03),  # KL coefficient

        # Generation
        max_completion_length=train_cfg.get("generation_max_length", 2048),
        temperature=train_cfg.get("temperature", 0.7),

        # vLLM - colocate mode keeps engine in same process, no server needed
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.70,
        vllm_max_model_length=cfg.get("vllm", {}).get("max_model_len", 4096),

        # Checkpointing
        save_steps=save_every,
        save_total_limit=None,

        # Logging
        logging_steps=10,
        report_to="none",

        # Other
        bf16=True,
        gradient_checkpointing=True,
        seed=42,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reward function
    reward_fn = make_reward_fn(reward_weights)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # Train!
    print("\nStarting training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"\nTraining complete! Model saved to {output_dir}/final")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args.config, args.data)
