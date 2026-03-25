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

    def reward_fn(completions, **kwargs) -> list[float]:
        """
        TRL calls this with completions (list of strings or list of message dicts).
        Batch-parallel reward computation for speed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import subprocess, tempfile, re

        test_cases_list = kwargs.get("test_cases", [None] * len(completions))

        # Step 1: Extract all codes
        codes = []
        tc_list = []
        for completion, tc_json in zip(completions, test_cases_list):
            try:
                if isinstance(completion, list):
                    text = completion[-1]["content"] if completion else ""
                elif isinstance(completion, dict):
                    text = completion.get("content", str(completion))
                else:
                    text = str(completion)
                codes.append(_extract_code(text))
                tc = json.loads(tc_json) if isinstance(tc_json, str) else tc_json
                tc_list.append(tc)
            except Exception:
                codes.append("")
                tc_list.append(None)

        n = len(codes)

        # Step 2: Parallel test execution (subprocess-based, ~32 at a time)
        test_scores = _batch_test(codes, tc_list)

        # Step 3: Parallel pylint (if in weights)
        pylint_scores = [0.5] * n
        if "pylint" in weights:
            pylint_scores = _batch_pylint(codes)

        # Step 4: Fast CPU metrics (complexity, comment, duplication)
        complexity_scores = [0.5] * n
        comment_scores = [0.5] * n
        dup_scores = [0.5] * n
        for i, code in enumerate(codes):
            if not code:
                continue
            if "complexity" in weights:
                from src.rewards.reward_functions import reward_complexity
                complexity_scores[i] = reward_complexity(code)
            if "comment" in weights:
                from src.rewards.reward_functions import reward_comment
                comment_scores[i] = reward_comment(code)
            if "duplication" in weights:
                from src.rewards.reward_functions import reward_duplication
                dup_scores[i] = reward_duplication(code)

        # Step 5: Combine
        rewards = []
        for i in range(n):
            total = 0.0
            if "test" in weights:
                total += weights["test"] * test_scores[i]
            if "pylint" in weights:
                total += weights["pylint"] * pylint_scores[i]
            if "complexity" in weights:
                total += weights["complexity"] * complexity_scores[i]
            if "comment" in weights:
                total += weights["comment"] * comment_scores[i]
            if "duplication" in weights:
                total += weights["duplication"] * dup_scores[i]
            rewards.append(total)

        return rewards

    return reward_fn


def _batch_test(codes: list[str], test_cases_list: list[dict | None], max_workers: int = 32) -> list[float]:
    """Run tests for all codes in parallel using subprocess."""
    import subprocess, tempfile, time
    from src.data.filter_sweet_spot import _build_fn_test_script, _build_stdio_test_script

    results = [0.0] * len(codes)
    tasks = []

    for i, (code, tc) in enumerate(zip(codes, test_cases_list)):
        if not code or tc is None:
            continue
        fn_name = tc.get("fn_name", "")
        inputs = tc.get("inputs", [])
        outputs = tc.get("outputs", [])
        if fn_name:
            script = _build_fn_test_script(code, fn_name, inputs, outputs)
        elif inputs and outputs:
            script = _build_stdio_test_script(code, inputs, outputs)
        else:
            continue
        tmp = f"/tmp/_rtest_{i}.py"
        with open(tmp, "w") as f:
            f.write(script)
        tasks.append((i, tmp))

    # Launch in waves
    wave_size = min(max_workers, len(tasks))
    idx = 0
    while idx < len(tasks):
        wave = tasks[idx:idx + wave_size]
        procs = []
        for i, path in wave:
            proc = subprocess.Popen(
                [sys.executable, path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            procs.append((i, path, proc))

        deadline = time.time() + 3
        for i, path, proc in procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
                out = proc.stdout.read().decode().strip() if proc.stdout else ""
                results[i] = 1.0 if (proc.returncode == 0 and out == "PASS") else 0.0
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except Exception:
                    proc.kill()
                results[i] = 0.0
            finally:
                try:
                    proc.stdout.close()
                    os.unlink(path)
                except Exception:
                    pass
        idx += wave_size

    return results


def _batch_pylint(codes: list[str], max_workers: int = 16) -> list[float]:
    """Run pylint on all codes in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor
    from src.rewards.reward_functions import reward_pylint

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(reward_pylint, codes))
    return results


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


def train(config_path: str, data_path: str, vllm_port: int = 8000, vllm_mode: str = "server"):
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
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_generations=rollouts,
        learning_rate=train_cfg.get("lr", 5e-7),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        beta=train_cfg.get("kl_coeff", 0.03),  # KL coefficient

        # Generation
        max_completion_length=train_cfg.get("generation_max_length", 2048),
        temperature=train_cfg.get("temperature", 0.7),

        # vLLM - each experiment gets unique server_port + group_port to avoid NCCL conflicts
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=vllm_port,
        vllm_group_port=51216 + vllm_port - 8000,  # unique per experiment
        vllm_gpu_memory_utilization=0.85,
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
    lora_cfg = cfg.get("lora", None)

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

    # Apply LoRA if configured
    peft_config = None
    if lora_cfg and lora_cfg.get("enabled", False):
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_cfg.get("rank", 128),
            lora_alpha=lora_cfg.get("alpha", 256),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            task_type="CAUSAL_LM",
        )
        print(f"  LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Create reward function
    reward_fn = make_reward_fn(reward_weights)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
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
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    args = parser.parse_args()
    train(args.config, args.data, args.vllm_port, args.vllm_mode)
