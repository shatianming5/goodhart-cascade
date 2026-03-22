"""
GRPO (Group Relative Policy Optimization) Trainer with vLLM.

Uses vLLM for fast rollout generation and implements GRPO's
group-relative advantage computation.
"""

import gc
import json
import os
import time
from dataclasses import dataclass, field

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from src.rewards.reward_functions import compute_reward, compute_rewards_batch
from src.rewards.all_metrics import compute_all_metrics


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    experiment_name: str = "R1_test_only"
    model_name: str = "Qwen/Qwen3-Coder-7B"
    trust_remote_code: bool = True

    # LoRA
    use_lora: bool = False
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Reward weights
    reward_weights: dict[str, float] = field(default_factory=lambda: {"test": 1.0})

    # Training
    total_steps: int = 1000
    batch_size: int = 16
    rollouts_per_prompt: int = 8
    lr: float = 5e-7
    kl_coeff: float = 0.03
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    clip_range: float = 0.2
    generation_max_length: int = 2048

    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096

    # Checkpointing
    save_every: int = 100
    output_dir: str = "results/R1_test_only"

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()
        config.experiment_name = raw.get("experiment_name", config.experiment_name)

        model_cfg = raw.get("model", {})
        config.model_name = model_cfg.get("name", config.model_name)
        config.trust_remote_code = model_cfg.get("trust_remote_code", True)
        config.use_lora = model_cfg.get("use_lora", False)
        config.lora_rank = model_cfg.get("lora_rank", 128)
        config.lora_alpha = model_cfg.get("lora_alpha", 256)
        config.lora_target_modules = model_cfg.get("lora_target_modules", config.lora_target_modules)

        config.reward_weights = raw.get("reward", {"test": 1.0})

        train_cfg = raw.get("training", {})
        config.total_steps = train_cfg.get("total_steps", config.total_steps)
        config.batch_size = train_cfg.get("batch_size", config.batch_size)
        config.rollouts_per_prompt = train_cfg.get("rollouts_per_prompt", config.rollouts_per_prompt)
        config.lr = train_cfg.get("lr", config.lr)
        config.kl_coeff = train_cfg.get("kl_coeff", config.kl_coeff)
        config.warmup_steps = train_cfg.get("warmup_steps", config.warmup_steps)
        config.max_grad_norm = train_cfg.get("max_grad_norm", config.max_grad_norm)
        config.clip_range = train_cfg.get("clip_range", config.clip_range)
        config.generation_max_length = train_cfg.get("generation_max_length", config.generation_max_length)

        vllm_cfg = raw.get("vllm", {})
        config.tensor_parallel_size = vllm_cfg.get("tensor_parallel_size", config.tensor_parallel_size)
        config.gpu_memory_utilization = vllm_cfg.get("gpu_memory_utilization", config.gpu_memory_utilization)
        config.max_model_len = vllm_cfg.get("max_model_len", config.max_model_len)

        ckpt_cfg = raw.get("checkpointing", {})
        config.save_every = ckpt_cfg.get("save_every", config.save_every)
        config.output_dir = ckpt_cfg.get("output_dir", config.output_dir)

        return config


class PromptDataset(Dataset):
    """Dataset of coding prompts for GRPO training."""

    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} prompts from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        test_cases = item.get("test_cases", {})
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)
        return {
            "prompt": item["prompt"],
            "starter_code": item.get("starter_code", ""),
            "test_cases": test_cases,
        }


def build_code_prompt(problem: dict) -> str:
    """Build code generation prompt."""
    prompt = (
        "Write a Python solution for the following problem. "
        "Only output the code, no explanations.\n\n"
        f"Problem:\n{problem['prompt']}\n"
    )
    if problem.get("starter_code"):
        prompt += f"\nStarter code:\n{problem['starter_code']}\n"
    prompt += "\nSolution:\n```python\n"
    return prompt


def extract_code(response: str) -> str:
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


class GRPOTrainer:
    """
    GRPO Trainer with vLLM for fast generation.

    Training loop:
    1. Sample a batch of prompts
    2. Generate K rollouts per prompt using vLLM
    3. Compute rewards for each rollout
    4. Compute group-relative advantages (within each prompt's group)
    5. Update policy with clipped surrogate objective + KL penalty
    """

    def __init__(self, config: GRPOConfig, data_path: str):
        self.config = config
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(config.output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(config), f, default_flow_style=False)

        self._init_model()
        self._init_dataset()
        self.training_log = []

    def _init_model(self):
        """Initialize policy model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model for training
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()

        self.policy_model.to(self.device)

        # Reference model (frozen) for KL divergence
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # LR scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _init_dataset(self):
        """Initialize prompt dataset."""
        self.dataset = PromptDataset(self.data_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        self.data_iter = iter(self.dataloader)

    def _collate_fn(self, batch):
        """Custom collate function."""
        return batch

    def _get_batch(self):
        """Get next batch, cycling through dataset."""
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def _generate_rollouts(self, prompts: list[str]) -> list[list[str]]:
        """Generate K rollouts per prompt using vLLM."""
        # Initialize vLLM for generation (we reload each time to free GPU mem)
        # In practice, you'd want to keep this alive or use a separate GPU
        self.policy_model.cpu()
        torch.cuda.empty_cache()

        # Save current policy weights for vLLM
        tmp_model_dir = os.path.join(self.config.output_dir, "_tmp_policy")
        self.policy_model.save_pretrained(tmp_model_dir)
        self.tokenizer.save_pretrained(tmp_model_dir)

        llm = LLM(
            model=tmp_model_dir,
            trust_remote_code=True,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
        )

        sampling_params = SamplingParams(
            n=self.config.rollouts_per_prompt,
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.config.generation_max_length,
            stop=["```\n", "\n\n\n"],
        )

        outputs = llm.generate(prompts, sampling_params)

        # Extract codes
        all_codes = []
        for output in outputs:
            codes = [extract_code(o.text) for o in output.outputs]
            all_codes.append(codes)

        # Cleanup vLLM
        del llm
        torch.cuda.empty_cache()
        gc.collect()

        # Move policy back to GPU
        self.policy_model.to(self.device)

        return all_codes

    def _compute_log_probs(self, model, prompt_text: str, completion_text: str) -> torch.Tensor:
        """Compute log probabilities of completion given prompt."""
        full_text = prompt_text + completion_text
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_model_len,
        ).to(self.device)

        prompt_tokens = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_model_len,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get log probs of completion tokens
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = inputs["input_ids"][:, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()

    def _compute_advantages(
        self, rewards: list[list[float]]
    ) -> list[list[float]]:
        """
        Compute group-relative advantages.
        For each prompt, normalize rewards within the group of K rollouts.
        """
        advantages = []
        for group_rewards in rewards:
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (
                sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)
            ) ** 0.5
            if std_r < 1e-8:
                # All same -> zero advantage (no gradient signal)
                advantages.append([0.0] * len(group_rewards))
            else:
                advantages.append([(r - mean_r) / std_r for r in group_rewards])
        return advantages

    def train_step(self, step: int) -> dict:
        """Execute one GRPO training step."""
        batch = self._get_batch()

        # Build prompts
        prompts = [build_code_prompt(item) for item in batch]

        # Generate rollouts
        all_codes = self._generate_rollouts(prompts)

        # Compute rewards (batch for speed)
        rewards = []
        reward_details = []
        for item, codes in zip(batch, all_codes):
            group_details = compute_rewards_batch(
                codes, item["test_cases"], self.config.reward_weights
            )
            group_rewards = [d["total"] for d in group_details]
            rewards.append(group_rewards)
            reward_details.append(group_details)

        # Compute advantages
        advantages = self._compute_advantages(rewards)

        # Policy gradient update
        self.policy_model.train()
        total_loss = 0.0
        total_kl = 0.0
        n_updates = 0

        for prompt_idx, (prompt, codes, advs) in enumerate(
            zip(prompts, all_codes, advantages)
        ):
            for code, adv in zip(codes, advs):
                if abs(adv) < 1e-8:
                    continue  # Skip zero-advantage samples

                completion = code
                # Policy log prob
                policy_log_prob = self._compute_log_probs(
                    self.policy_model, prompt, completion
                )
                # Reference log prob
                with torch.no_grad():
                    ref_log_prob = self._compute_log_probs(
                        self.ref_model, prompt, completion
                    )

                # KL divergence
                kl = (policy_log_prob - ref_log_prob).detach()

                # Ratio
                ratio = torch.exp(policy_log_prob - policy_log_prob.detach())

                # Clipped surrogate loss
                adv_tensor = torch.tensor(adv, device=self.device, dtype=torch.float32)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * adv_tensor
                policy_loss = -torch.min(surr1, surr2)

                # Total loss = policy loss + KL penalty
                loss = policy_loss + self.config.kl_coeff * kl

                loss.backward()
                total_loss += loss.item()
                total_kl += kl.item()
                n_updates += 1

        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Log metrics
        flat_rewards = [r for group in rewards for r in group]
        avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0
        pass_rate = sum(1 for r in flat_rewards if r > 0.5) / len(flat_rewards) if flat_rewards else 0

        # Per-component reward averages (all dimensions, not just reward ones)
        component_avgs = {}
        for component in self.config.reward_weights:
            vals = [
                d[component]
                for group in reward_details
                for d in group
                if component in d
            ]
            if vals:
                component_avgs[f"reward_{component}"] = sum(vals) / len(vals)

        # Log ALL quality dimensions for theory verification
        # Sample a few codes from this batch to compute full metrics
        all_dim_avgs = {}
        sample_codes = [codes[0] for codes in all_codes if codes][:4]  # first code from first 4 prompts
        if sample_codes:
            all_dims = ["test", "pylint", "complexity", "comment", "duplication",
                        "type_hint", "avg_func_length", "magic_numbers",
                        "nesting_depth", "dead_code", "naming_length"]
            for dim in all_dims:
                vals = []
                for code in sample_codes:
                    try:
                        m = compute_all_metrics(code)
                        if dim in m:
                            vals.append(m[dim])
                    except Exception:
                        pass
                if vals:
                    all_dim_avgs[f"dim_{dim}"] = sum(vals) / len(vals)

        # Write per-rollout logs for theory verification
        rollout_log_path = os.path.join(self.config.output_dir, "rollout_logs.jsonl")
        with open(rollout_log_path, "a") as f:
            for prompt_idx, (codes, details) in enumerate(zip(all_codes, reward_details)):
                for rollout_idx, (code, detail) in enumerate(zip(codes, details)):
                    entry = {
                        "step": step,
                        "prompt_idx": prompt_idx,
                        "rollout_idx": rollout_idx,
                        **detail,
                    }
                    f.write(json.dumps(entry) + "\n")

        metrics = {
            "step": step,
            "loss": total_loss / max(n_updates, 1),
            "kl": total_kl / max(n_updates, 1),
            "avg_reward": avg_reward,
            "pass_rate": pass_rate,
            "n_updates": n_updates,
            "lr": self.scheduler.get_last_lr()[0],
            **component_avgs,
            **all_dim_avgs,
        }
        return metrics

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.policy_model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        # Save training log
        log_path = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        print(f"  Checkpoint saved to {ckpt_dir}")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting GRPO training: {self.config.experiment_name}")
        print(f"  Steps: {self.config.total_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Rollouts/prompt: {self.config.rollouts_per_prompt}")
        print(f"  Reward weights: {self.config.reward_weights}")
        print(f"  Output: {self.config.output_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for step in range(1, self.config.total_steps + 1):
            step_start = time.time()
            metrics = self.train_step(step)
            step_time = time.time() - step_start

            metrics["step_time"] = step_time
            self.training_log.append(metrics)

            # Print progress
            if step % 10 == 0 or step == 1:
                elapsed = time.time() - start_time
                print(
                    f"Step {step}/{self.config.total_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={metrics['avg_reward']:.4f} | "
                    f"pass={metrics['pass_rate']:.2%} | "
                    f"kl={metrics['kl']:.4f} | "
                    f"time={step_time:.1f}s | "
                    f"elapsed={elapsed/60:.1f}min"
                )

            # Save checkpoint
            if step % self.config.save_every == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(self.config.total_steps)
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.1f} hours")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Path to sweet spot data JSON")
    args = parser.parse_args()

    config = GRPOConfig.from_yaml(args.config)
    trainer = GRPOTrainer(config, args.data)
    trainer.train()


if __name__ == "__main__":
    main()
