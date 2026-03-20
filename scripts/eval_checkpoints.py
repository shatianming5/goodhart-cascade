#!/usr/bin/env python3
"""Evaluate all checkpoints for calibration + code quality trajectory."""
import os, sys, json, argparse, glob
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from goodhart.eval.calibration import CalibrationEvaluator
from goodhart.eval.code_quality import CodeQualityEvaluator
from goodhart.rewards.test_passing import extract_code_from_response


def load_model_and_tokenizer(base_model, adapter_path, device="cuda:0"):
    """Load base model + LoRA adapter, return generate function."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device
    )
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"  Loaded LoRA adapter from {adapter_path}")
    model.eval()

    def generate_fn(prompt, return_logprobs=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                output_logits=return_logprobs, return_dict_in_generate=return_logprobs,
            )

        if return_logprobs:
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Extract Yes/No logprobs from first generated token logits
            logprobs_dict = {}
            if outputs.logits and len(outputs.logits) > 0:
                first_logits = outputs.logits[0][0]  # first token logits
                probs = torch.softmax(first_logits, dim=-1)
                for token_str in ["Yes", "yes", "No", "no"]:
                    token_id = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_id:
                        import math
                        p = probs[token_id[0]].item()
                        logprobs_dict[token_str] = math.log(p) if p > 0 else -100.0
            return response, logprobs_dict
        else:
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generate_fn


def prepare_eval_problems(data_path, n_eval=200):
    """Prepare evaluation problems from training data (sample subset)."""
    with open(data_path) as f:
        data = json.load(f)
    
    import random
    random.seed(42)
    # Use last n_eval as eval set (deterministic)
    eval_data = data[-n_eval:]
    
    problems = []
    for item in eval_data:
        gt = item["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        # Extract question from prompt
        prompt_text = item["prompt"]
        if isinstance(prompt_text, list):
            # chat format
            question = prompt_text[-1].get("content", "") if prompt_text else ""
        else:
            question = prompt_text
        
        problems.append({
            "question": question,
            "test_cases": gt,
            "starter_code": "",
            "difficulty": "unknown",
        })
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="e.g. test_only_7b")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/trl_train.json")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of eval problems")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_base", action="store_true", help="Also eval base model (no adapter)")
    args = parser.parse_args()

    output_root = Path(f"outputs/{args.experiment}")
    results_dir = Path(f"results/{args.experiment}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    ckpt_dirs = sorted(glob.glob(str(output_root / "checkpoint-*")),
                       key=lambda x: int(x.split("-")[-1]))
    
    # Add base model eval (step 0)
    eval_targets = []
    if args.eval_base:
        eval_targets.append(("step_0", None))
    for ckpt in ckpt_dirs:
        step = int(ckpt.split("-")[-1])
        eval_targets.append((f"step_{step}", ckpt))

    print(f"Eval targets: {len(eval_targets)} checkpoints")
    print(f"Preparing {args.n_eval} eval problems...")
    problems = prepare_eval_problems(args.data_path, args.n_eval)
    print(f"Loaded {len(problems)} problems")

    # Evaluators (no temptation for now - needs generated tasks)
    cal_eval = CalibrationEvaluator(problems, n_samples=4)
    qual_eval = CodeQualityEvaluator(problems)

    for step_name, adapter_path in eval_targets:
        step_dir = results_dir / step_name
        if (step_dir / "done.flag").exists():
            print(f"Skipping {step_name} (already done)")
            continue

        print("\n" + "="*60)
        adapter_label = adapter_path if adapter_path else "base"
        print("Evaluating: " + step_name + " (adapter: " + adapter_label + ")")
        print("="*60)

        generate_fn = load_model_and_tokenizer(args.base_model, adapter_path, args.device)

        step_dir.mkdir(parents=True, exist_ok=True)

        # Calibration (logprob only - faster)
        print("  Running calibration (logprob)...")
        try:
            cal_results = cal_eval.evaluate_logprob(generate_fn)
            with open(step_dir / "calibration.json", "w") as f:
                json.dump(cal_results, f, indent=2)
            print(f"  ECE={cal_results[ece]:.4f}, pass_rate={cal_results[pass_rate]:.4f}, "
                  f"overconf={cal_results[overconfidence_rate]:.4f}")
        except Exception as e:
            print(f"  Calibration failed: {e}")
            cal_results = {"error": str(e)}

        # Code quality
        print("  Running code quality...")
        try:
            qual_results = qual_eval.evaluate(generate_fn)
            with open(step_dir / "code_quality.json", "w") as f:
                json.dump(qual_results, f, indent=2)
            print(f"  pylint={qual_results.get(pylint_score, 0):.2f}, "
                  f"complexity={qual_results.get(cyclomatic_complexity, 0):.2f}")
        except Exception as e:
            print(f"  Code quality failed: {e}")
            qual_results = {"error": str(e)}

        # Mark done
        with open(step_dir / "done.flag", "w") as f:
            f.write("done")

        # Free GPU memory
        import gc
        del generate_fn
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("Summary across checkpoints:")
    print(f"{Step:>8} {ECE:>8} {PassRate:>10} {Overconf:>10} {Pylint:>8} {Complexity:>12}")
    for step_name, _ in eval_targets:
        step_dir = results_dir / step_name
        cal = {}; qual = {}
        try:
            cal = json.load(open(step_dir / "calibration.json"))
        except: pass
        try:
            qual = json.load(open(step_dir / "code_quality.json"))
        except: pass
        print(f"{step_name:>8} {cal.get(ece,0):.4f} {cal.get(pass_rate,0):>10.4f} "
              f"{cal.get(overconfidence_rate,0):>10.4f} {qual.get(pylint_score,0):>8.2f} "
              f"{qual.get(cyclomatic_complexity,0):>12.2f}")


if __name__ == "__main__":
    main()
