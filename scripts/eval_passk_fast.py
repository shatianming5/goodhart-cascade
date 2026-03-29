#!/usr/bin/env python3
"""Fast pass@k eval: merge LoRA + vLLM offline inference."""
import os, sys, json, argparse, math, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodhart.rewards.test_passing import extract_code_from_response
from goodhart.utils.code_exec import run_all_tests


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Merge LoRA into base and save to temp dir
    merge_dir = f"/tmp/merged_{args.name}"
    if not os.path.exists(os.path.join(merge_dir, "config.json")):
        print(f"[{args.name}] Merging adapter...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if args.adapter_path and os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
        model.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)
        del model
        torch.cuda.empty_cache()
        print(f"[{args.name}] Merged to {merge_dir}")

    # Step 2: Load with vLLM for fast generation
    from vllm import LLM, SamplingParams
    print(f"[{args.name}] Loading vLLM...")
    llm = LLM(model=merge_dir, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=3072)
    sampling = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, max_tokens=2048)

    # Step 3: Prepare prompts
    with open(args.data_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompt_text = prompt[-1].get("content", "") if prompt else ""
        else:
            prompt_text = prompt
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # Step 4: Batch generate ALL at once (vLLM handles batching)
    print(f"[{args.name}] Generating {len(prompts)} × {args.k} = {len(prompts)*args.k} samples...")
    outputs = llm.generate(prompts, sampling)

    # Step 5: Evaluate
    results_per_problem = []
    for i, (item, output) in enumerate(zip(data, outputs)):
        gt = item["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        correct = 0
        for completion in output.outputs:
            code = extract_code_from_response(completion.text)
            try:
                ok, _, _ = run_all_tests(code, gt, timeout=5, max_memory_mb=512)
                if ok:
                    correct += 1
            except:
                pass
        results_per_problem.append({"n": len(output.outputs), "c": correct})
        if (i + 1) % 20 == 0:
            p1_so_far = np.mean([pass_at_k(r["n"], r["c"], 1) for r in results_per_problem])
            print(f"  [{args.name}] {i+1}/{len(data)}: pass@1={p1_so_far:.4f}")

    p1 = np.mean([pass_at_k(r["n"], r["c"], 1) for r in results_per_problem])
    p4 = np.mean([pass_at_k(r["n"], r["c"], 4) for r in results_per_problem])
    p8 = np.mean([pass_at_k(r["n"], r["c"], min(8, args.k)) for r in results_per_problem])
    total_correct = sum(r["c"] for r in results_per_problem)
    total_samples = sum(r["n"] for r in results_per_problem)

    result = {
        "name": args.name, "k": args.k, "n_problems": len(data),
        "pass_at_1": float(p1), "pass_at_4": float(p4), "pass_at_8": float(p8),
        "raw_pass_rate": total_correct / total_samples,
        "total_correct": total_correct, "total_samples": total_samples,
        "per_problem": results_per_problem
    }
    os.makedirs("results/passk", exist_ok=True)
    with open(f"results/passk/{args.name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.name}] FINAL: pass@1={p1:.4f}  pass@4={p4:.4f}  pass@8={p8:.4f}")

    # Cleanup
    shutil.rmtree(merge_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
