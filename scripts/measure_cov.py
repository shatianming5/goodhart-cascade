#!/usr/bin/env python3
"""
Measure Cov(r_test, r_k) from base model rollouts.
This is the GATE STEP: independently verify that Cov signs match our theory.
"""
import json, os, sys, signal, time
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "4"))

def main():
    print("=" * 70)
    print("COV MATRIX MEASUREMENT (Gate Step)")
    print("=" * 70)

    # Load TACO data (use 500 problems for statistical power)
    with open("data/sweet_spot_7B.json") as f:
        raw = json.load(f)
    eval_data = raw[:500]
    print(f"Using {len(eval_data)} problems × 8 rollouts = {len(eval_data)*8} code samples")

    # Build prompts
    prompts = []
    test_cases = []
    for item in eval_data:
        p = ("Write a Python solution for the following problem. "
             "Only output the code, no explanations.\n\n"
             f"Problem:\n{item['prompt']}\n")
        if item.get("starter_code"):
            p += f"\nStarter code:\n{item['starter_code']}\n"
        p += "\nSolution:\n```python\n"
        prompts.append(p)
        tc = item.get("test_cases", {})
        if isinstance(tc, str):
            tc = json.loads(tc)
        test_cases.append(tc)

    # Generate rollouts with base model
    from vllm import LLM, SamplingParams
    print(f"\nLoading Qwen2.5-Coder-7B on GPU {GPU}...")
    llm = LLM(model="Qwen/Qwen2.5-Coder-7B", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)

    sample_params = SamplingParams(n=8, temperature=0.8, top_p=0.95,
                                   max_tokens=1024, stop=["```\n", "\n\n\n"])
    print("Generating 8 rollouts per problem...")
    t0 = time.time()
    outputs = llm.generate(prompts, sample_params)
    print(f"Generation done in {time.time()-t0:.0f}s")

    # Extract code
    from src.evaluation.evaluator import extract_code

    # Score each rollout on all 5 dimensions
    print("Scoring all rollouts on 5 dimensions...")
    from src.training.trl_grpo_trainer import _batch_test
    from src.rewards.reward_functions import reward_pylint, reward_complexity, reward_comment, reward_duplication

    all_scores = []  # list of dicts, one per rollout

    for i, (output, tc) in enumerate(zip(outputs, test_cases)):
        codes = [extract_code(o.text) for o in output.outputs]

        # Batch test
        test_results = _batch_test(codes, [tc] * len(codes))

        for j, code in enumerate(codes):
            if not code.strip():
                continue

            scores = {"test": test_results[j]}

            # Pylint (skip for speed if code is empty)
            try:
                scores["pylint"] = reward_pylint(code)
            except:
                scores["pylint"] = 0.0

            # Complexity
            try:
                scores["complexity"] = reward_complexity(code)
            except:
                scores["complexity"] = 0.5

            # Comment
            try:
                scores["comment"] = reward_comment(code)
            except:
                scores["comment"] = 0.0

            # Duplication
            try:
                scores["duplication"] = reward_duplication(code)
            except:
                scores["duplication"] = 0.5

            all_scores.append(scores)

        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(outputs)} problems ({len(all_scores)} rollouts)")

    print(f"\nTotal scored rollouts: {len(all_scores)}")

    # Cleanup GPU
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Compute Cov matrix
    dims = ["test", "pylint", "complexity", "comment", "duplication"]
    data = {d: np.array([s[d] for s in all_scores]) for d in dims}

    print(f"\n{'='*70}")
    print("RESULTS: Covariance and Correlation Matrix")
    print(f"{'='*70}")

    # Summary stats
    print(f"\n  Dimension statistics ({len(all_scores)} samples):")
    print(f"  {'Dim':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*48}")
    for d in dims:
        print(f"  {d:<14} {data[d].mean():>8.3f} {data[d].std():>8.3f} "
              f"{data[d].min():>8.3f} {data[d].max():>8.3f}")

    # Correlation matrix
    print(f"\n  Correlation matrix:")
    print(f"  {'':>14}", end="")
    for d in dims:
        print(f" {d:>12}", end="")
    print()
    print(f"  {'-'*74}")

    corr_matrix = {}
    for d1 in dims:
        print(f"  {d1:<14}", end="")
        corr_matrix[d1] = {}
        for d2 in dims:
            r, p = pearsonr(data[d1], data[d2])
            corr_matrix[d1][d2] = {"r": float(r), "p": float(p)}
            sig = "*" if p < 0.05 else ""
            print(f" {r:>+11.3f}{sig}", end="")
        print()

    # Key correlations with test
    print(f"\n  KEY: Correlation with test-passing:")
    print(f"  {'Constraint':<14} {'Corr':>8} {'p-value':>10} {'Sign':>8} {'Theory prediction':>20}")
    print(f"  {'-'*64}")
    for d in ["pylint", "complexity", "comment", "duplication"]:
        r = corr_matrix["test"][d]["r"]
        p = corr_matrix["test"][d]["p"]
        sign = "+" if r > 0 else "-"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        # Our theory predicts:
        if d == "complexity":
            pred = "HELPS (ρ>0)" if r > 0 else "HURTS (ρ<0)"
            expected = "HELPS"
        elif d == "comment":
            pred = "HURTS (ρ<0)" if r < 0 else "HELPS (ρ>0)"
            expected = "HURTS"
        elif d == "pylint":
            pred = "HELPS" if r > 0 else "HURTS"
            expected = "HELPS or NEUTRAL"
        else:
            pred = "HELPS" if r > 0 else "HURTS"
            expected = "NEUTRAL"

        match = "✓" if (expected.startswith("HELPS") and r > 0) or \
                       (expected == "HURTS" and r < 0) or \
                       (expected == "NEUTRAL") else "?"
        print(f"  {d:<14} {r:>+7.4f} {p:>10.6f} {sig:>3} {sign:>5}    {pred:<20} {match}")

    # Compute predicted optimal weights
    print(f"\n  OPTIMAL WEIGHT PREDICTION:")
    rhos = {d: corr_matrix["test"][d]["r"] for d in ["pylint","complexity","comment","duplication"]}
    priority = sorted(rhos.items(), key=lambda x: -x[1])
    print(f"  Priority order: {' > '.join(f'{k}({v:+.3f})' for k,v in priority)}")

    positive = [(k,v) for k,v in priority if v > 0]
    negative = [(k,v) for k,v in priority if v <= 0]
    print(f"  Add (ρ>0): {[k for k,v in positive]}")
    print(f"  Avoid (ρ≤0): {[k for k,v in negative]}")

    # Save results
    results = {
        "n_problems": len(eval_data),
        "n_rollouts": len(all_scores),
        "correlation_matrix": corr_matrix,
        "summary_stats": {d: {"mean": float(data[d].mean()), "std": float(data[d].std())}
                          for d in dims},
        "key_correlations_with_test": {d: {"r": float(corr_matrix["test"][d]["r"]),
                                            "p": float(corr_matrix["test"][d]["p"])}
                                       for d in ["pylint","complexity","comment","duplication"]},
        "priority_order": [k for k,v in priority],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs("results/final_report", exist_ok=True)
    with open("results/final_report/measured_cov_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/final_report/measured_cov_matrix.json")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
