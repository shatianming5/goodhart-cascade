#!/usr/bin/env python3
"""
Evaluate all checkpoints for 1.5B_Ropt_v2 and 1.5B_Ranti_v2.
Also evaluates the base model as baseline.
Usage: CUDA_VISIBLE_DEVICES=X python3 scripts/eval_1.5b_experiments.py --experiment <name>
"""
import json
import os
import sys
import gc
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.evaluator import Evaluator


def evaluate_experiment(experiment_name: str, base_model: str = "Qwen/Qwen2.5-Coder-1.5B"):
    results_dir = f"results/{experiment_name}"
    output_file = f"results/{experiment_name}/eval_results.json"

    # Collect checkpoints
    checkpoints = []
    for d in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, d)
        if d.startswith("checkpoint-") and os.path.isdir(path):
            step = int(d.split("-")[1])
            checkpoints.append((step, d, path))
    checkpoints.sort(key=lambda x: x[0])

    # Add final
    final_path = os.path.join(results_dir, "final")
    if os.path.isdir(final_path):
        checkpoints.append((99999, "final", final_path))

    print(f"Evaluating {experiment_name}: {len(checkpoints)} checkpoints")

    all_results = []

    # Check if we already have partial results
    if os.path.exists(output_file):
        with open(output_file) as f:
            all_results = json.load(f)
        done_steps = {r["step"] for r in all_results}
        print(f"  Resuming: {len(done_steps)} already done")
    else:
        done_steps = set()

    # Also evaluate base model if not done
    base_result_file = "results/base_1.5B_eval.json"
    if not os.path.exists(base_result_file):
        print(f"\n--- Evaluating base model: {base_model} ---")
        ev = Evaluator(model_path=base_model, n_mbpp=100, n_classeval=50,
                       gpu_memory_utilization=0.85)
        result = ev.evaluate_all()
        result["step"] = 0
        result["checkpoint"] = "base"
        result["experiment"] = "base"
        with open(base_result_file, "w") as f:
            json.dump(result, f, indent=2)
        ev.cleanup()
        del ev
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        print(f"Base model results saved to {base_result_file}")

    for step, name, path in checkpoints:
        if step in done_steps:
            print(f"  Skipping {name} (already done)")
            continue

        print(f"\n--- Evaluating {name} (step {step}) ---")
        try:
            ev = Evaluator(model_path=path, n_mbpp=100, n_classeval=50,
                           gpu_memory_utilization=0.85)
            result = ev.evaluate_all()
            result["step"] = step
            result["checkpoint"] = name
            result["experiment"] = experiment_name
            all_results.append(result)

            # Save incrementally
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)

            ev.cleanup()
            del ev
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll results saved to {output_file}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    evaluate_experiment(args.experiment)
