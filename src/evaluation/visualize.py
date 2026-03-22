"""
Visualization module for Goodhart Cascade experiments.

Produces three key figures:
1. R1 Training Dynamics Curve (rise-and-fall arc)
2. Escape Map (heatmap of metric changes across R1-R5)
3. Alignment Tax vs Constraint Dimensions (line chart)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results(experiment_dir: str) -> list[dict]:
    """Load evaluation results for an experiment."""
    path = os.path.join(experiment_dir, "eval_results.json")
    with open(path) as f:
        return json.load(f)


def load_training_log(experiment_dir: str) -> list[dict]:
    """Load training log for an experiment."""
    path = os.path.join(experiment_dir, "training_log.json")
    with open(path) as f:
        return json.load(f)


def plot_r1_dynamics(results_dir: str, output_path: str = "figures/r1_dynamics.png"):
    """
    Plot R1 training dynamics curve.
    Shows pass rate rising then falling, alongside quality metrics degrading.
    """
    results = load_results(results_dir)
    df = pd.DataFrame(results)
    df = df.sort_values("step")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("R1: Test-Only Training Dynamics", fontsize=16, fontweight="bold")

    # 1. Pass rate
    ax = axes[0, 0]
    ax.plot(df["step"], df["pass_at_1"] * 100, "b-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("Pass Rate (rise-then-fall)")
    ax.grid(True, alpha=0.3)

    # 2. ECE
    ax = axes[0, 1]
    ax.plot(df["step"], df["ece"], "r-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ECE")
    ax.set_title("Calibration Error (worsening)")
    ax.grid(True, alpha=0.3)

    # 3. Pylint
    ax = axes[0, 2]
    ax.plot(df["step"], df["pylint"], "g-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Pylint Score")
    ax.set_title("Pylint Score (declining)")
    ax.grid(True, alpha=0.3)

    # 4. Comment%
    ax = axes[1, 0]
    ax.plot(df["step"], df["comment_pct"], "m-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Comment %")
    ax.set_title("Comment Ratio (declining)")
    ax.grid(True, alpha=0.3)

    # 5. Complexity
    ax = axes[1, 1]
    ax.plot(df["step"], df["complexity"], "orange", marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cognitive Complexity")
    ax.set_title("Complexity (increasing)")
    ax.grid(True, alpha=0.3)

    # 6. Duplication
    ax = axes[1, 2]
    ax.plot(df["step"], df["duplication_pct"], "brown", marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Duplication %")
    ax.set_title("Duplication (increasing)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"R1 dynamics plot saved to {output_path}")


def plot_escape_map(
    results_dirs: dict[str, str],
    output_path: str = "figures/escape_map.png",
):
    """
    Plot escape map heatmap.
    Rows = quality dimensions, Columns = experiments R1-R5.
    Color = change from baseline (green=improved, red=degraded).
    """
    experiments = ["R1", "R2", "R3", "R4", "R5"]
    metrics = ["pass_at_1", "pylint", "complexity", "comment_pct", "duplication_pct", "type_hint_pct", "ece"]
    metric_labels = ["Pass@1", "Pylint", "Complexity", "Comment%", "Duplication%", "TypeHint%", "ECE"]

    # Collect final-step results
    final_results = {}
    for exp_name in experiments:
        if exp_name in results_dirs:
            results = load_results(results_dirs[exp_name])
            if results:
                final_results[exp_name] = results[-1]  # Last checkpoint

    if not final_results:
        print("No results found for escape map")
        return

    # Build matrix: relative change from base model
    # Use R1 step 0 or first checkpoint as baseline
    r1_results = load_results(results_dirs.get("R1", ""))
    if r1_results:
        baseline = r1_results[0]
    else:
        baseline = list(final_results.values())[0]

    # Direction: for each metric, is higher better or worse?
    higher_is_better = {
        "pass_at_1": True, "pylint": True, "comment_pct": True,
        "type_hint_pct": True,
    }
    higher_is_worse = {
        "complexity": True, "duplication_pct": True, "ece": True,
    }

    matrix = np.zeros((len(metrics), len(experiments)))
    for j, exp in enumerate(experiments):
        if exp not in final_results:
            continue
        for i, metric in enumerate(metrics):
            base_val = baseline.get(metric, 0)
            final_val = final_results[exp].get(metric, 0)

            if base_val == 0:
                change = 0
            else:
                change = (final_val - base_val) / abs(base_val) * 100

            # Flip sign for "higher is worse" metrics
            if metric in higher_is_worse:
                change = -change

            matrix[i, j] = change

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    vmax = max(abs(matrix.min()), abs(matrix.max()), 50)

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=experiments,
        yticklabels=metric_labels,
        ax=ax,
        cbar_kws={"label": "% Change (green=better, red=worse)"},
        linewidths=0.5,
    )

    ax.set_title("Escape Map: Gaming Displacement Across Experiments", fontsize=14, fontweight="bold")
    ax.set_xlabel("Experiment (cumulative constraints)")
    ax.set_ylabel("Quality Dimension")

    # Add constraint annotations
    constraints = {
        "R1": "test",
        "R2": "+pylint",
        "R3": "+complexity",
        "R4": "+comment",
        "R5": "+duplication",
    }
    for j, exp in enumerate(experiments):
        ax.text(j + 0.5, -0.3, constraints.get(exp, ""),
                ha="center", va="top", fontsize=9, style="italic")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Escape map saved to {output_path}")


def plot_alignment_tax(
    results_dirs: dict[str, str],
    output_path: str = "figures/alignment_tax.png",
):
    """
    Plot alignment tax vs number of constraint dimensions.
    Now supports 3 model scales on the same plot.
    """
    n_constraints = [1, 2, 3, 4, 5]
    labels = ["R1\ntest", "R2\n+pylint", "R3\n+complex", "R4\n+comment", "R5\nall 5"]

    scales = {"1.5B": "green", "7B": "blue", "14B": "red"}

    fig, ax = plt.subplots(figsize=(11, 7))

    for scale, color in scales.items():
        pass_rates = []
        for r_idx in range(1, 6):
            key = f"{scale}_R{r_idx}"
            if key in results_dirs:
                results = load_results(results_dirs[key])
                if results:
                    peak = max(r.get("pass_at_1", 0) for r in results)
                    pass_rates.append(peak * 100)
                else:
                    pass_rates.append(None)
            else:
                # Try old naming
                key2 = f"R{r_idx}"
                if key2 in results_dirs:
                    results = load_results(results_dirs[key2])
                    if results:
                        peak = max(r.get("pass_at_1", 0) for r in results)
                        pass_rates.append(peak * 100)
                    else:
                        pass_rates.append(None)
                else:
                    pass_rates.append(None)

        valid = [(n, p) for n, p in zip(n_constraints, pass_rates) if p is not None]
        if not valid:
            continue

        ns, ps = zip(*valid)
        ax.plot(ns, ps, f"-o", color=color, linewidth=2.5, markersize=8,
                label=f"{scale}", alpha=0.85)

        for n, p in zip(ns, ps):
            ax.annotate(f"{p:.1f}%", (n, p), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax.set_xticks(n_constraints)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Constrained Dimensions", fontsize=12)
    ax.set_ylabel("Peak Pass@1 (%)", fontsize=12)
    ax.set_title("Alignment Tax Across Model Scales",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, title="Model Scale")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Alignment tax plot saved to {output_path}")


def generate_all_figures(base_dir: str = "results", output_dir: str = "figures"):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    # Discover all experiment results
    results_dirs = {}
    for entry in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, entry)
        if os.path.exists(os.path.join(exp_dir, "eval_results.json")):
            results_dirs[entry] = exp_dir

    print(f"Found results for: {list(results_dirs.keys())}")

    # R1 dynamics for each scale
    for scale in ["1.5B", "7B", "14B"]:
        key = f"{scale}_R1"
        if key in results_dirs:
            plot_r1_dynamics(results_dirs[key],
                             os.path.join(output_dir, f"r1_dynamics_{scale}.png"))

    # Escape map per scale
    for scale in ["1.5B", "7B", "14B"]:
        scale_dirs = {f"R{i}": results_dirs[f"{scale}_R{i}"]
                      for i in range(1, 6) if f"{scale}_R{i}" in results_dirs}
        if len(scale_dirs) >= 2:
            plot_escape_map(scale_dirs,
                            os.path.join(output_dir, f"escape_map_{scale}.png"))

    # Alignment tax across all scales
    if len(results_dirs) >= 2:
        plot_alignment_tax(results_dirs, os.path.join(output_dir, "alignment_tax.png"))

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)
