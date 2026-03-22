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
    Shows pass rate declining as more quality constraints are added.
    """
    experiments = ["R1", "R2", "R3", "R4", "R5"]
    n_constraints = [1, 2, 3, 4, 5]
    labels = [
        "test",
        "test+pylint",
        "test+pylint\n+complexity",
        "test+pylint\n+complexity\n+comment",
        "all 5",
    ]

    pass_rates = []
    for exp in experiments:
        if exp in results_dirs:
            results = load_results(results_dirs[exp])
            if results:
                # Get peak pass rate
                peak = max(r.get("pass_at_1", 0) for r in results)
                pass_rates.append(peak * 100)
            else:
                pass_rates.append(None)
        else:
            pass_rates.append(None)

    # Filter out None
    valid = [(n, p, l) for n, p, l in zip(n_constraints, pass_rates, labels) if p is not None]
    if not valid:
        print("No results for alignment tax plot")
        return

    ns, ps, ls = zip(*valid)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ns, ps, "b-o", linewidth=2.5, markersize=10, markerfacecolor="red")

    for n, p, l in zip(ns, ps, ls):
        ax.annotate(
            f"{p:.1f}%",
            (n, p),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(list(ns))
    ax.set_xticklabels(ls, fontsize=10)
    ax.set_xlabel("Constrained Dimensions", fontsize=12)
    ax.set_ylabel("Peak Pass@1 (%)", fontsize=12)
    ax.set_title("Alignment Tax: Pass Rate vs Number of Quality Constraints",
                 fontsize=14, fontweight="bold")

    # Shade the alignment tax region
    if len(ps) >= 2:
        ax.fill_between(
            ns, ps, [max(ps)] * len(ps),
            alpha=0.15, color="red",
            label=f"Alignment Tax (up to {max(ps) - min(ps):.1f}pp)"
        )
        ax.legend(fontsize=11)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(ps) * 1.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Alignment tax plot saved to {output_path}")


def generate_all_figures(base_dir: str = "results", output_dir: str = "figures"):
    """Generate all three key figures."""
    os.makedirs(output_dir, exist_ok=True)

    results_dirs = {}
    for exp in ["R1_test_only", "R2_test_pylint", "R3_test_pylint_complexity",
                "R4_test_pylint_complexity_comment", "R5_all"]:
        exp_dir = os.path.join(base_dir, exp)
        if os.path.exists(os.path.join(exp_dir, "eval_results.json")):
            short = exp.split("_")[0]
            results_dirs[short] = exp_dir

    if "R1" in results_dirs:
        plot_r1_dynamics(results_dirs["R1"], os.path.join(output_dir, "r1_dynamics.png"))

    if len(results_dirs) >= 2:
        plot_escape_map(results_dirs, os.path.join(output_dir, "escape_map.png"))
        plot_alignment_tax(results_dirs, os.path.join(output_dir, "alignment_tax.png"))

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)
