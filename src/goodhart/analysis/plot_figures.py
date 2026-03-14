"""Paper figure generation for Goodhart Cascade analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_degradation_main(
    results: list[dict], output_path: str, title: str = "Goodhart Cascade"
):
    """Figure 1: Three-dimensional degradation (ECE + quality + shortcut vs step)."""
    steps = [r["step"] for r in results]
    ece = [r.get("ece", 0) for r in results]
    quality = [r.get("quality", 0) for r in results]
    shortcut = [r.get("shortcut_rate", 0) for r in results]
    pass_rate = [r.get("pass_rate", 0) for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Rate / Score")

    ax1.plot(steps, pass_rate, "g-o", label="Pass Rate", markersize=4)
    ax1.plot(steps, quality, "b-s", label="Code Quality", markersize=4)
    ax1.plot(steps, ece, "r-^", label="ECE (Miscalibration)", markersize=4)
    ax1.plot(steps, shortcut, "m-d", label="Shortcut Rate", markersize=4)

    ax1.legend(loc="center right")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(
    confidences: list[float], outcomes: list[float], output_path: str, n_bins: int = 10
):
    """Figure 2: Calibration reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = [(bin_edges[i] <= c < bin_edges[i + 1]) for c in confidences]
        count = sum(mask)
        if count > 0:
            acc = sum(o for o, m in zip(outcomes, mask) if m) / count
            conf = sum(c for c, m in zip(confidences, mask) if m) / count
        else:
            acc = 0
            conf = 0
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(count)

    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.5, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_spider(metrics: dict, output_path: str, title: str = "Code Quality"):
    """Figure 3: Spider/radar chart for code quality sub-dimensions."""
    keys = [k for k in metrics if k != "n_samples" and isinstance(metrics[k], (int, float))]
    if not keys:
        return

    values = [metrics[k] for k in keys]

    # Normalize to [0, 1] for display
    max_vals = {"pylint_score": 10.0, "lines_of_code": 100.0}
    norm_values = []
    for k, v in zip(keys, values):
        if k in max_vals:
            norm_values.append(min(1.0, v / max_vals[k]))
        else:
            norm_values.append(min(1.0, max(0.0, v)))

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    norm_values = norm_values + [norm_values[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, norm_values, alpha=0.25)
    ax.plot(angles, norm_values, "o-")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scale_comparison(
    comparison: dict[str, dict], output_path: str
):
    """Figure 4: Scale comparison chart."""
    scales = list(comparison.keys())
    valid = [s for s in scales if "error" not in comparison[s]]

    if not valid:
        return

    metrics = ["final_ece", "final_quality", "final_shortcut_rate", "final_pass_rate"]
    labels = ["ECE", "Quality", "Shortcut Rate", "Pass Rate"]

    x = np.arange(len(labels))
    width = 0.8 / len(valid)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, scale in enumerate(valid):
        values = [comparison[scale].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=scale)

    ax.set_xticks(x + width * (len(valid) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Scale Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_vs_test_only(
    test_only: list[dict], multi_obj: list[dict], output_path: str
):
    """Figure 5: Multi-objective vs test-only comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metric_pairs = [
        ("pass_rate", "Pass Rate"),
        ("ece", "ECE"),
        ("quality", "Code Quality"),
        ("shortcut_rate", "Shortcut Rate"),
    ]

    for ax, (key, label) in zip(axes.flat, metric_pairs):
        steps_to = [r["step"] for r in test_only]
        vals_to = [r.get(key, 0) for r in test_only]
        steps_mo = [r["step"] for r in multi_obj]
        vals_mo = [r.get(key, 0) for r in multi_obj]

        ax.plot(steps_to, vals_to, "r-o", label="Test-only", markersize=4)
        ax.plot(steps_mo, vals_mo, "b-s", label="Multi-obj", markersize=4)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_granger_heatmap(granger_results: dict, output_path: str):
    """Figure 6: Granger causality heatmap."""
    if not granger_results:
        return

    # Extract unique variable names
    vars_set = set()
    for key in granger_results:
        cause, effect = key.split("->")
        vars_set.add(cause)
        vars_set.add(effect)
    variables = sorted(vars_set)

    n = len(variables)
    matrix = np.ones((n, n))

    for key, val in granger_results.items():
        cause, effect = key.split("->")
        i = variables.index(cause)
        j = variables.index(effect)
        matrix[i, j] = val.get("p_value", 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(variables, rotation=45)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Effect")
    ax.set_ylabel("Cause")
    ax.set_title("Granger Causality (p-values)")
    fig.colorbar(im)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
