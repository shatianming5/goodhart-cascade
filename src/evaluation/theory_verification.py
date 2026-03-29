"""
Theory verification for alignment tax scaling law.

Verifies assumptions and fits the theoretical tax curve:
  tax(n) = C * (1-w_test)^2 / [n*w_test^2 + (1-w_test)^2]

With model scale dependence:
  tax(n, N) = C * (1-w_test)^2 / [n*w_test^2 + (1-w_test)^2 * N^(-delta)]
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# Reward weights for each experiment tier
W_TEST = {
    0: 1.0,   # R1: test only
    1: 0.7,   # R2: +pylint
    2: 0.6,   # R3: +complexity
    3: 0.5,   # R4: +comment
    4: 0.4,   # R5: +duplication
}


def load_rollout_logs(experiment_dir: str) -> list[dict]:
    """Load per-rollout component logs."""
    path = os.path.join(experiment_dir, "rollout_logs.jsonl")
    if not os.path.exists(path):
        path = os.path.join(experiment_dir, "training_log.json")

    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)


def verify_independence(experiment_dir: str) -> dict:
    """
    Assumption 1: Reward components are approximately independent.
    Check pairwise correlations between quality dimensions.
    """
    logs = load_rollout_logs(experiment_dir)

    dims = ["test", "pylint", "complexity", "comment", "duplication"]
    available_dims = [d for d in dims if d in logs[0] if isinstance(logs[0].get(d), (int, float))]

    if len(available_dims) < 2:
        return {"max_correlation": 0, "ok": True, "available_dims": available_dims}

    scores = {d: [entry[d] for entry in logs if d in entry] for d in available_dims}

    # Compute correlation matrix
    n = len(available_dims)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(scores[available_dims[i]]) == len(scores[available_dims[j]]) and len(scores[available_dims[i]]) > 2:
                r, _ = pearsonr(scores[available_dims[i]], scores[available_dims[j]])
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = 0 if i != j else 1

    max_off_diag = max(abs(corr_matrix[i][j])
                       for i in range(n) for j in range(n) if i != j)

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "dims": available_dims,
        "max_correlation": float(max_off_diag),
        "ok": max_off_diag < 0.3,
    }


def verify_equal_variance(experiment_dir: str) -> dict:
    """
    Assumption 2: Quality dimensions have roughly equal variance.
    """
    logs = load_rollout_logs(experiment_dir)

    dims = ["test", "pylint", "complexity", "comment", "duplication"]
    available_dims = [d for d in dims if d in logs[0] if isinstance(logs[0].get(d), (int, float))]

    if len(available_dims) < 2:
        return {"variance_ratio": 1.0, "ok": True}

    variances = {}
    for d in available_dims:
        vals = [entry[d] for entry in logs if d in entry and isinstance(entry[d], (int, float))]
        if vals:
            variances[d] = float(np.var(vals))

    if not variances or min(variances.values()) == 0:
        return {"variances": variances, "variance_ratio": float("inf"), "ok": False}

    ratio = max(variances.values()) / max(min(variances.values()), 1e-10)

    return {
        "variances": variances,
        "variance_ratio": float(ratio),
        "ok": ratio < 5.0,
    }


def fit_alignment_tax(results_by_scale: dict[str, list[dict]]) -> dict:
    """
    Fit alignment tax scaling law from experimental results.

    Args:
        results_by_scale: {"1.5B": [R1_result, R2_result, ...], "7B": [...], "14B": [...]}
    """
    def theoretical_tax_single(n, C):
        w = W_TEST.get(n, 0.5)
        if n == 0:
            return 0.0  # R1 is baseline
        return C * (1 - w) ** 2 / (n * w ** 2 + (1 - w) ** 2)

    def theoretical_tax_scaled(params, n, N):
        C, delta = params
        w = W_TEST.get(n, 0.5)
        if n == 0:
            return 0.0
        return C * (1 - w) ** 2 / (n * w ** 2 + (1 - w) ** 2 * N ** (-delta))

    fit_results = {}

    # Step 1: Fit C from 7B data
    if "7B" in results_by_scale:
        r7b = results_by_scale["7B"]
        baseline_pass = r7b[0].get("pass_at_1", 0)

        n_vals = list(range(len(r7b)))
        tax_vals = [baseline_pass - r.get("pass_at_1", 0) for r in r7b]
        # Normalize to [0, 1]
        tax_vals = [max(0, t) for t in tax_vals]

        try:
            popt, pcov = curve_fit(theoretical_tax_single, n_vals[1:], tax_vals[1:],
                                    p0=[0.15], bounds=(0, 1))
            C_fit = popt[0]
            fit_results["C"] = float(C_fit)

            # Predictions
            predicted = [theoretical_tax_single(n, C_fit) for n in n_vals]
            fit_results["7B_predicted"] = predicted
            fit_results["7B_actual"] = tax_vals

            # R^2
            ss_res = sum((a - p) ** 2 for a, p in zip(tax_vals, predicted))
            ss_tot = sum((a - np.mean(tax_vals)) ** 2 for a in tax_vals)
            fit_results["7B_r2"] = float(1 - ss_res / max(ss_tot, 1e-10))

        except Exception as e:
            fit_results["7B_fit_error"] = str(e)

    # Step 2: Predict other scales
    if "C" in fit_results:
        C = fit_results["C"]
        for scale_name, scale_N in [("1.5B", 1.5), ("14B", 14)]:
            if scale_name in results_by_scale:
                r_scale = results_by_scale[scale_name]
                baseline = r_scale[0].get("pass_at_1", 0)
                actual_tax = [max(0, baseline - r.get("pass_at_1", 0)) for r in r_scale]

                # Simple prediction using same C (no scale correction first)
                predicted = [theoretical_tax_single(n, C) for n in range(len(r_scale))]

                ss_res = sum((a - p) ** 2 for a, p in zip(actual_tax, predicted))
                ss_tot = sum((a - np.mean(actual_tax)) ** 2 for a in actual_tax)
                r2 = float(1 - ss_res / max(ss_tot, 1e-10))

                fit_results[f"{scale_name}_predicted"] = predicted
                fit_results[f"{scale_name}_actual"] = actual_tax
                fit_results[f"{scale_name}_r2"] = r2
                fit_results[f"{scale_name}_mae"] = float(np.mean(np.abs(
                    np.array(actual_tax) - np.array(predicted[:len(actual_tax)])
                )))

    return fit_results


def compute_rho_star(results: list[dict], n_total: int = 5) -> dict:
    """
    Find optimal constraint count (knee point) via efficiency analysis.

    rho* = n_optimal / n_measurable
    """
    if len(results) < 2:
        return {"rho_star": 0, "knee": 0}

    baseline_pass = results[0].get("pass_at_1", 0)
    tax = [max(0, baseline_pass - r.get("pass_at_1", 0)) for r in results]

    # Quality improvement at each step (average of constrained dims)
    quality_dims = ["pylint", "complexity", "comment_pct", "duplication_pct"]
    quality_scores = []
    for r in results:
        # Higher is better for pylint, lower is better for complexity/duplication
        score = r.get("pylint", 0) / 10.0  # normalize
        score -= r.get("complexity", 0) / 20.0
        score += r.get("comment_pct", 0) / 100.0
        score -= r.get("duplication_pct", 0) / 100.0
        quality_scores.append(score)

    # Marginal efficiency at each step
    efficiencies = []
    for i in range(1, min(len(results), n_total + 1)):
        delta_quality = quality_scores[i] - quality_scores[i - 1]
        delta_tax = tax[i] - tax[i - 1]
        eff = delta_quality / max(abs(delta_tax), 0.001)
        efficiencies.append(eff)

    # Knee = where efficiency drops below 50% of initial
    knee = len(efficiencies)
    if efficiencies:
        initial_eff = abs(efficiencies[0])
        for i, eff in enumerate(efficiencies):
            if abs(eff) < 0.5 * initial_eff:
                knee = i + 1
                break

    rho_star = knee / n_total if n_total > 0 else 0

    return {
        "knee": knee,
        "rho_star": float(rho_star),
        "efficiencies": [float(e) for e in efficiencies],
        "tax": [float(t) for t in tax],
        "quality_scores": [float(q) for q in quality_scores],
    }


def plot_theory_vs_experiment(fit_results: dict, output_path: str = "figures/theory_vs_experiment.png"):
    """Scatter plot: predicted vs actual alignment tax."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"7B": "blue", "1.5B": "green", "14B": "red"}

    for scale in ["1.5B", "7B", "14B"]:
        pred_key = f"{scale}_predicted"
        act_key = f"{scale}_actual"
        if pred_key in fit_results and act_key in fit_results:
            predicted = fit_results[pred_key]
            actual = fit_results[act_key]
            n = min(len(predicted), len(actual))
            ax.scatter(actual[:n], predicted[:n],
                       c=colors.get(scale, "gray"), s=100,
                       label=f"{scale} (R²={fit_results.get(f'{scale}_r2', 0):.3f})",
                       zorder=5)

    # Identity line
    all_vals = []
    for scale in ["1.5B", "7B", "14B"]:
        for key in [f"{scale}_predicted", f"{scale}_actual"]:
            if key in fit_results:
                all_vals.extend(fit_results[key])
    if all_vals:
        max_val = max(all_vals) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Actual Alignment Tax (pass rate loss)")
    ax.set_ylabel("Predicted Alignment Tax")
    ax.set_title(f"Theory vs Experiment (C={fit_results.get('C', '?'):.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Theory vs experiment plot saved to {output_path}")


def plot_rho_star(rho_results: dict[str, dict], output_path: str = "figures/rho_star.png"):
    """Bar chart of rho* across model scales."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scales = sorted(rho_results.keys())
    rho_vals = [rho_results[s]["rho_star"] for s in scales]
    knee_vals = [rho_results[s]["knee"] for s in scales]

    bars = ax.bar(scales, rho_vals, color=["green", "blue", "red"][:len(scales)], alpha=0.7)

    for bar, rho, knee in zip(bars, rho_vals, knee_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"ρ*={rho:.2f}\n(R{knee+1})",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("ρ* (optimal constraint ratio)")
    ax.set_xlabel("Model Scale")
    ax.set_title("Universal Alignment Ratio ρ* Across Scales")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.35, color="gray", linestyle="--", alpha=0.5, label="ρ*≈0.35 (hypothesized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ρ* plot saved to {output_path}")
