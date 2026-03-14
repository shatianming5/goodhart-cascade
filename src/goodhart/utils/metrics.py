"""Calibration metrics: ECE, overconfidence rate, difficulty-grouped stats."""

from __future__ import annotations

import math


def compute_ece(
    confidences: list[float], outcomes: list[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error. Bins confidences, compares mean confidence vs accuracy."""
    if not confidences:
        return 0.0
    n = len(confidences)
    bin_sums_conf = [0.0] * n_bins
    bin_sums_acc = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for conf, out in zip(confidences, outcomes):
        b = min(int(conf * n_bins), n_bins - 1)
        bin_sums_conf[b] += conf
        bin_sums_acc[b] += out
        bin_counts[b] += 1

    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_sums_conf[i] / bin_counts[i]
            avg_acc = bin_sums_acc[i] / bin_counts[i]
            ece += (bin_counts[i] / n) * abs(avg_conf - avg_acc)
    return ece


def compute_ece_sampling(results: list[dict]) -> float:
    """ECE from sampling results. Each dict has 'confidence' and 'pass_rate' (empirical)."""
    if not results:
        return 0.0
    confidences = [r["confidence"] for r in results]
    outcomes = [r["pass_rate"] for r in results]
    return compute_ece(confidences, outcomes)


def compute_overconfidence_rate(
    confidences: list[float],
    outcomes: list[float],
    threshold: float = 0.7,
) -> float:
    """Fraction of high-confidence predictions (>= threshold) that are wrong."""
    if not confidences:
        return 0.0
    high_conf = [(c, o) for c, o in zip(confidences, outcomes) if c >= threshold]
    if not high_conf:
        return 0.0
    wrong = sum(1 for _, o in high_conf if o < 0.5)
    return wrong / len(high_conf)


def compute_by_difficulty(results: list[dict]) -> dict[str, dict]:
    """Group results by difficulty level, compute per-group stats."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        groups.setdefault(diff, []).append(r)

    out = {}
    for diff, items in groups.items():
        confs = [r["confidence"] for r in items if "confidence" in r]
        outcomes = [r["pass_rate"] for r in items if "pass_rate" in r]
        out[diff] = {
            "count": len(items),
            "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
            "mean_pass_rate": sum(outcomes) / len(outcomes) if outcomes else 0.0,
            "ece": compute_ece(confs, outcomes) if confs else 0.0,
        }
    return out
