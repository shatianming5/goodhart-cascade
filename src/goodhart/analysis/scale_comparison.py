"""Scale comparison analysis across model sizes."""

from __future__ import annotations


def compare_scales(
    results_by_scale: dict[str, list[dict]],
) -> dict[str, dict]:
    """Compare degradation patterns across model scales.

    Args:
        results_by_scale: {"7b": [checkpoint_results...], "14b": [...]}

    Returns:
        Comparison dict with onset, severity, etc. per scale.
    """
    from goodhart.analysis.quality_submetrics import find_degradation_onset

    comparison = {}

    for scale, results in results_by_scale.items():
        if not results:
            comparison[scale] = {"error": "no data"}
            continue

        ece_series = [r.get("ece", 0.0) for r in results]
        quality_series = [r.get("quality", 0.0) for r in results]
        shortcut_series = [r.get("shortcut_rate", 0.0) for r in results]
        pass_series = [r.get("pass_rate", 0.0) for r in results]

        comparison[scale] = {
            "n_checkpoints": len(results),
            "ece_onset": find_degradation_onset(ece_series, direction="increase"),
            "quality_onset": find_degradation_onset(quality_series, direction="decrease"),
            "shortcut_onset": find_degradation_onset(shortcut_series, direction="increase"),
            "final_ece": ece_series[-1] if ece_series else 0.0,
            "final_quality": quality_series[-1] if quality_series else 0.0,
            "final_shortcut_rate": shortcut_series[-1] if shortcut_series else 0.0,
            "final_pass_rate": pass_series[-1] if pass_series else 0.0,
            "ece_delta": ece_series[-1] - ece_series[0] if len(ece_series) > 1 else 0.0,
            "quality_delta": quality_series[-1] - quality_series[0] if len(quality_series) > 1 else 0.0,
        }

    return comparison
