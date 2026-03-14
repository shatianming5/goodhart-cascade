"""Quality sub-metric degradation analysis."""

from __future__ import annotations


def find_degradation_onset(
    series: list[float],
    base: float | None = None,
    direction: str = "decrease",
    threshold: float = 0.1,
    window: int = 3,
) -> int | None:
    """Find the first index where metric degrades beyond threshold from baseline,
    confirmed by `window` consecutive points all exceeding the threshold.

    Args:
        series: Time series of metric values.
        base: Baseline value (defaults to first value).
        direction: "decrease" (quality drop) or "increase" (ECE rise).
        threshold: Fractional change threshold.
        window: Number of consecutive points required to confirm degradation.

    Returns:
        Index of first degradation point in the confirmed window, or None.
    """
    if not series or len(series) < window:
        return None

    if base is None:
        base = series[0]

    if base == 0:
        return None

    def _is_degraded(val: float) -> bool:
        if direction == "decrease":
            return (base - val) / abs(base) >= threshold
        return (val - base) / abs(base) >= threshold

    for i in range(len(series) - window + 1):
        if all(_is_degraded(series[i + j]) for j in range(window)):
            return i

    return None


def find_degradation_order(checkpoints_data: list[dict]) -> list[tuple[str, int]]:
    """Find the order in which different quality sub-metrics start degrading.

    checkpoints_data: list of dicts, each with sub-metric values.
    Returns: list of (metric_name, onset_index) sorted by onset.
    """
    if not checkpoints_data:
        return []

    # Identify sub-metric keys (skip 'step' and non-numeric)
    example = checkpoints_data[0]
    metric_keys = [
        k for k, v in example.items()
        if isinstance(v, (int, float)) and k != "step"
    ]

    onsets = []
    for key in metric_keys:
        series = [d.get(key, 0.0) for d in checkpoints_data]
        base = series[0] if series else 0.0

        # Determine direction: most quality metrics decrease = bad, some increase = bad
        increase_bad = key in ("cyclomatic_complexity", "cognitive_complexity", "duplication_ratio")
        direction = "increase" if increase_bad else "decrease"

        onset = find_degradation_onset(series, base=base, direction=direction, threshold=0.1)
        if onset is not None:
            onsets.append((key, onset))

    return sorted(onsets, key=lambda x: x[1])
