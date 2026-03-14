"""Temporal analysis: changepoint detection and Granger causality."""

from __future__ import annotations

import warnings


def detect_changepoints(
    series: list[float], min_size: int = 3, n_bkps: int = 2
) -> list[int]:
    """Detect changepoints in a time series using ruptures (Pelt).

    Returns list of changepoint indices (0-based, exclusive).
    """
    if len(series) < min_size * 2:
        return []

    import numpy as np
    import ruptures as rpt

    signal = np.array(series).reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
    try:
        bkps = algo.predict(pen=1.0)
    except Exception:
        return []

    # Remove the last breakpoint (always == len(series))
    return [b for b in bkps if b < len(series)]


def granger_causality(
    series_dict: dict[str, list[float]], maxlag: int = 5
) -> dict[str, dict]:
    """Test Granger causality between all pairs of time series.

    Returns dict of {cause->effect: {lag, p_value, significant}}.
    """
    import numpy as np

    keys = list(series_dict.keys())
    results = {}

    min_len = min(len(v) for v in series_dict.values())
    if min_len < maxlag + 3:
        return {}

    for cause in keys:
        for effect in keys:
            if cause == effect:
                continue

            try:
                from statsmodels.tsa.stattools import grangercausalitytests

                data = np.column_stack([
                    series_dict[effect][:min_len],
                    series_dict[cause][:min_len],
                ])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

                # Find best (lowest p-value) lag
                best_lag = 1
                best_p = 1.0
                for lag in range(1, maxlag + 1):
                    if lag in test_result:
                        p_val = test_result[lag][0]["ssr_ftest"][1]
                        if p_val < best_p:
                            best_p = p_val
                            best_lag = lag

                results[f"{cause}->{effect}"] = {
                    "lag": best_lag,
                    "p_value": best_p,
                    "significant": best_p < 0.05,
                }
            except Exception:
                results[f"{cause}->{effect}"] = {
                    "lag": 0,
                    "p_value": 1.0,
                    "significant": False,
                }

    return results


def full_temporal_analysis(all_results: list[dict]) -> dict:
    """Run full temporal analysis on checkpoint results.

    all_results: list of dicts with keys like 'step', 'ece', 'quality', 'shortcut_rate', 'pass_rate'.
    """
    if len(all_results) < 5:
        return {"error": "Not enough data points for temporal analysis"}

    # Extract time series
    series = {}
    metric_keys = ["ece", "quality", "shortcut_rate", "pass_rate"]
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in all_results]
        if any(v != 0.0 for v in values):
            series[key] = values

    result = {"steps": [r.get("step", i) for i, r in enumerate(all_results)]}

    # Changepoint detection per metric
    result["changepoints"] = {}
    for key, values in series.items():
        bkps = detect_changepoints(values)
        result["changepoints"][key] = bkps

    # Granger causality
    if len(series) >= 2:
        result["granger"] = granger_causality(series)
    else:
        result["granger"] = {}

    return result
