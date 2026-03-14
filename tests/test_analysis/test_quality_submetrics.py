"""Tests for quality sub-metric degradation analysis."""

import pytest

from goodhart.analysis.quality_submetrics import find_degradation_onset, find_degradation_order


class TestFindDegradationOnset:
    def test_decrease_found_with_window(self):
        # Need 3 consecutive degraded points (window=3 default)
        series = [0.8, 0.78, 0.75, 0.60, 0.50, 0.40]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1)
        # indices 3,4,5 are all degraded → onset = 3
        assert onset == 3

    def test_increase_found_with_window(self):
        series = [0.05, 0.06, 0.07, 0.20, 0.30, 0.40]
        onset = find_degradation_onset(series, direction="increase", threshold=0.1)
        assert onset is not None

    def test_no_degradation(self):
        series = [0.8, 0.79, 0.78, 0.77, 0.76, 0.75]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1)
        assert onset is None

    def test_transient_spike_not_confirmed(self):
        # One degraded point then recovery → no confirmed degradation
        series = [0.8, 0.60, 0.78, 0.77, 0.76]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1)
        assert onset is None

    def test_empty(self):
        assert find_degradation_onset([]) is None

    def test_too_short_for_window(self):
        assert find_degradation_onset([0.8, 0.5]) is None

    def test_custom_base(self):
        series = [0.5, 0.4, 0.3]
        onset = find_degradation_onset(series, base=0.8, direction="decrease", threshold=0.1)
        # All 3 points degraded from base 0.8 → onset = 0
        assert onset == 0

    def test_window_1_matches_single_point(self):
        series = [0.8, 0.78, 0.60, 0.75]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1, window=1)
        assert onset == 2


class TestFindDegradationOrder:
    def test_ordered_degradation(self):
        data = [
            {"step": i, "pylint_score": 8.0 - i * 0.5,
             "duplication_ratio": i * 0.05,
             "cyclomatic_complexity": 2.0 + i * 0.5}
            for i in range(8)
        ]
        order = find_degradation_order(data)
        assert len(order) > 0
        degraded_keys = [k for k, _ in order]
        assert "pylint_score" in degraded_keys or "cyclomatic_complexity" in degraded_keys

    def test_empty(self):
        assert find_degradation_order([]) == []
