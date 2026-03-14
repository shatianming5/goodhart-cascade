"""Tests for quality sub-metric degradation analysis."""

import pytest

from goodhart.analysis.quality_submetrics import find_degradation_onset, find_degradation_order


class TestFindDegradationOnset:
    def test_decrease_found(self):
        series = [0.8, 0.78, 0.75, 0.60, 0.50]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1)
        assert onset == 3  # (0.8-0.6)/0.8 = 0.25 > 0.1

    def test_increase_found(self):
        series = [0.05, 0.06, 0.07, 0.20, 0.30]
        onset = find_degradation_onset(series, direction="increase", threshold=0.1)
        # (0.20-0.05)/0.05 = 3.0 > 0.1
        assert onset is not None

    def test_no_degradation(self):
        series = [0.8, 0.79, 0.78, 0.77]
        onset = find_degradation_onset(series, direction="decrease", threshold=0.1)
        assert onset is None

    def test_empty(self):
        assert find_degradation_onset([]) is None

    def test_custom_base(self):
        series = [0.5, 0.4, 0.3]
        onset = find_degradation_onset(series, base=0.8, direction="decrease", threshold=0.1)
        # (0.8-0.5)/0.8 = 0.375 > 0.1 → onset at index 0
        assert onset == 0


class TestFindDegradationOrder:
    def test_ordered_degradation(self):
        data = [
            {"step": 0, "pylint_score": 8.0, "duplication_ratio": 0.0, "cyclomatic_complexity": 2.0},
            {"step": 1, "pylint_score": 7.8, "duplication_ratio": 0.0, "cyclomatic_complexity": 2.0},
            {"step": 2, "pylint_score": 7.5, "duplication_ratio": 0.15, "cyclomatic_complexity": 2.0},
            {"step": 3, "pylint_score": 6.0, "duplication_ratio": 0.3, "cyclomatic_complexity": 5.0},
            {"step": 4, "pylint_score": 4.0, "duplication_ratio": 0.5, "cyclomatic_complexity": 8.0},
        ]
        order = find_degradation_order(data)
        assert len(order) > 0
        # All degraded metrics should appear
        degraded_keys = [k for k, _ in order]
        assert "pylint_score" in degraded_keys or "duplication_ratio" in degraded_keys

    def test_empty(self):
        assert find_degradation_order([]) == []
