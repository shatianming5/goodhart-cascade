"""Tests for temporal analysis."""

import pytest

from goodhart.analysis.temporal import detect_changepoints, granger_causality, full_temporal_analysis


class TestDetectChangepoints:
    def test_obvious_changepoint(self):
        # Flat then jump
        series = [1.0] * 20 + [5.0] * 20
        bkps = detect_changepoints(series, min_size=3)
        # Should detect around index 20
        assert len(bkps) > 0
        assert any(15 <= b <= 25 for b in bkps)

    def test_constant_series(self):
        series = [1.0] * 20
        bkps = detect_changepoints(series, min_size=3)
        # May or may not find breakpoints in constant data, but shouldn't crash
        assert isinstance(bkps, list)

    def test_too_short(self):
        assert detect_changepoints([1.0, 2.0]) == []


class TestGrangerCausality:
    def test_basic_causality(self):
        # X causes Y with lag 1
        import random
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(100)]
        y = [0.0] + [0.8 * x[i] + rng.gauss(0, 0.3) for i in range(99)]

        result = granger_causality({"x": x, "y": y}, maxlag=3)
        assert "x->y" in result
        # The causal direction should be more significant
        assert result["x->y"]["p_value"] < result.get("y->x", {"p_value": 1.0})["p_value"]

    def test_too_short(self):
        result = granger_causality({"x": [1, 2, 3], "y": [4, 5, 6]}, maxlag=5)
        assert result == {}


class TestFullTemporalAnalysis:
    def test_with_sample_data(self, sample_eval_results):
        result = full_temporal_analysis(sample_eval_results)
        assert "steps" in result
        assert "changepoints" in result
        assert "granger" in result

    def test_too_few_points(self):
        result = full_temporal_analysis([{"step": 0, "ece": 0.1}])
        assert "error" in result
