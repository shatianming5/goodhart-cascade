"""Tests for scale comparison."""

import pytest

from goodhart.analysis.scale_comparison import compare_scales


class TestCompareScales:
    def test_two_scales(self):
        results = {
            "7b": [
                {"step": 0, "ece": 0.05, "quality": 0.8, "shortcut_rate": 0.0, "pass_rate": 0.3},
                {"step": 100, "ece": 0.15, "quality": 0.6, "shortcut_rate": 0.1, "pass_rate": 0.6},
                {"step": 200, "ece": 0.30, "quality": 0.4, "shortcut_rate": 0.3, "pass_rate": 0.8},
            ],
            "14b": [
                {"step": 0, "ece": 0.03, "quality": 0.85, "shortcut_rate": 0.0, "pass_rate": 0.4},
                {"step": 100, "ece": 0.08, "quality": 0.75, "shortcut_rate": 0.05, "pass_rate": 0.7},
                {"step": 200, "ece": 0.20, "quality": 0.55, "shortcut_rate": 0.15, "pass_rate": 0.85},
            ],
        }
        comparison = compare_scales(results)
        assert "7b" in comparison
        assert "14b" in comparison
        assert comparison["7b"]["n_checkpoints"] == 3
        assert comparison["7b"]["final_ece"] == 0.30

    def test_empty_scale(self):
        results = {"7b": [], "14b": [{"step": 0, "ece": 0.05, "quality": 0.8, "shortcut_rate": 0.0, "pass_rate": 0.3}]}
        comparison = compare_scales(results)
        assert "error" in comparison["7b"]
