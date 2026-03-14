"""Tests for calibration metrics."""

import pytest

from goodhart.utils.metrics import (
    compute_by_difficulty,
    compute_ece,
    compute_ece_sampling,
    compute_overconfidence_rate,
)


class TestComputeECE:
    def test_perfect_calibration(self):
        # confidence == accuracy in each bin → ECE = 0
        confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        outcomes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ece = compute_ece(confs, outcomes)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_total_overconfidence(self):
        # all confidence=1.0, 50% accuracy → ECE = 0.5
        confs = [1.0] * 100
        outcomes = [1.0] * 50 + [0.0] * 50
        ece = compute_ece(confs, outcomes)
        assert ece == pytest.approx(0.5, abs=0.01)

    def test_known_distribution(self):
        # 10 items in bin 0.8-0.9 with conf=0.85, 60% correct → |0.85-0.6|=0.25
        confs = [0.85] * 10
        outcomes = [1.0] * 6 + [0.0] * 4
        ece = compute_ece(confs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.25, abs=0.01)

    def test_empty_input(self):
        assert compute_ece([], []) == 0.0

    def test_single_bin(self):
        confs = [0.55, 0.55]
        outcomes = [1.0, 0.0]
        ece = compute_ece(confs, outcomes, n_bins=1)
        assert ece == pytest.approx(abs(0.55 - 0.5), abs=0.01)


class TestComputeECESampling:
    def test_from_results(self):
        results = [
            {"confidence": 0.9, "pass_rate": 0.9},
            {"confidence": 0.1, "pass_rate": 0.1},
        ]
        ece = compute_ece_sampling(results)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_empty(self):
        assert compute_ece_sampling([]) == 0.0


class TestOverconfidenceRate:
    def test_no_overconfidence(self):
        confs = [0.8, 0.9, 1.0]
        outcomes = [1.0, 1.0, 1.0]
        assert compute_overconfidence_rate(confs, outcomes) == 0.0

    def test_full_overconfidence(self):
        confs = [0.8, 0.9, 1.0]
        outcomes = [0.0, 0.0, 0.0]
        assert compute_overconfidence_rate(confs, outcomes) == 1.0

    def test_threshold(self):
        confs = [0.5, 0.8]
        outcomes = [0.0, 0.0]
        # only conf=0.8 is above threshold 0.7, and it's wrong
        assert compute_overconfidence_rate(confs, outcomes) == 1.0

    def test_empty(self):
        assert compute_overconfidence_rate([], []) == 0.0

    def test_none_above_threshold(self):
        confs = [0.1, 0.2, 0.3]
        outcomes = [0.0, 0.0, 0.0]
        assert compute_overconfidence_rate(confs, outcomes) == 0.0


class TestComputeByDifficulty:
    def test_grouping(self):
        results = [
            {"difficulty": "EASY", "confidence": 0.9, "pass_rate": 0.8},
            {"difficulty": "EASY", "confidence": 0.8, "pass_rate": 0.7},
            {"difficulty": "HARD", "confidence": 0.6, "pass_rate": 0.3},
        ]
        grouped = compute_by_difficulty(results)
        assert "EASY" in grouped
        assert "HARD" in grouped
        assert grouped["EASY"]["count"] == 2
        assert grouped["HARD"]["count"] == 1
        assert grouped["EASY"]["mean_confidence"] == pytest.approx(0.85)
        assert grouped["HARD"]["mean_pass_rate"] == pytest.approx(0.3)

    def test_unknown_difficulty(self):
        results = [{"confidence": 0.5, "pass_rate": 0.5}]
        grouped = compute_by_difficulty(results)
        assert "unknown" in grouped
