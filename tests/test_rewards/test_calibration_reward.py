"""Tests for calibration penalty reward."""

import pytest

from goodhart.rewards.calibration import compute_calibration_penalty, extract_confidence


class TestComputeCalibrationPenalty:
    def test_perfect_calibration(self):
        assert compute_calibration_penalty(0.8, 0.8) == 0.0

    def test_overconfident(self):
        assert compute_calibration_penalty(1.0, 0.0) == -1.0

    def test_underconfident(self):
        assert compute_calibration_penalty(0.0, 1.0) == -1.0

    def test_partial(self):
        assert compute_calibration_penalty(0.7, 0.5) == pytest.approx(-0.2)

    def test_range(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for out in [0.0, 0.5, 1.0]:
                p = compute_calibration_penalty(conf, out)
                assert -1.0 <= p <= 0.0


class TestExtractConfidence:
    def test_colon_format(self):
        assert extract_confidence("Confidence: 0.85") == pytest.approx(0.85)

    def test_percentage_format(self):
        assert extract_confidence("confidence: 85%") == pytest.approx(0.85)

    def test_percent_confident(self):
        assert extract_confidence("I am 90% confident") == pytest.approx(0.9)

    def test_standalone_decimal(self):
        assert extract_confidence("My answer is correct.\n0.75\n") == pytest.approx(0.75)

    def test_no_number(self):
        assert extract_confidence("I think this is right") == 0.5

    def test_empty(self):
        assert extract_confidence("") == 0.5

    def test_custom_default(self):
        assert extract_confidence("no numbers here", default=0.3) == 0.3

    def test_clamps_to_range(self):
        val = extract_confidence("confidence: 150%")
        assert 0.0 <= val <= 1.0

    def test_lowercase(self):
        assert extract_confidence("CONFIDENCE: 0.9") == pytest.approx(0.9)
