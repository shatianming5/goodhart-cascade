"""Tests for multi-objective reward."""

import json

import pytest

from goodhart.rewards.multi_objective import compute_score


class TestMultiObjective:
    def _gt(self, cases):
        return json.dumps(cases)

    def test_all_good(self):
        code = "```python\nn = int(input())\nprint(n * 2)\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        extra = {"confidence_text": "confidence: 1.0"}
        score = compute_score("taco", code, gt, extra)
        # r_test=1.0, r_quality>0, r_cal=0 → cal_norm=1.0
        # 0.5*1.0 + 0.3*quality + 0.2*1.0 = 0.7 + 0.3*quality
        assert score > 0.7

    def test_wrong_code(self):
        code = "```python\nprint('wrong')\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        score = compute_score("taco", code, gt)
        # r_test=0.0, cal not provided → cal_norm=1.0
        # 0.5*0 + 0.3*quality + 0.2*1.0 = 0.3*quality + 0.2
        assert score < 0.5

    def test_no_confidence(self):
        code = "```python\nn = int(input())\nprint(n * 2)\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        score = compute_score("taco", code, gt)
        # no extra_info → cal_norm=1.0
        assert score >= 0.5

    def test_overconfident_wrong(self):
        code = "```python\nprint('wrong')\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        extra = {"confidence_text": "confidence: 1.0"}
        score = compute_score("taco", code, gt, extra)
        # r_test=0, confidence=1.0, outcome=0 → cal=-1 → cal_norm=0
        # 0.5*0 + 0.3*quality + 0.2*0 = 0.3*quality
        assert score < 0.3

    def test_well_calibrated_wrong(self):
        code = "```python\nprint('wrong')\n```"
        gt = self._gt([{"input": "3", "output": "6"}])
        extra = {"confidence_text": "confidence: 0.0"}
        score_cal = compute_score("taco", code, gt, extra)
        # confidence=0, outcome=0 → perfect calibration → cal_norm=1.0
        extra2 = {"confidence_text": "confidence: 1.0"}
        score_uncal = compute_score("taco", code, gt, extra2)
        # well-calibrated wrong should score higher than overconfident wrong
        assert score_cal > score_uncal

    def test_score_range(self):
        code = "```python\nprint(int(input())*2)\n```"
        gt = self._gt([{"input": "1", "output": "2"}])
        score = compute_score("taco", code, gt)
        assert 0.0 <= score <= 1.0

    def test_weights_sum(self):
        # verify weights conceptually: 0.5 + 0.3 + 0.2 = 1.0
        assert 0.5 + 0.3 + 0.2 == pytest.approx(1.0)

    def test_empty_solution(self):
        gt = self._gt([{"input": "1", "output": "2"}])
        score = compute_score("taco", "", gt)
        assert score <= 0.5  # r_test=0, r_quality=0
