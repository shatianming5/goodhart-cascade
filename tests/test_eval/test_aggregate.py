"""Tests for result aggregation."""

import json
import os
import tempfile

import pytest

from goodhart.eval.aggregate import aggregate_checkpoint, merge_all_checkpoints, _extract_step


class TestAggregateCheckpoint:
    def test_loads_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = {"ece": 0.1, "pass_rate": 0.8}
            qual = {"pylint_score": 7.5}
            temp = {"overall_shortcut_rate": 0.05}

            for name, data in [("calibration", cal), ("code_quality", qual), ("temptation", temp)]:
                with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                    json.dump(data, f)

            result = aggregate_checkpoint(
                os.path.join(tmpdir, "calibration.json"),
                os.path.join(tmpdir, "code_quality.json"),
                os.path.join(tmpdir, "temptation.json"),
            )
            assert result["calibration"]["ece"] == 0.1
            assert result["code_quality"]["pylint_score"] == 7.5
            assert result["temptation"]["overall_shortcut_rate"] == 0.05

    def test_missing_file(self):
        result = aggregate_checkpoint("/nonexistent/a.json", "/nonexistent/b.json", "/nonexistent/c.json")
        assert result["calibration"] == {}


class TestMergeAllCheckpoints:
    def test_merges_multiple_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [0, 100, 200]:
                step_dir = os.path.join(tmpdir, f"step_{step}")
                os.makedirs(step_dir)
                for name in ["calibration", "code_quality", "temptation"]:
                    with open(os.path.join(step_dir, f"{name}.json"), "w") as f:
                        json.dump({"step": step, "value": step * 0.1}, f)

            results = merge_all_checkpoints(tmpdir)
            assert len(results) == 3
            assert results[0]["step"] == 0
            assert results[2]["step"] == 200

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert merge_all_checkpoints(tmpdir) == []

    def test_nonexistent_dir(self):
        assert merge_all_checkpoints("/nonexistent/path") == []


class TestExtractStep:
    def test_step_format(self):
        assert _extract_step("step_100") == 100
        assert _extract_step("step_0") == 0

    def test_global_step(self):
        assert _extract_step("global_step_200") == 200

    def test_pure_number(self):
        assert _extract_step("300") == 300

    def test_invalid(self):
        assert _extract_step("some_dir") is None
