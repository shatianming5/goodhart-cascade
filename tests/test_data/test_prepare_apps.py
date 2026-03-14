"""Tests for APPS data preparation (using mock data, no real download)."""

import json

import pytest

from goodhart.data.prepare_apps import (
    filter_apps,
    format_apps_problem,
    parse_apps_tests,
)


class TestParseAppsTests:
    def test_valid_json(self):
        raw = {
            "input_output": json.dumps(
                {"inputs": ["1\n", "2\n"], "outputs": ["2\n", "4\n"]}
            )
        }
        cases = parse_apps_tests(raw)
        assert len(cases) == 2
        assert cases[0]["input"] == "1"
        assert cases[0]["output"] == "2"

    def test_empty(self):
        assert parse_apps_tests({}) == []
        assert parse_apps_tests({"input_output": ""}) == []

    def test_invalid_json(self):
        assert parse_apps_tests({"input_output": "bad"}) == []


class TestFormatAppsProblem:
    def test_basic(self):
        raw = {
            "problem_id": 123,
            "question": "Sum two numbers",
            "input_output": json.dumps(
                {"inputs": ["1 2\n"], "outputs": ["3\n"]}
            ),
            "difficulty": "interview",
            "starter_code": "",
        }
        prob = format_apps_problem(raw)
        assert prob["id"] == "apps_123"
        assert prob["difficulty"] == "MEDIUM"
        assert prob["source"] == "apps"
        assert len(prob["test_cases"]) == 1

    def test_difficulty_mapping(self):
        for apps_diff, expected in [
            ("introductory", "EASY"),
            ("interview", "MEDIUM"),
            ("competition", "HARD"),
            ("unknown_val", "UNKNOWN"),
        ]:
            raw = {
                "question": "x",
                "difficulty": apps_diff,
                "input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]}),
            }
            assert format_apps_problem(raw)["difficulty"] == expected


class TestFilterApps:
    def _make_rows(self, configs):
        rows = []
        for i, (n_tests, diff) in enumerate(configs):
            inputs = [str(j) for j in range(n_tests)]
            outputs = [str(j) for j in range(n_tests)]
            rows.append(
                {
                    "problem_id": i,
                    "question": f"P{i}",
                    "input_output": json.dumps(
                        {"inputs": inputs, "outputs": outputs}
                    ),
                    "difficulty": diff,
                    "starter_code": "",
                }
            )
        return rows

    def test_min_tests_filter(self):
        ds = self._make_rows([(2, "interview"), (5, "interview"), (3, "interview")])
        result = filter_apps(ds, min_tests=3)
        assert len(result) == 2

    def test_difficulty_filter(self):
        ds = self._make_rows([(5, "introductory"), (5, "competition")])
        result = filter_apps(ds, difficulties=["EASY"], min_tests=1)
        assert len(result) == 1
        assert result[0]["difficulty"] == "EASY"

    def test_combined_filters(self):
        ds = self._make_rows([(1, "interview"), (5, "introductory"), (5, "competition")])
        result = filter_apps(ds, difficulties=["EASY"], min_tests=3)
        assert len(result) == 1
