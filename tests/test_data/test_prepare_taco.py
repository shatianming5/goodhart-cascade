"""Tests for TACO data preparation (using mock data, no real download)."""

import json
import os
import tempfile

import pytest

from goodhart.data.prepare_taco import (
    filter_taco,
    format_taco_problem,
    parse_input_output,
    prepare_verl_parquet,
)


class TestParseInputOutput:
    def test_valid_json(self):
        raw = json.dumps({"inputs": ["1", "2", "3"], "outputs": ["2", "4", "6"]})
        cases = parse_input_output(raw)
        assert len(cases) == 3
        assert cases[0] == {"input": "1", "output": "2"}

    def test_empty_string(self):
        assert parse_input_output("") == []

    def test_invalid_json(self):
        assert parse_input_output("not json") == []

    def test_missing_keys(self):
        assert parse_input_output(json.dumps({"inputs": []})) == []

    def test_none_input(self):
        assert parse_input_output(None) == []

    def test_whitespace_stripping(self):
        raw = json.dumps({"inputs": [" 1 \n"], "outputs": [" 2 \n"]})
        cases = parse_input_output(raw)
        assert cases[0] == {"input": "1", "output": "2"}


class TestFormatTacoProblem:
    def test_basic_format(self):
        raw = {
            "task_id": 42,
            "question": "Double the input",
            "input_output": json.dumps(
                {"inputs": ["1", "2"], "outputs": ["2", "4"]}
            ),
            "difficulty": "EASY",
            "starter_code": "",
        }
        prob = format_taco_problem(raw)
        assert prob["id"] == "taco_42"
        assert prob["question"] == "Double the input"
        assert len(prob["test_cases"]) == 2
        assert prob["difficulty"] == "EASY"
        assert prob["source"] == "taco"

    def test_numeric_difficulty(self):
        raw = {
            "question": "test",
            "input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]}),
            "difficulty": 3,
        }
        prob = format_taco_problem(raw)
        assert prob["difficulty"] == "HARD"

    def test_missing_starter_code(self):
        raw = {
            "question": "test",
            "input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]}),
        }
        prob = format_taco_problem(raw)
        assert prob["starter_code"] == ""


class TestFilterTaco:
    def _make_dataset(self, counts):
        """Make a list of mock rows with varying test case counts."""
        rows = []
        for i, n in enumerate(counts):
            inputs = [str(j) for j in range(n)]
            outputs = [str(j * 2) for j in range(n)]
            rows.append(
                {
                    "task_id": i,
                    "question": f"Problem {i}",
                    "input_output": json.dumps(
                        {"inputs": inputs, "outputs": outputs}
                    ),
                    "difficulty": "EASY",
                    "starter_code": "",
                }
            )
        return rows

    def test_filters_by_min_tests(self):
        ds = self._make_dataset([3, 5, 7, 2, 10])
        result = filter_taco(ds, min_tests=5)
        assert len(result) == 3  # 5, 7, 10

    def test_none_pass(self):
        ds = self._make_dataset([1, 2, 3])
        result = filter_taco(ds, min_tests=5)
        assert len(result) == 0


class TestPrepareVerlParquet:
    def test_writes_parquet(self):
        problems = [
            {
                "id": "t1",
                "question": "Double it",
                "test_cases": [{"input": "1", "output": "2"}],
                "difficulty": "EASY",
                "starter_code": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.parquet")
            prepare_verl_parquet(problems, path)
            assert os.path.exists(path)

            import pyarrow.parquet as pq

            table = pq.read_table(path)
            assert table.num_rows == 1
            assert "data_source" in table.column_names
            assert "prompt" in table.column_names
            assert "ground_truth" in table.column_names

            gt = json.loads(table.column("ground_truth")[0].as_py())
            assert gt[0]["input"] == "1"
