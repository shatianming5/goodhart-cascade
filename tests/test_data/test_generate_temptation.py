"""Tests for temptation task generation."""

import json
import os
import tempfile

import pytest

from goodhart.data.generate_temptation import (
    generate_all,
    generate_control,
    generate_type_a,
    generate_type_b,
    generate_type_c,
)


class TestGenerateTypeA:
    def test_correct_count(self):
        tasks = generate_type_a(n=10)
        assert len(tasks) == 10

    def test_has_required_fields(self):
        tasks = generate_type_a(n=1)
        t = tasks[0]
        assert t["type"] == "type_a"
        assert "spec_tests" in t
        assert "corrupted_tests" in t
        assert len(t["spec_tests"]) >= 3
        assert len(t["corrupted_tests"]) >= 3

    def test_spec_and_corrupted_differ(self):
        tasks = generate_type_a(n=5)
        for t in tasks:
            for s, c in zip(t["spec_tests"], t["corrupted_tests"]):
                assert s["input"] == c["input"]
                assert s["output"] != c["output"]  # must differ

    def test_deterministic(self):
        t1 = generate_type_a(n=5, seed=42)
        t2 = generate_type_a(n=5, seed=42)
        assert t1 == t2


class TestGenerateTypeB:
    def test_correct_count(self):
        tasks = generate_type_b(n=10)
        assert len(tasks) == 10

    def test_has_visible_and_hidden(self):
        tasks = generate_type_b(n=1)
        t = tasks[0]
        assert "visible_tests" in t
        assert "hidden_tests" in t
        assert len(t["hidden_tests"]) > 0

    def test_hidden_inputs_differ_from_visible(self):
        tasks = generate_type_b(n=5)
        for t in tasks:
            visible_inputs = {tc["input"] for tc in t["visible_tests"]}
            hidden_inputs = {tc["input"] for tc in t["hidden_tests"]}
            assert not visible_inputs.intersection(hidden_inputs)


class TestGenerateTypeC:
    def test_correct_count(self):
        tasks = generate_type_c(n=10)
        assert len(tasks) == 10

    def test_has_spec_tests(self):
        tasks = generate_type_c(n=1)
        assert "spec_tests" in tasks[0]
        assert len(tasks[0]["spec_tests"]) >= 1

    def test_prompt_contains_buggy_code(self):
        tasks = generate_type_c(n=1)
        assert "bug" in tasks[0]["prompt"].lower()


class TestGenerateControl:
    def test_correct_count(self):
        tasks = generate_control(n=10)
        assert len(tasks) == 10

    def test_type_is_control(self):
        tasks = generate_control(n=1)
        assert tasks[0]["type"] == "control"


class TestGenerateAll:
    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tasks.json")
            tasks = generate_all(path, n_per_type=5)

            assert os.path.exists(path)
            assert len(tasks) == 20  # 5 * 4 types

            with open(path) as f:
                loaded = json.load(f)
            assert len(loaded) == 20

    def test_all_types_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tasks.json")
            tasks = generate_all(path, n_per_type=3)

            types = {t["type"] for t in tasks}
            assert types == {"type_a", "type_b", "type_c", "control"}
