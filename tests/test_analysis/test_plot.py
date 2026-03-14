"""Tests for plot generation (verify files are created, no visual check)."""

import os
import tempfile

import pytest

from goodhart.analysis.plot_figures import (
    plot_degradation_main,
    plot_granger_heatmap,
    plot_multi_vs_test_only,
    plot_quality_spider,
    plot_reliability_diagram,
    plot_scale_comparison,
)


class TestPlots:
    def test_degradation_main(self, sample_eval_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig1.png")
            plot_degradation_main(sample_eval_results, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_reliability_diagram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig2.png")
            confs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.8, 0.2]
            outcomes = [0, 0, 1, 1, 1, 0, 1, 0]
            plot_reliability_diagram(confs, outcomes, path)
            assert os.path.exists(path)

    def test_quality_spider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig3.png")
            metrics = {
                "pylint_score": 7.5,
                "cyclomatic_complexity": 3.0,
                "duplication_ratio": 0.1,
                "comment_ratio": 0.2,
                "type_hint_ratio": 0.5,
                "n_samples": 10,
            }
            plot_quality_spider(metrics, path)
            assert os.path.exists(path)

    def test_scale_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig4.png")
            comparison = {
                "7b": {"final_ece": 0.3, "final_quality": 0.5, "final_shortcut_rate": 0.2, "final_pass_rate": 0.8, "n_checkpoints": 5},
                "14b": {"final_ece": 0.2, "final_quality": 0.6, "final_shortcut_rate": 0.1, "final_pass_rate": 0.85, "n_checkpoints": 5},
            }
            plot_scale_comparison(comparison, path)
            assert os.path.exists(path)

    def test_multi_vs_test_only(self, sample_eval_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig5.png")
            plot_multi_vs_test_only(sample_eval_results, sample_eval_results, path)
            assert os.path.exists(path)

    def test_granger_heatmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig6.png")
            granger = {
                "ece->quality": {"p_value": 0.01, "significant": True},
                "quality->ece": {"p_value": 0.3, "significant": False},
            }
            plot_granger_heatmap(granger, path)
            assert os.path.exists(path)
