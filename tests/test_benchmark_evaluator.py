"""Tests for the benchmark evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from src.benchmark.evaluator import (
    BenchmarkReport,
    TaskResult,
    build_report,
    evaluate_neighbourhood_task,
    evaluate_svg_task,
    evaluate_tda_task,
    write_report,
)

# ── Lightweight stand-ins for the result dataclasses ──────────────────


@dataclass
class FakeSVG:
    n_genes_tested: int = 16957
    n_svgs: int = 3315
    top_svgs: list = field(default_factory=lambda: ["Mbp", "Mog", "Plp1"])
    marker_genes_found: list = field(default_factory=lambda: ["Mbp", "Mog", "Plp1"])
    marker_genes_tested: list = field(
        default_factory=lambda: ["Mbp", "Mog", "Plp1", "Rorb", "Cux2",
                                 "Reln", "Prox1", "Ntsr1", "Bcl11b", "Scnn1a"]
    )


@dataclass
class FakeNhood:
    n_clusters: int = 15
    n_pairs_validated: int = 5
    n_pairs_passing: int = 4
    validated_pairs: dict = field(
        default_factory=lambda: {
            ("Hippocampus", "Pyramidal_layer"): True,
            ("Thalamus_1", "Thalamus_2"): True,
            ("Hypothalamus_1", "Hypothalamus_2"): True,
            ("Fiber_tract", "Cortex_5"): True,
            ("Pyramidal_layer", "Pyramidal_layer_dentate_gyrus"): False,
        }
    )


@dataclass
class FakeTDA:
    n_genes_analysed: int = 15
    tda_unique_genes: list = field(default_factory=lambda: ["Apbb2"])
    comparison_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "morans_i": [0.79, 0.087, 0.087],
                "persistence_entropy_h1": [3.0, 3.34, 1.44],
                "morans_rank": [1.0, 8.0, 9.0],
                "tda_h1_rank": [7.0, 1.0, 14.0],
                "rank_diff": [-6.0, 7.0, -5.0],
            },
            index=["Mbp", "Apbb2", "Grem1"],
        )
    )


class TestEvaluateSvgTask:

    def test_returns_task_result(self) -> None:
        result = evaluate_svg_task(FakeSVG())
        assert isinstance(result, TaskResult)

    def test_name_correct(self) -> None:
        assert evaluate_svg_task(FakeSVG()).name == "svg_detection"

    def test_sensitivity_computed(self) -> None:
        # 3 found / 10 tested = 0.3
        result = evaluate_svg_task(FakeSVG())
        assert result.score == pytest.approx(0.3, abs=0.001)

    def test_passes_at_threshold(self) -> None:
        # 0.3 >= 0.3 default threshold
        result = evaluate_svg_task(FakeSVG())
        assert result.passed

    def test_details_include_markers(self) -> None:
        result = evaluate_svg_task(FakeSVG())
        assert "markers_found" in result.details
        assert result.details["n_oracle_markers_found"] == 3


class TestEvaluateNeighbourhoodTask:

    def test_returns_task_result(self) -> None:
        result = evaluate_neighbourhood_task(FakeNhood())
        assert isinstance(result, TaskResult)

    def test_accuracy_computed(self) -> None:
        # 4 passing / 5 validated = 0.8
        result = evaluate_neighbourhood_task(FakeNhood())
        assert result.score == pytest.approx(0.8, abs=0.001)

    def test_passes_above_threshold(self) -> None:
        # 0.8 >= 0.6 default threshold
        result = evaluate_neighbourhood_task(FakeNhood())
        assert result.passed

    def test_validated_pairs_serialised(self) -> None:
        result = evaluate_neighbourhood_task(FakeNhood())
        assert "validated_pairs" in result.details
        # Keys should be joined strings
        for key in result.details["validated_pairs"]:
            assert "+" in key


class TestEvaluateTdaTask:

    def test_returns_task_result(self) -> None:
        result = evaluate_tda_task(FakeTDA())
        assert isinstance(result, TaskResult)

    def test_max_divergence_score(self) -> None:
        # max rank_diff = 7.0
        result = evaluate_tda_task(FakeTDA())
        assert result.score == pytest.approx(7.0, abs=0.001)

    def test_passes_with_divergence(self) -> None:
        # 7.0 >= 15 * 0.25 = 3.75
        result = evaluate_tda_task(FakeTDA())
        assert result.passed

    def test_most_divergent_gene_identified(self) -> None:
        result = evaluate_tda_task(FakeTDA())
        assert result.details["most_divergent_gene"] == "Apbb2"


class TestBuildReport:

    def test_returns_benchmark_report(self) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        assert isinstance(report, BenchmarkReport)

    def test_three_tasks(self) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        assert report.n_tasks_total == 3
        assert len(report.tasks) == 3

    def test_counts_passed(self) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        # All three should pass with these fixtures
        assert report.n_tasks_passed == 3

    def test_to_dict_serialisable(self) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        d = report.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)
        assert d["n_spots"] == 2688
        assert len(d["tasks"]) == 3


class TestWriteReport:

    def test_writes_json_file(self, tmp_path: Path) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        out = tmp_path / "report.json"
        result_path = write_report(report, out)
        assert result_path.exists()

    def test_written_json_valid(self, tmp_path: Path) -> None:
        report = build_report(
            "Visium", 2688, 16957, 15,
            FakeSVG(), FakeNhood(), FakeTDA(),
        )
        out = tmp_path / "report.json"
        write_report(report, out)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["n_tasks_total"] == 3
        assert loaded["dataset"] == "Visium"
