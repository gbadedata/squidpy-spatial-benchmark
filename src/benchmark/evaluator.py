"""Unified benchmark evaluator.

Combines the three benchmark tasks into a single structured report:

Task 1 -- SVG detection: how many Allen Brain Atlas oracle markers does
    Moran's I recover among its top-ranked spatially variable genes?
    Scored by sensitivity (recall of known markers).

Task 2 -- Neighbourhood enrichment: do compact, anatomically co-localised
    brain structures show statistically significant spatial enrichment
    consistent with known neuroanatomy? Scored by fraction of validated
    anatomical pairs passing.

Task 3 -- TDA vs Moran's I: do persistent homology features (H1 entropy)
    and spatial autocorrelation (Moran's I) produce different gene
    rankings, identifying genes whose topologically complex spatial
    patterns autocorrelation underestimates? Scored by rank divergence.

Each task has an explicit pass criterion and the report records both the
quantitative result and a pass/fail verdict, mirroring the evaluation
design used across the portfolio (oracle, validators, documented limits).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single benchmark task.

    Attributes:
        name: Task identifier.
        passed: Whether the task met its pass criterion.
        score: Primary scalar score for the task.
        threshold: Pass threshold for the score.
        details: Task-specific supporting metrics.
    """

    name: str
    passed: bool
    score: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all three tasks.

    Attributes:
        dataset: Dataset name.
        n_spots: Number of spatial spots.
        n_genes: Number of genes after filtering.
        n_clusters: Number of anatomical clusters.
        tasks: List of TaskResult, one per benchmark task.
        n_tasks_passed: Count of tasks passing their criterion.
        n_tasks_total: Total number of tasks.
    """

    dataset: str
    n_spots: int
    n_genes: int
    n_clusters: int
    tasks: list[TaskResult]
    n_tasks_passed: int
    n_tasks_total: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "dataset": self.dataset,
            "n_spots": self.n_spots,
            "n_genes": self.n_genes,
            "n_clusters": self.n_clusters,
            "n_tasks_passed": self.n_tasks_passed,
            "n_tasks_total": self.n_tasks_total,
            "tasks": [asdict(t) for t in self.tasks],
        }


def evaluate_svg_task(svg_result: Any) -> TaskResult:
    """Score Task 1: SVG detection against oracle markers.

    Sensitivity = fraction of known Allen Brain Atlas markers that appear
    in the top-ranked SVGs by Moran's I.

    Args:
        svg_result: SVGResult from run_svg_detection.

    Returns:
        TaskResult for the SVG detection task.
    """
    from config.settings import settings

    n_markers_tested = len(svg_result.marker_genes_tested)
    n_markers_found = len(svg_result.marker_genes_found)
    sensitivity = n_markers_found / n_markers_tested if n_markers_tested else 0.0

    passed = sensitivity >= settings.min_svg_sensitivity

    return TaskResult(
        name="svg_detection",
        passed=passed,
        score=round(sensitivity, 4),
        threshold=settings.min_svg_sensitivity,
        details={
            "n_genes_tested": svg_result.n_genes_tested,
            "n_svgs_called": svg_result.n_svgs,
            "n_oracle_markers_tested": n_markers_tested,
            "n_oracle_markers_found": n_markers_found,
            "markers_found": svg_result.marker_genes_found,
            "top_svg": svg_result.top_svgs[0] if svg_result.top_svgs else None,
        },
    )


def evaluate_neighbourhood_task(nhood_result: Any) -> TaskResult:
    """Score Task 2: neighbourhood enrichment against anatomy.

    Accuracy = fraction of validated anatomical pairs that show the
    expected spatial enrichment.

    Args:
        nhood_result: NeighbourhoodResult from run_neighbourhood_analysis.

    Returns:
        TaskResult for the neighbourhood enrichment task.
    """
    from config.settings import settings

    n_validated = nhood_result.n_pairs_validated
    n_passing = nhood_result.n_pairs_passing
    accuracy = n_passing / n_validated if n_validated else 0.0

    passed = accuracy >= settings.min_enrichment_accuracy

    return TaskResult(
        name="neighbourhood_enrichment",
        passed=passed,
        score=round(accuracy, 4),
        threshold=settings.min_enrichment_accuracy,
        details={
            "n_clusters": nhood_result.n_clusters,
            "n_pairs_validated": n_validated,
            "n_pairs_passing": n_passing,
            "validated_pairs": {
                f"{a}+{b}": passed
                for (a, b), passed in nhood_result.validated_pairs.items()
            },
        },
    )


def evaluate_tda_task(tda_result: Any) -> TaskResult:
    """Score Task 3: TDA vs Moran's I rank divergence.

    This task does not have a simple pass/fail in the same sense as the
    others -- its purpose is to demonstrate that persistent homology and
    spatial autocorrelation measure different structure. The score is the
    maximum rank divergence observed (how strongly TDA promotes a gene
    that Moran's I ranks low). A divergence above the threshold confirms
    the two methods produce genuinely different rankings.

    Args:
        tda_result: TDAResult from run_tda_analysis.

    Returns:
        TaskResult for the TDA comparison task.
    """
    df = tda_result.comparison_df
    max_divergence = float(df["rank_diff"].max()) if len(df) else 0.0
    n_genes = tda_result.n_genes_analysed
    # Pass if at least one gene's TDA rank exceeds its Moran rank by
    # more than 25% of the gene set size -- evidence of genuine divergence
    threshold = n_genes * 0.25
    passed = max_divergence >= threshold

    # Identify the gene with the largest divergence
    top_divergent = None
    if len(df):
        top_divergent = str(df["rank_diff"].idxmax())

    return TaskResult(
        name="tda_vs_morans",
        passed=passed,
        score=round(max_divergence, 2),
        threshold=round(threshold, 2),
        details={
            "n_genes_analysed": n_genes,
            "n_tda_unique_genes": len(tda_result.tda_unique_genes),
            "tda_unique_genes": tda_result.tda_unique_genes,
            "most_divergent_gene": top_divergent,
        },
    )


def build_report(
    dataset: str,
    n_spots: int,
    n_genes: int,
    n_clusters: int,
    svg_result: Any,
    nhood_result: Any,
    tda_result: Any,
) -> BenchmarkReport:
    """Combine all three task results into a unified benchmark report.

    Args:
        dataset: Dataset name.
        n_spots: Number of spatial spots.
        n_genes: Number of genes after filtering.
        n_clusters: Number of anatomical clusters.
        svg_result: SVGResult from Task 1.
        nhood_result: NeighbourhoodResult from Task 2.
        tda_result: TDAResult from Task 3.

    Returns:
        Complete BenchmarkReport.
    """
    tasks = [
        evaluate_svg_task(svg_result),
        evaluate_neighbourhood_task(nhood_result),
        evaluate_tda_task(tda_result),
    ]
    n_passed = sum(1 for t in tasks if t.passed)

    report = BenchmarkReport(
        dataset=dataset,
        n_spots=n_spots,
        n_genes=n_genes,
        n_clusters=n_clusters,
        tasks=tasks,
        n_tasks_passed=n_passed,
        n_tasks_total=len(tasks),
    )

    logger.info(
        "benchmark_report_built: %d/%d tasks passed",
        n_passed, len(tasks),
    )
    return report


def write_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write the benchmark report to a JSON file.

    Args:
        report: BenchmarkReport to serialise.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("benchmark_report_written: %s", path)
    return path
