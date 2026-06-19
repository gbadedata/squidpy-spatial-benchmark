"""End-to-end spatial transcriptomics benchmark pipeline.

Runs all phases in sequence and produces a unified benchmark report:

  1. Load Visium mouse brain dataset
  2. Preprocess: filter, normalise, embed, build spatial graph
  3. Task 1: SVG detection (Moran's I) vs oracle markers
  4. Task 2: neighbourhood enrichment vs neuroanatomy
  5. Task 3: TDA (gudhi persistent homology) vs Moran's I
  6. Combine into a JSON benchmark report

Run with:  python3 -m src.pipeline

For memory efficiency on large datasets, SVG detection and TDA operate
on the highly variable gene subset.
"""

from __future__ import annotations

import gc
import logging
import sys
import time

import structlog

from config.settings import settings
from src.benchmark.evaluator import build_report, write_report
from src.data_loader import get_dataset
from src.neighbourhood import run_neighbourhood_analysis
from src.preprocessing import run_preprocessing
from src.svg_detection import run_svg_detection
from src.tda import run_tda_analysis

# Structured logging: JSON events to stdout
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger()


def run_pipeline(data_path: str | None = None) -> dict:
    """Run the full benchmark pipeline.

    Args:
        data_path: Optional path to a local h5ad. If None, downloads the
            Visium H&E mouse brain dataset.

    Returns:
        The benchmark report as a dict.
    """
    t0 = time.time()
    log.info("pipeline_started", dataset=settings.dataset_name)

    # ── Phase 1: Load ──────────────────────────────────────────────────
    adata = get_dataset(filepath=data_path)
    log.info("phase1_complete", n_spots=adata.n_obs, n_genes=adata.n_vars)

    # ── Phase 2: Preprocess ────────────────────────────────────────────
    adata, pp_report = run_preprocessing(adata)
    log.info(
        "phase2_complete",
        n_genes=pp_report.n_genes,
        n_hvg=pp_report.n_hvg,
        n_spatial_connections=pp_report.n_spatial_connections,
    )

    # ── Phase 4 (Task 2): Neighbourhood enrichment ─────────────────────
    # Run before subsetting genes (uses cluster labels, not gene subset)
    nhood_result = run_neighbourhood_analysis(adata)
    log.info(
        "task2_complete",
        n_pairs_validated=nhood_result.n_pairs_validated,
        n_pairs_passing=nhood_result.n_pairs_passing,
    )

    # Subset to HVGs for gene-level analysis (memory efficiency)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    gc.collect()

    # ── Phase 3 (Task 1): SVG detection ────────────────────────────────
    svg_result = run_svg_detection(adata_hvg)
    log.info(
        "task1_complete",
        n_svgs=svg_result.n_svgs,
        n_markers_found=len(svg_result.marker_genes_found),
        n_markers_tested=len(svg_result.marker_genes_tested),
    )

    # ── Phase 5 (Task 3): TDA vs Moran's I ─────────────────────────────
    tda_result = run_tda_analysis(adata_hvg, svg_result.scores)
    log.info(
        "task3_complete",
        n_genes_analysed=tda_result.n_genes_analysed,
        n_tda_unique=len(tda_result.tda_unique_genes),
    )

    # ── Phase 6: Build and write report ────────────────────────────────
    report = build_report(
        dataset=settings.dataset_name,
        n_spots=pp_report.n_spots,
        n_genes=pp_report.n_genes,
        n_clusters=pp_report.n_clusters,
        svg_result=svg_result,
        nhood_result=nhood_result,
        tda_result=tda_result,
    )

    report_path = settings.evidence_dir / "reports" / "benchmark_report.json"
    write_report(report, report_path)

    runtime = round(time.time() - t0, 1)
    log.info(
        "pipeline_complete",
        runtime_seconds=runtime,
        tasks_passed=report.n_tasks_passed,
        tasks_total=report.n_tasks_total,
        report=str(report_path),
    )

    _print_summary(report, runtime)
    return report.to_dict()


def _print_summary(report, runtime: float) -> None:
    """Print a human-readable benchmark summary to stdout."""
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY -- Spatial Transcriptomics & TDA")
    print("=" * 65)
    print(f"  Dataset:   {report.dataset}")
    print(f"  Spots:     {report.n_spots}")
    print(f"  Genes:     {report.n_genes}")
    print(f"  Clusters:  {report.n_clusters}")
    print()

    for task in report.tasks:
        verdict = "PASS" if task.passed else "FAIL"
        print(f"  [{verdict}] {task.name}")
        print(f"         score={task.score}  threshold={task.threshold}")

    print()
    print(f"  Tasks passed: {report.n_tasks_passed}/{report.n_tasks_total}")
    print(f"  Runtime:      {runtime}s")
    print("=" * 65)


def main() -> None:
    """CLI entry point."""
    data_path = None
    if len(sys.argv) > 2 and sys.argv[1] == "--data":
        data_path = sys.argv[2]
    run_pipeline(data_path=data_path)


if __name__ == "__main__":
    main()
