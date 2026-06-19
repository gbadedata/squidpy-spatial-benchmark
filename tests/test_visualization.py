"""Tests for the visualization module.

These tests verify that each plotting function runs without error and
produces a PNG file. They use the synthetic fixture and a temporary
figures directory, so no network or real data is required. Rendering
correctness is validated by visual inspection of the real-data figures.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd
import pytest

from src import visualization as viz
from src.neighbourhood import compute_neighbourhood_enrichment
from src.preprocessing import run_preprocessing
from src.svg_detection import run_svg_detection
from src.tda import run_tda_analysis


@pytest.fixture
def preprocessed(synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
    adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
    return adata


@pytest.fixture(autouse=True)
def tmp_figures(monkeypatch, tmp_path):
    """Redirect figure output to a temp directory for all tests."""
    monkeypatch.setattr(viz.settings, "evidence_dir", tmp_path)
    return tmp_path


class TestPlotSpatialClusters:

    def test_creates_png(self, preprocessed: ad.AnnData) -> None:
        path = viz.plot_spatial_clusters(preprocessed)
        assert path.exists()
        assert path.suffix == ".png"

    def test_file_non_empty(self, preprocessed: ad.AnnData) -> None:
        path = viz.plot_spatial_clusters(preprocessed)
        assert path.stat().st_size > 0


class TestPlotSvgExpression:

    def test_creates_png(self, preprocessed: ad.AnnData) -> None:
        genes = ["Grad_0", "Ring_0", "Mbp", "Cluster_0"]
        path = viz.plot_svg_expression(preprocessed, genes)
        assert path.exists()

    def test_handles_missing_genes(self, preprocessed: ad.AnnData) -> None:
        """Genes not in the dataset are silently skipped."""
        genes = ["Grad_0", "NONEXISTENT_GENE"]
        path = viz.plot_svg_expression(preprocessed, genes)
        assert path.exists()


class TestPlotNeighbourhoodMatrix:

    def test_creates_png(self, preprocessed: ad.AnnData) -> None:
        enr = compute_neighbourhood_enrichment(preprocessed)
        path = viz.plot_neighbourhood_matrix(enr)
        assert path.exists()


class TestPlotPersistenceDiagram:

    def test_creates_png(self, preprocessed: ad.AnnData) -> None:
        path = viz.plot_persistence_diagram(preprocessed, "Ring_0")
        assert path.exists()

    def test_handles_low_expression_gene(self, preprocessed: ad.AnnData) -> None:
        """A gene with few expressing spots should not crash the plot."""
        path = viz.plot_persistence_diagram(preprocessed, "House_000")
        assert path.exists()


class TestPlotMoransVsTda:

    def test_creates_png(self) -> None:
        df = pd.DataFrame(
            {
                "morans_rank": [1.0, 2.0, 3.0, 4.0],
                "tda_h1_rank": [3.0, 1.0, 4.0, 2.0],
            },
            index=["GeneA", "GeneB", "GeneC", "GeneD"],
        )
        path = viz.plot_morans_vs_tda(df)
        assert path.exists()


class TestGenerateAllFigures:

    def test_generates_five_figures(self, preprocessed: ad.AnnData) -> None:
        enr = compute_neighbourhood_enrichment(preprocessed)
        svg = run_svg_detection(preprocessed)
        tda = run_tda_analysis(preprocessed, svg.scores, n_genes=9)
        paths = viz.generate_all_figures(
            preprocessed, enr, tda.comparison_df,
            top_svgs=svg.top_svgs[:4],
            tda_gene="Ring_0",
        )
        assert len(paths) == 5
        for p in paths:
            assert p.exists()
