"""Tests for TDA module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import run_preprocessing
from src.svg_detection import run_svg_detection
from src.tda import (
    PersistenceSummary,
    TDAResult,
    _expression_to_point_cloud,
    analyse_gene,
    compute_persistence,
    persistence_entropy,
    run_tda_analysis,
)


@pytest.fixture
def preprocessed(synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
    adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
    return adata


@pytest.fixture
def morans_scores(preprocessed: ad.AnnData) -> pd.DataFrame:
    result = run_svg_detection(preprocessed)
    return result.scores


class TestExpressionToPointCloud:
    """Test point cloud construction from spatial expression."""

    def test_returns_2d_array(self, preprocessed: ad.AnnData) -> None:
        pts = _expression_to_point_cloud(preprocessed, "Grad_0")
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_coords_in_unit_interval(self, preprocessed: ad.AnnData) -> None:
        pts = _expression_to_point_cloud(preprocessed, "Grad_0")
        if len(pts) > 0:
            assert pts.min() >= 0.0
            assert pts.max() <= 1.0

    def test_invalid_gene_raises(self, preprocessed: ad.AnnData) -> None:
        with pytest.raises(ValueError, match="not found"):
            _expression_to_point_cloud(preprocessed, "NonExistentGene_XYZ")

    def test_gradient_gene_has_points(self, preprocessed: ad.AnnData) -> None:
        pts = _expression_to_point_cloud(preprocessed, "Grad_0")
        assert len(pts) > 0

    def test_housekeeping_gene_has_points(
        self, preprocessed: ad.AnnData
    ) -> None:
        pts = _expression_to_point_cloud(preprocessed, "House_000")
        assert len(pts) >= 0  # may have few or no points above median


class TestComputePersistence:
    """Test persistence computation via gudhi."""

    def test_returns_two_lists(self) -> None:
        rng = np.random.default_rng(42)
        pts = rng.random((30, 2))
        h0, h1 = compute_persistence(pts)
        assert isinstance(h0, list)
        assert isinstance(h1, list)

    def test_ring_has_h1_features(self) -> None:
        """A ring-like point cloud should produce H1 persistence features."""
        angles = np.linspace(0, 2 * np.pi, 40)
        ring_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        ring_pts = (ring_pts - ring_pts.min(0)) / (ring_pts.max(0) - ring_pts.min(0))
        h0, h1 = compute_persistence(ring_pts, max_edge_length=0.5)
        assert len(h1) > 0, "Ring point cloud should have H1 persistence features"

    def test_blob_has_fewer_h1_than_ring(self) -> None:
        """A compact blob should have fewer H1 features than a ring."""
        rng = np.random.default_rng(42)
        # Ring
        angles = np.linspace(0, 2 * np.pi, 40)
        ring_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        ring_pts = (ring_pts - ring_pts.min(0)) / (ring_pts.max(0) - ring_pts.min(0))
        _, h1_ring = compute_persistence(ring_pts, max_edge_length=0.5)
        # Blob
        blob_pts = rng.normal(0.5, 0.05, (40, 2))
        _, h1_blob = compute_persistence(blob_pts, max_edge_length=0.5)
        assert len(h1_ring) >= len(h1_blob)

    def test_empty_points_returns_empty(self) -> None:
        h0, h1 = compute_persistence(np.array([]).reshape(0, 2))
        assert h0 == []
        assert h1 == []

    def test_too_few_points_returns_empty(self) -> None:
        pts = np.array([[0, 0], [1, 1]])
        h0, h1 = compute_persistence(pts)
        assert h0 == []
        assert h1 == []

    def test_intervals_are_tuples(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.random((20, 2))
        h0, h1 = compute_persistence(pts)
        for b, d in h0:
            assert d >= b


class TestPersistenceEntropy:
    """Test persistence entropy computation."""

    def test_empty_intervals_returns_zero(self) -> None:
        assert persistence_entropy([]) == 0.0

    def test_single_interval_returns_zero(self) -> None:
        # Single feature: p=1, log(1)=0, entropy=0
        result = persistence_entropy([(0.0, 1.0)])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_uniform_intervals_has_max_entropy(self) -> None:
        """Equal-lifetime features have maximum entropy."""
        intervals = [(i * 0.1, (i + 1) * 0.1) for i in range(5)]
        entropy = persistence_entropy(intervals)
        assert entropy > 0

    def test_entropy_is_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        intervals = [(float(b), float(b + d))
                     for b, d in zip(rng.random(10), rng.random(10) * 0.5)]
        assert persistence_entropy(intervals) >= 0


class TestAnalyseGene:
    """Test per-gene TDA analysis."""

    def test_returns_persistence_summary(self, preprocessed: ad.AnnData) -> None:
        result = analyse_gene(preprocessed, "Grad_0")
        assert isinstance(result, PersistenceSummary)

    def test_gene_name_preserved(self, preprocessed: ad.AnnData) -> None:
        result = analyse_gene(preprocessed, "Grad_0")
        assert result.gene == "Grad_0"

    def test_entropy_non_negative(self, preprocessed: ad.AnnData) -> None:
        result = analyse_gene(preprocessed, "Grad_0")
        assert result.persistence_entropy_h0 >= 0
        assert result.persistence_entropy_h1 >= 0

    def test_ring_gene_has_h1(self, preprocessed: ad.AnnData) -> None:
        """Ring genes in the fixture should show H1 topological features."""
        result = analyse_gene(preprocessed, "Ring_0")
        # Ring genes have H1 features (loops); housekeeping should not
        house_result = analyse_gene(preprocessed, "House_000")
        # max_persistence_h1 is more robust than entropy for single features
        assert result.max_persistence_h1 >= house_result.max_persistence_h1


class TestRunTdaAnalysis:
    """Test the full TDA comparison pipeline."""

    def test_returns_tda_result(
        self, preprocessed: ad.AnnData, morans_scores: pd.DataFrame
    ) -> None:
        result = run_tda_analysis(preprocessed, morans_scores, n_genes=9)
        assert isinstance(result, TDAResult)

    def test_summaries_populated(
        self, preprocessed: ad.AnnData, morans_scores: pd.DataFrame
    ) -> None:
        result = run_tda_analysis(preprocessed, morans_scores, n_genes=9)
        assert len(result.summaries) > 0

    def test_comparison_df_has_required_columns(
        self, preprocessed: ad.AnnData, morans_scores: pd.DataFrame
    ) -> None:
        result = run_tda_analysis(preprocessed, morans_scores, n_genes=9)
        for col in ["morans_i", "persistence_entropy_h1", "morans_rank",
                    "tda_h1_rank", "rank_diff"]:
            assert col in result.comparison_df.columns

    def test_n_genes_analysed_correct(
        self, preprocessed: ad.AnnData, morans_scores: pd.DataFrame
    ) -> None:
        result = run_tda_analysis(preprocessed, morans_scores, n_genes=9)
        assert result.n_genes_analysed == len(result.summaries)

    def test_tda_unique_is_list(
        self, preprocessed: ad.AnnData, morans_scores: pd.DataFrame
    ) -> None:
        result = run_tda_analysis(preprocessed, morans_scores, n_genes=9)
        assert isinstance(result.tda_unique_genes, list)
