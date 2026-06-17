"""Tests for SVG detection module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import run_preprocessing
from src.svg_detection import (
    SVGResult,
    call_svgs,
    compute_morans_i,
    get_oracle_markers,
    run_svg_detection,
)


@pytest.fixture
def preprocessed(synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData after full preprocessing."""
    adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
    return adata


class TestComputeMoransI:
    """Test Moran's I computation."""

    def test_returns_dataframe(self, preprocessed: ad.AnnData) -> None:
        result = compute_morans_i(preprocessed)
        assert isinstance(result, pd.DataFrame)

    def test_has_morans_i_column(self, preprocessed: ad.AnnData) -> None:
        result = compute_morans_i(preprocessed)
        assert "morans_i" in result.columns

    def test_one_row_per_gene(self, preprocessed: ad.AnnData) -> None:
        result = compute_morans_i(preprocessed)
        assert len(result) == preprocessed.n_vars

    def test_scores_in_valid_range(self, preprocessed: ad.AnnData) -> None:
        """Moran's I should be in [-1, 1]."""
        result = compute_morans_i(preprocessed)
        assert result["morans_i"].between(-1, 1).all()

    def test_sorted_descending(self, preprocessed: ad.AnnData) -> None:
        result = compute_morans_i(preprocessed)
        scores = result["morans_i"].values
        assert np.all(scores[:-1] >= scores[1:])

    def test_missing_graph_raises(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Should raise if spatial graph not computed."""
        from src.preprocessing import filter_and_normalise
        adata = filter_and_normalise(synthetic_spatial_adata.copy())
        with pytest.raises(ValueError, match="spatial_connectivities"):
            compute_morans_i(adata)

    def test_missing_layer_raises(self, preprocessed: ad.AnnData) -> None:
        """Should raise if log_norm layer missing."""
        adata = preprocessed.copy()
        del adata.layers["log_norm"]
        with pytest.raises(ValueError, match="log_norm"):
            compute_morans_i(adata)

    def test_gradient_genes_have_high_morans_i(
        self, preprocessed: ad.AnnData
    ) -> None:
        """Gradient genes (Grad_0..4) should score higher than housekeeping."""
        result = compute_morans_i(preprocessed)
        grad_genes = [g for g in result.index if g.startswith("Grad_")]
        house_genes = [g for g in result.index if g.startswith("House_")]
        if grad_genes and house_genes:
            grad_mean = result.loc[grad_genes, "morans_i"].mean()
            house_mean = result.loc[house_genes, "morans_i"].mean()
            assert grad_mean > house_mean, (
                f"Gradient genes should have higher Moran's I "
                f"({grad_mean:.3f}) than housekeeping ({house_mean:.3f})"
            )


class TestCallSvgs:
    """Test SVG calling."""

    def test_adds_is_svg_column(self, preprocessed: ad.AnnData) -> None:
        scores = compute_morans_i(preprocessed)
        result = call_svgs(scores)
        assert "is_svg" in result.columns

    def test_is_svg_is_boolean(self, preprocessed: ad.AnnData) -> None:
        scores = compute_morans_i(preprocessed)
        result = call_svgs(scores)
        assert result["is_svg"].dtype == bool

    def test_high_scoring_genes_called(self, preprocessed: ad.AnnData) -> None:
        """Genes with high Moran's I should be called as SVGs."""
        scores = compute_morans_i(preprocessed)
        result = call_svgs(scores)
        high_mask = result["morans_i"] >= 0.5
        if high_mask.any():
            assert result.loc[high_mask, "is_svg"].all()


class TestGetOracleMarkers:
    """Test oracle marker retrieval."""

    def test_returns_list(self, preprocessed: ad.AnnData) -> None:
        result = get_oracle_markers(preprocessed)
        assert isinstance(result, list)

    def test_all_markers_present_in_dataset(
        self, preprocessed: ad.AnnData
    ) -> None:
        """All returned markers must exist in the dataset var_names."""
        markers = get_oracle_markers(preprocessed)
        for m in markers:
            assert m in preprocessed.var_names

    def test_known_markers_detected(
        self, preprocessed: ad.AnnData
    ) -> None:
        """Fixture contains Mbp, Mog, Rorb, Cux2, Reln as markers."""
        markers = get_oracle_markers(preprocessed)
        # At least some fixture markers should be found
        fixture_markers = {"Mbp", "Mog", "Rorb", "Cux2", "Reln"}
        found = fixture_markers.intersection(set(markers))
        assert len(found) > 0


class TestRunSvgDetection:
    """Test the full SVG detection pipeline."""

    def test_returns_svg_result(self, preprocessed: ad.AnnData) -> None:
        result = run_svg_detection(preprocessed)
        assert isinstance(result, SVGResult)

    def test_scores_dataframe_populated(self, preprocessed: ad.AnnData) -> None:
        result = run_svg_detection(preprocessed)
        assert len(result.scores) > 0

    def test_top_svgs_length(self, preprocessed: ad.AnnData) -> None:
        from config.settings import settings
        result = run_svg_detection(preprocessed)
        assert len(result.top_svgs) <= settings.n_top_svgs

    def test_n_genes_tested_correct(self, preprocessed: ad.AnnData) -> None:
        result = run_svg_detection(preprocessed)
        assert result.n_genes_tested == preprocessed.n_vars

    def test_marker_genes_tested_subset_of_var(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_svg_detection(preprocessed)
        for m in result.marker_genes_tested:
            assert m in preprocessed.var_names

    def test_marker_genes_found_subset_of_tested(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_svg_detection(preprocessed)
        for m in result.marker_genes_found:
            assert m in result.marker_genes_tested

    def test_gradient_genes_in_top_svgs(
        self, preprocessed: ad.AnnData
    ) -> None:
        """At least some gradient genes should rank in the top SVGs."""
        result = run_svg_detection(preprocessed)
        grad_in_top = [g for g in result.top_svgs if g.startswith("Grad_")]
        assert len(grad_in_top) > 0
