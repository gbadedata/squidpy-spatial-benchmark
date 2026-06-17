"""Tests for preprocessing module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.preprocessing import (
    PreprocessingReport,
    build_spatial_graph,
    filter_and_normalise,
    run_preprocessing,
    select_hvg_and_embed,
)


class TestFilterAndNormalise:
    """Test gene filtering and expression normalisation."""

    def test_preserves_spot_count(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        result = filter_and_normalise(synthetic_spatial_adata.copy())
        assert result.n_obs == synthetic_spatial_adata.n_obs

    def test_reduces_gene_count(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Genes expressed in fewer than 10 spots should be removed."""
        result = filter_and_normalise(synthetic_spatial_adata.copy())
        assert result.n_vars <= synthetic_spatial_adata.n_vars

    def test_log_norm_layer_saved(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Log-normalised layer must be saved before scaling."""
        result = filter_and_normalise(synthetic_spatial_adata.copy())
        assert "log_norm" in result.layers

    def test_no_negative_values_after_log(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Log1p of non-negative counts should be non-negative."""
        import scipy.sparse as sp
        result = filter_and_normalise(synthetic_spatial_adata.copy())
        X = result.layers["log_norm"]
        if sp.issparse(X):
            X = X.toarray()
        assert np.all(X >= 0)

    def test_log_norm_finite(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Log-normalised values should be finite."""
        import scipy.sparse as sp
        result = filter_and_normalise(synthetic_spatial_adata.copy())
        X = result.layers["log_norm"]
        if sp.issparse(X):
            X = X.toarray()
        assert np.all(np.isfinite(X))


class TestSelectHvgAndEmbed:
    """Test HVG selection, PCA, and UMAP."""

    @pytest.fixture
    def normalised(self, synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
        return filter_and_normalise(synthetic_spatial_adata.copy())

    def test_hvg_column_added(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        assert "highly_variable" in result.var.columns

    def test_at_least_one_hvg(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        n_hvg = int(result.var["highly_variable"].sum())
        assert n_hvg > 0

    def test_pca_embedding_present(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        assert "X_pca" in result.obsm

    def test_umap_embedding_present(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        assert "X_umap" in result.obsm

    def test_umap_shape(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        assert result.obsm["X_umap"].shape == (normalised.n_obs, 2)

    def test_umap_finite(self, normalised: ad.AnnData) -> None:
        result = select_hvg_and_embed(normalised)
        assert np.all(np.isfinite(result.obsm["X_umap"]))


class TestBuildSpatialGraph:
    """Test spatial neighbourhood graph construction."""

    @pytest.fixture
    def embedded(self, synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
        adata = filter_and_normalise(synthetic_spatial_adata.copy())
        return select_hvg_and_embed(adata)

    def test_spatial_connectivities_added(self, embedded: ad.AnnData) -> None:
        result = build_spatial_graph(embedded)
        assert "spatial_connectivities" in result.obsp

    def test_spatial_distances_added(self, embedded: ad.AnnData) -> None:
        result = build_spatial_graph(embedded)
        assert "spatial_distances" in result.obsp

    def test_graph_shape(self, embedded: ad.AnnData) -> None:
        result = build_spatial_graph(embedded)
        n = result.n_obs
        assert result.obsp["spatial_connectivities"].shape == (n, n)

    def test_graph_has_connections(self, embedded: ad.AnnData) -> None:
        result = build_spatial_graph(embedded)
        n_conn = result.obsp["spatial_connectivities"].nnz
        assert n_conn > 0

    def test_graph_is_symmetric(self, embedded: ad.AnnData) -> None:
        """Spatial neighbourhood graph should be symmetric."""
        result = build_spatial_graph(embedded)
        conn = result.obsp["spatial_connectivities"]
        diff = (conn - conn.T).data
        assert np.allclose(diff, 0, atol=1e-6)


class TestRunPreprocessing:
    """Test the full preprocessing pipeline."""

    def test_returns_adata_and_report(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata, report = run_preprocessing(synthetic_spatial_adata.copy())
        assert isinstance(adata, ad.AnnData)
        assert isinstance(report, PreprocessingReport)

    def test_report_spots_consistent(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata, report = run_preprocessing(synthetic_spatial_adata.copy())
        assert report.n_spots == adata.n_obs

    def test_report_n_clusters_correct(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        _, report = run_preprocessing(synthetic_spatial_adata.copy())
        assert report.n_clusters == 4  # fixture has 4 domains

    def test_report_spatial_connections_positive(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        _, report = run_preprocessing(synthetic_spatial_adata.copy())
        assert report.n_spatial_connections > 0

    def test_all_layers_present(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
        assert "log_norm" in adata.layers

    def test_spatial_graph_present(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
        assert "spatial_connectivities" in adata.obsp

    def test_embeddings_present(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm
