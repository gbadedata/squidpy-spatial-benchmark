"""Tests for synthetic spatial fixture."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp


class TestSyntheticSpatialAdata:

    def test_shape(self, synthetic_spatial_adata: ad.AnnData) -> None:
        assert synthetic_spatial_adata.n_obs == 120
        assert synthetic_spatial_adata.n_vars == 60

    def test_spatial_coords_present(self, synthetic_spatial_adata: ad.AnnData) -> None:
        assert "spatial" in synthetic_spatial_adata.obsm

    def test_spatial_coords_shape(self, synthetic_spatial_adata: ad.AnnData) -> None:
        coords = synthetic_spatial_adata.obsm["spatial"]
        assert coords.shape == (120, 2)

    def test_cluster_labels_present(self, synthetic_spatial_adata: ad.AnnData) -> None:
        assert "cluster" in synthetic_spatial_adata.obs.columns

    def test_four_domains(self, synthetic_spatial_adata: ad.AnnData) -> None:
        domains = synthetic_spatial_adata.obs["cluster"].unique()
        assert len(domains) == 4

    def test_sparse_matrix(self, synthetic_spatial_adata: ad.AnnData) -> None:
        assert sp.issparse(synthetic_spatial_adata.X)

    def test_no_negative_counts(self, synthetic_spatial_adata: ad.AnnData) -> None:
        X = synthetic_spatial_adata.X.toarray()
        assert np.all(X >= 0)

    def test_gradient_genes_spatially_variable(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Gradient genes should have higher expression variance than housekeeping."""
        X = synthetic_spatial_adata.X.toarray()
        gradient_var = X[:, :5].var(axis=0).mean()
        house_var = X[:, 20:].var(axis=0).mean()
        assert gradient_var > house_var, (
            f"Gradient genes variance ({gradient_var:.3f}) should exceed "
            f"housekeeping variance ({house_var:.3f})"
        )

    def test_ring_genes_peripheral_enriched(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        """Ring genes should have higher expression at periphery than centre.

        The ring pattern is defined relative to the grid centroid (0.5, 0.5)
        in normalised coordinates, with peak at distance 0.35 from centre.
        Peripheral spots (dist > 0.3) should have higher expression than
        central spots (dist < 0.15).
        """
        coords = synthetic_spatial_adata.obsm["spatial"]
        x = coords[:, 0]
        y = coords[:, 1]
        # Use the normalised centroid -- same as the fixture construction
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        dist = np.sqrt((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2)

        # Spots near the ring radius (0.35) should be enriched
        near_ring = (dist > 0.25) & (dist < 0.45)
        central = dist < 0.15

        X = synthetic_spatial_adata.X.toarray()
        if near_ring.sum() > 0 and central.sum() > 0:
            ring_mean = X[near_ring, 5:10].mean()
            central_mean = X[central, 5:10].mean()
            assert ring_mean > central_mean, (
                f"Ring-pattern genes should be higher near ring ({ring_mean:.3f}) "
                f"than at centre ({central_mean:.3f})"
            )

    def test_known_markers_present(self, synthetic_spatial_adata: ad.AnnData) -> None:
        """Known Allen Brain Atlas marker genes are in the fixture."""
        gene_names = set(synthetic_spatial_adata.var_names)
        for marker in ["Mbp", "Mog", "Rorb", "Cux2", "Reln"]:
            assert marker in gene_names

    def test_uns_spatial_structure(self, synthetic_spatial_adata: ad.AnnData) -> None:
        """uns['spatial'] has expected Visium-compatible structure."""
        assert "spatial" in synthetic_spatial_adata.uns
        assert "V1_Adult_Mouse_Brain" in synthetic_spatial_adata.uns["spatial"]
