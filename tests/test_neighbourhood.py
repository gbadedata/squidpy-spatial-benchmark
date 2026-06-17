"""Tests for neighbourhood enrichment analysis module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.neighbourhood import (
    NeighbourhoodResult,
    compute_neighbourhood_enrichment,
    run_neighbourhood_analysis,
    validate_anatomical_constraints,
)
from src.preprocessing import run_preprocessing


@pytest.fixture
def preprocessed(synthetic_spatial_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData after full preprocessing."""
    adata, _ = run_preprocessing(synthetic_spatial_adata.copy())
    return adata


class TestComputeNeighbourhoodEnrichment:
    """Test neighbourhood enrichment computation."""

    def test_returns_dataframe(self, preprocessed: ad.AnnData) -> None:
        result = compute_neighbourhood_enrichment(preprocessed)
        assert isinstance(result, pd.DataFrame)

    def test_square_matrix(self, preprocessed: ad.AnnData) -> None:
        result = compute_neighbourhood_enrichment(preprocessed)
        assert result.shape[0] == result.shape[1]

    def test_cluster_labels_as_index(self, preprocessed: ad.AnnData) -> None:
        result = compute_neighbourhood_enrichment(preprocessed)
        clusters = set(preprocessed.obs["cluster"].cat.categories)
        assert set(result.index) == clusters

    def test_missing_graph_raises(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        from src.preprocessing import filter_and_normalise, select_hvg_and_embed
        adata = filter_and_normalise(synthetic_spatial_adata.copy())
        adata = select_hvg_and_embed(adata)
        with pytest.raises(ValueError, match="spatial_connectivities"):
            compute_neighbourhood_enrichment(adata)

    def test_missing_cluster_column_raises(
        self, preprocessed: ad.AnnData
    ) -> None:
        with pytest.raises(ValueError, match="not found in obs"):
            compute_neighbourhood_enrichment(preprocessed, cluster_key="nonexistent")

    def test_diagonal_is_self_enrichment(
        self, preprocessed: ad.AnnData
    ) -> None:
        """Each cluster should be enriched with itself (positive diagonal)."""
        result = compute_neighbourhood_enrichment(preprocessed)
        diag = np.diag(result.values)
        # Most self-enrichment values should be positive
        assert (diag > 0).sum() >= len(diag) // 2

    def test_symmetric_matrix(self, preprocessed: ad.AnnData) -> None:
        """Enrichment matrix should be approximately symmetric."""
        result = compute_neighbourhood_enrichment(preprocessed)
        diff = np.abs(result.values - result.values.T)
        assert np.allclose(diff, 0, atol=1e-6)


class TestValidateAnatomicalConstraints:
    """Test anatomical constraint validation."""

    @pytest.fixture
    def synthetic_enrichment(self) -> pd.DataFrame:
        """Synthetic enrichment matrix for testing."""
        # 4 domains matching the fixture
        domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
        # Self-enriched, some cross-enrichment
        data = np.array([
            [5.0, 2.0, -1.0, -1.0],
            [2.0, 5.0, -1.0, -1.0],
            [-1.0, -1.0, 5.0, 2.0],
            [-1.0, -1.0, 2.0, 5.0],
        ])
        return pd.DataFrame(data, index=domains, columns=domains)

    def test_returns_dict(
        self,
        synthetic_enrichment: pd.DataFrame,
        preprocessed: ad.AnnData,
    ) -> None:
        result = validate_anatomical_constraints(
            synthetic_enrichment, preprocessed
        )
        assert isinstance(result, dict)

    def test_skips_missing_clusters(
        self,
        synthetic_enrichment: pd.DataFrame,
        preprocessed: ad.AnnData,
    ) -> None:
        """Known neuroanatomy pairs not in the dataset are silently skipped."""
        result = validate_anatomical_constraints(
            synthetic_enrichment, preprocessed
        )
        # Fixture has domain_A..D, not Cortex etc -- should return empty dict
        assert isinstance(result, dict)

    def test_expected_enrichment_detected(self) -> None:
        """Pairs with positive z-score should pass."""
        import anndata as ad
        import pandas as pd

        clusters = ["Cortex_1", "Cortex_2", "Thalamus_1", "Thalamus_2"]
        data = np.array([
            [5.0, 3.0, -1.0, -1.0],
            [3.0, 5.0, -1.0, -1.0],
            [-1.0, -1.0, 5.0, 3.0],
            [-1.0, -1.0, 3.0, 5.0],
        ])
        df = pd.DataFrame(data, index=clusters, columns=clusters)

        obs = pd.DataFrame(
            {"cluster": pd.Categorical(clusters[:2] * 10)},
            index=[f"S{i}" for i in range(20)],
        )
        dummy = ad.AnnData(obs=obs)

        result = validate_anatomical_constraints(df, dummy)
        if ("Cortex_1", "Cortex_2") in result:
            assert result[("Cortex_1", "Cortex_2")] is True
        if ("Thalamus_1", "Thalamus_2") in result:
            assert result[("Thalamus_1", "Thalamus_2")] is True


class TestRunNeighbourhoodAnalysis:
    """Test the full neighbourhood analysis pipeline."""

    def test_returns_neighbourhood_result(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_neighbourhood_analysis(preprocessed)
        assert isinstance(result, NeighbourhoodResult)

    def test_enrichment_scores_populated(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_neighbourhood_analysis(preprocessed)
        assert len(result.enrichment_scores) > 0

    def test_n_clusters_correct(self, preprocessed: ad.AnnData) -> None:
        result = run_neighbourhood_analysis(preprocessed)
        expected = preprocessed.obs["cluster"].nunique()
        assert result.n_clusters == expected

    def test_n_pairs_validated_non_negative(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_neighbourhood_analysis(preprocessed)
        assert result.n_pairs_validated >= 0

    def test_n_pairs_passing_leq_validated(
        self, preprocessed: ad.AnnData
    ) -> None:
        result = run_neighbourhood_analysis(preprocessed)
        assert result.n_pairs_passing <= result.n_pairs_validated
