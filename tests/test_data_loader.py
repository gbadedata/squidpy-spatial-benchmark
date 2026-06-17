"""Tests for data_loader module."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

from src.data_loader import get_dataset, load_h5ad, validate_dataset


class TestLoadH5ad:
    """Test h5ad loading."""

    def test_load_existing_h5ad(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == synthetic_spatial_adata.n_obs
        assert result.n_vars == synthetic_spatial_adata.n_vars

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_h5ad("/nonexistent/path/fake.h5ad")

    def test_preserves_spatial_coords(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert "spatial" in result.obsm

    def test_preserves_cluster_labels(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert "cluster" in result.obs.columns

    def test_preserves_gene_names(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert list(result.var_names) == list(synthetic_spatial_adata.var_names)


class TestValidateDataset:
    """Test dataset structure validation."""

    def test_valid_dataset_passes(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        validate_dataset(synthetic_spatial_adata)  # should not raise

    def test_missing_spatial_raises(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata = synthetic_spatial_adata.copy()
        del adata.obsm["spatial"]
        with pytest.raises(ValueError, match="Missing obsm"):
            validate_dataset(adata)

    def test_missing_cluster_column_raises(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        import pandas as pd
        adata = synthetic_spatial_adata.copy()
        adata.obs = pd.DataFrame(index=adata.obs.index)
        with pytest.raises(ValueError, match="Missing required obs columns"):
            validate_dataset(adata)

    def test_empty_dataset_raises(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata = synthetic_spatial_adata[:0, :].copy()
        with pytest.raises(ValueError, match="Empty dataset"):
            validate_dataset(adata)

    def test_spatial_coords_shape(
        self, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        validate_dataset(synthetic_spatial_adata)
        coords = synthetic_spatial_adata.obsm["spatial"]
        assert coords.shape[1] == 2


class TestGetDataset:
    """Test main entry point."""

    def test_get_dataset_from_h5ad(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = get_dataset(filepath=filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 120

    def test_get_dataset_validates_structure(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        adata = synthetic_spatial_adata.copy()
        del adata.obsm["spatial"]
        filepath = tmp_path / "invalid.h5ad"
        adata.write_h5ad(filepath)
        with pytest.raises(ValueError, match="Missing obsm"):
            get_dataset(filepath=filepath)

    def test_get_dataset_returns_validated_adata(
        self, tmp_path: Path, synthetic_spatial_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_spatial_adata.write_h5ad(filepath)
        result = get_dataset(filepath=filepath)
        assert "spatial" in result.obsm
        assert "cluster" in result.obs.columns
