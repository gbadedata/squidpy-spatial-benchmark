"""Data acquisition and loading.

Downloads the 10x Genomics Visium H&E mouse brain dataset and loads
it into AnnData format. The dataset contains 2,688 spots across 15
named anatomical brain regions, with pre-computed PCA, UMAP, and
Leiden clustering. Named cluster labels ('cluster' column) provide
the anatomical ground truth for benchmark evaluation.

Dataset: V1 Adult Mouse Brain (10x Genomics Visium)
Source:  squidpy.datasets.visium_hne_adata() via scverse CDN
Size:    ~329 MB h5ad
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import squidpy as sq

from config.settings import settings

logger = logging.getLogger(__name__)


def download_visium(dest_dir: Path | str | None = None) -> ad.AnnData:
    """Download the Visium H&E mouse brain dataset via squidpy.

    Uses squidpy.datasets.visium_hne_adata() which fetches from the
    scverse CDN. The file is cached locally so subsequent calls do
    not re-download.

    Dataset structure after download:
    - 2,688 spots, 18,078 genes (pre-filtered)
    - obs['cluster']: 15 named anatomical brain regions (oracle)
    - obs['leiden']: numeric Leiden cluster IDs
    - obsm['spatial']: (n_spots, 2) physical coordinates
    - obsm['X_pca'], obsm['X_umap']: pre-computed embeddings
    - uns['spatial']: Visium scalefactors and image metadata

    Args:
        dest_dir: Directory to cache the h5ad. Defaults to
            settings.data_dir / 'anndata'.

    Returns:
        AnnData with raw counts and anatomical cluster labels.
    """
    dest_dir = Path(dest_dir) if dest_dir else settings.data_dir / "anndata"
    dest_dir.mkdir(parents=True, exist_ok=True)

    cache_path = dest_dir / settings.dataset_filename

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("dataset_cached: %s (%.1f MB)", cache_path, size_mb)
        return load_h5ad(cache_path)

    logger.info("downloading_visium via squidpy (scverse CDN)")
    import scanpy as sc
    sc.settings.datasetdir = str(dest_dir)
    adata = sq.datasets.visium_hne_adata()
    adata.var_names_make_unique()

    adata.write_h5ad(cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(
        "downloaded_and_cached: %d spots x %d genes -> %s (%.1f MB)",
        adata.n_obs, adata.n_vars, cache_path, size_mb,
    )
    return adata


def load_h5ad(filepath: str | Path) -> ad.AnnData:
    """Load an h5ad file into AnnData.

    Args:
        filepath: Path to the .h5ad file.

    Returns:
        AnnData object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info("loading_h5ad: %s", filepath)
    adata = ad.read_h5ad(filepath)
    adata.var_names_make_unique()
    logger.info("loaded: %d spots x %d genes", adata.n_obs, adata.n_vars)
    return adata


def validate_dataset(adata: ad.AnnData) -> None:
    """Validate the loaded dataset has required structure.

    Checks for spatial coordinates, anatomical cluster labels, and
    gene expression matrix. Raises ValueError if anything is missing.

    Args:
        adata: Loaded AnnData to validate.

    Raises:
        ValueError: If required components are missing.
    """
    if "spatial" not in adata.obsm:
        raise ValueError(
            "Missing obsm['spatial']. The dataset must contain "
            "physical spot coordinates for spatial analysis."
        )

    required_obs = ["cluster"]
    missing = [c for c in required_obs if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Missing required obs columns: {missing}. "
            "The dataset must contain named anatomical cluster labels "
            "for oracle-based benchmark evaluation."
        )

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            f"Empty dataset: {adata.n_obs} spots x {adata.n_vars} genes."
        )

    n_clusters = adata.obs["cluster"].nunique()
    coords = adata.obsm["spatial"]
    logger.info(
        "dataset_validated: %d spots x %d genes, %d clusters, "
        "spatial_coords_shape=%s",
        adata.n_obs, adata.n_vars, n_clusters, coords.shape,
    )


def get_dataset(filepath: str | Path | None = None) -> ad.AnnData:
    """Download (if needed) and load the Visium dataset.

    Main entry point. Downloads via squidpy if no explicit path given.
    Validates structure before returning.

    Args:
        filepath: Optional path to a local h5ad file. If None,
            downloads the Visium H&E mouse brain dataset.

    Returns:
        Validated AnnData with spatial coordinates and cluster labels.
    """
    if filepath is not None:
        adata = load_h5ad(filepath)
    else:
        adata = download_visium()

    validate_dataset(adata)
    return adata
