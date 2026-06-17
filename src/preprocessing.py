"""Preprocessing for spatial transcriptomics analysis.

Prepares the Visium mouse brain dataset for spatial analysis:
1. Filter spots and genes by quality metrics
2. Normalise and log-transform expression
3. Select highly variable genes
4. PCA and UMAP (transcriptional embedding)
5. Build the Visium spatial neighbourhood graph

The spatial neighbourhood graph connects each spot to its immediate
hexagonal grid neighbours. This graph is the foundation for all
downstream spatial statistics: Moran's I, neighbourhood enrichment,
and co-occurrence analysis. The Visium coord_type uses the known
hexagonal grid topology rather than a distance-based kNN graph,
which is more appropriate for the regular Visium array layout.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import scanpy as sc
import squidpy as sq

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingReport:
    """Summary statistics from preprocessing.

    Attributes:
        n_spots: Spots after filtering.
        n_genes: Genes after filtering.
        n_hvg: Highly variable genes selected.
        n_clusters: Number of anatomical cluster labels.
        spatial_graph_key: Key in obsp where spatial graph is stored.
        n_spatial_connections: Total edges in the spatial graph.
    """

    n_spots: int
    n_genes: int
    n_hvg: int
    n_clusters: int
    spatial_graph_key: str
    n_spatial_connections: int


def filter_and_normalise(adata: ad.AnnData) -> ad.AnnData:
    """Filter low-quality spots/genes and normalise expression.

    Steps:
    1. Filter genes expressed in fewer than 10 spots
    2. Normalise each spot to 10,000 total counts
    3. Log1p transform
    4. Save log-normalised layer before scaling

    Args:
        adata: Raw AnnData with counts in .X.

    Returns:
        Filtered, normalised, log-transformed AnnData.
    """
    n_spots_before = adata.n_obs
    n_genes_before = adata.n_vars

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save log-normalised layer -- required by downstream tools
    # that need this specific transformation state
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        adata.layers["log_norm"] = adata.X.copy()
    else:
        adata.layers["log_norm"] = adata.X.copy()

    logger.info(
        "filter_and_normalise: %d->%d spots, %d->%d genes",
        n_spots_before, adata.n_obs,
        n_genes_before, adata.n_vars,
    )
    return adata


def select_hvg_and_embed(adata: ad.AnnData) -> ad.AnnData:
    """Select highly variable genes, run PCA and UMAP.

    HVG selection uses the Seurat flavour on log-normalised data.
    PCA is computed on scaled HVG expression. UMAP provides the
    transcriptional embedding for visualisation.

    Note: if the dataset already has X_pca and X_umap (as the
    squidpy Visium dataset does), we recompute them on the
    filtered/normalised data for consistency.

    Args:
        adata: Filtered and normalised AnnData.

    Returns:
        AnnData with highly_variable, X_pca, X_umap in place.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=settings.n_top_genes,
            flavor="seurat",
        )

    n_hvg = int(adata.var["highly_variable"].sum())

    # Scale before PCA (on HVG subset)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=settings.n_pcs,
        mask_var="highly_variable",
        random_state=settings.random_seed,
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=settings.n_neighbors,
        n_pcs=settings.n_pcs,
        random_state=settings.random_seed,
    )
    sc.tl.umap(adata, random_state=settings.random_seed)

    logger.info(
        "hvg_and_embed: %d HVGs, PCA n_comps=%d, UMAP done",
        n_hvg, settings.n_pcs,
    )
    return adata


def build_spatial_graph(adata: ad.AnnData) -> ad.AnnData:
    """Build the Visium hexagonal spatial neighbourhood graph.

    Uses squidpy's spatial_neighbors with coord_type='grid' which
    correctly handles regular array geometry (squidpy 1.8.x renamed
    'visium' to 'grid'). Each spot is connected to its immediate
    hex-grid neighbours (n_rings=1, giving 6 neighbours per interior spot).

    The spatial graph is the foundation for:
    - Moran's I spatial autocorrelation (Task 1)
    - Neighbourhood enrichment analysis (Task 2)
    - Co-occurrence scores (Task 2)

    Args:
        adata: AnnData with obsm['spatial'] coordinates.

    Returns:
        AnnData with spatial connectivities in obsp.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sq.gr.spatial_neighbors(
            adata,
            coord_type=settings.coord_type,
            n_rings=settings.n_rings,
            key_added="spatial",
        )

    graph_key = "spatial_connectivities"
    if graph_key in adata.obsp:
        n_conn = int(adata.obsp[graph_key].nnz)
    else:
        n_conn = 0

    logger.info(
        "spatial_graph_built: coord_type=%s, n_rings=%d, "
        "n_connections=%d",
        settings.coord_type, settings.n_rings, n_conn,
    )
    return adata


def run_preprocessing(adata: ad.AnnData) -> tuple[ad.AnnData, PreprocessingReport]:
    """Full preprocessing pipeline.

    Steps:
    1. Filter and normalise
    2. HVG selection, PCA, UMAP
    3. Spatial neighbourhood graph

    Args:
        adata: Raw AnnData from data loader.

    Returns:
        Tuple of (preprocessed AnnData, PreprocessingReport).
    """
    adata = filter_and_normalise(adata)
    adata = select_hvg_and_embed(adata)
    adata = build_spatial_graph(adata)

    n_hvg = int(adata.var["highly_variable"].sum()) \
        if "highly_variable" in adata.var.columns else 0
    graph_key = "spatial_connectivities"
    n_conn = int(adata.obsp[graph_key].nnz) if graph_key in adata.obsp else 0

    report = PreprocessingReport(
        n_spots=adata.n_obs,
        n_genes=adata.n_vars,
        n_hvg=n_hvg,
        n_clusters=adata.obs["cluster"].nunique(),
        spatial_graph_key=graph_key,
        n_spatial_connections=n_conn,
    )

    logger.info(
        "preprocessing_complete: %d spots, %d genes, %d HVGs, "
        "%d connections",
        report.n_spots, report.n_genes,
        report.n_hvg, report.n_spatial_connections,
    )
    return adata, report
