"""Topological Data Analysis using gudhi.

Computes persistence-based topological features on spatial gene
expression landscapes. This module implements Task 3 of the benchmark:
comparing TDA features against Moran's I to identify genes with
spatial structure that standard autocorrelation underestimates.

Topological Data Analysis captures multi-scale spatial structure
through persistent homology. For a gene's spatial expression landscape:

- H0 (connected components): measures how many spatially coherent
  expression domains exist and how strongly they persist
- H1 (loops/cycles): detects ring-like or annular expression patterns
  that cannot be captured by pairwise autocorrelation

The key scientific claim: Moran's I measures PAIRWISE spatial
correlation -- whether nearby spots have similar expression. Persistent
homology measures GLOBAL TOPOLOGY -- connected components, loops, and
voids in the expression landscape at multiple spatial scales. A gene
with a ring-like expression domain (e.g. surrounding a brain structure)
has low Moran's I but high H1 persistence. This is a genuine case where
TDA detects spatial structure that Moran's I misses.

Persistence entropy is used as a scalar summary of the persistence
diagram, enabling direct comparison with Moran's I rankings.

Reference: Chazal & Michel (2021). An Introduction to Topological
Data Analysis: Fundamental and Practical Aspects for Data Scientists.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import gudhi as gd
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PersistenceSummary:
    """Topological summary for a single gene.

    Attributes:
        gene: Gene name.
        persistence_entropy_h0: Entropy of H0 persistence diagram.
            High entropy = many persistent connected components.
        persistence_entropy_h1: Entropy of H1 persistence diagram.
            High entropy = ring-like or looping expression patterns.
        max_persistence_h0: Largest H0 persistence value.
        max_persistence_h1: Largest H1 persistence value.
        n_features_h0: Number of significant H0 features.
        n_features_h1: Number of significant H1 features.
        betti_0: Number of connected components at median filtration.
        betti_1: Number of loops at median filtration.
    """

    gene: str
    persistence_entropy_h0: float
    persistence_entropy_h1: float
    max_persistence_h0: float
    max_persistence_h1: float
    n_features_h0: int
    n_features_h1: int
    betti_0: int
    betti_1: int


@dataclass
class TDAResult:
    """Result from the TDA benchmark comparison.

    Attributes:
        summaries: Dict mapping gene name to PersistenceSummary.
        comparison_df: DataFrame comparing Moran's I rank vs TDA
            persistence entropy rank for each gene.
        tda_unique_genes: Genes where TDA rank >> Moran's I rank,
            suggesting ring-like or topologically complex patterns
            that autocorrelation underestimates.
        n_genes_analysed: Total genes analysed.
    """

    summaries: dict[str, PersistenceSummary]
    comparison_df: pd.DataFrame
    tda_unique_genes: list[str]
    n_genes_analysed: int


def _expression_to_point_cloud(
    adata: ad.AnnData,
    gene: str,
    layer: str = "log_norm",
) -> np.ndarray:
    """Convert a gene's spatial expression to a weighted point cloud.

    Each spot with above-threshold expression becomes a point in 2D space.
    Spots are weighted by expression -- higher expression creates more
    representative points (achieved by thresholding at the gene's median).

    Args:
        adata: AnnData with spatial coordinates and log_norm layer.
        gene: Gene name to extract.
        layer: Expression layer to use.

    Returns:
        Array of shape (n_expressing_spots, 2) with spatial coordinates
        of spots expressing this gene above its median.
    """
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names.")

    gene_idx = adata.var_names.get_loc(gene)
    X = adata.layers[layer]
    if sp.issparse(X):
        expr = np.asarray(X[:, gene_idx].todense()).flatten()
    else:
        expr = X[:, gene_idx]

    # Use spots above median expression as the point cloud
    threshold = float(np.median(expr[expr > 0])) if np.any(expr > 0) else 0.0
    mask = expr > threshold

    coords = adata.obsm["spatial"][mask].astype(float)

    # Normalise spatial coordinates to [0, 1] range
    if len(coords) > 0:
        coords = coords - coords.min(axis=0)
        scale = coords.max(axis=0)
        scale[scale == 0] = 1.0
        coords = coords / scale

    return coords


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float | None = None,
) -> tuple[list, list]:
    """Compute persistent homology via Rips complex filtration.

    Constructs a Vietoris-Rips complex on the point cloud and computes
    persistence in dimensions 0 (connected components) and 1 (loops).

    Args:
        points: Array of shape (n_points, 2) with 2D coordinates.
        max_edge_length: Maximum edge length for Rips filtration.
            Defaults to settings.rips_max_edge_length.

    Returns:
        Tuple of (H0 intervals, H1 intervals) where each is a list
        of (birth, death) pairs. Infinite death values are replaced
        with max_edge_length.
    """
    if max_edge_length is None:
        max_edge_length = settings.rips_max_edge_length

    if len(points) < 3:
        return [], []

    rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    h0 = []
    for birth, death in simplex_tree.persistence_intervals_in_dimension(0):
        if np.isinf(death):
            death = max_edge_length
        if death - birth >= settings.min_persistence:
            h0.append((float(birth), float(death)))

    h1 = []
    for birth, death in simplex_tree.persistence_intervals_in_dimension(1):
        if np.isinf(death):
            death = max_edge_length
        if death - birth >= settings.min_persistence:
            h1.append((float(birth), float(death)))

    return h0, h1


def persistence_entropy(intervals: list[tuple[float, float]]) -> float:
    """Compute persistence entropy of a diagram.

    Persistence entropy measures the complexity of a persistence diagram.
    A uniform distribution of lifetimes gives maximum entropy.
    A single dominant feature gives minimum entropy.

    Formula: H = -sum(p_i * log(p_i)) where p_i = L_i / sum(L_j)
    and L_i is the lifetime (death - birth) of feature i.

    Args:
        intervals: List of (birth, death) pairs.

    Returns:
        Scalar entropy value. Returns 0 if no intervals.
    """
    if not intervals:
        return 0.0

    lifetimes = np.array([d - b for b, d in intervals])
    total = lifetimes.sum()
    if total <= 0:
        return 0.0

    probs = lifetimes / total
    probs = probs[probs > 0]
    return float(max(0.0, -np.sum(probs * np.log(probs + 1e-12))))


def analyse_gene(
    adata: ad.AnnData,
    gene: str,
    layer: str = "log_norm",
) -> PersistenceSummary:
    """Compute full topological summary for one gene.

    Args:
        adata: Preprocessed AnnData.
        gene: Gene to analyse.
        layer: Expression layer.

    Returns:
        PersistenceSummary with all topological features.
    """
    points = _expression_to_point_cloud(adata, gene, layer)

    if len(points) < 3:
        return PersistenceSummary(
            gene=gene,
            persistence_entropy_h0=0.0,
            persistence_entropy_h1=0.0,
            max_persistence_h0=0.0,
            max_persistence_h1=0.0,
            n_features_h0=0,
            n_features_h1=0,
            betti_0=0,
            betti_1=0,
        )

    h0, h1 = compute_persistence(points)

    max_h0 = max((d - b for b, d in h0), default=0.0)
    max_h1 = max((d - b for b, d in h1), default=0.0)

    # Betti numbers: count features alive at median filtration value
    median_scale = settings.rips_max_edge_length / 2
    betti_0 = sum(1 for b, d in h0 if b <= median_scale <= d)
    betti_1 = sum(1 for b, d in h1 if b <= median_scale <= d)

    return PersistenceSummary(
        gene=gene,
        persistence_entropy_h0=persistence_entropy(h0),
        persistence_entropy_h1=persistence_entropy(h1),
        max_persistence_h0=max_h0,
        max_persistence_h1=max_h1,
        n_features_h0=len(h0),
        n_features_h1=len(h1),
        betti_0=betti_0,
        betti_1=betti_1,
    )


def run_tda_analysis(
    adata: ad.AnnData,
    morans_scores: pd.DataFrame,
    n_genes: int | None = None,
    layer: str = "log_norm",
) -> TDAResult:
    """Run TDA on selected genes and compare with Moran's I rankings.

    Selects three gene categories for comparison:
    - Top Moran's I genes (highest spatial autocorrelation)
    - Low Moran's I genes (near threshold, spatially variable but not extreme)
    - Random housekeeping-like genes (control)

    For each gene, computes persistence entropy and constructs a rank
    comparison DataFrame. Genes where TDA rank >> Moran's I rank are
    candidates for ring-like or topologically complex patterns.

    Args:
        adata: Preprocessed AnnData with log_norm layer and spatial coords.
        morans_scores: DataFrame from run_svg_detection with morans_i column.
        n_genes: Number of genes per category. Defaults to n_tda_genes.
        layer: Expression layer.

    Returns:
        TDAResult with persistence summaries and rank comparison.
    """
    if n_genes is None:
        n_genes = settings.n_tda_genes

    n_per_cat = max(3, n_genes // 3)

    # Select genes: top Moran's I, mid-range, and low (control)
    n_total = len(morans_scores)
    top_genes = list(morans_scores.index[:n_per_cat])
    mid_start = max(0, n_total // 2 - n_per_cat // 2)
    mid_genes = list(morans_scores.index[mid_start: mid_start + n_per_cat])
    low_genes = list(morans_scores.index[-n_per_cat:])

    selected = list(dict.fromkeys(top_genes + mid_genes + low_genes))
    logger.info(
        "tda_analysis: analysing %d genes (%d top, %d mid, %d low Moran's I)",
        len(selected), len(top_genes), len(mid_genes), len(low_genes),
    )

    # Compute persistence for each gene
    summaries: dict[str, PersistenceSummary] = {}
    for i, gene in enumerate(selected):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            summary = analyse_gene(adata, gene, layer)
        summaries[gene] = summary
        if (i + 1) % 5 == 0:
            logger.info("tda_progress: %d/%d genes", i + 1, len(selected))

    # Build comparison DataFrame
    records = []
    for gene, summary in summaries.items():
        morans_i = float(morans_scores.loc[gene, "morans_i"]) \
            if gene in morans_scores.index else 0.0
        records.append({
            "gene": gene,
            "morans_i": morans_i,
            "persistence_entropy_h0": summary.persistence_entropy_h0,
            "persistence_entropy_h1": summary.persistence_entropy_h1,
            "max_persistence_h1": summary.max_persistence_h1,
            "n_features_h1": summary.n_features_h1,
            "betti_1": summary.betti_1,
        })

    comparison_df = pd.DataFrame(records).set_index("gene")

    # Rank genes by each method
    comparison_df["morans_rank"] = comparison_df["morans_i"].rank(ascending=False)
    comparison_df["tda_h1_rank"] = comparison_df["persistence_entropy_h1"].rank(
        ascending=False
    )
    comparison_df["rank_diff"] = (
        comparison_df["morans_rank"] - comparison_df["tda_h1_rank"]
    )

    # Genes where TDA rank is much better than Moran's I rank
    # (positive rank_diff = TDA says more spatial structure than Moran's I)
    tda_unique = list(
        comparison_df[comparison_df["rank_diff"] > len(selected) * 0.3].index
    )

    logger.info(
        "tda_analysis_complete: %d genes, %d TDA-unique genes "
        "(TDA rank >> Moran's I rank)",
        len(selected), len(tda_unique),
    )
    return TDAResult(
        summaries=summaries,
        comparison_df=comparison_df,
        tda_unique_genes=tda_unique,
        n_genes_analysed=len(selected),
    )
