"""Visualisation for the spatial transcriptomics benchmark.

Generates publication-quality figures that form the visual evidence
for the benchmark:

  1. Spatial clusters       -- 15 anatomical regions on the tissue
  2. SVG spatial expression -- top spatially variable genes on tissue
  3. Neighbourhood matrix   -- enrichment z-scores between regions
  4. Persistence diagram    -- H0/H1 topological features for a gene
  5. Moran vs TDA scatter   -- the two methods' rankings compared

All figures use spatial coordinates (the tissue layout), not UMAP,
because spatial transcriptomics is fundamentally about tissue geometry.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import settings
from src.tda import _expression_to_point_cloud, compute_persistence

logger = logging.getLogger(__name__)

FIGSIZE = (8, 7)
DPI = 150


def _figures_dir() -> Path:
    d = settings.evidence_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_spatial_clusters(adata: ad.AnnData, cluster_key: str = "cluster") -> Path:
    """Plot the anatomical cluster labels on the tissue coordinates.

    Args:
        adata: AnnData with spatial coords and cluster labels.
        cluster_key: obs column with anatomical labels.

    Returns:
        Path to the saved figure.
    """
    coords = adata.obsm["spatial"]
    clusters = adata.obs[cluster_key].astype("category")
    categories = clusters.cat.categories

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap("tab20")
    for i, cat in enumerate(categories):
        mask = clusters == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, color=cmap(i % 20), label=cat,
        )
    ax.invert_yaxis()  # image coordinate convention
    ax.set_aspect("equal")
    ax.set_title("Anatomical regions (Visium mouse brain)")
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, markerscale=1.5)
    fig.tight_layout()

    out = _figures_dir() / "01_spatial_clusters.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_figure: %s", out)
    return out


def plot_svg_expression(
    adata: ad.AnnData,
    genes: list[str],
    layer: str = "log_norm",
) -> Path:
    """Plot spatial expression of top SVGs on the tissue.

    Args:
        adata: AnnData with spatial coords and expression.
        genes: Gene names to plot (up to 4).
        layer: Expression layer.

    Returns:
        Path to the saved figure.
    """
    import scipy.sparse as sp

    genes = [g for g in genes if g in adata.var_names][:4]
    coords = adata.obsm["spatial"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, gene in enumerate(genes):
        gene_idx = adata.var_names.get_loc(gene)
        X = adata.layers[layer] if layer in adata.layers else adata.X
        if sp.issparse(X):
            expr = np.asarray(X[:, gene_idx].todense()).flatten()
        else:
            expr = X[:, gene_idx]

        sc = axes[i].scatter(
            coords[:, 0], coords[:, 1],
            c=expr, s=10, cmap="viridis",
        )
        axes[i].invert_yaxis()
        axes[i].set_aspect("equal")
        axes[i].set_title(f"{gene} (log-normalised expression)")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        fig.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(len(genes), 4):
        axes[j].axis("off")

    fig.suptitle("Top spatially variable genes (Moran's I)", fontsize=14)
    fig.tight_layout()

    out = _figures_dir() / "02_svg_expression.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_figure: %s", out)
    return out


def plot_neighbourhood_matrix(enrichment_df: pd.DataFrame) -> Path:
    """Plot the neighbourhood enrichment z-score matrix as a heatmap.

    Args:
        enrichment_df: Cluster x cluster z-score DataFrame.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.nanpercentile(np.abs(enrichment_df.values), 95)
    im = ax.imshow(
        enrichment_df.values, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, aspect="auto",
    )
    ax.set_xticks(range(len(enrichment_df.columns)))
    ax.set_yticks(range(len(enrichment_df.index)))
    ax.set_xticklabels(enrichment_df.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(enrichment_df.index, fontsize=8)
    ax.set_title("Neighbourhood enrichment z-scores\n(red = co-localised, blue = segregated)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")
    fig.tight_layout()

    out = _figures_dir() / "03_neighbourhood_matrix.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_figure: %s", out)
    return out


def plot_persistence_diagram(
    adata: ad.AnnData,
    gene: str,
    layer: str = "log_norm",
) -> Path:
    """Plot the persistence diagram (H0 and H1) for a gene.

    A persistence diagram plots each topological feature as a point
    at (birth, death). Points far from the diagonal are persistent
    (long-lived) features; points near the diagonal are short-lived
    noise. H0 = connected components, H1 = loops.

    Args:
        adata: Preprocessed AnnData.
        gene: Gene to analyse.
        layer: Expression layer.

    Returns:
        Path to the saved figure.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        points = _expression_to_point_cloud(adata, gene, layer)
        h0, h1 = compute_persistence(points)

    fig, ax = plt.subplots(figsize=(7, 7))

    max_val = settings.rips_max_edge_length
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="diagonal")

    if h0:
        h0_arr = np.array(h0)
        ax.scatter(h0_arr[:, 0], h0_arr[:, 1], c="tab:blue",
                   s=40, alpha=0.7, label=f"H0 (components, n={len(h0)})")
    if h1:
        h1_arr = np.array(h1)
        ax.scatter(h1_arr[:, 0], h1_arr[:, 1], c="tab:red",
                   s=40, alpha=0.7, marker="^", label=f"H1 (loops, n={len(h1)})")

    ax.set_xlabel("birth (filtration scale)")
    ax.set_ylabel("death (filtration scale)")
    ax.set_title(f"Persistence diagram: {gene}\n(distance from diagonal = persistence)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    out = _figures_dir() / "04_persistence_diagram.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_figure: %s", out)
    return out


def plot_morans_vs_tda(comparison_df: pd.DataFrame) -> Path:
    """Scatter plot comparing Moran's I rank vs TDA H1 entropy rank.

    Genes far from the diagonal are ranked very differently by the two
    methods. A gene low on Moran's rank but high on TDA rank (upper-left)
    has topologically complex spatial structure that autocorrelation
    underestimates.

    Args:
        comparison_df: DataFrame with morans_rank and tda_h1_rank columns.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    mr = comparison_df["morans_rank"]
    tr = comparison_df["tda_h1_rank"]

    ax.scatter(mr, tr, s=60, alpha=0.7, c="tab:purple")

    # Label the most divergent genes
    diff = (mr - tr).abs()
    top_divergent = diff.nlargest(min(5, len(diff))).index
    for gene in top_divergent:
        ax.annotate(
            gene, (mr[gene], tr[gene]),
            fontsize=9, xytext=(5, 5), textcoords="offset points",
        )

    max_rank = max(mr.max(), tr.max())
    ax.plot([0, max_rank], [0, max_rank], "k--", alpha=0.4,
            label="equal ranking")
    ax.set_xlabel("Moran's I rank (1 = most spatially autocorrelated)")
    ax.set_ylabel("TDA H1 entropy rank (1 = most topologically complex)")
    ax.set_title("Moran's I vs TDA gene rankings\n(off-diagonal = methods disagree)")
    ax.legend()
    fig.tight_layout()

    out = _figures_dir() / "05_morans_vs_tda.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_figure: %s", out)
    return out


def generate_all_figures(
    adata: ad.AnnData,
    enrichment_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    top_svgs: list[str],
    tda_gene: str,
) -> list[Path]:
    """Generate all five benchmark figures.

    Args:
        adata: Preprocessed AnnData.
        enrichment_df: Neighbourhood enrichment matrix.
        comparison_df: Moran vs TDA comparison.
        top_svgs: Top spatially variable gene names.
        tda_gene: Gene to show the persistence diagram for.

    Returns:
        List of saved figure paths.
    """
    paths = [
        plot_spatial_clusters(adata),
        plot_svg_expression(adata, top_svgs),
        plot_neighbourhood_matrix(enrichment_df),
        plot_persistence_diagram(adata, tda_gene),
        plot_morans_vs_tda(comparison_df),
    ]
    logger.info("all_figures_generated: %d figures", len(paths))
    return paths
