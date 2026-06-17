"""Spatially variable gene detection using Moran's I.

Computes Moran's I spatial autocorrelation for all genes using the
Visium spatial neighbourhood graph. Moran's I measures whether nearby
spots have similar expression -- a positive value indicates spatial
clustering, a value near zero indicates random spatial distribution.

The top-ranked SVGs are validated against the Allen Brain Atlas oracle
marker set in the benchmark evaluator (Phase 6). This module only
computes and ranks; validation is separate.

Moran's I formula:
    I = (N / W) * (sum_i sum_j w_ij (x_i - x_mean)(x_j - x_mean)) /
        sum_i (x_i - x_mean)^2

where N is number of spots, W is sum of all weights, w_ij is the
spatial weight between spots i and j, and x_i is the expression of
the gene at spot i.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import pandas as pd
import squidpy as sq

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SVGResult:
    """Result from spatially variable gene detection.

    Attributes:
        scores: DataFrame with one row per gene, columns:
            - morans_i: Moran's I statistic [-1, 1]
            - pvalue: p-value from permutation test
            - pvalue_adj: BH-corrected p-value
            - is_svg: True if above threshold and significant
        n_genes_tested: Total genes tested.
        n_svgs: Genes called as spatially variable.
        top_svgs: Names of top n_top_svgs genes by Moran's I score.
        marker_genes_found: Known oracle markers present in top SVGs.
        marker_genes_tested: Known oracle markers present in dataset.
    """

    scores: pd.DataFrame
    n_genes_tested: int
    n_svgs: int
    top_svgs: list[str]
    marker_genes_found: list[str]
    marker_genes_tested: list[str]


def compute_morans_i(
    adata: ad.AnnData,
    layer: str = "log_norm",
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Moran's I for all genes using the spatial graph.

    Args:
        adata: AnnData with spatial_connectivities in obsp and
            log_norm layer with normalised expression.
        layer: Which layer to use for expression values.
        genes: Optional subset of genes to test. If None, tests all.

    Returns:
        DataFrame with morans_i, pvalue, pvalue_adj columns,
        indexed by gene name, sorted descending by Moran's I.
    """
    if "spatial_connectivities" not in adata.obsp:
        raise ValueError(
            "spatial_connectivities not found in obsp. "
            "Run build_spatial_graph() before computing Moran's I."
        )
    if layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' not found. "
            "Run filter_and_normalise() to create the log_norm layer."
        )

    # Work on a view with the requested genes
    if genes is not None:
        genes_present = [g for g in genes if g in adata.var_names]
        adata_sub = adata[:, genes_present].copy()
    else:
        adata_sub = adata

    # squidpy.gr.spatial_autocorr computes Moran's I using the
    # spatial_connectivities graph in obsp
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sq.gr.spatial_autocorr(
            adata_sub,
            mode="moran",
            layer=layer,
            genes=adata_sub.var_names.tolist(),
            n_perms=100,
            corr_method="fdr_bh",
            seed=settings.random_seed,
            n_jobs=1,
            show_progress_bar=False,
        )

    # Results are stored in uns['moranI']
    if "moranI" not in adata_sub.uns:
        raise RuntimeError(
            "Moran's I results not found in uns['moranI'] after "
            "sq.gr.spatial_autocorr. Check squidpy version compatibility."
        )

    df = adata_sub.uns["moranI"].copy()

    # Standardise column names across squidpy versions
    col_map = {
        "I": "morans_i",
        "pval_norm": "pvalue",
        "pval_norm_fdr_bh": "pvalue_adj",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure we have the key columns
    if "morans_i" not in df.columns and "I" not in df.columns:
        # Try to find the score column
        score_col = [c for c in df.columns if "I" in c or "moran" in c.lower()]
        if score_col:
            df = df.rename(columns={score_col[0]: "morans_i"})

    df = df.sort_values("morans_i", ascending=False)
    logger.info(
        "morans_i_computed: %d genes, top gene=%s (I=%.4f)",
        len(df),
        df.index[0] if len(df) > 0 else "none",
        df["morans_i"].iloc[0] if len(df) > 0 else 0,
    )
    return df


def call_svgs(scores: pd.DataFrame) -> pd.DataFrame:
    """Annotate genes as spatially variable based on score and significance.

    A gene is called an SVG if:
    1. Moran's I >= settings.morans_i_threshold
    2. BH-adjusted p-value < 0.05

    Args:
        scores: DataFrame from compute_morans_i.

    Returns:
        Same DataFrame with is_svg column added.
    """
    has_pval = "pvalue_adj" in scores.columns
    if has_pval:
        scores["is_svg"] = (
            (scores["morans_i"] >= settings.morans_i_threshold)
            & (scores["pvalue_adj"] < 0.05)
        )
    else:
        scores["is_svg"] = scores["morans_i"] >= settings.morans_i_threshold

    n_svg = int(scores["is_svg"].sum())
    logger.info(
        "svg_calling: %d/%d genes called as SVGs (threshold=%.2f)",
        n_svg, len(scores), settings.morans_i_threshold,
    )
    return scores


def get_oracle_markers(adata: ad.AnnData) -> list[str]:
    """Return all Allen Brain Atlas oracle markers present in the dataset.

    Flattens settings.layer_markers and filters to genes that exist
    in adata.var_names.

    Args:
        adata: AnnData with var_names.

    Returns:
        List of marker gene names present in the dataset.
    """
    all_markers = []
    for markers in settings.layer_markers.values():
        all_markers.extend(markers)
    # Deduplicate and filter to present genes
    present = [m for m in dict.fromkeys(all_markers) if m in adata.var_names]
    logger.info(
        "oracle_markers: %d total markers, %d present in dataset",
        len(set(all_markers)), len(present),
    )
    return present


def run_svg_detection(adata: ad.AnnData) -> SVGResult:
    """Full SVG detection pipeline.

    Computes Moran's I for all genes, calls SVGs, and compares the
    top-ranked SVGs against the Allen Brain Atlas oracle marker set.

    Args:
        adata: Preprocessed AnnData with spatial graph and log_norm layer.

    Returns:
        SVGResult with scores, top SVGs, and oracle marker overlap.
    """
    scores = compute_morans_i(adata, layer="log_norm")
    scores = call_svgs(scores)

    top_svgs = list(scores.index[: settings.n_top_svgs])
    oracle_markers = get_oracle_markers(adata)
    markers_found = [m for m in oracle_markers if m in top_svgs]

    result = SVGResult(
        scores=scores,
        n_genes_tested=len(scores),
        n_svgs=int(scores["is_svg"].sum()),
        top_svgs=top_svgs,
        marker_genes_found=markers_found,
        marker_genes_tested=oracle_markers,
    )

    sensitivity = (
        len(markers_found) / len(oracle_markers) if oracle_markers else 0.0
    )
    logger.info(
        "svg_detection_complete: %d SVGs, %d/%d oracle markers in top %d "
        "(sensitivity=%.3f)",
        result.n_svgs,
        len(markers_found),
        len(oracle_markers),
        settings.n_top_svgs,
        sensitivity,
    )
    return result
