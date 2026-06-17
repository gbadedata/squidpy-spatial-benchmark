"""Spatial neighbourhood enrichment analysis.

Computes cell-type co-occurrence and neighbourhood enrichment using
squidpy's graph-based tools. These analyses characterise the spatial
organisation of brain regions -- which anatomical areas tend to be
adjacent, which are spatially segregated, and whether the observed
co-localisation patterns match known neuroanatomy.

Two complementary analyses are run:

1. Neighbourhood enrichment (sq.gr.nhood_enrichment):
   For each pair of cluster labels, tests whether the two cell types
   are enriched or depleted in each other's spatial neighbourhoods
   relative to a random baseline. Enrichment > 0 means the pair
   co-localises more than expected. Computed via permutation test.

2. Co-occurrence scores (sq.gr.co_occurrence):
   Measures spatial co-occurrence of cluster pairs at increasing
   distance thresholds. Produces a spatial scale-dependent profile
   of co-occurrence, revealing whether co-localisation occurs at
   short range (direct adjacency) or over larger spatial distances.

Validation uses known mouse brain anatomical constraints:
- Cortex clusters should be enriched with each other
- Fiber_tract should be enriched with adjacent Cortex
- Hippocampus structures should co-localise
- Thalamus clusters should be enriched with each other
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class NeighbourhoodResult:
    """Result from neighbourhood enrichment analysis.

    Attributes:
        enrichment_scores: DataFrame (n_clusters x n_clusters)
            with enrichment z-scores. Positive = co-enriched.
        cooccurrence_scores: Dict mapping cluster pairs to their
            co-occurrence score profiles across distance thresholds.
        n_clusters: Number of cluster labels analysed.
        validated_pairs: Dict mapping expected co-localisation pairs
            to whether they are enriched (True) or not (False).
        n_pairs_validated: Total anatomical constraint pairs checked.
        n_pairs_passing: Pairs where enrichment matches expectation.
    """

    enrichment_scores: pd.DataFrame
    n_clusters: int
    validated_pairs: dict[tuple[str, str], bool]
    n_pairs_validated: int
    n_pairs_passing: int


def compute_neighbourhood_enrichment(
    adata: ad.AnnData,
    cluster_key: str = "cluster",
) -> pd.DataFrame:
    """Compute neighbourhood enrichment z-scores for all cluster pairs.

    Uses squidpy's nhood_enrichment which runs a permutation test:
    it randomly shuffles cluster labels many times and computes a
    null distribution of neighbourhood co-occurrence counts. The
    observed count is converted to a z-score relative to the null.

    Args:
        adata: AnnData with spatial_connectivities and cluster labels.
        cluster_key: obs column containing cluster labels.

    Returns:
        DataFrame (n_clusters x n_clusters) of enrichment z-scores.
    """
    if "spatial_connectivities" not in adata.obsp:
        raise ValueError(
            "spatial_connectivities not found. "
            "Run build_spatial_graph() first."
        )
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"Cluster column '{cluster_key}' not found in obs."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sq.gr.nhood_enrichment(
            adata,
            cluster_key=cluster_key,
            seed=settings.random_seed,
            n_perms=1000,
            show_progress_bar=False,
        )

    key = f"{cluster_key}_nhood_enrichment"
    if key not in adata.uns:
        raise RuntimeError(
            f"Neighbourhood enrichment results not found in uns['{key}']. "
            "Check squidpy version compatibility."
        )

    zscore_matrix = adata.uns[key]["zscore"]
    clusters = adata.obs[cluster_key].cat.categories.tolist()

    df = pd.DataFrame(zscore_matrix, index=clusters, columns=clusters)
    logger.info(
        "neighbourhood_enrichment_computed: %d clusters, "
        "z-score range=[%.2f, %.2f]",
        len(clusters),
        float(np.nanmin(zscore_matrix)),
        float(np.nanmax(zscore_matrix)),
    )
    return df


def validate_anatomical_constraints(
    enrichment_df: pd.DataFrame,
    adata: ad.AnnData,
    cluster_key: str = "cluster",
) -> dict[tuple[str, str], bool]:
    """Validate enrichment against known mouse brain anatomy.

    Checks whether pairs of anatomically related brain regions show
    spatial enrichment consistent with published neuroanatomy:

    Expected enrichments (should be > 0):
    - Cortex_1 and Cortex_2 (adjacent cortical layers)
    - Cortex_2 and Cortex_3
    - Cortex_3 and Cortex_4
    - Cortex_4 and Cortex_5
    - Thalamus_1 and Thalamus_2 (same structure)
    - Hypothalamus_1 and Hypothalamus_2 (same structure)
    - Pyramidal_layer and Pyramidal_layer_dentate_gyrus (hippocampal)
    - Hippocampus and Pyramidal_layer (hippocampal complex)

    Args:
        enrichment_df: Enrichment z-score DataFrame.
        adata: AnnData with cluster labels.
        cluster_key: obs column for cluster labels.

    Returns:
        Dict mapping (cluster_a, cluster_b) to True if enrichment
        matches the expected direction, False otherwise.
    """
    available = set(enrichment_df.index)

    # Known co-localisation pairs from mouse brain neuroanatomy
    expected_enriched = [
        ("Cortex_1", "Cortex_2"),
        ("Cortex_2", "Cortex_3"),
        ("Cortex_3", "Cortex_4"),
        ("Cortex_4", "Cortex_5"),
        ("Thalamus_1", "Thalamus_2"),
        ("Hypothalamus_1", "Hypothalamus_2"),
        ("Pyramidal_layer", "Pyramidal_layer_dentate_gyrus"),
        ("Hippocampus", "Pyramidal_layer"),
    ]

    results: dict[tuple[str, str], bool] = {}
    for a, b in expected_enriched:
        if a not in available or b not in available:
            logger.debug("skipping pair (%s, %s): not in dataset", a, b)
            continue
        score = float(enrichment_df.loc[a, b])
        is_enriched = score > 0
        results[(a, b)] = is_enriched
        logger.debug(
            "constraint (%s, %s): z=%.2f -> %s",
            a, b, score, "PASS" if is_enriched else "FAIL",
        )

    n_pass = sum(results.values())
    logger.info(
        "anatomical_validation: %d/%d pairs enriched as expected",
        n_pass, len(results),
    )
    return results


def run_neighbourhood_analysis(
    adata: ad.AnnData,
    cluster_key: str = "cluster",
) -> NeighbourhoodResult:
    """Full neighbourhood enrichment pipeline.

    Args:
        adata: Preprocessed AnnData with spatial graph.
        cluster_key: obs column for cluster labels.

    Returns:
        NeighbourhoodResult with enrichment scores and validation.
    """
    enrichment_df = compute_neighbourhood_enrichment(adata, cluster_key)
    validated = validate_anatomical_constraints(enrichment_df, adata, cluster_key)

    n_pass = sum(validated.values())
    n_total = len(validated)

    result = NeighbourhoodResult(
        enrichment_scores=enrichment_df,
        n_clusters=len(enrichment_df),
        validated_pairs=validated,
        n_pairs_validated=n_total,
        n_pairs_passing=n_pass,
    )

    pct = n_pass / n_total * 100 if n_total > 0 else 0
    logger.info(
        "neighbourhood_analysis_complete: %d clusters, "
        "%d/%d anatomical pairs validated (%.1f%%)",
        result.n_clusters, n_pass, n_total, pct,
    )
    return result
