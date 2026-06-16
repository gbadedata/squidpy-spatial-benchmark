"""Shared test fixtures for spatial transcriptomics pipeline.

Provides synthetic AnnData objects that simulate Visium spatial
transcriptomics data with:
- Hexagonal grid spatial coordinates (matching Visium layout)
- Known spatially variable genes with controlled expression patterns
- Known non-variable housekeeping genes
- Cell-type cluster labels for neighbourhood analysis
- TDA-detectable expression patterns (ring, gradient, blob)

No network access is required for any test.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _hex_grid(n_rows: int, n_cols: int) -> np.ndarray:
    """Generate Visium-like hexagonal grid spatial coordinates.

    Returns:
        Array of shape (n_rows * n_cols, 2) with x, y coordinates.
    """
    coords = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            y = row * 0.866  # sqrt(3)/2 for hex grid
            coords.append([x, y])
    return np.array(coords)


@pytest.fixture
def synthetic_spatial_adata() -> ad.AnnData:
    """Synthetic Visium-like AnnData with known spatial gene patterns.

    Structure:
    - 120 spots on a 10x12 hexagonal grid
    - 60 genes: 20 SVGs with spatial patterns, 40 housekeeping genes
    - 4 spatial domains (cell types) corresponding to grid quadrants
    - Spatial coordinates in obsm['spatial']
    - cluster labels in obs['cluster']

    Gene categories:
    - Genes 0-4:   high-gradient SVGs (expression increases L->R)
    - Genes 5-9:   ring-pattern SVGs (high at periphery, low at centre)
    - Genes 10-14: cluster-specific SVGs (high in one quadrant only)
    - Genes 15-19: known oracle markers (from settings.layer_markers)
    - Genes 20-59: housekeeping genes (random, spatially uniform)
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = 10, 12
    n_spots = n_rows * n_cols  # 120
    n_genes = 60

    # Spatial coordinates (hexagonal grid)
    coords = _hex_grid(n_rows, n_cols)
    x = coords[:, 0]
    y = coords[:, 1]

    # Normalise to [0, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Spatial domain assignment (4 quadrants)
    domains = np.where(
        (x < 0.5) & (y < 0.5), "domain_A",
        np.where(
            (x >= 0.5) & (y < 0.5), "domain_B",
            np.where(
                (x < 0.5) & (y >= 0.5), "domain_C",
                "domain_D",
            ),
        ),
    )

    # Centre of grid for ring pattern
    cx, cy = 0.5, 0.5
    dist_from_centre = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Build expression matrix
    expr = np.zeros((n_spots, n_genes), dtype=np.float32)

    # Genes 0-4: gradient SVGs (Moran's I detects these well)
    for g in range(5):
        expr[:, g] = x * 8 + rng.poisson(1, n_spots)

    # Genes 5-9: ring SVGs (TDA detects H1; Moran's I may underdetect)
    for g in range(5, 10):
        ring_signal = np.exp(-((dist_from_centre - 0.35) ** 2) / 0.02) * 10
        expr[:, g] = ring_signal + rng.poisson(0.5, n_spots)

    # Genes 10-14: cluster-specific SVGs (domain A only)
    for g in range(10, 15):
        domain_signal = (domains == "domain_A").astype(float) * 8
        expr[:, g] = domain_signal + rng.poisson(0.5, n_spots)

    # Genes 15-19: oracle marker genes (known layer markers, named)
    for g in range(15, 20):
        # These simulate layer-specific expression in domain B
        domain_signal = (domains == "domain_B").astype(float) * 7
        expr[:, g] = domain_signal + rng.poisson(0.5, n_spots)

    # Genes 20-59: housekeeping (spatially uniform)
    expr[:, 20:] = rng.poisson(3, (n_spots, 40)).astype(np.float32)

    # Gene names: include known markers from settings
    marker_names = ["Mbp", "Mog", "Rorb", "Cux2", "Reln"]
    housekeeping_names = [f"House_{i:03d}" for i in range(40)]
    gene_names = (
        [f"Grad_{i}" for i in range(5)]
        + [f"Ring_{i}" for i in range(5)]
        + [f"Cluster_{i}" for i in range(5)]
        + marker_names
        + housekeeping_names
    )

    obs = pd.DataFrame(
        {"cluster": pd.Categorical(domains)},
        index=[f"SPOT_{i:04d}" for i in range(n_spots)],
    )

    var = pd.DataFrame(
        {
            "is_svg": [True] * 20 + [False] * 40,
            "is_marker": [False] * 15 + [True] * 5 + [False] * 40,
            "pattern": (
                ["gradient"] * 5
                + ["ring"] * 5
                + ["cluster"] * 5
                + ["marker"] * 5
                + ["housekeeping"] * 40
            ),
        },
        index=gene_names,
    )

    adata = ad.AnnData(X=sp.csr_matrix(expr), obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.uns["spatial"] = {
        "V1_Adult_Mouse_Brain": {
            "scalefactors": {"spot_diameter_fullres": 89.0},
        }
    }

    return adata
