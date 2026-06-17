"""Pipeline configuration.

All parameters are centralised here. Override via environment variables
prefixed SPATIAL_ or a .env file (see .env.example).
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """Spatial transcriptomics pipeline settings."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    evidence_dir: Path = Path(__file__).resolve().parent.parent / "evidence"

    # ── Dataset ────────────────────────────────────────────────────────
    # 10x Genomics Visium V1 Adult Mouse Brain
    # Downloaded via squidpy.datasets.visium_hne_adata() from scverse CDN
    dataset_name: str = "V1 Adult Mouse Brain (10x Genomics Visium)"
    dataset_filename: str = "visium_hne_adata.h5ad"

    # ── Spatial graph ──────────────────────────────────────────────────
    n_rings: int = 1          # rings of neighbours for Visium hexagonal grid
    # squidpy 1.8.x uses 'grid' (not 'visium') for regular array layouts
    coord_type: str = "grid"

    # ── Preprocessing ──────────────────────────────────────────────────
    n_top_genes: int = 3000   # highly variable genes for clustering
    n_pcs: int = 30
    n_neighbors: int = 15
    leiden_resolution: float = 0.5

    # ── SVG detection ──────────────────────────────────────────────────
    # Number of top SVGs to evaluate against the oracle marker set
    n_top_svgs: int = 200
    # Moran's I score threshold to call a gene spatially variable
    morans_i_threshold: float = 0.1

    # ── TDA (topological data analysis) ────────────────────────────────
    # Maximum edge length for Rips complex filtration
    # Set relative to typical inter-spot distance in Visium (~100 microns)
    rips_max_edge_length: float = 3.0   # in normalised spatial coordinates
    # Minimum persistence to retain a topological feature (noise filter)
    min_persistence: float = 0.1
    # Number of top genes per category for TDA comparison
    n_tda_genes: int = 20

    # ── Oracle: Allen Brain Atlas cortical layer markers ───────────────
    # Published layer-specific markers for mouse brain cortex
    # Source: Lein et al. 2007 (Allen Brain Atlas); Zeisel et al. 2018
    layer_markers: dict[str, list[str]] = {
        "Layer 1":  ["Reln", "Ndnf", "Cxcl14"],
        "Layer 2/3": ["Cux1", "Cux2", "Calb1"],
        "Layer 4":  ["Rorb", "Scnn1a", "Rspo1"],
        "Layer 5":  ["Bcl11b", "Fezf2", "Etv1"],
        "Layer 6":  ["Ntsr1", "Syt6", "Ctgf"],
        "White matter": ["Mbp", "Mog", "Plp1"],
        "Hippocampus": ["Prox1", "Dkk3", "C1ql2"],
    }

    # ── Benchmark ──────────────────────────────────────────────────────
    random_seed: int = 42
    # Minimum sensitivity (fraction of known markers detected) to pass Task 1
    min_svg_sensitivity: float = 0.3
    # Minimum fraction of neighbourhood pairs correctly enriched
    min_enrichment_accuracy: float = 0.6

    model_config = {
        "env_prefix": "SPATIAL_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = PipelineSettings()
