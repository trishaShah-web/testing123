"""PCA Overlay — Component 7(c) of 7 (ARCHITECTURE.md).

Role: project latent tokens D -> 3 with a SHARED basis (fit once on a fixed
reference set, reused across Raw/Blind/LLM), map to RGB, tile into a patch
grid, overlay as a heatmap mp4.

Hard constraint (INSTRUCTIONS.md): refit-per-clip makes colors non-
comparable across arms and is a bug, not a style choice — the basis must be
fit exactly once and reused everywhere.

Known Unknowns: which fixed reference set the basis is fit on, and the
patch-grid layout — not decided here (see STATUS.md UNKNOWN;
configs/base.yaml `visualization.pca_reference_set`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def fit_shared_pca_basis(reference_latents: torch.Tensor) -> Any:
    """Fit a 3-component PCA basis ONCE on a fixed reference set. Call this
    exactly once per run and reuse the returned basis for every arm and
    every clip — never refit per clip (hard constraint).

    TODO: not specified by project definition — reference set composition
    is not yet chosen.
    """
    raise NotImplementedError(
        "Shared PCA basis fitting is TODO — reference set is not specified "
        "by project definition."
    )


def project_to_rgb_overlay(
    latent_trajectory: torch.Tensor,
    pca_basis: Any,
    output_path: Path,
) -> Path:
    """latent trajectory + shared PCA basis -> RGB patch-grid heatmap mp4.

    TODO: not specified by project definition — patch-grid layout and
    color-mapping normalization are not yet chosen.
    """
    raise NotImplementedError(
        "PCA overlay rendering is TODO — patch-grid layout and color "
        "normalization are not specified by project definition."
    )
