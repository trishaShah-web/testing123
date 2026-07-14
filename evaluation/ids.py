"""Identity Drift Score (IDS).

Purpose: measure the degree to which predictions depend on performer-specific
appearance information (clothing, gender, identity, background, viewpoint)
rather than action semantics. Lower values indicate less forecast drift.

Computed entirely on latents (AGENT.md "Evaluation metrics (all computed on
latents)") via an external, read-only linear probe predicting *performer*
from the predicted future latent; report top-1 accuracy vs. chance. The
probe is external and read-only — V-JEPA stays frozen; this is not a
violation of "no training" (AGENT.md IDS definition).

Known Unknowns (STATUS.md): probe architecture; aggregation strategy;
normalization.
"""

from __future__ import annotations

from typing import Any


def compute_ids(predicted_latent_trajectory: Any, performer_labels: Any) -> float:
    """predicted latent trajectory + ground-truth performer labels -> IDS
    (probe top-1 accuracy vs. chance).

    TODO: not specified by project definition.
    """
    raise NotImplementedError(
        "IDS implementation is TODO — probe architecture, aggregation, and "
        "normalization are not specified by project definition."
    )
