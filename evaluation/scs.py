"""Semantic Consistency Score (SCS).

Purpose: measure whether predicted futures preserve intended action semantics.
Higher values indicate better semantic preservation.

Computed entirely on latents (AGENT.md "Evaluation metrics (all computed on
latents)") via an external, read-only linear probe predicting *action class*
from the predicted future latent; higher = action preserved through the
nudge (AGENT.md SCS definition).

Known Unknowns (STATUS.md): probe architecture; aggregation strategy;
normalization. None of these may be invented.
"""

from __future__ import annotations

from typing import Any


def compute_scs(predicted_latent_trajectory: Any, action_labels: Any) -> float:
    """predicted latent trajectory + ground-truth action labels -> SCS
    (probe accuracy).

    TODO: not specified by project definition.
    """
    raise NotImplementedError(
        "SCS implementation is TODO — probe architecture, aggregation, and "
        "normalization are not specified by project definition."
    )
