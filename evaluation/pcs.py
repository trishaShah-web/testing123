"""Physics Consistency Score (PCS).

Purpose: measure whether semantic steering preserves physically plausible
dynamics and realistic future trajectories. Higher values indicate better
physical consistency.

Computed entirely on latents (AGENT.md "Evaluation metrics (all computed on
latents)"): compare step-to-step latent deltas of the steered trajectory
against real clips' deltas; large deviation = off-manifold / broken motion
(AGENT.md PCS definition). There is no decoder, so this never touches pixels.

Known Unknowns (STATUS.md): delta comparison method (e.g. norm vs.
distributional distance); aggregation strategy; normalization.
"""

from __future__ import annotations

from typing import Any


def compute_pcs(steered_latent_trajectory: Any, real_latent_trajectory: Any) -> float:
    """steered latent trajectory + real (observed, non-predicted) latent
    trajectory for the same clip -> PCS, from comparing step-to-step latent
    deltas between the two.

    TODO: not specified by project definition.
    """
    raise NotImplementedError(
        "PCS implementation is TODO — delta comparison method, aggregation, "
        "and normalization are not specified by project definition."
    )
