"""Guidance Vector — Component 6 of 7 (ARCHITECTURE.md).

Role: pure latent arithmetic, nothing else. This is a documented decision
now, not a Known Unknown:

    g = alpha * (anchor - predicted)
    steered = predicted + g

Never modifies model parameters, never retrains V-JEPA, never edits any
output artifact — this is the only place the correction is actually applied,
and it is one line of tensor arithmetic.
"""

from __future__ import annotations

import torch


def guidance_vector(predicted_latent: torch.Tensor, anchor_latent: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha * (anchor - predicted)."""
    return alpha * (anchor_latent - predicted_latent)


def apply_steering(predicted_latent: torch.Tensor, anchor_latent: torch.Tensor, alpha: float) -> torch.Tensor:
    """predicted + guidance_vector(predicted, anchor, alpha). alpha=0 (Raw
    arm) returns predicted unchanged, by construction.
    """
    return predicted_latent + guidance_vector(predicted_latent, anchor_latent, alpha)
