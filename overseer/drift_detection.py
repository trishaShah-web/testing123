"""Drift Detection — Component 5 of 7 (ARCHITECTURE.md).

Role: produce the scalar drift signal handed to the controller. This is a
documented decision now, not a Known Unknown:

    d = 1 - cos(predicted_latent, anchor_latent)

One number per step. Threshold / trigger behavior (whether `d` is large
enough to act on) lives in the controller (arms/*.py, overseer/llm_overseer.py
Job 2), not here — this module only computes the number.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_drift(predicted_latent: torch.Tensor, anchor_latent: torch.Tensor) -> float:
    """predicted latent + anchor latent -> scalar drift d = 1 - cos(pred, anchor).

    Both tensors are flattened before the cosine similarity so this works
    whether the latent is a single token, a token sequence, or already
    pooled — matching whatever shape SemanticAnchor.latent and a single
    predicted-step latent share.
    """
    pred_flat = predicted_latent.reshape(-1)
    anchor_flat = anchor_latent.reshape(-1)
    cos_sim = F.cosine_similarity(pred_flat, anchor_flat, dim=0)
    return (1.0 - cos_sim).item()
