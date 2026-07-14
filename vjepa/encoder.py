"""Frozen V-JEPA Encoder — Component 1 of 7 (ARCHITECTURE.md).

Role: observed video frames -> latent context.
Frozen throughout the entire project. Never trained, never fine-tuned.

Loading API verified against primary source (see vjepa/_hub.py docstring),
2026-07-13 — not yet execution-tested (STATUS.md Day-1 smoke test). Uses
ViT-L, 64 frames-per-clip, 256px (ARCHITECTURE.md component 1 decision),
same weights as `facebook/vjepa2-vitl-fpc64-256` on HuggingFace.

Latent token dimensionality and exact output shape are Known Unknowns per
AGENT.md / ARCHITECTURE.md and are determined entirely by the underlying
pretrained checkpoint — not chosen here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._hub import load_vjepa2_vitl


class VJEPAEncoder(nn.Module):
    """Thin frozen wrapper around the V-JEPA 2 ViT-L encoder."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._load_frozen_model(checkpoint_path)

    def _load_frozen_model(self, checkpoint_path: str | None) -> nn.Module:
        # checkpoint_path is currently informational only — the verified
        # loading path is the official torch.hub entrypoint (pretrained
        # weights resolved automatically), not an arbitrary local file. A
        # local-checkpoint override is an explicit TODO, not invented here.
        encoder, _predictor = load_vjepa2_vitl(pretrained=True)
        return encoder.to(self.device)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """observed frames -> latent context. Encoder is always in eval mode; no grad."""
        self.model.eval()
        return self.model(frames)
