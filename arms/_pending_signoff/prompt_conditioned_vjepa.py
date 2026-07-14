"""Baseline 3: Prompt Conditioned V-JEPA.

Conditions the frozen world model's rollout on the intended action via some
form of prompt/text conditioning, WITHOUT the Overseer's iterative monitor-
detect-correct loop — this is the ablation-style baseline that isolates the
value of active steering vs. static conditioning.

TODO: not specified by project definition — the exact conditioning
mechanism (cross-attention injection, latent concatenation, text-embedding
prefix, etc.) is a Known Unknown and must not be invented here.
"""

from __future__ import annotations

import torch

from vjepa import VJEPAEncoder, VJEPAWorldModel, VJEPADecoder


class PromptConditionedVJEPABaseline:
    def __init__(self, encoder: VJEPAEncoder, world_model: VJEPAWorldModel, decoder: VJEPADecoder):
        self.encoder = encoder
        self.world_model = world_model
        self.decoder = decoder

    @torch.no_grad()
    def run(self, observed_frames: torch.Tensor, intended_action: str, rollout_steps: int) -> torch.Tensor:
        raise NotImplementedError(
            "Prompt conditioning mechanism is TODO — not specified by "
            "project definition."
        )
