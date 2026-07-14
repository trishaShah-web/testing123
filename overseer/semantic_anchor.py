"""Semantic Anchor — Component 4 of 7 (ARCHITECTURE.md).

Role: cross-performer mean latent for a given action (AGENT.md DEVIATIONS
#3) — NOT a language embedding, and NOT built by the LLM. It is built from
the same action performed by other performers (NTU RGB+D, performer Q != the
clip's own performer P), with latents averaged at matched action phases.
Lives entirely in V-JEPA's own latent space; fixed once constructed for a
given rollout.

The LLM's only involvement is upstream and text-only: Job 1
(overseer/llm_overseer.py `build_target`) describes identity-free
sub-motions to help select *which* reference clips get pooled here. The LLM
never sees or produces the actual anchor tensor — this module does the
pooling and averaging in plain code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SemanticAnchor:
    """Cross-performer mean latent, phase-matched, for one action label."""

    action_label: str
    latent: torch.Tensor          # mean latent, shape matches a single predicted-latent step
    phase_length: int             # T: fixed length every reference clip was resampled to
    num_reference_clips: int      # how many other-performer clips were pooled

    @classmethod
    def from_reference_clips(
        cls,
        action_label: str,
        reference_latents: list[torch.Tensor],
        phase_length: int,
    ) -> "SemanticAnchor":
        """other-performer latents (already phase-aligned to `phase_length`,
        one tensor per reference clip) -> mean latent anchor.

        Phase alignment (resampling every clip to fixed length T) is a
        documented decision (ARCHITECTURE.md component 4: "resample every
        clip to fixed length T"); DTW-based alignment is a stretch goal, not
        assumed. This method expects alignment to have already happened —
        it only pools and averages.
        """
        if not reference_latents:
            raise ValueError(
                "reference_latents is empty — need at least one other-"
                "performer clip for the same action to build an anchor."
            )
        mismatched = [t.shape for t in reference_latents if t.shape != reference_latents[0].shape]
        if mismatched:
            raise ValueError(
                "reference_latents must all share one shape after phase "
                f"alignment to phase_length={phase_length}; got mismatched "
                f"shapes: {mismatched}"
            )
        stacked = torch.stack(reference_latents, dim=0)
        mean_latent = stacked.mean(dim=0)
        return cls(
            action_label=action_label,
            latent=mean_latent,
            phase_length=phase_length,
            num_reference_clips=len(reference_latents),
        )
