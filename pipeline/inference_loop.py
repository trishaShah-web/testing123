"""Wires the full data flow (ARCHITECTURE.md "Data Flow") end-to-end.

    Input Video (WHOLE fixed-length clip, real pixels throughout)
       -> TorchCodec (decode mp4 -> frames)         [data/video_dataset.py]
       -> Frozen V-JEPA Encoder                     -> latent tokens for the full clip
       -> temporal-split mask (context vs one future target region) [vjepa/masking.py]
       -> Frozen V-JEPA World Model (predict)        -> predicted future latent tokens
       -> SEMANTIC STEERING:
            d = compute_drift(predicted, anchor)                  [overseer/drift_detection.py]
            alpha = arm.get_alpha(action_label, d, step=0)         [arms/*.py]
            steered = apply_steering(predicted, anchor, alpha)     [overseer/guidance_vector.py]
       -> steered future latent tokens, cached ONCE per clip per arm
       -> fork into three consumers (reusing the same cached trajectory):
            (a) metrics: IDS / SCS / PCS probes on latents  [evaluation/*.py]
            (b) NN retrieval -> mp4, labeled retrieval       [visualization/nn_retrieval.py]
            (c) shared-basis PCA overlay -> heatmap mp4      [visualization/pca_overlay.py]

Only single future-chunk prediction (ARCHITECTURE.md component 2 default)
is implemented — one predictor call, one target region, `step` is always 0.
Autoregressive chaining (rollout_steps > 1) is an explicit stretch goal with
an unresolved embedding-space question (predicted latents live in the
predictor's output space, not verified to match the encoder's input space —
see vjepa/world_model.py predict_future) and is intentionally not
implemented or approximated here.

The Semantic Anchor is built once, before this call, from a pool of
other-performer reference clips (overseer/semantic_anchor.py) — it is not
recomputed here. Its `.latent` must be the same shape as this pipeline's
predicted target-region latents (same temporal split applied to each
reference clip) for compute_drift's cosine similarity to be well-defined.
There is no decoder anywhere in this path (ARCHITECTURE.md: "No pixel
decoder is trained or added").

This module only wires components together; it must not contain any
scientific logic itself (no similarity functions, no thresholds beyond what
the arm/controller already decided) — that logic lives in overseer/*.py and
arms/*.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import build_temporal_split_masks
from overseer import SemanticAnchor, compute_drift, apply_steering
from arms import AlphaController

# Verified ViT-L / 64-frame / 256px grid dims (vjepa/_hub.py,
# facebookresearch/vjepa2 src/hub/backbones.py defaults: num_frames=64,
# tubelet_size=2, patch_size=16, img_size=256), 2026-07-13.
_NUM_TEMPORAL_BLOCKS = 64 // 2
_NUM_SPATIAL_PATCHES = (256 // 16) ** 2


@dataclass
class SteeredRollout:
    """Cached output of one clip through one arm — the single source every
    downstream consumer (metrics, NN retrieval, PCA overlay) reads from, so
    the rollout is never recomputed per branch (INSTRUCTIONS.md "Compute the
    rollout once").
    """
    steered_latents: torch.Tensor   # predicted-target-region latents, post-steering
    drift_per_step: list[float]     # length 1 until autoregressive chaining exists
    alpha_per_step: list[float]


@dataclass
class SteeringPipeline:
    encoder: VJEPAEncoder
    world_model: VJEPAWorldModel
    arm: AlphaController

    @torch.no_grad()
    def run(
        self,
        full_clip_frames: torch.Tensor,
        action_label: str,
        anchor: SemanticAnchor,
        rollout_steps: int = 1,
    ) -> SteeredRollout:
        """full_clip_frames: the WHOLE fixed-length clip (64 real frames for
        the verified ViT-L config), not just an "observed prefix" — the
        encoder is run over the full clip and the future/target region is
        carved out afterward via masking, not by withholding frames from
        the encoder (see vjepa/world_model.py predict_future docstring).
        """
        if rollout_steps != 1:
            raise NotImplementedError(
                "rollout_steps > 1 requires autoregressive chaining, which "
                "is a stretch goal with an unresolved design question — see "
                "vjepa/world_model.py predict_future docstring. Only "
                "rollout_steps=1 (single future-chunk) is implemented."
            )

        encoded_full_clip = self.encoder(full_clip_frames)
        masks_x, masks_y = build_temporal_split_masks(
            num_temporal_blocks=_NUM_TEMPORAL_BLOCKS,
            num_spatial_patches=_NUM_SPATIAL_PATCHES,
            batch_size=full_clip_frames.shape[0],
            device=full_clip_frames.device,
        )
        predicted = self.world_model.predict_future(encoded_full_clip, masks_x, masks_y)

        drift = compute_drift(predicted, anchor.latent)
        alpha = self.arm.get_alpha(action_label, drift, step=0)
        steered = apply_steering(predicted, anchor.latent, alpha)

        return SteeredRollout(
            steered_latents=steered,
            drift_per_step=[drift],
            alpha_per_step=[alpha],
        )
