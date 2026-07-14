"""Frozen V-JEPA World Model — Component 2 of 7 (ARCHITECTURE.md).

Role: predict future latent tokens from observed context, via masked
prediction (not literal autoregressive re-encoding — see predict_future
docstring).

Frozen throughout the entire project — this module never updates its own
parameters. (arms/_pending_signoff/lora_finetuned_vjepa.py trains a
*separate* copy of the world model for comparison purposes only and is
dropped pending sign-off, AGENT.md DEVIATIONS #4; it does not affect this
class either way.)

Loading API verified against primary source (see vjepa/_hub.py docstring),
2026-07-13. This is the standard masked-future predictor
(`use_mask_tokens=True`), NOT the action-conditioned predictor that ships
only with the separate Giant model (`vjepa2_ac_vit_giant`) — using that one
instead would silently swap in a different, action-conditioned architecture
and is not what this project calls for.

predict_future's forward call is verified against
facebookresearch/vjepa2 src/models/predictor.py
(`forward(x, masks_x, masks_y, mask_index=1, has_cls=False)`) and
src/masks/utils.py (`apply_masks`), both 2026-07-13. Loading AND the
forward call are still **not execution-tested** (STATUS.md Day-1 smoke
test) — source-verified is not the same as confirmed-working.

Known Unknowns (AGENT.md/STATUS.md): final rollout length; whether
autoregressive chaining is used at all (the default per ARCHITECTURE.md is
single future-chunk prediction, which is what predict_future implements —
chaining is an explicit stretch goal, not assumed here or anywhere else in
this module).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._hub import load_vjepa2_vitl
from .masking import apply_masks


class VJEPAWorldModel(nn.Module):
    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._load_frozen_model(checkpoint_path)

    def _load_frozen_model(self, checkpoint_path: str | None) -> nn.Module:
        # checkpoint_path is currently informational only — see
        # vjepa/encoder.py's identical note; both encoder and predictor come
        # from the same verified torch.hub entrypoint / checkpoint file.
        _encoder, predictor = load_vjepa2_vitl(pretrained=True)
        return predictor.to(self.device)

    @torch.no_grad()
    def predict_future(
        self,
        encoded_full_clip: torch.Tensor,
        masks_x: torch.Tensor,
        masks_y: torch.Tensor,
    ) -> torch.Tensor:
        """encoded_full_clip: encoder output over the WHOLE fixed-length
        clip (real pixels throughout — the encoder does not itself take
        "only observed frames", see vjepa/encoder.py), shape [B, N, D].
        masks_x / masks_y: (B, K) index tensors from
        vjepa/masking.build_temporal_split_masks — which patch positions are
        "observed context" vs the single "future" target region.

        -> predicted latents at the target positions only, shape
        [B, len(masks_y), predictor_out_embed_dim].

        This is ONE predictor call for ONE target region — not a per-step
        autoregressive loop. Feeding this method's own output back in as
        new context for another call is an unresolved design question: the
        output lives in the predictor's embedding space
        (`predictor_embed_dim`/`out_embed_dim`), which is not guaranteed to
        match the encoder's embedding space (`embed_dim`) that masks_x
        selects from — so it cannot be assumed to be valid input here
        without an explicit, verified re-projection. Do not add
        autoregressive chaining on top of this method without resolving
        that first (see STATUS.md).
        """
        self.model.eval()
        context = apply_masks(encoded_full_clip, [masks_x])
        return self.model(context, [masks_x], [masks_y])
