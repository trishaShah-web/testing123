"""Mask construction for the V-JEPA 2 predictor (vjepa/world_model.py).

`apply_masks` is vendored verbatim from facebookresearch/vjepa2
src/masks/utils.py (verified 2026-07-13, MIT licensed) rather than imported
from torch.hub's cache directory — relying on `from src.masks.utils import
apply_masks` reaching into torch.hub's cached repo path is fragile (not a
stable public API, depends on torch.hub's sys.path behavior across
torch versions/OS) for a two-line gather function.

`build_temporal_split_masks` is NOT from upstream — it is this project's
own pragmatic mask construction for "single future-chunk prediction"
(ARCHITECTURE.md component 2: "start with single future-chunk prediction;
autoregressive chaining is a stretch goal, not assumed"): split the clip's
temporal blocks at the midpoint, first half = context, second half = one
target region. This is a documented decision for getting a first clip
through the pipeline, not a claim about the final rollout design — see
STATUS.md UNKNOWN ("final rollout length / whether autoregressive chaining
is used").
"""

from __future__ import annotations

import torch


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor], concat: bool = True):
    """Vendored from facebookresearch/vjepa2 src/masks/utils.py.

    x: [B, N, D]. masks: list of [B, K] index tensors (patch indices to keep).
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def build_temporal_split_masks(
    num_temporal_blocks: int,
    num_spatial_patches: int,
    batch_size: int = 1,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch-grid token layout is time-major: token_index = t *
    num_spatial_patches + spatial_index (verified against
    facebookresearch/vjepa2 src/models/vision_transformer.py
    `interpolate_pos_encoding`, 2026-07-13 — the literal PatchEmbed3D
    flatten call itself was not directly inspected, so treat this ordering
    as high-confidence, not certainty, until a real run confirms tensor
    shapes match expectations).

    Splits at the temporal midpoint: first half of temporal blocks ->
    context (masks_x), second half -> one target region (masks_y).

    num_temporal_blocks = num_frames // tubelet_size (32 for the verified
    ViT-L config: num_frames=64, tubelet_size=2).
    num_spatial_patches = (img_size // patch_size) ** 2 (256 for
    img_size=256, patch_size=16).

    Returns (masks_x, masks_y), each shape (batch_size, K) — pass each
    wrapped in a list (`[masks_x]`, `[masks_y]`) to
    VJEPAWorldModel.predict_future / vjepa2's predictor.forward.
    """
    if num_temporal_blocks < 2:
        raise ValueError(
            f"need at least 2 temporal blocks to split context/target, got {num_temporal_blocks}"
        )
    split = num_temporal_blocks // 2
    context_tokens = split * num_spatial_patches
    target_tokens = (num_temporal_blocks - split) * num_spatial_patches

    context_idx = torch.arange(0, context_tokens, device=device)
    target_idx = torch.arange(context_tokens, context_tokens + target_tokens, device=device)

    masks_x = context_idx.unsqueeze(0).repeat(batch_size, 1)
    masks_y = target_idx.unsqueeze(0).repeat(batch_size, 1)
    return masks_x, masks_y
