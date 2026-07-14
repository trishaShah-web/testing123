"""Shared, cached entrypoint into the official V-JEPA 2 repo.

Verified against primary source, 2026-07-13 (not yet execution-tested — see
STATUS.md Day-1 smoke test):
  - facebookresearch/vjepa2 hubconf.py -> src/hub/backbones.py
  - `vjepa2_vit_large(pretrained=True)` builds encoder + predictor via
    `_make_vjepa2_model(model_name="vit_large", ...)`, which downloads ONE
    checkpoint file and loads `state_dict["target_encoder"]` into the
    encoder and `state_dict["predictor"]` into the predictor. The predictor
    is the standard masked-future predictor (`use_mask_tokens=True`), NOT
    the action-conditioned predictor — `vjepa2_ac_vit_giant` is a different,
    action-conditioned Giant model and is not used here.
  - This is the ViT-L / 64 frames-per-clip / 256px variant, i.e. the same
    underlying weights as the HuggingFace `facebook/vjepa2-vitl-fpc64-256`
    checkpoint referenced in ARCHITECTURE.md component 1 — but loaded via
    torch.hub rather than `transformers.AutoModel`, because the HF
    `AutoModel` path only exposes encoder features and has no documented
    way to retrieve the predictor (world model), which component 2 needs.

Requires `timm` and `einops` (hubconf.py `dependencies`), not just `torch`.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def load_vjepa2_vitl(pretrained: bool = True):
    """-> (encoder, predictor), both frozen nn.Modules, from one shared
    download/build so VJEPAEncoder and VJEPAWorldModel don't each trigger
    their own copy of the checkpoint fetch.
    """
    return torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large", pretrained=pretrained)
