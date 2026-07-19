"""PCA Overlay — Component 7(c) of 7 (ARCHITECTURE.md).

Role: project latent tokens D -> 3 with a SHARED basis (fit once on a fixed
reference set, reused across Raw/Blind/LLM), map to RGB, tile into a patch
grid, overlay as a heatmap.

Hard constraint (INSTRUCTIONS.md): refit-per-clip makes colors non-
comparable across arms and is a bug, not a style choice — the basis must be
fit exactly once and reused everywhere. This module goes one step further
than just sharing the 3 projection axes: `PCABasis` also stores a fixed
per-component min/max (computed on the SAME fit call, from the SAME
reference set) and `project_to_rgb_overlay` always normalizes through that
fixed range. Per-image min/max normalization would silently reintroduce the
exact "non-comparable colors across arms" bug one level down (axes shared,
color scale not) — see scripts/build_pca_basis.py, which fits+saves a
PCABasis once, and scripts/visualize_steering.py, which only ever loads it.

Reference set composition (which clips feed fit_shared_pca_basis) is a
documented decision made in scripts/build_pca_basis.py's CLI, not here —
this module only does the fitting/projection math.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


@dataclass
class PCABasis:
    """A 3-component PCA basis fit once on a fixed reference set, plus the
    fixed per-component (min, max) from that same fit — both are required to
    turn a projection into a directly-comparable RGB color across arms/clips.
    """

    mean: torch.Tensor            # [D]
    components: torch.Tensor      # [3, D]
    component_min: torch.Tensor   # [3]
    component_max: torch.Tensor   # [3]

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., 3] via the fixed, already-fit basis."""
        return (latent - self.mean) @ self.components.T

    def to_rgb(self, latent: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., 3] in [0, 1], normalized through the FIXED
        component_min/component_max from the original fit — never rescaled
        per-call, so colors stay comparable across every clip and arm that
        reuses this basis.
        """
        projected = self.project(latent)
        span = (self.component_max - self.component_min).clamp(min=1e-8)
        return ((projected - self.component_min) / span).clamp(0.0, 1.0)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "components": self.components,
            "component_min": self.component_min,
            "component_max": self.component_max,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PCABasis":
        return cls(
            mean=state["mean"],
            components=state["components"],
            component_min=state["component_min"],
            component_max=state["component_max"],
        )


def fit_shared_pca_basis(reference_latents: torch.Tensor) -> PCABasis:
    """reference_latents: [M, D] — every reference token, already pooled
    across whichever clips/blocks make up the fixed reference set (see
    scripts/build_pca_basis.py). Fit a 3-component PCA basis via SVD on the
    mean-centered reference set, plus the per-component (min, max) of the
    SAME reference set's projections — call this exactly once per run and
    reuse the returned basis for every arm and every clip (hard constraint).
    """
    if reference_latents.dim() != 2:
        raise ValueError(f"reference_latents must be [M, D] (already flattened across clips/blocks/tokens), got shape {tuple(reference_latents.shape)}")
    reference_latents = reference_latents.float()
    mean = reference_latents.mean(dim=0)
    centered = reference_latents - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:3].clone()  # [3, D], top-3 principal directions

    projected = centered @ components.T  # [M, 3]
    component_min = projected.min(dim=0).values
    component_max = projected.max(dim=0).values

    return PCABasis(mean=mean, components=components, component_min=component_min, component_max=component_max)


def project_to_rgb_overlay(
    latent_blocks: torch.Tensor,
    original_frames: torch.Tensor,
    pca_basis: PCABasis,
    output_path: Path,
    alpha: float = 0.5,
) -> Path:
    """latent_blocks: [T_blocks, num_spatial_patches, D] (target-region
    tokens for ONE clip/arm, grouped by temporal block — time-major token
    order per vjepa/masking.py, so this reshape is a documented decision
    consistent with that ordering, not a new assumption).
    original_frames: [T_blocks, C, H, W] in [0, 1] — one real representative
    raw frame per temporal block (see scripts/visualize_steering.py: first
    raw frame of each tubelet, a documented decision since a tubelet covers
    tubelet_size=2 raw frames and V-JEPA tokens don't resolve single frames).
    pca_basis: already-fit, shared PCABasis — never refit here.

    -> saves ONE image per call: the T_blocks per-step heatmap-over-frame
    overlays concatenated horizontally (left = earliest target step), and
    returns its path. Caller composes Raw/Blind side by side (see
    scripts/visualize_steering.py) rather than this function knowing about
    arms at all.

    Upsampling patch grid -> pixel grid uses 'nearest': a patch is a
    discrete region with no sub-patch resolution, so a smooth interpolation
    would imply precision the representation doesn't have.
    """
    t_blocks, num_spatial_patches, _d = latent_blocks.shape
    grid_size = int(round(num_spatial_patches ** 0.5))
    if grid_size * grid_size != num_spatial_patches:
        raise ValueError(f"num_spatial_patches={num_spatial_patches} is not a perfect square (grid_size={grid_size}) — cannot reshape to a square patch grid")
    if original_frames.shape[0] != t_blocks:
        raise ValueError(f"original_frames has {original_frames.shape[0]} frames but latent_blocks has {t_blocks} temporal blocks — need exactly one representative frame per block")

    _c, h, w = original_frames.shape[1:]
    strips = []
    for t in range(t_blocks):
        rgb = pca_basis.to_rgb(latent_blocks[t])              # [num_spatial_patches, 3]
        heatmap = rgb.view(grid_size, grid_size, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, grid, grid]
        heatmap = F.interpolate(heatmap, size=(h, w), mode="nearest").squeeze(0)   # [3, H, W]
        blended = alpha * heatmap + (1.0 - alpha) * original_frames[t]
        strips.append(blended.clamp(0.0, 1.0))

    strip = torch.cat(strips, dim=-1)  # [3, H, W * T_blocks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(strip, output_path)
    return output_path
