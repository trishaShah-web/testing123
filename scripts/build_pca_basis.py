"""Build a shared PCA basis for the PCA overlay visualization
(visualization/pca_overlay.py) — the "fit ONCE on a fixed reference set"
step INSTRUCTIONS.md requires before any Raw/Blind/LLM overlay is rendered.
Run this once, reuse pca_basis.pt for every clip and every arm afterwards
(scripts/visualize_steering.py only ever loads it, never refits).

Reference set (documented decision, approved 2026-07-19): the real
target-region latents of a small, explicit pool of clips — either passed
directly (--clips) or reused from an already-built Semantic Anchor's
`reference_paths` (--from-anchor), so building the basis doesn't require
curating yet another separate pool. No default/implicit pool is invented —
one of the two must be passed.

Same code path as scripts/build_semantic_anchor.py for getting a real
target-region latent per clip (NUM_FRAMES=16, deterministic sampling,
encoder + apply_masks(masks_y)) — reused here, not reinvented.

Usage:
    python scripts/build_pca_basis.py /path/to/ntu_root \
        --clips /path/to/ntu_root/S004C002P007R001A015_rgb.avi \
                /path/to/ntu_root/S004C002P008R001A015_rgb.avi \
        --output pca_basis.pt

    python scripts/build_pca_basis.py /path/to/ntu_root \
        --from-anchor anchor_A015.pt --output pca_basis.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.ntu_rgbd import NTURGBDDataset
from vjepa import VJEPAEncoder
from vjepa.masking import apply_masks, build_temporal_split_masks
from visualization.pca_overlay import fit_shared_pca_basis

NUM_FRAMES = 16  # = steering.phase_length (configs/base.yaml); matches every other script in this repo
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntu_root", type=Path, help="root directory of NTU RGB+D clips")
    parser.add_argument("--clips", type=Path, nargs="+", default=None, help="explicit clip paths to fit the PCA reference set on")
    parser.add_argument("--from-anchor", type=Path, default=None, help="reuse an existing anchor .pt's reference_paths as the reference set instead of --clips")
    parser.add_argument("--output", type=Path, required=True, help="path to save the fitted basis via torch.save")
    args = parser.parse_args()

    if args.clips is None and args.from_anchor is None:
        raise SystemExit("must pass either --clips (explicit paths) or --from-anchor (reuse an anchor's reference_paths)")

    clip_paths = list(args.clips) if args.clips else []
    if args.from_anchor is not None:
        anchor_data = torch.load(args.from_anchor, weights_only=False)
        if "reference_paths" not in anchor_data:
            raise SystemExit(f"{args.from_anchor} has no 'reference_paths' (built before that field existed) — pass --clips explicitly instead")
        clip_paths += [Path(p) for p in anchor_data["reference_paths"]]
        print(f"reusing {len(anchor_data['reference_paths'])} reference clip(s) from {args.from_anchor} as the PCA reference set")

    if not clip_paths:
        raise SystemExit("reference set is empty — nothing to fit the PCA basis on")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"PCA reference set: {len(clip_paths)} clip(s)")

    dataset = NTURGBDDataset(root=args.ntu_root, num_frames=NUM_FRAMES, deterministic=True)
    encoder = VJEPAEncoder(device=device)
    _masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=1,
        device=device,
    )

    all_tokens = []
    for path in clip_paths:
        clip = dataset._load_clip(path)  # (T, C, H, W), deterministic full-span sampling
        frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as other scripts
        with torch.no_grad():
            encoded = encoder(frames)
            real_target = apply_masks(encoded, [masks_y]).squeeze(0).cpu()  # [target_tokens, D]
        all_tokens.append(real_target)
        print(f"  encoded {path.name} ({real_target.shape[0]} target tokens)")
        del clip, frames, encoded
        if device == "cuda":
            torch.cuda.empty_cache()

    reference_latents = torch.cat(all_tokens, dim=0)  # [sum(target_tokens across clips), D]
    print(f"fitting PCA basis on {reference_latents.shape[0]} tokens, D={reference_latents.shape[1]}...")
    basis = fit_shared_pca_basis(reference_latents)

    torch.save(
        {
            **basis.state_dict(),
            "reference_clip_paths": [str(p) for p in clip_paths],
            "num_reference_tokens": reference_latents.shape[0],
        },
        args.output,
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
