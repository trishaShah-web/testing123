"""Day-1 smoke test (STATUS.md "IN PROGRESS"): run ONE real NTU clip through
the encoder + predictor and check the prediction is at least in the right
ballpark, before trusting anything built on top of them.

This intentionally does NOT go through pipeline/inference_loop.py's
SteeringPipeline or build a real Semantic Anchor — a real anchor needs a
pool of *other-performer* reference clips (AGENT.md DEVIATIONS #3), which a
single clip cannot provide. What this script checks instead: does the
predictor's guess for the held-out second half of the clip resemble what
the encoder itself sees when it's given that footage directly? That's a
plumbing/sanity signal, not a scientific result — do not report this number
as IDS/SCS/PCS or as evidence steering works.

Known unverified assumption (the single highest-risk one in this script):
the exact tensor layout VJEPAEncoder.forward() expects. TorchCodec gives us
(T, C, H, W); this script permutes to (1, C, T, H, W) as a best guess
matching standard 3D-conv video conventions. If this script fails with a
shape error, this line is the first thing to check.

Usage:
    python scripts/smoke_test_single_clip.py /path/to/one_ntu_clip.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from data.video_dataset import VideoDataset
from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer.drift_detection import compute_drift

# Verified ViT-L / 64-frame / 256px config (vjepa/_hub.py), 2026-07-13.
NUM_FRAMES = 64
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2


def main(video_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = VideoDataset(root=".", num_frames=NUM_FRAMES, frame_stride=1)
    clip = loader._load_clip(Path(video_path))  # (T, C, H, W), values in [0, 1]
    print(f"loaded clip: {tuple(clip.shape)}")

    # UNVERIFIED: (T,C,H,W) -> (1,C,T,H,W). First thing to check on failure.
    frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    print(f"frames into encoder: {tuple(frames.shape)}")

    print("loading encoder + predictor (torch.hub, first run will download)...")
    encoder = VJEPAEncoder(device=device)
    world_model = VJEPAWorldModel(device=device)

    encoded_full_clip = encoder(frames)
    print(f"encoded_full_clip: {tuple(encoded_full_clip.shape)}")

    masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=frames.shape[0],
        device=device,
    )
    print(f"context tokens: {masks_x.shape[1]}, target tokens: {masks_y.shape[1]}")

    predicted = world_model.predict_future(encoded_full_clip, masks_x, masks_y)
    print(f"predicted (target region): {tuple(predicted.shape)}")

    # Sanity comparison only — NOT the Semantic Anchor, NOT a metric.
    real_target = apply_masks(encoded_full_clip, [masks_y])
    sanity_drift = compute_drift(predicted, real_target)
    print(f"sanity check — 1-cos(predicted, real target encoding): {sanity_drift:.4f}")
    print("(lower = predictor's guess for the future resembles the real footage; "
          "this is a plumbing check, not IDS/SCS/PCS)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
