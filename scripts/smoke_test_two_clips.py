"""Two-clip smoke test: same action, two different performers — the
smallest real test of the actual steering method (not just the raw
predictor, see scripts/smoke_test_single_clip.py).

Verified 2026-07-14: the predictor's output dimensionality defaults to the
encoder's embed_dim (facebookresearch/vjepa2 src/models/predictor.py
VisionTransformerPredictor.__init__), so a predicted future latent and a
real encoded latent from another clip live in the same space and can be
compared/arithmetic'd directly — this is what makes the anchor math below
valid, not just shape-compatible by luck.

What this does NOT do: build a real Semantic Anchor (AGENT.md DEVIATIONS
#3 anchor = MEAN over a POOL of other-performer clips; this uses exactly
one reference clip, which is a degenerate one-clip "mean"). Do not report
its output as an anchor result — it is a plumbing check that the whole
predict -> drift -> steer chain runs on two real clips without crashing,
and that steering visibly moves the latent toward the reference performer.

Usage:
    python scripts/smoke_test_two_clips.py target.avi reference.avi [--blind-alpha 0.3]

target.avi and reference.avi should be the same action (A code), different
performer (P code) — the script checks this from the filenames and warns
(does not block) if it looks wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.ntu_rgbd import parse_ntu_filename
from data.video_dataset import VideoDataset
from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer import SemanticAnchor, compute_drift, apply_steering
from arms import RawController, BlindController

NUM_FRAMES = 16
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2


def load_clip(loader: VideoDataset, path: Path, device: str) -> torch.Tensor:
    clip = loader._load_clip(path)  # (T, C, H, W)
    # UNVERIFIED: same tensor-layout guess as smoke_test_single_clip.py.
    return clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--blind-alpha", type=float, default=0.3,
        help="Smoke-test-only value, NOT the documented steering.blind_alpha decision "
             "(configs/base.yaml) — that is still an open unknown. This flag exists so "
             "you can see steering do something today, nothing more.",
    )
    args = parser.parse_args()

    target_meta = parse_ntu_filename(args.target)
    reference_meta = parse_ntu_filename(args.reference)
    if target_meta is None or reference_meta is None:
        print("WARNING: could not parse SsssCcccPpppRrrrAaaa out of one or both filenames "
              "— continuing anyway, but action/performer checks below are skipped.")
    else:
        print(f"target:    {target_meta.path.name} -> action A{target_meta.action} "
              f"({target_meta.action_label}), performer P{target_meta.performer}")
        print(f"reference: {reference_meta.path.name} -> action A{reference_meta.action} "
              f"({reference_meta.action_label}), performer P{reference_meta.performer}")
        if target_meta.action != reference_meta.action:
            print("WARNING: different action codes — this is not a valid anchor pair.")
        if target_meta.performer == reference_meta.performer:
            print("WARNING: same performer — anchor needs a *different* performer, "
                  "same action (AGENT.md DEVIATIONS #3).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = VideoDataset(root=".", num_frames=NUM_FRAMES, frame_stride=1)
    target_frames = load_clip(loader, args.target, device)
    reference_frames = load_clip(loader, args.reference, device)

    print("loading encoder + predictor (torch.hub, first run will download)...")
    encoder = VJEPAEncoder(device=device)
    world_model = VJEPAWorldModel(device=device)

    encoded_target = encoder(target_frames)
    encoded_reference = encoder(reference_frames)

    masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=1,
        device=device,
    )

    predicted_target = world_model.predict_future(encoded_target, masks_x, masks_y)
    reference_future = apply_masks(encoded_reference, [masks_y])  # real, not predicted

    action_label = target_meta.action_label if target_meta else "unknown action"
    anchor = SemanticAnchor.from_reference_clips(
        action_label=action_label,
        reference_latents=[reference_future.squeeze(0)],
        phase_length=NUM_TEMPORAL_BLOCKS - NUM_TEMPORAL_BLOCKS // 2,
    )
    # from_reference_clips averages over dim=0 of the stacked list; re-add
    # the batch dim apply_steering/compute_drift expect.
    anchor_latent = anchor.latent.unsqueeze(0)

    drift = compute_drift(predicted_target, anchor_latent)
    print(f"\ndrift d = 1 - cos(predicted, anchor) = {drift:.4f}")

    for name, arm in [("Raw", RawController()), ("Blind", BlindController(alpha=args.blind_alpha))]:
        alpha = arm.get_alpha(action_label, drift, step=0)
        steered = apply_steering(predicted_target, anchor_latent, alpha)
        moved = (steered - predicted_target).norm().item()
        print(f"{name:6s} arm: alpha={alpha:.3f}  ||steered - predicted|| = {moved:.4f}")

    print("\n(Raw should show ||steered-predicted||=0 by construction, alpha=0. "
          "Blind should show a nonzero move toward the reference performer's real "
          "future encoding. This is a plumbing check, not a scientific result.)")


if __name__ == "__main__":
    main()
