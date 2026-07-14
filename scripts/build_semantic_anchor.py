"""Build a REAL Semantic Anchor from a pool of other-performer NTU clips —
the step after the two-clip plumbing check (scripts/smoke_test_two_clips.py),
per HANDOFF.md "Where things stand": (1) two-clip smoke test confirmed
working, (2) THIS SCRIPT — real pooled anchor, (3) Raw vs. Blind on that
real anchor (Day-2 spike gate), (4) only then the LLM.

What smoke_test_two_clips.py did NOT do: it used exactly one reference clip
as a degenerate one-clip "mean" (AGENT.md DEVIATIONS #3 requires a POOL).
This script pools every other-performer clip for a given action (via
data/ntu_rgbd.py NTURGBDDataset.clips_for_action) and averages their real
target-region latents into one SemanticAnchor.

Phase alignment (configs/base.yaml steering.phase_length=16, decided
2026-07-14): every reference clip is loaded with
VideoDataset(deterministic=True) — uniform full-span index subsampling, NOT
the random-crop default — so frame i/16 lands at the same fraction of each
clip's real duration across performers. Random start offsets would average
unaligned action phases together, which would make the anchor scientifically
meaningless. See data/video_dataset.py for the deterministic-mode
implementation.

phase_length=16 is the RAW FRAME count every clip (target and reference
alike) is resampled to — matches SemanticAnchor.phase_length's own
docstring ("T: fixed length every reference clip was resampled to",
overseer/semantic_anchor.py) and scripts/smoke_test_two_clips.py's
NUM_FRAMES=16. This is a deviation from the vjepa2-vitl-fpc64-256
checkpoint's native 64-frame training config (AGENT.md DEVIATIONS #6) —
whether the frozen encoder/predictor behaves correctly on a 16-frame input
(different temporal-block count, different positional-embedding
interpolation) has NOT been re-verified against source the way the
64-frame case was (vjepa/masking.py). Kept because
scripts/smoke_test_two_clips.py ran successfully with it on Kaggle
2026-07-14, but treat this as a real open technical risk, not a settled fact.

Camera view: pooling across camera angles would mix "viewpoint" variation
into the anchor alongside performer identity, which the anchor is supposed
to isolate. This script therefore restricts the reference pool to the
TARGET clip's own camera by default (`NTURGBDDataset.clips_for_action`'s
`camera` filter) — pass `--all-cameras` to opt out.

Usage:
    python scripts/build_semantic_anchor.py /path/to/ntu_root \
        --target-clip /path/to/ntu_root/S004C002P003R001A013_rgb.avi \
        --output anchor_A013.pt

    (infers action=13, exclude_performer=3, camera=2 straight from the
    target clip's filename via data.ntu_rgbd.parse_ntu_filename. Or specify
    --action/--exclude-performer/--camera explicitly instead of
    --target-clip. Run --help for all args.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.ntu_rgbd import NTURGBDDataset, parse_ntu_filename
from vjepa import VJEPAEncoder
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer import SemanticAnchor

NUM_FRAMES = 16  # = steering.phase_length (configs/base.yaml) = T. Matches
                 # scripts/smoke_test_two_clips.py's NUM_FRAMES; see module
                 # docstring — deviates from the checkpoint's native 64-frame
                 # (fpc64) config, AGENT.md DEVIATIONS #6.
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2
PHASE_LENGTH = NUM_FRAMES  # T itself — the raw resample length, per SemanticAnchor.phase_length's docstring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntu_root", type=Path, help="root directory of NTU RGB+D clips")
    parser.add_argument(
        "--target-clip", type=Path, default=None,
        help="path to the clip you intend to steer — infers action/exclude-performer/camera "
             "from its filename (SsssCcccPpppRrrrAaaa). Overrides --action/--exclude-performer/--camera.",
    )
    parser.add_argument("--action", type=int, default=None, help="NTU action code, e.g. 13 for A013")
    parser.add_argument(
        "--exclude-performer", type=int, default=None,
        help="performer ID to exclude from pooling (the TARGET clip's own performer P) — "
             "AGENT.md DEVIATIONS #3 requires other performers Q != P",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="restrict the reference pool to one camera view (recommended — see module docstring)",
    )
    parser.add_argument(
        "--all-cameras", action="store_true",
        help="opt out of the camera restriction and pool across all camera views (not recommended: "
             "mixes viewpoint variation into the anchor alongside performer identity)",
    )
    parser.add_argument("--output", type=Path, default=None, help="path to save the anchor via torch.save")
    args = parser.parse_args()

    action, exclude_performer, camera = args.action, args.exclude_performer, args.camera
    if args.target_clip is not None:
        meta = parse_ntu_filename(args.target_clip)
        if meta is None:
            raise SystemExit(f"--target-clip {args.target_clip} does not match SsssCcccPpppRrrrAaaa")
        action, exclude_performer, camera = meta.action, meta.performer, meta.camera
        print(f"inferred from target clip: action={action} ({meta.action_label}), "
              f"exclude_performer={exclude_performer}, camera={camera}")
    if args.all_cameras:
        camera = None
    if action is None:
        raise SystemExit("must pass either --target-clip or --action")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = NTURGBDDataset(root=args.ntu_root, num_frames=NUM_FRAMES, deterministic=True)
    records = dataset.clips_for_action(action, exclude_performer=exclude_performer, camera=camera)
    if not records:
        raise SystemExit(
            f"no other-performer clips found for action {action} "
            f"(exclude_performer={exclude_performer}, camera={camera}) under {args.ntu_root}"
        )
    print(f"pooling {len(records)} reference clip(s) for action {action} "
          f"({records[0].action_label}), excluding performer {exclude_performer}, camera={camera}")

    print("loading encoder + predictor (torch.hub, first run will download)...")
    encoder = VJEPAEncoder(device=device)
    # No VJEPAWorldModel here — the anchor pools REAL target-region
    # encodings, not predictions (see smoke_test_two_clips.py: "real, not
    # predicted").

    masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=1,
        device=device,
    )

    reference_latents = []
    for record in records:
        clip = dataset._load_clip(record.path)  # (T, C, H, W), deterministic full-span sampling
        frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as smoke tests
        with torch.no_grad():
            encoded = encoder(frames)
            real_target = apply_masks(encoded, [masks_y]).squeeze(0).cpu()  # real, not predicted
        reference_latents.append(real_target)
        print(f"  encoded {record.path.name} (performer {record.performer})")
        del clip, frames, encoded
        if device == "cuda":
            torch.cuda.empty_cache()  # one clip's activations at a time — we've OOM'd on this loop before

    anchor = SemanticAnchor.from_reference_clips(
        action_label=records[0].action_label,
        reference_latents=reference_latents,
        phase_length=PHASE_LENGTH,
    )
    print(f"\nanchor built: action={anchor.action_label!r}, "
          f"num_reference_clips={anchor.num_reference_clips}, "
          f"phase_length={anchor.phase_length}, latent shape={tuple(anchor.latent.shape)}")

    if args.output is not None:
        torch.save(
            {
                "action_label": anchor.action_label,
                "latent": anchor.latent,
                "phase_length": anchor.phase_length,
                "num_reference_clips": anchor.num_reference_clips,
                # provenance, for downstream leakage exclusion (e.g.
                # scripts/spike_blind_vs_raw.py probe training must not
                # train on the exact clips that fed this anchor):
                "reference_paths": [str(r.path) for r in records],
                "action": action,
                "camera": camera,
            },
            args.output,
        )
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
