"""Day-2-adjacent visualization: for ONE target clip, render Raw-vs-Blind
comparisons using the SAME encode/predict/steer code path
scripts/spike_blind_vs_raw.py already uses (inlined here, not routed through
pipeline/inference_loop.py's SteeringPipeline — that class still hardcodes
the pre-DEVIATIONS-#6 64-frame mask dims and would break on the real
16-frame config; see HANDOFF.md, flagged separately, not fixed here).

Two outputs, both purely latent-based (no pixel decoder anywhere):

1. PCA overlay (primary): predicted (Raw) and steered (Blind) target-region
   latents projected through a SHARED PCA basis (scripts/build_pca_basis.py,
   loaded here, never refit) -> RGB -> patch grid -> upsampled -> alpha-
   blended over the clip's own real frames -> saved per arm + one combined
   Raw-vs-Blind image.
2. NN retrieval (secondary, cheaper): nearest real (clip, temporal-block)
   pair from the anchor's own reference clips, by cosine distance — labeled
   "retrieval" everywhere (filenames, prints, side-by-side images). No
   frame is ever generated, only retrieved from real footage.

Kaggle-runnable: one clip at a time, GPU cache cleared between every encode
call (same discipline as build_semantic_anchor.py / spike_blind_vs_raw.py —
we've OOM'd on this before). Outputs default to
/kaggle/working/checkpoints/viz/<target-clip-stem>/.

Usage:
    python scripts/visualize_steering.py /path/to/ntu_root \
        --target-clip /path/to/ntu_root/S004C002P003R001A015_rgb.avi \
        --anchor anchor_A015.pt \
        --pca-basis pca_basis.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torchvision.io import read_image
from torchvision.utils import save_image

from data.ntu_rgbd import NTURGBDDataset, parse_ntu_filename
from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer import SemanticAnchor, compute_drift, apply_steering
from arms import RawController, BlindController
from visualization.pca_overlay import PCABasis, project_to_rgb_overlay
from visualization.nn_retrieval import ReferenceBankEntry, retrieve_nearest_real_frames, stitch_to_mp4, save_frame_as_image

NUM_FRAMES = 16  # = steering.phase_length (configs/base.yaml); matches every other script in this repo
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2
TARGET_BLOCK_OFFSET = NUM_TEMPORAL_BLOCKS // 2                    # first target-region block, absolute index
NUM_TARGET_BLOCKS = NUM_TEMPORAL_BLOCKS - TARGET_BLOCK_OFFSET     # 4 for the current 16-frame config
TUBELET_SIZE = 2

REPO_ROOT = Path(__file__).resolve().parent.parent


def clear_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def load_blind_alpha() -> float:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        config = yaml.safe_load(f)
    alpha = config["steering"]["blind_alpha"]
    if alpha is None:
        raise SystemExit("steering.blind_alpha is null in configs/base.yaml — set it before running.")
    return alpha


def read_image_float(path: Path) -> torch.Tensor:
    return read_image(str(path)).float() / 255.0


def representative_frames(clip: torch.Tensor, block_offset: int, num_blocks: int) -> torch.Tensor:
    """clip: [T, C, H, W] (full clip, real pixels) -> [num_blocks, C, H, W],
    one representative real frame per temporal block starting at
    `block_offset` — the FIRST raw frame of each tubelet (documented
    decision: a V-JEPA token covers TUBELET_SIZE=2 raw frames, so there is
    no single "the" frame for a block; picking the first is simple and
    consistent everywhere it's used in this script).
    """
    idx = [(block_offset + t) * TUBELET_SIZE for t in range(num_blocks)]
    return clip[idx]


def encode_predict(dataset, path: Path, encoder: VJEPAEncoder, world_model: VJEPAWorldModel, masks_x, masks_y, device: str):
    """-> (predicted_target_latent [1, target_tokens, D] on CPU, full clip
    frames [T, C, H, W] on CPU). Same call sequence as
    scripts/spike_blind_vs_raw.py's target-clip block.
    """
    clip = dataset._load_clip(path)
    frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as other scripts
    with torch.no_grad():
        encoded = encoder(frames)
        predicted = world_model.predict_future(encoded, masks_x, masks_y).cpu()
    del frames, encoded
    clear_gpu(device)
    return predicted, clip


def encode_real_target(dataset, path: Path, encoder: VJEPAEncoder, masks_y, device: str):
    """-> (real target-region latent [target_tokens, D] on CPU, full clip
    frames [T, C, H, W] on CPU). Same call sequence as
    scripts/build_semantic_anchor.py's reference-encoding loop.
    """
    clip = dataset._load_clip(path)
    frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as other scripts
    with torch.no_grad():
        encoded = encoder(frames)
        real_target = apply_masks(encoded, [masks_y]).squeeze(0).cpu()
    del frames, encoded
    clear_gpu(device)
    return real_target, clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntu_root", type=Path, help="root directory of NTU RGB+D clips")
    parser.add_argument("--target-clip", type=Path, required=True, help="the clip to steer and visualize")
    parser.add_argument("--anchor", type=Path, required=True, help="path to an anchor .pt saved by scripts/build_semantic_anchor.py")
    parser.add_argument("--pca-basis", type=Path, required=True, help="path to a basis .pt saved by scripts/build_pca_basis.py")
    parser.add_argument("--blind-alpha", type=float, default=None, help="override steering.blind_alpha from configs/base.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/checkpoints/viz"), help="outputs saved under <output-dir>/<target-clip-stem>/")
    parser.add_argument("--skip-nn-retrieval", action="store_true",
                         help="skip NN retrieval entirely (PCA overlay only) — NN retrieval re-encodes "
                              "every one of the anchor's reference clips to build its comparison bank, "
                              "which can dominate runtime on a slow/contended GPU if the anchor has many "
                              "references; PCA overlay only needs the target clip + the anchor's already-"
                              "computed mean latent, so it stays fast regardless of anchor size")
    parser.add_argument("--max-nn-reference-clips", type=int, default=None,
                         help="cap how many of the anchor's reference clips get re-encoded for the NN "
                              "retrieval bank (default: use all of them). Lowering this trades retrieval "
                              "diversity for speed without needing to rebuild the anchor itself smaller.")
    args = parser.parse_args()

    target_meta = parse_ntu_filename(args.target_clip)
    if target_meta is None:
        raise SystemExit(f"--target-clip {args.target_clip} does not match SsssCcccPpppRrrrAaaa")
    print(f"target: {target_meta.path.name} -> action A{target_meta.action} ({target_meta.action_label}), "
          f"performer P{target_meta.performer}, camera C{target_meta.camera}")

    anchor_data = torch.load(args.anchor, weights_only=False)
    anchor = SemanticAnchor(
        action_label=anchor_data["action_label"],
        latent=anchor_data["latent"],
        phase_length=anchor_data["phase_length"],
        num_reference_clips=anchor_data["num_reference_clips"],
    )
    if anchor.action_label != target_meta.action_label:
        raise SystemExit(f"anchor is for action {anchor.action_label!r}, target clip is "
                          f"{target_meta.action_label!r} — these must match")

    blind_alpha = args.blind_alpha if args.blind_alpha is not None else load_blind_alpha()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"blind_alpha = {blind_alpha}, device = {device}")

    dataset = NTURGBDDataset(root=args.ntu_root, num_frames=NUM_FRAMES, deterministic=True)

    print("loading encoder + predictor (torch.hub, first run will download)...")
    encoder = VJEPAEncoder(device=device)
    world_model = VJEPAWorldModel(device=device)
    masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=1,
        device=device,
    )

    pca_basis = PCABasis.from_state_dict(torch.load(args.pca_basis, weights_only=False))
    print(f"loaded shared PCA basis from {args.pca_basis} (fit once, reused for every arm below)")

    output_dir = args.output_dir / target_meta.path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nencoding + predicting target clip {target_meta.path.name}...")
    predicted_target, target_clip_frames = encode_predict(dataset, target_meta.path, encoder, world_model, masks_x, masks_y, device)
    anchor_latent = anchor.latent.unsqueeze(0)  # add batch dim, matches predicted_target's [1, N, D]
    drift = compute_drift(predicted_target, anchor_latent)
    print(f"drift d = 1 - cos(predicted, anchor) = {drift:.4f}")

    steered_latents = {}
    for name, arm in [("raw", RawController()), ("blind", BlindController(alpha=blind_alpha))]:
        alpha = arm.get_alpha(target_meta.action_label, drift, step=0)
        steered_latents[name] = apply_steering(predicted_target, anchor_latent, alpha)
        print(f"  {name}: alpha={alpha}")

    def reshape_blocks(latent: torch.Tensor) -> torch.Tensor:
        return latent.squeeze(0).view(NUM_TARGET_BLOCKS, NUM_SPATIAL_PATCHES, -1)

    latent_blocks = {name: reshape_blocks(t) for name, t in steered_latents.items()}
    original_frames = representative_frames(target_clip_frames, TARGET_BLOCK_OFFSET, NUM_TARGET_BLOCKS)

    print("\nrendering PCA overlays (shared basis, fit once, reused across arms)...")
    pca_paths = {}
    for name, blocks in latent_blocks.items():
        pca_paths[name] = project_to_rgb_overlay(blocks, original_frames, pca_basis, output_dir / f"pca_{name}.png")
        print(f"  {name}: {pca_paths[name]}")

    combined = torch.cat([read_image_float(pca_paths["raw"]), read_image_float(pca_paths["blind"])], dim=-2)  # stacked vertically
    save_image(combined, output_dir / "pca_raw_vs_blind.png")
    print(f"  side-by-side: {output_dir / 'pca_raw_vs_blind.png'} (top=Raw, bottom=Blind)")

    del predicted_target, target_clip_frames
    clear_gpu(device)

    if args.skip_nn_retrieval:
        print("\n--skip-nn-retrieval passed — skipping NN retrieval (PCA overlay above is already saved).")
    elif "reference_paths" not in anchor_data:
        print("\nWARNING: anchor has no 'reference_paths' — skipping NN retrieval (no reference bank "
              "available without exact provenance). Rebuild the anchor with the current "
              "scripts/build_semantic_anchor.py to enable this.")
    else:
        reference_paths = anchor_data["reference_paths"]
        if args.max_nn_reference_clips is not None and len(reference_paths) > args.max_nn_reference_clips:
            print(f"\ncapping NN retrieval reference bank to {args.max_nn_reference_clips} of the anchor's "
                  f"{len(reference_paths)} reference clip(s) (--max-nn-reference-clips)")
            reference_paths = reference_paths[:args.max_nn_reference_clips]
        print(f"\nbuilding NN retrieval reference bank from {len(reference_paths)} reference clip(s)...")
        reference_bank: list[ReferenceBankEntry] = []
        for ref_path_str in reference_paths:
            ref_path = Path(ref_path_str)
            real_target, ref_clip = encode_real_target(dataset, ref_path, encoder, masks_y, device)
            ref_blocks = real_target.view(NUM_TARGET_BLOCKS, NUM_SPATIAL_PATCHES, -1)
            ref_frames = representative_frames(ref_clip, TARGET_BLOCK_OFFSET, NUM_TARGET_BLOCKS)
            for t in range(NUM_TARGET_BLOCKS):
                frame_path = save_frame_as_image(ref_frames[t], output_dir / "reference_frames" / f"{ref_path.stem}_block{t}.png")
                reference_bank.append(ReferenceBankEntry(
                    clip_path=ref_path, block_index=t, frame_path=frame_path,
                    feature=ref_blocks[t].mean(dim=0),
                ))
            del ref_clip
            clear_gpu(device)
        print(f"  reference bank: {len(reference_bank)} (clip, block) entries from "
              f"{len(anchor_data['reference_paths'])} reference clip(s)")

        print("\nretrieving nearest real frames (RETRIEVAL, not generation — cosine distance, "
              "consistent with overseer/drift_detection.py's convention)...")
        for name, blocks in latent_blocks.items():
            frame_paths = retrieve_nearest_real_frames(blocks, reference_bank)
            stitch_to_mp4(frame_paths, output_dir / f"retrieval_{name}.mp4", label=f"retrieval-{name}")

            rows = [
                torch.cat([original_frames[t], read_image_float(frame_paths[t])], dim=-1)
                for t in range(NUM_TARGET_BLOCKS)
            ]
            comparison = torch.cat(rows, dim=-2)
            comparison_path = output_dir / f"nn_retrieval_{name}_side_by_side.png"
            save_image(comparison, comparison_path)
            print(f"  {name}: {comparison_path} (left=target's own real frame, right=retrieved real frame)")

    print(f"\nall outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
