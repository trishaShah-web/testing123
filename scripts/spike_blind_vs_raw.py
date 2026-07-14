"""Day-2 spike gate (STATUS.md hard gate): "On a few NTU clips: measure IDS
+ SCS on Raw, apply Blind averaging, re-measure. Green if IDS drops and SCS
holds." This script does that for ONE target clip against a REAL pooled
Semantic Anchor (scripts/build_semantic_anchor.py) — not the two-clip
smoke test's degenerate one-clip stand-in.

Pipeline: load the target clip -> encode full clip -> predict_future once
(the single predicted latent both arms steer from) -> Raw (alpha=0, i.e.
steered==predicted, unchanged) and Blind (alpha=steering.blind_alpha) both
applied via overseer.apply_steering -> score each arm's steered latent with
IDS (performer probe) and SCS (action probe), evaluation/probes.py.

CAVEAT (read before trusting the verdict): this scores exactly ONE target
clip. IDS/SCS are top-1 correct/incorrect (0 or 1) for that single example
per arm — not a statistically meaningful "IDS dropped" signal on their own.
STATUS.md's gate says "a few NTU clips"; running this script over several
target clips (same action, different target performers) and looking at the
trend is the actual gate check. This script does one clip because that's
what was asked for this iteration; extending --target-clip to accept
several paths and aggregating is the natural next step, not done here.

Probe training set (evaluation/probes.py train_linear_probe): the target
clip's own action + 5 other diverse actions (arm-only/whole-body/
dynamic/static, so SCS isn't a trivial 1-class problem), all performers
present in the dataset, restricted to the target clip's own camera (avoids
the viewpoint confound already fixed in data/ntu_rgbd.py). Deliberately
small (Day-2 spike, not the final evaluation) -- same order of magnitude as
anchor building, not the full dataset.

LEAKAGE GUARD: the target clip itself, and the anchor's own reference
clips, are excluded from probe training. Anchors built with the current
scripts/build_semantic_anchor.py save `reference_paths` for exact
exclusion. Anchors built BEFORE that field existed (no `reference_paths`
key in the .pt file) fall back to a heuristic exclusion — same action,
performer != target performer, same camera as target — which is what the
anchor was almost certainly built from, but isn't a path-exact guarantee;
a warning is printed when this fallback is used. Rebuild the anchor with
the current script if you need exact provenance.

Same GPU-memory care as build_semantic_anchor.py: one clip encoded at a
time, moved to CPU immediately, torch.cuda.empty_cache() between clips
(we've OOM'd more than once today).

Usage:
    python scripts/spike_blind_vs_raw.py /path/to/ntu_root \
        --target-clip /path/to/ntu_root/S004C002P003R001A015_rgb.avi \
        --anchor anchor_A015.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.ntu_rgbd import NTURGBDDataset, parse_ntu_filename
from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer import SemanticAnchor, compute_drift, apply_steering
from arms import RawController, BlindController
from evaluation import compute_ids, compute_scs
from evaluation.probes import LabelEncoder, pool_latent, train_linear_probe

NUM_FRAMES = 16  # = steering.phase_length (configs/base.yaml); see AGENT.md DEVIATIONS #6
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2

# Fixed "filler" actions for probe training diversity — arm-only/whole-body/
# dynamic/static mix, per HANDOFF.md. The target clip's own action is added
# to this set at runtime (see build_probe_training_set).
FILLER_ACTIONS = [6, 24, 27, 8, 1]  # pickup, kicking, jump up, sitting down, drink water

REPO_ROOT = Path(__file__).resolve().parent.parent


def clear_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def load_blind_alpha() -> float:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        config = yaml.safe_load(f)
    alpha = config["steering"]["blind_alpha"]
    if alpha is None:
        raise SystemExit("steering.blind_alpha is null in configs/base.yaml — set it before running the spike.")
    return alpha


def encode_real_target_region(dataset: NTURGBDDataset, path: Path, encoder: VJEPAEncoder, masks_y: torch.Tensor, device: str) -> torch.Tensor:
    """One clip -> its REAL (not predicted) target-region latent, on CPU,
    GPU cache cleared. Used for probe training examples.
    """
    clip = dataset._load_clip(path)  # (T, C, H, W), deterministic full-span sampling
    frames = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as smoke tests
    with torch.no_grad():
        encoded = encoder(frames)
        real_target = apply_masks(encoded, [masks_y]).squeeze(0).cpu()
    del clip, frames, encoded
    clear_gpu(device)
    return real_target


def build_probe_training_set(
    dataset: NTURGBDDataset,
    encoder: VJEPAEncoder,
    masks_y: torch.Tensor,
    device: str,
    target_action: int,
    target_camera: int | None,
    exclude_paths: set[Path],
) -> tuple[torch.Tensor, list[int], list[int]]:
    """-> (pooled_features [N, D], performer_labels_raw, action_labels_raw).
    Every training clip contributes ONE encode pass, reused as a labeled
    example for BOTH the performer probe and the action probe.
    """
    actions = sorted({target_action, *FILLER_ACTIONS})
    performers = sorted({r.performer for r in dataset.records})

    features, performer_labels, action_labels = [], [], []
    for action in actions:
        pool = dataset.clips_for_action(action, camera=target_camera)
        for performer in performers:
            candidates = [r for r in pool if r.performer == performer and r.path.resolve() not in exclude_paths]
            if not candidates:
                print(f"  WARNING: no probe-training clip for action={action} performer={performer} "
                      f"(camera={target_camera}) after exclusions — skipping this cell")
                continue
            record = candidates[0]
            latent = encode_real_target_region(dataset, record.path, encoder, masks_y, device)
            features.append(pool_latent(latent))
            performer_labels.append(performer)
            action_labels.append(action)
            print(f"  probe-training clip encoded: {record.path.name} (P{performer}, A{action})")

    if not features:
        raise SystemExit("no probe-training clips survived leakage exclusion — cannot train IDS/SCS probes")

    print("  class balance (total training examples per class, after exclusion):")
    for action in actions:
        count = action_labels.count(action)
        flag = "  <-- thin, interpret with caution" if count < 2 else ""
        print(f"    action A{action}: {count} example(s){flag}")
    for performer in performers:
        count = performer_labels.count(performer)
        flag = "  <-- thin, interpret with caution" if count < 2 else ""
        print(f"    performer P{performer}: {count} example(s){flag}")

    return torch.stack(features), performer_labels, action_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntu_root", type=Path, help="root directory of NTU RGB+D clips")
    parser.add_argument("--target-clip", type=Path, required=True, help="the clip to steer and score")
    parser.add_argument("--anchor", type=Path, required=True, help="path to an anchor .pt saved by scripts/build_semantic_anchor.py")
    parser.add_argument("--blind-alpha", type=float, default=None, help="override steering.blind_alpha from configs/base.yaml")
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

    exclude_paths = {args.target_clip.resolve()}
    if "reference_paths" in anchor_data:
        exclude_paths |= {Path(p).resolve() for p in anchor_data["reference_paths"]}
        print(f"leakage guard: excluding target clip + {len(anchor_data['reference_paths'])} "
              f"anchor reference clip(s) (exact paths, from anchor file)")
    else:
        print("WARNING: anchor file has no 'reference_paths' (built before that field existed) — "
              "falling back to a HEURISTIC leakage exclusion (same action, performer != target's, "
              "same camera as target). Not a path-exact guarantee; rebuild the anchor with the "
              "current scripts/build_semantic_anchor.py for exact provenance.")

    blind_alpha = args.blind_alpha if args.blind_alpha is not None else load_blind_alpha()
    print(f"blind_alpha = {blind_alpha}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = NTURGBDDataset(root=args.ntu_root, num_frames=NUM_FRAMES, deterministic=True)

    if "reference_paths" not in anchor_data:
        # heuristic fallback exclusion, see WARNING above
        heuristic_refs = dataset.clips_for_action(
            target_meta.action, exclude_performer=target_meta.performer, camera=target_meta.camera,
        )
        exclude_paths |= {r.path.resolve() for r in heuristic_refs}

    print("loading encoder + predictor (torch.hub, first run will download)...")
    encoder = VJEPAEncoder(device=device)
    world_model = VJEPAWorldModel(device=device)

    masks_x, masks_y = build_temporal_split_masks(
        num_temporal_blocks=NUM_TEMPORAL_BLOCKS,
        num_spatial_patches=NUM_SPATIAL_PATCHES,
        batch_size=1,
        device=device,
    )

    print("\nbuilding probe training set...")
    features, performer_labels_raw, action_labels_raw = build_probe_training_set(
        dataset, encoder, masks_y, device,
        target_action=target_meta.action, target_camera=target_meta.camera,
        exclude_paths=exclude_paths,
    )
    performer_encoder = LabelEncoder.fit(performer_labels_raw)
    action_encoder = LabelEncoder.fit(action_labels_raw)
    if target_meta.performer not in performer_encoder.classes:
        raise SystemExit(f"target performer {target_meta.performer} has no probe-training examples "
                          f"(after leakage exclusion) — IDS cannot score it. Need another performer's "
                          f"clip for P{target_meta.performer}/A{target_meta.action} at camera "
                          f"{target_meta.camera} in the dataset.")
    print(f"performer probe classes: {performer_encoder.classes} (chance = {1/performer_encoder.num_classes:.3f})")
    print(f"action probe classes: {action_encoder.classes} (chance = {1/action_encoder.num_classes:.3f})")

    performer_probe = train_linear_probe(
        features, performer_encoder.encode(performer_labels_raw), num_classes=performer_encoder.num_classes,
    )
    action_probe = train_linear_probe(
        features, action_encoder.encode(action_labels_raw), num_classes=action_encoder.num_classes,
    )
    del features
    clear_gpu(device)

    print("\nencoding + predicting target clip...")
    target_clip = dataset._load_clip(args.target_clip)
    target_frames = target_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as smoke tests
    with torch.no_grad():
        encoded_target = encoder(target_frames)
        predicted_target = world_model.predict_future(encoded_target, masks_x, masks_y).cpu()
    del target_clip, target_frames, encoded_target
    clear_gpu(device)

    anchor_latent = anchor.latent.unsqueeze(0)  # add batch dim, matches predicted_target's [1, N, D]
    drift = compute_drift(predicted_target, anchor_latent)
    print(f"drift d = 1 - cos(predicted, anchor) = {drift:.4f}")

    performer_label_idx = performer_encoder.encode([target_meta.performer])
    action_label_idx = action_encoder.encode([target_meta.action])

    print(f"\n{'arm':6s}  {'alpha':>6s}  {'||steered-predicted||':>22s}  {'IDS':>10s}  {'SCS':>10s}")
    results = {}
    for name, arm in [("Raw", RawController()), ("Blind", BlindController(alpha=blind_alpha))]:
        alpha = arm.get_alpha(target_meta.action_label, drift, step=0)
        steered = apply_steering(predicted_target, anchor_latent, alpha)
        moved = (steered - predicted_target).norm().item()
        ids = compute_ids(steered, performer_label_idx, performer_probe)
        scs = compute_scs(steered, action_label_idx, action_probe)
        results[name] = {"ids": ids, "scs": scs}
        print(f"{name:6s}  {alpha:6.3f}  {moved:22.4f}  {ids:10.3f}  {scs:10.3f}")

    print(f"\nchance: IDS={1/performer_encoder.num_classes:.3f}, SCS={1/action_encoder.num_classes:.3f}")
    ids_dropped = results["Blind"]["ids"] < results["Raw"]["ids"]
    scs_held = results["Blind"]["scs"] >= results["Raw"]["scs"]
    print(f"IDS dropped (Blind < Raw): {ids_dropped}")
    print(f"SCS held (Blind >= Raw): {scs_held}")
    print(f"gate condition (both true): {ids_dropped and scs_held}")
    print("\nCAVEAT: n=1 target clip -- IDS/SCS above are single 0/1 outcomes, not a statistically "
          "meaningful trend. STATUS.md's gate needs 'a few NTU clips'; treat this as one data point, "
          "not a verdict, until run across several target clips.")


if __name__ == "__main__":
    main()
