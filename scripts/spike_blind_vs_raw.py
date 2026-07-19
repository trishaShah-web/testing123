"""Day-2 spike gate (STATUS.md hard gate): "On a few NTU clips: measure IDS
+ SCS on Raw, apply Blind averaging, re-measure. Green if IDS drops and SCS
holds." This script does that against a REAL pooled Semantic Anchor
(scripts/build_semantic_anchor.py) — not the two-clip smoke test's
degenerate one-clip stand-in — and now accepts SEVERAL target clips (same
action, different performers) in one run, per STATUS.md's actual gate
wording ("a few NTU clips") and HANDOFF.md's stated next step.

Pipeline per target clip: encode full clip -> predict_future once (the
single predicted latent both arms steer from) -> Raw (alpha=0, i.e.
steered==predicted, unchanged) and Blind (alpha=steering.blind_alpha) both
applied via overseer.apply_steering -> score each arm's steered latent with
IDS (performer probe) and SCS (action probe), evaluation/probes.py.

Probes are trained ONCE per run (train_probes), not once per target clip —
this was a real performance bug: every target clip used to re-encode the
entire ~24-clip probe training pool from scratch. score_clip() now takes the
already-trained probes and only does the target clip's own encode/predict/
steer/score. All target clips in one run must share the SAME action (the
anchor's action) since the probe training pool and its leakage exclusion are
built once for that action.

Probe training set (evaluation/probes.py train_linear_probe): the target
action + 5 other diverse filler actions (arm-only/whole-body/dynamic/static,
so SCS isn't a trivial 1-class problem), all performers present in the
dataset, restricted to the target clips' shared camera (avoids the viewpoint
confound already fixed in data/ntu_rgbd.py). Deliberately small (Day-2
spike, not the final evaluation) -- same order of magnitude as anchor
building, not the full dataset.

CLASS BALANCING (fixed 2026-07-19): the target action's own cross-performer
clips are largely consumed by the anchor (they're excluded from probe
training as leakage), while filler actions are never touched by that
exclusion — so filler classes could end up with several times as many
training examples as the target class, structurally starving the action
probe of target-class signal (SCS meaningless by construction, not by bad
luck). Fix: after building the per-class candidate pool, every action class
is capped down to the SMALLEST surviving class's count via deterministic
(seeded) subsampling — no class is ever fabricated or oversampled, and the
cap value + seed are printed every run (INSTRUCTIONS.md "no silent magic
numbers"). Seed comes from configs/base.yaml `seed` (override with
--seed).

LEAKAGE GUARD: all target clips passed in this run, and the anchor's own
reference clips, are excluded from probe training. Anchors built with the
current scripts/build_semantic_anchor.py save `reference_paths` for exact
exclusion. Anchors built BEFORE that field existed (no `reference_paths`
key in the .pt file) fall back to a heuristic exclusion — same action,
performer != any target performer, same camera as the targets — which is
what the anchor was almost certainly built from, but isn't a path-exact
guarantee; a warning is printed when this fallback is used. Rebuild the
anchor with the current script if you need exact provenance.

Same GPU-memory care as build_semantic_anchor.py: one clip encoded at a
time, moved to CPU immediately, torch.cuda.empty_cache() between clips
(we've OOM'd more than once today).

Usage:
    python scripts/spike_blind_vs_raw.py /path/to/ntu_root \
        --target-clip /path/to/ntu_root/S004C002P003R001A015_rgb.avi \
                       /path/to/ntu_root/S004C002P007R001A015_rgb.avi \
        --anchor anchor_A015.pt

    (one --target-clip is still fine; several is the real Day-2 gate check
    per STATUS.md's "a few NTU clips" wording — all must be the same action
    as the anchor.)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from data.ntu_rgbd import NTURGBDDataset, NTUClipMeta, parse_ntu_filename
from vjepa import VJEPAEncoder, VJEPAWorldModel
from vjepa.masking import apply_masks, build_temporal_split_masks
from overseer import SemanticAnchor, compute_drift, apply_steering
from arms import RawController, BlindController
from evaluation import compute_ids, compute_scs
from evaluation.probes import LabelEncoder, LinearProbe, pool_latent, train_linear_probe

NUM_FRAMES = 16  # = steering.phase_length (configs/base.yaml); see AGENT.md DEVIATIONS #6
NUM_TEMPORAL_BLOCKS = NUM_FRAMES // 2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2

# Fixed "filler" actions for probe training diversity — arm-only/whole-body/
# dynamic/static mix, per HANDOFF.md. The target clips' own (shared) action
# is added to this set at runtime (see build_probe_training_set).
FILLER_ACTIONS = [6, 24, 27, 8, 1]  # pickup, kicking, jump up, sitting down, drink water

REPO_ROOT = Path(__file__).resolve().parent.parent


def clear_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


def load_blind_alpha(config: dict) -> float:
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
    seed: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """-> (pooled_features [N, D], performer_labels_raw, action_labels_raw).
    Every training clip contributes ONE encode pass, reused as a labeled
    example for BOTH the performer probe and the action probe.

    Collects up to one clip per (action, performer) cell, then caps every
    action class down to the smallest surviving class's count (deterministic,
    seeded subsampling) so the action probe (SCS) isn't structurally starved
    of target-class examples relative to the filler classes — see module
    docstring "CLASS BALANCING".
    """
    actions = sorted({target_action, *FILLER_ACTIONS})
    performers = sorted({r.performer for r in dataset.records})

    per_action: dict[int, list[tuple[torch.Tensor, int]]] = {a: [] for a in actions}
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
            per_action[action].append((pool_latent(latent), performer))
            print(f"  probe-training clip encoded: {record.path.name} (P{performer}, A{action})")

    surviving_counts = {a: len(items) for a, items in per_action.items() if items}
    if not surviving_counts:
        raise SystemExit("no probe-training clips survived leakage exclusion — cannot train IDS/SCS probes")

    print("  class balance (before capping):")
    for action in actions:
        count = len(per_action[action])
        flag = "  <-- thin, interpret with caution" if count < 2 else ""
        print(f"    action A{action}: {count} example(s){flag}")

    cap = min(surviving_counts.values())
    print(f"  balancing: capping every action class to {cap} example(s) — the smallest surviving "
          f"class, chosen so no class is fabricated or oversampled (deterministic, seed={seed})")
    rng = random.Random(seed)

    features, performer_labels, action_labels = [], [], []
    for action in actions:
        items = per_action[action]
        if not items:
            continue
        if len(items) > cap:
            items = rng.sample(items, cap)
        for feature, performer in items:
            features.append(feature)
            performer_labels.append(performer)
            action_labels.append(action)

    print("  class balance (after capping):")
    for action in actions:
        count = action_labels.count(action)
        print(f"    action A{action}: {count} example(s)")
    for performer in performers:
        count = performer_labels.count(performer)
        flag = "  <-- thin, interpret with caution" if count < 2 else ""
        print(f"    performer P{performer}: {count} example(s){flag}")

    return torch.stack(features), performer_labels, action_labels


@dataclass
class TrainedProbes:
    performer_probe: LinearProbe
    action_probe: LinearProbe
    performer_encoder: LabelEncoder
    action_encoder: LabelEncoder


def train_probes(
    dataset: NTURGBDDataset,
    encoder: VJEPAEncoder,
    masks_y: torch.Tensor,
    device: str,
    target_action: int,
    target_camera: int | None,
    exclude_paths: set[Path],
    seed: int,
) -> TrainedProbes:
    """Build the probe training set ONCE and fit both probes on it. Callers
    scoring several target clips (same action) call this exactly once and
    reuse the result across all of them via score_clip — the training pool
    no longer gets re-encoded per target clip.
    """
    features, performer_labels_raw, action_labels_raw = build_probe_training_set(
        dataset, encoder, masks_y, device,
        target_action=target_action, target_camera=target_camera,
        exclude_paths=exclude_paths, seed=seed,
    )
    performer_encoder = LabelEncoder.fit(performer_labels_raw)
    action_encoder = LabelEncoder.fit(action_labels_raw)
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
    return TrainedProbes(
        performer_probe=performer_probe,
        action_probe=action_probe,
        performer_encoder=performer_encoder,
        action_encoder=action_encoder,
    )


def score_clip(
    target_meta: NTUClipMeta,
    dataset: NTURGBDDataset,
    encoder: VJEPAEncoder,
    world_model: VJEPAWorldModel,
    masks_x: torch.Tensor,
    masks_y: torch.Tensor,
    anchor: SemanticAnchor,
    probes: TrainedProbes,
    blind_alpha: float,
    device: str,
) -> dict:
    """One target clip -> encode + predict once, Raw/Blind steer, IDS/SCS
    score each arm. Uses already-trained `probes` — does not touch the
    probe training pool at all.
    """
    if target_meta.performer not in probes.performer_encoder.classes:
        raise SystemExit(f"target performer {target_meta.performer} has no probe-training examples "
                          f"(after leakage exclusion) — IDS cannot score it. Need another performer's "
                          f"clip for P{target_meta.performer}/A{target_meta.action} at camera "
                          f"{target_meta.camera} in the dataset.")
    if target_meta.action not in probes.action_encoder.classes:
        raise SystemExit(f"target action {target_meta.action} has no probe-training examples "
                          f"(after leakage exclusion) — SCS cannot score it.")

    target_clip = dataset._load_clip(target_meta.path)
    target_frames = target_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # UNVERIFIED layout guess, same as smoke tests
    with torch.no_grad():
        encoded_target = encoder(target_frames)
        predicted_target = world_model.predict_future(encoded_target, masks_x, masks_y).cpu()
    del target_clip, target_frames, encoded_target
    clear_gpu(device)

    anchor_latent = anchor.latent.unsqueeze(0)  # add batch dim, matches predicted_target's [1, N, D]
    drift = compute_drift(predicted_target, anchor_latent)

    performer_label_idx = probes.performer_encoder.encode([target_meta.performer])
    action_label_idx = probes.action_encoder.encode([target_meta.action])

    arms = {}
    for name, arm in [("Raw", RawController()), ("Blind", BlindController(alpha=blind_alpha))]:
        alpha = arm.get_alpha(target_meta.action_label, drift, step=0)
        steered = apply_steering(predicted_target, anchor_latent, alpha)
        moved = (steered - predicted_target).norm().item()
        ids = compute_ids(steered, performer_label_idx, probes.performer_probe)
        scs = compute_scs(steered, action_label_idx, probes.action_probe)
        arms[name] = {"alpha": alpha, "moved": moved, "ids": ids, "scs": scs}

    return {"target_meta": target_meta, "drift": drift, "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntu_root", type=Path, help="root directory of NTU RGB+D clips")
    parser.add_argument(
        "--target-clip", type=Path, nargs="+", required=True,
        help="one or more clips to steer and score — all must share the anchor's action "
             "(different performers is the real Day-2 gate check, STATUS.md 'a few NTU clips')",
    )
    parser.add_argument("--anchor", type=Path, required=True, help="path to an anchor .pt saved by scripts/build_semantic_anchor.py")
    parser.add_argument("--blind-alpha", type=float, default=None, help="override steering.blind_alpha from configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override configs/base.yaml seed (used for deterministic probe class-balance capping)")
    args = parser.parse_args()

    config = load_config()
    blind_alpha = args.blind_alpha if args.blind_alpha is not None else load_blind_alpha(config)
    seed = args.seed if args.seed is not None else config["seed"]
    print(f"blind_alpha = {blind_alpha}, seed = {seed}")

    anchor_data = torch.load(args.anchor, weights_only=False)
    anchor = SemanticAnchor(
        action_label=anchor_data["action_label"],
        latent=anchor_data["latent"],
        phase_length=anchor_data["phase_length"],
        num_reference_clips=anchor_data["num_reference_clips"],
    )

    target_metas: list[NTUClipMeta] = []
    for clip in args.target_clip:
        meta = parse_ntu_filename(clip)
        if meta is None:
            raise SystemExit(f"--target-clip {clip} does not match SsssCcccPpppRrrrAaaa")
        if meta.action_label != anchor.action_label:
            raise SystemExit(f"--target-clip {clip} is action {meta.action_label!r}, anchor is "
                              f"{anchor.action_label!r} — all target clips in one run must match the anchor's action")
        target_metas.append(meta)
        print(f"target: {meta.path.name} -> action A{meta.action} ({meta.action_label}), "
              f"performer P{meta.performer}, camera C{meta.camera}")

    exclude_paths = {m.path.resolve() for m in target_metas}
    if "reference_paths" in anchor_data:
        exclude_paths |= {Path(p).resolve() for p in anchor_data["reference_paths"]}
        print(f"leakage guard: excluding {len(target_metas)} target clip(s) + {len(anchor_data['reference_paths'])} "
              f"anchor reference clip(s) (exact paths, from anchor file)")
    else:
        print("WARNING: anchor file has no 'reference_paths' (built before that field existed) — "
              "falling back to a HEURISTIC leakage exclusion (same action, performer != any target's, "
              "same camera as the targets). Not a path-exact guarantee; rebuild the anchor with the "
              "current scripts/build_semantic_anchor.py for exact provenance.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = NTURGBDDataset(root=args.ntu_root, num_frames=NUM_FRAMES, deterministic=True)

    if "reference_paths" not in anchor_data:
        # heuristic fallback exclusion, see WARNING above
        heuristic_refs = dataset.clips_for_action(
            target_metas[0].action,
            exclude_performer=[m.performer for m in target_metas],
            camera=target_metas[0].camera,
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

    print("\ntraining probes (once for this run, shared across all target clips)...")
    probes = train_probes(
        dataset, encoder, masks_y, device,
        target_action=target_metas[0].action, target_camera=target_metas[0].camera,
        exclude_paths=exclude_paths, seed=seed,
    )

    print(f"\n{'clip':32s} {'arm':6s}  {'alpha':>6s}  {'||steered-predicted||':>22s}  {'IDS':>10s}  {'SCS':>10s}")
    results = []
    skipped = []
    for meta in target_metas:
        print(f"\nencoding + predicting {meta.path.name}...")
        try:
            result = score_clip(meta, dataset, encoder, world_model, masks_x, masks_y, anchor, probes, blind_alpha, device)
        except SystemExit as e:
            # One clip lacking training examples for its own performer/action
            # (more likely now that class balancing caps the training set
            # down to the smallest surviving class, see build_probe_training_set)
            # should not throw away every OTHER clip's already-computed
            # results in a multi-clip run — skip it and keep going.
            print(f"  SKIPPING {meta.path.name}: {e}")
            skipped.append((meta.path.name, str(e)))
            continue
        results.append(result)
        print(f"  drift d = 1 - cos(predicted, anchor) = {result['drift']:.4f}")
        for name, arm_res in result["arms"].items():
            print(f"{meta.path.name:32s} {name:6s}  {arm_res['alpha']:6.3f}  {arm_res['moved']:22.4f}  "
                  f"{arm_res['ids']:10.3f}  {arm_res['scs']:10.3f}")

    if not results:
        raise SystemExit(f"no target clips were successfully scored — all {len(skipped)} were skipped, see warnings above")
    if skipped:
        print(f"\n{len(skipped)}/{len(target_metas)} target clip(s) skipped (see SKIPPING lines above): "
              f"{[name for name, _ in skipped]}")

    print(f"\nchance: IDS={1/probes.performer_encoder.num_classes:.3f}, SCS={1/probes.action_encoder.num_classes:.3f}")

    for name in ("Raw", "Blind"):
        ids_vals = [r["arms"][name]["ids"] for r in results]
        scs_vals = [r["arms"][name]["scs"] for r in results]
        print(f"{name}: mean IDS={sum(ids_vals)/len(ids_vals):.3f}, mean SCS={sum(scs_vals)/len(scs_vals):.3f} (n={len(results)})")

    mean_ids_raw = sum(r["arms"]["Raw"]["ids"] for r in results) / len(results)
    mean_ids_blind = sum(r["arms"]["Blind"]["ids"] for r in results) / len(results)
    mean_scs_raw = sum(r["arms"]["Raw"]["scs"] for r in results) / len(results)
    mean_scs_blind = sum(r["arms"]["Blind"]["scs"] for r in results) / len(results)
    ids_dropped = mean_ids_blind < mean_ids_raw
    scs_held = mean_scs_blind >= mean_scs_raw
    print(f"IDS dropped (mean Blind < mean Raw): {ids_dropped}")
    print(f"SCS held (mean Blind >= mean Raw): {scs_held}")
    print(f"gate condition (both true): {ids_dropped and scs_held}")
    print(f"\nn={len(results)} target clip(s) scored.")
    if len(results) == 1:
        print("CAVEAT: n=1 target clip -- IDS/SCS above are a single 0/1 outcome per arm, not a "
              "statistically meaningful trend. STATUS.md's gate needs 'a few NTU clips'; pass "
              "several --target-clip paths (same action, different performers) for a real gate check.")


if __name__ == "__main__":
    main()
