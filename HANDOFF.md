# HANDOFF.md

Session-to-session engineering log. Distinct from AGENT.md / ARCHITECTURE.md
/ INSTRUCTIONS.md / SKILLS.md / STATUS.md, which are the locked research-spec
files (read those for *what* the project is). This file is for *what's been
built and where we left off* — read it first when resuming so you don't have
to re-derive it from the conversation.

Update this file, don't replace it, as work continues — append to "Session
summary" and rewrite "Where things stand" each time.

---

## Session summary (chronological)

**1. Repo scaffold audit + realignment to revised spec.**
The existing code implemented the OLD pre-revision design that AGENT.md's
DEVIATIONS block had already overturned: a pixel decoder existed, the LLM
overseer read raw latent tensors directly, and dropped baselines
(LoRA-Fine-Tuned, Prompt-Conditioned) were still live. Fixed:
- Deleted `vjepa/decoder.py` (forbidden — no pixel decoder, ARCHITECTURE.md).
- Quarantined LoRA/Prompt-Conditioned baselines to `arms/_pending_signoff/`
  (dropped pending sign-off, AGENT.md DEVIATIONS #4).
- Renamed `baselines/` -> `arms/`, restructured around a shared
  `AlphaController` interface (`arms/raw.py`, `blind.py`, `llm.py`).
- Rewrote `overseer/llm_overseer.py` as a text-only controller (Job 1
  `build_target`, Job 2 `schedule_correction`) — no longer touches latents.
- Rewrote `overseer/semantic_anchor.py` -> cross-performer mean latent
  (`from_reference_clips`), not LLM-built.
- Implemented `overseer/drift_detection.py` (`d = 1-cos(...)`) and
  `overseer/guidance_vector.py` (`steered = predicted + alpha*(anchor-predicted)`)
  for real — these are documented decisions per ARCHITECTURE.md, not TODOs.
- Rewrote `pipeline/inference_loop.py` to match the revised data flow.
- Added missing `visualization/` package (`nn_retrieval.py`, `pca_overlay.py`
  — stubs; nothing existed for this before).
- Fixed `evaluation/ids.py|scs.py|pcs.py` signatures (dropped a
  `predicted_frames` param that can never exist — no decoder).
- Updated `configs/base.yaml`, `requirements.txt`, `STATUS.md` to match.

**2. V-JEPA 2 checkpoint/API verification** (source-grounded via WebFetch on
the real `facebookresearch/vjepa2` repo, not invented from memory):
- `facebook/vjepa2-vitl-fpc64-256` exists on HF, but `transformers.AutoModel`
  only exposes encoder features — no predictor.
- Verified via `hubconf.py` -> `src/hub/backbones.py`
  (`_make_vjepa2_model`) that
  `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large', pretrained=True)`
  returns `(encoder, predictor)` from ONE checkpoint — the standard
  (non-action-conditioned) predictor. NOT `vjepa2_ac_vit_giant`, which is a
  different, action-conditioned Giant model.
- Added `vjepa/_hub.py` (cached loader), wired into `vjepa/encoder.py` and
  `vjepa/world_model.py`'s `_load_frozen_model`.
- Added `timm`, `einops` to `requirements.txt` (hubconf.py's actual deps —
  were missing).

**3. Predictor forward/masking API verification + implementation**, all
verified 2026-07-13/14 against primary source:
- `predictor.forward(x, masks_x, masks_y, mask_index=1, has_cls=False)`
  (`src/models/predictor.py`).
- `apply_masks` gather logic (`src/masks/utils.py`) — vendored into
  `vjepa/masking.py` with attribution, rather than cross-imported from
  torch.hub's cache dir (fragile, not a stable public API for a 2-line fn).
- Token ordering is time-major (`src/models/vision_transformer.py`):
  `token_index = t * num_spatial_patches + spatial_index`.
- Predictor output dim defaults to encoder `embed_dim` when `out_embed_dim`
  is unset (`VisionTransformerPredictor.__init__`) — so a predicted future
  latent and a real encoded latent are directly comparable/addable, not
  just shape-compatible by luck.
- `vjepa/masking.py`: `apply_masks` + `build_temporal_split_masks` (context
  = first half of the clip's 32 temporal blocks, target = second half —
  single future-chunk prediction, the ARCHITECTURE.md default).
- `VJEPAWorldModel.predict_next` (a stub) replaced with
  `predict_future(encoded_full_clip, masks_x, masks_y)` — real
  implementation, single-shot (not autoregressive).
- `pipeline/inference_loop.py` reconciled to this single-shot call
  (`rollout_steps` must be 1; >1 explicitly raises — chaining is an
  unresolved embedding-space question, not silently approximated).

**4. NTU RGB+D dataset support**, filename convention verified against
`shahroudy/NTURGB-D` README:
- `data/ntu_action_labels.py` — full A1-A120 label dict. Cross-checked
  against the user's own copy of the list: 118/120 exact, 2 cosmetic
  apostrophe-character-only diffs (A109, A117) — not fixed, doesn't matter.
- Rewrote `data/ntu_rgbd.py` — real `_build_index`, shared
  `parse_ntu_filename()` helper, `NTUClipMeta` dataclass,
  `clips_for_action(action, exclude_performer=...)` for anchor pooling.

**5. Smoke test scripts** (both source-verified but NOT execution-tested by
me — no runtime environment available in this session):
- `scripts/smoke_test_single_clip.py` — one clip in, encoder+predictor run,
  prints a sanity drift number vs. the real held-out half. Bypasses
  anchor/arms entirely.
- `scripts/smoke_test_two_clips.py` — two clips (same action, different
  performer) in, full predict -> drift -> steer chain, Raw vs. Blind
  comparison, using ONE reference clip as a degenerate one-clip anchor.
  **Not** a real scientific anchor test (needs a pool, not one clip) — a
  plumbing check only, do not report its numbers as results.

**6. Environment/running troubleshooting (live, with the user)**:
- `pip install` hit macOS's `externally-managed-environment` error ->
  resolved via a venv: `python3 -m venv venv && source venv/bin/activate`.
- `ModuleNotFoundError: No module named 'data'` when running
  `python scripts/smoke_test_two_clips.py` directly -> scripts/ isn't on
  `sys.path` for absolute imports of `data`/`vjepa`/`overseer`/`arms`. Fix:
  prefix with `PYTHONPATH=.` (run from the `ECCV/` root).
- **Last state**: about to re-run with
  `PYTHONPATH=. python scripts/smoke_test_two_clips.py <target> <reference>`
  from `~/Desktop/ECCV` with the venv active. Not yet confirmed working —
  this is the very next thing to check when resuming.

---

**7. Two-clip smoke test executed successfully on Kaggle (2026-07-14).**
Real run, two NTU clips (same action, different performers):
- `drift d = 1 - cos(predicted, anchor) = 0.5049` (cos_sim ≈ 0.495, ~60°
  apart) — moderate divergence, consistent with "related action, different
  performer," not identical and not random.
- Raw arm (`alpha=0.000`): `||steered-predicted|| = 0.0000` — correct by
  construction.
- Blind arm (`alpha=0.300`): `||steered-predicted|| = 898.9101` — nonzero,
  correct direction of movement toward the reference clip's real future
  encoding. (Implies `||anchor-predicted|| ≈ 2996.4`; this raw norm is not
  itself meaningful — V-JEPA latents are not unit-normalized, so only
  cosine-based drift and the later IDS/SCS/PCS metrics carry scientific
  weight.)
- **Conclusion**: the encoder -> predictor -> drift -> steer chain is now
  CONFIRMED WORKING end-to-end (not merely source-verified). Still only a
  plumbing check — anchor was a single reference clip (degenerate), not a
  real pooled Semantic Anchor. Do not cite these numbers as a result.
- Tensor layout assumption `(1, C, T, H, W)` (see unverified assumption #1
  below) evidently worked — no shape-mismatch error. Downgrading from
  "unverified" to "confirmed correct" for this checkpoint/predictor path.

**8. Phase-alignment decisions made + real anchor-pool builder written
(2026-07-14, same session as item 7).**
- `steering.phase_length` decided: **16**, units = target-region temporal
  blocks (not raw frames) — every clip is still resampled to the encoder's
  fixed 64-frame input (32 temporal blocks), and the existing context/target
  mask split already takes the second half (16 blocks); this reuses that
  shape rather than inventing a new one. Documented in `configs/base.yaml`.
- Resample method decided: uniform full-span index subsampling, applied
  **deterministically** — found and fixed a real phase-alignment bug in the
  process: `VideoDataset._load_clip` used a *random* start offset
  (`torch.randint`) when a clip is longer than needed, which is fine for a
  single smoke-test clip but would silently break anchor pooling (each
  reference clip's window would land at an arbitrary, unaligned point in the
  action, making cross-performer averaging meaningless). Fixed via a new
  opt-in `deterministic=True` constructor flag on `VideoDataset`
  (`data/video_dataset.py`) that uses `torch.linspace(0, total-1,
  num_frames)` instead — full clip span, evenly spaced, no randomness.
  Default (`deterministic=False`) is unchanged, so the two already-confirmed
  smoke test scripts are unaffected.
- New `scripts/build_semantic_anchor.py`: pools every other-performer clip
  for a given NTU action via `NTURGBDDataset.clips_for_action`, loads each
  with `deterministic=True`, encodes, extracts the REAL (not predicted)
  target-region latent, and calls `SemanticAnchor.from_reference_clips`.
  This is the actual "step 2" from the prior session's plan. **Not yet
  execution-tested** — no real pool of other-performer clips is uploaded to
  Kaggle yet (only the two smoke-test clips exist there so far); this is
  next.

**9. NTU RGB+D data audited + prepared for Kaggle upload (2026-07-14).**
User has `~/Downloads/nturgb+d_rgb/` locally: 1440 clips, setup S004 only, 4
performers (P003, P007, P008, P020), 3 camera views (C001-C003), 2
replications, 60 actions (A001-A060), 5.5GB total. Verified against the
real project code (not just eyeballed):
- All 1440 filenames parse correctly via `parse_ntu_filename` (0 unparsed).
- `NTURGBDDataset(root=..., deterministic=True).clips_for_action(13,
  exclude_performer=3)` returns a real 18-clip pool end to end.
- **Found + fixed a second methodological gap**: `clips_for_action` had no
  way to control camera view, so pooling would mix all 3 camera angles
  into the anchor — conflating "viewpoint" drift with "performer identity"
  drift, which the anchor is supposed to isolate from. Added an optional
  `camera` filter param to `clips_for_action` (`data/ntu_rgbd.py`).
- `scripts/build_semantic_anchor.py` updated: now restricts the reference
  pool to the target clip's own camera by default (opt out with
  `--all-cameras`), and added `--target-clip` to infer
  action/exclude_performer/camera straight from a clip's filename instead
  of specifying each by hand.
- No file reorganization needed — filenames already match
  `SsssCcccPpppRrrrAaaa` exactly, so the folder can be pointed at directly
  as `ntu_root`. Wrote `dataset-metadata.json` into
  `~/Downloads/nturgb+d_rgb/` (kaggle CLI's required manifest for
  `kaggle datasets create -p <folder>`) — `id` field has a placeholder
  `REPLACE_WITH_YOUR_KAGGLE_USERNAME`, not yet filled in (don't know the
  user's Kaggle username). **Not yet uploaded** — needs the user's Kaggle
  CLI + credentials, which this session doesn't have.
- Flagged for the user, not yet confirmed: NTU RGB+D's own usage
  agreement is research-only / no-redistribution, so this Kaggle dataset
  should be created as **Private**, not Public.

**10. Synced with teammate's commits (2026-07-14).** Pulled + merged two
commits from Devanshi Kashyap (`origin/main`): `vjepa/_hub.py` now passes
`trust_repo=True` to `torch.hub.load` (avoids an interactive trust prompt
hanging non-interactive environments like Kaggle — no conflict). Also
`scripts/smoke_test_two_clips.py`'s `NUM_FRAMES` changed from 64 to 16 —
user confirmed this is intentional/crucial, not a mistake. **Flagging, not
resolving**: this was not reconciled against today's `configs/base.yaml`
`steering.phase_length=16` decision, whose documented reasoning explicitly
assumed clips still enter the encoder at the full 64-frame `fpc64` input
(with 16 being the downstream target-region latent block count, a
different quantity). If the encoder is now actually fed 16 raw frames, the
`vitl-fpc64-256` checkpoint's positional embeddings (verified/documented
for 64 frames) may not behave as documented — needs reconciliation with
Devanshi on what NUM_FRAMES=16 means architecturally before trusting
results built on it. Merged via `fadda28`, pushed to `origin/main`
(GitHub reported the repo moved to
`github.com/trishaShah-web/testing123.git` — push succeeded via redirect,
remote URL not yet updated).

**11. Item 8's reasoning corrected — item 10's flag resolved (2026-07-14,
same session).** Re-read `SemanticAnchor.phase_length`'s own docstring
(`overseer/semantic_anchor.py`): `# T: fixed length every reference clip
was resampled to`. That's the RAW FRAME count, not a target-region
temporal-block count — item 8 above was wrong about the units (it matched
16 to "target-region blocks" only because the *original* smoke-test code's
`phase_length=NUM_TEMPORAL_BLOCKS - NUM_TEMPORAL_BLOCKS//2` formula
happened to equal 16 when `NUM_FRAMES` was still 64; that formula was
itself inconsistent with the dataclass field's documented meaning). Correct
reading: `steering.phase_length=16` and Devanshi's `NUM_FRAMES=16` are the
SAME quantity, not a conflict — resolves item 10's flag. Fixed
`scripts/build_semantic_anchor.py` to use `NUM_FRAMES=16` (was still
hardcoded 64, inconsistent with `smoke_test_two_clips.py` after the merge)
and `PHASE_LENGTH = NUM_FRAMES` directly (was the wrong
temporal-block-count formula). Updated `configs/base.yaml`,
`ARCHITECTURE.md` component 1, and `STATUS.md` to match. Added
**AGENT.md DEVIATIONS #6**: input clip length is 16 frames, not the
`vjepa2-vitl-fpc64-256` checkpoint's native 64-frame training config — this
is the real remaining open question (not the units confusion, which is now
resolved), since whether the frozen encoder/predictor's positional
embeddings behave correctly on a 16-frame input (8 temporal blocks, not the
source-verified 32) has not been checked against primary source. It ran
without a shape error on Kaggle, which is plumbing-compatible, not
confirmation the predictions are meaningful at this length — flagged for
sign-off, not silently accepted as correct.
`scripts/smoke_test_single_clip.py` still hardcodes `NUM_FRAMES=64` and
was NOT changed (it doesn't use `SemanticAnchor`/`phase_length` at all, so
it's an independent script, but it's now the only place in the repo still
exercising the original 64-frame path — worth the team noting).

## Known unverified assumptions (check these first if something breaks)

1. **Tensor layout into `VJEPAEncoder.forward()`.** Both smoke-test scripts
   guess `(1, C, T, H, W)` from TorchCodec's `(T, C, H, W)` output via
   `permute(1,0,2,3).unsqueeze(0)`. This was never confirmed from source —
   if either script throws a shape-mismatch error here, this is the first
   suspect.
2. Everything under "source-verified, not execution-tested" — i.e. all of
   `vjepa/encoder.py`, `vjepa/world_model.py`, `vjepa/masking.py` — is
   read-from-source-correct as of 2026-07-13/14 but has never actually run
   end to end. Treat every claim of "verified" in this file and in
   STATUS.md as "verified by reading the real source code," not "confirmed
   working."

**12. IDS/SCS probes implemented + Day-2 spike script written (2026-07-14,
same session).** Anchor building confirmed working for real
(`anchor_A015.pt`, 3 reference clips, action "take off jacket") — step 2 of
the plan below is done.
- `evaluation/probes.py` (new): shared linear-probe utilities —
  `LabelEncoder` (raw NTU IDs -> contiguous class indices), `pool_latent`
  (mean-pool token sequence -> one feature vector), `LinearProbe`
  (`nn.Linear(embed_dim, num_classes)`, the documented simple architecture),
  `train_linear_probe` (full-batch Adam + cross-entropy, CPU, no GPU
  contention), `probe_accuracy`. Sanity-tested standalone with synthetic
  separable data (100% train accuracy) before wiring into anything.
- `evaluation/ids.py` / `evaluation/scs.py`: implemented (were
  `NotImplementedError` stubs). Both now take an already-trained
  `LinearProbe` — training and scoring the same example in one call would
  leak it into its own training set, so probe training is a separate step
  the caller (the spike script) owns.
- `configs/base.yaml` `steering.blind_alpha` set to **0.3** (was `null`) —
  matches the value already informally used in
  `scripts/smoke_test_two_clips.py`'s `--blind-alpha` default.
- `scripts/build_semantic_anchor.py`: added `torch.cuda.empty_cache()`
  between each reference clip's encode (was missing — likely part of why
  OOMs happened today) and now saves `reference_paths`/`action`/`camera` in
  the output `.pt` file, so downstream scripts can exclude the anchor's
  exact source clips from probe training. **`anchor_A015.pt` was built
  before this fix and has no `reference_paths`** — the spike script falls
  back to a heuristic exclusion for it (same action, performer != target's,
  same camera) with a printed warning; rebuild the anchor with the current
  script for exact provenance.
- New `scripts/spike_blind_vs_raw.py`: the actual Day-2 gate script. Loads
  one target clip, runs `predict_future` once, applies Raw/Blind steering
  to that one predicted latent, scores each arm with IDS/SCS using probes
  trained on a small (~24-clip) held-out set — target clip's own action + 5
  filler actions (pickup, kicking, jump up, sitting down, drink water) × all
  4 performers × the target's own camera, with the anchor's reference clips
  and the target clip itself excluded. Prints a class-balance summary
  (flags classes with <2 surviving examples) and a Raw-vs-Blind comparison
  table, then the gate condition (IDS dropped AND SCS held).
  **Verified via an offline dry run** (real dataset, real anchor file, real
  leakage/probe-training logic — only the GPU encoder/predictor mocked with
  random tensors of the correct shape, since this machine has no CUDA and a
  real run would trigger a multi-GB checkpoint download): ran end to end
  without error. The dry run also concretely surfaced the class-thinness
  risk flagged during design — for A015, performer P7 ended up with 0
  surviving training examples (both its replications had fed the anchor)
  while the other 3 performers had 1 each; the class-balance printout
  caught this correctly rather than hiding it. **Not yet run for real** —
  needs a GPU (Kaggle).

**13. Anchor/probe-training overlap bug fixed (2026-07-14, same session).**
The first real run of `spike_blind_vs_raw.py` (against the actual
`anchor_A015.pt`, 3 reference clips) crashed: with only 4 total NTU
performers, excluding the target (P003) + all 3 anchor references (P007,
P008, P020) left ZERO other-performer A015 clips for probe training.
- `data/ntu_rgbd.py` `clips_for_action`'s `exclude_performer` now accepts
  a single ID or a list.
- `scripts/build_semantic_anchor.py`: new `--max-references` flag caps how
  many DISTINCT performers feed the anchor (not raw clip count — a
  capped-in performer's clips at all replications still get used).
  Default: pool's performer count minus 1, always reserving exactly one
  performer's clips as disjoint probe-training material. `--exclude-performer`
  now also accepts multiple IDs for manual reservation. Saved anchor files
  now also record `reference_performers` and `reserved_performers`.
- **Verified via a real (mocked-encoder) end-to-end dry run**: rebuilding
  the A015 anchor with the new default correctly used only performers
  [7, 8], reserved [20], and `spike_blind_vs_raw.py` correctly picked up
  P20's A015 clip as a surviving probe-training example (action A15 went
  from 0 training examples, the crash, to 2). P7/P8 still correctly warn
  (fully consumed by the anchor) — the fix is precise, not a blanket
  workaround.
- `anchor_A015.pt` (the one already saved) predates this fix and still
  has all 3 references baked in — **must be rebuilt** before the spike
  script can run on it without hitting the same crash.

**14. Probe class-imbalance fixed + spike script refactored for multi-clip runs
(2026-07-19).** Confirmed the reported problem by reading the real selection
logic in `build_probe_training_set` before changing anything: filler actions
are never touched by leakage exclusion (up to ~4 examples/class, one per
performer), while the target action's own cross-performer clips are largely
consumed by the anchor's `reference_paths` exclusion (typically 1-2 surviving
examples) — a structural asymmetry from anchor-building, not bad luck. SCS
was measuring a probe that had almost no target-class signal to learn from.
- `scripts/spike_blind_vs_raw.py` rewritten:
  - **Class balancing**: `build_probe_training_set` now collects all
    surviving candidates per action class, then caps every class down to the
    SMALLEST surviving class's count via deterministic (seeded) subsampling
    — no class fabricated or oversampled, cap value + seed printed every run
    (`configs/base.yaml` `seed: 42`, override with `--seed`). Both
    before/after class-balance tables are printed.
  - **Perf fix**: split into `train_probes(dataset, encoder, masks_y,
    device, target_action, target_camera, exclude_paths, seed) ->
    TrainedProbes` (builds the training pool + fits both probes, called
    ONCE per run) and `score_clip(target_meta, dataset, encoder,
    world_model, masks_x, masks_y, anchor, probes, blind_alpha, device) ->
    dict` (one target clip's encode/predict/steer/score, called in a loop).
    Previously the ~24-clip training pool was re-encoded from scratch for
    every target clip; now it's encoded once regardless of how many target
    clips are scored.
  - **`--target-clip` now accepts multiple paths** (`nargs="+"`), all must
    share the anchor's action (validated, raises `SystemExit` otherwise).
    `main()` trains probes once, loops `score_clip` per clip, and prints a
    per-arm mean IDS/SCS across all target clips plus the gate condition on
    those means — this is the actual "run across a few target clips, look at
    the trend" gate check STATUS.md/HANDOFF item 12 called for, not just a
    performance fix. Single-clip invocations still work (n=1 caveat still
    printed).
  - Added a matching `target_meta.action not in probes.action_encoder.classes`
    guard next to the existing performer-side one (an existing gap where a
    fully-excluded target action would previously crash inside
    `LabelEncoder.encode` with a less clear `ValueError` instead of a clean
    `SystemExit`).
  - Did NOT touch `scripts/build_semantic_anchor.py`, the anchor/steering
    arithmetic (`overseer/*.py`), or `arms/*.py` — out of scope, confirmed
    unchanged.
- **Not execution-tested this session** (no GPU/Kaggle access here) —
  `python3 -m py_compile` passes; needs a real Kaggle run before trusting the
  printed numbers. This is the next thing to run.
- **Found, not fixed (flagged only, per explicit instruction not to touch it
  in this session)**: `pipeline/inference_loop.py`'s `SteeringPipeline` still
  hardcodes `_NUM_TEMPORAL_BLOCKS = 64 // 2` / `_NUM_SPATIAL_PATCHES = 256`
  — the pre-DEVIATIONS-#6 64-frame config. It was never updated when
  `NUM_FRAMES` moved to 16 (item 11 above). As written it would build masks
  sized for 32 temporal blocks while being fed a 16-frame/8-block clip and
  likely fail inside `apply_masks`'s `torch.gather` (out-of-range index) —
  this class is not used by `spike_blind_vs_raw.py` or
  `build_semantic_anchor.py` (both bypass it and call
  `encoder`/`world_model.predict_future`/`apply_steering` directly with the
  correct 16-frame constants), so it hasn't caused a failure yet, but should
  not be trusted or wired into anything new until fixed separately.

**15. Visualization implemented: PCA overlay (primary) + NN retrieval
(secondary) (2026-07-19, same session as item 14).** Both
`visualization/pca_overlay.py` and `visualization/nn_retrieval.py` were pure
`NotImplementedError` stubs before this; both are now real implementations.
Reuses the same encode/predict/steer call sequence
`scripts/spike_blind_vs_raw.py` uses (inlined into a new driver script, see
below) — explicitly NOT routed through `pipeline/inference_loop.py`'s
`SteeringPipeline`, which still hardcodes the pre-DEVIATIONS-#6 64-frame
mask dims (`_NUM_TEMPORAL_BLOCKS = 64 // 2`, `_NUM_SPATIAL_PATCHES = 256`)
and would build wrongly-sized masks for the actual 16-frame/8-block input,
likely failing inside `apply_masks`'s `torch.gather`. **Flagged, not
fixed** — out of scope for this task per explicit instruction; needs a
separate fix before anything routes through it.
- **PCA overlay** (`visualization/pca_overlay.py`): new `PCABasis` dataclass
  (`mean`, `components` [3,D], plus `component_min`/`component_max` [3] —
  the min/max are fit on the SAME reference set as the axes, at the SAME
  `fit_shared_pca_basis` call, specifically so per-image color rescaling
  can't quietly reintroduce the "non-comparable colors across arms" bug one
  level below the axis-sharing fix). `fit_shared_pca_basis` (SVD-based,
  no new dependency — no sklearn in requirements.txt, so this uses
  `torch.linalg.svd` directly). `project_to_rgb_overlay(latent_blocks
  [T_blocks, num_spatial_patches, D], original_frames [T_blocks, C, H, W],
  pca_basis, output_path, alpha=0.5)` projects each temporal block's tokens
  through the fixed basis, reshapes to a 16x16 grid (verified square:
  `num_spatial_patches=256`), upsamples with **nearest** interpolation
  (documented: a patch has no sub-patch resolution, so smooth interpolation
  would imply false precision), alpha-blends over that block's real
  representative frame, and saves one horizontally-concatenated strip image
  per clip/arm.
  - New `scripts/build_pca_basis.py`: the "fit once" step — takes an
    explicit `--clips` list or `--from-anchor <anchor.pt>` (reuses that
    anchor's `reference_paths`, no separate reference-set curation
    decision needed), encodes each clip's real target-region latent (same
    code path as `build_semantic_anchor.py`), fits+saves `pca_basis.pt`.
    **Not yet run for real** (needs Kaggle/GPU) — verified via a synthetic-
    tensor unit test instead (see below).
- **NN retrieval** (`visualization/nn_retrieval.py`): new `ReferenceBankEntry`
  dataclass (`clip_path`, `block_index`, `frame_path`, `feature` — one
  real (clip, target-temporal-block) candidate). `retrieve_nearest_real_frames`
  mean-pools each query temporal block and picks the bank entry with the
  highest cosine similarity (documented decision, approved: cosine, matching
  `overseer/drift_detection.py`'s existing `1-cos` convention elsewhere in
  the project — "distance in latent space" now means one consistent thing
  everywhere). `stitch_to_mp4` and `save_frame_as_image` materialize
  retrieved/query frames to real files and stitch them into a small mp4,
  always printed/labeled "retrieval" (no pixel watermark — would need a
  font-rendering dependency this project doesn't otherwise need).
  - **Found + fixed while implementing**: `stitch_to_mp4` was first written
    against `torchvision.io.write_video`, which does not exist in the
    installed `torchvision==0.28` — torchvision's video I/O was deprecated
    and removed, which is the exact reason `data/video_dataset.py` already
    switched frame *decoding* to torchcodec (see that file's own docstring).
    Switched `stitch_to_mp4` to `torchcodec.encoders.VideoEncoder(...).to_file(...)`
    (torchcodec 0.14, installed, has a real encoder now) — consistent with
    the project's existing torchcodec decision rather than reintroducing a
    dead torchvision path. Verified this actually encodes a real (small)
    mp4 file via a synthetic-frame test, not just import-checked.
- New driver script `scripts/visualize_steering.py`: for one target clip +
  anchor + `pca_basis.pt`, computes Raw (alpha=0) and Blind (alpha=
  `steering.blind_alpha`) predicted/steered latents (same inlined
  encode/predict/steer sequence as `spike_blind_vs_raw.py`), renders PCA
  overlays for both arms plus one combined Raw-vs-Blind comparison image,
  builds an NN reference bank from the anchor's `reference_paths` (skips
  NN retrieval with a clear warning if the anchor predates that field),
  retrieves + stitches an mp4 per arm, and saves a real-vs-retrieved
  side-by-side comparison image per arm. All outputs default under
  `/kaggle/working/checkpoints/viz/<target-clip-stem>/`. One clip at a
  time, `torch.cuda.empty_cache()` between every encode call (same
  discipline as the other Kaggle scripts). **Not yet run for real** (needs
  Kaggle/GPU + a built anchor + a built PCA basis).
- **Verified via synthetic-tensor smoke tests this session** (no GPU/
  checkpoint needed — pure tensor-shape/math correctness, run in the local
  venv, not on Kaggle): `fit_shared_pca_basis` produces the right shapes,
  `PCABasis` state-dict round-trips exactly, `project_to_rgb_overlay`
  produces the correct output image shape, `retrieve_nearest_real_frames`
  correctly retrieves an exact cosine-similarity match in a constructed
  test case, and `stitch_to_mp4` produces a real playable mp4 file. These
  confirm the tensor plumbing is correct; they do NOT confirm anything
  about real V-JEPA latents (still untested end-to-end on Kaggle).

## Where things stand right now / next steps

- **Real anchor building CONFIRMED WORKING**, but the anchor/probe overlap
  bug (item 13) means `anchor_A015.pt` as currently saved must be
  **rebuilt** with the updated `build_semantic_anchor.py` before
  `spike_blind_vs_raw.py` can run on it for real.
- **IDS/SCS implemented; `scripts/spike_blind_vs_raw.py` rewritten (item 14)
  to fix class imbalance (deterministic cap-to-min-class-count) and to
  accept multiple `--target-clip` paths in one run (`train_probes` once +
  `score_clip` looped + per-arm mean IDS/SCS across clips) — this is now
  code-complete for the ACTUAL "a few NTU clips" gate check, not just one
  clip. Dry-run-verified this session with a mocked encoder/dataset
  (tensor-shape/logic correctness only); NOT yet run for real** (needs
  GPU/Kaggle) — this is the very next thing to run, high priority (blocks
  the running accounts).
- **Visualization implemented** (item 15): `visualization/pca_overlay.py`
  and `visualization/nn_retrieval.py` are real (were stubs), plus two new
  scripts (`scripts/build_pca_basis.py`, `scripts/visualize_steering.py`).
  Verified via synthetic-tensor unit tests only — NOT yet run against real
  V-JEPA latents on Kaggle.
- **Order**: (1) DONE — two-clip smoke test. (2) DONE — real pooled
  Semantic Anchor (needs rebuilding with `--max-references`, see item 13).
  (3) NEXT, two parallel tracks now that both are code-complete:
  (3a) rebuild `anchor_A015.pt`
  (`python scripts/build_semantic_anchor.py <ntu_root> --target-clip
  <P003 A015 clip path> --output anchor_A015.pt`), then run the rewritten
  `spike_blind_vs_raw.py` **across several target clips at once**
  (`--target-clip <clip1> <clip2> ...`, same action, different performers)
  on Kaggle and read the real, now-balanced IDS/SCS numbers — this is the
  first run that can actually answer the Day-2 gate question rather than
  produce a single anecdotal data point. (3b) `python
  scripts/build_pca_basis.py <ntu_root> --from-anchor anchor_A015.pt
  --output pca_basis.pt`, then `python scripts/visualize_steering.py
  <ntu_root> --target-clip <clip> --anchor anchor_A015.pt --pca-basis
  pca_basis.pt` and eyeball the outputs under
  `/kaggle/working/checkpoints/viz/`. (4) only then wire up the LLM (Job
  1/Job 2 in `overseer/llm_overseer.py`, both still `NotImplementedError`).
- **Still untouched stubs**: `data/something_something_v2.py`,
  `data/ucf101.py`, `overseer/llm_overseer.py` (Job1/Job2 prompt logic),
  `evaluation/pcs.py`.
- **Known bug, flagged not fixed (item 15)**: `pipeline/inference_loop.py`'s
  `SteeringPipeline` still hardcodes the pre-16-frame mask dimensions
  (`_NUM_TEMPORAL_BLOCKS = 64 // 2`) and would likely break if anything
  routed a real 16-frame clip through it. Nothing currently does (every
  working script bypasses it and calls encoder/world_model/apply_steering
  directly) — but fix it before wiring anything new through that class.
- **Known limitation to watch**: probe training data comes from the same
  small local NTU subset the anchor is built from. The class-balancing cap
  added in item 14 fixes the *action*-class imbalance (SCS), but caps down
  to the smallest surviving class — with this few performers, that can mean
  very small per-class counts (the mocked dry run saw cap=1), which in turn
  makes it MORE likely that any single target clip's own performer ends up
  with zero surviving *performer*-class training examples purely by
  capping's random subsampling, not by an actual data gap. `score_clip` now
  raises a clear per-clip error for this and `main()`'s multi-clip loop
  catches it and skips that one clip rather than aborting the whole run
  (fixed this session after the mocked dry run surfaced exactly this
  failure mode) — but it means a run can silently score fewer clips than
  requested; check the "N/M target clip(s) skipped" line every run.

## Exact commands reference

```
cd ~/Desktop/ECCV
source venv/bin/activate                          # every new terminal
PYTHONPATH=. python scripts/smoke_test_single_clip.py /path/to/clip.avi
PYTHONPATH=. python scripts/smoke_test_two_clips.py /path/to/target.avi /path/to/reference.avi
PYTHONPATH=. python scripts/build_semantic_anchor.py <ntu_root> --target-clip <clip.avi> --output anchor.pt
PYTHONPATH=. python scripts/spike_blind_vs_raw.py <ntu_root> --target-clip <clip1.avi> <clip2.avi> --anchor anchor.pt
PYTHONPATH=. python scripts/build_pca_basis.py <ntu_root> --from-anchor anchor.pt --output pca_basis.pt
PYTHONPATH=. python scripts/visualize_steering.py <ntu_root> --target-clip <clip.avi> --anchor anchor.pt --pca-basis pca_basis.pt
```

## File inventory — created or substantively rewritten this session

New: `vjepa/_hub.py`, `vjepa/masking.py`, `visualization/` (whole package),
`arms/` (whole package, renamed from `baselines/`),
`arms/_pending_signoff/README.md`, `data/ntu_action_labels.py`,
`scripts/smoke_test_single_clip.py`, `scripts/smoke_test_two_clips.py`,
this file.

Rewritten: `vjepa/decoder.py` (deleted), `vjepa/encoder.py`,
`vjepa/world_model.py`, `overseer/llm_overseer.py`,
`overseer/semantic_anchor.py`, `overseer/drift_detection.py`,
`overseer/guidance_vector.py`, `overseer/__init__.py`,
`pipeline/inference_loop.py`, `data/ntu_rgbd.py`,
`evaluation/ids.py|scs.py|pcs.py`, `configs/base.yaml`,
`requirements.txt`, `STATUS.md`.

## File inventory — 2026-07-19 session (items 14-15)

New: `scripts/build_pca_basis.py`, `scripts/visualize_steering.py`.

Rewritten: `scripts/spike_blind_vs_raw.py` (class-balancing cap,
`train_probes`/`score_clip` split, multi-clip `--target-clip`, per-clip
skip-on-error), `visualization/pca_overlay.py` (stub -> real),
`visualization/nn_retrieval.py` (stub -> real).

Not touched (confirmed, per explicit scope): `scripts/build_semantic_anchor.py`,
`overseer/*.py`, `arms/*.py`, `pipeline/inference_loop.py` (bug found and
flagged above, fix deferred).
