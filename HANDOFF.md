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

## Where things stand right now / next steps

- **Two-clip smoke test CONFIRMED WORKING** (2026-07-14, Kaggle) — see
  session summary item 7. Step (1) of the order below is done.
- **Immediate next step**: do NOT jump to the LLM. Order is: (1) DONE —
  confirm the two-clip smoke test works, (2) build a *real* Semantic Anchor
  from a pool of clips (not the current one-clip degenerate version) — this
  needs the phase-alignment/resampling utility (does not exist yet, see
  below), (3) validate Raw vs. Blind on that real anchor (this is the Day-2
  spike gate), (4) only then wire up the LLM (Job 1/Job 2 in
  `overseer/llm_overseer.py`, both still `NotImplementedError`).
- **Still untouched stubs**: `data/something_something_v2.py`,
  `data/ucf101.py`, `overseer/llm_overseer.py` (Job1/Job2 prompt logic),
  `evaluation/ids.py|scs.py|pcs.py` (probe architecture), and
  `visualization/nn_retrieval.py|pca_overlay.py`.
- **Phase-alignment/resampling utility does not exist yet** — needed to
  turn a pool of same-action clips (possibly different lengths/framerates)
  into the fixed-length reference set `SemanticAnchor.from_reference_clips`
  expects. This is required before step 2 above.

## Exact commands reference

```
cd ~/Desktop/ECCV
source venv/bin/activate                          # every new terminal
PYTHONPATH=. python scripts/smoke_test_single_clip.py /path/to/clip.avi
PYTHONPATH=. python scripts/smoke_test_two_clips.py /path/to/target.avi /path/to/reference.avi
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
