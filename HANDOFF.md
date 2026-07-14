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

- **Immediate**: get `scripts/smoke_test_two_clips.py` to run successfully
  (`PYTHONPATH=. python scripts/smoke_test_two_clips.py <target> <reference>`).
  This was mid-troubleshooting when this file was written.
- **After that passes**: do NOT jump to the LLM. Order is: (1) confirm the
  two-clip smoke test works, (2) build a *real* Semantic Anchor from a pool
  of clips (not the current one-clip degenerate version), (3) validate
  Raw vs. Blind on that real anchor (this is the Day-2 spike gate), (4) only
  then wire up the LLM (Job 1/Job 2 in `overseer/llm_overseer.py`, both
  still `NotImplementedError`).
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
