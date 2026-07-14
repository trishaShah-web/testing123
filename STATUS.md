# STATUS.md

## Project
Zero-Shot Semantic Steering for Video World Models

## Current Stage
**Design locked (revised), scaffold realigned, pre-implementation.**
Feasibility review complete; design corrected (no decoder, LLM as
text-only controller, cross-performer anchor). A repo scaffold exists
(module skeletons + interfaces), but no working scientific logic yet — every
component that requires a documented decision still raises
`NotImplementedError` until that decision is made. 7-day build window,
3 people.

## Status Categories
COMPLETED / IN PROGRESS / NOT STARTED / UNKNOWN.

---

## Current Project Status

### COMPLETED
- problem + Forecast Drift definition
- scientific nugget, hypotheses, contributions (revised)
- architecture (revised: decoder removed, LLM reframed, anchor = cross-performer mean)
- methodology + experimental arms (Raw / Blind / LLM)
- metric definitions with concrete latent-based implementations (IDS/SCS/PCS)
- dataset plan (curated subsets)
- visualization plan (NN retrieval + shared-basis PCA overlay)
- feasibility review + audit fallback defined
- repo scaffold realigned to the revised architecture: removed the pixel
  decoder (`vjepa/decoder.py`) and the per-step latent-reading LLM loop that
  had been left over from the pre-revision design; quarantined
  LoRA-Fine-Tuned/Prompt-Conditioned baselines to
  `arms/_pending_signoff/` (dropped, AGENT.md DEVIATIONS #4); added the
  missing `visualization/` package (NN retrieval + PCA overlay stubs) that
  component 7 requires but had no code for
- V-JEPA 2 encoder + predictor loading API verified against primary source
  (facebookresearch/vjepa2 `hubconf.py` -> `src/hub/backbones.py`,
  2026-07-13): `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large',
  pretrained=True)` returns `(encoder, predictor)` from one checkpoint —
  the standard non-action-conditioned predictor, not the
  `vjepa2_ac_vit_giant` action-conditioned one. `vjepa/encoder.py` and
  `vjepa/world_model.py` now call this (via shared cached loader
  `vjepa/_hub.py`) instead of raising `NotImplementedError` on load.
  `requirements.txt` updated to add `timm`/`einops`, which the hub loader
  needs. **Not yet execution-tested** — see IN PROGRESS.
- V-JEPA predictor forward()/masking call verified against primary source
  (`src/models/predictor.py` forward signature, `src/masks/utils.py
  apply_masks`, `src/models/vision_transformer.py` token ordering), all
  2026-07-13. `vjepa/masking.py` (new) constructs context/target masks by
  splitting the clip's 32 temporal blocks at the midpoint (single
  future-chunk prediction — the ARCHITECTURE.md default, not autoregressive
  chaining). `VJEPAWorldModel.predict_next` replaced with
  `predict_future(encoded_full_clip, masks_x, masks_y)`, which actually
  calls the predictor instead of raising `NotImplementedError`.
  `pipeline/inference_loop.py` reconciled to this single-shot call
  (`rollout_steps` must be 1; >1 explicitly raises, not silently
  approximated). **Not yet execution-tested.**
- `scripts/smoke_test_single_clip.py` (new): standalone script — one real
  NTU clip in, encoder + predictor run on it, prints a sanity
  drift number (predicted vs. real encoding of the held-out half). Bypasses
  SteeringPipeline/anchor/arms entirely, since a real Semantic Anchor needs
  other-performer reference clips a single video can't provide. This is a
  plumbing check, not IDS/SCS/PCS — do not report its output as a result.
- `data/ntu_action_labels.py` (new, A1-A120 names) and a real
  `data/ntu_rgbd.py` (parses `SsssCcccPpppRrrrAaaa` from filenames; both
  verified against the dataset's own GitHub README, 2026-07-14; cross-checked
  against the user's own copy, 118/120 exact + 2 cosmetic apostrophe-only
  diffs). `NTURGBDDataset.clips_for_action()` finds same-action
  other-performer clips.
- `scripts/smoke_test_two_clips.py` (new): same action, two performers ->
  runs predict -> drift -> Raw/Blind steer end to end using one real
  reference clip as a degenerate one-clip anchor. Confirmed 2026-07-14 that
  predictor output dim defaults to encoder embed_dim (source-verified), so
  this arithmetic is dimensionally valid, not just shape-compatible by
  luck. Still NOT a real Semantic Anchor (needs a POOL of reference clips,
  not one) — plumbing check only.

- Two-clip smoke test (`scripts/smoke_test_two_clips.py`) executed
  successfully on Kaggle, 2026-07-14, on two real NTU clips (same action,
  different performers): `drift d = 1-cos(predicted, anchor) = 0.5049`;
  Raw arm `alpha=0.000` gives `||steered-predicted||=0.0000` (correct by
  construction); Blind arm `alpha=0.300` gives `||steered-predicted||=
  898.9101` (nonzero, correct-direction move toward the reference
  performer's real future encoding). Encoder -> predictor -> drift -> steer
  chain is now CONFIRMED WORKING end-to-end, not just source-verified.
  Tensor layout guess `(1, C, T, H, W)` confirmed correct (no shape
  mismatch). Still a plumbing check only — anchor was one degenerate
  reference clip, not a real pooled Semantic Anchor; do not cite these
  numbers as a result.
- Phase alignment decided (2026-07-14): `steering.phase_length = 16`
  RAW FRAMES — the fixed length every clip (target and reference alike) is
  resampled to before encoding, matching `SemanticAnchor.phase_length`'s
  own docstring and Devanshi Kashyap's `NUM_FRAMES=16` in
  `scripts/smoke_test_two_clips.py` (confirmed working on Kaggle same day).
  This deviates from the `vjepa2-vitl-fpc64-256` checkpoint's native
  64-frame training config — see AGENT.md DEVIATIONS #6: whether the
  frozen encoder/predictor behaves correctly on a 16-frame input (8
  temporal blocks, not the source-verified 32) is an open technical risk,
  not a resolved fact — it ran without a shape error, which is
  plumbing-compatible, not confirmation the predictions are meaningful at
  this length. Resample method = uniform full-span index subsampling,
  applied **deterministically**. Fixed a real bug found in the process:
  `VideoDataset._load_clip`'s random start-offset crop would have averaged
  unaligned action phases across performers; added an opt-in
  `deterministic=True` mode (`torch.linspace` full-clip-span sampling) used
  only for anchor reference clips, existing random-crop default unchanged.
  New `scripts/build_semantic_anchor.py`: pools other-performer clips for
  an action (`NTURGBDDataset.clips_for_action`), encodes each, extracts the
  real (not predicted) target-region latent, calls
  `SemanticAnchor.from_reference_clips`. **Not yet execution-tested** — no
  real multi-clip pool uploaded to Kaggle yet.

- NTU RGB+D local data audited (`~/Downloads/nturgb+d_rgb/`, 2026-07-14):
  1440 clips, S004, 4 performers, 3 cameras, 2 reps, 60 actions, 5.5GB — all
  filenames verified parseable via the real `parse_ntu_filename`, 0
  unparsed. Found + fixed a camera-view confound: `clips_for_action`
  (`data/ntu_rgbd.py`) now accepts an optional `camera` filter so anchor
  pooling doesn't mix viewpoint variation in with performer-identity
  variation; `scripts/build_semantic_anchor.py` uses the target clip's own
  camera by default. `dataset-metadata.json` written into the local folder
  for `kaggle datasets create`.
- Real Semantic Anchor built successfully (`anchor_A015.pt`, 3 reference
  clips, action "take off jacket") — the actual pooled anchor, not the
  smoke test's one-clip stand-in. **This exact file must be rebuilt**: with
  only 4 total NTU performers, using all 3 non-target performers as
  references left 0 clips for IDS/SCS probe training on that action,
  crashing the spike script. Fixed via `build_semantic_anchor.py`'s new
  `--max-references` (default: always reserve one performer's clips for
  probe training) — verified via a real end-to-end dry run. Rebuild with
  no extra flags needed (the reservation is now the default).
- IDS + SCS implemented (`evaluation/ids.py`, `evaluation/scs.py`, new
  `evaluation/probes.py`): external, read-only linear probes
  (`nn.Linear(embed_dim, num_classes)`, mean-pooled latent tokens — a
  documented decision) trained on a small held-out labeled set, top-1
  accuracy reported against chance. `steering.blind_alpha` decided: 0.3.
  New `scripts/spike_blind_vs_raw.py` runs the actual Day-2 gate
  comparison for one target clip; dry-run-verified (real dataset/anchor/
  probe-training logic, GPU encoder/predictor mocked) but not yet run for
  real — needs Kaggle GPU.

### IN PROGRESS
- Actual Kaggle upload: `dataset-metadata.json`'s `id` field needs the
  user's real Kaggle username, and needs `kaggle` CLI + API credentials
  configured (not available in this session) to run
  `kaggle datasets create -p ~/Downloads/nturgb+d_rgb`. Should be created
  **Private** — NTU RGB+D's usage agreement is research-only, no
  redistribution.
- Running `scripts/spike_blind_vs_raw.py` for real (needs GPU) and reading
  actual IDS/SCS numbers — this is the literal Day-2 gate check.

### NOT STARTED
- data curation (NTU subset, UCF101) + Kaggle dataset upload; dataset index
  builders are still TODO stubs
- LLM Overseer Job 1 (`build_target`) and Job 2 (`schedule_correction`)
  prompt templates — interfaces exist, prompting logic is TODO
- PCA reference set (still TODO in configs/base.yaml — genuinely open, not
  invented; phase length T and blind_alpha are now decided, see COMPLETED
  above)
- PCS (interfaces exist, delta-comparison method is TODO — IDS/SCS are now
  implemented, PCS is not)
- NN retrieval + PCA overlay implementations (interfaces exist, distance
  metric / reference set / rendering are TODO)
- **Day-2 spike (GATE)** — script exists and is dry-run-verified, but has
  not been executed for real yet
- three-arm experiment, ablation (alpha)
- figures, writing

### UNKNOWN (still genuinely open, not yet decided)
- final rollout length / whether autoregressive chaining is used
- final probe architecture details
- whether SSv2 makes the timeline
- Ollama model choice (`overseer.model_name`)
- PCA reference set (phase-alignment length T and Blind arm's fixed alpha
  are now decided — 16 and 0.3 — see COMPLETED above)

---

## Gate
**Day-2 spike is a hard gate.** On a few NTU clips: measure IDS + SCS on Raw, apply
Blind averaging, re-measure. Green if IDS drops and SCS holds. If red, pivot to
**audit-only** (measure drift, no intervention) — a legitimate WiCV paper.

## Datasets — NOT STARTED
NTU RGB+D 120 (curated subset), Something-Something v2 (subset, if time), UCF101.
Nothing downloaded, curated, or uploaded yet.

## Arms — Raw + Blind IMPLEMENTED, LLM NOT IMPLEMENTED
Raw (`arms/raw.py`, alpha=0 — implemented, trivial), Blind (`arms/blind.py`,
fixed alpha=0.3 — implemented, `steering.blind_alpha` now set), LLM
(`arms/llm.py`, scheduled alpha — delegates to `LLMOverseer.schedule_correction`,
which is still TODO).
(Dropped pending sign-off: LoRA-Fine-Tuned, Prompt-Conditioned — see
`arms/_pending_signoff/README.md`.)

## Metrics — IDS + SCS IMPLEMENTED, PCS NOT IMPLEMENTED
IDS (performer probe) and SCS (action probe): implemented, `evaluation/probes.py`
(linear probe, mean-pooled latent tokens, trained on a small held-out labeled set —
see `scripts/spike_blind_vs_raw.py`). PCS (latent-delta vs real): still
`NotImplementedError`, delta comparison method not chosen. All on latents.

## Experiments — none run. No benchmarks, ablations, or results exist.

## Anti-Hallucination
Never mark anything COMPLETED that is not. Distinguish documented decisions from open
unknowns. If absent: **"Not specified by project definition."**
