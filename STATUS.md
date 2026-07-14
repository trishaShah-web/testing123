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

### IN PROGRESS
- Day-1 smoke test: run `scripts/smoke_test_single_clip.py` on one real NTU
  clip to confirm the source-verified loading + predictor call above
  actually work. Nothing downstream should be trusted until this passes.
  Highest-risk unverified assumption going in: the exact tensor layout
  `VJEPAEncoder.forward()` expects (script currently guesses
  `(1, C, T, H, W)`) — flagged prominently in the script's own docstring as
  the first thing to check on a shape-mismatch failure.

### NOT STARTED
- data curation (NTU subset, UCF101) + Kaggle dataset upload; dataset index
  builders are still TODO stubs
- anchor construction inputs: phase-alignment/resampling implementation,
  reference-clip pooling (SemanticAnchor.from_reference_clips itself is
  implemented — it needs real phase-aligned tensors to consume)
- LLM Overseer Job 1 (`build_target`) and Job 2 (`schedule_correction`)
  prompt templates — interfaces exist, prompting logic is TODO
- Blind arm alpha value, phase length T, PCA reference set (all TODO in
  configs/base.yaml — genuinely open, not invented)
- IDS/SCS/PCS probes (interfaces exist, probe architecture is TODO)
- NN retrieval + PCA overlay implementations (interfaces exist, distance
  metric / reference set / rendering are TODO)
- **Day-2 spike (GATE)**
- three-arm experiment, ablation (alpha)
- figures, writing

### UNKNOWN (still genuinely open, not yet decided)
- final rollout length / whether autoregressive chaining is used
- final probe architecture details
- whether SSv2 makes the timeline
- Ollama model choice (`overseer.model_name`)
- Blind arm's fixed alpha, phase-alignment length T, PCA reference set

---

## Gate
**Day-2 spike is a hard gate.** On a few NTU clips: measure IDS + SCS on Raw, apply
Blind averaging, re-measure. Green if IDS drops and SCS holds. If red, pivot to
**audit-only** (measure drift, no intervention) — a legitimate WiCV paper.

## Datasets — NOT STARTED
NTU RGB+D 120 (curated subset), Something-Something v2 (subset, if time), UCF101.
Nothing downloaded, curated, or uploaded yet.

## Arms — scaffolded, NOT IMPLEMENTED
Raw (`arms/raw.py`, alpha=0 — implemented, trivial), Blind (`arms/blind.py`,
fixed alpha — implemented, but `steering.blind_alpha` is still unset),
LLM (`arms/llm.py`, scheduled alpha — delegates to `LLMOverseer.schedule_correction`,
which is still TODO).
(Dropped pending sign-off: LoRA-Fine-Tuned, Prompt-Conditioned — see
`arms/_pending_signoff/README.md`.)

## Metrics — defined, NOT IMPLEMENTED
IDS (performer probe), SCS (action probe), PCS (latent-delta vs real). All on latents.

## Experiments — none run. No benchmarks, ablations, or results exist.

## Anti-Hallucination
Never mark anything COMPLETED that is not. Distinguish documented decisions from open
unknowns. If absent: **"Not specified by project definition."**
