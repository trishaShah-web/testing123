# AGENT.md

## Project Title

Zero-Shot Semantic Steering for Video World Models

## Authority of This Document

This file is derived from the project specification as revised after a feasibility
review. Anything not explicitly stated is treated as UNKNOWN and must never be
invented, assumed, inferred, or hallucinated. Where information is genuinely absent,
state: **"Not specified by project definition."**

Where this file deviates from the original locked specification, the change is listed
in the DEVIATIONS block below. Deviations require author + mentor sign-off; until then
they are documented decisions, not silent edits.

---

## DEVIATIONS FROM ORIGINAL SPEC (require sign-off)

1. **The "Frozen V-JEPA Decoder" component is removed.** V-JEPA (v1 and 2, as
   released) has no pixel decoder — it predicts in latent space and stops. The only
   pixel path in the original paper was a *separately trained conditional diffusion
   decoder*, which is (a) a training step and (b) a diffusion module, both explicitly
   forbidden by the original spec. Evaluation and visualization are therefore
   latent-based, not decoded-frame-based. See ARCHITECTURE.md.
2. **The LLM Overseer does not read latents or operate in latent space.** V-JEPA
   latents are not language-aligned; no LLM can read them and none can emit a
   latent-space vector. The LLM's role is redefined as a text-only controller (see
   "LLM role" below). The claim "LLM continuously monitors semantic consistency of
   the latent rollout" from the original spec is false and is removed.
3. **The Semantic Anchor is a cross-performer mean latent, not a language embedding.**
   It is built from the same action performed by other performers (NTU), averaged at
   matched action phases. This keeps the anchor in V-JEPA's own space, so drift and
   nudging are well-defined without a cross-space bridge.
4. **Baselines changed** to Raw / Blind / LLM (see Baselines). LoRA-Fine-Tuned and
   Prompt-Conditioned V-JEPA are dropped for the current 7-day scope; reinstating them
   requires sign-off and a training budget the current timeline does not have.
5. **An audit-only fallback is defined.** If the Day-2 spike fails, the project ships
   as a measurement paper (identity leaks into the rollout) with no intervention.
6. **Input clip length is 16 frames, not the checkpoint's native 64-frame (fpc64)
   config.** `steering.phase_length` (configs/base.yaml) = T = 16 raw frames every
   clip — target and reference alike — is resampled to before encoding (Devanshi
   Kashyap, `scripts/smoke_test_two_clips.py`, 2026-07-14; confirmed working on
   Kaggle same day). The checkpoint in use is still `facebook/vjepa2-vitl-fpc64-256`
   (ARCHITECTURE.md component 1) — its name reflects its *training* config (64
   frames-per-clip), not a hard requirement we've independently verified at
   inference time. Whether the frozen encoder/predictor's positional embeddings
   behave correctly on a 16-frame input (8 temporal blocks instead of the
   source-verified 32, `vjepa/masking.py`) has not been checked against primary
   source the way the 64-frame case was — it empirically ran without a shape
   error, which is evidence it's *plumbing-compatible*, not confirmation the
   predictions are meaningful at this length. Flagged as an open technical risk,
   not a resolved one.

---

## Problem

Video world models such as V-JEPA predict the future of an action in latent space.
Their predictions can drift toward performer-specific appearance information (identity,
clothing, build, viewpoint) rather than the action itself. We call this **Forecast
Drift**: *"future predictions becoming more dependent on performer-specific appearance
information rather than action semantics."*

We assume the model already understands the action and that drift is a failure of
focus, not of capability.

## Scientific Nugget

Bias in a world model's *forecast* can be reduced at inference time, with no retraining,
by pulling the predicted trajectory toward an identity-free version of the same action
built from other performers. Whether a language model reasoning about the action can
schedule that correction better than a fixed rule is the question we test.

## Primary Hypothesis

Nudging the predicted latent trajectory toward a cross-performer action anchor reduces
measured identity leakage in the forecast without retraining the world model.

## Secondary Hypothesis

An LLM controller that decides *when and how strongly* to nudge (from the action label
and a scalar drift signal) preserves action semantics and motion realism better than a
fixed nudge schedule.

## Contributions

1. We identify and measure **Forecast Drift** — performer identity leaking into and
   compounding through a frozen video world model's predicted future, distinct from
   prior work that audits the *encoder* of an already-observed clip.
2. We propose a **zero-shot, latent-space steering** method: a cross-performer action
   anchor plus an LLM controller that schedules the correction. No V-JEPA weights are
   changed.
3. We evaluate identity leakage, action preservation, and motion realism across a
   Raw / Blind / LLM ladder, and report honestly where the LLM does and does not beat
   a fixed rule.

---

## LLM role (precise)

The LLM is a **controller**, not a corrector. It reads text, writes decisions. It never
touches a latent.

- **Job 1 — build the target (pre-rollout).** From the action label, the LLM describes
  the action's identity-free sub-motions; this decides which reference clips pool into
  the cross-performer anchor.
- **Job 2 — schedule the correction (per step).** Code computes a scalar drift `d`; the
  LLM receives the action label + `d` (+ step index) as text and returns whether to
  nudge and the strength alpha.

The LLM does NOT: read V-JEPA latents; compute the guidance vector (that is
`anchor - latent`, pure arithmetic); or monitor latent-space semantics.

Known reviewer risk: Job 2 alone could be a threshold rule. The LLM's harder-to-replace
value is Job 1. Justify the LLM there.

---

## Architecture

Two entities: **Overseer** (LLM controller, text-only) and **Actor** (frozen V-JEPA).
See ARCHITECTURE.md for the full component list and data flow.

---

## Datasets

- NTU RGB+D 120 — primary; same action, many performers (builds the anchor + the
  counterfactual). Curated subset only (full RGB is ~1TB+).
- Something-Something v2 — intent-sensitivity check (secondary; if time).
- UCF101 — natural-drift control.

No additional datasets without sign-off.

---

## Baselines / experimental arms

1. **Raw** — frozen V-JEPA, alpha = 0. Establishes the drift baseline.
2. **Blind** — fixed alpha at every step. Tests whether dumb averaging helps.
3. **LLM** — LLM-scheduled alpha toward an LLM-built anchor. Tests whether reasoning
   beats a fixed rule.

(Dropped from original spec, pending sign-off: LoRA-Fine-Tuned V-JEPA,
Prompt-Conditioned V-JEPA.)

---

## Evaluation metrics (all computed on latents)

Every experiment reports all three.

### IDS — Identity Drift Score
Linear probe predicting *performer* from the predicted future latent; report top-1
accuracy vs. chance. Lower after steering = less identity leakage. Probe is external and
read-only; V-JEPA stays frozen (document this — it is not a violation of "no training").

### SCS — Semantic Consistency Score
Linear probe predicting *action class* from the predicted future latent; higher = action
preserved through the nudge.

### PCS — Physics Consistency Score
Compare step-to-step latent deltas of the steered trajectory against real clips' deltas;
large deviation = off-manifold / broken motion.

### Metric rules
- All three reported every experiment.
- Improving one while catastrophically degrading another is failure.
- Success is the joint condition: IDS down, SCS held, PCS held.

The specific probe architecture, similarity function, phase-alignment method, and
normalization are documented decisions (see repo config), not silent assumptions.

---

## Agent role and priorities

Acts as PI / research lead / reviewer. Priorities in order: scientific validity,
reproducibility, interpretability, minimal assumptions. Rejects: additional datasets or
baselines without sign-off, additional training of V-JEPA, unsupported claims,
overclaiming the LLM's role, and any assumption not documented as a decision.

## Anti-Hallucination Rules

Never invent datasets, baselines, modules, or results. Distinguish "documented decision"
(a choice we made and logged) from "not specified" (genuinely open). If absent, state:
**"Not specified by project definition."**
