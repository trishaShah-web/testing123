# CLAUDE.md

One project: **Zero-Shot Semantic Steering for Video World Models** (revised design after
feasibility review).

## Read First
- **HANDOFF.md** — session-to-session engineering log: what's been built, exact
  commands, verified-vs-untested facts, and where to resume. Read this FIRST when
  picking up work on this repo.
- **AGENT.md** — problem, hypotheses, contributions, LLM role, arms, metrics, and the
  DEVIATIONS block (what changed from the original locked spec and why).
- **ARCHITECTURE.md** — components (decoder removed), data flow, the three output
  branches.
- **STATUS.md** — current state; the Day-2 spike gate.
- **INSTRUCTIONS.md** — non-negotiable rules, hard correctness constraints, claim rules.
- **SKILLS.md** — required skills.

If two files conflict, stop and flag it — do not resolve silently.

## Absolute Rules (summary)
- V-JEPA is frozen: never train, fine-tune, or modify it.
- Steering is inference-time, latent-space arithmetic:
  `steered = predicted + alpha*(anchor - predicted)`.
- The LLM is a **text-only controller** (targets + schedules the nudge). It never reads
  latents or emits a latent vector.
- **No pixel decoder.** Visualization is latent-based: NN retrieval (labeled retrieval,
  not generation) and shared-basis PCA overlay.
- Anchor = cross-performer mean latent (same action, other performers).
- Every experiment reports IDS, SCS, PCS. Success is the joint condition.
- No extra datasets/baselines without sign-off.

## Honesty Rules (these matter for acceptance)
- Do not overclaim the LLM's role. Say it schedules/targets the correction; do not say it
  "steers latents" or "monitors latent semantics."
- Report negative results (e.g. Blind == LLM) honestly.
- If the Day-2 spike fails, ship the pre-approved audit-only paper.
- Distinguish documented decisions from open unknowns; mark genuine gaps TODO, never
  invent.

## How to Behave
Act as PI / research lead / reviewer. Prioritize scientific validity, reproducibility,
interpretability, minimal assumptions. Reject anything that trains V-JEPA, feeds latents
to the LLM, adds a decoder, adds datasets/baselines without sign-off, or overclaims.
