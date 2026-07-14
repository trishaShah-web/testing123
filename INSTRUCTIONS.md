# INSTRUCTIONS.md

## For Contributors

One project: **Zero-Shot Semantic Steering for Video World Models** (revised design;
see AGENT.md DEVIATIONS). These rules govern the repo.

## Authority
The revised spec files are the source of truth. Distinguish **documented decisions**
(chosen and logged) from **open unknowns** (mark TODO). Never silently assume. Where
absent: **"Not specified by project definition."**

---

## Non-Negotiable Rules
1. Never modify, train, or fine-tune V-JEPA. It stays frozen.
2. Steering only at inference, only in latent space.
3. Guidance is arithmetic on latents (`steered = predicted + alpha*(anchor - predicted)`);
   it never edits any output artifact or model weight.
4. Method stays zero-shot. External **read-only** probes for metrics are allowed and
   must be documented; they never touch V-JEPA.
5. The LLM is a text-only controller. It must never be coded to read latents or emit a
   latent vector. If a PR does this, reject it.
6. No pixel decoder is trained or added. Visualization is latent-based (NN retrieval,
   PCA overlay) only.
7. No additional datasets or baselines without author + mentor sign-off.

---

## Allowed Datasets
NTU RGB+D 120 (curated subset), Something-Something v2 (subset), UCF101.

## Experimental Arms
Raw / Blind / LLM. (LoRA-Fine-Tuned and Prompt-Conditioned are out of current scope;
reinstating needs sign-off and a training budget.)

---

## Hard Constraints (correctness, not style)
- **Shared PCA basis:** fit PCA once on a fixed reference set; reuse across all arms.
  Refit-per-clip makes colors non-comparable and is a bug.
- **Compute the rollout once** per clip per arm; the metrics, NN, and PCA branches all
  consume the same cached steered latents.
- **NN frames are retrieval, not generation** — must be labeled so everywhere.
- **Joint success only:** IDS down AND SCS held AND PCS held. One metric improving while
  another collapses is a failure, not a result.

---

## Documented-Decision Rules
Log in repo config: model variant, rollout length, phase-alignment method, probe
architecture, anchor pooling set, alpha values, PCA reference set. Each is a decision to
be defended, not hidden. Missing genuinely-open details: mark **TODO**, do not invent.

---

## Research Claim Rules
Claims need: evidence, comparison against the Raw/Blind/LLM arms, reproducibility,
documented methodology. Do not overclaim the LLM's role — it schedules and targets the
correction; it does not "steer latents" or "monitor latent semantics." Overclaiming is
the fastest path to rejection.

## Fallback
If the Day-2 spike fails, ship the audit-only paper. This is pre-approved, not a defeat.

## Anti-Hallucination
Never invent datasets, baselines, results, or a decoder. If absent, state:
**"Not specified by project definition."**
