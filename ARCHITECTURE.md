# ARCHITECTURE.md

## System Name

Zero-Shot Semantic Steering for Video World Models

## Authority of This Document

Revised after feasibility review. Deviations from the original locked architecture are
listed in AGENT.md's DEVIATIONS block and require sign-off. Where information is
genuinely absent, state: **"Not specified in project definition."** Where a value was
chosen, it is a **documented decision**, not an assumption.

## System Purpose

Reduce measured Forecast Drift in a frozen video world model's predicted future, at
inference time, in latent space only.

---

## System Components

Seven components. Component 3 (the pixel decoder) from the original spec is **removed** —
V-JEPA has no pixel decoder. It is replaced by two latent-based visualization paths.

1. Frozen V-JEPA Encoder
2. Frozen V-JEPA World Model (predictor)
3. LLM Overseer (text-only controller)
4. Semantic Anchor (cross-performer mean latent)
5. Drift Detection (scalar)
6. Guidance Vector (latent arithmetic)
7. Visualization + Evaluation outputs (latent-based)

---

## Data Flow

```
Input Video
   -> TorchCodec (decode mp4 -> frames)
   -> Frozen V-JEPA Encoder
   -> Observed Latent Tokens
   -> Frozen V-JEPA World Model (mask future, predict)
   -> Predicted Future Latent Tokens
   -> SEMANTIC STEERING:
        anchor = mean latent of same action, other performers (phase-matched)
        d = 1 - cos(predicted, anchor)                 # scalar drift
        alpha = controller(action_label, d, step)      # Raw:0 / Blind:fixed / LLM:scheduled
        steered = predicted + alpha * (anchor - predicted)
   -> Steered Future Latent Tokens
   -> fork into three consumers (compute rollout ONCE per clip per arm, then reuse):
        (a) METRICS   : IDS / SCS / PCS probes on latents   <- paper evidence
        (b) NN RETRIEVAL : nearest real frame per step -> TorchCodec -> mp4 (label: retrieval)
        (c) PCA OVERLAY  : project D->3 (SHARED basis) -> RGB -> patch grid -> overlay -> heatmap mp4
```

---

## Component Definitions

### 1. Frozen V-JEPA Encoder
Frames -> observed latent tokens. Frozen. Decision: ViT-L (`facebook/vjepa2-vitl-fpc64-256`).

### 2. Frozen V-JEPA World Model (predictor)
Masks future tokens, predicts them from observed context. Frozen. Rollout length is a
documented decision; start with single future-chunk prediction, autoregressive chaining
is a stretch goal, not assumed.

### 3. LLM Overseer (text-only controller)
Inputs: action label (text); scalar drift `d` (text); step index (text).
Outputs: (Job 1) identity-free action description used to select anchor reference clips;
(Job 2) nudge decision + alpha.
Never reads latents. Never emits a latent vector. Never computes the guidance vector.
Served via Ollama (documented decision).

### 4. Semantic Anchor (cross-performer mean latent)
Same action, other performers (Q != P), latents averaged at matched action phases.
Phase alignment: resample every clip to fixed length T (documented decision; DTW is a
stretch). Lives in V-JEPA latent space. Fixed during a given rollout.

### 5. Drift Detection (scalar)
`d = 1 - cos(predicted_latent, anchor)`. One number per step, handed to the controller.
Threshold / trigger behavior lives in the controller, not here.

### 6. Guidance Vector (latent arithmetic)
`g = alpha * (anchor - predicted)`; `steered = predicted + g`. Pure arithmetic. Never
changes weights, never edits any decoded output.

### 7. Visualization + Evaluation outputs
- Metrics (IDS/SCS/PCS) — the evidence; see AGENT.md.
- NN retrieval video — closest *real* frame to each steered latent, stitched to mp4.
  Must be labeled **retrieval, not generation**.
- PCA overlay heatmap — PCA fit ONCE on a fixed reference set, reused across
  Raw/Blind/LLM. Refit-per-clip breaks cross-arm comparability (hard constraint).

---

## Non-Negotiable Rules
- V-JEPA weights never modified; no training, no fine-tuning of V-JEPA.
- Steering only at inference, only in latent space.
- Guidance never edits any output artifact or model parameter.
- Method stays zero-shot (external read-only probes for metrics are allowed and
  documented; they do not touch V-JEPA).
- No pixel decoder is trained or added. Visualization is latent-based only.

## Forbidden Content
Do not add: a trained decoder, diffusion modules, adapters, extra intelligent entities,
losses, or additional datasets/baselines without sign-off. If absent, state:
**"Not specified in project definition."**
