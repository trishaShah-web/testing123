# SKILLS.md

## Purpose
Skills needed to build **Zero-Shot Semantic Steering for Video World Models** (revised
design; see AGENT.md DEVIATIONS).

## Scope
In scope: frozen V-JEPA inference, video dataset curation, latent handling, PyTorch
inference, LLM prompting (text-only controller), latent-based evaluation, latent-based
visualization (NN retrieval, PCA overlay), experiment tracking.

Explicitly excluded: training/fine-tuning any model (incl. V-JEPA), diffusion decoders,
RL, reward modeling, online learning. (Note: the excluded "diffusion decoder" is exactly
why there is no pixel-generation path.)

---

## Skill Categories

### 1. Frozen V-JEPA inference
Encoder -> predictor flow, inference only. No training. Load ViT-L, run masked
future prediction, extract predicted latents.

### 2. Video dataset curation
Curate small subsets (NTU: same action x many performers; UCF101 control). Frame
sampling per model (64 fpc). Upload once as Kaggle datasets to avoid re-download.

### 3. Latent handling
Anchor construction (cross-performer mean, phase-matched), cosine drift, the guidance
arithmetic, caching the rollout once per clip per arm. No latent optimization/training.

### 4. PyTorch inference on Kaggle
Frozen model loading, VRAM budgeting (ViT-L + quantized 7B on a T4/P100), deterministic
inference, results checkpointing across 9h sessions.

### 5. LLM prompting (text-only controller)
Prompt design for Job 1 (identity-free action description -> anchor selection) and Job 2
(action label + scalar drift -> nudge decision + alpha) via Ollama. Contributors must
NOT attempt to feed latents to the LLM.

### 6. Latent-based visualization
NN retrieval (nearest real frame per steered latent -> TorchCodec -> mp4, labeled
retrieval) and PCA overlay (shared basis -> RGB -> patch grid -> overlay -> heatmap mp4).

### 7. Evaluation methodology
IDS (performer probe), SCS (action probe), PCS (latent-delta vs real). Report all three,
every experiment. Understand the joint-success rule.

---

## Non-Negotiable
- Report IDS, SCS, PCS every experiment.
- Never train/fine-tune V-JEPA; never feed latents to the LLM; never add a trained decoder.

## Anti-Hallucination
Never invent skills, datasets, metrics, or a decoder path. If absent:
**"Not specified in project definition."**
