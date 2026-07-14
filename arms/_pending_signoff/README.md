# Pending sign-off

These two arms are **not part of the current 7-day scope**. They are dropped
per AGENT.md DEVIATIONS #4:

> Baselines changed to Raw / Blind / LLM. LoRA-Fine-Tuned and
> Prompt-Conditioned V-JEPA are dropped for the current 7-day scope;
> reinstating them requires sign-off and a training budget the current
> timeline does not have.

`lora_finetuned_vjepa.py` additionally trains a copy of the V-JEPA world
model via LoRA adapters — reinstating it must not be read as license to
train the frozen Actor used by the Overseer-Actor pipeline
(`vjepa/world_model.py`); that model stays frozen regardless.

Both files still import `VJEPADecoder`, which no longer exists
(ARCHITECTURE.md: no pixel decoder). They will need rework in addition to
sign-off before they can run again.

Do not import from this directory in active code. Reinstating requires
author + mentor sign-off (INSTRUCTIONS.md rule 7).
