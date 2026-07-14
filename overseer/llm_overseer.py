"""LLM Overseer — Component 3 of 7 (ARCHITECTURE.md).

Role: text-only controller. Reads text, writes decisions. Never touches a
latent — see AGENT.md DEVIATIONS #2 and "LLM role (precise)", and
INSTRUCTIONS.md rule 5 ("must never be coded to read latents or emit a
latent vector. If a PR does this, reject it.").

Two jobs, both pure text in / text out:

- Job 1 — build_target (pre-rollout): from the action label, describe the
  action's identity-free sub-motions; this decides which reference clips
  pool into the cross-performer anchor (overseer/semantic_anchor.py).
- Job 2 — schedule_correction (per step): given the action label + a scalar
  drift value `d` (+ step index) as text, return whether to nudge and the
  strength alpha. `d` is computed upstream by overseer/drift_detection.py
  from latents; only the resulting number ever reaches this class.

Served via Ollama (documented decision, configs/base.yaml `overseer.*`).
"""

from __future__ import annotations

from dataclasses import dataclass

import ollama


@dataclass
class AnchorTarget:
    """Job 1 output: identity-free sub-motion description used to select
    which reference clips (other performers, same action) pool into the
    cross-performer anchor. Does not itself contain a latent or embedding.
    """
    action_label: str
    submotion_description: str


@dataclass
class NudgeDecision:
    """Job 2 output: whether to nudge this step, and how strongly."""
    should_nudge: bool
    alpha: float


class LLMOverseer:
    """Text-only controller. Constructor and methods only ever accept/return
    strings, numbers, and dataclasses of those — never a tensor.
    """

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        if model_name is None:
            raise ValueError(
                "overseer.model_name is not set in config — TODO: not "
                "specified by project definition, must be chosen explicitly "
                "(see STATUS.md UNKNOWN)."
            )
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def build_target(self, action_label: str) -> AnchorTarget:
        """Job 1. action label (text) -> identity-free sub-motion description
        (text), used upstream to select which NTU reference clips pool into
        the cross-performer anchor (overseer/semantic_anchor.py).

        TODO: not specified by project definition — exact prompt template is
        an open implementation detail (see SKILLS.md #5), not invented here.
        """
        raise NotImplementedError(
            "LLM Overseer Job 1 (build_target) is TODO — prompt template for "
            "identity-free sub-motion description is not specified by "
            "project definition."
        )

    def schedule_correction(
        self, action_label: str, drift: float, step: int
    ) -> NudgeDecision:
        """Job 2. action label (text) + scalar drift `d` (text) + step index
        (text) -> nudge decision + alpha. `drift` must already be a plain
        float computed by overseer/drift_detection.py — this method must
        never be passed a latent tensor.

        TODO: not specified by project definition — exact prompt template
        and alpha range/quantization are open implementation details (see
        SKILLS.md #5), not invented here.
        """
        raise NotImplementedError(
            "LLM Overseer Job 2 (schedule_correction) is TODO — prompt "
            "template and alpha selection are not specified by project "
            "definition."
        )
