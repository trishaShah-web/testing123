"""Arm 2: Blind — fixed alpha at every step, regardless of drift or action
(AGENT.md "Baselines"). Tests whether dumb averaging toward the anchor helps
at all, independent of any reasoning about when/how much to nudge.

`alpha` is a documented decision (INSTRUCTIONS.md "Documented-Decision
Rules"), logged in configs/base.yaml under `steering.blind_alpha` — not
invented here, and not yet chosen (see STATUS.md UNKNOWN).
"""

from __future__ import annotations


class BlindController:
    def __init__(self, alpha: float):
        if alpha is None:
            raise ValueError(
                "steering.blind_alpha is not set in config — TODO: not yet "
                "chosen, must be an explicit documented decision before this "
                "arm can run (see STATUS.md UNKNOWN)."
            )
        self.alpha = alpha

    def get_alpha(self, action_label: str, drift: float, step: int) -> float:
        return self.alpha
