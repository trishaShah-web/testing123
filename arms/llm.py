"""Arm 3: LLM — LLM-scheduled alpha toward an LLM-built anchor pool
(AGENT.md "Baselines"). Tests whether reasoning about the action beats a
fixed rule (arms/blind.py).

Thin adapter: delegates to LLMOverseer Job 2 (overseer/llm_overseer.py),
which is the only place the actual text-only controller logic lives. This
class exists so the pipeline can treat all three arms uniformly via
arms/controller.py's AlphaController interface.
"""

from __future__ import annotations

from overseer import LLMOverseer


class LLMController:
    def __init__(self, overseer: LLMOverseer):
        self.overseer = overseer

    def get_alpha(self, action_label: str, drift: float, step: int) -> float:
        decision = self.overseer.schedule_correction(action_label, drift, step)
        return decision.alpha if decision.should_nudge else 0.0
