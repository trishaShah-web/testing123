"""Shared controller interface for the three experimental arms (AGENT.md
"Baselines / experimental arms").

Each arm supplies `alpha` in the steering equation
(ARCHITECTURE.md data flow):

    steered = predicted + alpha * (anchor - predicted)

All three arms share this one interface so the rest of the pipeline
(pipeline/inference_loop.py) does not need to know which arm is active —
it just calls `get_alpha(action_label, drift, step)` every step.
"""

from __future__ import annotations

from typing import Protocol


class AlphaController(Protocol):
    """alpha = controller(action_label, drift, step) — see ARCHITECTURE.md."""

    def get_alpha(self, action_label: str, drift: float, step: int) -> float:
        ...
