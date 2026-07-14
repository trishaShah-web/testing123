"""Arm 1: Raw — frozen V-JEPA, alpha = 0 always (AGENT.md "Baselines").

Establishes the drift baseline: no steering is applied, so
`steered == predicted` at every step. Exists so the pipeline can run the
exact same encode/predict/metrics/visualization path for all three arms
with no special-casing.
"""

from __future__ import annotations


class RawController:
    def get_alpha(self, action_label: str, drift: float, step: int) -> float:
        return 0.0
