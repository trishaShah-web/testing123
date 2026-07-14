"""Enforces the repository's mandatory evaluation rule:

Every experiment must report SCS, IDS, and PCS. Reporting only a subset is
an incomplete evaluation (AGENT.md, INSTRUCTIONS.md, SKILLS.md).

This module encodes that rule structurally rather than relying on convention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentReport:
    experiment_name: str
    config_snapshot: dict
    scs: Optional[float] = None
    ids: Optional[float] = None
    pcs: Optional[float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def validate(self) -> None:
        missing = [
            name for name, value in (("SCS", self.scs), ("IDS", self.ids), ("PCS", self.pcs))
            if value is None
        ]
        if missing:
            raise ValueError(
                f"Incomplete evaluation: missing {missing}. Every experiment "
                "must report SCS, IDS, and PCS — see AGENT.md 'Metric Rules'."
            )

    def save(self, output_dir: str | Path) -> Path:
        self.validate()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.experiment_name}_{int(self.timestamp)}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
