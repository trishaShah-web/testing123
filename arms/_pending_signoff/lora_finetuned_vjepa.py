"""Baseline 2: LoRA Fine-Tuned V-JEPA.

The only place in this repository where any V-JEPA weights are updated —
this trains a separate copy of the world model via LoRA adapters, purely as
a comparison baseline. It does not affect the frozen Actor used by the
Overseer-Actor pipeline (vjepa/world_model.py).

TODO: not specified by project definition — adapter target layers, rank,
alpha, learning rate, training data split, and number of steps are all
unresolved and must be decided explicitly before training can run.
"""

from __future__ import annotations

import torch.nn as nn


class LoRAFinetunedVJEPABaseline:
    def __init__(self, checkpoint_path: str, lora_config: dict | None = None):
        if lora_config is None:
            raise ValueError(
                "lora_config is required — TODO: not specified by project "
                "definition (rank, alpha, target modules, lr, etc.)."
            )
        self.checkpoint_path = checkpoint_path
        self.lora_config = lora_config
        self.model = self._build_lora_model()

    def _build_lora_model(self) -> nn.Module:
        # TODO: not specified by project definition — instantiate the V-JEPA
        # world model and wrap target modules with LoRA adapters (e.g. via
        # `peft`), once target layers/rank/alpha are explicitly decided.
        raise NotImplementedError(
            "LoRA baseline construction is TODO pending explicit adapter "
            "configuration decisions."
        )

    def train(self, dataloader, num_steps: int):
        # TODO: not specified by project definition — training loop,
        # optimizer, loss function, and schedule are all unresolved.
        raise NotImplementedError(
            "LoRA baseline training is TODO pending explicit training "
            "configuration decisions."
        )
