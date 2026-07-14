"""Shared linear-probe utilities for IDS/SCS (evaluation/ids.py,
evaluation/scs.py).

External, read-only probes only (AGENT.md "Evaluation metrics (all
computed on latents)") — V-JEPA is never touched or backpropagated
through; probes train on already-computed, detached latent features only.

Probe architecture: `nn.Linear(embed_dim, num_classes)` — one linear layer,
full-batch cross-entropy training, no hidden layers, no regularization
search. A documented decision (not a research question, per project
instructions), not derived from any architecture search.

Probe TRAINING DATA is not this module's concern — it just fits/scores
whatever labeled features it's given. The caller (e.g.
scripts/spike_blind_vs_raw.py) is responsible for choosing a held-out
training set that does not overlap with the clip(s) actually being
evaluated (see that script's docstring for the exclusion policy it uses).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """[..., N, D] token sequence -> [..., D] single feature vector via mean
    pooling over the token dimension. A linear probe needs one fixed-size
    vector per example, not a variable-length token sequence; mean pooling
    is the simplest choice (documented decision, not a research question).
    """
    return latent.mean(dim=-2)


@dataclass
class LabelEncoder:
    """Maps arbitrary integer labels (NTU performer IDs, action codes) to
    the contiguous 0..num_classes-1 class indices CrossEntropyLoss requires,
    and back. Fit once on the probe's training labels; the exact same
    encoder must be reused to encode the evaluation label(s) so class
    indices line up.
    """

    classes: list[int]

    @classmethod
    def fit(cls, labels: list[int]) -> "LabelEncoder":
        return cls(classes=sorted(set(labels)))

    def encode(self, labels: list[int]) -> torch.Tensor:
        index = {c: i for i, c in enumerate(self.classes)}
        missing = [l for l in labels if l not in index]
        if missing:
            raise ValueError(
                f"label(s) {missing} were not seen when this LabelEncoder was fit "
                f"(known classes: {self.classes}) — the probe has no way to score them."
            )
        return torch.tensor([index[l] for l in labels], dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


class LinearProbe(nn.Module):
    """One nn.Linear layer, embed_dim -> num_classes. External and
    read-only w.r.t. V-JEPA — trains on frozen, detached latent features
    only; never touches V-JEPA's own parameters or gradients.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    @property
    def num_classes(self) -> int:
        return self.linear.out_features


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 200,
    lr: float = 0.01,
) -> LinearProbe:
    """features: [N, D] pooled, detached latent vectors (see pool_latent).
    labels: [N] long tensor of class indices (see LabelEncoder.encode) —
    NOT raw performer/action IDs. Full-batch gradient descent — N is small
    (a few dozen examples for a Day-2 spike, not a large training set), so
    this is under a second on CPU; no GPU involvement, so it does not
    contend with the encoder's GPU memory.
    """
    features = features.detach().cpu().float()
    labels = labels.detach().cpu().long()
    probe = LinearProbe(embed_dim=features.shape[-1], num_classes=num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(probe(features), labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


@torch.no_grad()
def probe_accuracy(probe: LinearProbe, features: torch.Tensor, labels: torch.Tensor) -> float:
    """features: [D] (one example) or [N, D] (several) pooled latent
    vector(s); labels: matching scalar or [N] class indices (same
    LabelEncoder used for training) -> top-1 accuracy. Compare against
    chance = 1/probe.num_classes, not against zero.
    """
    features = features.detach().cpu().float()
    if features.dim() == 1:
        features = features.unsqueeze(0)
    labels = labels.detach().cpu().long()
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    preds = probe(features).argmax(dim=-1)
    return (preds == labels).float().mean().item()
