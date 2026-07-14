"""Semantic Consistency Score (SCS).

Purpose: measure whether predicted futures preserve intended action semantics.
Higher values indicate better semantic preservation.

Computed entirely on latents (AGENT.md "Evaluation metrics (all computed on
latents)") via an external, read-only linear probe predicting *action class*
from the predicted future latent; higher = action preserved through the
nudge (AGENT.md SCS definition).

Probe architecture decided 2026-07-14 (evaluation/probes.py): a single
`nn.Linear(embed_dim, num_classes)` over mean-pooled latent tokens — a
documented decision, not a research question. Aggregation: none beyond
mean-pooling the token dimension per example; SCS is the plain top-1
accuracy across whatever example(s) are passed in. Normalization: none —
compare the returned value against chance (`1/probe.num_classes`), which
this function does not compute for you (see LinearProbe.num_classes).

Probe TRAINING is a separate step from this function — see
evaluation/probes.py train_linear_probe and scripts/spike_blind_vs_raw.py
for where the training set/labels come from and how eval-set leakage is
avoided. Training the probe and scoring the same example here would leak
that example into its own training set.
"""

from __future__ import annotations

import torch

from .probes import LinearProbe, pool_latent, probe_accuracy


def compute_scs(predicted_latent_trajectory: torch.Tensor, action_labels: torch.Tensor, probe: LinearProbe) -> float:
    """predicted (or steered) latent trajectory, shape [..., N, D] (one or
    more examples, token dim second-to-last) + ground-truth action label(s)
    as class indices from the SAME LabelEncoder used to train `probe`
    (evaluation/probes.py LabelEncoder.encode — NOT raw NTU action codes) +
    an already-trained external LinearProbe -> SCS (top-1 accuracy).
    Compare against `1/probe.num_classes`, not zero.
    """
    features = pool_latent(predicted_latent_trajectory)
    return probe_accuracy(probe, features, action_labels)
