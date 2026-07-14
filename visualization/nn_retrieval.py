"""NN Retrieval — Component 7(b) of 7 (ARCHITECTURE.md).

Role: nearest *real* frame per steered latent step, stitched to mp4. This is
retrieval, not generation — there is no decoder, and outputs must be labeled
"retrieval" everywhere they are shown (INSTRUCTIONS.md hard constraint).

Known Unknowns: nearest-neighbor distance metric, size/composition of the
real-frame reference bank searched against, and tie-breaking — not decided
here (see STATUS.md UNKNOWN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def retrieve_nearest_real_frames(
    steered_latent_trajectory: torch.Tensor,
    reference_latent_bank: Any,
) -> list[Path]:
    """steered latent trajectory + bank of (real frame -> real latent) pairs
    -> one real frame path per step (the nearest neighbor by latent
    distance).

    TODO: not specified by project definition — distance metric and
    reference bank composition are open Known Unknowns, not invented here.
    """
    raise NotImplementedError(
        "NN retrieval is TODO — distance metric and reference bank are not "
        "specified by project definition."
    )


def stitch_to_mp4(frame_paths: list[Path], output_path: Path, label: str = "retrieval") -> Path:
    """Stitch retrieved frames into an mp4, tagged with `label` so it is
    never mistaken for a generated/decoded video (hard constraint).

    TODO: not specified by project definition — output fps/encoding are not
    yet chosen.
    """
    raise NotImplementedError(
        "NN retrieval video stitching is TODO — output encoding parameters "
        "are not specified by project definition."
    )
