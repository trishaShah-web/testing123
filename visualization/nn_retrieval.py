"""NN Retrieval — Component 7(b) of 7 (ARCHITECTURE.md).

Role: nearest *real* frame per steered latent step. This is retrieval, not
generation — there is no decoder, and outputs must be labeled "retrieval"
everywhere they are shown (INSTRUCTIONS.md hard constraint).

Documented decisions (approved 2026-07-19, consistent with the rest of the
codebase rather than invented fresh):
- Distance metric: cosine, matching overseer/drift_detection.py's existing
  `1 - cos(...)` convention (drift already uses cosine; reusing it here
  keeps "distance in latent space" meaning one thing across the project).
- Reference bank composition: built by the caller
  (scripts/visualize_steering.py) from the same anchor's `reference_paths` —
  no new dataset pull, no separate reference-bank curation decision.
- Granularity: one comparison per TARGET TEMPORAL BLOCK (a tubelet-sized
  group of tokens), not per raw frame — a V-JEPA token corresponds to a
  tubelet (tubelet_size=2 raw frames), not a single frame, so "per step"
  here means "per temporal block," represented on screen by that block's
  first raw frame (same convention visualization/pca_overlay.py uses for
  its own per-block original_frames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


@dataclass
class ReferenceBankEntry:
    """One (real clip, target temporal block) pair available for retrieval."""

    clip_path: Path
    block_index: int          # index within the target region's temporal blocks (0-based)
    frame_path: Path          # saved real frame representing this block (see save_frame_as_image)
    feature: torch.Tensor     # [D], mean-pooled real latent for this block


def save_frame_as_image(frame: torch.Tensor, output_path: Path) -> Path:
    """frame: [C, H, W] in [0, 1] -> saved PNG, path returned. Used to
    materialize real reference frames on disk so they have a Path
    (`retrieve_nearest_real_frames`'s and `stitch_to_mp4`'s contracts both
    operate on real files, not in-memory tensors, matching how a human
    would actually inspect a retrieval result).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(frame, output_path)
    return output_path


def retrieve_nearest_real_frames(
    steered_latent_trajectory: torch.Tensor,
    reference_latent_bank: list[ReferenceBankEntry],
) -> list[Path]:
    """steered_latent_trajectory: [T_blocks, num_spatial_patches, D] — the
    target-region latents for ONE clip/arm, grouped by temporal block (same
    layout as visualization/pca_overlay.py project_to_rgb_overlay's
    `latent_blocks`).
    reference_latent_bank: real (clip, block) candidates to retrieve from —
    built by the caller, see module docstring.

    -> one real frame path per temporal block (the nearest neighbor by
    cosine similarity between that block's mean-pooled query feature and
    every bank entry's feature). This is RETRIEVAL, not generation — no
    decoder is involved anywhere in this function.
    """
    if not reference_latent_bank:
        raise ValueError("reference_latent_bank is empty — nothing to retrieve from")

    bank_features = torch.stack([e.feature for e in reference_latent_bank])  # [B, D]

    results: list[Path] = []
    for t in range(steered_latent_trajectory.shape[0]):
        query = steered_latent_trajectory[t].mean(dim=0)  # [D], mean-pool over spatial tokens
        similarities = F.cosine_similarity(query.unsqueeze(0), bank_features, dim=-1)  # [B]
        best = int(similarities.argmax().item())
        results.append(reference_latent_bank[best].frame_path)
    return results


def stitch_to_mp4(frame_paths: list[Path], output_path: Path, label: str = "retrieval", fps: int = 2) -> Path:
    """Stitch retrieved frames into an mp4, tagged with `label` in the
    printed output so it is never mistaken for a generated/decoded video
    (hard constraint) — there is no pixel-level watermark since that would
    need a font-rendering dependency this project doesn't otherwise need;
    the filename/caller and this print are the label.

    Uses torchcodec.encoders.VideoEncoder, not torchvision's video I/O:
    torchvision's write_video/read_video is deprecated and was removed in
    the torchvision version this project uses (see data/video_dataset.py's
    docstring — the same reason decoding already goes through torchcodec
    instead of torchvision).

    fps=2 (documented default, not re-derived from a search): these are a
    handful of retrieved stills, not real playback-rate footage, so a slow
    fps that lets a viewer actually see each retrieved frame is more useful
    than matching the source clip's native rate.
    """
    from torchcodec.encoders import VideoEncoder
    from torchvision.io import read_image

    frames = torch.stack([read_image(str(p)) for p in frame_paths])  # [T, C, H, W] uint8, what VideoEncoder expects
    output_path.parent.mkdir(parents=True, exist_ok=True)
    VideoEncoder(frames, frame_rate=fps).to_file(str(output_path))
    print(f"[{label}] saved {output_path} ({len(frame_paths)} retrieved real frame(s), not generated)")
    return output_path
