"""Shared video loading utilities for the three permitted datasets.

This is plumbing (frame extraction, sequence handling) explicitly listed as a
required skill in SKILLS.md #2 — not a research decision, so no TODO markers
are needed here. Dataset-specific annotation parsing lives in each dataset's
own module and IS marked TODO where the exact downloaded directory/annotation
layout is not yet known.

Uses torchcodec for decoding rather than torchvision.io: torchvision's video
I/O (read_video, VideoReader, etc.) is deprecated as of torchvision 0.22 and
slated for removal in 0.24, and torchcodec is what the official V-JEPA2
HuggingFace example uses to load frames before the video processor — matching
that reduces the chance of a frame-format mismatch once vjepa/encoder.py is
wired up to the real model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class VideoDataset(Dataset):
    """Base class handling frame extraction and temporal sampling.

    Subclasses populate `self.samples`: a list of (video_path, label_or_action_text) pairs.
    """

    def __init__(
        self,
        root: str | Path,
        num_frames: int = 16,
        frame_stride: int = 1,
        transform: Optional[Callable] = None,
        deterministic: bool = False,
    ):
        self.root = Path(root) if root is not None else None
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        # deterministic=True: uniform full-span sampling (evenly spaced
        # indices across the WHOLE clip, no random crop). Required for
        # Semantic Anchor reference clips (ARCHITECTURE.md component 4,
        # phase alignment: STATUS.md-documented decision, resample via
        # uniform index subsampling) — a random start offset would land
        # each reference clip's sampled window at an arbitrary, unaligned
        # point in the action, making cross-performer averaging meaningless.
        # Default is False (existing random-crop behavior, unchanged) since
        # it is not a phase-alignment concern for a single non-anchor clip.
        self.deterministic = deterministic
        self.samples: list[tuple[Path, str]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, video_path: Path) -> torch.Tensor:
        decoder = VideoDecoder(str(video_path), dimension_order="NCHW")

        try:
            total = len(decoder)
        except TypeError:
            # TODO: exact metadata attribute name for frame count has varied
            # across torchcodec releases — verify against the installed
            # version once torchcodec is actually available on Kaggle.
            total = decoder.metadata.num_frames

        if self.deterministic:
            # Evenly spaced indices across the full [0, total-1] span, so
            # frame i/num_frames always lands at the same fraction of the
            # clip's real duration regardless of the clip's native length —
            # this is the phase alignment step, not just frame-count
            # matching. frame_stride does not apply here (it describes a
            # local window stride, not a full-span span).
            idx = torch.linspace(0, total - 1, self.num_frames).round().long()
        else:
            needed = self.num_frames * self.frame_stride
            start = 0 if total <= needed else torch.randint(0, total - needed + 1, (1,)).item()
            # clamp handles the short-video case by repeating the last available
            # frame index, equivalent to the old pad-by-repeat behavior.
            idx = torch.arange(start, start + needed, self.frame_stride).clamp(max=total - 1)

        clip = decoder.get_frames_at(indices=idx).data.float() / 255.0  # (T, C, H, W)

        if self.transform is not None:
            clip = self.transform(clip)
        return clip

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        clip = self._load_clip(video_path)
        return clip, label
