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
    ):
        self.root = Path(root) if root is not None else None
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
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
