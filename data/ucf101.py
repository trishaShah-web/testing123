"""UCF101 dataset loader.

TODO: not specified by project definition — exact annotation/split file
format and directory layout must be confirmed once the dataset download
completes (UCF101 is publicly downloadable and typically ships with
train/test split list files under a ucfTrainTestlist/ directory).
"""

from __future__ import annotations

from pathlib import Path

from .video_dataset import VideoDataset


class UCF101Dataset(VideoDataset):
    def __init__(self, root: str | Path, split: str | None = None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.split = split
        self._build_index()

    def _build_index(self) -> None:
        # TODO: not specified by project definition — populate self.samples
        # by parsing UCF101's actual split list files once the exact
        # downloaded layout is confirmed.
        raise NotImplementedError(
            "UCF101 index building is TODO pending confirmed dataset layout."
        )
