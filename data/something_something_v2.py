"""Something-Something V2 dataset loader.

TODO: not specified by project definition — exact annotation/split file
format and directory layout must be confirmed once the dataset download
completes.
"""

from __future__ import annotations

from pathlib import Path

from .video_dataset import VideoDataset


class SomethingSomethingV2Dataset(VideoDataset):
    def __init__(self, root: str | Path, split: str | None = None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.split = split
        self._build_index()

    def _build_index(self) -> None:
        # TODO: not specified by project definition — populate self.samples
        # by parsing Something-Something V2's actual annotation files once
        # available.
        raise NotImplementedError(
            "Something-Something V2 index building is TODO pending confirmed "
            "dataset layout."
        )
