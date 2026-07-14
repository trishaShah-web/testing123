"""NTU RGB+D dataset loader.

Filename convention verified against the dataset's official GitHub repo
(shahroudy/NTURGB-D README.md, fetched 2026-07-14):

    SsssCcccPpppRrrrAaaa   e.g. S001C002P003R002A013

    S = setup number, C = camera ID, P = performer/subject ID,
    R = replication (1 or 2), A = action class (1-120).

This is exactly the "same action, other performers" pooling AGENT.md
DEVIATIONS #3 needs for the Semantic Anchor: same A, different P.
Action-name text comes from data/ntu_action_labels.py (also fetched from
the same README).

TODO: not specified by project definition — which performer IDs go in
train/test ("Cross-Subject") or which cameras/setups go in train/test
("Cross-View") is the dataset's own published split protocol, not
transcribed here yet; `split` is accepted but not yet applied. Do not invent
a split assignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ntu_action_labels import NTU_ACTION_LABELS
from .video_dataset import VideoDataset

_FILENAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
_VIDEO_EXTENSIONS = (".avi", ".mp4")


@dataclass(frozen=True)
class NTUClipMeta:
    path: Path
    setup: int
    camera: int
    performer: int
    replication: int
    action: int          # 1-120
    action_label: str    # human-readable text, e.g. "drink water"


def parse_ntu_filename(path: Path) -> NTUClipMeta | None:
    """One NTU file path -> NTUClipMeta, or None if the filename doesn't
    match SsssCcccPpppRrrrAaaa. Shared by NTURGBDDataset._build_index and
    scripts/ that need to parse just one or two files directly (e.g. a
    quick two-clip test) without indexing an entire curated folder.
    """
    match = _FILENAME_RE.search(path.stem)
    if match is None:
        return None
    setup, camera, performer, replication, action = (int(g) for g in match.groups())
    return NTUClipMeta(
        path=path,
        setup=setup,
        camera=camera,
        performer=performer,
        replication=replication,
        action=action,
        action_label=NTU_ACTION_LABELS.get(action, f"A{action:03d}"),
    )


class NTURGBDDataset(VideoDataset):
    def __init__(self, root: str | Path, split: str | None = None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.split = split
        self.records: list[NTUClipMeta] = []
        self.unparsed: list[Path] = []
        self._build_index()

    def _build_index(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            record = parse_ntu_filename(path)
            if record is None:
                self.unparsed.append(path)
                continue

            self.records.append(record)
            self.samples.append((path, record.action_label))

        if not self.records:
            raise RuntimeError(
                f"no NTU RGB+D clips matched the SsssCcccPpppRrrrAaaa filename "
                f"pattern under {self.root} — check the path and file extensions "
                f"({_VIDEO_EXTENSIONS}); {len(self.unparsed)} files were seen but unparsed."
            )

    def clips_for_action(
        self,
        action: int,
        exclude_performer: int | list[int] | None = None,
        camera: int | None = None,
    ) -> list[NTUClipMeta]:
        """Same action, optionally excluding one or more performers — the
        exact pooling query the Semantic Anchor needs (AGENT.md DEVIATIONS
        #3: "same action performed by other performers"). `exclude_performer`
        accepts a single performer ID (the common case: excluding the
        target clip's own performer) or a list (e.g. also reserving one or
        more OTHER performers out of the anchor's reference pool so their
        clips remain available as disjoint probe-training examples —
        scripts/build_semantic_anchor.py `--max-references`).

        `camera`, if given, restricts the pool to one camera view. NTU RGB+D
        clips of the same action/performer are shot from multiple camera
        angles (e.g. S004 has C001/C002/C003); pooling across all of them
        would mix "camera viewpoint" variation into the anchor alongside
        performer identity, which the anchor is supposed to isolate.
        Recommended: pass the TARGET clip's own camera here so reference
        clips share its viewpoint.
        """
        if exclude_performer is None:
            excluded = set()
        elif isinstance(exclude_performer, int):
            excluded = {exclude_performer}
        else:
            excluded = set(exclude_performer)
        return [
            r for r in self.records
            if r.action == action
            and r.performer not in excluded
            and (camera is None or r.camera == camera)
        ]
