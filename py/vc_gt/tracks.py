"""Candidate boxes linked into objects (spec 2026-10-22 §2 step 2, §3): one track per plate or face over time."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from vc_label.boxes import iou
from vc_label.contracts import Box


@dataclass(frozen=True)
class TrackParams:
    min_iou: float = 0.3
    max_center_dist: float = 1.5  # in widths of the wider of the two boxes
    max_gap: int = 3  # sampled frames a track may go without a box and still continue


@dataclass(frozen=True)
class Track:
    id: int
    cls: str
    frames: tuple[int, ...]  # increasing
    boxes: tuple[Box, ...]  # one per frame

    def box_at(self, frame: int) -> Box | None:
        return self.boxes[self.frames.index(frame)] if frame in self.frames else None


def _match(a: Box, b: Box, p: TrackParams) -> float | None:
    """How well b continues a track whose last box is a (higher is better), None when it cannot. Any overlap match
    (> 1) beats any distance match (<= 1)."""
    overlap = iou(a, b)
    if overlap >= p.min_iou:
        return 1.0 + overlap
    (ax, ay), (bx, by) = a.center, b.center
    limit = p.max_center_dist * max(a.width, b.width)
    dist = math.hypot(ax - bx, ay - by)
    return 1.0 - dist / limit if limit > 0 and dist < limit else None


def link_tracks(by_frame: dict[int, list[Box]], order: list[int], p: TrackParams = TrackParams()) -> list[Track]:
    """Links per-frame boxes into tracks of one class each. At every frame of `order` the best (track, box) pairs are
    taken greedily; a box no track takes starts a track; a track that misses more than p.max_gap frames of `order` in
    a row ends. Ids count from 0 by first frame, then left edge."""
    open_tracks: list[dict] = []
    ended: list[dict] = []
    for i, frame in enumerate(order):
        alive = []
        for t in open_tracks:
            (ended if i - t["last"] - 1 > p.max_gap else alive).append(t)
        open_tracks = alive
        boxes = by_frame.get(frame, [])
        pairs = [(score, ti, bi) for ti, t in enumerate(open_tracks) for bi, b in enumerate(boxes)
                 if b.cls == t["cls"] and (score := _match(t["boxes"][-1], b, p)) is not None]
        taken_t: set[int] = set()
        taken_b: set[int] = set()
        for _, ti, bi in sorted(pairs, key=lambda x: -x[0]):
            if ti in taken_t or bi in taken_b:
                continue
            taken_t.add(ti)
            taken_b.add(bi)
            t = open_tracks[ti]
            t["frames"].append(frame)
            t["boxes"].append(boxes[bi])
            t["last"] = i
        open_tracks += [{"cls": b.cls, "frames": [frame], "boxes": [b], "last": i}
                        for bi, b in enumerate(boxes) if bi not in taken_b]
    every = sorted(ended + open_tracks, key=lambda t: (t["frames"][0], t["boxes"][0].x1, t["cls"]))
    return [Track(k, t["cls"], tuple(t["frames"]), tuple(t["boxes"])) for k, t in enumerate(every)]


def write_tracks(path: Path, video: str, tracks: list[Track]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for t in tracks:
            boxes = [[round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(b.score, 4), b.src]
                     for b in t.boxes]
            f.write(json.dumps({"id": t.id, "cls": t.cls, "video": video, "frames": list(t.frames), "boxes": boxes},
                               ensure_ascii=False) + "\n")


def read_tracks(path: Path) -> list[Track]:
    tracks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                boxes = tuple(Box(r["cls"], *map(float, b[:5]), src=str(b[5])) for b in r["boxes"])
                tracks.append(Track(int(r["id"]), r["cls"], tuple(int(x) for x in r["frames"]), boxes))
    return tracks
