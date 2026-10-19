"""Viewing metrics of a redaction (spec 2026-10-19 §5): how much of the picture the masks cover, how often the masked
region jumps between frames, how many boxes a frame carries, and whether a track ever has two boxes on one frame.

Every frame is measured, on a raster of RASTER_PX x RASTER_PX pixel cells: a region covers every cell it touches, so
it is rounded outwards (by less than one cell on each side).

Two areas are measured. The area on screen is what the viewer sees pixelated: a box with a `block` is first rounded
outwards to whole pixels and clipped to the picture, then snapped outwards to multiples of its block counted from the
picture's top left corner and clipped again, exactly as vc_redact's mosaic does (cpp/src/mask/mosaic.cpp); a box without
a `block` (an older vc_redact) counts as it is. The flicker uses the area on screen. The box area is the plain union of
the boxes, without the snapping, which keeps the numbers comparable with runs made before the mosaic grid. One caveat:
masks.jsonl holds coordinates rounded to 0.1 px, vc_redact snapped the unrounded ones, so an edge within 0.05 px of a
grid line can fall on the other side of it.

Percentiles are numpy's default, linear interpolation between the sorted values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

RASTER_PX = 4  # cells of 4 x 4 px: a quarter of the resolution along each axis
FLICKER_FRACTION = 0.01  # a frame pair jumps when the masked regions differ on more than 1 % of the picture


@dataclass(frozen=True, eq=False)
class ViewingReport:
    video: str
    area: np.ndarray  # per frame: share of the picture pixelated on screen (boxes snapped to the mosaic grid)
    box_area: np.ndarray  # per frame: share of the picture under the plain union of the boxes
    boxes: np.ndarray  # per frame: number of mask boxes
    flicker_pairs: int  # consecutive frame pairs whose masked regions differ on more than FLICKER_FRACTION
    pairs: int  # consecutive frame pairs
    multi_box_frames: int | None  # frames on which one track has more than one box; None when no box names a track

    @property
    def frames(self) -> int:
        return len(self.area)


class _Rect(NamedTuple):
    """Half-open pixel rectangle [x1, x2) x [y1, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1


def _covering(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> _Rect:
    """The smallest pixel rectangle covering the box, clipped to the picture (vc_redact's coveringRect)."""
    return _Rect(
        min(width, max(0, math.floor(x1))), min(height, max(0, math.floor(y1))),
        min(width, max(0, math.ceil(x2))), min(height, max(0, math.ceil(y2))),
    )


def _snapped(r: _Rect, block: int, width: int, height: int) -> _Rect:
    """`r` with its edges moved outwards to multiples of `block`, clipped to the picture (vc_redact's mosaic area)."""
    return _Rect(
        r.x1 // block * block, r.y1 // block * block,
        min(width, -(-r.x2 // block) * block), min(height, -(-r.y2 // block) * block),
    )


def _cells(r: _Rect) -> tuple[slice, slice]:
    """Raster rows and columns a pixel rectangle inside the picture touches (empty for an empty rectangle)."""
    if r.empty:
        return slice(0, 0), slice(0, 0)
    return slice(r.y1 // RASTER_PX, -(-r.y2 // RASTER_PX)), slice(r.x1 // RASTER_PX, -(-r.x2 // RASTER_PX))


def _parse(line: str) -> tuple[str, int, list[tuple[float, float, float, float, int | None]], list[int]]:
    """The video, frame number, boxes (x1, y1, x2, y2, block or None) and track ids of one masks.jsonl line.

    Raises KeyError, TypeError, ValueError or OverflowError for a line that is not a mask record.
    """
    rec = json.loads(line)
    name, frame = rec["video"], int(rec["frame"])
    if not isinstance(name, str):
        raise ValueError("video is not a string")
    boxes = []
    for b in rec["boxes"]:
        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("a box coordinate is not finite")
        block = int(b["block"]) if "block" in b else None
        if block is not None and block < 1:
            raise ValueError(f"block {block} is not positive")
        boxes.append((x1, y1, x2, y2, block))
    return name, frame, boxes, [int(b["track"]) for b in rec["boxes"] if "track" in b]


def _rasterise(boxes: list, width: int, height: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """The raster of the plain union of the boxes and the raster of the region on screen (the same array if no box has a block).

    A box's snapped region contains the box, so the region on screen is the plain raster plus the snapped regions.
    """
    plain = np.zeros((rows, cols), dtype=bool)
    snapped = []
    for x1, y1, x2, y2, block in boxes:
        r = _covering(x1, y1, x2, y2, width, height)
        plain[_cells(r)] = True
        if block is not None and not r.empty:
            snapped.append(_snapped(r, block, width, height))
    if not snapped:
        return plain, plain
    screen = plain.copy()
    for area in snapped:
        screen[_cells(area)] = True
    return plain, screen


def evaluate_viewing(masks_path: Path, width: int, height: int, *, require_inside: bool = False) -> ViewingReport:
    """The viewing metrics of a masks.jsonl of a width x height video.

    The file is read line by line: only a frame's two rasters and the per-frame numbers are held in memory.

    Raises ValueError for a bad line (a `video` that is not a string, a `block` that is not a positive integer, a coordinate
    that is not finite...), frames that are not 0, 1, 2, ... one line each in order, lines of several videos, a file without
    frames or a picture without pixels. A box beyond the picture is clipped to it, or with `require_inside` refused: vc_redact
    clips its boxes to the frame, so such a box means that width x height is not the size of the video.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"the picture size must be positive, got {width}x{height}")
    cols, rows = math.ceil(width / RASTER_PX), math.ceil(height / RASTER_PX)
    cells = cols * rows
    areas: list[float] = []
    box_areas: list[float] = []
    counts: list[int] = []
    flicker = 0
    multi = 0
    tracked = False
    video: str | None = None
    previous: np.ndarray | None = None
    with masks_path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                name, frame, boxes, tracks = _parse(line)
            except (KeyError, TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"{masks_path} line {n}: bad mask record ({e})") from None
            if frame != len(areas):
                raise ValueError(
                    f"{masks_path} line {n}: frame {frame} where frame {len(areas)} was expected (one line per frame, in order)"
                )
            if video is None:
                video = name
            elif name != video:
                raise ValueError(f"{masks_path} line {n}: video {name!r}, but the lines before are for {video!r}")
            if require_inside and any(x1 < 0 or y1 < 0 or x2 > width or y2 > height for x1, y1, x2, y2, _ in boxes):
                raise ValueError(
                    f"{masks_path} line {n}: a box extends beyond the {width}x{height} picture; "
                    "give the video's real size (display size for rotated video)"
                )
            plain, screen = _rasterise(boxes, width, height, rows, cols)
            box_areas.append(np.count_nonzero(plain) / cells)
            areas.append(box_areas[-1] if screen is plain else np.count_nonzero(screen) / cells)
            counts.append(len(boxes))
            tracked = tracked or bool(tracks)
            multi += len(tracks) != len(set(tracks))
            if previous is not None and np.count_nonzero(screen ^ previous) > FLICKER_FRACTION * cells:
                flicker += 1
            previous = screen
    if video is None:
        raise ValueError(f"{masks_path} has no frames")
    return ViewingReport(
        video, np.array(areas), np.array(box_areas), np.array(counts), flicker, len(areas) - 1, multi if tracked else None
    )


def pool_viewing(reports: list[ViewingReport]) -> ViewingReport:
    """All frames of all reports together; a frame pair never spans two reports."""
    if not reports:
        raise ValueError("no viewing reports to pool")
    multis = [r.multi_box_frames for r in reports if r.multi_box_frames is not None]
    return ViewingReport(
        ",".join(r.video for r in reports),
        np.concatenate([r.area for r in reports]),
        np.concatenate([r.box_area for r in reports]),
        np.concatenate([r.boxes for r in reports]),
        sum(r.flicker_pairs for r in reports),
        sum(r.pairs for r in reports),
        sum(multis) if multis else None,
    )


def _area_json(area: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(area)),
        "p95": float(np.percentile(area, 95)),
        "p99": float(np.percentile(area, 99)),
        "max": float(np.max(area)),
    }


def viewing_json(r: ViewingReport) -> dict:
    return {
        "frames": r.frames,
        "area": _area_json(r.area),
        "box_area": _area_json(r.box_area),
        "flicker": {"pairs": r.pairs, "over": r.flicker_pairs, "share": None if r.pairs == 0 else r.flicker_pairs / r.pairs},
        "boxes": {"median": float(np.median(r.boxes)), "p95": float(np.percentile(r.boxes, 95))},
        "multi_box_frames": r.multi_box_frames,
    }


def summary_line(r: ViewingReport) -> str:
    j = viewing_json(r)
    a, ba, fl, b = j["area"], j["box_area"], j["flicker"], j["boxes"]
    flicker = "n/a (one frame)" if fl["share"] is None else f"{fl['share']:.2%} ({fl['over']}/{fl['pairs']} pairs)"
    multi = "n/a (no track ids)" if j["multi_box_frames"] is None else str(j["multi_box_frames"])
    return (
        f"viewing {r.video}: area mean {a['mean']:.2%}, p95 {a['p95']:.2%}, p99 {a['p99']:.2%}, max {a['max']:.2%}; "
        f"flicker {flicker}; boxes median {b['median']:g}, p95 {b['p95']:g}; same-track multi-box frames {multi}; "
        f"box-union area mean {ba['mean']:.2%}, p95 {ba['p95']:.2%}, p99 {ba['p99']:.2%}, max {ba['max']:.2%}"
    )
