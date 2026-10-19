"""Viewing metrics of a redaction (spec 2026-10-19 §5): how much of the picture the masks cover, how often the masked
region jumps between frames, how many boxes a frame carries, and whether a track ever has two boxes on one frame.

Every frame is measured, on a raster of RASTER_PX x RASTER_PX pixel cells: a box covers every cell it touches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RASTER_PX = 4  # cells of 4 x 4 px: a quarter of the resolution along each axis
FLICKER_FRACTION = 0.01  # a frame pair jumps when the masked regions differ on more than 1 % of the picture


@dataclass(frozen=True, eq=False)
class ViewingReport:
    video: str
    area: np.ndarray  # per frame: share of the picture under the union of its masks
    boxes: np.ndarray  # per frame: number of mask boxes
    flicker_pairs: int  # consecutive frame pairs whose masked regions differ on more than FLICKER_FRACTION
    pairs: int  # consecutive frame pairs
    multi_box_frames: int | None  # frames on which one track has more than one box; None when no box names a track

    @property
    def frames(self) -> int:
        return len(self.area)


def _cells(x1: float, y1: float, x2: float, y2: float, cols: int, rows: int) -> tuple[slice, slice]:
    """Raster rows and columns a box touches, clipped to the picture (empty for a box outside it)."""
    c0 = min(cols, max(0, math.floor(x1 / RASTER_PX)))
    c1 = max(c0, min(cols, math.ceil(x2 / RASTER_PX)))
    r0 = min(rows, max(0, math.floor(y1 / RASTER_PX)))
    r1 = max(r0, min(rows, math.ceil(y2 / RASTER_PX)))
    return slice(r0, r1), slice(c0, c1)


def evaluate_viewing(masks_path: Path, width: int, height: int) -> ViewingReport:
    """The viewing metrics of a masks.jsonl of a width x height video.

    The file is read line by line: only one frame's raster and the per-frame numbers are held in memory.

    Raises ValueError for a bad line, frames that are not 0, 1, 2, ... one line each in order, lines of several videos,
    a file without frames or a picture without pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"the picture size must be positive, got {width}x{height}")
    cols, rows = math.ceil(width / RASTER_PX), math.ceil(height / RASTER_PX)
    cells = cols * rows
    areas: list[float] = []
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
                rec = json.loads(line)
                name, frame = rec["video"], int(rec["frame"])
                coords = [(float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])) for b in rec["boxes"]]
                tracks = [int(b["track"]) for b in rec["boxes"] if "track" in b]
                if not all(math.isfinite(v) for box in coords for v in box):
                    raise ValueError("a box coordinate is not finite")
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{masks_path} line {n}: bad mask record ({e})") from None
            if frame != len(areas):
                raise ValueError(
                    f"{masks_path} line {n}: frame {frame} where frame {len(areas)} was expected (one line per frame, in order)"
                )
            if video is None:
                video = name
            elif name != video:
                raise ValueError(f"{masks_path} line {n}: video {name!r}, but the lines before are for {video!r}")
            grid = np.zeros((rows, cols), dtype=bool)
            for x1, y1, x2, y2 in coords:
                grid[_cells(x1, y1, x2, y2, cols, rows)] = True
            areas.append(np.count_nonzero(grid) / cells)
            counts.append(len(coords))
            tracked = tracked or bool(tracks)
            multi += len(tracks) != len(set(tracks))
            if previous is not None and np.count_nonzero(grid ^ previous) > FLICKER_FRACTION * cells:
                flicker += 1
            previous = grid
    if video is None:
        raise ValueError(f"{masks_path} has no frames")
    return ViewingReport(video, np.array(areas), np.array(counts), flicker, len(areas) - 1, multi if tracked else None)


def pool_viewing(reports: list[ViewingReport]) -> ViewingReport:
    """All frames of all reports together; a frame pair never spans two reports."""
    if not reports:
        raise ValueError("no viewing reports to pool")
    multis = [r.multi_box_frames for r in reports if r.multi_box_frames is not None]
    return ViewingReport(
        ",".join(r.video for r in reports),
        np.concatenate([r.area for r in reports]),
        np.concatenate([r.boxes for r in reports]),
        sum(r.flicker_pairs for r in reports),
        sum(r.pairs for r in reports),
        sum(multis) if multis else None,
    )


def viewing_json(r: ViewingReport) -> dict:
    return {
        "frames": r.frames,
        "area": {
            "mean": float(np.mean(r.area)),
            "p95": float(np.percentile(r.area, 95)),
            "p99": float(np.percentile(r.area, 99)),
            "max": float(np.max(r.area)),
        },
        "flicker": {"pairs": r.pairs, "over": r.flicker_pairs, "share": None if r.pairs == 0 else r.flicker_pairs / r.pairs},
        "boxes": {"median": float(np.median(r.boxes)), "p95": float(np.percentile(r.boxes, 95))},
        "multi_box_frames": r.multi_box_frames,
    }


def summary_line(r: ViewingReport) -> str:
    j = viewing_json(r)
    a, fl, b = j["area"], j["flicker"], j["boxes"]
    flicker = "n/a (one frame)" if fl["share"] is None else f"{fl['share']:.2%} ({fl['over']}/{fl['pairs']} pairs)"
    multi = "n/a (no track ids)" if j["multi_box_frames"] is None else str(j["multi_box_frames"])
    return (
        f"viewing {r.video}: area mean {a['mean']:.2%}, p95 {a['p95']:.2%}, p99 {a['p99']:.2%}, max {a['max']:.2%}; "
        f"flicker {flicker}; boxes median {b['median']:g}, p95 {b['p95']:g}; same-track multi-box frames {multi}"
    )
