"""Coverage geometry and statistics (proxy-validation spec §3.3, §6; main spec §8.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vc_label.contracts import Box

FACE_MIN_4K = 24.0  # must-mask face height at 2160 px
PLATE_MIN_4K = 48.0  # must-mask plate width at 2160 px
BIN_EDGES_4K = (24.0, 48.0, 96.0, 192.0)
COVERED_FRACTION = 0.8
GRID_COLS = 6
GRID_ROWS = 4
HEATMAP_MIN_TARGETS = 30
CLASS_PASS = 0.99
CELL_PASS = 0.97


def target_size(box: Box) -> float:
    """Faces are sized by height, plates by width."""
    return box.height if box.cls == "face" else box.width


def must_mask(box: Box, frame_height: int) -> bool:
    limit = FACE_MIN_4K if box.cls == "face" else PLATE_MIN_4K
    return target_size(box) >= limit * frame_height / 2160


def _edges(frame_height: int) -> list[float]:
    return [e * frame_height / 2160 for e in BIN_EDGES_4K]


def size_bin_labels(frame_height: int) -> list[str]:
    e = _edges(frame_height)
    return [f"<{e[0]:g}"] + [f"{e[i]:g}-{e[i + 1]:g}" for i in range(len(e) - 1)] + [f">{e[-1]:g}"]


def size_bin(box: Box, frame_height: int) -> str:
    edges = _edges(frame_height)
    labels = size_bin_labels(frame_height)
    size = target_size(box)
    for i, edge in enumerate(edges):
        if size < edge:
            return labels[i]
    return labels[-1]


def covered_fraction(target: Box, masks: list[Box]) -> float:
    """Exact share of the target's area under the union of the masks (any class)."""
    if target.area <= 0:
        return 0.0
    parts = [
        (max(m.x1, target.x1), max(m.y1, target.y1), min(m.x2, target.x2), min(m.y2, target.y2))
        for m in masks
    ]
    parts = [p for p in parts if p[2] > p[0] and p[3] > p[1]]
    if not parts:
        return 0.0
    xs = sorted({target.x1, target.x2, *(p[0] for p in parts), *(p[2] for p in parts)})
    ys = sorted({target.y1, target.y2, *(p[1] for p in parts), *(p[3] for p in parts)})
    covered = 0.0
    for xa, xb in zip(xs, xs[1:]):
        cx = (xa + xb) / 2
        for ya, yb in zip(ys, ys[1:]):
            cy = (ya + yb) / 2
            if any(p[0] <= cx < p[2] and p[1] <= cy < p[3] for p in parts):
                covered += (xb - xa) * (yb - ya)
    return covered / target.area


def _pixels(box: Box, width: int, height: int) -> tuple[slice, slice]:
    rows = slice(max(0, math.floor(box.y1)), min(height, math.ceil(box.y2)))
    cols = slice(max(0, math.floor(box.x1)), min(width, math.ceil(box.x2)))
    return rows, cols


def stray_mask_fraction(masks: list[Box], targets: list[Box], width: int, height: int) -> float:
    """Share of the frame covered by masks but by no target box (the false-positive area)."""
    if not masks:
        return 0.0
    grid = np.zeros((height, width), dtype=bool)
    for m in masks:
        grid[_pixels(m, width, height)] = True
    for t in targets:
        grid[_pixels(t, width, height)] = False
    return float(grid.sum()) / (width * height)


def heat_cell(box: Box, width: int, height: int) -> tuple[int, int]:
    cx, cy = box.center
    col = min(GRID_COLS - 1, max(0, int(cx / width * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(cy / height * GRID_ROWS)))
    return row, col


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    hits: int = 0
    total: int = 0

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.hits, self.total)

    def add(self, hit: bool) -> Rate:
        return Rate(self.hits + int(hit), self.total + 1)
