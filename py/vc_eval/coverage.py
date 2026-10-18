"""Final coverage of the redacted video against the held-out teacher labels (proxy-validation spec §6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vc_label.contracts import LABEL_CLASSES, Box, read_frames_csv
from vc_label.pipeline import MERGED, PROCESSED, check_labels_match_frames, read_processed
from vc_label.review import image_size
from vc_label.rules import TEMPORAL_SOURCE
from vc_eval.metrics import (
    CELL_PASS,
    CLASS_PASS,
    COVERED_FRACTION,
    GRID_COLS,
    GRID_ROWS,
    HEATMAP_MIN_TARGETS,
    Rate,
    covered_fraction,
    heat_cell,
    must_mask,
    size_bin,
    size_bin_labels,
    stray_mask_fraction,
)
from vc_eval.records import detections_by_frame, read_masks


@dataclass(frozen=True)
class Miss:
    frame: int
    box: Box
    covered: float


@dataclass
class ClassCoverage:
    rate: Rate
    bins: dict[str, Rate]
    heatmap: list[list[Rate]]
    temporal_fill: Rate
    best_effort: Rate


def _empty_class(frame_height: int) -> ClassCoverage:
    return ClassCoverage(
        rate=Rate(),
        bins={label: Rate() for label in size_bin_labels(frame_height)},
        heatmap=[[Rate() for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)],
        temporal_fill=Rate(),
        best_effort=Rate(),
    )


@dataclass
class CoverageReport:
    video: str
    frames: int
    width: int
    height: int
    classes: dict[str, ClassCoverage]
    misses: list[Miss] = field(default_factory=list)
    stray_max: float = 0.0
    stray_mean: float = 0.0
    stray_p95: float = 0.0

    def class_passed(self, cls: str) -> bool:
        rate = self.classes[cls].rate
        return rate.total > 0 and rate.value >= CLASS_PASS

    def weakest_cell(self) -> tuple[str, int, int, Rate] | None:
        cells = [
            (cls, r, c, rate)
            for cls, cc in self.classes.items()
            for r, row in enumerate(cc.heatmap)
            for c, rate in enumerate(row)
            if rate.total >= HEATMAP_MIN_TARGETS
        ]
        return min(cells, key=lambda x: x[3].value, default=None)

    def heatmap_passed(self) -> bool:
        weakest = self.weakest_cell()
        return weakest is None or weakest[3].value >= CELL_PASS

    def passed(self) -> bool:
        return all(self.class_passed(c) for c in LABEL_CLASSES) and self.heatmap_passed()


def _fused_labels(labels_dir: Path) -> Path:
    """merged.jsonl, which vc_label fuse writes after processed.txt; a missing or stale copy is refused."""
    merged = labels_dir / MERGED
    if not merged.exists():
        raise ValueError(f"{merged} does not exist; run vc_label fuse on {labels_dir} first")
    if merged.stat().st_mtime < (labels_dir / PROCESSED).stat().st_mtime:
        raise ValueError(f"{merged} is older than processed.txt; rerun vc_label fuse")
    return merged


def evaluate_coverage(frames_dir: Path, labels_dir: Path, masks_path: Path) -> CoverageReport:
    refs = read_frames_csv(frames_dir)
    check_labels_match_frames(frames_dir, labels_dir, refs)
    done = read_processed(labels_dir)
    used = [r for r in refs if r.frame in done]
    if not used:
        raise ValueError(f"{labels_dir} has no labelled frames")
    video = used[0].video
    merged = _fused_labels(labels_dir)
    merged_videos, targets = detections_by_frame(merged)
    if merged_videos - {video}:
        raise ValueError(f"{merged} holds labels for {', '.join(sorted(merged_videos - {video}))}, not {video}")
    mask_videos, masks = read_masks(masks_path)
    if mask_videos - {video}:
        raise ValueError(f"{masks_path} is for {', '.join(sorted(mask_videos))}, not {video}")
    width, height = image_size(used[0].image)
    report = CoverageReport(video, len(used), width, height, {c: _empty_class(height) for c in LABEL_CLASSES})
    stray = []
    for ref in used:
        if ref.frame not in masks:
            raise ValueError(f"{masks_path} has no line for frame {ref.frame}")
        frame_targets = [b for b in targets.get(ref.frame, []) if b.cls in LABEL_CLASSES]
        frame_masks = masks[ref.frame]
        stray.append(stray_mask_fraction(frame_masks, frame_targets, width, height))
        for t in frame_targets:
            cc = report.classes[t.cls]
            covered = covered_fraction(t, frame_masks)
            hit = covered >= COVERED_FRACTION
            if not must_mask(t, height):
                cc.best_effort = cc.best_effort.add(hit)
                continue
            cc.rate = cc.rate.add(hit)
            label = size_bin(t, height)
            cc.bins[label] = cc.bins[label].add(hit)
            r, c = heat_cell(t, width, height)
            cc.heatmap[r][c] = cc.heatmap[r][c].add(hit)
            if t.src == TEMPORAL_SOURCE:
                cc.temporal_fill = cc.temporal_fill.add(hit)
            if not hit:
                report.misses.append(Miss(ref.frame, t, covered))
    report.stray_max = float(max(stray))
    report.stray_mean = float(np.mean(stray))
    report.stray_p95 = float(np.percentile(stray, 95))
    return report
