"""Temporal coverage: sparse keyframes + BoxTimeline against per-frame detection (proxy-validation spec §6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vc_label.contracts import LABEL_CLASSES
from vc_eval.coverage import Miss
from vc_eval.metrics import COVERED_FRACTION, Rate, covered_fraction, must_mask
from vc_eval.records import detections_by_frame, read_masks


@dataclass
class TemporalReport:
    video: str
    frames: int
    classes: dict[str, Rate]
    misses: list[Miss] = field(default_factory=list)


def evaluate_temporal(dense_detections: Path, sparse_masks: Path, frame_height: int) -> TemporalReport:
    """Every must-mask box of the --detect-every-frame run must be >= 80 % under that frame's sparse masks."""
    dense_videos, dense = detections_by_frame(dense_detections)
    mask_videos, masks = read_masks(sparse_masks)
    if dense_videos and dense_videos != mask_videos:
        raise ValueError(
            f"{dense_detections} ({', '.join(sorted(dense_videos))}) and {sparse_masks} "
            f"({', '.join(sorted(mask_videos))}) are for different videos"
        )
    report = TemporalReport(next(iter(mask_videos), ""), len(masks), {c: Rate() for c in LABEL_CLASSES})
    for frame in sorted(dense):
        if frame not in masks:
            raise ValueError(f"{sparse_masks} has no line for frame {frame}")
        for box in dense[frame]:
            if box.cls not in LABEL_CLASSES or not must_mask(box, frame_height):
                continue
            covered = covered_fraction(box, masks[frame])
            hit = covered >= COVERED_FRACTION
            report.classes[box.cls] = report.classes[box.cls].add(hit)
            if not hit:
                report.misses.append(Miss(frame, box, covered))
    return report
