"""Temporal coverage: sparse keyframes + BoxTimeline against per-frame detection (proxy-validation spec §6.2)."""

from __future__ import annotations

import json
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
    bins: dict[str, Rate] = field(default_factory=dict)  # the same boxes by score_bin


def score_bin(cls: str, score: float) -> str:
    """The score bands of docs/reports/stable-masks-2026-10-19.md (its temporal coverage table)."""
    if cls == "face":
        return "face>=0.8" if score >= 0.8 else "face0.5-0.8" if score >= 0.5 else "face<0.5"
    return "plate>=0.5" if score >= 0.5 else "plate0.2-0.5" if score >= 0.2 else "plate<0.2"


def _dense_frame_count(dense_detections: Path) -> int:
    """Frame count of the dense run, from the stats.json next to its detections.jsonl, which must say the run detected every frame.

    detections.jsonl alone cannot show that: a frame without detections leaves no line, and a sparse run's file looks the same.
    """
    stats_path = dense_detections.parent / "stats.json"
    if not stats_path.is_file():
        raise ValueError(f"{stats_path} does not exist; it is what shows {dense_detections} comes from a --detect-every-frame run")
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {stats_path}: {exc}") from None
    if not isinstance(stats, dict) or stats.get("detect_every_frame") is not True:
        raise ValueError(f"{stats_path} does not say \"detect_every_frame\": true; the dense reference must come from a --detect-every-frame run")
    frames = stats.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError(f"{stats_path} has no valid \"frames\" count ({frames!r})")
    return frames


def _listed(frames: list[int]) -> str:
    return "frames " + ", ".join(str(f) for f in frames[:5]) + (f" and {len(frames) - 5} more" if len(frames) > 5 else "")


def evaluate_temporal(dense_detections: Path, sparse_masks: Path, frame_height: int) -> TemporalReport:
    """Every must-mask box of the --detect-every-frame run must be >= 80 % under that frame's sparse masks.

    The dense run's stats.json (next to its detections.jsonl) must say it detected every frame, and the masks must cover
    exactly frames 0 .. stats.frames-1: a dense reference is only a reference for the frames it really covers.
    Boxes the dense run never masked (written "confirmed": false) are no reference.
    """
    dense_videos, dense = detections_by_frame(dense_detections, confirmed_only=True)
    dense_frames = _dense_frame_count(dense_detections)
    mask_videos, masks = read_masks(sparse_masks)
    if not masks:
        raise ValueError(f"{sparse_masks} has no frames")
    if len(mask_videos) > 1:
        raise ValueError(f"{sparse_masks} is for several videos: {', '.join(sorted(mask_videos))}")
    if dense_videos and dense_videos != mask_videos:
        raise ValueError(
            f"{dense_detections} ({', '.join(sorted(dense_videos))}) and {sparse_masks} "
            f"({', '.join(sorted(mask_videos))}) are for different videos"
        )
    for frame in sorted(dense):
        if frame not in masks:
            raise ValueError(f"{sparse_masks} has no line for frame {frame}")
    missing = [f for f in range(dense_frames) if f not in masks]
    if missing:
        raise ValueError(f"{sparse_masks} has no line for {_listed(missing)} of the dense run's {dense_frames} frames")
    extra = sorted(f for f in masks if f >= dense_frames)
    if extra:
        raise ValueError(f"{sparse_masks} covers frames beyond the {dense_frames} frames of the dense run: {_listed(extra)}")
    video = next(iter(mask_videos))
    report = TemporalReport(video, len(masks), {c: Rate() for c in LABEL_CLASSES})
    for frame in sorted(dense):
        for box in dense[frame]:
            if box.cls not in LABEL_CLASSES or not must_mask(box, frame_height):
                continue
            covered = covered_fraction(box, masks[frame])
            hit = covered >= COVERED_FRACTION
            report.classes[box.cls] = report.classes[box.cls].add(hit)
            key = score_bin(box.cls, box.score)
            report.bins[key] = report.bins.get(key, Rate()).add(hit)
            if not hit:
                report.misses.append(Miss(frame, box, covered))
    return report
