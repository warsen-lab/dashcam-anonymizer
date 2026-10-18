"""Readers for vc_redact outputs (masks.jsonl, detections.jsonl) and for the fused teacher labels' flags.csv."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from vc_label.contracts import Box, Detection
from vc_label.pipeline import FLAGS, PROCESSED

# flags.csv rules that mark a merged box the teachers were unsure about; the other rules (temporal_fill,
# person_without_face, vehicle_without_plate) are not doubts about the box itself
UNCERTAIN_RULES = frozenset({"low_score", "face_single_source"})
FLAG_COLUMNS = ("frame", "rule", "cls", "x1", "y1", "x2", "y2", "score")


def read_masks(path: Path) -> tuple[set[str], dict[int, list[Box]]]:
    """(video names, frame -> mask boxes); each Box.src holds the mask origin."""
    videos: set[str] = set()
    by_frame: dict[int, list[Box]] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                video = rec["video"]
                frame = int(rec["frame"])
                boxes = [
                    Box(b["cls"], float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]), src=b["origin"])
                    for b in rec["boxes"]
                ]
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path} line {n}: bad mask record ({e})") from None
            if frame in by_frame:
                raise ValueError(f"{path} line {n}: frame {frame} appears twice")
            videos.add(video)
            by_frame[frame] = boxes
    return videos, by_frame


def detections_by_frame(path: Path) -> tuple[set[str], dict[int, list[Box]]]:
    """(video names, frame -> boxes) of a detections JSONL; an empty file gives no detections."""
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    videos: set[str] = set()
    by_frame: dict[int, list[Box]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = Detection.from_record(json.loads(line))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path} line {n}: bad detection record ({e})") from None
            videos.add(d.video)
            by_frame[d.frame].append(d.box)
    return videos, dict(by_frame)


def fused_file(labels_dir: Path, name: str) -> Path:
    """labels_dir/name, a product of vc_label fuse, which writes it after processed.txt; a missing or stale copy is refused."""
    path = labels_dir / name
    if not path.exists():
        raise ValueError(f"{path} does not exist; run vc_label fuse on {labels_dir} first")
    processed = labels_dir / PROCESSED
    if not processed.exists():
        raise ValueError(f"{processed} does not exist; run vc_label teach and fuse on {labels_dir} first")
    if path.stat().st_mtime < processed.stat().st_mtime:
        raise ValueError(f"{path} is older than processed.txt; rerun vc_label fuse")
    return path


def read_uncertain(labels_dir: Path) -> dict[int, list[Box]]:
    """frame -> the boxes flags.csv marks `low_score` or `face_single_source`, each box once whatever rules flagged it.

    Raises ValueError for a missing or stale flags.csv (it must not be older than processed.txt), a header without
    the needed columns, or a bad row (the message names the line).
    """
    path = fused_file(labels_dir, FLAGS)
    seen: set[tuple] = set()
    by_frame: dict[int, list[Box]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FLAG_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no {', '.join(missing)} column")
        for row in reader:
            if row["rule"] not in UNCERTAIN_RULES:
                continue
            try:
                frame = int(row["frame"])
                box = Box(row["cls"], float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), score=float(row["score"]))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path} line {reader.line_num}: bad flag row ({e})") from None
            key = (frame, box.cls, box.x1, box.y1, box.x2, box.y2)
            if key not in seen:
                seen.add(key)
                by_frame.setdefault(frame, []).append(box)
    return by_frame
