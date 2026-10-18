"""Readers for vc_redact outputs: masks.jsonl (one line per frame) and detections.jsonl."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vc_label.contracts import Box, Detection


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
