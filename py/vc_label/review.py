"""Human review round trip through X-AnyLabeling (LabelMe JSON), kept outside the frames dir."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from vc_label.contracts import LABEL_CLASSES, Box, FrameRef, to_yolo_lines
from vc_label.rules import Flag

SCENE_FLAGS = ("day", "night", "rain", "tunnel", "city", "highway")
LABELME_VERSION = "5.4.1"


def ensure_outside(frames_dir: Path, path: Path) -> None:
    """Refuses any output inside the frames dir: vc_extract --overwrite deletes that whole dir."""
    frames = frames_dir.resolve()
    target = path.resolve()
    if target == frames or frames in target.parents:
        raise ValueError(f"{path} is inside the frames directory {frames_dir}; write labels elsewhere")


def labelme_document(ref: FrameRef, boxes: list[Box], flags: list[Flag], review_dir: Path, width: int, height: int) -> dict:
    notes: dict[Box, list[str]] = defaultdict(list)
    for f in flags:
        notes[f.box].append(f.rule)
    shapes = [
        {
            "label": b.cls,
            "points": [[round(b.x1, 1), round(b.y1, 1)], [round(b.x2, 1), round(b.y2, 1)]],
            "group_id": None,
            "description": " ".join([f"score={b.score:.2f}", f"src={b.src}", *notes.get(b, [])]),
            "shape_type": "rectangle",
            "flags": {},
        }
        for b in boxes
    ]
    return {
        "version": LABELME_VERSION,
        "flags": {name: False for name in SCENE_FLAGS},
        "shapes": shapes,
        "imagePath": os.path.relpath(ref.image.resolve(), review_dir.resolve()),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def write_review_queue(path: Path, flags: list[Flag]) -> None:
    """One row per flagged frame, most flags first; `where` lists rule@x1,y1,x2,y2 per flag."""
    per_frame: dict[tuple[str, int], list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[(f.video, f.frame)].append(f)
    rows = sorted(per_frame.items(), key=lambda item: (-len(item[1]), item[0][1]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "frame", "flags", "rules", "where"])
        for (video, frame), frame_flags in rows:
            where = " ".join(
                f"{f.rule}@{f.box.x1:.0f},{f.box.y1:.0f},{f.box.x2:.0f},{f.box.y2:.0f}" for f in frame_flags
            )
            rules = ";".join(sorted({f.rule for f in frame_flags}))
            writer.writerow([video, frame, len(frame_flags), rules, where])


def read_labelme_boxes(path: Path) -> tuple[list[Box], list[str], int, int]:
    """(boxes, true scene flags, width, height). Non-rectangles, unknown labels and empty boxes are errors."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(doc["imageWidth"]), int(doc["imageHeight"])
    boxes = []
    for i, shape in enumerate(doc.get("shapes", [])):
        label = shape.get("label")
        if label not in LABEL_CLASSES:
            raise ValueError(f"{path}: unknown label {label!r} (expected one of {LABEL_CLASSES})")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"{path}: shape {label!r} is {shape.get('shape_type')!r}, not a rectangle")
        points = shape.get("points") or []
        if len(points) < 2:
            raise ValueError(f"{path}: shape {i} {label!r} has {len(points)} points, need at least 2")
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        box = Box(label, min(xs), min(ys), max(xs), max(ys), src="human").clipped(width, height)
        if box.area <= 0:
            raise ValueError(f"{path}: shape {i} {label!r} is empty after clipping to the image")
        boxes.append(box)
    tags = sorted(name for name, on in doc.get("flags", {}).items() if on)
    return boxes, tags, width, height


def import_review(review_dir: Path, refs: list[FrameRef], gold_dir: Path) -> int:
    """Converts reviewed LabelMe JSONs to gold YOLO labels plus gold_meta.csv; returns frame count."""
    by_name = {f"{r.frame:06d}": r for r in refs}
    yolo_dir = gold_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(review_dir.glob("*.json")):
        ref = by_name.get(path.stem)
        if ref is None:
            raise ValueError(f"{path}: no frame {path.stem} in frames.csv")
        boxes, tags, width, height = read_labelme_boxes(path)
        yolo_text = "".join(f"{line}\n" for line in to_yolo_lines(boxes, width, height))
        (yolo_dir / f"{path.stem}.txt").write_text(yolo_text, encoding="utf-8")
        rows.append([ref.image.name, ref.video, ref.frame, ";".join(tags)])
    with (gold_dir / "gold_meta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "video", "frame", "scene_tags"])
        writer.writerows(rows)
    return len(rows)
