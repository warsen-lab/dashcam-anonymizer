"""teach (expensive, resumable) and fuse/export (cheap, recomputed from raw teacher output)."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from vc_label.boxes import fuse
from vc_label.config import LabelingConfig
from vc_label.contracts import LABEL_CLASSES, Box, Detection, FrameRef, read_frames_csv, read_jsonl, to_yolo_lines, write_jsonl
from vc_label.review import labelme_document, write_review_queue
from vc_label.rules import Flag, flag_frame, temporal_fill

RAW = "raw.jsonl"
AUX = "aux.jsonl"
PROCESSED = "processed.txt"
MERGED = "merged.jsonl"
FLAGS = "flags.csv"


class FrameDetector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Box]: ...


class PlateDetector(Protocol):
    def detect(self, bgr: np.ndarray, vehicles: list[Box]) -> list[Box]: ...


@dataclass
class Teachers:
    faces: list[FrameDetector]
    coco: FrameDetector
    plates: PlateDetector


def label_frame(bgr: np.ndarray, teachers: Teachers, cfg: LabelingConfig) -> tuple[list[Box], list[Box]]:
    """(raw face/plate boxes from every teacher, auxiliary person/vehicle boxes)."""
    raw = [b for teacher in teachers.faces for b in teacher.detect(bgr)]
    aux = teachers.coco.detect(bgr)
    min_width = cfg.plate_search_min_vehicle_frac * bgr.shape[1]
    vehicles = [a for a in aux if a.cls == "vehicle" and a.width >= min_width]
    raw += teachers.plates.detect(bgr, vehicles)
    return raw, aux


def read_processed(labels_dir: Path) -> set[int]:
    path = labels_dir / PROCESSED
    return {int(line) for line in path.read_text().split()} if path.exists() else set()


def _drop_unfinished(labels_dir: Path, done: set[int]) -> None:
    """Drops detections of frames a crashed run never marked processed, and a torn last line
    (an append cut short by a crash). Rewrites via a temp file and os.replace, and only when needed."""
    for name in (RAW, AUX):
        path = labels_dir / name
        if not path.exists():
            continue
        lines = [line for line in path.read_bytes().split(b"\n") if line.strip()]
        kept: list[bytes] = []
        for i, line in enumerate(lines):
            try:
                frame = int(json.loads(line.decode("utf-8"))["frame"])
            except (UnicodeDecodeError, json.JSONDecodeError):
                if i == len(lines) - 1:
                    continue  # torn last line: drop it
                raise
            if frame in done:
                kept.append(line)
        if len(kept) == len(lines):
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(b"".join(line + b"\n" for line in kept))
        os.replace(tmp, path)


def run_teach(
    frames_dir: Path,
    labels_dir: Path,
    teachers: Teachers,
    cfg: LabelingConfig,
    every: int = 1,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Labels every `every`-th frame of frames.csv not yet processed, at most `limit`; resumable."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    done = read_processed(labels_dir)
    _drop_unfinished(labels_dir, done)
    todo = [r for r in read_frames_csv(frames_dir)[::every] if r.frame not in done][:limit]
    started = time.monotonic()
    for i, ref in enumerate(todo, start=1):
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise FileNotFoundError(f"cannot read {ref.image}")
        raw, aux = label_frame(bgr, teachers, cfg)
        write_jsonl(labels_dir / RAW, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in raw], append=True)
        write_jsonl(labels_dir / AUX, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in aux], append=True)
        with (labels_dir / PROCESSED).open("a") as f:
            f.write(f"{ref.frame}\n")
        if i % 10 == 0 or i == len(todo):
            per_frame = (time.monotonic() - started) / i
            log(f"teach: {i}/{len(todo)} frames, {per_frame:.2f} s/frame, ~{per_frame * (len(todo) - i) / 60:.1f} min left")
    return len(todo)


@dataclass(frozen=True)
class FuseSummary:
    frames: int
    boxes: int
    filled: int
    flags: int


def _by_frame(path: Path) -> dict[int, list[Box]]:
    grouped: dict[int, list[Box]] = defaultdict(list)
    for d in read_jsonl(path):
        grouped[d.frame].append(d.box)
    return grouped


def _image_size(ref: FrameRef) -> tuple[int, int]:
    with Image.open(ref.image) as img:
        return img.size


def run_fuse(frames_dir: Path, labels_dir: Path, cfg: LabelingConfig) -> tuple[FuseSummary, list[tuple[FrameRef, list[Box]]], list[Flag]]:
    """Fuses raw teacher boxes, fills temporal holes, applies rules, writes merged/flags/YOLO."""
    done = read_processed(labels_dir)
    refs = [r for r in read_frames_csv(frames_dir) if r.frame in done]
    raw, aux = _by_frame(labels_dir / RAW), _by_frame(labels_dir / AUX)
    fused = [(r, fuse(raw.get(r.frame, []), cfg.fuse_iou)) for r in refs]
    filled_lists, flags = temporal_fill(fused, cfg)
    sequence = [(ref, boxes) for (ref, _), boxes in zip(fused, filled_lists)]
    yolo_dir = labels_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / "classes.txt").write_text("".join(f"{c}\n" for c in LABEL_CLASSES))
    for ref, boxes in sequence:
        width, height = _image_size(ref)
        flags += flag_frame(ref, boxes, aux.get(ref.frame, []), width, height, cfg)
        (yolo_dir / f"{ref.frame:06d}.txt").write_text("".join(f"{line}\n" for line in to_yolo_lines(boxes, width, height)))
    write_jsonl(labels_dir / MERGED, [Detection(r.video, r.frame, r.pts_ms, b) for r, boxes in sequence for b in boxes])
    with (labels_dir / FLAGS).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "frame", "rule", "cls", "x1", "y1", "x2", "y2", "score", "detail"])
        for f in sorted(flags, key=lambda f: (f.frame, f.rule)):
            b = f.box
            writer.writerow([f.video, f.frame, f.rule, b.cls, round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(b.score, 4), f.detail])
    summary = FuseSummary(
        frames=len(sequence),
        boxes=sum(len(b) for _, b in sequence),
        filled=sum(1 for f in flags if f.rule == "temporal_fill"),
        flags=len(flags),
    )
    return summary, sequence, flags


def run_export_review(frames_dir: Path, labels_dir: Path, review_dir: Path, cfg: LabelingConfig) -> int:
    """Writes one LabelMe JSON per labelled frame plus review_queue.csv; returns the frame count."""
    _, sequence, flags = run_fuse(frames_dir, labels_dir, cfg)
    review_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[f.frame].append(f)
    for ref, boxes in sequence:
        width, height = _image_size(ref)
        doc = labelme_document(ref, boxes, per_frame.get(ref.frame, []), review_dir, width, height)
        (review_dir / f"{ref.frame:06d}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    write_review_queue(review_dir / "review_queue.csv", flags)
    return len(sequence)
