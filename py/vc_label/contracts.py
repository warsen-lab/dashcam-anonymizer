"""Shared data contracts (spec §3.2): boxes in original pixel coordinates."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

LABEL_CLASSES = ("face", "plate")


@dataclass(frozen=True)
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    src: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, width: int, height: int) -> Box:
        return replace(
            self,
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
        )

    def shifted(self, dx: float, dy: float) -> Box:
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


@dataclass(frozen=True)
class FrameRef:
    video: str
    frame: int
    pts_ms: float
    image: Path


def read_frames_csv(frames_dir: Path) -> list[FrameRef]:
    """Rows of <frames_dir>/frames.csv (written by vc_extract), in file order."""
    with (frames_dir / "frames.csv").open(newline="") as f:
        return [
            FrameRef(
                video=row["video"],
                frame=int(row["frame"]),
                pts_ms=float(row["pts_ms"]),
                image=frames_dir / f"{int(row['frame']):06d}.jpg",
            )
            for row in csv.DictReader(f)
        ]


@dataclass(frozen=True)
class Detection:
    video: str
    frame: int
    pts_ms: float
    box: Box

    def to_record(self) -> dict:
        b = self.box
        return {
            "video": self.video,
            "frame": self.frame,
            "pts_ms": round(self.pts_ms, 1),
            "cls": b.cls,
            "x1": round(b.x1, 1),
            "y1": round(b.y1, 1),
            "x2": round(b.x2, 1),
            "y2": round(b.y2, 1),
            "score": round(b.score, 4),
            "src": b.src,
        }

    @staticmethod
    def from_record(record: dict) -> Detection:
        box = Box(
            record["cls"],
            float(record["x1"]),
            float(record["y1"]),
            float(record["x2"]),
            float(record["y2"]),
            score=float(record["score"]),
            src=record["src"],
        )
        return Detection(record["video"], int(record["frame"]), float(record["pts_ms"]), box)


def write_jsonl(path: Path, detections: Iterable[Detection], append: bool = False) -> None:
    with path.open("a" if append else "w") as f:
        for d in detections:
            f.write(json.dumps(d.to_record(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Detection]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield Detection.from_record(json.loads(line))


def to_yolo_lines(boxes: Iterable[Box], width: int, height: int) -> list[str]:
    """YOLO txt rows `<class> <cx> <cy> <w> <h>` normalised to [0, 1], label classes only."""
    lines = []
    for b in boxes:
        if b.cls not in LABEL_CLASSES:
            raise ValueError(f"not a label class: {b.cls}")
        cx, cy = b.center
        lines.append(
            f"{LABEL_CLASSES.index(b.cls)} {cx / width:.6f} {cy / height:.6f} "
            f"{b.width / width:.6f} {b.height / height:.6f}"
        )
    return lines


def from_yolo_lines(lines: Iterable[str], width: int, height: int, src: str = "") -> list[Box]:
    boxes = []
    for line in lines:
        if not line.strip():
            continue
        cls_id, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append(
            Box(LABEL_CLASSES[int(cls_id)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, src=src)
        )
    return boxes
