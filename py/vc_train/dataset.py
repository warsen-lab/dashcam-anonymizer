"""YOLO dataset from teacher labels: native-scale crops plus one full frame per labelled frame (spec §4.2)."""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from vc_label.contracts import LABEL_CLASSES, Box, from_yolo_lines, read_frames_csv, to_yolo_lines
from vc_label.pipeline import check_labels_match_frames, read_processed
from vc_label.review import image_size
from vc_train.geometry import letterbox_scale

Window = tuple[int, int, int, int]  # x0, y0, w, h in frame pixels
MANIFEST = "manifest.csv"
DATA_YAML = "data.yaml"
_MANIFEST_FIELDS = ["sample", "split", "video", "frame", "x0", "y0", "w", "h", "scale", "boxes"]


@dataclass(frozen=True)
class DatasetSummary:
    train_samples: int
    val_samples: int
    train_frames: int
    val_frames: int
    boxes: int


def _positions(size: int, window: int) -> list[int]:
    return sorted({0, size - window})


def crop_windows(width: int, height: int, scale: float, crop: int = 1280) -> list[Window]:
    """Windows that become crop x crop samples at `scale`; together they contain every box of width <= W - w."""
    w = min(width, round(crop / scale))
    h = min(height, round(crop / scale))
    return [(x0, y0, w, h) for y0 in _positions(height, h) for x0 in _positions(width, w)]


def crop_boxes(boxes: list[Box], window: Window, min_visible: float = 0.3) -> list[Box]:
    """Boxes in window coordinates; a box with less than `min_visible` of its area inside is dropped."""
    x0, y0, w, h = window
    out = []
    for b in boxes:
        moved = b.shifted(-x0, -y0)
        clipped = moved.clipped(w, h)
        if b.area > 0 and clipped.area / b.area >= min_visible:
            out.append(clipped)
    return out


def _scaled(box: Box, s: float) -> Box:
    return replace(box, x1=box.x1 * s, y1=box.y1 * s, x2=box.x2 * s, y2=box.y2 * s)


def _write_sample(out_dir: Path, split: str, name: str, image, boxes: list[Box]) -> None:
    height, width = image.shape[:2]
    cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = to_yolo_lines(boxes, width, height)
    (out_dir / "labels" / split / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines))


def _link_full_frame(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"dataset dir {out_dir} is not empty; choose a new --out")
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path) -> None:
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(LABEL_CLASSES))
    (out_dir / DATA_YAML).write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}"
    )


def build_dataset(
    segments: list[tuple[Path, Path]],
    out_dir: Path,
    in_width: int,
    in_height: int,
    val_tail_ms: float = 20000.0,
    crop: int = 1280,
) -> DatasetSummary:
    """segments: (frames_dir, labels_dir) pairs; only frames in processed.txt are used."""
    _prepare_out_dir(out_dir)
    counts = {"train_samples": 0, "val_samples": 0, "train_frames": 0, "val_frames": 0, "boxes": 0}
    rows = []
    for frames_dir, labels_dir in segments:
        refs = read_frames_csv(frames_dir)
        check_labels_match_frames(frames_dir, labels_dir, refs)
        done = read_processed(labels_dir)
        used = [r for r in refs if r.frame in done]
        if not used:
            continue
        val_from = max(r.pts_ms for r in used) - val_tail_ms
        for ref in used:
            label_file = labels_dir / "yolo" / f"{ref.frame:06d}.txt"
            if not label_file.exists():
                raise ValueError(f"{label_file} is missing; run vc_label fuse for {labels_dir} first")
            width, height = image_size(ref.image)
            boxes = from_yolo_lines(label_file.read_text().splitlines(), width, height)
            split = "val" if ref.pts_ms >= val_from else "train"
            stem = Path(ref.video).stem
            s = letterbox_scale(width, height, in_width, in_height)
            image = cv2.imread(str(ref.image))
            if image is None:
                raise ValueError(f"cannot read {ref.image}")
            samples = []
            for i, (x0, y0, w, h) in enumerate(crop_windows(width, height, s, crop)):
                part = image[y0 : y0 + h, x0 : x0 + w]
                part_boxes = crop_boxes(boxes, (x0, y0, w, h))
                if s != 1.0:
                    part = cv2.resize(part, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                    part_boxes = [_scaled(b, s) for b in part_boxes]
                name = f"{stem}_{ref.frame:06d}_c{i}"
                _write_sample(out_dir, split, name, part, part_boxes)
                samples.append((name, x0, y0, w, h, s, len(part_boxes)))
            full = f"{stem}_{ref.frame:06d}_full"
            _link_full_frame(ref.image, out_dir / "images" / split / f"{full}.jpg")
            lines = to_yolo_lines(boxes, width, height)
            (out_dir / "labels" / split / f"{full}.txt").write_text("".join(f"{line}\n" for line in lines))
            samples.append((full, 0, 0, width, height, 1.0, len(boxes)))
            for name, x0, y0, w, h, scale, n in samples:
                rows.append([f"{name}.jpg", split, ref.video, ref.frame, x0, y0, w, h, scale, n])
                counts["boxes"] += n
            counts[f"{split}_samples"] += len(samples)
            counts[f"{split}_frames"] += 1
    rows.sort(key=lambda r: (r[1] != "train", r[0]))
    with (out_dir / MANIFEST).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MANIFEST_FIELDS)
        writer.writerows(rows)
    _write_data_yaml(out_dir)
    return DatasetSummary(**counts)
