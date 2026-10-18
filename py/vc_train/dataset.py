"""YOLO dataset from teacher labels: native-scale crops plus one full frame per labelled frame (spec §4.2)."""

from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from vc_label.contracts import LABEL_CLASSES, Box, FrameRef, from_yolo_lines, read_frames_csv, to_yolo_lines
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


@dataclass(frozen=True)
class _Frame:
    """A labelled frame whose inputs phase 1 has read and checked."""

    stem: str
    ref: FrameRef
    width: int
    height: int
    boxes: list[Box]
    split: str


def _round_half_up(x: float) -> int:
    # Python's round() is half to even (2.5 -> 2); C++ std::lround and vc_train.geometry round half up
    return math.floor(x + 0.5)


def _positions(size: int, window: int) -> list[int]:
    return sorted({0, size - window})


def crop_windows(width: int, height: int, scale: float, crop: int = 1280) -> list[Window]:
    """Windows that become crop x crop samples at `scale`; together they contain every box of width <= W - w."""
    w = min(width, _round_half_up(crop / scale))
    h = min(height, _round_half_up(crop / scale))
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


def _write_data_yaml(out_dir: Path) -> None:
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(LABEL_CLASSES))
    (out_dir / DATA_YAML).write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}"
    )


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"dataset dir {out_dir} is not empty; choose a new --out")


def _plan(segments: list[tuple[Path, Path]], val_tail_ms: float) -> list[_Frame]:
    """Phase 1: reads and checks every segment, writes nothing."""
    frames: list[_Frame] = []
    owner: dict[str, int] = {}  # video stem -> index of the segment that writes its samples
    for index, (frames_dir, labels_dir) in enumerate(segments):
        refs = read_frames_csv(frames_dir)
        check_labels_match_frames(frames_dir, labels_dir, refs)
        done = read_processed(labels_dir)
        used = [r for r in refs if r.frame in done]
        if not used:
            continue
        val_from = max(r.pts_ms for r in used) - val_tail_ms
        for ref in used:
            stem = Path(ref.video).stem
            if owner.setdefault(stem, index) != index:
                raise ValueError(
                    f"video stem {stem!r} is in two segments; their samples would overwrite each other"
                )
            label_file = labels_dir / "yolo" / f"{ref.frame:06d}.txt"
            if not label_file.exists():
                raise ValueError(f"{label_file} is missing; run vc_label fuse for {labels_dir} first")
            width, height = image_size(ref.image)
            boxes = from_yolo_lines(label_file.read_text().splitlines(), width, height)
            split = "val" if ref.pts_ms >= val_from else "train"
            frames.append(_Frame(stem, ref, width, height, boxes, split))
    return frames


def _manifest_row(
    name: str, frame: _Frame, x0: int, y0: int, w: int, h: int, scale: float, boxes: int
) -> dict:
    return {
        "sample": f"{name}.jpg",
        "split": frame.split,
        "video": frame.ref.video,
        "frame": frame.ref.frame,
        "x0": x0,
        "y0": y0,
        "w": w,
        "h": h,
        "scale": scale,
        "boxes": boxes,
    }


def _write_frame(out_dir: Path, frame: _Frame, in_width: int, in_height: int, crop: int) -> list[dict]:
    image = cv2.imread(str(frame.ref.image))
    if image is None:
        raise ValueError(f"cannot read {frame.ref.image}")
    s = letterbox_scale(frame.width, frame.height, in_width, in_height)
    rows = []
    for i, (x0, y0, w, h) in enumerate(crop_windows(frame.width, frame.height, s, crop)):
        part = image[y0 : y0 + h, x0 : x0 + w]
        part_boxes = crop_boxes(frame.boxes, (x0, y0, w, h))
        if s != 1.0:
            part = cv2.resize(part, (_round_half_up(w * s), _round_half_up(h * s)), interpolation=cv2.INTER_AREA)
            part_boxes = [_scaled(b, s) for b in part_boxes]
        name = f"{frame.stem}_{frame.ref.frame:06d}_c{i}"
        _write_sample(out_dir, frame.split, name, part, part_boxes)
        rows.append(_manifest_row(name, frame, x0, y0, w, h, s, len(part_boxes)))
    full = f"{frame.stem}_{frame.ref.frame:06d}_full"
    _link_full_frame(frame.ref.image, out_dir / "images" / frame.split / f"{full}.jpg")
    lines = to_yolo_lines(frame.boxes, frame.width, frame.height)
    (out_dir / "labels" / frame.split / f"{full}.txt").write_text("".join(f"{line}\n" for line in lines))
    rows.append(_manifest_row(full, frame, 0, 0, frame.width, frame.height, 1.0, len(frame.boxes)))
    return rows


def _write_dataset(out_dir: Path, frames: list[_Frame], in_width: int, in_height: int, crop: int) -> DatasetSummary:
    """Phase 2: writes the samples, the manifest and data.yaml."""
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for frame in frames:
        rows.extend(_write_frame(out_dir, frame, in_width, in_height, crop))
    rows.sort(key=lambda r: (r["split"] != "train", r["sample"]))
    with (out_dir / MANIFEST).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_data_yaml(out_dir)
    return DatasetSummary(
        train_samples=sum(r["split"] == "train" for r in rows),
        val_samples=sum(r["split"] == "val" for r in rows),
        train_frames=sum(f.split == "train" for f in frames),
        val_frames=sum(f.split == "val" for f in frames),
        boxes=sum(r["boxes"] for r in rows),
    )


def _remove_partial_output(out_dir: Path, created: bool) -> None:
    """After a failed write: remove the dir if this call created it, otherwise empty it again."""
    if created:
        shutil.rmtree(out_dir, ignore_errors=True)
        return
    for child in out_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def build_dataset(
    segments: list[tuple[Path, Path]],
    out_dir: Path,
    in_width: int,
    in_height: int,
    val_tail_ms: float = 20000.0,
    crop: int = 1280,
) -> DatasetSummary:
    """segments: (frames_dir, labels_dir) pairs; only frames in processed.txt are used.

    Every input is checked before anything is written. If writing fails, the output dir is removed
    (or emptied, when it existed empty), so the same --out can be used again after the fix.
    """
    _check_out_dir(out_dir)
    frames = _plan(segments, val_tail_ms)
    created = not out_dir.exists()
    try:
        return _write_dataset(out_dir, frames, in_width, in_height, crop)
    except BaseException:
        _remove_partial_output(out_dir, created)
        raise
