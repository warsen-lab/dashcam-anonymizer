"""YOLO dataset from teacher labels: native-scale crops plus one full frame per labelled frame (spec §4.2); optionally
with the labels cleaned (the dropped boxes optionally covered in grey) and extra crops around judged student false
positives (spec 2026-10-21 §5)."""

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
from vc_train.cleaning import hard_negative_window
from vc_train.geometry import check_input_size, letterbox_scale

Window = tuple[int, int, int, int]  # x0, y0, w, h in frame pixels
MANIFEST = "manifest.csv"
DATA_YAML = "data.yaml"
_MANIFEST_FIELDS = ["sample", "split", "video", "frame", "x0", "y0", "w", "h", "scale", "boxes"]
GREY = 114  # the letterbox padding colour: a covered box teaches the student neither "plate" nor "background"


@dataclass(frozen=True)
class SegmentSummary:
    name: str  # the segment's frames dir name
    train_frames: int
    val_frames: int


@dataclass(frozen=True)
class DatasetSummary:
    train_samples: int
    val_samples: int
    train_frames: int
    val_frames: int
    boxes: int
    segments: tuple[SegmentSummary, ...] = ()
    labels_dropped: int = 0  # label boxes the cleaning left out
    hard_negatives: int = 0  # extra samples around judged student false positives


@dataclass(frozen=True)
class _Frame:
    """A labelled frame whose inputs phase 1 has read and checked."""

    segment: int  # index into the segments list
    stem: str
    ref: FrameRef
    width: int
    height: int
    boxes: list[Box]
    split: str
    greyed: tuple[Box, ...] = ()  # dropped label boxes to cover in every sample of the frame


def _round_half_up(x: float) -> int:
    # Python's round() is half to even (2.5 -> 2); C++ std::lround and vc_train.geometry round half up
    return math.floor(x + 0.5)


def _positions(size: int, window: int) -> list[int]:
    return sorted({0, size - window})


def crop_windows(width: int, height: int, scale: float, crop: int = 1280) -> list[Window]:
    """Windows that become crop x crop samples at `scale`: on each axis the two ends {0, size - window}.

    For a frame of width W and a window of width w <= W <= 2w the two windows overlap, and together they contain every
    box of width <= 2w - W whole (the same for heights). Wider boxes can be cut, and for W > 2w the middle of the
    frame is in no window at all; the full frame sample of every labelled frame covers both.
    """
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


def grey_out(image, boxes: list[Box], keep: list[Box]):
    """A copy of image with the boxes (rounded outward) filled with GREY, except where a box in keep lies."""
    out = image.copy()
    height, width = image.shape[:2]

    def pixels(b: Box) -> tuple[slice, slice]:
        x1, y1 = max(0, math.floor(b.x1)), max(0, math.floor(b.y1))
        x2, y2 = min(width, math.ceil(b.x2)), min(height, math.ceil(b.y2))
        return slice(y1, y2), slice(x1, x2)

    for b in boxes:
        out[pixels(b)] = GREY
    for b in keep:
        out[pixels(b)] = image[pixels(b)]
    return out


def _read_frame_image(frame: _Frame):
    image = cv2.imread(str(frame.ref.image))
    if image is None:
        raise ValueError(f"cannot read {frame.ref.image}")
    return grey_out(image, list(frame.greyed), frame.boxes) if frame.greyed else image


def _scaled(box: Box, s: float) -> Box:
    return replace(box, x1=box.x1 * s, y1=box.y1 * s, x2=box.x2 * s, y2=box.y2 * s)


def _write_sample(out_dir: Path, split: str, name: str, image, boxes: list[Box]) -> None:
    height, width = image.shape[:2]
    path = out_dir / "images" / split / f"{name}.jpg"
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):  # cv2 reports a failed write only by returning False
        raise ValueError(f"cannot write {path}")
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


def _plan(
    segments: list[tuple[Path, Path]], val_tail_ms: float, cleaning=None, grey_dropped: bool = False
) -> tuple[list[_Frame], int]:
    """Phase 1: reads and checks every segment, writes nothing.

    A segment that is missing its labels, or whose frames all fall into the validation tail, is an error: skipping
    it quietly would hand the training run less data (or none at all) than the command line asked for.
    """
    if not segments:
        raise ValueError("no segments given; pass at least one segment")
    frames: list[_Frame] = []
    dropped = 0
    owner: dict[str, int] = {}  # video stem -> index of the segment that writes its samples
    for index, (frames_dir, labels_dir) in enumerate(segments):
        refs = read_frames_csv(frames_dir)
        check_labels_match_frames(frames_dir, labels_dir, refs)
        done = read_processed(labels_dir)
        used = [r for r in refs if r.frame in done]
        if not used:
            raise ValueError(
                f"{labels_dir} has no processed frames for {frames_dir} (the labels dir is missing or empty); "
                f"run vc_label teach and vc_label fuse for this segment first"
            )
        val_from = max(r.pts_ms for r in used) - val_tail_ms
        segment_frames: list[_Frame] = []
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
            left_out: list[Box] = []
            if cleaning is not None:
                boxes, left_out = cleaning.clean(stem, ref.frame, boxes)
                dropped += len(left_out)
            split = "val" if ref.pts_ms >= val_from else "train"
            greyed = tuple(left_out) if grey_dropped else ()
            segment_frames.append(_Frame(index, stem, ref, width, height, boxes, split, greyed))
        if all(f.split == "val" for f in segment_frames):
            span = max(r.pts_ms for r in used) - min(r.pts_ms for r in used)
            raise ValueError(
                f"{labels_dir} would have no train frames: its {len(used)} processed frames span {span:.0f} ms, "
                f"which is not longer than val_tail_ms={val_tail_ms:.0f}. Label a longer segment or lower --val-tail-ms"
            )
        frames.extend(segment_frames)
    return frames, dropped


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
    image = _read_frame_image(frame)
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
    if frame.greyed:
        _write_sample(out_dir, frame.split, full, image, frame.boxes)
    else:
        _link_full_frame(frame.ref.image, out_dir / "images" / frame.split / f"{full}.jpg")
        lines = to_yolo_lines(frame.boxes, frame.width, frame.height)
        (out_dir / "labels" / frame.split / f"{full}.txt").write_text("".join(f"{line}\n" for line in lines))
    rows.append(_manifest_row(full, frame, 0, 0, frame.width, frame.height, 1.0, len(frame.boxes)))
    return rows


def _write_hard_negatives(out_dir: Path, frames: list[_Frame], hard_negatives, count: int) -> list[dict]:
    """The first `count` hard negatives that lie on a dataset frame, each as a native-scale crop centred on it with the
    frame's (cleaned) labels inside it; the false positive itself has no label."""
    by_frame = {(f.stem, f.ref.frame): f for f in frames}
    taken = [h for h in hard_negatives if (h.stem, h.frame) in by_frame][:count]
    rows: list[dict] = []
    per_frame: dict[tuple[str, int], int] = {}
    image, image_of = None, None
    for hn in taken:
        frame = by_frame[(hn.stem, hn.frame)]
        if image_of != frame.ref.image:
            image, image_of = _read_frame_image(frame), frame.ref.image
        x0, y0, w, h = hard_negative_window(hn.box, frame.width, frame.height)
        k = per_frame.get((hn.stem, hn.frame), 0)
        per_frame[(hn.stem, hn.frame)] = k + 1
        name = f"{frame.stem}_{frame.ref.frame:06d}_hn{k}"
        part_boxes = crop_boxes(frame.boxes, (x0, y0, w, h))
        _write_sample(out_dir, frame.split, name, image[y0 : y0 + h, x0 : x0 + w], part_boxes)
        rows.append(_manifest_row(name, frame, x0, y0, w, h, 1.0, len(part_boxes)))
    return rows


def _write_dataset(
    out_dir: Path, segments: list[tuple[Path, Path]], frames: list[_Frame], in_width: int, in_height: int, crop: int,
    hard_negatives=(), hard_negative_share: float = 0.10,
) -> DatasetSummary:
    """Phase 2: writes the samples, the manifest and data.yaml."""
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for frame in frames:
        rows.extend(_write_frame(out_dir, frame, in_width, in_height, crop))
    extra = _write_hard_negatives(out_dir, frames, hard_negatives, round(hard_negative_share * len(rows)))
    rows.extend(extra)
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
        hard_negatives=len(extra),
        segments=tuple(
            SegmentSummary(
                frames_dir.name,
                sum(f.split == "train" for f in frames if f.segment == index),
                sum(f.split == "val" for f in frames if f.segment == index),
            )
            for index, (frames_dir, _) in enumerate(segments)
        ),
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
    cleaning=None,
    hard_negatives=(),
    hard_negative_share: float = 0.10,
    grey_dropped: bool = False,
) -> DatasetSummary:
    """segments: (frames_dir, labels_dir) pairs; only frames in processed.txt are used.

    Every input is checked before anything is written: a segment without processed frames (labels not taught or fused
    yet) or without any frame older than the validation tail raises ValueError instead of being skipped. If writing fails, the output dir is removed
    (or emptied, when it existed empty), so the same --out can be used again after the fix.

    `cleaning` (vc_train.cleaning.LabelCleaning) leaves label boxes out of every sample of their frame; with
    `grey_dropped` those boxes are also covered in grey, so the student does not learn them as background (kept boxes
    overlapping them stay visible, and such a frame's full sample is a written copy, not a link). `hard_negatives`
    (vc_train.cleaning.HardNegative, in the order to take them) add up to hard_negative_share times the other samples.
    """
    check_input_size(in_width, in_height)
    if not 0 <= hard_negative_share <= 1:
        raise ValueError(f"hard_negative_share must be in [0, 1], got {hard_negative_share}")
    if grey_dropped and cleaning is None:
        raise ValueError("grey_dropped needs a cleaning: there are no dropped label boxes to cover")
    _check_out_dir(out_dir)
    frames, dropped = _plan(segments, val_tail_ms, cleaning, grey_dropped)
    created = not out_dir.exists()
    try:
        summary = _write_dataset(out_dir, segments, frames, in_width, in_height, crop, hard_negatives, hard_negative_share)
        return replace(summary, labels_dropped=dropped)
    except BaseException:
        _remove_partial_output(out_dir, created)
        raise
