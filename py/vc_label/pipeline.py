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

from vc_label.boxes import fuse
from vc_label.config import LabelingConfig
from vc_label.contracts import LABEL_CLASSES, Box, Detection, FrameRef, read_frames_csv, to_yolo_lines, write_jsonl
from vc_label.review import image_size, labelme_document, write_review_queue
from vc_label.rules import Flag, flag_frame, temporal_fill

RAW = "raw.jsonl"
AUX = "aux.jsonl"
PROCESSED = "processed.txt"
MERGED = "merged.jsonl"
FLAGS = "flags.csv"
TEACH_CONFIG = "teach_config.json"


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


@dataclass(frozen=True)
class _JsonlLine:
    raw: bytes
    record: dict


def _read_records(path: Path) -> tuple[list[_JsonlLine], bool]:
    """(parsed non-blank lines of a detections JSONL file, whether a torn last line was dropped).

    An undecodable last line is an append cut short by a crash and is dropped. Any other bad line,
    or a record without `video` and `frame`, raises ValueError naming the file and line number.
    """
    numbered = [(n, line) for n, line in enumerate(path.read_bytes().split(b"\n"), start=1) if line.strip()]
    lines: list[_JsonlLine] = []
    for i, (number, raw) in enumerate(numbered):
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            if i == len(numbered) - 1:
                return lines, True
            raise ValueError(f"{path}:{number}: not a JSON line ({e})") from e
        if not isinstance(record, dict) or "video" not in record or "frame" not in record:
            raise ValueError(f"{path}:{number}: not a detection record (needs video and frame)")
        lines.append(_JsonlLine(raw, record))
    return lines, False


def _preview(items: list, limit: int = 8) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    return f"[{shown}{f', ... (+{len(items) - limit} more)' if len(items) > limit else ''}]"


def check_labels_match_frames(frames_dir: Path, labels_dir: Path, refs: list[FrameRef]) -> None:
    """A labels dir is keyed on frame numbers only; refuse one that was taught on other frames."""
    videos = {r.video for r in refs}
    frame_numbers = {r.frame for r in refs}
    foreign_videos: set[str] = set()
    for name in (RAW, AUX):
        path = labels_dir / name
        if path.exists():
            foreign_videos |= {str(line.record["video"]) for line in _read_records(path)[0]} - videos
    stray_frames = sorted(read_processed(labels_dir) - frame_numbers)
    problems = []
    if foreign_videos:
        problems.append(f"{RAW}/{AUX} hold videos {_preview(sorted(foreign_videos))} which frames.csv lacks (it has {_preview(sorted(videos))})")
    if stray_frames:
        problems.append(f"{PROCESSED} lists frames {_preview(stray_frames)} which frames.csv lacks")
    if problems:
        raise ValueError(f"labels dir {labels_dir} does not match {frames_dir}: " + "; ".join(problems))


def _drop_unfinished(labels_dir: Path, done: set[int]) -> None:
    """Drops detections of frames a crashed run never marked processed, and a torn last line
    (an append cut short by a crash). Rewrites via a temp file and os.replace, and only when needed."""
    for name in (RAW, AUX):
        path = labels_dir / name
        if not path.exists():
            continue
        lines, torn = _read_records(path)
        kept = [line.raw for line in lines if int(line.record["frame"]) in done]
        if not torn and len(kept) == len(lines):
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(b"".join(line + b"\n" for line in kept))
        os.replace(tmp, path)


def _read_pin(path: Path) -> dict:
    try:
        pinned = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(pinned, dict):
        raise ValueError(f"{path}: expected a JSON object of teacher thresholds, got {type(pinned).__name__}")
    return pinned


def _pinned_threshold(pinned: dict, path: Path, name: str) -> float:
    value = pinned.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {name} must be a number, got {value!r}")
    return value


def check_bands_against_pin(labels_dir: Path, cfg: LabelingConfig) -> None:
    """Refuses low-score bands that start above the teacher thresholds this labels dir was taught with.

    LabelingConfig checks the same thing against its own thresholds, but fuse may run with a config
    that differs from the one teach pinned: raw boxes scoring between the pinned threshold and the
    band's lower edge would become labels without being flagged for review.
    A labels dir without teach_config.json (taught before pinning) cannot be checked.
    """
    path = labels_dir / TEACH_CONFIG
    if not path.exists():
        return
    pinned = _read_pin(path)
    retinaface, yunet = (_pinned_threshold(pinned, path, name) for name in ("retinaface_score", "yunet_score"))
    owlv2 = _pinned_threshold(pinned, path, "owlv2_score")
    tail = "boxes below the band would become labels without being flagged for review; lower the band or teach a new labels dir"
    if cfg.face_low_score[0] > min(retinaface, yunet):
        raise ValueError(
            f"{path}: face_low_score lower edge {cfg.face_low_score[0]} is above a face teacher threshold this labels dir "
            f"was taught with (retinaface_score {retinaface}, yunet_score {yunet}): {tail}"
        )
    if cfg.plate_low_score[0] > owlv2:
        raise ValueError(
            f"{path}: plate_low_score lower edge {cfg.plate_low_score[0]} is above the plate teacher threshold this labels dir "
            f"was taught with (owlv2_score {owlv2}): {tail}"
        )


def pin_teach_settings(labels_dir: Path, cfg: LabelingConfig) -> None:
    """Records the teacher thresholds on first use; refuses a later run with different ones.

    Raw teacher output depends on these thresholds, and a resumed run only labels the frames still
    missing, so mixing two sets in one labels dir would leave frames that are not comparable.
    """
    path = labels_dir / TEACH_CONFIG
    settings = cfg.teach_settings()
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return
    pinned = _read_pin(path)
    if pinned == settings:
        return
    changes = ", ".join(
        f"{name} {pinned.get(name)} -> {settings[name]}" for name in settings if pinned.get(name) != settings[name]
    )
    raise ValueError(
        f"{path} pins the teacher thresholds this labels dir was taught with ({changes}). "
        f"A labels dir must be taught with one set of teacher thresholds: restore the config, "
        f"or use a new labels dir (fuse-time settings may change freely)"
    )


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
    refs = read_frames_csv(frames_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    check_labels_match_frames(frames_dir, labels_dir, refs)
    pin_teach_settings(labels_dir, cfg)
    done = read_processed(labels_dir)
    _drop_unfinished(labels_dir, done)
    todo = [r for r in refs[::every] if r.frame not in done][:limit]
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
    """Boxes per frame, read like teach reads them: a torn last line is dropped, any other bad line raises."""
    grouped: dict[int, list[Box]] = defaultdict(list)
    if path.exists():
        for line in _read_records(path)[0]:
            d = Detection.from_record(line.record)
            grouped[d.frame].append(d.box)
    return grouped


def run_fuse(frames_dir: Path, labels_dir: Path, cfg: LabelingConfig) -> tuple[FuseSummary, list[tuple[FrameRef, list[Box]]], list[Flag]]:
    """Fuses raw teacher boxes, fills temporal holes, applies rules, writes merged/flags/YOLO.

    Refuses a labels dir of another video, and low-score bands that start above the pinned teacher thresholds."""
    all_refs = read_frames_csv(frames_dir)
    check_labels_match_frames(frames_dir, labels_dir, all_refs)
    check_bands_against_pin(labels_dir, cfg)
    done = read_processed(labels_dir)
    refs = [r for r in all_refs if r.frame in done]
    raw, aux = _by_frame(labels_dir / RAW), _by_frame(labels_dir / AUX)
    fused = [(r, fuse(raw.get(r.frame, []), cfg.fuse_iou)) for r in refs]
    filled_lists, flags = temporal_fill(fused, cfg)
    sequence = [(ref, boxes) for (ref, _), boxes in zip(fused, filled_lists)]
    yolo_dir = labels_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / "classes.txt").write_text("".join(f"{c}\n" for c in LABEL_CLASSES))
    for ref, boxes in sequence:
        width, height = image_size(ref.image)
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


def run_export_review(
    frames_dir: Path, labels_dir: Path, review_dir: Path, cfg: LabelingConfig, force: bool = False
) -> tuple[int, int]:
    """Writes a LabelMe JSON for every labelled frame that has none yet, plus review_queue.csv.

    The review dir is where X-AnyLabeling saves human review, so an existing `<frame>.json` is left
    untouched unless `force`; review_queue.csv is derived and always rewritten.
    Returns (written, kept).
    """
    _, sequence, flags = run_fuse(frames_dir, labels_dir, cfg)
    review_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[f.frame].append(f)
    written = kept = 0
    for ref, boxes in sequence:
        target = review_dir / f"{ref.frame:06d}.json"
        if target.exists() and not force:
            kept += 1
            continue
        width, height = image_size(ref.image)
        doc = labelme_document(ref, boxes, per_frame.get(ref.frame, []), review_dir, width, height)
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
    write_review_queue(review_dir / "review_queue.csv", flags)
    return written, kept
