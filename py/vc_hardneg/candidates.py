"""Verifier candidates (spec 2026-10-21 §3.2): unsure boxes vc_redact dumped and masked, a few per track."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from vc_eval.records import EXCLUDED_REFS, read_excluded
from vc_hardneg.config import HardnegConfig
from vc_label.boxes import iou
from vc_label.contracts import Box, read_jsonl

CANDIDATES = "candidates.jsonl"
CROPS_DIR = "crops"
LOW_BANDS = {"plate": (0.1, 0.3), "face": (0.3, 0.5)}  # configs/labeling.json plate_low_score / face_low_score
EXCLUDED_IOU = 0.3  # a teacher box overlapping an excluded_refs.csv row this much is a known teacher false positive


@dataclass(frozen=True)
class Candidate:
    id: str
    video: str
    stem: str
    frame: int
    cls: str
    box: tuple[float, float, float, float]  # frame pixels
    score: float
    src: str
    track: int
    dump_dir: str  # the vc_redact run directory holding verify_crops.bin
    dump_index: int  # the crop's index in it
    frame_teacher: float  # best same-class teacher box at IoU >= frame_teacher_iou on the frame, 0 if none
    image: str  # native-resolution crop, relative to the candidates file's directory
    crop_origin: tuple[int, int]  # the crop's top-left corner in frame pixels

    def to_record(self) -> dict:
        d = asdict(self)
        d["box"], d["crop_origin"] = list(self.box), list(self.crop_origin)
        return d

    @staticmethod
    def from_record(r: dict) -> Candidate:
        return Candidate(**{**r, "box": tuple(r["box"]), "crop_origin": tuple(r["crop_origin"])})


@dataclass(frozen=True)
class MineSummary:
    rows: int  # dump rows read
    masked: int  # ... on labelled frames that a track masked
    candidates: int  # ... kept after the per-track limit


def candidate_id(video: str, frame: int, cls: str, box: tuple[float, float, float, float]) -> str:
    key = f"{video}:{frame}:{cls}:" + ",".join(f"{round(v, 1):.1f}" for v in box)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def crop_window(box: tuple[float, float, float, float], factor: float, min_px: float) -> tuple[float, float, float]:
    """(x, y, side) of the verifier crop's square, as vc_redact's verifierWindow."""
    x1, y1, x2, y2 = box
    side = max(factor * max(x2 - x1, y2 - y1), min_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return cx - side / 2, cy - side / 2, side


def native_crop(image: np.ndarray, window: tuple[float, float, float], pad: int) -> tuple[np.ndarray, tuple[int, int]]:
    """The window's whole pixels at the image's own resolution, pad outside the image; and its top-left corner."""
    x, y, side = window
    x0, y0 = math.floor(x), math.floor(y)
    w, h = math.ceil(x + side) - x0, math.ceil(y + side) - y0
    out = np.full((h, w, 3), pad, np.uint8)
    ih, iw = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(iw, x0 + w), min(ih, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out, (x0, y0)


def box_in_crop(c: Candidate, cfg: HardnegConfig) -> Box:
    ox, oy = c.crop_origin
    x1, y1, x2, y2 = c.box
    return Box(c.cls, x1 - ox, y1 - oy, x2 - ox, y2 - oy, score=c.score, src=c.src)


def read_candidates(path: Path) -> list[Candidate]:
    with Path(path).open(encoding="utf-8") as f:
        return [Candidate.from_record(json.loads(line)) for line in f if line.strip()]


def write_candidates(path: Path, cands: list[Candidate]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")


def _masked_tracks(masks_path: Path, frames: set[int]) -> dict[int, list[tuple[str, Box, int]]]:
    """frame -> (cls, mask box, track) of the masks a keyframe detection made ("origin": "detected")."""
    out: dict[int, list[tuple[str, Box, int]]] = defaultdict(list)
    with masks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["frame"] not in frames:
                continue
            for b in rec["boxes"]:
                if b.get("origin") == "detected" and "track" in b:
                    box = Box(b["cls"], b["x1"], b["y1"], b["x2"], b["y2"])
                    out[rec["frame"]].append((b["cls"], box, int(b["track"])))
    return out


def _track_of(cls: str, box: Box, masks: list[tuple[str, Box, int]]) -> int | None:
    """The track whose detected mask on that frame contains the box's centre, the nearest centre first."""
    cx, cy = box.center
    inside = [(m, t) for c, m, t in masks if c == cls and m.x1 <= cx <= m.x2 and m.y1 <= cy <= m.y2]
    if not inside:
        return None
    return min(inside, key=lambda mt: (mt[0].center[0] - cx) ** 2 + (mt[0].center[1] - cy) ** 2)[1]


def _spread(items: list, k: int) -> list:
    """k items spread evenly over the list (both ends), all of them when there are at most k."""
    if len(items) <= k:
        return items
    if k == 1:
        return [items[(len(items) - 1) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _frame_teacher(cls: str, box: Box, teacher: list[Box], excluded: list[Box], min_iou: float) -> float:
    real = [t for t in teacher if t.cls == cls and not any(e.cls == cls and iou(e, t) >= EXCLUDED_IOU for e in excluded)]
    return max((t.score for t in real if iou(t, box) >= min_iou), default=0.0)


def write_crops(cands: list[Candidate], frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> list[Candidate]:
    """Each candidate's native-resolution crop to out_dir/crops/<id>.png; the candidates with image and crop_origin
    filled in, ordered by stem, frame and id."""
    (out_dir / CROPS_DIR).mkdir(parents=True, exist_ok=True)
    final: list[Candidate] = []
    images: dict[tuple[str, int], np.ndarray] = {}
    for c in sorted(cands, key=lambda c: (c.stem, c.frame, c.id)):
        key = (c.stem, c.frame)
        if key not in images:
            images.clear()  # candidates come frame by frame: keep one image
            path = frames_root / c.stem / f"{c.frame:06d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"{path} cannot be read")
            images[key] = image
        crop, origin = native_crop(images[key], crop_window(c.box, cfg.crop_factor, cfg.min_crop_px), cfg.pad_value)
        rel = f"{CROPS_DIR}/{c.id}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        final.append(replace(c, image=rel, crop_origin=origin))
    return final


def mine(runs: list[Path], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> MineSummary:
    """Candidates from vc_redact runs made with --dump-verify-crops (each run dir named after its stem)."""
    found: list[Candidate] = []
    rows_read = 0
    for run in runs:
        stem = run.name
        labels = labels_root / stem
        processed = {int(v) for v in (labels / "processed.txt").read_text().split()}
        with (run / "verify_crops.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows_read += len(rows)
        frames = {r["frame"] for r in rows
                  if r["frame"] in processed and (frames_root / stem / f"{r['frame']:06d}.jpg").is_file()}
        masks = _masked_tracks(run / "masks.jsonl", frames)
        teacher: dict[int, list[Box]] = defaultdict(list)
        for d in read_jsonl(labels / "merged.jsonl"):
            teacher[d.frame].append(d.box)
        excluded: dict[int, list[Box]] = defaultdict(list)
        if (labels / EXCLUDED_REFS).is_file():
            for e in read_excluded(labels):
                excluded[e.frame].append(e.box)
        for r in rows:
            if r["frame"] not in frames:
                continue
            box = Box(r["cls"], r["x1"], r["y1"], r["x2"], r["y2"], score=r["score"], src=r["src"])
            track = _track_of(r["cls"], box, masks[r["frame"]])
            if track is None:
                continue
            coords = (box.x1, box.y1, box.x2, box.y2)
            found.append(Candidate(
                id=candidate_id(r["video"], r["frame"], r["cls"], coords), video=r["video"], stem=stem,
                frame=r["frame"], cls=r["cls"], box=coords, score=r["score"], src=r["src"], track=track,
                dump_dir=str(run), dump_index=r["index"],
                frame_teacher=_frame_teacher(r["cls"], box, teacher[r["frame"]], excluded[r["frame"]], cfg.frame_teacher_iou),
                image="", crop_origin=(0, 0),
            ))
    by_track: dict[tuple[str, int, str], list[Candidate]] = defaultdict(list)
    for c in found:
        by_track[(c.stem, c.track, c.cls)].append(c)
    picked = [c for key in sorted(by_track) for c in _spread(sorted(by_track[key], key=lambda c: c.frame), cfg.per_track)]
    final = write_crops(picked, frames_root, out_dir, cfg)
    write_candidates(out_dir / CANDIDATES, final)
    return MineSummary(rows=rows_read, masked=len(found), candidates=len(final))


def mine_labels(stems: list[str], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig,
                bands: dict[str, tuple[float, float]] = LOW_BANDS) -> MineSummary:
    """The teachers' own boxes in a score band, as candidates (spec 2026-10-21 §5.1): judging them cleans the labels
    the student learns from. Only boxes on processed frames whose image is there."""
    found: list[Candidate] = []
    rows_read = 0
    for stem in stems:
        labels = labels_root / stem
        processed = {int(v) for v in (labels / "processed.txt").read_text().split()}
        for d in read_jsonl(labels / "merged.jsonl"):
            rows_read += 1
            b = d.box
            lo, hi = bands.get(b.cls, (1.0, 0.0))
            if d.frame not in processed or not lo <= b.score < hi:
                continue
            if not (frames_root / stem / f"{d.frame:06d}.jpg").is_file():
                continue
            coords = (b.x1, b.y1, b.x2, b.y2)
            found.append(Candidate(
                id=candidate_id(d.video, d.frame, b.cls, coords), video=d.video, stem=stem, frame=d.frame, cls=b.cls,
                box=coords, score=b.score, src=b.src, track=-1, dump_dir="", dump_index=-1, frame_teacher=0.0,
                image="", crop_origin=(0, 0),
            ))
    final = write_crops(found, frames_root, out_dir, cfg)
    write_candidates(out_dir / CANDIDATES, final)
    return MineSummary(rows=rows_read, masked=len(found), candidates=len(final))
