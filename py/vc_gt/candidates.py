"""Every plausible plate and face on every sampled frame, linked into tracks (spec 2026-10-22 §2 step 2): the teachers
at low thresholds, OWLv2 also on 2 x 2 tiles of the whole frame for small far plates, and the student's boxes."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vc_gt.config import GtConfig
from vc_gt.tracks import link_tracks, write_tracks
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box, Detection, read_frames_csv, read_jsonl, write_jsonl

BOXES = "boxes.jsonl"
PROCESSED = "processed.txt"  # "<frame> <persons>" per finished frame, written after its boxes
TRACKS = "tracks.jsonl"


@dataclass(frozen=True)
class FrameFinds:
    boxes: list[Box]  # faces and plates
    persons: int  # COCO person boxes: the face audit samples frames with people (spec §2 step 5)


class Finder(Protocol):
    def find(self, bgr: np.ndarray) -> FrameFinds: ...


def tile_windows(width: int, height: int, grid: int, overlap: float) -> list[Box]:
    """grid x grid windows over the frame, each grown by `overlap` of a tile into its neighbours, inside the frame."""
    tw, th = width / grid, height / grid
    return [Box("vehicle", max(0.0, c * tw - overlap * tw), max(0.0, r * th - overlap * th),
                min(float(width), (c + 1) * tw + overlap * tw), min(float(height), (r + 1) * th + overlap * th))
            for r in range(grid) for c in range(grid)]


def merge_duplicates(boxes: list[Box], merge_iou: float) -> list[Box]:
    """Per class, a box overlapping a higher-scoring kept box by merge_iou or more folds into it: the kept box stays,
    its src lists every detector that found it (sorted, '+'-joined)."""
    kept: list[tuple[Box, set[str]]] = []
    for b in sorted(boxes, key=lambda x: -x.score):
        home = next((srcs for k, srcs in kept if k.cls == b.cls and iou(k, b) >= merge_iou), None)
        if home is None:
            kept.append((b, {b.src}))
        else:
            home.add(b.src)
    return [replace(k, src="+".join(sorted(s for s in srcs if s))) for k, srcs in kept]


class TeacherFinder:
    """RetinaFace and YuNet for faces, COCO for vehicles and people, OWLv2 in every vehicle and on the tiles; cached
    weights only."""

    def __init__(self, cfg: GtConfig, device: str = "auto") -> None:
        from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env, required_model_files

        offline_model_env()
        missing = [p for p in required_model_files() if not p.exists()]
        if missing:
            raise FileNotFoundError("missing model files: " + ", ".join(str(p) for p in missing))
        import torch

        from vc_label.teachers.coco import CocoTeacher
        from vc_label.teachers.owlv2 import Owlv2PlateTeacher
        from vc_label.teachers.retinaface import RetinaFaceTeacher
        from vc_label.teachers.yunet import YuNetTeacher

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        dev = torch.device(device)
        self.cfg = cfg
        self.faces = [RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), dev, score_threshold=cfg.retinaface_score),
                      YuNetTeacher(str(YUNET_MODEL), score_threshold=cfg.yunet_score)]
        self.coco = CocoTeacher(dev, score_threshold=cfg.coco_score)
        self.owl = Owlv2PlateTeacher(dev, score_threshold=cfg.owlv2_score)

    def find(self, bgr: np.ndarray) -> FrameFinds:
        height, width = bgr.shape[:2]
        faces = [b for t in self.faces for b in t.detect(bgr)]
        aux = self.coco.detect(bgr)
        windows = [a for a in aux if a.cls == "vehicle"] + tile_windows(width, height, self.cfg.tile_grid,
                                                                         self.cfg.tile_overlap)
        return FrameFinds(faces + self.owl.detect(bgr, windows), sum(a.cls == "person" for a in aux))


def read_persons(out_dir: Path) -> dict[int, int]:
    """frame -> COCO person boxes, for every finished frame."""
    path = Path(out_dir) / PROCESSED
    if not path.exists():
        return {}
    return {int(f): int(p) for f, p in (line.split() for line in path.read_text().splitlines() if line.strip())}


def _drop_unfinished(out_dir: Path, done: set[int]) -> None:
    """Keeps only the boxes of finished frames; a torn last line (a crash mid-append) goes too."""
    path = out_dir / BOXES
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line)["frame"] in done:
                keep.append(line)
        except (ValueError, KeyError):
            continue
    path.write_text("".join(f"{line}\n" for line in keep), encoding="utf-8")


def linkable(box: Box, cfg: GtConfig) -> bool:
    """Whether a box takes part in linking: every face; a plate only of plate shape (spec §8: the pilot's square lamps
    and whole bus backs)."""
    return box.cls != "plate" or (box.width >= cfg.plate_min_aspect * box.height
                                  and box.width <= cfg.plate_max_width_px)


def relink(frames_dir: Path, out_dir: Path, cfg: GtConfig) -> int:
    """Links the linkable boxes of boxes.jsonl into tracks.jsonl again (no detection); returns the number of tracks."""
    refs = read_frames_csv(frames_dir)
    by_frame: dict[int, list[Box]] = defaultdict(list)
    for d in read_jsonl(Path(out_dir) / BOXES):
        if linkable(d.box, cfg):
            by_frame[d.frame].append(d.box)
    tracks = link_tracks(by_frame, [r.frame for r in refs], cfg.track)
    write_tracks(Path(out_dir) / TRACKS, refs[0].video, tracks)
    return len(tracks)


def build_candidates(frames_dir: Path, out_dir: Path, finder: Finder, student: dict[int, list[Box]], cfg: GtConfig,
                     limit: int | None = None) -> int:
    """Runs `finder` on every frame of frames.csv not finished yet (at most `limit` in this call), merges its boxes
    with the student's for that frame and appends them to boxes.jsonl, then marks the frame in processed.txt. Once
    every frame is finished, links them into tracks.jsonl. Returns the frames finished in this call."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = read_frames_csv(frames_dir)
    done = set(read_persons(out_dir))
    _drop_unfinished(out_dir, done)
    count = 0
    for ref in refs:
        if ref.frame in done:
            continue
        if limit is not None and count >= limit:
            break
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise FileNotFoundError(f"{ref.image} cannot be read")
        finds = finder.find(bgr)
        theirs = [replace(b, src="student") for b in student.get(ref.frame, []) if b.cls in LABEL_CLASSES]
        merged = merge_duplicates([b for b in finds.boxes if b.cls in LABEL_CLASSES] + theirs, cfg.merge_iou)
        write_jsonl(out_dir / BOXES, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in merged], append=True)
        with (out_dir / PROCESSED).open("a", encoding="utf-8") as f:
            f.write(f"{ref.frame} {finds.persons}\n")
        done.add(ref.frame)
        count += 1
    if refs and all(r.frame in done for r in refs):
        relink(frames_dir, out_dir, cfg)
    return count
