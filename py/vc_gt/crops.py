"""Crops of tracked objects for the LLM and the review pages (spec 2026-10-22 §2 steps 4-5, §3): which frames show an
object best, and on which frames vc_redact left it unmasked."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_eval.metrics import COVERED_FRACTION, covered_fraction
from vc_gt.config import GtConfig
from vc_gt.tracks import Track
from vc_label.contracts import Box

CROPS_INDEX = "crops.jsonl"
SHARPNESS_CANDIDATES = 5  # sharpness is measured on this many of the largest boxes only (each costs a frame read)


@dataclass(frozen=True)
class Crop:
    path: Path
    box: tuple[float, float, float, float]  # the target in the crop's pixels


def crop_window(box: Box, width: int, height: int, factor: float, min_px: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): a square of factor x the box's long side (at least min_px) around its centre, inside the frame."""
    side = max(min_px, factor * max(box.width, box.height))
    cx, cy = box.center
    return (int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2))),
            int(min(width, round(cx + side / 2))), int(min(height, round(cy + side / 2))))


def write_crop(bgr: np.ndarray, box: Box, path: Path, factor: float, min_px: float) -> Crop:
    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = crop_window(box, width, height, factor, min_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr[y0:y1, x0:x1]):
        raise ValueError(f"cannot write {path}")
    return Crop(path, (box.x1 - x0, box.y1 - y0, box.x2 - x0, box.y2 - y0))


def sharpness(bgr: np.ndarray, box: Box) -> float:
    """Variance of the Laplacian inside the box: higher is sharper."""
    patch = bgr[max(0, int(box.y1)) : math.ceil(box.y2), max(0, int(box.x1)) : math.ceil(box.x2)]
    if patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pick_frames(track: Track, sharp: dict[int, float], k: int) -> list[int]:
    """Up to k of the track's frames, in time order: the largest box, the sharpest (by `sharp`, missing = 0), the
    middle frame, then the next largest until there are k."""
    by_area = sorted(track.frames, key=lambda f: (-track.box_at(f).area, f))
    sharpest = max(track.frames, key=lambda f: (sharp.get(f, 0.0), -f))
    picked: list[int] = []
    for f in [by_area[0], sharpest, track.frames[len(track.frames) // 2], *by_area]:
        if f not in picked:
            picked.append(f)
        if len(picked) == k:
            break
    return sorted(picked)


def unmasked_frames(track: Track, masks_by_frame: dict[int, list[Box]]) -> list[int]:
    """The track's frames where the masks cover less than COVERED_FRACTION of its box (vc_eval's covered test)."""
    return [f for f, b in zip(track.frames, track.boxes)
            if covered_fraction(b, masks_by_frame.get(f, [])) < COVERED_FRACTION]


def _read(images: dict[int, Path], frame: int) -> np.ndarray:
    bgr = cv2.imread(str(images[frame]))
    if bgr is None:
        raise FileNotFoundError(f"{images[frame]} cannot be read")
    return bgr


def write_object_crops(track: Track, images: dict[int, Path], out_dir: Path, cfg: GtConfig) -> list[Crop]:
    """Question A's crops: cfg.crops_per_object picked frames, out_dir/o<id>_<frame>.jpg."""
    largest = sorted(track.frames, key=lambda f: -track.box_at(f).area)[:SHARPNESS_CANDIDATES]
    sharp = {f: sharpness(_read(images, f), track.box_at(f)) for f in largest}
    return [write_crop(_read(images, f), track.box_at(f), Path(out_dir) / f"o{track.id}_{f:06d}.jpg",
                       cfg.crop_factor, cfg.min_crop_px)
            for f in pick_frames(track, sharp, cfg.crops_per_object)]


def write_frame_crop(track: Track, frame: int, images: dict[int, Path], out_dir: Path, cfg: GtConfig) -> Crop:
    """Question B's crop of one unmasked frame: out_dir/o<id>@<frame>.jpg."""
    return write_crop(_read(images, frame), track.box_at(frame), Path(out_dir) / f"o{track.id}@{frame:06d}.jpg",
                      cfg.crop_factor, cfg.min_crop_px)


def append_crop_index(path: Path, item_id: str, crops: list[Crop] | tuple[Crop, ...]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item_id, "crops": [{"path": str(c.path), "box": list(c.box)} for c in crops]}) + "\n")


def read_crop_index(path: Path) -> dict[str, tuple[Crop, ...]]:
    """item id -> its crops; a later line for an id replaces an earlier one."""
    path = Path(path)
    out: dict[str, tuple[Crop, ...]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = tuple(Crop(Path(c["path"]), tuple(float(v) for v in c["box"])) for c in r["crops"])
    return out
