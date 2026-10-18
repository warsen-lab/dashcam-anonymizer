"""Inference geometry shared by training, export, the C++ detector and evaluation (spec §4.1).

The frame is scaled by r = min(W_in / W, H_in / H); the scaled width and height are rounded
half up, like C++ std::lround, so both sides agree on the size. The result is placed at the
top-left corner of a W_in x H_in canvas filled with 114, converted to RGB and divided by 255.
Model outputs are in canvas pixels; dividing by r maps them back to the frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vc_label.boxes import nms
from vc_label.contracts import Box

PAD_VALUE = 114


def letterbox_scale(width: int, height: int, in_width: int, in_height: int) -> float:
    return min(in_width / width, in_height / height)


def letterbox(bgr: np.ndarray, in_width: int, in_height: int) -> tuple[np.ndarray, float]:
    """BGR uint8 image -> ([1, 3, in_height, in_width] float32 RGB in [0, 1], r)."""
    height, width = bgr.shape[:2]
    if min(width, height, in_width, in_height) <= 0:
        raise ValueError(f"letterbox needs positive sizes, got image {width}x{height} and input {in_width}x{in_height}")
    r = letterbox_scale(width, height, in_width, in_height)
    # half up like std::lround (Python's round() is half-to-even and would send 360.5 to 360)
    new_w = min(in_width, math.floor(width * r + 0.5))
    new_h = min(in_height, math.floor(height * r + 0.5))
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"letterbox scaled size {new_w}x{new_h} is empty for image {width}x{height} at scale {r}")
    resized = bgr if (new_w, new_h) == (width, height) else cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((in_height, in_width, 3), PAD_VALUE, np.uint8)
    canvas[:new_h, :new_w] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None].astype(np.float32) / 255.0), r


def decode_yolov8(
    pred: np.ndarray,
    r: float,
    classes: tuple[str, ...] | list[str],
    frame_width: int,
    frame_height: int,
    score_threshold: float,
    iou_threshold: float,
    src: str = "student",
) -> list[Box]:
    """Raw YOLOv8 output -> boxes in frame pixels after thresholding and per-class NMS.

    A broken model must stop the run instead of quietly losing detections, so this raises ValueError when any class
    score of any anchor is not finite (NaN would otherwise fail every threshold test and the anchor would just vanish),
    and when a kept anchor has a non-finite coordinate. Coordinates of anchors below the threshold are never read.
    The C++ decoder does the same.
    """
    p = pred[0] if pred.ndim == 3 else pred
    if p.shape[0] != 4 + len(classes):
        raise ValueError(f"output has {p.shape[0] - 4} classes, model.json lists {len(classes)} classes")
    scores = p[4:]
    if not np.isfinite(scores).all():
        raise ValueError("model output has non-finite scores")
    cls_ids = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    keep = np.nonzero(conf >= score_threshold)[0]
    if not np.isfinite(p[:4, keep]).all():
        raise ValueError("model output has non-finite box coordinates")
    cx, cy, w, h = (p[i, keep] for i in range(4))
    x1 = np.clip((cx - w / 2) / r, 0, frame_width)
    y1 = np.clip((cy - h / 2) / r, 0, frame_height)
    x2 = np.clip((cx + w / 2) / r, 0, frame_width)
    y2 = np.clip((cy + h / 2) / r, 0, frame_height)
    boxes = [
        Box(classes[int(cls_ids[j])], float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), score=float(conf[j]), src=src)
        for i, j in enumerate(keep)
    ]
    return nms(boxes, iou_threshold)
