"""RetinaFace-R50 face teacher: full-frame inference, no tiling seams."""

from __future__ import annotations

import math

import numpy as np
import torch
from torchvision.ops import nms

from vc_label.contracts import Box
from vc_label.teachers.retinaface_net import load_retinaface

SOURCE = "teacher_retinaface"
_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_STEPS = (8, 16, 32)
_VARIANCES = (0.1, 0.2)
_BGR_MEAN = (104.0, 117.0, 123.0)


def prior_boxes(height: int, width: int) -> np.ndarray:
    """Anchors as (cx, cy, w, h) normalised to the image, in network output order."""
    rows = []
    for step, sizes in zip(_STEPS, _MIN_SIZES):
        fh, fw = math.ceil(height / step), math.ceil(width / step)
        cy, cx = np.meshgrid(
            (np.arange(fh) + 0.5) * step / height,
            (np.arange(fw) + 0.5) * step / width,
            indexing="ij",
        )
        for_sizes = [
            np.stack(
                [cx, cy, np.full_like(cx, s / width), np.full_like(cy, s / height)], axis=-1
            )
            for s in sizes
        ]
        # Per cell the anchors of each size are adjacent: (fh, fw, len(sizes), 4).
        rows.append(np.stack(for_sizes, axis=2).reshape(-1, 4))
    return np.concatenate(rows, axis=0).astype(np.float32)


def decode(deltas: np.ndarray, priors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Box deltas -> absolute (x1, y1, x2, y2) pixel boxes."""
    centers = priors[:, :2] + deltas[:, :2] * _VARIANCES[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(deltas[:, 2:] * _VARIANCES[1])
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    return boxes * np.array([width, height, width, height], dtype=np.float32)


class RetinaFaceTeacher:
    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        score_threshold: float = 0.3,
        nms_iou: float = 0.4,
    ) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.net = load_retinaface(weights_path, device)

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        tensor = torch.from_numpy(
            (bgr.astype(np.float32) - np.array(_BGR_MEAN, dtype=np.float32)).transpose(2, 0, 1)
        )
        deltas, scores = self.net(tensor.unsqueeze(0).to(self.device))
        deltas = deltas[0].float().cpu().numpy()
        scores = scores[0].float().cpu().numpy()
        keep = scores >= self.score_threshold
        boxes = decode(deltas[keep], prior_boxes(height, width)[keep], width, height)
        scores = scores[keep]
        if len(scores) == 0:
            return []
        kept = nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.nms_iou).numpy()
        return [
            Box("face", *map(float, boxes[i]), score=float(scores[i]), src=SOURCE).clipped(
                width, height
            )
            for i in kept
        ]
