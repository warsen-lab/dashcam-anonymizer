"""OWLv2 (Google, Apache-2.0) plate teacher: open-vocabulary search inside vehicle crops."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vc_label.contracts import Box
from vc_label.paths import OWLV2_REVISION

SOURCE = "teacher_owlv2"
MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = OWLV2_REVISION
QUERIES = ("a photo of a license plate",)
_CROP_MARGIN = 0.1


def crop_window(vehicle: Box, width: int, height: int, margin: float = _CROP_MARGIN) -> tuple[int, int, int, int]:
    """Integer crop (x0, y0, x1, y1) around a vehicle, grown by `margin` per side, inside the image."""
    dx, dy = vehicle.width * margin, vehicle.height * margin
    x0 = max(0, int(vehicle.x1 - dx))
    y0 = max(0, int(vehicle.y1 - dy))
    x1 = min(width, int(round(vehicle.x2 + dx)))
    y1 = min(height, int(round(vehicle.y2 + dy)))
    return x0, y0, x1, y1


class Owlv2PlateTeacher:
    def __init__(self, device: torch.device, score_threshold: float = 0.1, batch_size: int = 8) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = self.model.eval().to(device)

    def detect(self, bgr: np.ndarray, vehicles: list[Box]) -> list[Box]:
        height, width = bgr.shape[:2]
        windows = [crop_window(v, width, height) for v in vehicles]
        windows = [w for w in windows if w[2] - w[0] >= 8 and w[3] - w[1] >= 8]
        plates: list[Box] = []
        for start in range(0, len(windows), self.batch_size):
            plates.extend(self._detect_batch(bgr, windows[start : start + self.batch_size]))
        return plates

    @torch.inference_mode()
    def _detect_batch(self, bgr: np.ndarray, windows: list[tuple[int, int, int, int]]) -> list[Box]:
        crops = [Image.fromarray(bgr[y0:y1, x0:x1, ::-1]) for x0, y0, x1, y1 in windows]
        inputs = self.processor(
            text=[list(QUERIES)] * len(crops), images=crops, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # OWLv2 pads each crop to a square at the bottom/right; boxes are relative to that square.
        sides = [max(c.width, c.height) for c in crops]
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(s, s) for s in sides]
        )
        plates = []
        for (x0, y0, x1, y1), result in zip(windows, results):
            for (bx1, by1, bx2, by2), score in zip(result["boxes"].tolist(), result["scores"].tolist()):
                box = Box("plate", bx1, by1, bx2, by2, score=score, src=SOURCE)
                plates.append(box.clipped(x1 - x0, y1 - y0).shifted(x0, y0))
        return plates
