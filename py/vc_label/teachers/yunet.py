"""YuNet face teacher (OpenCV Zoo, MIT) — the second opinion next to RetinaFace."""

from __future__ import annotations

import cv2
import numpy as np

from vc_label.contracts import Box

SOURCE = "teacher_yunet"
_TOP_K = 5000


class YuNetTeacher:
    def __init__(self, model_path: str, score_threshold: float = 0.3, nms_iou: float = 0.3) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_iou, _TOP_K
        )

    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        if faces is None:
            return []
        # Each row: x, y, w, h, 5 landmark pairs, score.
        return [
            Box("face", float(x), float(y), float(x + w), float(y + h), score=float(row[-1]),
                src=SOURCE).clipped(width, height)
            for row in faces
            for x, y, w, h in [row[:4]]
        ]
