"""COCO Faster R-CNN v2 (torchvision, BSD): persons and vehicles as auxiliary boxes."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from vc_label.contracts import Box

SOURCE = "teacher_coco"
# COCO category ids used by torchvision detection models.
_PERSON = {1: "person"}
_VEHICLES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}


class CocoTeacher:
    """Returns Box(cls="person" | "vehicle"); src carries the COCO name, e.g. "teacher_coco:bus"."""

    def __init__(self, device: torch.device, score_threshold: float = 0.3) -> None:
        self.device = device
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_score_thresh=score_threshold,
        ).eval().to(device)

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1)
        out = self.model([rgb.float().div(255).to(self.device)])[0]
        boxes = []
        for (x1, y1, x2, y2), label, score in zip(
            out["boxes"].cpu().tolist(), out["labels"].cpu().tolist(), out["scores"].cpu().tolist()
        ):
            if label in _PERSON:
                cls, name = "person", _PERSON[label]
            elif label in _VEHICLES:
                cls, name = "vehicle", _VEHICLES[label]
            else:
                continue
            boxes.append(
                Box(cls, x1, y1, x2, y2, score=score, src=f"{SOURCE}:{name}").clipped(width, height)
            )
        return boxes
