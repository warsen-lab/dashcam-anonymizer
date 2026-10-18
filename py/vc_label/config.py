"""Labeling thresholds. Defaults come from the 2026-10-17 prototype on sg_drive_600s (1080p60)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class LabelingConfig:
    # Teacher score thresholds (recall first: low, review catches the noise).
    retinaface_score: float = 0.3
    yunet_score: float = 0.5
    coco_score: float = 0.3
    owlv2_score: float = 0.1
    # OWLv2 only searches vehicles at least this wide (fraction of image width): a must-mask
    # plate (>= 48 px at 4K, 1.25 % of the width) sits on a vehicle >= ~4 % of the width.
    plate_search_min_vehicle_frac: float = 0.03
    fuse_iou: float = 0.5
    # Boxes scoring inside [low, high) for their class go to the review queue.
    face_low_score: tuple[float, float] = (0.3, 0.5)
    plate_low_score: tuple[float, float] = (0.1, 0.3)
    # "Should have but none": large persons need a face in their top part, large vehicles a plate.
    min_person_height_frac: float = 0.12
    face_region_frac: float = 0.35
    min_vehicle_width_frac: float = 0.05
    # Temporal hole filling between neighbouring labelled frames.
    temporal_max_gap_ms: float = 500.0
    temporal_match_iou: float = 0.3
    temporal_present_iou: float = 0.1

    def low_score_band(self, cls: str) -> tuple[float, float]:
        return {"face": self.face_low_score, "plate": self.plate_low_score}[cls]

    @staticmethod
    def from_json(path: Path) -> LabelingConfig:
        """Defaults overridden by the keys in a JSON object; unknown keys are an error."""
        overrides = json.loads(path.read_text())
        known = {f.name for f in fields(LabelingConfig)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown labeling config keys in {path}: {sorted(unknown)}")
        tuples = {k: tuple(v) for k, v in overrides.items() if isinstance(v, list)}
        return replace(LabelingConfig(), **{**overrides, **tuples})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"
