"""Labeling thresholds. Defaults come from the 2026-10-17 prototype on sg_drive_600s (1080p60)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

# Applied while the teachers run (`teach`); pinned per labels dir in teach_config.json. Every other
# field only shapes `fuse` / `export-review`, which recompute from the raw teacher output.
TEACH_FIELDS = (
    "retinaface_score",
    "yunet_score",
    "coco_score",
    "owlv2_score",
    "plate_search_min_vehicle_frac",
)
SCORE_FIELDS = ("retinaface_score", "yunet_score", "coco_score", "owlv2_score", "fuse_iou", "temporal_match_iou", "temporal_present_iou")
FRACTION_FIELDS = ("plate_search_min_vehicle_frac", "min_person_height_frac", "face_region_frac", "min_vehicle_width_frac")
BAND_FIELDS = ("face_low_score", "plate_low_score")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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

    def __post_init__(self) -> None:
        for name in (*SCORE_FIELDS, *FRACTION_FIELDS, "temporal_max_gap_ms"):
            if not _is_number(getattr(self, name)):
                raise ValueError(f"{name} must be a number, got {getattr(self, name)!r}")
        for name in SCORE_FIELDS:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1], got {getattr(self, name)}")
        for name in FRACTION_FIELDS:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within (0, 1], got {getattr(self, name)}")
        if not self.temporal_max_gap_ms > 0:
            raise ValueError(f"temporal_max_gap_ms must be positive, got {self.temporal_max_gap_ms}")
        for name in BAND_FIELDS:
            band = getattr(self, name)
            if not isinstance(band, (tuple, list)) or len(band) != 2 or not all(_is_number(v) for v in band):
                raise ValueError(f"{name} must be a [low, high] pair of numbers, got {band!r}")
            low, high = band
            if not 0 <= low < high <= 1:
                raise ValueError(f"{name} must satisfy 0 <= low < high <= 1, got {list(band)}")
            object.__setattr__(self, name, (low, high))  # lists from JSON become tuples; keeps the config hashable
        # A box below the band's lower edge but above the teacher threshold would become a label
        # without ever being flagged for review.
        if self.face_low_score[0] > min(self.retinaface_score, self.yunet_score):
            raise ValueError(
                f"face_low_score lower edge {self.face_low_score[0]} is above a face teacher threshold "
                f"(retinaface_score {self.retinaface_score}, yunet_score {self.yunet_score}): "
                f"boxes below the band would be labelled without being flagged"
            )
        if self.plate_low_score[0] > self.owlv2_score:
            raise ValueError(
                f"plate_low_score lower edge {self.plate_low_score[0]} is above owlv2_score {self.owlv2_score}: "
                f"boxes below the band would be labelled without being flagged"
            )

    def low_score_band(self, cls: str) -> tuple[float, float]:
        bands = {"face": self.face_low_score, "plate": self.plate_low_score}
        if cls not in bands:
            raise ValueError(f"no low-score band for class {cls!r} (expected one of {sorted(bands)})")
        return bands[cls]

    def teach_settings(self) -> dict[str, float]:
        """The thresholds baked into teacher output; a labels dir must be taught with one set."""
        return {name: getattr(self, name) for name in TEACH_FIELDS}

    @staticmethod
    def from_json(path: Path) -> LabelingConfig:
        """Defaults overridden by the keys in a JSON object; unknown keys and invalid values are errors."""
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON ({e})") from e
        if not isinstance(overrides, dict):
            raise ValueError(f"{path}: a labeling config must be a JSON object, got {type(overrides).__name__}")
        known = {f.name for f in fields(LabelingConfig)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown labeling config keys in {path}: {sorted(unknown)}")
        try:
            return replace(LabelingConfig(), **overrides)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"
