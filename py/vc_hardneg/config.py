"""configs/hardneg.json: crop geometry, judge thresholds and the optional LLM judge (spec 2026-10-21 §3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "hardneg.json"  # the repository's, wherever this runs
_KEYS = {
    "crop_factor", "min_crop_px", "pad_value", "per_track", "frame_teacher_yes", "frame_teacher_iou", "upscale",
    "yes_iou", "overlap_iou", "plate", "face", "llm",
}


@dataclass(frozen=True)
class ClassThresholds:
    yes_score: float  # a teacher box at the candidate (IoU >= yes_iou) scoring this much: "yes"
    no_score: float  # no overlapping teacher box scoring this much: "no"
    max_side: int  # the upscaled crop's longest side is capped here


@dataclass(frozen=True)
class LlmConfig:
    name: str
    command: tuple[str, ...]  # argv; {image}, {cls}, {cls_zh} and {prompt} are filled in; empty: no LLM judge
    timeout_s: float
    prompt: str


@dataclass(frozen=True)
class HardnegConfig:
    crop_factor: float
    min_crop_px: float
    pad_value: int
    per_track: int
    frame_teacher_yes: float
    frame_teacher_iou: float
    upscale: float
    yes_iou: float
    overlap_iou: float
    plate: ClassThresholds
    face: ClassThresholds
    llm: LlmConfig

    def thresholds(self, cls: str) -> ClassThresholds:
        return {"plate": self.plate, "face": self.face}[cls]


def _unit(name: str, v: float) -> float:
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return float(v)


def _class(name: str, d: dict) -> ClassThresholds:
    t = ClassThresholds(_unit(f"{name}.yes_score", d["yes_score"]), _unit(f"{name}.no_score", d["no_score"]), int(d["max_side"]))
    if t.no_score > t.yes_score:
        raise ValueError(f"{name}.no_score must not exceed {name}.yes_score")
    if t.max_side < 32:
        raise ValueError(f"{name}.max_side must be at least 32")
    return t


def load_config(path: Path = DEFAULT_CONFIG) -> HardnegConfig:
    """Raises ValueError for unknown or missing keys and values out of range."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or set(d) != _KEYS:
        raise ValueError(f"{path}: keys must be exactly {sorted(_KEYS)}")
    llm = d["llm"]
    cfg = HardnegConfig(
        crop_factor=float(d["crop_factor"]),
        min_crop_px=float(d["min_crop_px"]),
        pad_value=int(d["pad_value"]),
        per_track=int(d["per_track"]),
        frame_teacher_yes=_unit("frame_teacher_yes", d["frame_teacher_yes"]),
        frame_teacher_iou=_unit("frame_teacher_iou", d["frame_teacher_iou"]),
        upscale=float(d["upscale"]),
        yes_iou=_unit("yes_iou", d["yes_iou"]),
        overlap_iou=_unit("overlap_iou", d["overlap_iou"]),
        plate=_class("plate", d["plate"]),
        face=_class("face", d["face"]),
        llm=LlmConfig(str(llm["name"]), tuple(str(a) for a in llm["command"]), float(llm["timeout_s"]), str(llm["prompt"])),
    )
    if cfg.crop_factor <= 0 or cfg.min_crop_px < 0 or not 0 <= cfg.pad_value <= 255 or cfg.per_track < 1 or cfg.upscale < 1:
        raise ValueError(f"{path}: crop_factor > 0, min_crop_px >= 0, pad_value in [0, 255], per_track >= 1, upscale >= 1")
    return cfg
