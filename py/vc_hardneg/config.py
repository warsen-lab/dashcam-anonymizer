"""configs/hardneg.json: crop geometry, judge thresholds and the optional LLM judge (spec 2026-10-21 §3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "hardneg.json"  # the repository's, wherever this runs
_KEYS = {
    "crop_factor", "min_crop_px", "pad_value", "per_track", "frame_teacher_yes", "frame_teacher_iou", "upscale",
    "yes_iou", "overlap_iou", "contain_frac", "tiny_px", "plate", "face", "llm",
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
    ask_on: tuple[str, ...] = ("no", "unsure")  # the model verdicts the LLM is asked about
    parallel: int = 1  # LLM calls at once
    image_side: int = 384  # the crop it sees is enlarged to about this, the box drawn in red
    batch: int = 1  # crops per call: more than 1 sends them as one numbered grid, with batch_prompt
    batch_prompt: str = ""  # {count} and {cls_zh} are filled in; it must ask for a JSON array of {n, verdict}
    tile: int = 256  # a grid cell's side in px


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
    contain_frac: float  # a teacher box this much inside the candidate counts as at it (a mask over a whole bumper)
    tiny_px: float  # a candidate whose long side is below this (frame px) is never judged "no"
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
        contain_frac=_unit("contain_frac", d["contain_frac"]),
        tiny_px=float(d["tiny_px"]),
        plate=_class("plate", d["plate"]),
        face=_class("face", d["face"]),
        llm=LlmConfig(str(llm["name"]), tuple(str(a) for a in llm["command"]), float(llm["timeout_s"]), str(llm["prompt"]),
                      tuple(str(v) for v in llm.get("ask_on", ("no", "unsure"))), int(llm.get("parallel", 1)),
                      int(llm.get("image_side", 384)), int(llm.get("batch", 1)), str(llm.get("batch_prompt", "")),
                      int(llm.get("tile", 256))),
    )
    if not set(cfg.llm.ask_on) <= {"no", "unsure"} or cfg.llm.parallel < 1 or cfg.llm.image_side < 64 or cfg.tiny_px < 0:
        raise ValueError(f"{path}: llm.ask_on within [no, unsure], llm.parallel >= 1, llm.image_side >= 64, tiny_px >= 0")
    if cfg.llm.batch < 1 or cfg.llm.tile < 64 or (cfg.llm.batch > 1 and "{count}" not in cfg.llm.batch_prompt):
        raise ValueError(f"{path}: llm.batch >= 1, llm.tile >= 64, and a batch_prompt holding {{count}} when batch > 1")
    if cfg.crop_factor <= 0 or cfg.min_crop_px < 0 or not 0 <= cfg.pad_value <= 255 or cfg.per_track < 1 or cfg.upscale < 1:
        raise ValueError(f"{path}: crop_factor > 0, min_crop_px >= 0, pad_value in [0, 255], per_track >= 1, upscale >= 1")
    return cfg
