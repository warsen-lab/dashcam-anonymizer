"""configs/gt.json: candidate search, linking, crops, the LLM's two questions and the audit sample
(spec 2026-10-22 §3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vc_gt.tracks import TrackParams
from vc_hardneg.config import LlmConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gt.json"  # the repository's, wherever this runs
_KEYS = {"owlv2_score", "tile_grid", "tile_overlap", "retinaface_score", "yunet_score", "coco_score", "student_score",
         "merge_iou", "plate_min_aspect", "plate_max_width_px", "track", "judge_min_score", "crops_per_object",
         "crop_factor", "min_crop_px", "audit", "llm"}
_TRACK_KEYS = {"min_iou", "max_size_ratio", "max_gap"}
_AUDIT_KEYS = {"objects", "frames", "frames_per_class", "seed"}
_LLM_KEYS = {"name", "command", "timeout_s", "parallel", "batch", "tile", "object_prompt", "frame_prompt"}


@dataclass(frozen=True)
class AuditConfig:
    objects: int  # H2: model verdicts on objects people re-judge
    frames: int  # H2: model verdicts on unmasked frames people re-judge
    frames_per_class: int  # H3: whole frames people box every plate (face) on
    seed: int


@dataclass(frozen=True)
class GtLlm:
    name: str
    command: tuple[str, ...]  # argv; {image}, {cls}, {cls_zh} and {prompt} are filled in
    timeout_s: float
    parallel: int
    batch: int  # objects (frames) per call, on one numbered grid
    tile: int  # a crop's side on the grid, px
    object_prompt: str  # question A; {count}, {cls_zh} filled in
    frame_prompt: str  # question B

    def command_config(self) -> LlmConfig:
        """The settings vc_hardneg's CommandJudge runs the command with."""
        return LlmConfig(self.name, self.command, self.timeout_s, prompt="")


@dataclass(frozen=True)
class GtConfig:
    owlv2_score: float
    tile_grid: int
    tile_overlap: float
    retinaface_score: float
    yunet_score: float
    coco_score: float
    student_score: float
    merge_iou: float  # boxes of one class overlapping this much on a frame are one candidate
    plate_min_aspect: float  # a plate box narrower than this many heights is not linked (lamps, signs)
    plate_max_width_px: float  # nor one wider than this (the back of a bus)
    track: TrackParams
    judge_min_score: float  # only tracks some detector scored this much on some frame are judged (spec §8)
    crops_per_object: int
    crop_factor: float  # a crop's side in target sizes
    min_crop_px: float
    audit: AuditConfig
    llm: GtLlm


def _check(ok: bool, path: Path, what: str) -> None:
    if not ok:
        raise ValueError(f"{path}: {what}")


def _keys(d: object, keys: set[str], path: Path, where: str) -> dict:
    _check(isinstance(d, dict) and set(d) == keys, path, f"{where} keys must be exactly {sorted(keys)}")
    return d


def load_config(path: Path = DEFAULT_CONFIG) -> GtConfig:
    """Raises ValueError naming the key for unknown or missing keys and values out of range."""
    path = Path(path)
    d = _keys(json.loads(path.read_text(encoding="utf-8")), _KEYS, path, "top-level")
    t = _keys(d["track"], _TRACK_KEYS, path, "track")
    a = _keys(d["audit"], _AUDIT_KEYS, path, "audit")
    m = _keys(d["llm"], _LLM_KEYS, path, "llm")
    for k in ("owlv2_score", "retinaface_score", "yunet_score", "coco_score", "student_score", "merge_iou",
              "judge_min_score"):
        _check(0 <= d[k] <= 1, path, f"{k} must be in [0, 1]")
    _check(int(d["tile_grid"]) >= 1, path, "tile_grid must be >= 1")
    _check(0 <= d["tile_overlap"] < 0.5, path, "tile_overlap must be in [0, 0.5)")
    _check(int(d["crops_per_object"]) >= 1, path, "crops_per_object must be >= 1")
    _check(d["crop_factor"] >= 1 and d["min_crop_px"] >= 16, path, "crop_factor must be >= 1 and min_crop_px >= 16")
    _check(0 <= t["min_iou"] <= 1, path, "track.min_iou must be in [0, 1]")
    _check(t["max_size_ratio"] >= 1, path, "track.max_size_ratio must be >= 1")
    _check(d["plate_min_aspect"] > 0 and d["plate_max_width_px"] > 0, path,
           "plate_min_aspect and plate_max_width_px must be > 0")
    _check(int(t["max_gap"]) >= 0, path, "track.max_gap must be >= 0")
    _check(all(int(a[k]) >= 0 for k in ("objects", "frames", "frames_per_class")), path, "audit sizes must be >= 0")
    _check(int(m["parallel"]) >= 1, path, "llm.parallel must be >= 1")
    _check(int(m["batch"]) >= 1, path, "llm.batch must be >= 1")
    _check(int(m["tile"]) >= 64, path, "llm.tile must be >= 64")
    for k in ("object_prompt", "frame_prompt"):
        _check("{count}" in m[k] and "{cls_zh}" in m[k], path, f"llm.{k} must hold {{count}} and {{cls_zh}}")
    return GtConfig(
        owlv2_score=float(d["owlv2_score"]), tile_grid=int(d["tile_grid"]), tile_overlap=float(d["tile_overlap"]),
        retinaface_score=float(d["retinaface_score"]), yunet_score=float(d["yunet_score"]),
        coco_score=float(d["coco_score"]), student_score=float(d["student_score"]), merge_iou=float(d["merge_iou"]),
        plate_min_aspect=float(d["plate_min_aspect"]), plate_max_width_px=float(d["plate_max_width_px"]),
        track=TrackParams(float(t["min_iou"]), float(t["max_size_ratio"]), int(t["max_gap"])),
        judge_min_score=float(d["judge_min_score"]),
        crops_per_object=int(d["crops_per_object"]), crop_factor=float(d["crop_factor"]),
        min_crop_px=float(d["min_crop_px"]),
        audit=AuditConfig(int(a["objects"]), int(a["frames"]), int(a["frames_per_class"]), int(a["seed"])),
        llm=GtLlm(str(m["name"]), tuple(str(x) for x in m["command"]), float(m["timeout_s"]), int(m["parallel"]),
                  int(m["batch"]), int(m["tile"]), str(m["object_prompt"]), str(m["frame_prompt"])),
    )
