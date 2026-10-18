"""Rule-based flags that route likely label errors to human review (spec §5.3, no LLMs)."""

from __future__ import annotations

from dataclasses import dataclass

from vc_label.boxes import contains_point, iou, nms, sources
from vc_label.config import LabelingConfig
from vc_label.contracts import Box, FrameRef

FACE_SOURCES = frozenset({"teacher_retinaface", "teacher_yunet"})
TEMPORAL_SOURCE = "temporal_fill"
AUX_DEDUP_IOU = 0.6


@dataclass(frozen=True)
class Flag:
    video: str
    frame: int
    rule: str
    box: Box
    detail: str = ""


def flag_frame(
    ref: FrameRef, merged: list[Box], aux: list[Box], width: int, height: int, cfg: LabelingConfig
) -> list[Flag]:
    """Per-frame rules: single-source faces, low scores, persons without faces, vehicles without plates."""
    flags: list[Flag] = []
    for b in merged:
        if b.src == TEMPORAL_SOURCE:
            continue
        if b.cls == "face" and not FACE_SOURCES <= sources(b):
            flags.append(Flag(ref.video, ref.frame, "face_single_source", b, f"only {b.src}"))
        low, high = cfg.low_score_band(b.cls)
        if low <= b.score < high:
            flags.append(Flag(ref.video, ref.frame, "low_score", b, f"score {b.score:.2f}"))
    faces = [b for b in merged if b.cls == "face" and b.src != TEMPORAL_SOURCE]
    plates = [b for b in merged if b.cls == "plate" and b.src != TEMPORAL_SOURCE]
    # COCO often reports one vehicle as both car and truck; flag it once.
    deduped = nms(aux, AUX_DEDUP_IOU)
    for person in (a for a in deduped if a.cls == "person"):
        if person.height < cfg.min_person_height_frac * height:
            continue
        region = Box("person", person.x1, person.y1, person.x2, person.y1 + cfg.face_region_frac * person.height)
        if not any(contains_point(region, *f.center) for f in faces):
            flags.append(Flag(ref.video, ref.frame, "person_without_face", person))
    for vehicle in (a for a in deduped if a.cls == "vehicle"):
        if vehicle.width < cfg.min_vehicle_width_frac * width:
            continue
        if not any(contains_point(vehicle, *p.center) for p in plates):
            flags.append(Flag(ref.video, ref.frame, "vehicle_without_plate", vehicle))
    return flags


def _interpolate(a: Box, b: Box, t: float) -> tuple[float, float, float, float]:
    return tuple(getattr(a, k) + (getattr(b, k) - getattr(a, k)) * t for k in ("x1", "y1", "x2", "y2"))


def temporal_fill(
    sequence: list[tuple[FrameRef, list[Box]]], cfg: LabelingConfig
) -> tuple[list[list[Box]], list[Flag]]:
    """Adds a box at frame t when a matching box exists at t-1 and t+1 but nothing similar at t.

    `sequence` is ordered by time. Neighbours are used only when timestamps strictly increase across the
    three frames and each step is at most `temporal_max_gap_ms` apart.
    A match pairs boxes of the same class with IoU >= `temporal_match_iou`; the filled box is the
    linear interpolation at t's timestamp, scored min(prev, next) with src "temporal_fill".
    Returns the per-frame box lists (originals plus fills) and one flag per fill.
    """
    out = [list(boxes) for _, boxes in sequence]
    flags: list[Flag] = []
    for i in range(1, len(sequence) - 1):
        (prev_ref, prev), (ref, _), (next_ref, nxt) = sequence[i - 1], sequence[i], sequence[i + 1]
        gap = cfg.temporal_max_gap_ms
        if not (0 < ref.pts_ms - prev_ref.pts_ms <= gap and 0 < next_ref.pts_ms - ref.pts_ms <= gap):
            continue
        t = (ref.pts_ms - prev_ref.pts_ms) / (next_ref.pts_ms - prev_ref.pts_ms)
        for p in (b for b in prev if b.src != TEMPORAL_SOURCE):
            match = max(
                (n for n in nxt if n.cls == p.cls and n.src != TEMPORAL_SOURCE),
                key=lambda n: iou(p, n),
                default=None,
            )
            if match is None or iou(p, match) < cfg.temporal_match_iou:
                continue
            filled = Box(p.cls, *_interpolate(p, match, t), score=min(p.score, match.score), src=TEMPORAL_SOURCE)
            if any(c.cls == p.cls and iou(c, filled) >= cfg.temporal_present_iou for c in out[i]):
                continue
            out[i].append(filled)
            flags.append(
                Flag(
                    ref.video,
                    ref.frame,
                    "temporal_fill",
                    filled,
                    f"between frames {prev_ref.frame} and {next_ref.frame}",
                )
            )
    return out, flags
