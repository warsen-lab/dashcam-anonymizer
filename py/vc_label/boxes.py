"""Box geometry: IoU, NMS and weighted box fusion across teacher sources."""

from __future__ import annotations

from collections import defaultdict

from vc_label.contracts import Box

SOURCE_SEPARATOR = "+"


def intersection_area(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def contains_point(box: Box, x: float, y: float) -> bool:
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def sources(box: Box) -> set[str]:
    """Teacher sources that contributed to a (possibly fused) box."""
    return set(box.src.split(SOURCE_SEPARATOR)) if box.src else set()


def nms(boxes: list[Box], iou_threshold: float) -> list[Box]:
    """Greedy per-class non-maximum suppression, highest score first."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: x.score, reverse=True):
        if all(k.cls != b.cls or iou(k, b) < iou_threshold for k in kept):
            kept.append(b)
    return kept


def _weighted(members: list[Box]) -> Box:
    weights = [m.score for m in members]
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * len(members), float(len(members))
    coords = [
        sum(w * getattr(m, k) for w, m in zip(weights, members)) / total
        for k in ("x1", "y1", "x2", "y2")
    ]
    return Box(
        members[0].cls,
        *coords,
        score=max(m.score for m in members),
        src=SOURCE_SEPARATOR.join(sorted(set().union(*(sources(m) for m in members)))),
    )


def fuse(boxes: list[Box], iou_threshold: float) -> list[Box]:
    """Weighted box fusion per class.

    Boxes are visited by descending score; each joins the cluster whose fused box overlaps it most,
    if that IoU is at least `iou_threshold`, otherwise it starts a new cluster. A fused box has
    score-weighted coordinates, the maximum member score (recall first: no down-weighting of
    single-source boxes) and the union of member sources joined with '+'.
    """
    by_class: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        by_class[b.cls].append(b)
    fused: list[Box] = []
    for cls in sorted(by_class):
        clusters: list[list[Box]] = []
        centers: list[Box] = []
        for b in sorted(by_class[cls], key=lambda x: x.score, reverse=True):
            best = max(range(len(centers)), key=lambda i: iou(centers[i], b), default=None)
            if best is not None and iou(centers[best], b) >= iou_threshold:
                clusters[best].append(b)
                centers[best] = _weighted(clusters[best])
            else:
                clusters.append([b])
                centers.append(_weighted([b]))
        fused.extend(centers)
    return fused
