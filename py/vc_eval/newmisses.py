"""Reference boxes the candidate masks miss but the baseline masks covered (spec 2026-10-21 §6), with their crops."""

from __future__ import annotations

from pathlib import Path

import cv2

from vc_eval.coverage import Miss, evaluate_coverage
from vc_label.boxes import iou


def new_misses(frames: Path, labels: Path, baseline_masks: Path, candidate_masks: Path, recall_targets) -> list[Miss]:
    base = evaluate_coverage(frames, labels, baseline_masks, recall_targets)
    cand = evaluate_coverage(frames, labels, candidate_masks, recall_targets)
    old = [(m.frame, m.box) for m in base.misses + base.uncertain_misses]
    return [m for m in cand.misses + cand.uncertain_misses
            if not any(f == m.frame and b.cls == m.box.cls and iou(b, m.box) >= 0.9 for f, b in old)]


def write_crops(misses: list[Miss], frames: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(misses):
        image = cv2.imread(str(frames / f"{m.frame:06d}.jpg"))
        b = m.box
        side = max(96.0, 3 * max(b.width, b.height))
        cx, cy = b.center
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        crop = image[y0 : int(cy + side / 2), x0 : int(cx + side / 2)].copy()
        cv2.rectangle(crop, (int(b.x1 - x0), int(b.y1 - y0)), (int(b.x2 - x0), int(b.y2 - y0)), (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{i:03d}_{m.frame:06d}_{b.cls}_{b.score:.2f}.jpg"), crop)
