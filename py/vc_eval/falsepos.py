"""Masks no teacher box explains (spec 2026-10-21 §1, §6): on labelled frames, masked detections that overlap no
same-class teacher box (IoU >= ref_iou), teacher boxes on an excluded_refs.csv row not counting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vc_eval.records import EXCLUDED_REFS, read_excluded
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box, read_jsonl


def count_false_positives(detections: Path, labels_dir: Path, ref_iou: float = 0.1, excluded_iou: float = 0.3) -> dict:
    frames = {int(v) for v in (labels_dir / "processed.txt").read_text().split()}
    teacher: dict[int, list[Box]] = defaultdict(list)
    for d in read_jsonl(labels_dir / "merged.jsonl"):
        teacher[d.frame].append(d.box)
    excluded: dict[int, list[Box]] = defaultdict(list)
    if (labels_dir / EXCLUDED_REFS).is_file():
        for e in read_excluded(labels_dir):
            excluded[e.frame].append(e.box)
    out = {c: {"masked": 0, "unreferenced": 0, "by_src": {}} for c in LABEL_CLASSES}
    with Path(detections).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] not in frames or r.get("confirmed") is False or r.get("verified") is False or r["cls"] not in out:
                continue
            box = Box(r["cls"], r["x1"], r["y1"], r["x2"], r["y2"])
            refs = [t for t in teacher[r["frame"]] if t.cls == box.cls
                    and not any(e.cls == box.cls and iou(e, t) >= excluded_iou for e in excluded[r["frame"]])]
            unref = not any(iou(t, box) >= ref_iou for t in refs)
            o = out[box.cls]
            o["masked"] += 1
            o["unreferenced"] += int(unref)
            s = o["by_src"].setdefault(r["src"], [0, 0])
            s[0] += 1
            s[1] += int(unref)
    return out
