"""C++ (vc_detect) vs Python onnxruntime on the same JPEGs (main spec §9 cross-language check)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from vc_label.boxes import iou
from vc_label.contracts import Box, FrameRef
from vc_eval.records import detections_by_frame
from vc_train.geometry import decode_yolov8, letterbox

CONFIDENT = 0.3
MATCH_IOU = 0.5


@dataclass(frozen=True)
class ConsistencyReport:
    images: int
    boxes: int
    matched: int
    max_diff_px: float
    unmatched_confident: int
    unmatched_low: int

    def passed(self, tol_px: float = 1.0) -> bool:
        return self.unmatched_confident == 0 and self.max_diff_px < tol_px


def python_detections(model_dir: Path, refs: list[FrameRef], score: float = 0.15, iou_threshold: float = 0.5) -> dict[int, list[Box]]:
    import onnxruntime as ort

    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    width_in, height_in = meta["input"]["width"], meta["input"]["height"]
    out: dict[int, list[Box]] = {}
    for ref in refs:
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise ValueError(f"cannot read {ref.image}")
        arr, r = letterbox(bgr, width_in, height_in)
        pred = session.run(None, {name: arr})[0]
        out[ref.frame] = decode_yolov8(pred, r, meta["classes"], bgr.shape[1], bgr.shape[0], score, iou_threshold)
    return out


def compare_detections(cpp: dict[int, list[Box]], py: dict[int, list[Box]]) -> ConsistencyReport:
    frames = sorted(set(cpp) | set(py))
    boxes = matched = confident = low = 0
    worst = 0.0
    for frame in frames:
        remaining = list(py.get(frame, []))
        for box in cpp.get(frame, []):
            boxes += 1
            same = [o for o in remaining if o.cls == box.cls]
            match = max(same, key=lambda o: iou(o, box), default=None)
            if match is None or iou(match, box) < MATCH_IOU:
                confident, low = (confident + 1, low) if box.score >= CONFIDENT else (confident, low + 1)
                continue
            remaining.remove(match)
            matched += 1
            worst = max(worst, *(abs(p - q) for p, q in zip((box.x1, box.y1, box.x2, box.y2), (match.x1, match.y1, match.x2, match.y2))))
        for box in remaining:
            confident, low = (confident + 1, low) if box.score >= CONFIDENT else (confident, low + 1)
    return ConsistencyReport(len(frames), boxes, matched, worst, confident, low)


def run_vc_detect(vc_detect: Path, model_dir: Path, refs: list[FrameRef], work_dir: Path, runner=subprocess.run) -> dict[int, list[Box]]:
    """Runs the C++ detector on `refs` through a scratch frames dir holding only those frames."""
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with (frames_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for ref in refs:
            link = frames_dir / f"{ref.frame:06d}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(ref.image.resolve(), link)
            writer.writerow([ref.video, ref.frame, ref.pts_ms, "base", 0.0])
    out = work_dir / "cpp_detections.jsonl"
    runner(
        [str(vc_detect), "--model", str(model_dir), "--frames", str(frames_dir), "--out", str(out), "--ep", "cpu"],
        check=True, capture_output=True, text=True,
    )
    return detections_by_frame(out)[1]
