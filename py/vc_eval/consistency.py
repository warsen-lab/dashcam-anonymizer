"""C++ (vc_detect) vs Python onnxruntime on the same JPEGs (main spec §9 cross-language check)."""

from __future__ import annotations

import csv
import dataclasses
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
from vc_train.ort_cpu import cpu_session

CONFIDENT = 0.3
MATCH_IOU = 0.5


@dataclass(frozen=True)
class ConsistencyReport:
    """`max_diff_px` is the largest corner difference over matched pairs whose C++ box scores >= CONFIDENT; it is the gated
    figure. Low-score boxes sit next to the threshold and may legitimately move between runtimes, so they only show up in
    `max_diff_all_px`, the largest difference over every matched pair."""

    images: int
    boxes: int
    matched: int
    max_diff_px: float
    unmatched_confident: int
    unmatched_low: int
    max_diff_all_px: float = 0.0
    matched_confident: int = 0

    def passed(self, tol_px: float = 1.0) -> bool:
        # no images or no confident matched pair is no evidence of agreement (a gate over zero pairs reads 0.0 px),
        # however empty the unmatched counts are
        return (
            self.images > 0 and self.matched > 0 and self.matched_confident > 0
            and self.unmatched_confident == 0 and self.max_diff_px < tol_px
        )


def python_detections(model_dir: Path, refs: list[FrameRef], score: float = 0.15, iou_threshold: float = 0.5) -> dict[int, list[Box]]:
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    session = cpu_session(model_dir / "model.onnx")
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
    boxes = matched = matched_confident = confident = low = 0
    worst = worst_all = 0.0
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
            diff = max(abs(p - q) for p, q in zip((box.x1, box.y1, box.x2, box.y2), (match.x1, match.y1, match.x2, match.y2)))
            worst_all = max(worst_all, diff)
            if box.score >= CONFIDENT:
                matched_confident += 1
                worst = max(worst, diff)
        for box in remaining:
            confident, low = (confident + 1, low) if box.score >= CONFIDENT else (confident, low + 1)
    return ConsistencyReport(len(frames), boxes, matched, worst, confident, low, worst_all, matched_confident)


def same_pixel_subset(refs: list[FrameRef], out_dir: Path) -> list[FrameRef]:
    """Copies of `refs` re-encoded as 4:4:4 JPEG at quality 100, for both runtimes to read.

    FFmpeg (vc_detect) and OpenCV upsample 4:2:0 chroma differently; on the source frames that alone moved boxes by more
    than the 1 px gate. Without chroma subsampling the two decoders agree to within IDCT rounding, so the check measures
    preprocessing and inference rather than the JPEG decoders.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = []
    for ref in refs:
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise ValueError(f"cannot read {ref.image}")
        image = out_dir / f"{ref.frame:06d}.jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        if not cv2.imwrite(str(image), bgr, params):
            raise ValueError(f"cannot write {image}")
        subset.append(dataclasses.replace(ref, image=image))
    return subset


def run_vc_detect(
    vc_detect: Path, model_dir: Path, refs: list[FrameRef], work_dir: Path, runner=subprocess.run, score: float = 0.15
) -> dict[int, list[Box]]:
    """Runs the C++ detector on `refs` through a scratch frames dir holding only those frames.

    `score` is forwarded as --score; a failed run raises ValueError carrying the detector's stderr.
    """
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with (frames_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF, like the frames.csv vc_extract writes
        writer.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for ref in refs:
            link = frames_dir / f"{ref.frame:06d}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(ref.image.resolve(), link)
            writer.writerow([ref.video, ref.frame, ref.pts_ms, "base", 0.0])
    out = work_dir / "cpp_detections.jsonl"
    out.unlink(missing_ok=True)  # a run that writes nothing must not be read as the previous run's result
    command = [
        str(vc_detect), "--model", str(model_dir), "--frames", str(frames_dir), "--out", str(out),
        "--ep", "cpu", "--score", str(score),
    ]
    try:
        runner(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"vc_detect failed (exit {exc.returncode}): {(exc.stderr or '').strip()}") from exc
    return detections_by_frame(out)[1]
