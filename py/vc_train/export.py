"""ONNX export with model.json, and the PyTorch vs onnxruntime consistency check (spec §4.4)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_label.boxes import iou
from vc_train.geometry import PAD_VALUE, decode_yolov8, letterbox
from vc_train.offline import load_yolo

MODEL_ONNX = "model.onnx"
MODEL_JSON = "model.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_map50(weights: Path) -> float:
    results = weights.parent.parent / "results.csv"
    if not results.exists():
        return 0.0
    with results.open(newline="") as f:
        rows = list(csv.DictReader(f))
    values = [float(v) for r in rows for k, v in r.items() if k.strip() == "metrics/mAP50(B)" and v.strip()]
    return max(values, default=0.0)


def export_onnx(
    weights: Path,
    out_dir: Path,
    in_width: int,
    in_height: int,
    manifest: Path | None = None,
    name: str | None = None,
) -> Path:
    """Static [1, 3, in_height, in_width] ONNX without NMS, plus model.json, into out_dir."""
    model = load_yolo(weights)
    exported = Path(
        model.export(format="onnx", imgsz=(in_height, in_width), opset=17, simplify=False, dynamic=False, nms=False, device="cpu")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / MODEL_ONNX
    shutil.move(str(exported), target)
    classes = [model.names[i] for i in sorted(model.names)]
    meta = {
        "name": name or out_dir.name,
        "classes": classes,
        "input": {"width": in_width, "height": in_height, "color": "RGB", "scale": 1 / 255, "letterbox": "top_left", "pad_value": PAD_VALUE},
        "output_format": "yolov8",
        "train_manifest_sha256": _sha256(manifest) if manifest else "",
        "metrics": {"val_map50": float(_best_map50(Path(weights)))},
    }
    (out_dir / MODEL_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return target


def read_model_json(out_dir: Path) -> dict:
    return json.loads((out_dir / MODEL_JSON).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ConsistencyReport:
    images: int
    boxes: int
    max_diff_px: float
    unmatched: int

    def passed(self, tol_px: float = 1.0) -> bool:
        return self.unmatched == 0 and self.max_diff_px < tol_px


def check_consistency(
    weights: Path,
    onnx_path: Path,
    images: list[Path],
    score_threshold: float = 0.15,
    iou_threshold: float = 0.5,
) -> ConsistencyReport:
    """Runs the same preprocessed tensors through PyTorch (CPU) and onnxruntime and compares decoded boxes."""
    import onnxruntime as ort
    import torch

    net = load_yolo(weights).model.float().eval()
    classes = [net.names[i] for i in sorted(net.names)]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    in_height, in_width = inp.shape[2], inp.shape[3]
    total, unmatched, worst = 0, 0, 0.0
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cannot read {path}")
        height, width = bgr.shape[:2]
        arr, r = letterbox(bgr, in_width, in_height)
        with torch.no_grad():
            out = net(torch.from_numpy(arr))
        torch_pred = (out[0] if isinstance(out, (list, tuple)) else out).numpy()
        ort_pred = session.run(None, {inp.name: arr})[0]
        a = decode_yolov8(torch_pred, r, classes, width, height, score_threshold, iou_threshold)
        b = decode_yolov8(ort_pred, r, classes, width, height, score_threshold, iou_threshold)
        total += len(a)
        remaining = list(b)
        for box in a:
            same = [o for o in remaining if o.cls == box.cls]
            match = max(same, key=lambda o: iou(o, box), default=None)
            if match is None or iou(match, box) < 0.5:
                unmatched += 1
                continue
            remaining.remove(match)
            worst = max(worst, *(abs(p - q) for p, q in zip((box.x1, box.y1, box.x2, box.y2), (match.x1, match.y1, match.x2, match.y2))))
        unmatched += len(remaining)
    return ConsistencyReport(images=len(images), boxes=total, max_diff_px=worst, unmatched=unmatched)
