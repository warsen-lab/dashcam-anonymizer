"""ONNX export with model.json, and the PyTorch vs onnxruntime consistency check (spec §4.4)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from vc_label.boxes import iou
from vc_label.contracts import Box
from vc_train.geometry import PAD_VALUE, check_input_size, decode_yolov8, letterbox
from vc_train.offline import load_yolo, ultralytics_offline_env
from vc_train.ort_cpu import cpu_session

MODEL_ONNX = "model.onnx"
MODEL_JSON = "model.json"
MAP50_KEY = "metrics/mAP50(B)"
CONFIDENT = 0.3  # an unmatched box at or above this score fails the consistency check; same value as vc_eval.consistency
MATCH_IOU = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_map50(weights: Path) -> float:
    """mAP50 that ultralytics stored in the checkpoint: the validation of the very weights being exported."""
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    train_metrics = ckpt.get("train_metrics") if isinstance(ckpt, dict) else None
    if not isinstance(train_metrics, dict) or train_metrics.get(MAP50_KEY) is None:
        raise ValueError(f"{weights} has no train_metrics[{MAP50_KEY!r}]; cannot report val_map50")
    return float(train_metrics[MAP50_KEY])


def _check_onnx_io(onnx_path: Path, in_width: int, in_height: int, n_classes: int) -> None:
    """Reads the exported graph back with onnxruntime.

    One static NCHW input, float32 inputs and outputs only, and one output of shape [1, 4 + n_classes, N] with a fixed N > 0
    (the layout decode_yolov8 and the C++ decoder read).
    """
    session = cpu_session(onnx_path)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    shapes = [list(i.shape) for i in inputs]
    if shapes != [[1, 3, in_height, in_width]]:
        raise ValueError(f"{onnx_path} has input shapes {shapes}, expected one input of shape [1, 3, {in_height}, {in_width}]")
    types = [i.type for i in inputs] + [o.type for o in outputs]
    if any(t != "tensor(float)" for t in types):
        raise ValueError(f"{onnx_path} must use tensor(float) for every input and output, got {types}")
    out_shapes = [list(o.shape) for o in outputs]
    ok = (
        len(out_shapes) == 1
        and len(out_shapes[0]) == 3
        and out_shapes[0][:2] == [1, 4 + n_classes]
        and isinstance(out_shapes[0][2], int)
        and out_shapes[0][2] > 0
    )
    if not ok:
        raise ValueError(
            f"{onnx_path} has output shapes {out_shapes}, expected one output of shape [1, {4 + n_classes}, N] with a fixed N > 0"
        )


def export_onnx(
    weights: Path,
    out_dir: Path,
    in_width: int,
    in_height: int,
    manifest: Path | None = None,
    name: str | None = None,
) -> Path:
    """Static [1, 3, in_height, in_width] ONNX without NMS, plus model.json, into out_dir.

    out_dir only receives files once the exported graph has been read back and has the expected static float shape.
    """
    # first: reading the checkpoint unpickles ultralytics classes, and importing ultralytics already runs its network check
    ultralytics_offline_env()
    check_input_size(in_width, in_height)
    map50 = _checkpoint_map50(Path(weights))
    model = load_yolo(weights)
    exported = Path(
        model.export(format="onnx", imgsz=(in_height, in_width), opset=17, simplify=False, dynamic=False, nms=False, device="cpu")
    )
    classes = [model.names[i] for i in sorted(model.names)]
    _check_onnx_io(exported, in_width, in_height, len(classes))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / MODEL_ONNX
    shutil.move(str(exported), target)
    meta = {
        "name": name or out_dir.name,
        "classes": classes,
        "input": {"width": in_width, "height": in_height, "color": "RGB", "scale": 1 / 255, "letterbox": "top_left", "pad_value": PAD_VALUE},
        "output_format": "yolov8",
        "train_manifest_sha256": _sha256(manifest) if manifest else "",
        "metrics": {"val_map50": map50},
    }
    (out_dir / MODEL_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return target


def read_model_json(out_dir: Path) -> dict:
    return json.loads((out_dir / MODEL_JSON).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ConsistencyReport:
    """PyTorch vs onnxruntime on the same tensors; unmatched boxes are split by score like vc_eval.consistency does."""

    images: int
    boxes: int  # boxes the PyTorch run found
    matched: int
    max_diff_px: float
    unmatched_confident: int  # unmatched boxes (either side) with score >= CONFIDENT: these fail the check
    unmatched_low: int  # unmatched boxes below CONFIDENT: reported, tolerated (a score next to the threshold may flip)

    def passed(self, tol_px: float = 1.0) -> bool:
        # no images or no matched boxes is no evidence of agreement, however empty the unmatched counts are
        return self.images > 0 and self.matched > 0 and self.unmatched_confident == 0 and self.max_diff_px < tol_px


def match_boxes(a: list[Box], b: list[Box]) -> tuple[int, int, int, float]:
    """(matched, unmatched_confident, unmatched_low, worst corner difference in px) for the boxes of one image.

    Each box of `a` takes the best-overlapping unused box of `b` with the same class, if the IoU is at least MATCH_IOU.
    """
    matched = confident = low = 0
    worst = 0.0
    remaining = list(b)
    for box in a:
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
    return matched, confident, low, worst


def check_consistency(
    weights: Path,
    onnx_path: Path,
    images: list[Path],
    score_threshold: float = 0.15,
    iou_threshold: float = 0.5,
) -> ConsistencyReport:
    """Runs the same preprocessed tensors through PyTorch (CPU) and onnxruntime and compares decoded boxes."""
    ultralytics_offline_env()
    import torch

    net = load_yolo(weights).model.float().eval()
    classes = [net.names[i] for i in sorted(net.names)]
    session = cpu_session(onnx_path)
    inp = session.get_inputs()[0]
    in_height, in_width = inp.shape[2], inp.shape[3]
    total = matched = confident = low = 0
    worst = 0.0
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
        image_matched, image_confident, image_low, image_worst = match_boxes(a, b)
        total += len(a)
        matched += image_matched
        confident += image_confident
        low += image_low
        worst = max(worst, image_worst)
    return ConsistencyReport(
        images=len(images), boxes=total, matched=matched, max_diff_px=worst, unmatched_confident=confident, unmatched_low=low
    )
