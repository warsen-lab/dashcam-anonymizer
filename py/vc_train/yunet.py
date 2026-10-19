"""YuNet face model for vc_redact --face-model, and the reference decoder the C++ YuNetDetector follows.

The OpenCV Zoo ONNX (MIT) declares a fixed [1, 3, 640, 640] input, but its graph does not depend on the size (Reshape to
[1, -1, k], Resize by scale 2). vc_train yunet marks the input height and width and the anchor count of every output
symbolic, so onnxruntime runs it at the frame's own resolution padded to multiples of 32, as cv2.FaceDetectorYN does.

Preprocessing and decoding are OpenCV's (FaceDetectorYN, OpenCV 4.8+): BGR as float 0-255 with no scale or mean,
zero-padded on the right and bottom; per stride s in (8, 16, 32) the anchor at row r, column c scores
sqrt(clamp(cls) * clamp(obj)) and its box is centred at ((c + b0) s, (r + b1) s) with size (exp(b2) s, exp(b3) s);
NMS (cv::dnn::NMSBoxes) takes the highest score first and suppresses an overlap above the IoU threshold, measured on the
boxes truncated to integer rectangles (cv::Rect2i), and keeps at most top_k candidates. One deliberate difference:
NMSBoxes also drops a score exactly equal to the threshold when there are several candidates; this keeps it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from vc_label.contracts import Box

MODEL_ONNX = "model.onnx"
MODEL_JSON = "model.json"
NAME = "yunet-2023mar"
SOURCE = "yunet"  # the `src` of a YuNet box in detections.jsonl
CLASS = "face"
STRIDES = (8, 16, 32)
PAD_MULTIPLE = 32
TOP_K = 5000
INPUT = "input"
# output kind -> values per anchor
OUTPUT_WIDTHS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
# int() of a float beyond the int range is undefined behaviour in C++; both decoders saturate here first
_INT_LIMIT = 1 << 30
_CHECK_SIZE = (96, 64)  # width, height of the test run that proves a written model is not tied to 640x640


def pad_size(width: int, height: int) -> tuple[int, int]:
    """Frame size rounded up to multiples of 32, as FaceDetectorYN.setInputSize pads it."""
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size {width}x{height} must be positive")
    return (width - 1) // PAD_MULTIPLE * PAD_MULTIPLE + PAD_MULTIPLE, (height - 1) // PAD_MULTIPLE * PAD_MULTIPLE + PAD_MULTIPLE


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> [1, 3, pad_h, pad_w] float32 0-255, zero-padded on the right and bottom."""
    height, width = bgr.shape[:2]
    pad_w, pad_h = pad_size(width, height)
    canvas = np.zeros((pad_h, pad_w, 3), np.float32)
    canvas[:height, :width] = bgr
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])


def _level(outputs: Mapping[str, np.ndarray], kind: str, stride: int, anchors: int) -> np.ndarray:
    name = f"{kind}_{stride}"
    width = OUTPUT_WIDTHS[kind]
    value = np.asarray(outputs[name], dtype=np.float32)
    if value.shape != (1, anchors, width):
        raise ValueError(f"{name} has shape {list(value.shape)}, expected [1, {anchors}, {width}]")
    return value[0]


def _candidates(outputs, pad_w: int, pad_h: int, score_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(boxes [n, 4] float32 x, y, w, h; scores [n] float32) of every anchor at or above the threshold, stride by stride."""
    boxes, scores = [], []
    for stride in STRIDES:
        cols, rows = pad_w // stride, pad_h // stride
        cls = _level(outputs, "cls", stride, rows * cols)[:, 0]
        obj = _level(outputs, "obj", stride, rows * cols)[:, 0]
        bbox = _level(outputs, "bbox", stride, rows * cols)
        # a healthy model never emits NaN / inf: such an anchor would fail every threshold test and vanish unmasked
        if not (np.isfinite(cls).all() and np.isfinite(obj).all()):
            raise ValueError("model output has non-finite scores")
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        keep = np.nonzero(score >= np.float32(score_threshold))[0]
        b = bbox[keep]
        s = np.float32(stride)
        with np.errstate(over="ignore", invalid="ignore"):
            w = np.exp(b[:, 2]) * s
            h = np.exp(b[:, 3]) * s
            x = (keep % cols).astype(np.float32) + b[:, 0]
            y = (keep // cols).astype(np.float32) + b[:, 1]
            xywh = np.stack([x * s - w / np.float32(2), y * s - h / np.float32(2), w, h], 1)
        if not (np.isfinite(b).all() and np.isfinite(xywh).all()):
            raise ValueError("model output has non-finite box coordinates")
        boxes.append(xywh.astype(np.float32))
        scores.append(score[keep])
    return np.concatenate(boxes), np.concatenate(scores)


def _int_rects(boxes: np.ndarray) -> np.ndarray:
    """x, y, w, h truncated toward zero like int(float) (cv::Rect2i), saturated to +-2^30 first."""
    return np.trunc(np.clip(boxes.astype(np.float64), -_INT_LIMIT, _INT_LIMIT)).astype(np.int64)


def _overlaps(rects: np.ndarray, kept: list[int], i: int) -> np.ndarray:
    """OpenCV's rectOverlap of rect i with each kept rect: 1.f - float(jaccardDistance) on integer rectangles."""
    k = rects[kept]
    x, y, w, h = rects[i]
    area, areas = w * h, k[:, 2] * k[:, 3]
    iw = np.minimum(x + w, k[:, 0] + k[:, 2]) - np.maximum(x, k[:, 0])
    ih = np.minimum(y + h, k[:, 1] + k[:, 3]) - np.maximum(y, k[:, 1])
    inter = np.where((iw > 0) & (ih > 0) & (w > 0) & (h > 0) & (k[:, 2] > 0) & (k[:, 3] > 0), iw * ih, 0)
    union = area + areas - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(area + areas <= 0, 0.0, 1.0 - inter / np.where(union > 0, union, 1))
    return np.float32(1.0) - distance.astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, nms_iou: float, top_k: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    if top_k > 0:
        order = order[:top_k]
    rects = _int_rects(boxes)
    kept: list[int] = []
    limit = np.float32(nms_iou)
    for i in order:
        if not kept or (_overlaps(rects, kept, int(i)) <= limit).all():
            kept.append(int(i))
    return kept


def decode_yunet(
    outputs: Mapping[str, np.ndarray],
    pad_w: int,
    pad_h: int,
    width: int,
    height: int,
    score_threshold: float,
    nms_iou: float,
    top_k: int = TOP_K,
) -> list[Box]:
    """Raw YuNet outputs ({"cls_8": [1, N, 1], ...}) for a pad_w x pad_h input -> face boxes in a width x height frame.

    Highest score first; boxes are clipped to the frame after NMS. Raises ValueError when an output has the wrong shape,
    when any cls / obj value is not finite, and when a box that passes the threshold has a non-finite value.
    """
    boxes, scores = _candidates(outputs, pad_w, pad_h, score_threshold)
    return [
        Box(CLASS, float(x), float(y), float(x + w), float(y + h), score=float(scores[i]), src=SOURCE).clipped(width, height)
        for i in _nms(boxes, scores, nms_iou, top_k)
        for x, y, w, h in [boxes[i]]
    ]


def yunet_session(onnx_path: Path):
    from vc_train.ort_cpu import cpu_session

    return cpu_session(onnx_path)


def detect_yunet(session, bgr: np.ndarray, score_threshold: float, nms_iou: float, top_k: int = TOP_K) -> list[Box]:
    """Faces in a BGR uint8 image at its own resolution (onnxruntime session on a model written by write_yunet_model)."""
    height, width = bgr.shape[:2]
    pad_w, pad_h = pad_size(width, height)
    names = [f"{kind}_{s}" for kind in ("cls", "obj", "bbox") for s in STRIDES]
    outputs = dict(zip(names, session.run(names, {INPUT: preprocess(bgr)})))
    return decode_yunet(outputs, pad_w, pad_h, width, height, score_threshold, nms_iou, top_k)


def _dims(value_info) -> list:
    return [d.dim_param if d.HasField("dim_param") else d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _check_source(model, source: Path) -> None:
    """The graph must have YuNet's input and its twelve outputs, all float32 (shapes as OpenCV Zoo publishes them)."""
    from onnx import TensorProto

    problems = []
    inputs = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    if [i.name for i in inputs] != [INPUT] or len(_dims(inputs[0])) != 4 or _dims(inputs[0])[:2] != [1, 3]:
        problems.append(f"inputs {[(i.name, _dims(i)) for i in inputs]}, expected one '{INPUT}' of shape [1, 3, H, W]")
    outputs = {o.name: o for o in model.graph.output}
    for kind, width in OUTPUT_WIDTHS.items():
        for s in STRIDES:
            o = outputs.get(f"{kind}_{s}")
            if o is None or len(_dims(o)) != 3 or _dims(o)[0] != 1 or _dims(o)[2] != width:
                problems.append(f"output {kind}_{s} is {None if o is None else _dims(o)}, expected [1, N, {width}]")
    types = [v.type.tensor_type.elem_type for v in (*inputs, *model.graph.output)]
    if any(t != TensorProto.FLOAT for t in types):
        problems.append("every input and output must be float32")
    if problems:
        raise ValueError(f"{source} is not the YuNet face model: " + "; ".join(problems))


def make_dynamic(model):
    """A copy of the model whose input height / width are the symbols H / W and whose outputs have N anchors.

    value_info is dropped: it records the 640x640 shapes of the intermediate tensors, which onnxruntime would check.
    """
    import onnx

    dynamic = onnx.ModelProto()
    dynamic.CopyFrom(model)
    dims = dynamic.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param = "H"
    dims[3].dim_param = "W"
    for output in dynamic.graph.output:
        output.type.tensor_type.shape.dim[1].dim_param = "N"
    del dynamic.graph.value_info[:]
    onnx.checker.check_model(dynamic)
    return dynamic


def _check_runs(onnx_path: Path) -> None:
    """Runs the written model on a small input that is not 640x640 and checks every output's anchor count."""
    width, height = _CHECK_SIZE
    session = yunet_session(onnx_path)
    names = [o.name for o in session.get_outputs()]
    results = dict(zip(names, session.run(None, {INPUT: np.zeros((1, 3, height, width), np.float32)})))
    for kind in OUTPUT_WIDTHS:
        for s in STRIDES:
            _level(results, kind, s, (width // s) * (height // s))


def write_yunet_model(source: Path, out_dir: Path) -> Path:
    """model.onnx (free input size) + model.json into out_dir, which must not exist or be empty.

    Everything is checked before out_dir receives a file: the source is YuNet, and the converted model runs at a size
    other than 640x640 with the expected anchor counts.
    """
    import onnx

    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} is not empty; choose a new --out")
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist; see py/README.md, section 模型权重")
    try:
        model = onnx.load(str(source))
    except Exception as e:  # a file that is not ONNX at all fails in protobuf, with its own exception types
        raise ValueError(f"{source} is not the YuNet face model: {e}") from e
    _check_source(model, source)
    dynamic = make_dynamic(model)
    meta = {
        "name": NAME,
        "classes": [CLASS],
        "input": {"color": "BGR", "scale": 1.0, "letterbox": "top_left", "pad_value": 0, "pad_multiple": PAD_MULTIPLE},
        "output_format": "yunet",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / MODEL_ONNX
        onnx.save(dynamic, str(staged))
        _check_runs(staged)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), out_dir / MODEL_ONNX)
    (out_dir / MODEL_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir / MODEL_ONNX
