"""Writes yunet_synthetic_expected.json: what cv2.FaceDetectorYN finds on a synthetic image, for the C++ YuNet test.

The image is drawn from integers only (a colour gradient and filled ellipses: four cartoon faces, no real person), so
the C++ test paints the same pixels from the ellipse list in the JSON and checks image_sha256 before comparing boxes.
The boxes are cv2's (OpenCV's own decoder and NMS, on the cached OpenCV Zoo model) at score 0.01, NMS IoU 0.3, top_k
5000, clipped to the image like vc_redact clips them, rounded to 4 decimals.

Run with the repo venv after changing the image or the model; the output is committed:
    .venv/bin/python -I cpp/tests/data/make_yunet_expected.py cpp/tests/data/yunet_synthetic_expected.json
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

WIDTH = HEIGHT = 640
SCORE_THRESHOLD = 0.01
NMS_IOU = 0.3
TOP_K = 5000
# (cx, cy, rx, ry) of each cartoon face in pixels
FACES = ((160, 200, 60, 80), (420, 260, 90, 115), (300, 500, 30, 38), (540, 520, 20, 26))


def ellipses() -> list[tuple[int, int, int, int, int, int, int]]:
    """Filled ellipses (cx, cy, rx, ry, b, g, r) in painting order: hair, skin, eye whites and pupils, nose, mouth."""
    out = []
    for cx, cy, rx, ry in FACES:
        out.append((cx, cy - ry // 4, rx + rx // 10, ry, 40, 50, 70))
        out.append((cx, cy + ry // 10, rx, ry, 120, 160, 215))
        for side in (-1, 1):
            ex, ey = cx + side * (rx * 2 // 5), cy - ry // 10
            out.append((ex, ey, rx // 5, ry // 10, 245, 245, 245))
            out.append((ex, ey, rx // 10, ry // 10, 40, 30, 30))
        out.append((cx, cy + ry * 2 // 10, rx // 10, ry // 6, 100, 130, 190))
        out.append((cx, cy + ry * 5 // 10, rx * 3 // 10, ry // 12, 70, 70, 160))
    return out


def synthetic_faces(width: int, height: int) -> np.ndarray:
    """BGR uint8: B = 60 + 80x/W, G = 90 + 60y/H, R = 70 (integer division), then every ellipse of ellipses() filled."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    img = np.stack([60 + x * 80 // width, 90 + y * 60 // height, np.full_like(x, 70)], -1)
    for cx, cy, rx, ry, b, g, r in ellipses():
        inside = (x - cx) ** 2 * ry * ry + (y - cy) ** 2 * rx * rx <= rx * rx * ry * ry
        img[inside] = (b, g, r)
    return np.ascontiguousarray(img.astype(np.uint8))


def expectation(model: Path) -> dict:
    import cv2

    bgr = synthetic_faces(WIDTH, HEIGHT)
    detector = cv2.FaceDetectorYN.create(str(model), "", (WIDTH, HEIGHT), SCORE_THRESHOLD, NMS_IOU, TOP_K)
    _, faces = detector.detect(bgr)
    clip = lambda v, hi: round(min(max(float(v), 0.0), float(hi)), 4)  # noqa: E731
    return {
        "generator": "cpp/tests/data/make_yunet_expected.py",
        "width": WIDTH,
        "height": HEIGHT,
        "ellipses": [list(e) for e in ellipses()],
        "image_sha256": hashlib.sha256(bgr.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        "score_threshold": SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "top_k": TOP_K,
        "faces": [
            {
                "x1": clip(f[0], WIDTH),
                "y1": clip(f[1], HEIGHT),
                "x2": clip(f[0] + f[2], WIDTH),
                "y2": clip(f[1] + f[3], HEIGHT),
                "score": round(float(f[-1]), 6),
            }
            for f in ([] if faces is None else faces)
        ],
    }


def dumps(data: dict) -> str:
    """Indented JSON with each list of numbers (an ellipse) on one line."""
    text = json.dumps(data, indent=2)
    return re.sub(r"\[\s+([-\d.,\s]+?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)


def main(out: Path) -> None:
    cache = Path(os.environ.get("VC_CACHE", Path.home() / ".cache" / "video-compliance"))
    model = cache / "models" / "face_detection_yunet_2023mar.onnx"
    if not model.is_file():
        sys.exit(f"missing {model}; see py/README.md, section 模型权重")
    out.write_text(dumps(expectation(model)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_yunet_expected.py <out.json>")
    main(Path(sys.argv[1]))
