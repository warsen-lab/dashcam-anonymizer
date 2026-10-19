"""Writes <out>/model.onnx + model.json: a tiny model with YuNet's inputs and outputs, for the C++ YuNetDetector tests.

input [1, 3, H, W] (H and W symbolic) -> cls_s / obj_s [1, N, 1], bbox_s [1, N, 4], kps_s [1, N, 10] for s in 8, 16, 32,
N = (H / s) * (W / s). Stride 32 reads the image: an anchor's cls is the mean of channel 0 (blue) over its 32x32 cell
divided by 255, its obj the mean of channel 1 (green) / 255, and its box is the cell itself (bbox 0.5, 0.5, 0, 0). So a
cell filled with B = G = 255 scores sqrt(1 * 1) = 1, a red one 0; feeding RGB instead of BGR, scaling the input to 0-1 or
padding with anything but 0 all change the scores. Strides 8 and 16 score 0 everywhere. The outputs are listed in a
scrambled order, so the detector must find them by name.

With --static the input is a fixed [1, 3, 64, 64] (like the OpenCV Zoo file before vc_train yunet).

Run once with the repo venv (onnx is installed there); the output is committed:
    .venv/bin/python -I cpp/tests/data/make_const_yunet.py cpp/tests/data/const_yunet
    .venv/bin/python -I cpp/tests/data/make_const_yunet.py --static cpp/tests/data/const_yunet_static
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

STRIDES = (8, 16, 32)
WIDTHS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}


def _const(name, values, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(values, dtype), name)


def main(out_dir: Path, static: bool = False) -> None:
    nodes, inits = [], [
        _const("shape_n1", [1, -1, 1], np.int64),
        _const("zero", [0.0]),
        _const("inv255", [1.0 / 255.0]),
        _const("cell", [0.5, 0.5, 0.0, 0.0]),
        _const("kps0", np.zeros(10)),
        _const("ax1", [1], np.int64),
        _const("c0", [0], np.int64),
        _const("c1", [1], np.int64),
        _const("c2", [2], np.int64),
    ]
    for s in STRIDES:
        p = f"s{s}_"
        nodes += [
            # explicit pads: onnxruntime's CoreML converter needs them
            helper.make_node("AveragePool", ["input"], [p + "pool"], kernel_shape=[s, s], strides=[s, s], pads=[0, 0, 0, 0]),
            helper.make_node("Slice", [p + "pool", "c0", "c1", "ax1"], [p + "b"]),
            helper.make_node("Slice", [p + "pool", "c1", "c2", "ax1"], [p + "g"]),
        ]
        for channel, kind in ((p + "b", "cls"), (p + "g", "obj")):
            nodes += [
                helper.make_node("Transpose", [channel], [p + kind + "_nhwc"], perm=[0, 2, 3, 1]),
                helper.make_node("Reshape", [p + kind + "_nhwc", "shape_n1"], [p + kind + "_raw"]),
                helper.make_node("Mul", [p + kind + "_raw", "inv255" if s == 32 else "zero"], [f"{kind}_{s}"]),
            ]
        nodes += [
            helper.make_node("Mul", [p + "cls_raw", "zero"], [p + "zeros"]),
            helper.make_node("Add", [p + "zeros", "cell"], [f"bbox_{s}"]),
            helper.make_node("Add", [p + "zeros", "kps0"], [f"kps_{s}"]),
        ]
    height, width = (64, 64) if static else ("H", "W")
    outputs = [
        helper.make_tensor_value_info(f"{kind}_{s}", TensorProto.FLOAT, [1, (height * width // (s * s)) if static else "N", WIDTHS[kind]])
        for kind in ("kps", "bbox", "obj", "cls")
        for s in reversed(STRIDES)
    ]
    graph = helper.make_graph(
        nodes, "const_yunet", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])], outputs, initializer=inits
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))
    meta = {
        "name": "const-yunet",
        "classes": ["face"],
        "input": {"color": "BGR", "scale": 1.0, "letterbox": "top_left", "pad_value": 0, "pad_multiple": 32},
        "output_format": "yunet",
        "source_sha256": "",
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    static = "--static" in args
    paths = [a for a in args if a != "--static"]
    if len(paths) != 1:
        sys.exit("usage: make_const_yunet.py [--static] <out_dir>")
    main(Path(paths[0]), static)
