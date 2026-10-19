"""Writes <out>/model.onnx: images [1,3,64,64] -> output0 [1,6,4] = fixed YOLOv8 columns + 0 * mean(images).

With --fp16 the same graph ends in a Cast, so output0 is FLOAT16 (a model exported in half precision);
OnnxYoloDetector must refuse it.

Run once with the repo venv (onnx is installed there); the output is committed:
    .venv/bin/python -I cpp/tests/data/make_const_yolo.py cpp/tests/data/const_yolo
    .venv/bin/python -I cpp/tests/data/make_const_yolo.py --fp16 cpp/tests/data/const_yolo_fp16
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COLUMNS = [  # cx, cy, w, h, face, plate in 64x64 input pixels
    (16, 16, 8, 8, 0.9, 0.0),
    (17, 16, 8, 8, 0.6, 0.0),  # overlaps the first face, suppressed by NMS
    (40, 40, 16, 8, 0.0, 0.7),
    (50, 10, 4, 4, 0.05, 0.05),  # below the 0.15 threshold
]


def main(out_dir: Path, fp16: bool = False) -> None:
    boxes = np.array(COLUMNS, np.float32).T[None]  # [1, 6, 4]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["boxes", "zeroed"], ["output0" if not fp16 else "sum"]),
    ]
    if fp16:
        nodes.append(helper.make_node("Cast", ["sum"], ["output0"], to=TensorProto.FLOAT16))
    output_type = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "const_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", output_type, [1, 6, 4])],
        initializer=[numpy_helper.from_array(boxes, "boxes"), numpy_helper.from_array(np.zeros(1, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))


if __name__ == "__main__":
    args = sys.argv[1:]
    fp16 = "--fp16" in args
    paths = [a for a in args if a != "--fp16"]
    if len(paths) != 1:
        sys.exit("usage: make_const_yolo.py [--fp16] <out_dir>")
    main(Path(paths[0]), fp16)
