"""Writes <out>/model.onnx: images [1,3,64,64] -> output0 [1,6,4] = fixed YOLOv8 columns + 0 * mean(images).

Run once with the repo venv (onnx is installed there); the output is committed:
    .venv/bin/python cpp/tests/data/make_const_yolo.py cpp/tests/data/const_yolo
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


def main(out_dir: Path) -> None:
    boxes = np.array(COLUMNS, np.float32).T[None]  # [1, 6, 4]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["boxes", "zeroed"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "const_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])],
        initializer=[numpy_helper.from_array(boxes, "boxes"), numpy_helper.from_array(np.zeros(1, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
