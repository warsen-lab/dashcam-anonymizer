"""Writes <out>/model.onnx: crops [4,3,64,64] -> logits [4,2] = (-2, +2) per row + 0 * mean(crops).

The face logit -2 (sigmoid 0.119) fails and the plate logit +2 (0.881) passes the 0.5 thresholds of the
model.json next to it, so tests can tell a rejected face from a kept plate.

Run once with the repo venv; the output is committed:
    .venv/bin/python -I cpp/tests/data/make_const_verifier.py cpp/tests/data/const_verifier
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BATCH, SIZE = 4, 64


def main(out_dir: Path) -> None:
    logits = np.tile(np.array([[-2.0, 2.0]], np.float32), (BATCH, 1))
    nodes = [
        helper.make_node("ReduceMean", ["crops"], ["mean"], axes=[1, 2, 3], keepdims=1),  # [4, 1, 1, 1]
        helper.make_node("Reshape", ["mean", "shape"], ["flat"]),  # [4, 1]
        helper.make_node("Mul", ["flat", "zero"], ["zeroed"]),
        helper.make_node("Add", ["logits_const", "zeroed"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "const_verifier",
        [helper.make_tensor_value_info("crops", TensorProto.FLOAT, [BATCH, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, 2])],
        initializer=[
            numpy_helper.from_array(logits, "logits_const"),
            numpy_helper.from_array(np.zeros(1, np.float32), "zero"),
            numpy_helper.from_array(np.array([BATCH, 1], np.int64), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_const_verifier.py <out_dir>")
    main(Path(sys.argv[1]))
