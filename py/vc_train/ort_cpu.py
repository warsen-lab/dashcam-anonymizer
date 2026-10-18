"""The onnxruntime CPU session the Python reference paths use (export checks, the C++/Python consistency check)."""

from __future__ import annotations

from pathlib import Path


def cpu_session(onnx_path: Path):
    """A single-threaded CPU session.

    onnxruntime's own thread pools racing interpreter teardown showed up as an intermittent libc++ "recursive_mutex lock
    failed" abort at exit; these paths only run a handful of frames, so one thread costs little.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
