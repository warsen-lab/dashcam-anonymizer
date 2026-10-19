"""The onnxruntime CPU session the Python reference paths use (export checks, the C++/Python consistency check)."""

from __future__ import annotations

import os
from pathlib import Path


def cpu_session(onnx_path: Path):
    """A single-threaded CPU session with onnxruntime's usage telemetry switched off.

    Telemetry: onnxruntime (macOS wheel) uploads usage events to Microsoft from a background thread, which is traffic on
    a metered link, and exiting the interpreter while one of those responses is being handled destroys a mutex that thread
    still locks (an intermittent libc++ "recursive_mutex lock failed" abort, exit 134, about ten seconds after the first
    session). The switch is process-wide and must be thrown before the session is created.

    One thread: these paths only run a handful of frames, so it costs little.
    """
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"  # read if this import is the first one and creates the environment
    import onnxruntime as ort

    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
