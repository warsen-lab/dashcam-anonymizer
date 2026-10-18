import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, Detection, read_frames_csv, write_jsonl
from vc_eval.consistency import ConsistencyReport, compare_detections, python_detections, run_vc_detect


def _const_model(dir_path):
    """64x64 model returning fixed YOLOv8 columns: a face (0.9) and a plate (0.7)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    columns = [(16, 16, 8, 8, 0.9, 0.0), (40, 40, 16, 8, 0.0, 0.7), (50, 10, 4, 4, 0.05, 0.05)]
    boxes = np.array(columns, np.float32).T[None]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["boxes", "zeroed"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "const", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 3])],
        initializer=[numpy_helper.from_array(boxes, "boxes"), numpy_helper.from_array(np.zeros(1, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    dir_path.mkdir(parents=True)
    onnx.save(model, str(dir_path / "model.onnx"))
    (dir_path / "model.json").write_text(json.dumps({
        "name": "const", "classes": ["face", "plate"],
        "input": {"width": 64, "height": 64, "color": "RGB", "scale": 1 / 255, "letterbox": "top_left", "pad_value": 114},
        "output_format": "yolov8", "train_manifest_sha256": "", "metrics": {"val_map50": 0.0},
    }))
    return dir_path


def _frames(root, n=2, size=128):
    d = root / "frames" / "t"
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i in range(n):
            cv2.imwrite(str(d / f"{i * 5:06d}.jpg"), np.full((size, size, 3), 128, np.uint8))
            w.writerow(["t.mp4", i * 5, i * 83.3, "base", 0.0])
    return d


def test_python_detections_decode_the_model(tmp_path):
    model = _const_model(tmp_path / "m")
    refs = read_frames_csv(_frames(tmp_path))
    dets = python_detections(model, refs)
    assert sorted(dets) == [0, 5]
    face, plate = sorted(dets[0], key=lambda b: b.cls)
    assert (face.x1, face.y1, face.x2, face.y2) == pytest.approx((24, 24, 40, 40))
    assert (plate.x1, plate.y1, plate.x2, plate.y2) == pytest.approx((64, 72, 96, 88))


def test_compare_detections_matches_per_class():
    a = {0: [Box("face", 10, 10, 20, 20, score=0.9), Box("plate", 50, 50, 70, 60, score=0.2)]}
    b = {0: [Box("face", 10.4, 10, 20, 20.2, score=0.9)]}
    r = compare_detections(a, b)
    assert (r.images, r.boxes, r.matched) == (1, 2, 1)
    assert r.max_diff_px == pytest.approx(0.4)
    assert (r.unmatched_confident, r.unmatched_low) == (0, 1)
    assert r.passed()
    worse = compare_detections({0: [Box("face", 10, 10, 20, 20, score=0.9)]}, {0: []})
    assert worse.unmatched_confident == 1 and not worse.passed()


def test_run_vc_detect_builds_a_subset_frames_dir_and_reads_its_output(tmp_path):
    frames = _frames(tmp_path)
    refs = read_frames_csv(frames)[1:]
    calls = []

    def fake_runner(cmd, check, capture_output, text):
        calls.append(cmd)
        frames_arg = cmd[cmd.index("--frames") + 1]
        out = cmd[cmd.index("--out") + 1]
        with open(f"{frames_arg}/frames.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["frame"] for r in rows] == ["5"]
        write_jsonl(__import__("pathlib").Path(out), [Detection("t.mp4", 5, 83.3, Box("face", 1, 2, 3, 4, score=0.8, src="student"))])

    dets = run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, tmp_path / "work", runner=fake_runner)
    assert list(dets) == [5] and dets[5][0].x2 == 3
    assert calls[0][calls[0].index("--ep") + 1] == "cpu"


def test_zero_boxes_on_both_sides_is_not_a_pass():
    assert not compare_detections({}, {}).passed()
    assert not compare_detections({0: []}, {0: []}).passed()
    assert not ConsistencyReport(0, 0, 0, 0.0, 0, 0).passed()


def test_nothing_matched_is_not_a_pass_even_without_confident_misses():
    r = compare_detections({0: [Box("face", 10, 10, 20, 20, score=0.2)]}, {0: []})
    assert (r.images, r.matched, r.unmatched_confident) == (1, 0, 0)
    assert not r.passed()


def test_run_vc_detect_forwards_the_score_and_defaults_to_the_decoder_default(tmp_path):
    refs = read_frames_csv(_frames(tmp_path))[:1]
    calls = []

    def fake_runner(cmd, check, capture_output, text):
        calls.append(cmd)
        write_jsonl(Path(cmd[cmd.index("--out") + 1]), [])

    run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, tmp_path / "w1", runner=fake_runner)
    run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, tmp_path / "w2", runner=fake_runner, score=0.3)
    assert [c[c.index("--score") + 1] for c in calls] == ["0.15", "0.3"]


def test_run_vc_detect_removes_a_stale_output_before_running(tmp_path):
    refs = read_frames_csv(_frames(tmp_path))[:1]
    work = tmp_path / "work"
    work.mkdir()
    stale = work / "cpp_detections.jsonl"
    write_jsonl(stale, [Detection("t.mp4", 0, 0.0, Box("face", 9, 9, 19, 19, score=0.9, src="student"))])
    seen = []

    def fresh_runner(cmd, check, capture_output, text):
        out = Path(cmd[cmd.index("--out") + 1])
        seen.append(out.exists())
        write_jsonl(out, [])

    assert run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, work, runner=fresh_runner) == {}
    assert seen == [False]

    write_jsonl(stale, [Detection("t.mp4", 0, 0.0, Box("face", 9, 9, 19, 19, score=0.9, src="student"))])
    with pytest.raises(ValueError):  # a run that writes nothing must not fall back to the stale file
        run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, work, runner=lambda *a, **k: None)


def test_run_vc_detect_turns_a_failed_run_into_a_value_error_with_its_stderr(tmp_path):
    refs = read_frames_csv(_frames(tmp_path))[:1]

    def failing_runner(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(3, cmd, output="", stderr="model.onnx: bad magic number")

    with pytest.raises(ValueError, match="bad magic number"):
        run_vc_detect(tmp_path / "vc_detect", tmp_path / "m", refs, tmp_path / "work", runner=failing_runner)
