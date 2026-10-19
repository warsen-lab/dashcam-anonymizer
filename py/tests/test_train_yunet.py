"""vc_train yunet: the OpenCV Zoo YuNet face model with a free input size, and the reference decoder vc_redact follows."""

import hashlib
import importlib.util
import json
import math
import os
import stat
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from vc_label.paths import YUNET_MODEL
from vc_train.cli import main
from vc_train.yunet import (
    MODEL_JSON,
    MODEL_ONNX,
    STRIDES,
    decode_yunet,
    detect_yunet,
    pad_size,
    preprocess,
    yunet_session,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_DATA = REPO_ROOT / "cpp" / "tests" / "data"
needs_yunet = pytest.mark.skipif(not YUNET_MODEL.exists(), reason=f"missing {YUNET_MODEL}")


def _generator():
    """cpp/tests/data/make_yunet_expected.py, which draws the synthetic image both languages test on."""
    spec = importlib.util.spec_from_file_location("make_yunet_expected", CPP_DATA / "make_yunet_expected.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outputs(pad_w, pad_h, anchors=()):
    """Raw YuNet outputs for a pad_w x pad_h input: every score 0, except `anchors` = (stride, row, col, cls, obj, bbox)."""
    out = {}
    for s in STRIDES:
        n = (pad_w // s) * (pad_h // s)
        out[f"cls_{s}"] = np.zeros((1, n, 1), np.float32)
        out[f"obj_{s}"] = np.zeros((1, n, 1), np.float32)
        out[f"bbox_{s}"] = np.zeros((1, n, 4), np.float32)
        out[f"kps_{s}"] = np.zeros((1, n, 10), np.float32)
    for s, r, c, cls, obj, bbox in anchors:
        idx = r * (pad_w // s) + c
        out[f"cls_{s}"][0, idx, 0] = cls
        out[f"obj_{s}"][0, idx, 0] = obj
        out[f"bbox_{s}"][0, idx] = bbox
    return out


def _bbox_for(stride, col, row, x1, y1, w, h):
    """The raw bbox values that decode to (x1, y1, w, h) at this anchor."""
    return ((x1 + w / 2) / stride - col, (y1 + h / 2) / stride - row, math.log(w / stride), math.log(h / stride))


def _decode(out, pad_w=64, pad_h=32, width=64, height=32, score=0.3, nms=0.3, **kw):
    return decode_yunet(out, pad_w, pad_h, width, height, score, nms, **kw)


def test_pad_size_rounds_up_to_multiples_of_32():
    assert pad_size(1920, 1080) == (1920, 1088)
    assert pad_size(3840, 2160) == (3840, 2176)
    assert pad_size(32, 33) == (32, 64)
    assert pad_size(1, 1) == (32, 32)


def test_preprocess_keeps_bgr_unscaled_and_pads_right_and_bottom_with_zeros():
    bgr = np.zeros((30, 40, 3), np.uint8)
    bgr[:, :] = (10, 20, 30)
    bgr[0, 0] = (255, 128, 1)
    blob = preprocess(bgr)
    assert blob.shape == (1, 3, 32, 64) and blob.dtype == np.float32
    assert blob[0, :, 0, 0].tolist() == [255.0, 128.0, 1.0]  # B, G, R planes; no scale, no mean
    assert blob[0, :, 29, 39].tolist() == [10.0, 20.0, 30.0]
    assert not blob[0, :, 30:, :].any() and not blob[0, :, :, 40:].any()


def test_decode_scores_the_root_of_cls_times_obj_and_maps_the_anchor_to_pixels():
    # stride 16 on a 64x32 input: 4 columns, 2 rows; anchor row 1, column 2
    out = _outputs(64, 32, [(16, 1, 2, 0.81, 0.25, (0.5, 0.25, math.log(2), 0.0))])
    [box] = _decode(out, width=64, height=32)
    assert box.cls == "face" and box.src == "yunet"
    assert box.score == pytest.approx(0.45, abs=1e-6)
    # cx = (2 + 0.5) * 16 = 40, cy = (1 + 0.25) * 16 = 20, w = 2 * 16, h = 16
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((24, 12, 56, 28), abs=1e-4)


def test_decode_reads_every_stride_in_row_major_order():
    anchors = [(8, 3, 7, 1, 0.64, (0.5, 0.5, 0, 0)), (32, 0, 1, 1, 0.36, (0.5, 0.5, 0, 0))]
    boxes = _decode(_outputs(64, 32, anchors), score=0.1)
    assert [(b.x1, b.y1, b.x2, b.y2) for b in boxes] == [(56, 24, 64, 32), (32, 0, 64, 32)]
    assert [b.score for b in boxes] == pytest.approx([0.8, 0.6])


def test_decode_clamps_cls_and_obj_to_zero_one_before_the_root():
    anchors = [(32, 0, 0, 1.5, 1.2, (0.5, 0.5, 0, 0)), (32, 0, 1, -0.5, 1.0, (0.5, 0.5, 0, 0))]
    boxes = _decode(_outputs(64, 32, anchors), score=0.0)
    assert boxes[0].score == 1.0
    assert all(b.score == 0.0 for b in boxes[1:])


def test_the_lower_clamp_makes_a_negative_score_zero():
    # unclamped, cls = obj = -0.5 would score sqrt(0.25) = 0.5
    out = _outputs(64, 32, [(32, 0, 1, -0.5, -0.5, _bbox_for(32, 1, 0, 40, 4, 20, 10))])
    assert _decode(out, score=0.3) == []
    # at threshold 0 every anchor is kept and nothing is suppressed (an overlap is never above 1): the clamped one scores 0
    boxes = _decode(out, score=0.0, nms=1.0)
    [clamped] = [b for b in boxes if (b.x1, b.y1, b.x2, b.y2) == pytest.approx((40, 4, 60, 14), abs=1e-3)]
    assert clamped.score == 0.0


def test_decode_keeps_a_score_equal_to_the_threshold():
    out = _outputs(64, 32, [(32, 0, 0, 0.25, 0.25, (0.5, 0.5, 0, 0))])  # sqrt(0.0625) = 0.25 exactly
    assert len(_decode(out, score=0.25)) == 1
    assert _decode(out, score=0.2500001) == []


def test_nms_suppresses_only_an_overlap_above_the_threshold():
    # A is 20x10, B is 10x10 inside it: IoU exactly 0.5
    anchors = [
        (8, 0, 0, 1, 0.81, _bbox_for(8, 0, 0, 0, 0, 20, 10)),
        (8, 0, 1, 1, 0.64, _bbox_for(8, 1, 0, 0, 0, 10, 10)),
    ]
    out = _outputs(64, 32, anchors)
    assert len(_decode(out, nms=0.5)) == 2
    assert len(_decode(out, nms=0.49)) == 1


def test_nms_measures_overlap_on_truncated_integer_rectangles_like_opencv():
    # A = (0, 0, 10, 10); B = (5.9, 0, 10.9, 10): IoU 0.244 in float, but (5, 0, 10, 10) after truncation: IoU 1/3 > 0.3
    anchors = [
        (8, 0, 0, 1, 0.81, _bbox_for(8, 0, 0, 0, 0, 10, 10)),
        (8, 0, 1, 1, 0.64, _bbox_for(8, 1, 0, 5.9, 0, 10.9, 10)),
    ]
    boxes = _decode(_outputs(64, 32, anchors), nms=0.3)
    assert len(boxes) == 1 and boxes[0].score == pytest.approx(0.9)


def test_nms_runs_highest_score_first_and_keeps_the_top_k():
    anchors = [
        (8, 0, 0, 1, 0.36, (0.5, 0.5, 0, 0)),
        (8, 2, 4, 1, 0.81, (0.5, 0.5, 0, 0)),
        (8, 1, 6, 1, 0.64, (0.5, 0.5, 0, 0)),
    ]
    out = _outputs(64, 32, anchors)
    assert [b.score for b in _decode(out)] == pytest.approx([0.9, 0.8, 0.6])
    assert [b.score for b in _decode(out, top_k=2)] == pytest.approx([0.9, 0.8])


def test_the_top_k_cut_comes_before_nms():
    # A (0.9) and B (0.8) overlap with IoU 0.5; C (0.6) is separate
    anchors = [
        (8, 0, 0, 1, 0.81, _bbox_for(8, 0, 0, 0, 0, 20, 10)),
        (8, 0, 1, 1, 0.64, _bbox_for(8, 1, 0, 0, 0, 10, 10)),
        (8, 2, 4, 1, 0.36, (0.5, 0.5, 0, 0)),
    ]
    out = _outputs(64, 32, anchors)
    assert [b.score for b in _decode(out, top_k=3)] == pytest.approx([0.9, 0.6])  # B is suppressed
    # A and B make the cut and B is suppressed; C was already cut and does not come back
    assert [b.score for b in _decode(out, top_k=2)] == pytest.approx([0.9])


def test_an_equal_score_keeps_the_earlier_anchor():
    # stride 8 before 16 before 32, so the 16 x 16 box wins over the 16 x 12 one
    anchors = [
        (8, 0, 0, 1, 0.64, _bbox_for(8, 0, 0, 0, 0, 16, 16)),
        (16, 0, 0, 1, 0.64, _bbox_for(16, 0, 0, 0, 0, 16, 12)),
    ]
    [box] = _decode(_outputs(64, 32, anchors))
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0, 0, 16, 16), abs=1e-4)


def test_nms_truncates_negative_coordinates_toward_zero():
    # x = -0.9 is 0, so the 10-wide A covers [0, 10) and overlaps B = [8, 18) by 2 of 18 (0.111); floored or rounded to -1 it
    # would be 1 of 19 (0.053)
    anchors = [
        (8, 0, 0, 1, 0.81, _bbox_for(8, 0, 0, -0.9, 0, 10, 10)),
        (8, 0, 1, 1, 0.64, _bbox_for(8, 1, 0, 8, 0, 10, 10)),
    ]
    out = _outputs(64, 32, anchors)
    assert len(_decode(out, nms=0.08)) == 1
    assert len(_decode(out, nms=0.12)) == 2


def test_decode_clips_boxes_to_the_frame_not_the_padded_input():
    out = _outputs(64, 32, [(32, 0, 1, 1, 1, _bbox_for(32, 1, 0, -6, -4, 80, 40))])
    [box] = _decode(out, width=50, height=30)
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0, 0, 50, 30))


@pytest.mark.parametrize("kind, value", [("cls", np.nan), ("obj", np.inf), ("cls", -np.inf)])
def test_a_non_finite_score_anywhere_stops_the_decode(kind, value):
    out = _outputs(64, 32)
    out[f"{kind}_16"][0, 5, 0] = value  # an anchor nowhere near any box
    with pytest.raises(ValueError, match="non-finite scores"):
        _decode(out)


@pytest.mark.parametrize(
    "bbox",
    [
        (np.nan, 0.5, 0, 0), (0.5, 0.5, np.inf, 0), (0.5, 0.5, 0, 1000.0),  # the last one: exp overflows
        (0.5, 0.5, -np.inf, 0), (0.5, 0.5, 0, -np.inf), (-np.inf, 0.5, 0, 0), (0.5, -np.inf, 0, 0),  # -inf size: exp is 0
    ],
)
def test_a_kept_box_with_non_finite_coordinates_stops_the_decode(bbox):
    out = _outputs(64, 32, [(32, 0, 0, 1, 1, bbox)])
    with pytest.raises(ValueError, match="non-finite box"):
        _decode(out)


def test_a_dropped_anchor_may_have_any_coordinates():
    out = _outputs(64, 32, [(32, 0, 0, 0.1, 0.1, (np.nan, np.inf, 1000.0, -np.inf))])
    assert _decode(out) == []


def test_decode_refuses_outputs_of_the_wrong_size():
    out = _outputs(64, 32)
    out["bbox_16"] = np.zeros((1, 7, 4), np.float32)
    with pytest.raises(ValueError, match="bbox_16"):
        _decode(out)


@pytest.mark.parametrize("name", ["cls_8", "obj_16", "bbox_32"])
def test_decode_refuses_outputs_with_a_missing_level_as_a_value_error(name):
    out = _outputs(64, 32)
    del out[name]
    with pytest.raises(ValueError, match=name):
        _decode(out)


def test_yunet_refuses_a_missing_source_before_writing(tmp_path, capsys):
    out = tmp_path / "yunet"
    assert main(["yunet", "--out", str(out), "--source", str(tmp_path / "missing.onnx"), "--frames-root", str(tmp_path / "frames")]) == 2
    assert capsys.readouterr().err.startswith("vc_train: ")
    assert not out.exists()


def test_yunet_refuses_a_model_that_is_not_yunet(tmp_path, capsys):
    out = tmp_path / "yunet"
    other = CPP_DATA / "const_yolo" / "model.onnx"
    assert main(["yunet", "--out", str(out), "--source", str(other), "--frames-root", str(tmp_path / "frames")]) == 2
    assert "not the YuNet face model" in capsys.readouterr().err
    assert not out.exists()


def test_yunet_refuses_a_non_empty_out(tmp_path, capsys):
    out = tmp_path / "yunet"
    out.mkdir()
    (out / "keep.txt").write_text("previous run")
    assert main(["yunet", "--out", str(out), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "frames")]) == 2
    assert "not empty" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_yunet_refuses_an_out_under_the_frames_root(tmp_path, capsys):
    (tmp_path / "frames" / "seg").mkdir(parents=True)
    out = tmp_path / "frames" / "seg" / "yunet"
    assert main(["yunet", "--out", str(out), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "frames")]) == 2
    assert "frames directory" in capsys.readouterr().err
    assert not out.exists()


STATIC = CPP_DATA / "const_yunet_static" / "model.onnx"  # YuNet's inputs and outputs fixed at 64x64, a few KB, committed


def _convert(tmp_path, source, out=None):
    out = tmp_path / "models" / "yunet" if out is None else out
    return main(["yunet", "--out", str(out), "--source", str(source), "--frames-root", str(tmp_path / "frames")])


def _variant(tmp_path, change):
    """The committed static model with `change(model)` applied, written to a file of its own."""
    model = onnx.load(STATIC)
    change(model)
    path = tmp_path / "source" / "variant.onnx"
    path.parent.mkdir(exist_ok=True)
    onnx.save(model, path)
    return path


def _assert_same_weights(converted, source):
    """The initializers are the source's: same names, same values (not the same serialisation)."""
    got = {i.name: numpy_helper.to_array(i) for i in converted.graph.initializer}
    want = {i.name: numpy_helper.to_array(i) for i in source.graph.initializer}
    assert got.keys() == want.keys() and want
    for name, array in want.items():
        assert got[name].dtype == array.dtype and np.array_equal(got[name], array), name


def _hard_code_the_reshape(model):
    """Every Reshape of the const model takes [1, -1, 1]; with [1, 64, 1] the graph depends on the 64x64 input."""
    [shape] = [i for i in model.graph.initializer if i.name == "shape_n1"]
    shape.CopyFrom(numpy_helper.from_array(np.array([1, 64, 1], np.int64), "shape_n1"))


def _add_a_rank_one_output(model):
    model.graph.node.append(onnx.helper.make_node("Identity", ["zero"], ["extra"]))
    model.graph.output.append(onnx.helper.make_tensor_value_info("extra", onnx.TensorProto.FLOAT, [1]))


def test_yunet_converts_the_committed_static_model_without_the_cached_one(tmp_path, capsys):
    out = tmp_path / "models" / "yunet"
    assert _convert(tmp_path, STATIC, out) == 0
    assert str(out / MODEL_ONNX) in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == [MODEL_JSON, MODEL_ONNX]
    assert json.loads((out / MODEL_JSON).read_text())["source_sha256"] == hashlib.sha256(STATIC.read_bytes()).hexdigest()
    model = onnx.load(out / MODEL_ONNX)
    [inp] = model.graph.input
    assert [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim] == [1, 3, "H", "W"]
    assert len(model.graph.output) == 12 and len(model.graph.value_info) == 0
    assert all(o.type.tensor_type.shape.dim[1].dim_param == "N" for o in model.graph.output)
    _assert_same_weights(model, onnx.load(STATIC))
    # it runs at sizes other than 64x64, and the reference decoder finds the cyan cell the const model scores 1
    session = yunet_session(out / MODEL_ONNX)
    bgr = np.zeros((64, 96, 3), np.uint8)
    bgr[:, :, 2] = 255  # red
    bgr[0:32, 64:96] = (255, 255, 0)  # cyan: B = G = 255
    [box] = detect_yunet(session, bgr, 0.3, 0.3)
    assert (box.x1, box.y1, box.x2, box.y2, box.score) == pytest.approx((64, 0, 96, 32, 1.0), abs=1e-4)


def test_yunet_refuses_a_graph_that_depends_on_the_size_it_was_written_for(tmp_path, capsys):
    out = tmp_path / "models" / "yunet"
    out.parent.mkdir()
    assert _convert(tmp_path, _variant(tmp_path, _hard_code_the_reshape), out) == 2
    err = capsys.readouterr().err
    assert err.startswith("vc_train: ") and "96x64" in err
    assert list(out.parent.iterdir()) == []  # neither out nor a staging directory


def test_yunet_refuses_a_graph_onnx_checker_rejects(tmp_path, capsys):
    def break_a_node(model):
        model.graph.node[0].op_type = "NoSuchOperator"

    out = tmp_path / "models" / "yunet"
    out.parent.mkdir()
    assert _convert(tmp_path, _variant(tmp_path, break_a_node), out) == 2
    assert capsys.readouterr().err.startswith("vc_train: ")
    assert list(out.parent.iterdir()) == []


def test_yunet_leaves_an_extra_output_of_any_rank_alone(tmp_path):
    out = tmp_path / "models" / "yunet"
    assert _convert(tmp_path, _variant(tmp_path, _add_a_rank_one_output), out) == 0
    extra = [o for o in onnx.load(out / MODEL_ONNX).graph.output if o.name == "extra"]
    assert [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in extra] == [[1]]


def test_yunet_writes_into_an_empty_out_that_already_exists(tmp_path):
    out = tmp_path / "models" / "yunet"
    out.mkdir(parents=True)
    assert _convert(tmp_path, STATIC, out) == 0
    assert sorted(p.name for p in out.iterdir()) == [MODEL_JSON, MODEL_ONNX]
    assert sorted(p.name for p in out.parent.iterdir()) == ["yunet"]


def test_the_written_directory_has_the_mode_mkdir_would_give_it(tmp_path):
    out = tmp_path / "models" / "yunet"
    old = os.umask(0o022)
    try:
        assert _convert(tmp_path, STATIC, out) == 0
    finally:
        os.umask(old)
    assert stat.S_IMODE(out.stat().st_mode) == 0o755 and stat.S_IMODE((out / MODEL_JSON).stat().st_mode) == 0o644


def test_a_failed_write_of_model_json_leaves_no_half_written_out(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()
    write_text = Path.write_text

    def disk_full(self, *args, **kwargs):
        if self.name == MODEL_JSON:
            raise OSError(28, "No space left on device")
        return write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", disk_full)
    assert _convert(tmp_path, STATIC, models / "yunet") == 2
    assert "No space left on device" in capsys.readouterr().err
    assert list(models.iterdir()) == []  # not a model.onnx without its model.json, and no staging directory


def test_a_failed_move_into_place_leaves_nothing_behind(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()

    def refuse(src, dst):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "replace", refuse)
    assert _convert(tmp_path, STATIC, models / "yunet") == 2
    assert "Operation not permitted" in capsys.readouterr().err
    assert list(models.iterdir()) == []


@needs_yunet
def test_yunet_writes_a_model_with_a_free_input_size_and_its_model_json(tmp_path, capsys):
    out = tmp_path / "models" / "yunet"
    assert main(["yunet", "--out", str(out), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "frames")]) == 0
    assert str(out / MODEL_ONNX) in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == [MODEL_JSON, MODEL_ONNX]
    assert json.loads((out / MODEL_JSON).read_text()) == {
        "name": "yunet-2023mar",
        "classes": ["face"],
        "input": {"color": "BGR", "scale": 1.0, "letterbox": "top_left", "pad_value": 0, "pad_multiple": 32},
        "output_format": "yunet",
        "source_sha256": hashlib.sha256(YUNET_MODEL.read_bytes()).hexdigest(),
    }
    model = onnx.load(out / MODEL_ONNX)
    [inp] = model.graph.input
    assert [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim] == [1, 3, "H", "W"]
    assert all(o.type.tensor_type.shape.dim[1].dim_param == "N" for o in model.graph.output)
    assert len(model.graph.output) == 12 and len(model.graph.value_info) == 0
    _assert_same_weights(model, onnx.load(YUNET_MODEL))


@needs_yunet
@pytest.mark.parametrize("width, height", [(64, 96), (96, 64), (1920, 1088)])
def test_the_written_model_runs_at_any_multiple_of_32(tmp_path, width, height):
    out = tmp_path / "yunet"
    assert main(["yunet", "--out", str(out), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "frames")]) == 0
    session = yunet_session(out / MODEL_ONNX)
    names = [o.name for o in session.get_outputs()]
    results = dict(zip(names, session.run(None, {"input": np.zeros((1, 3, height, width), np.float32)})))
    for s in STRIDES:
        n = (width // s) * (height // s)
        assert [results[f"{k}_{s}"].shape for k in ("cls", "obj", "bbox", "kps")] == [(1, n, 1), (1, n, 1), (1, n, 4), (1, n, 10)]


def _assert_finds_what_cv2_finds(tmp_path, bgr, threshold, at_least):
    """detect_yunet on the converted model and cv2.FaceDetectorYN on the original, boxes and scores in the same order."""
    import cv2

    height, width = bgr.shape[:2]
    assert main(["yunet", "--out", str(tmp_path / "yunet"), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "f")]) == 0
    ours = detect_yunet(yunet_session(tmp_path / "yunet" / MODEL_ONNX), bgr, threshold, 0.3)
    detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (width, height), threshold, 0.3, 5000)
    _, faces = detector.detect(bgr)
    theirs = [] if faces is None else [tuple(float(v) for v in (*f[:2], f[0] + f[2], f[1] + f[3], f[-1])) for f in faces]
    assert len(theirs) >= at_least  # enough boxes for the comparison to mean something
    assert len(ours) == len(theirs)
    for box, (x1, y1, x2, y2, score) in zip(ours, theirs):
        clipped = (min(max(x1, 0), width), min(max(y1, 0), height), min(max(x2, 0), width), min(max(y2, 0), height))
        assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(clipped, abs=0.01)
        assert box.score == pytest.approx(score, abs=1e-4)


@needs_yunet
@pytest.mark.parametrize("threshold", [0.01, 0.3])
def test_the_reference_decoder_finds_what_cv2_face_detector_yn_finds(tmp_path, threshold):
    gen = _generator()
    _assert_finds_what_cv2_finds(tmp_path, gen.synthetic_faces(gen.WIDTH, gen.HEIGHT), threshold, 5 if threshold < 0.1 else 3)


# neither size is a multiple of 32: the input is padded on the right and bottom (1024x576, 608x480), and the padded
# strip must neither shift the anchors nor hide a face
@needs_yunet
@pytest.mark.parametrize("width, height, at_least", [(1000, 562, 4), (600, 450, 2)])
@pytest.mark.parametrize("threshold", [0.01, 0.3])
def test_the_reference_decoder_matches_cv2_on_a_size_that_needs_padding(tmp_path, width, height, at_least, threshold):
    assert pad_size(width, height) != (width, height)
    _assert_finds_what_cv2_finds(tmp_path, _generator().synthetic_faces(width, height), threshold, at_least)


@needs_yunet
def test_the_committed_cpp_expectation_is_what_cv2_finds_on_the_generator_image():
    gen = _generator()
    committed = json.loads((CPP_DATA / "yunet_synthetic_expected.json").read_text())
    fresh = gen.expectation(YUNET_MODEL)
    assert {k: v for k, v in committed.items() if k != "faces"} == {k: v for k, v in fresh.items() if k != "faces"}
    assert len(committed["faces"]) == len(fresh["faces"]) >= 5
    for a, b in zip(committed["faces"], fresh["faces"]):
        assert a == pytest.approx(b, abs=1e-3)
