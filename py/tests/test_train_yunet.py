"""vc_train yunet: the OpenCV Zoo YuNet face model with a free input size, and the reference decoder vc_redact follows."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import onnx
import pytest

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


@pytest.mark.parametrize("bbox", [(np.nan, 0.5, 0, 0), (0.5, 0.5, np.inf, 0), (0.5, 0.5, 0, 1000.0)])
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
    # the weights are the source's, untouched
    source = onnx.load(YUNET_MODEL)
    assert [i.raw_data for i in model.graph.initializer] == [i.raw_data for i in source.graph.initializer]


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


@needs_yunet
@pytest.mark.parametrize("threshold", [0.01, 0.3])
def test_the_reference_decoder_finds_what_cv2_face_detector_yn_finds(tmp_path, threshold):
    import cv2

    gen = _generator()
    bgr = gen.synthetic_faces(gen.WIDTH, gen.HEIGHT)
    assert main(["yunet", "--out", str(tmp_path / "yunet"), "--source", str(YUNET_MODEL), "--frames-root", str(tmp_path / "f")]) == 0
    ours = detect_yunet(yunet_session(tmp_path / "yunet" / MODEL_ONNX), bgr, threshold, 0.3)
    detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (gen.WIDTH, gen.HEIGHT), threshold, 0.3, 5000)
    _, faces = detector.detect(bgr)
    theirs = [] if faces is None else [tuple(float(v) for v in (*f[:2], f[0] + f[2], f[1] + f[3], f[-1])) for f in faces]
    assert len(theirs) >= (5 if threshold < 0.1 else 3)  # enough boxes for the comparison to mean something
    assert len(ours) == len(theirs)
    for box, (x1, y1, x2, y2, score) in zip(ours, theirs):
        clipped = (min(max(x1, 0), gen.WIDTH), min(max(y1, 0), gen.HEIGHT), min(max(x2, 0), gen.WIDTH), min(max(y2, 0), gen.HEIGHT))
        assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(clipped, abs=0.01)
        assert box.score == pytest.approx(score, abs=1e-4)


@needs_yunet
def test_the_committed_cpp_expectation_is_what_cv2_finds_on_the_generator_image():
    gen = _generator()
    committed = json.loads((CPP_DATA / "yunet_synthetic_expected.json").read_text())
    fresh = gen.expectation(YUNET_MODEL)
    assert {k: v for k, v in committed.items() if k != "faces"} == {k: v for k, v in fresh.items() if k != "faces"}
    assert len(committed["faces"]) == len(fresh["faces"]) >= 5
    for a, b in zip(committed["faces"], fresh["faces"]):
        assert a == pytest.approx(b, abs=1e-3)
