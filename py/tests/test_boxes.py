import pytest

from vc_label.boxes import contains_point, fuse, iou, nms, sources
from vc_label.contracts import Box


def test_iou():
    a = Box("face", 0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, Box("face", 20, 20, 30, 30)) == 0.0
    assert iou(a, Box("face", 5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_contains_point():
    assert contains_point(Box("vehicle", 0, 0, 10, 10), 10, 5)
    assert not contains_point(Box("vehicle", 0, 0, 10, 10), 10.1, 5)


def test_nms_keeps_best_box_per_class():
    boxes = [
        Box("vehicle", 0, 0, 100, 100, score=0.6),
        Box("vehicle", 2, 2, 100, 100, score=0.9),
        Box("person", 0, 0, 100, 100, score=0.5),
    ]
    kept = nms(boxes, 0.6)
    assert [(b.cls, b.score) for b in kept] == [("vehicle", 0.9), ("person", 0.5)]


def test_fuse_merges_overlapping_sources_with_score_weighted_coordinates():
    retina = Box("face", 0, 0, 10, 10, score=0.9, src="teacher_retinaface")
    yunet = Box("face", 1, 1, 11, 11, score=0.6, src="teacher_yunet")

    [fused] = fuse([yunet, retina], 0.5)

    assert fused.x1 == pytest.approx((0 * 0.9 + 1 * 0.6) / 1.5)
    assert fused.x2 == pytest.approx((10 * 0.9 + 11 * 0.6) / 1.5)
    assert fused.score == 0.9
    assert fused.src == "teacher_retinaface+teacher_yunet"
    assert sources(fused) == {"teacher_retinaface", "teacher_yunet"}


def test_fuse_keeps_classes_and_distant_boxes_apart():
    boxes = [
        Box("face", 0, 0, 10, 10, score=0.9, src="a"),
        Box("plate", 0, 0, 10, 10, score=0.8, src="b"),
        Box("face", 50, 50, 60, 60, score=0.7, src="a"),
    ]
    fused = fuse(boxes, 0.5)
    assert sorted((b.cls, b.x1) for b in fused) == [("face", 0), ("face", 50), ("plate", 0)]


def test_fuse_of_nothing_is_nothing():
    assert fuse([], 0.5) == []
