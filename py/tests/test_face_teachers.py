import numpy as np
import pytest

from conftest import best_match
from vc_label.boxes import iou
from vc_label.contracts import Box
from vc_label.teachers.retinaface import decode, prior_boxes

# Frame 006005 of the sg_drive_600s sample: a man facing the camera on the left.
SAMPLE_FACE = Box("face", 254, 704, 281, 737)


def test_prior_boxes_follow_the_network_output_order():
    priors = prior_boxes(64, 64)
    assert priors.shape == (8 * 8 * 2 + 4 * 4 * 2 + 2 * 2 * 2, 4)
    np.testing.assert_allclose(priors[0], [4 / 64, 4 / 64, 16 / 64, 16 / 64])
    np.testing.assert_allclose(priors[1], [4 / 64, 4 / 64, 32 / 64, 32 / 64])
    np.testing.assert_allclose(priors[2], [12 / 64, 4 / 64, 16 / 64, 16 / 64])


def test_prior_boxes_match_odd_feature_map_sizes():
    # 1080 / 16 = 67.5 -> 68 rows, as the ResNet strides produce.
    assert prior_boxes(1080, 1920).shape == (135 * 240 * 2 + 68 * 120 * 2 + 34 * 60 * 2, 4)


def test_decode_of_zero_deltas_returns_the_prior_boxes():
    priors = np.array([[0.5, 0.5, 0.25, 0.5]], np.float32)
    np.testing.assert_allclose(decode(np.zeros((1, 4), np.float32), priors, 200, 100), [[75, 25, 125, 75]])


@pytest.mark.models
def test_retinaface_finds_the_sample_face(sample_frame, device):
    from vc_label.paths import RETINAFACE_WEIGHTS
    from vc_label.teachers.retinaface import RetinaFaceTeacher

    best = best_match(RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device).detect(sample_frame), SAMPLE_FACE)
    assert best is not None and iou(best, SAMPLE_FACE) >= 0.5 and best.score >= 0.9
    assert best.src == "teacher_retinaface"


@pytest.mark.models
def test_yunet_finds_the_sample_face(sample_frame):
    from vc_label.paths import YUNET_MODEL
    from vc_label.teachers.yunet import YuNetTeacher

    best = best_match(YuNetTeacher(str(YUNET_MODEL)).detect(sample_frame), SAMPLE_FACE)
    assert best is not None and iou(best, SAMPLE_FACE) >= 0.5 and best.score >= 0.8
