import numpy as np
import pytest

from vc_train.geometry import PAD_VALUE, decode_yolov8, letterbox, letterbox_scale


def test_letterbox_scale_fits_the_limiting_side():
    assert letterbox_scale(1920, 1080, 1920, 1088) == 1.0
    assert letterbox_scale(3840, 2160, 1920, 1088) == 0.5
    assert letterbox_scale(100, 50, 64, 64) == pytest.approx(0.64)


def test_letterbox_places_image_top_left_in_rgb_and_pads_with_114():
    bgr = np.zeros((50, 100, 3), np.uint8)
    bgr[:, :] = (255, 0, 0)  # pure blue in BGR
    arr, r = letterbox(bgr, 64, 64)
    assert r == pytest.approx(0.64)
    assert arr.shape == (1, 3, 64, 64) and arr.dtype == np.float32
    # resized content is 64 x 32 (round(100*0.64), round(50*0.64)) at the top left
    assert np.allclose(arr[0, 2, :32, :], 1.0)  # blue lands in RGB channel 2
    assert np.allclose(arr[0, 0, :32, :], 0.0)
    assert np.allclose(arr[0, :, 32:, :], PAD_VALUE / 255.0)


def test_letterbox_without_resize_keeps_pixels_and_pads_bottom_rows():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    arr, r = letterbox(bgr, 1920, 1088)
    assert r == 1.0
    assert np.array_equal((arr[0, :, :1080, :] * 255).round().astype(np.uint8), bgr[:, :, ::-1].transpose(2, 0, 1))
    assert np.allclose(arr[0, :, 1080:, :], PAD_VALUE / 255.0)


def _pred(columns):
    """columns: list of (cx, cy, w, h, face_score, plate_score) -> [1, 6, N]."""
    return np.array(columns, np.float32).T[None]


def test_decode_maps_back_by_r_thresholds_and_suppresses_per_class():
    pred = _pred(
        [
            (32, 16, 8, 4, 0.9, 0.0),    # face, kept
            (100, 100, 10, 10, 0.1, 0.05),  # below threshold
            (33, 16, 8, 4, 0.6, 0.0),    # overlaps the first face -> suppressed
            (32, 16, 8, 4, 0.0, 0.5),    # same place, but plate -> kept
        ]
    )
    boxes = decode_yolov8(pred, 0.5, ("face", "plate"), 1000, 1000, 0.15, 0.5)
    faces = [b for b in boxes if b.cls == "face"]
    plates = [b for b in boxes if b.cls == "plate"]
    assert len(faces) == 1 and len(plates) == 1
    f = faces[0]
    assert (f.x1, f.y1, f.x2, f.y2) == pytest.approx((56, 28, 72, 36))
    assert f.score == pytest.approx(0.9) and f.src == "student"


def test_decode_clips_to_the_frame_and_accepts_2d_input():
    pred = _pred([(2, 2, 10, 10, 0.8, 0.0)])[0]
    (b,) = decode_yolov8(pred, 1.0, ("face", "plate"), 20, 20, 0.15, 0.5)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0, 0, 7, 7))


def test_decode_rejects_a_class_count_mismatch():
    with pytest.raises(ValueError, match="classes"):
        decode_yolov8(_pred([(1, 1, 1, 1, 0.9, 0.1)]), 1.0, ("face",), 10, 10, 0.15, 0.5)
