import cv2
import numpy as np

from vc_gt.config import load_config
from vc_gt.crops import (append_crop_index, crop_window, pick_frames, read_crop_index, sharpness, unmasked_frames,
                         write_crop, write_object_crops)
from vc_gt.tracks import Track
from vc_label.contracts import Box


def track(widths, frames=None):
    frames = tuple(frames or range(0, 6 * len(widths), 6))
    return Track(0, "plate", frames, tuple(Box("plate", 100, 100, 100 + w, 110) for w in widths))


def test_crop_window_is_a_square_around_the_box_inside_the_frame():
    assert crop_window(Box("plate", 100, 100, 140, 112), 1920, 1080, 3.0, 96) == (60, 46, 180, 166)
    assert crop_window(Box("plate", 0, 0, 10, 10), 1920, 1080, 3.0, 96) == (0, 0, 53, 53)  # min_px, clipped


def test_write_crop_gives_the_box_in_crop_pixels(tmp_path):
    crop = write_crop(np.zeros((1080, 1920, 3), np.uint8), Box("plate", 100, 100, 140, 112), tmp_path / "a.jpg", 3.0, 96)
    assert crop.box == (40, 54, 80, 66) and cv2.imread(str(crop.path)).shape[:2] == (120, 120)


def test_sharpness_ranks_detail_over_flat():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
    flat = np.full((50, 50, 3), 90, np.uint8)
    assert sharpness(noisy, Box("plate", 0, 0, 50, 50)) > sharpness(flat, Box("plate", 0, 0, 50, 50))


def test_pick_frames_takes_largest_sharpest_and_middle_in_time_order():
    t = track([20, 30, 50, 40, 10])  # frames 0, 6, 12, 18, 24
    assert pick_frames(t, {24: 9.0}, 3) == [12, 18, 24]  # largest 12 (also the middle), sharpest 24, next largest 18
    assert pick_frames(track([20]), {}, 3) == [0]


def test_unmasked_frames_are_those_under_80_percent_cover():
    t = track([40, 40, 40])
    masks = {0: [Box("plate", 90, 95, 150, 115)], 6: [Box("plate", 100, 100, 120, 110)]}  # 100 %, 50 %, none
    assert unmasked_frames(t, masks) == [6, 12]


def test_object_crops_and_the_crop_index_round_trip(tmp_path):
    images = {}
    for f in (0, 6, 12):
        images[f] = tmp_path / f"{f:06d}.jpg"
        cv2.imwrite(str(images[f]), np.zeros((300, 400, 3), np.uint8))
    crops = write_object_crops(track([20, 30, 40]), images, tmp_path / "crops", load_config())
    assert [c.path.name for c in crops] == ["o0_000000.jpg", "o0_000006.jpg", "o0_000012.jpg"]
    append_crop_index(tmp_path / "crops.jsonl", "o0", crops)
    assert read_crop_index(tmp_path / "crops.jsonl") == {"o0": tuple(crops)}
