import math

import pytest

from vc_label.contracts import Box
from vc_eval.metrics import (
    Rate,
    covered_fraction,
    heat_cell,
    must_mask,
    size_bin,
    size_bin_labels,
    stray_mask_fraction,
    wilson_interval,
)


def test_must_mask_scales_with_frame_height():
    assert must_mask(Box("face", 0, 0, 5, 12), 1080)
    assert not must_mask(Box("face", 0, 0, 50, 11.9), 1080)
    assert must_mask(Box("plate", 0, 0, 24, 5), 1080)
    assert not must_mask(Box("plate", 0, 0, 23, 50), 1080)
    assert not must_mask(Box("face", 0, 0, 5, 12), 2160)


def test_size_bins_use_height_for_faces_and_width_for_plates():
    assert size_bin_labels(1080) == ["<12", "12-24", "24-48", "48-96", ">96"]
    assert size_bin(Box("face", 0, 0, 100, 30), 1080) == "24-48"
    assert size_bin(Box("plate", 0, 0, 30, 100), 1080) == "24-48"
    assert size_bin(Box("face", 0, 0, 5, 10), 1080) == "<12"
    assert size_bin(Box("plate", 0, 0, 96, 5), 1080) == ">96"


def test_covered_fraction_uses_the_union_of_masks():
    target = Box("face", 0, 0, 10, 10)
    assert covered_fraction(target, []) == 0.0
    assert covered_fraction(target, [Box("face", -5, -5, 20, 20)]) == 1.0
    # two overlapping halves cover 0..8 in x: 80 %
    assert covered_fraction(target, [Box("face", 0, 0, 5, 10), Box("plate", 3, 0, 8, 10)]) == pytest.approx(0.8)
    assert covered_fraction(target, [Box("face", 20, 20, 30, 30)]) == 0.0


def test_stray_mask_fraction_counts_mask_pixels_outside_targets():
    masks = [Box("face", 0, 0, 10, 10)]
    assert stray_mask_fraction(masks, [], 100, 100) == pytest.approx(0.01)
    assert stray_mask_fraction(masks, [Box("face", 0, 0, 10, 5)], 100, 100) == pytest.approx(0.005)
    assert stray_mask_fraction([], [], 100, 100) == 0.0


def test_heat_cell_places_the_centre_in_a_6_by_4_grid():
    assert heat_cell(Box("face", 0, 0, 10, 10), 1920, 1080) == (0, 0)
    assert heat_cell(Box("face", 1900, 1070, 1920, 1080), 1920, 1080) == (3, 5)
    assert heat_cell(Box("face", 950, 530, 970, 550), 1920, 1080) == (2, 3)


def test_wilson_interval():
    lo, hi = wilson_interval(99, 100)
    assert lo == pytest.approx(0.9455, abs=1e-3) and hi == pytest.approx(0.9983, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == 1.0


def test_rate_accumulates():
    r = Rate(0, 0).add(True).add(False).add(True)
    assert (r.hits, r.total) == (2, 3)
    assert r.value == pytest.approx(2 / 3)
    assert math.isnan(Rate(0, 0).value)
