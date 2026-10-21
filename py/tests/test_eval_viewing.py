import json

import pytest

from vc_eval.cli import main
from vc_eval.viewing import FLICKER_FRACTION, RASTER_PX, evaluate_viewing, pool_viewing, viewing_json

W, H = 160, 120  # a 40 x 30 raster of 4 px cells: 1200 cells


def _box(x1, y1, x2, y2, track=None, cls="face", block=None, snap=None):
    b = {"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "origin": "detected"}
    if track is not None:
        b["track"] = track
    if block is not None:
        b["block"] = block
    if snap is not None:
        b["snap"] = snap
    return b


def _masks(path, frames, video="a.mp4"):
    with path.open("w") as f:
        for i, boxes in enumerate(frames):
            f.write(json.dumps({"video": video, "frame": i, "boxes": boxes}) + "\n")
    return path


def test_the_raster_is_four_pixels_and_a_jump_is_one_percent():
    assert RASTER_PX == 4 and FLICKER_FRACTION == 0.01


def test_area_is_the_union_of_the_masks_on_the_raster(tmp_path):
    frames = [
        [_box(0, 0, 40, 40)],  # 10 x 10 cells
        [_box(0, 0, 40, 40), _box(20, 0, 60, 40)],  # overlapping: 15 x 10 cells, not 200
        [],
        [_box(1, 1, 3, 3)],  # rounded out to one whole cell
        [_box(-50, -50, 8, 400)],  # clipped to the picture: 2 columns x 30 rows
    ]
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert r.frames == 5 and r.video == "a.mp4"
    assert r.area.tolist() == pytest.approx([100 / 1200, 150 / 1200, 0.0, 1 / 1200, 60 / 1200])
    assert r.boxes.tolist() == [1, 2, 0, 1, 1]


def _cell_boxes(k):
    """Boxes whose union is exactly k raster cells of the 40 x 30 raster: whole rows of 40 cells, then a partial row."""
    rows, rest = divmod(k, 40)
    return ([_box(0, 0, 160, 4 * rows)] if rows else []) + ([_box(0, 4 * rows, 4 * rest, 4 * rows + 4)] if rest else [])


def test_the_acceptance_statistics_are_pinned_on_frames_of_every_different_area(tmp_path):
    # frame k covers k cells, k = 1..100; the percentiles are numpy's default (linear interpolation between the sorted values)
    r = viewing_json(evaluate_viewing(_masks(tmp_path / "m.jsonl", [_cell_boxes(k) for k in range(1, 101)]), W, H))
    assert r["area"] == {
        "mean": pytest.approx(50.5 / 1200, rel=1e-12),
        "p95": pytest.approx(95.05 / 1200, rel=1e-12),  # the value at sorted index 0.95 * 99 = 94.05
        "p99": pytest.approx(99.01 / 1200, rel=1e-12),  # the value at sorted index 0.99 * 99 = 98.01
        "max": pytest.approx(100 / 1200, rel=1e-12),
    }
    assert r["box_area"] == r["area"]  # no block on any box: the region on screen is the boxes


def test_a_box_covers_every_raster_cell_it_touches_and_nothing_outside_the_picture(tmp_path):
    frames = [
        [_box(2, 2, 10, 10)],  # cells 0..2 on both axes: 9 cells (sampling the cell centres would give 4)
        [_box(200, 0, 240, 40)],  # right of the 160 px picture
        [_box(0, 130, 40, 170)],  # below the 120 px picture
        [_box(-40, -40, -4, -4)],  # left of and above the picture
        [_box(2, 2, 2, 2)],  # no pixel, like vc_redact's covering rectangle of an empty box
        [_box(0, 0, 160, 120)],  # the whole picture
    ]
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert r.area.tolist() == pytest.approx([9 / 1200, 0, 0, 0, 0, 1.0])
    assert r.box_area.tolist() == pytest.approx([9 / 1200, 0, 0, 0, 0, 1.0])
    assert r.boxes.tolist() == [1, 1, 1, 1, 1, 1]  # a box off the picture is still a box of the frame


@pytest.mark.parametrize(
    "box, cells, box_cells",
    [
        # x 111..151 snaps to 96..160, y 21..61 to 16..64 (blocks of 16 from the frame origin): 16 x 12 cells; the box alone 11 x 11
        (_box(111, 21, 151, 61, block=16), 192, 121),
        # x 100..150 snaps up to 168 and is clipped to the 160 px picture
        (_box(100, 10, 150, 30, block=24), 192, 78),
        # fractional edges are rounded outwards to whole pixels before snapping
        (_box(1.5, 1.5, 2.5, 2.5, block=16), 16, 1),
        # a box partly outside the picture is clipped first, then snapped: 0..8 x 0..120 becomes 0..16 x 0..120
        (_box(-50, -50, 8, 400, block=16), 120, 60),
        # a box wholly outside the picture masks nothing, whatever its block
        (_box(200, 0, 240, 40, block=16), 0, 0),
        # no block (an older vc_redact): the box itself
        (_box(111, 21, 151, 61), 121, 121),
        # "snap": true is the default
        (_box(111, 21, 151, 61, block=16, snap=True), 192, 121),
        # "snap": false (a plate): the mosaic stays inside the box
        (_box(111, 21, 151, 61, cls="plate", block=16, snap=False), 121, 121),
    ],
)
def test_the_area_on_screen_is_the_box_snapped_to_the_mosaic_grid_and_box_area_is_the_box(tmp_path, box, cells, box_cells):
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", [[box]]), W, H)
    assert r.area.tolist() == pytest.approx([cells / 1200])
    assert r.box_area.tolist() == pytest.approx([box_cells / 1200])


def test_boxes_with_and_without_a_block_are_unioned_on_screen(tmp_path):
    frames = [[_box(111, 21, 151, 61, block=16), _box(0, 0, 40, 40)]]  # 192 snapped cells and 100 plain ones, apart
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert r.area.tolist() == pytest.approx([(192 + 100) / 1200])  # x 0..40 and 96..160 do not meet
    assert r.box_area.tolist() == pytest.approx([(121 + 100) / 1200])


def test_flicker_is_measured_on_the_region_that_is_pixelated(tmp_path):
    # the box moves 4 px: its plain regions differ on 38 cells (over 1 %), the 16 px grid it snaps to does not change
    moving = [[_box(0, 0, 40, 40, block=16)], [_box(4, 4, 44, 44, block=16)]]
    assert evaluate_viewing(_masks(tmp_path / "a.jsonl", moving), W, H).flicker_pairs == 0
    plain = [[_box(0, 0, 40, 40)], [_box(4, 4, 44, 44)]]
    assert evaluate_viewing(_masks(tmp_path / "b.jsonl", plain), W, H).flicker_pairs == 1


def test_flicker_counts_frame_pairs_whose_masked_regions_differ_on_more_than_one_percent(tmp_path):
    twelve = _box(0, 0, 48, 4)  # 12 cells: exactly 1 % of 1200
    thirteen = _box(0, 0, 52, 4)  # 13 cells
    frames = [[], [twelve], [twelve], [thirteen], [], [thirteen]]
    # pairs: 12 cells differ (not more than 1 %), 0, 1, 13 (over), 13 (over)
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert (r.flicker_pairs, r.pairs) == (2, 5)
    assert viewing_json(r)["flicker"] == {"pairs": 5, "over": 2, "share": 0.4}


def test_area_and_box_statistics(tmp_path):
    frames = [[_box(0, 0, 4, 4)] * k for k in range(100)]  # k boxes on frame k, all on the same cell
    r = viewing_json(evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H))
    assert r["frames"] == 100
    assert r["boxes"] == {"median": 49.5, "p95": pytest.approx(94.05)}
    assert r["area"] == {
        "mean": pytest.approx(0.99 / 1200),
        "p95": pytest.approx(1 / 1200),
        "p99": pytest.approx(1 / 1200),
        "max": pytest.approx(1 / 1200),
    }


def test_frames_where_a_track_has_two_boxes_are_counted(tmp_path):
    frames = [
        [_box(0, 0, 8, 8, 1), _box(20, 0, 28, 8, 2)],
        [_box(0, 0, 8, 8, 1), _box(30, 0, 38, 8, 1)],
        [_box(0, 0, 8, 8, 1)],
    ]
    assert evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H).multi_box_frames == 1
    untracked = [[_box(0, 0, 8, 8)], [_box(0, 0, 8, 8), _box(30, 0, 38, 8)]]  # an older vc_redact: no track ids
    assert evaluate_viewing(_masks(tmp_path / "old.jsonl", untracked), W, H).multi_box_frames is None


@pytest.mark.parametrize(
    "lines, message",
    [
        ([{"video": "a.mp4", "frame": 1, "boxes": []}], "frame 1 where frame 0 was expected"),
        ([{"video": "a.mp4", "frame": 0, "boxes": []}, {"video": "a.mp4", "frame": 2, "boxes": []}], "frame 2 where frame 1"),
        ([{"video": "a.mp4", "frame": 0, "boxes": []}, {"video": "b.mp4", "frame": 1, "boxes": []}], "'b.mp4'"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [{"x1": 0}]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [{**_box(0, 0, 4, 4), "track": "x"}]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, float("inf"), 4)]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": float("inf"), "boxes": []}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, track=float("inf"))]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, block=float("inf"))]}], "line 1: bad mask record"),
        ([{"video": 5, "frame": 0, "boxes": []}], "line 1: bad mask record"),
        ([{"video": None, "frame": 0, "boxes": []}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, block=0)]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, block=-16)]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, block="x")]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, 0, 4, 4, block=16, snap="no")]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [_box(0, float("nan"), 4, 4)]}], "line 1: bad mask record"),
        ([], "has no frames"),
    ],
)
def test_refuses_masks_that_are_not_one_ordered_line_per_frame_of_one_video(tmp_path, lines, message):
    path = tmp_path / "m.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match=message):
        evaluate_viewing(path, W, H)


def test_refuses_a_picture_without_pixels(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_viewing(_masks(tmp_path / "m.jsonl", [[]]), 0, H)


@pytest.mark.parametrize("box", [_box(-1, 0, 40, 40), _box(0, -1, 40, 40), _box(0, 0, 161, 40), _box(0, 0, 40, 121)])
def test_a_box_beyond_the_declared_picture_is_refused_on_request(tmp_path, box):
    path = _masks(tmp_path / "m.jsonl", [[_box(0, 0, 4, 4)], [box]])
    assert evaluate_viewing(path, W, H).frames == 2  # measured by default: clipped to the picture
    with pytest.raises(ValueError, match=r"line 2: .*beyond the 160x120 picture"):
        evaluate_viewing(path, W, H, require_inside=True)


def test_pooling_concatenates_the_frames_and_adds_up_the_pairs(tmp_path):
    a = evaluate_viewing(_masks(tmp_path / "a.jsonl", [[_box(0, 0, 40, 40, 1)], []]), W, H)
    b = evaluate_viewing(_masks(tmp_path / "b.jsonl", [[], [], [_box(0, 0, 4, 4)]], video="b.mp4"), W, H)
    pooled = viewing_json(pool_viewing([a, b]))
    assert pooled["frames"] == 5
    assert pooled["flicker"] == {"pairs": 3, "over": 1, "share": pytest.approx(1 / 3)}  # no pair across a and b
    assert pooled["area"]["max"] == pytest.approx(100 / 1200)
    assert pooled["box_area"]["max"] == pytest.approx(100 / 1200)
    assert pooled["multi_box_frames"] == 0  # a has track ids and never repeats one; b has none


def test_cli_viewing_prints_the_metrics_and_writes_json(tmp_path, capsys):
    masks = _masks(tmp_path / "m.jsonl", [[_box(0, 0, 40, 40, 1)], [_box(0, 0, 40, 40, 1)]])
    args = ["viewing", "--masks", str(masks), "--width", str(W), "--height", str(H), "--json", str(tmp_path / "v.json")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("viewing a.mp4: area mean 8.33%, p95 8.33%, p99 8.33%, max 8.33%;")
    assert "flicker 0.00% (0/1 pairs)" in out and "boxes median 1, p95 1" in out and "same-track multi-box frames 0" in out
    assert out.rstrip().endswith("box-union area mean 8.33%, p95 8.33%, p99 8.33%, max 8.33%")
    assert json.loads((tmp_path / "v.json").read_text())["area"]["max"] == pytest.approx(100 / 1200)
    assert main(["viewing", "--masks", str(tmp_path / "missing.jsonl"), "--width", "160", "--height", "120"]) == 2
    assert "vc_eval:" in capsys.readouterr().err


def test_cli_viewing_refuses_boxes_beyond_the_given_size_but_takes_boxes_up_to_its_edge(tmp_path, capsys):
    edge = _masks(tmp_path / "edge.jsonl", [[_box(120, 80, 160, 120, 1)]])
    assert main(["viewing", "--masks", str(edge), "--width", str(W), "--height", str(H)]) == 0
    capsys.readouterr()
    # vc_redact clips its boxes to the frame, so a box beyond the picture means the size given is too small
    over = _masks(tmp_path / "over.jsonl", [[_box(0, 0, 40, 40, 1)], [_box(100, 0, 170, 40, 1)]])
    out_json = tmp_path / "v.json"
    assert main(["viewing", "--masks", str(over), "--width", str(W), "--height", str(H), "--json", str(out_json)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vc_eval:") and "line 2" in err and "160x120" in err and "real size" in err
    assert not out_json.exists()
