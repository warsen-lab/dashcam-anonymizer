import json

import pytest

from vc_eval.cli import main
from vc_eval.viewing import FLICKER_FRACTION, RASTER_PX, evaluate_viewing, pool_viewing, viewing_json

W, H = 160, 120  # a 40 x 30 raster of 4 px cells: 1200 cells


def _box(x1, y1, x2, y2, track=None, cls="face"):
    b = {"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "origin": "detected"}
    if track is not None:
        b["track"] = track
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


def test_pooling_concatenates_the_frames_and_adds_up_the_pairs(tmp_path):
    a = evaluate_viewing(_masks(tmp_path / "a.jsonl", [[_box(0, 0, 40, 40, 1)], []]), W, H)
    b = evaluate_viewing(_masks(tmp_path / "b.jsonl", [[], [], [_box(0, 0, 4, 4)]], video="b.mp4"), W, H)
    pooled = viewing_json(pool_viewing([a, b]))
    assert pooled["frames"] == 5
    assert pooled["flicker"] == {"pairs": 3, "over": 1, "share": pytest.approx(1 / 3)}  # no pair across a and b
    assert pooled["area"]["max"] == pytest.approx(100 / 1200)
    assert pooled["multi_box_frames"] == 0  # a has track ids and never repeats one; b has none


def test_cli_viewing_prints_the_metrics_and_writes_json(tmp_path, capsys):
    masks = _masks(tmp_path / "m.jsonl", [[_box(0, 0, 40, 40, 1)], [_box(0, 0, 40, 40, 1)]])
    args = ["viewing", "--masks", str(masks), "--width", str(W), "--height", str(H), "--json", str(tmp_path / "v.json")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("viewing a.mp4: area mean 8.33%, p95 8.33%, p99 8.33%, max 8.33%;")
    assert "flicker 0.00% (0/1 pairs)" in out and "boxes median 1, p95 1" in out and "same-track multi-box frames 0" in out
    assert json.loads((tmp_path / "v.json").read_text())["area"]["max"] == pytest.approx(100 / 1200)
    assert main(["viewing", "--masks", str(tmp_path / "missing.jsonl"), "--width", "160", "--height", "120"]) == 2
    assert "vc_eval:" in capsys.readouterr().err
