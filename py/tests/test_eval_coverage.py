import csv
import json
import os

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, Detection, write_jsonl
from vc_label.rules import TEMPORAL_SOURCE
from vc_eval.coverage import CoverageReport, _empty_class, evaluate_coverage, parse_recall_targets, pool
from vc_eval.metrics import Rate, heat_cell
from vc_eval.records import detections_by_frame, read_excluded, read_masks, read_uncertain
from vc_eval.temporal import evaluate_temporal

W, H = 1920, 1080


def _frames(root, stem, frames):
    d = root / "frames" / stem
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for fr in frames:
            w.writerow([f"{stem}.mp4", fr, fr * 16.7, "base", 0.0])
            cv2.imwrite(str(d / f"{fr:06d}.jpg"), np.zeros((H, W, 3), np.uint8))
    return d


FLAG_HEADER = ["video", "frame", "rule", "cls", "x1", "y1", "x2", "y2", "score", "detail"]


def _write_flags(labels, stem, flags):
    """flags.csv as vc_label fuse writes it; `flags` lists (frame, rule, Box), an empty list leaves just the header.

    The coordinates are written as they are (fuse rounds them to 0.1) so a test can put a flagged box a few
    hundredths of a pixel away from its target.
    """
    with (labels / "flags.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FLAG_HEADER)
        for frame, rule, b in flags:
            w.writerow([f"{stem}.mp4", frame, rule, b.cls, b.x1, b.y1, b.x2, b.y2, b.score, ""])


def _labels(root, stem, boxes_by_frame, flags=()):
    d = root / "labels" / stem
    d.mkdir(parents=True)
    # processed.txt first, then merged.jsonl and flags.csv, so neither is ever older than processed (the fuse order)
    (d / "processed.txt").write_text("".join(f"{fr}\n" for fr in boxes_by_frame))
    write_jsonl(d / "merged.jsonl", [Detection(f"{stem}.mp4", fr, fr * 16.7, b) for fr, bs in boxes_by_frame.items() for b in bs])
    _write_flags(d, stem, flags)
    return d


def _dense_stats(dense, frames, every=True):
    """The stats.json vc_redact writes next to detections.jsonl; a --detect-every-frame run says so and counts its frames."""
    stats = dense.parent / "stats.json"
    stats.write_text(json.dumps({"frames": frames, "detect_every_frame": every}))
    return stats


def _cell_targets():
    """30 must-mask faces packed into heatmap cell (row 0, col 0) of a 1920x1080 frame: 6 columns x 5 rows."""
    return [Box("face", 10 + 50 * c, 10 + 50 * r, 50 + 50 * c, 50 + 50 * r, src="teacher_retinaface") for r in range(5) for c in range(6)]


def _masks(path, video, masks_by_frame, frames):
    with path.open("w") as f:
        for fr in frames:
            boxes = [{"cls": b.cls, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "origin": "detected"} for b in masks_by_frame.get(fr, [])]
            f.write(json.dumps({"video": video, "frame": fr, "boxes": boxes}) + "\n")
    return path


# A parked bus's tail light the teachers took for a plate: 60 x 20 px, must-mask at 1080 p.
LIGHT = Box("plate", 1800, 1000, 1860, 1020, src="teacher_owlv2")


def _excluded(labels, rows):
    with (labels / "excluded_refs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cls", "x1", "y1", "x2", "y2", "reason"])
        w.writerows(rows)


def test_read_masks_returns_boxes_by_frame(tmp_path):
    p = _masks(tmp_path / "m.jsonl", "a.mp4", {1: [Box("face", 1, 2, 3, 4)]}, [0, 1])
    videos, masks = read_masks(p)
    assert videos == {"a.mp4"}
    assert masks[0] == [] and masks[1][0].x2 == 3 and masks[1][0].src == "detected"
    (tmp_path / "bad.jsonl").write_text('{"video": "a.mp4", "frame": 0}\n')
    with pytest.raises(ValueError, match="line 1"):
        read_masks(tmp_path / "bad.jsonl")


def test_read_masks_refuses_a_frame_that_appears_twice(tmp_path):
    p = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0, 1, 0])
    with pytest.raises(ValueError, match="line 3: frame 0 appears twice"):
        read_masks(p)


def test_detections_by_frame_refuses_a_missing_file_and_accepts_an_empty_one(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        detections_by_frame(tmp_path / "missing.jsonl")
    (tmp_path / "empty.jsonl").write_text("")
    assert detections_by_frame(tmp_path / "empty.jsonl") == (set(), {})


def test_detections_by_frame_names_the_file_and_line_of_a_bad_record(tmp_path):
    p = tmp_path / "dense.jsonl"
    write_jsonl(p, [Detection("a.mp4", 0, 0.0, Box("face", 1, 2, 3, 4, score=0.9, src="student"))])
    with p.open("a") as f:
        f.write('{"video": "a.mp4"}\n')  # line 2 has no cls/box fields
    with pytest.raises(ValueError, match=r"dense\.jsonl line 2"):
        detections_by_frame(p)


def test_coverage_counts_must_mask_targets_bins_heatmap_and_misses(tmp_path):
    frames = _frames(tmp_path, "a", [0, 10])
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")  # 40 px high, must-mask, bin 24-48
    plate = Box("plate", 1800, 1000, 1830, 1010, src="teacher_owlv2")  # 30 px wide, must-mask
    tiny = Box("face", 500, 500, 505, 508, src="teacher_yunet")  # 8 px high, best effort
    filled = Box("face", 900, 500, 920, 530, src="temporal_fill")
    labels = _labels(tmp_path, "a", {0: [face, plate, tiny], 10: [face, filled]})
    masks = _masks(
        tmp_path / "m.jsonl",
        "a.mp4",
        {0: [Box("face", 95, 95, 135, 145), Box("face", 0, 0, 100, 100)], 10: [Box("face", 95, 95, 135, 145)]},
        [0, 5, 10],
    )
    r = evaluate_coverage(frames, labels, masks)
    assert (r.video, r.frames, r.width, r.height) == ("a.mp4", 2, W, H)
    faces = r.classes["face"]
    assert (faces.rate.hits, faces.rate.total) == (2, 3)  # filled face is missed
    assert (faces.temporal_fill.hits, faces.temporal_fill.total) == (0, 1)
    assert (faces.best_effort.hits, faces.best_effort.total) == (0, 1)
    assert faces.bins["24-48"].total == 3
    assert faces.heatmap[0][0].total == 2
    plates = r.classes["plate"]
    assert (plates.rate.hits, plates.rate.total) == (0, 1)
    assert [(m.frame, m.box.cls) for m in r.misses] == [(0, "plate"), (10, "face")]
    assert not r.class_passed("face") and not r.passed()
    # frame 0 stray: the 100x100 corner mask (10000 px) plus the 775 px ring of the 40x50 face mask outside the
    # face (2000 px mask - 1200 px face - 25 px overlap with the corner mask, which is counted once, in the corner)
    assert r.stray_max == pytest.approx(10775 / (W * H))
    # frame 10 stray: the 40x50 face mask (2000 px) minus the 30x40 face (1200 px) = 800 px. Per-frame strays are
    # [10775, 800] px, so the mean is 5787.5 px and numpy's default linear p95 is 800 + 0.95 * (10775 - 800).
    assert r.stray_mean == pytest.approx(5787.5 / (W * H))
    assert r.stray_p95 == pytest.approx((800 + 0.95 * (10775 - 800)) / (W * H))


def test_coverage_passes_when_every_target_is_masked(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140), Box("plate", 300, 300, 340, 310)]})
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 90, 90, 140, 150), Box("plate", 290, 295, 350, 315)]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    assert r.class_passed("face") and r.class_passed("plate") and r.heatmap_passed() and r.passed()


def test_coverage_rejects_masks_of_another_video_or_missing_frames(tmp_path):
    frames = _frames(tmp_path, "a", [0, 10])
    labels = _labels(tmp_path, "a", {0: [], 10: []})
    other = _masks(tmp_path / "other.jsonl", "b.mp4", {}, [0, 10])
    with pytest.raises(ValueError, match="b.mp4"):
        evaluate_coverage(frames, labels, other)
    short = _masks(tmp_path / "short.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="frame 10"):
        evaluate_coverage(frames, labels, short)


def test_coverage_refuses_a_duplicate_mask_frame(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: []})
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0, 0])
    with pytest.raises(ValueError, match="frame 0 appears twice"):
        evaluate_coverage(frames, labels, masks)


def test_coverage_refuses_a_missing_merged_file(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    (labels / "merged.jsonl").unlink()
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="vc_label fuse"):
        evaluate_coverage(frames, labels, masks)


def test_coverage_refuses_a_merged_file_older_than_processed(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    processed_mtime = (labels / "processed.txt").stat().st_mtime
    os.utime(labels / "merged.jsonl", (processed_mtime - 100, processed_mtime - 100))
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="older than processed.txt"):
        evaluate_coverage(frames, labels, masks)


def test_coverage_refuses_merged_labels_of_another_video(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: []})
    write_jsonl(labels / "merged.jsonl", [Detection("b.mp4", 0, 0.0, Box("face", 100, 100, 130, 140))])
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="b.mp4"):
        evaluate_coverage(frames, labels, masks)


@pytest.mark.parametrize(("covered", "passed"), [(29, False), (30, True)])
def test_heatmap_cell_needs_97_percent_of_its_30_targets(tmp_path, covered, passed):
    frames = _frames(tmp_path, "a", [0])
    targets = _cell_targets()
    labels = _labels(tmp_path, "a", {0: targets})
    masks = _masks(
        tmp_path / "m.jsonl",
        "a.mp4",
        {0: [Box("face", t.x1 - 2, t.y1 - 2, t.x2 + 2, t.y2 + 2) for t in targets[:covered]]},
        [0],
    )
    r = evaluate_coverage(frames, labels, masks)
    assert r.heatmap_passed() is passed
    cls, row, col, rate = r.weakest_cell()
    assert (cls, row, col, rate.hits, rate.total) == ("face", 0, 0, covered, 30)


def test_class_with_no_must_mask_targets_does_not_pass(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140), Box("plate", 300, 300, 310, 305)]})  # 10 px plate: best effort
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 90, 90, 140, 150)]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    assert r.class_passed("face")
    assert r.classes["plate"].rate.total == 0 and not r.class_passed("plate")


def test_covered_temporal_fill_target_counts_as_a_hit(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    filled = Box("face", 100, 100, 130, 140, src="temporal_fill")
    labels = _labels(tmp_path, "a", {0: [filled]})
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 95, 95, 135, 145)]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    assert (r.classes["face"].temporal_fill.hits, r.classes["face"].temporal_fill.total) == (1, 1)
    assert (r.classes["face"].rate.hits, r.classes["face"].rate.total) == (1, 1)


def test_temporal_coverage_uses_dense_detections_as_reference(tmp_path):
    dense = tmp_path / "dense.jsonl"
    write_jsonl(
        dense,
        [
            Detection("a.mp4", 0, 0.0, Box("face", 100, 100, 120, 130, score=0.9, src="student")),
            Detection("a.mp4", 1, 16.7, Box("face", 102, 100, 122, 130, score=0.9, src="student")),
            Detection("a.mp4", 1, 16.7, Box("plate", 500, 500, 540, 510, score=0.5, src="student")),
            Detection("a.mp4", 1, 16.7, Box("face", 800, 800, 803, 803, score=0.5, src="student")),  # too small
        ],
    )
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)], 1: [Box("face", 95, 95, 125, 135)]}, [0, 1, 2])
    _dense_stats(dense, 3)
    r = evaluate_temporal(dense, sparse, 1080)
    assert r.video == "a.mp4" and r.frames == 3
    assert (r.classes["face"].hits, r.classes["face"].total) == (2, 2)
    assert (r.classes["plate"].hits, r.classes["plate"].total) == (0, 1)
    assert [(m.frame, m.box.cls) for m in r.misses] == [(1, "plate")]


def test_temporal_refuses_a_missing_dense_file(tmp_path):
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="does not exist"):
        evaluate_temporal(tmp_path / "dense.jsonl", sparse, 1080)


def test_temporal_refuses_masks_of_another_video(tmp_path):
    dense = tmp_path / "dense.jsonl"
    write_jsonl(dense, [Detection("a.mp4", 0, 0.0, Box("face", 100, 100, 120, 130, score=0.9, src="student"))])
    sparse = _masks(tmp_path / "sparse.jsonl", "b.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    _dense_stats(dense, 1)
    with pytest.raises(ValueError, match="different videos"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_refuses_a_dense_frame_without_a_masks_line(tmp_path):
    dense = tmp_path / "dense.jsonl"
    write_jsonl(dense, [Detection("a.mp4", 5, 83.5, Box("face", 100, 100, 120, 130, score=0.9, src="student"))])
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    _dense_stats(dense, 6)
    with pytest.raises(ValueError, match="frame 5"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_takes_the_video_from_masks_when_dense_is_empty(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {}, [0, 1])
    _dense_stats(dense, 2)
    r = evaluate_temporal(dense, sparse, 1080)
    assert (r.video, r.frames, r.misses) == ("a.mp4", 2, [])


def test_temporal_refuses_an_empty_dense_file_with_masks_of_several_videos(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = tmp_path / "sparse.jsonl"
    sparse.write_text(
        json.dumps({"video": "a.mp4", "frame": 0, "boxes": []}) + "\n" + json.dumps({"video": "b.mp4", "frame": 1, "boxes": []}) + "\n"
    )
    _dense_stats(dense, 2)
    with pytest.raises(ValueError, match="a.mp4, b.mp4"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_refuses_an_empty_masks_file(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = tmp_path / "sparse.jsonl"
    sparse.write_text("")
    _dense_stats(dense, 1)
    with pytest.raises(ValueError, match="no frames"):
        evaluate_temporal(dense, sparse, 1080)


def _dense_with_one_face(tmp_path):
    dense = tmp_path / "dense" / "detections.jsonl"
    dense.parent.mkdir()
    write_jsonl(dense, [Detection("a.mp4", 0, 0.0, Box("face", 100, 100, 120, 130, score=0.9, src="student"))])
    return dense


def test_temporal_refuses_a_dense_run_without_stats_json(tmp_path):
    dense = _dense_with_one_face(tmp_path)
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    with pytest.raises(ValueError, match=r"stats\.json does not exist"):
        evaluate_temporal(dense, sparse, 1080)


@pytest.mark.parametrize(
    "stats, message",
    [
        ({"frames": 1, "detect_every_frame": False}, "detect_every_frame"),  # a sparse run is no per-frame reference
        ({"frames": 1}, "detect_every_frame"),
        ({"frames": 1, "detect_every_frame": 1}, "detect_every_frame"),  # truthy is not true
        ({"detect_every_frame": True}, "frames"),
        ({"frames": 0, "detect_every_frame": True}, "frames"),
        ({"frames": "1", "detect_every_frame": True}, "frames"),
        ({"frames": True, "detect_every_frame": True}, "frames"),
        ([1], "detect_every_frame"),
    ],
)
def test_temporal_refuses_a_dense_run_that_is_not_a_complete_every_frame_run(tmp_path, stats, message):
    dense = _dense_with_one_face(tmp_path)
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    (dense.parent / "stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match=message) as info:
        evaluate_temporal(dense, sparse, 1080)
    assert "stats.json" in str(info.value)


def test_temporal_refuses_a_stats_json_that_is_not_json(tmp_path):
    dense = _dense_with_one_face(tmp_path)
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    (dense.parent / "stats.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"stats\.json"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_needs_masks_for_every_frame_of_the_dense_run_not_just_the_detected_ones(tmp_path):
    dense = _dense_with_one_face(tmp_path)  # a detection in frame 0 only; the run had 5 frames
    _dense_stats(dense, 5)
    short = _masks(tmp_path / "short.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0, 1, 2])
    with pytest.raises(ValueError, match=r"frames 3, 4"):
        evaluate_temporal(dense, short, 1080)
    gap = _masks(tmp_path / "gap.jsonl", "a.mp4", {}, [0, 1, 3, 4])
    with pytest.raises(ValueError, match=r"frames 2"):
        evaluate_temporal(dense, gap, 1080)


def test_temporal_refuses_masks_of_a_longer_video_than_the_dense_run(tmp_path):
    dense = _dense_with_one_face(tmp_path)
    _dense_stats(dense, 2)
    long = _masks(tmp_path / "long.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0, 1, 2, 3])
    with pytest.raises(ValueError, match=r"beyond.*frames 2, 3"):
        evaluate_temporal(dense, long, 1080)


def test_temporal_accepts_masks_covering_exactly_the_frames_of_the_dense_run(tmp_path):
    dense = _dense_with_one_face(tmp_path)
    _dense_stats(dense, 4)
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0, 1, 2, 3])
    r = evaluate_temporal(dense, sparse, 1080)
    assert r.frames == 4 and (r.classes["face"].hits, r.classes["face"].total) == (1, 1)


def test_temporal_leaves_out_dense_boxes_vc_redact_never_masked(tmp_path):
    """A box the dense run marked "confirmed": false (a low-score box seen once) is no reference; the others are."""
    dense = tmp_path / "dense" / "detections.jsonl"
    dense.parent.mkdir()
    box = Box("face", 100, 100, 140, 150, score=0.3, src="student")
    records = [
        {**Detection("a.mp4", 0, 0.0, box).to_record(), "confirmed": False},
        {**Detection("a.mp4", 1, 16.7, box).to_record(), "confirmed": True},
    ]
    dense.write_text("".join(json.dumps(r) + "\n" for r in records))
    _dense_stats(dense, 2)
    sparse = _masks(tmp_path / "s.jsonl", "a.mp4", {}, [0, 1])
    report = evaluate_temporal(dense, sparse, H)
    assert (report.classes["face"].hits, report.classes["face"].total) == (0, 1)
    assert [m.frame for m in report.misses] == [1]
    assert set(detections_by_frame(dense)[1]) == {0, 1}  # without confirmed_only every box is read
    assert set(detections_by_frame(dense, confirmed_only=True)[1]) == {1}


def test_coverage_refuses_a_missing_processed_file(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    (labels / "processed.txt").unlink()
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="processed.txt does not exist"):
        evaluate_coverage(frames, labels, masks)


def _labels_dir(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    (d / "processed.txt").write_text("0\n")
    return d


def _shape(boxes):
    return [(b.cls, b.x1, b.y1, b.x2, b.y2) for b in boxes]


def test_read_uncertain_keeps_low_score_and_single_source_boxes_once(tmp_path):
    labels = _labels_dir(tmp_path)
    face = Box("face", 100, 100, 130, 140, score=0.4)
    plate = Box("plate", 300, 300, 340, 310, score=0.3)
    other = Box("face", 500, 500, 530, 540, score=0.6)
    flags = [
        (3, "low_score", face),
        (3, "face_single_source", face),  # the same box under both rules: once
        (3, "low_score", plate),
        (7, "face_single_source", other),
        (3, "temporal_fill", Box("face", 700, 100, 730, 140)),  # other rules are not uncertainty
        (3, "person_without_face", Box("person", 10, 10, 90, 400)),
        (7, "vehicle_without_plate", Box("vehicle", 10, 10, 900, 400)),
    ]
    _write_flags(labels, "a", flags)
    got = read_uncertain(labels)
    assert sorted(got) == [3, 7]
    assert _shape(got[3]) == [("face", 100, 100, 130, 140), ("plate", 300, 300, 340, 310)]
    assert _shape(got[7]) == [("face", 500, 500, 530, 540)]
    assert got[3][1].score == 0.3


def test_read_uncertain_of_a_header_only_file_is_empty(tmp_path):
    labels = _labels_dir(tmp_path)
    _write_flags(labels, "a", [])
    assert read_uncertain(labels) == {}


def test_read_uncertain_refuses_a_missing_flags_file(tmp_path):
    labels = _labels_dir(tmp_path)
    with pytest.raises(ValueError, match=r"flags\.csv does not exist; run vc_label fuse"):
        read_uncertain(labels)


def test_read_uncertain_refuses_a_flags_file_older_than_processed(tmp_path):
    labels = _labels_dir(tmp_path)
    _write_flags(labels, "a", [])
    processed_mtime = (labels / "processed.txt").stat().st_mtime
    os.utime(labels / "flags.csv", (processed_mtime - 100, processed_mtime - 100))
    with pytest.raises(ValueError, match=r"flags\.csv is older than processed\.txt; rerun vc_label fuse"):
        read_uncertain(labels)


def test_read_uncertain_names_the_file_and_line_of_a_bad_row_or_header(tmp_path):
    labels = _labels_dir(tmp_path)
    _write_flags(labels, "a", [])
    with (labels / "flags.csv").open("a") as f:
        f.write("a.mp4,3,low_score,face,1,2,3,4,0.5,\n")
        f.write("a.mp4,x,low_score,face,1,2,3,4,0.5,\n")  # line 3: the frame is not a number
    with pytest.raises(ValueError, match=r"flags\.csv line 3"):
        read_uncertain(labels)
    (labels / "flags.csv").write_text("video,frame\n")
    with pytest.raises(ValueError, match=r"flags\.csv.*rule"):
        read_uncertain(labels)


def test_a_target_is_uncertain_when_a_flagged_box_of_its_class_matches_every_coordinate_within_0_05_px(tmp_path):
    frames = _frames(tmp_path, "a", [0, 10])
    exact = Box("face", 100, 100, 130, 140, src="teacher_retinaface")  # flagged with the same coordinates
    near = Box("face", 300, 100, 330, 140, src="teacher_retinaface")  # flagged 0.04 px off on every coordinate
    off = Box("face", 500, 100, 530, 140, src="teacher_retinaface")  # flagged 0.06 px off on one coordinate: not it
    plate = Box("plate", 700, 100, 760, 120, src="teacher_owlv2")  # flagged, but as a face
    elsewhere = Box("face", 900, 100, 930, 140, src="teacher_retinaface")  # flagged in frame 10, which has no targets
    flags = [
        (0, "low_score", exact),
        (0, "face_single_source", Box("face", 300.04, 100.04, 329.96, 139.96)),
        (0, "low_score", Box("face", 500.06, 100, 530, 140)),
        (0, "low_score", Box("face", 700, 100, 760, 120)),
        (10, "low_score", elsewhere),
    ]
    labels = _labels(tmp_path, "a", {0: [exact, near, off, plate, elsewhere], 10: []}, flags)
    covering = [Box(t.cls, t.x1 - 5, t.y1 - 5, t.x2 + 5, t.y2 + 5) for t in (exact, off, elsewhere)]
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: covering}, [0, 10])
    r = evaluate_coverage(frames, labels, masks)
    faces, plates = r.classes["face"], r.classes["plate"]
    assert (faces.uncertain.hits, faces.uncertain.total) == (1, 2)  # exact is covered, near is not
    assert (faces.rate.hits, faces.rate.total) == (2, 2)  # off and elsewhere, both covered
    assert (plates.uncertain.hits, plates.uncertain.total) == (0, 0)
    assert (plates.rate.hits, plates.rate.total) == (0, 1)  # a face flag does not make a plate uncertain
    assert [(m.frame, m.box.cls) for m in r.misses] == [(0, "plate")]
    assert [(m.frame, m.box) for m in r.uncertain_misses] == [(0, near)]
    assert r.uncertain_misses[0].covered == 0.0


def test_bins_heatmap_and_temporal_fill_count_the_confident_tier_only(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    confident = Box("face", 100, 100, 130, 140, src="teacher_retinaface")  # 40 px high: bin 24-48, cell (0, 0)
    flagged = Box("face", 1000, 700, 1030, 740, src="teacher_retinaface")  # bin 24-48, cell (2, 3)
    filled = Box("face", 1500, 100, 1530, 140, src="temporal_fill")
    labels = _labels(tmp_path, "a", {0: [confident, flagged, filled]}, [(0, "low_score", flagged)])
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 95, 95, 135, 145)]}, [0])
    faces = evaluate_coverage(frames, labels, masks).classes["face"]
    assert (faces.rate.hits, faces.rate.total) == (1, 2)
    assert (faces.uncertain.hits, faces.uncertain.total) == (0, 1)
    assert (faces.bins["24-48"].hits, faces.bins["24-48"].total) == (1, 2)
    assert faces.heatmap[0][0] == Rate(1, 1) and faces.heatmap[2][3] == Rate() and faces.heatmap[0][4] == Rate(0, 1)
    assert (faces.temporal_fill.hits, faces.temporal_fill.total) == (0, 1)


def test_best_effort_counts_small_targets_of_both_tiers_and_uncertain_only_must_mask_ones(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    small = Box("face", 500, 500, 505, 508, src="teacher_yunet")  # 8 px high: best effort
    small_flagged = Box("face", 600, 500, 605, 508, src="teacher_yunet")
    big_flagged = Box("face", 100, 100, 130, 140, src="teacher_yunet")
    flags = [(0, "face_single_source", small_flagged), (0, "face_single_source", big_flagged)]
    labels = _labels(tmp_path, "a", {0: [small, small_flagged, big_flagged]}, flags)
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 595, 495, 610, 512)]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    faces = r.classes["face"]
    assert (faces.best_effort.hits, faces.best_effort.total) == (1, 2)
    assert (faces.uncertain.hits, faces.uncertain.total) == (0, 1)  # only the 40 px face
    assert (faces.rate.hits, faces.rate.total) == (0, 0)
    assert r.misses == [] and [m.box for m in r.uncertain_misses] == [big_flagged]


def test_an_uncertain_miss_fails_neither_p1_nor_p2(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    confident = _cell_targets()  # 30 faces in heatmap cell (0, 0)
    flagged = [Box("face", 303, 10 + 50 * r, 333, 50 + 50 * r, src="teacher_yunet") for r in range(5)]  # same cell, clear of the others
    plate = Box("plate", 1000, 600, 1040, 610, src="teacher_owlv2")
    labels = _labels(tmp_path, "a", {0: [*confident, *flagged, plate]}, [(0, "face_single_source", b) for b in flagged])
    masks = _masks(
        tmp_path / "m.jsonl", "a.mp4",
        {0: [Box(b.cls, b.x1 - 2, b.y1 - 2, b.x2 + 2, b.y2 + 2) for b in [*confident, plate]]},
        [0],
    )
    r = evaluate_coverage(frames, labels, masks)
    faces = r.classes["face"]
    assert (faces.rate.hits, faces.rate.total) == (30, 30)
    assert (faces.uncertain.hits, faces.uncertain.total) == (0, 5)
    assert r.class_passed("face") and r.heatmap_passed() and r.passed()
    assert r.weakest_cell()[:3] == ("face", 0, 0) and r.weakest_cell()[3] == Rate(30, 30)
    assert r.misses == [] and len(r.uncertain_misses) == 5
    assert all(m.frame == 0 and m.box.cls == "face" and m.covered == 0.0 for m in r.uncertain_misses)


def test_a_confident_miss_still_fails_whatever_the_uncertain_tier_does(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    missed = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    flagged = Box("face", 300, 100, 330, 140, src="teacher_yunet")
    plate = Box("plate", 1000, 600, 1040, 610, src="teacher_owlv2")
    labels = _labels(tmp_path, "a", {0: [missed, flagged, plate]}, [(0, "low_score", flagged)])
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [Box("face", 295, 95, 335, 145), Box("plate", 990, 590, 1050, 620)]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    assert (r.classes["face"].uncertain.hits, r.classes["face"].uncertain.total) == (1, 1)
    assert not r.class_passed("face") and not r.passed()
    assert [m.box for m in r.misses] == [missed] and r.uncertain_misses == []


def test_stray_area_treats_the_boxes_of_both_tiers_as_references(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    confident = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    flagged = Box("face", 300, 100, 330, 140, src="teacher_yunet")
    labels = _labels(tmp_path, "a", {0: [confident, flagged]}, [(0, "low_score", flagged)])
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [confident, flagged]}, [0])
    r = evaluate_coverage(frames, labels, masks)
    assert r.stray_max == 0.0  # a mask on a flagged box is not a false positive


def test_coverage_refuses_a_missing_flags_file(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    (labels / "flags.csv").unlink()
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match=r"flags\.csv does not exist; run vc_label fuse"):
        evaluate_coverage(frames, labels, masks)


def test_coverage_refuses_a_flags_file_older_than_processed(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    processed_mtime = (labels / "processed.txt").stat().st_mtime
    os.utime(labels / "flags.csv", (processed_mtime - 100, processed_mtime - 100))
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match=r"flags\.csv is older than processed\.txt"):
        evaluate_coverage(frames, labels, masks)


def _pool_inputs(tmp_path):
    """Two segments sharing one face position: `a` (2 frames: face and tiny face covered, plate and filled face missed)
    and `b` (1 frame: face covered, plate and filled face missed, plus a 50 x 50 px stray mask)."""
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    plate = Box("plate", 300, 300, 340, 310, src="teacher_owlv2")
    tiny = Box("face", 500, 500, 505, 508, src="teacher_yunet")  # best effort
    filled = Box("face", 900, 500, 920, 530, src="temporal_fill")
    frames_a = _frames(tmp_path, "a", [0, 10])
    labels_a = _labels(tmp_path, "a", {0: [face, plate, tiny], 10: [filled]})
    masks_a = _masks(tmp_path / "ma.jsonl", "a.mp4", {0: [Box("face", 90, 90, 140, 150), Box("face", 495, 495, 510, 512)]}, [0, 10])
    frames_b = _frames(tmp_path, "b", [0])
    labels_b = _labels(tmp_path, "b", {0: [face, plate, Box("face", 700, 100, 740, 150, src="temporal_fill")]})
    masks_b = _masks(tmp_path / "mb.jsonl", "b.mp4", {0: [Box("face", 90, 90, 140, 150), Box("face", 0, 0, 50, 50)]}, [0])
    return (
        evaluate_coverage(frames_a, labels_a, masks_a),
        evaluate_coverage(frames_b, labels_b, masks_b),
    )


def test_pool_adds_every_rate_and_concatenates_misses(tmp_path):
    a, b = _pool_inputs(tmp_path)
    p = pool([a, b])
    assert (p.video, p.frames, p.width, p.height) == ("a.mp4,b.mp4", 3, W, H)
    for cls in ("face", "plate"):
        for name in ("rate", "temporal_fill", "best_effort"):
            x, y, z = getattr(a.classes[cls], name), getattr(b.classes[cls], name), getattr(p.classes[cls], name)
            assert (z.hits, z.total) == (x.hits + y.hits, x.total + y.total), (cls, name)
        for label, rate in p.classes[cls].bins.items():
            assert (rate.hits, rate.total) == (
                a.classes[cls].bins[label].hits + b.classes[cls].bins[label].hits,
                a.classes[cls].bins[label].total + b.classes[cls].bins[label].total,
            )
        for r in range(len(p.classes[cls].heatmap)):
            for c, rate in enumerate(p.classes[cls].heatmap[r]):
                assert (rate.hits, rate.total) == (
                    a.classes[cls].heatmap[r][c].hits + b.classes[cls].heatmap[r][c].hits,
                    a.classes[cls].heatmap[r][c].total + b.classes[cls].heatmap[r][c].total,
                )
    # the same face (100,100) sits in the same heatmap cell of both segments: the cell adds up
    assert (a.classes["face"].heatmap[0][0].total, b.classes["face"].heatmap[0][0].total, p.classes["face"].heatmap[0][0].total) == (1, 1, 2)
    assert (p.classes["face"].rate.hits, p.classes["face"].rate.total) == (2, 4)  # a face and a filled face in each segment
    assert [(m.frame, m.box.cls) for m in p.misses] == [(0, "plate"), (10, "face"), (0, "plate"), (0, "face")]
    assert p.misses[:2] == a.misses and p.misses[2:] == b.misses
    # the originals are not touched
    assert (a.classes["face"].rate.total, b.classes["face"].rate.total) == (2, 2)


def test_pool_takes_the_worst_stray_and_weights_the_mean_by_frames(tmp_path):
    a, b = _pool_inputs(tmp_path)
    p = pool([a, b])
    assert p.stray_max == max(a.stray_max, b.stray_max)
    assert p.stray_p95 == max(a.stray_p95, b.stray_p95)
    assert p.stray_mean == pytest.approx((a.stray_mean * a.frames + b.stray_mean * b.frames) / (a.frames + b.frames))
    assert p.stray_mean != pytest.approx((a.stray_mean + b.stray_mean) / 2)  # 2 frames against 1: not a plain average


def test_pool_adds_the_uncertain_rates_and_concatenates_the_uncertain_misses(tmp_path):
    flagged_face = Box("face", 100, 100, 130, 140, src="teacher_yunet")
    flagged_plate = Box("plate", 300, 300, 340, 310, src="teacher_owlv2")
    confident = Box("face", 700, 100, 740, 150, src="teacher_retinaface")
    frames_a = _frames(tmp_path, "a", [0, 10])
    labels_a = _labels(
        tmp_path, "a", {0: [flagged_face, confident], 10: [flagged_plate]},
        [(0, "face_single_source", flagged_face), (10, "low_score", flagged_plate)],
    )
    masks_a = _masks(tmp_path / "ma.jsonl", "a.mp4", {0: [Box("face", 90, 90, 140, 150), Box("face", 690, 90, 750, 160)]}, [0, 10])
    frames_b = _frames(tmp_path, "b", [0])
    labels_b = _labels(
        tmp_path, "b", {0: [flagged_face, flagged_plate]},
        [(0, "low_score", flagged_face), (0, "low_score", flagged_plate)],
    )
    masks_b = _masks(tmp_path / "mb.jsonl", "b.mp4", {}, [0])
    a = evaluate_coverage(frames_a, labels_a, masks_a)
    b = evaluate_coverage(frames_b, labels_b, masks_b)
    p = pool([a, b])
    assert (a.classes["face"].uncertain, b.classes["face"].uncertain) == (Rate(1, 1), Rate(0, 1))
    assert (a.classes["plate"].uncertain, b.classes["plate"].uncertain) == (Rate(0, 1), Rate(0, 1))
    assert p.classes["face"].uncertain == Rate(1, 2) and p.classes["plate"].uncertain == Rate(0, 2)
    assert p.classes["face"].rate == Rate(1, 1)  # the confident tier is pooled on its own
    assert [(m.frame, m.box) for m in p.uncertain_misses] == [(10, flagged_plate), (0, flagged_face), (0, flagged_plate)]
    assert p.uncertain_misses[:1] == a.uncertain_misses and p.uncertain_misses[1:] == b.uncertain_misses
    assert p.misses == []
    assert b.classes["face"].uncertain == Rate(0, 1)  # the originals are not touched


def test_pool_judges_the_summed_counts_not_each_segment(tmp_path):
    # 100 faces and 100 plates per segment; b misses 2 faces: b alone is at 98 %, the two together at exactly 99 %
    def report(video, missed):
        r = CoverageReport(video, 10, W, H, {c: _empty_class(H) for c in ("face", "plate")})
        for cls in ("face", "plate"):
            hits = 100 - (missed if cls == "face" else 0)
            r.classes[cls].rate = Rate(hits, 100)
        return r

    a, b = report("a.mp4", 0), report("b.mp4", 2)
    assert a.class_passed("face") and not b.class_passed("face")
    assert pool([a, b]).class_passed("face") and pool([a, b]).passed()
    assert not pool([a, report("b.mp4", 3)]).class_passed("face")  # 197 / 200


def test_pool_of_one_report_is_that_report_and_pooling_nothing_is_refused(tmp_path):
    a, _ = _pool_inputs(tmp_path)
    p = pool([a])
    assert (p.video, p.frames, p.stray_max, p.stray_p95) == (a.video, a.frames, a.stray_max, a.stray_p95)
    assert p.stray_mean == pytest.approx(a.stray_mean)
    assert p.classes["face"].rate == a.classes["face"].rate and p.misses == a.misses
    with pytest.raises(ValueError, match="no coverage reports"):
        pool([])


def test_pool_refuses_reports_of_different_frame_sizes(tmp_path):
    a, _ = _pool_inputs(tmp_path)
    small = CoverageReport("c.mp4", 1, 640, 360, {c: _empty_class(360) for c in ("face", "plate")})
    with pytest.raises(ValueError, match="frame size"):
        pool([a, small])


def test_excluded_refs_leave_matching_targets_out_of_every_rate(tmp_path):
    frames = _frames(tmp_path, "a", [0, 10])
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    labels = _labels(tmp_path, "a", {0: [face, LIGHT], 10: [face, LIGHT]})
    _excluded(labels, [[0, "plate", 1802, 1001, 1862, 1021, "公交尾灯"]])  # IoU 0.85 with the light, frame 0 only
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [face], 10: [face]}, [0, 10])
    r = evaluate_coverage(frames, labels, masks)
    assert (r.classes["plate"].rate.hits, r.classes["plate"].rate.total) == (0, 1)  # frame 10's light still counts
    assert sum(rate.total for rate in r.classes["plate"].bins.values()) == 1
    assert [(e.frame, e.box.cls, e.box.x1, e.reason) for e in r.excluded] == [(0, "plate", 1800, "公交尾灯")]
    assert r.unmatched_exclusions == []
    assert [m.frame for m in r.misses] == [10]


def test_an_iou_of_exactly_one_half_excludes(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [LIGHT]})
    _excluded(labels, [[0, "plate", 1800, 1000, 1860, 1040, "x"]])  # 1200 / 2400 px^2
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert r.classes["plate"].rate.total == 0 and len(r.excluded) == 1


@pytest.mark.parametrize(
    "row",
    [
        [0, "plate", 1830, 1000, 1890, 1020, "x"],  # IoU 1/3
        [0, "face", 1800, 1000, 1860, 1020, "x"],  # another class
        [5, "plate", 1800, 1000, 1860, 1020, "x"],  # a frame without labels
    ],
)
def test_a_row_that_matches_no_reference_box_excludes_nothing_and_is_reported(tmp_path, row):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [LIGHT]})
    _excluded(labels, [row])
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert r.classes["plate"].rate.total == 1 and r.excluded == []
    assert [(e.frame, e.box.cls, e.line) for e in r.unmatched_exclusions] == [(row[0], row[1], 2)]


def test_pooled_exclusions_are_concatenated(tmp_path):
    for stem in ("a", "b"):
        frames = _frames(tmp_path, stem, [0])
        labels = _labels(tmp_path, stem, {0: [LIGHT]})
        _excluded(labels, [[0, "plate", 1800, 1000, 1860, 1020, "公交尾灯"], [3, "plate", 1, 1, 2, 2, "x"]])
    reports = [
        evaluate_coverage(tmp_path / "frames" / s, tmp_path / "labels" / s, _masks(tmp_path / f"{s}.jsonl", f"{s}.mp4", {}, [0]))
        for s in ("a", "b")
    ]
    pooled = pool(reports)
    assert len(pooled.excluded) == 2 and len(pooled.unmatched_exclusions) == 2 and pooled.exclusion_rows == 4


def test_read_excluded_without_a_file_excludes_nothing(tmp_path):
    assert read_excluded(tmp_path) == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("frame,cls,x1,y1,x2,y2\n0,plate,1,1,2,2\n", "no reason column"),
        ("frame,cls,x1,y1,x2,y2,reason\nzero,plate,1,1,2,2,x\n", "line 2: bad row"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,sign,1,1,2,2,x\n", "line 2: class 'sign'"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,x\n0,plate,5,5,5,9,y\n", "line 3: the box is empty"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,nan,1,2,2,x\n", "line 2: coordinates must be finite"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,x\n0,plate,1,1,inf,2,y\n", "line 3: coordinates must be finite"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,  \n", "line 2: no reason"),
    ],
)
def test_read_excluded_names_the_line_of_a_bad_row(tmp_path, text, message):
    (tmp_path / "excluded_refs.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_excluded(tmp_path)


def test_read_excluded_accepts_the_bom_excel_writes(tmp_path):
    text = "frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,公交尾灯\n"
    (tmp_path / "excluded_refs.csv").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert [(r.frame, r.box.cls, r.reason, r.line) for r in read_excluded(tmp_path)] == [(0, "plate", "公交尾灯", 2)]


def test_read_excluded_names_the_file_when_it_is_not_utf8(tmp_path):
    (tmp_path / "excluded_refs.csv").write_bytes("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,尾灯\n".encode("gbk"))
    with pytest.raises(ValueError, match=r"excluded_refs\.csv is not UTF-8"):
        read_excluded(tmp_path)


# Excluding a reference must be exactly as if it had never been labelled, in every tier and every statistic.
FAR_PLATE = Box("plate", 400, 500, 460, 520, src="teacher_owlv2")  # a second confident plate, never excluded
FAR_FACE = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
FLAGGED_LIGHT = Box("plate", 1800, 1000, 1860, 1020, score=0.35, src="teacher_owlv2")  # low_score in flags.csv: uncertain
TINY_LIGHT = Box("plate", 1800, 1000, 1812, 1004, src="teacher_owlv2")  # 12 x 4 px: too small to demand, best effort
FILL_LIGHT = Box("plate", 1800, 1000, 1860, 1020, src=TEMPORAL_SOURCE)  # a confident box the temporal fill made up


def _three_reports(tmp_path, extra, covered, flagged=False):
    """(excluded, kept, gone): the coverage with `extra` in the labels and a row excluding it, with it and no row, and without it.

    The masks sit exactly on the face and the far plate, and on `extra` too when `covered`.
    """
    reports = {}
    for stem, in_labels, excluded in (("excl", True, True), ("kept", True, False), ("gone", False, False)):
        frames = _frames(tmp_path, stem, [0])
        labels = _labels(tmp_path, stem, {0: [FAR_FACE, FAR_PLATE, *([extra] if in_labels else [])]},
                         flags=[(0, "low_score", extra)] if in_labels and flagged else [])
        if excluded:
            _excluded(labels, [[0, extra.cls, extra.x1, extra.y1, extra.x2, extra.y2, "x"]])
        masks = _masks(tmp_path / f"{stem}.jsonl", f"{stem}.mp4", {0: [FAR_FACE, FAR_PLATE, *([extra] if covered else [])]}, [0])
        reports[stem] = evaluate_coverage(frames, labels, masks)
    return reports["excl"], reports["kept"], reports["gone"]


@pytest.mark.parametrize(
    "extra, covered, flagged, tier",
    [
        pytest.param(LIGHT, True, False, "confident", id="confident-covered"),
        pytest.param(LIGHT, False, False, "confident", id="confident-missed"),
        pytest.param(FILL_LIGHT, True, False, "confident", id="temporal-fill"),
        pytest.param(TINY_LIGHT, True, False, "best_effort", id="best-effort-covered"),
        pytest.param(TINY_LIGHT, False, False, "best_effort", id="best-effort-missed"),
        pytest.param(FLAGGED_LIGHT, True, True, "uncertain", id="uncertain-covered"),
        pytest.param(FLAGGED_LIGHT, False, True, "uncertain", id="uncertain-missed"),
    ],
)
def test_excluding_a_reference_equals_never_having_labelled_it(tmp_path, extra, covered, flagged, tier):
    excl, kept, gone = _three_reports(tmp_path, extra, covered, flagged)
    assert kept.classes != gone.classes  # the extra box does count when nothing excludes it
    assert excl.classes == gone.classes  # every rate, size bin, heatmap cell, temporal fill, best-effort and uncertain count
    assert excl.misses == gone.misses and excl.uncertain_misses == gone.uncertain_misses
    assert (excl.stray_max, excl.stray_mean, excl.stray_p95) == (gone.stray_max, gone.stray_mean, gone.stray_p95)
    assert [(e.tier, e.covered) for e in excl.excluded] == [(tier, covered)]


def test_an_excluded_reference_that_was_covered_is_no_hit_and_its_mask_is_stray_area(tmp_path):
    excl, kept, _ = _three_reports(tmp_path, LIGHT, covered=True)
    plate = lambda r: (r.classes["plate"].rate.hits, r.classes["plate"].rate.total)  # noqa: E731
    assert plate(kept) == (2, 2) and plate(excl) == (1, 1)  # both counts drop, not just the denominator
    assert kept.stray_max == 0.0
    assert excl.stray_max == pytest.approx(60 * 20 / (W * H))  # the mask on the light has no reference any more


def test_an_excluded_temporal_fill_box_leaves_its_heatmap_cell_and_the_fill_rate(tmp_path):
    excl, kept, _ = _three_reports(tmp_path, FILL_LIGHT, covered=True)
    row, col = heat_cell(FILL_LIGHT, W, H)
    assert (kept.classes["plate"].temporal_fill, kept.classes["plate"].heatmap[row][col]) == (Rate(1, 1), Rate(1, 1))
    assert (excl.classes["plate"].temporal_fill, excl.classes["plate"].heatmap[row][col]) == (Rate(0, 0), Rate(0, 0))


def test_one_row_excludes_every_reference_box_it_overlaps(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    other = Box("plate", 1810, 1000, 1870, 1020, src="teacher_owlv2")  # IoU 0.85 with the row, as is LIGHT
    labels = _labels(tmp_path, "a", {0: [LIGHT, other]})
    _excluded(labels, [[0, "plate", 1805, 1000, 1865, 1020, "公交尾灯"]])
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert r.classes["plate"].rate.total == 0
    assert [e.box.x1 for e in r.excluded] == [1800, 1810] and [e.reason for e in r.excluded] == ["公交尾灯"] * 2
    assert r.exclusion_rows == 1 and r.unmatched_exclusions == []


def test_rows_that_all_overlap_one_reference_box_are_all_matched_and_the_closest_gives_the_reason(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [LIGHT]})
    _excluded(labels, [
        [0, "plate", 1802, 1001, 1862, 1021, "roughly"],  # IoU 0.85
        [0, "plate", 1800, 1000, 1860, 1020, "exactly"],  # IoU 1
        [0, "plate", 1810, 1000, 1870, 1020, "also"],  # IoU 0.71
    ])
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert [e.reason for e in r.excluded] == ["exactly"]
    assert r.unmatched_exclusions == [] and r.exclusion_rows == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {"face": 0.99, "plate": 0.99}),
        ("face=0.95,plate=0.98", {"face": 0.95, "plate": 0.98}),
        (" plate = 0.98 ", {"face": 0.99, "plate": 0.98}),
        ("face=1", {"face": 1.0, "plate": 0.99}),
    ],
)
def test_parse_recall_targets(text, expected):
    assert parse_recall_targets(text) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("car=0.9", "car"),
        ("face", "<class>=<value>"),
        ("face=0.9,face=0.95", "twice"),
        ("face=high", "not a number"),
        ("face=0", r"\(0, 1\]"),
        ("plate=1.5", r"\(0, 1\]"),
        ("face=nan", r"\(0, 1\]"),
    ],
)
def test_parse_recall_targets_refuses_bad_text(text, message):
    with pytest.raises(ValueError, match=message):
        parse_recall_targets(text)


def test_a_class_passes_at_its_own_recall_target(tmp_path):
    frames = _frames(tmp_path, "a", list(range(20)))
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    labels = _labels(tmp_path, "a", {fr: [face] for fr in range(20)})
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {fr: [face] for fr in range(1, 20)}, list(range(20)))  # 19 of 20
    default = evaluate_coverage(frames, labels, masks)
    assert default.recall_targets == {"face": 0.99, "plate": 0.99} and not default.class_passed("face")
    lowered = evaluate_coverage(frames, labels, masks, {"face": 0.95})
    assert lowered.recall_targets == {"face": 0.95, "plate": 0.99}
    assert lowered.class_passed("face")  # 19 / 20 is exactly 95 %


def test_pooling_needs_the_same_recall_targets():
    def classes():
        return {c: _empty_class(H) for c in ("face", "plate")}

    a = CoverageReport("a.mp4", 1, W, H, classes(), recall_targets={"face": 0.95, "plate": 0.98})
    b = CoverageReport("b.mp4", 1, W, H, classes())
    with pytest.raises(ValueError, match="different recall targets"):
        pool([a, b])
    assert pool([a, a]).recall_targets == {"face": 0.95, "plate": 0.98}
