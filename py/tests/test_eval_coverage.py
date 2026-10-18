import csv
import json
import os

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, Detection, write_jsonl
from vc_eval.coverage import CoverageReport, _empty_class, evaluate_coverage, pool
from vc_eval.metrics import Rate
from vc_eval.records import detections_by_frame, read_masks
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


def _labels(root, stem, boxes_by_frame):
    d = root / "labels" / stem
    d.mkdir(parents=True)
    # processed.txt first, then merged.jsonl, so merged is never older than processed (the fuse order)
    (d / "processed.txt").write_text("".join(f"{fr}\n" for fr in boxes_by_frame))
    write_jsonl(d / "merged.jsonl", [Detection(f"{stem}.mp4", fr, fr * 16.7, b) for fr, bs in boxes_by_frame.items() for b in bs])
    return d


def _cell_targets():
    """30 must-mask faces packed into heatmap cell (row 0, col 0) of a 1920x1080 frame: 6 columns x 5 rows."""
    return [Box("face", 10 + 50 * c, 10 + 50 * r, 50 + 50 * c, 50 + 50 * r, src="teacher_retinaface") for r in range(5) for c in range(6)]


def _masks(path, video, masks_by_frame, frames):
    with path.open("w") as f:
        for fr in frames:
            boxes = [{"cls": b.cls, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "origin": "detected"} for b in masks_by_frame.get(fr, [])]
            f.write(json.dumps({"video": video, "frame": fr, "boxes": boxes}) + "\n")
    return path


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
    with pytest.raises(ValueError, match="different videos"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_refuses_a_dense_frame_without_a_masks_line(tmp_path):
    dense = tmp_path / "dense.jsonl"
    write_jsonl(dense, [Detection("a.mp4", 5, 83.5, Box("face", 100, 100, 120, 130, score=0.9, src="student"))])
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {0: [Box("face", 95, 95, 125, 135)]}, [0])
    with pytest.raises(ValueError, match="frame 5"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_takes_the_video_from_masks_when_dense_is_empty(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = _masks(tmp_path / "sparse.jsonl", "a.mp4", {}, [0, 1])
    r = evaluate_temporal(dense, sparse, 1080)
    assert (r.video, r.frames, r.misses) == ("a.mp4", 2, [])


def test_temporal_refuses_an_empty_dense_file_with_masks_of_several_videos(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = tmp_path / "sparse.jsonl"
    sparse.write_text(
        json.dumps({"video": "a.mp4", "frame": 0, "boxes": []}) + "\n" + json.dumps({"video": "b.mp4", "frame": 1, "boxes": []}) + "\n"
    )
    with pytest.raises(ValueError, match="a.mp4, b.mp4"):
        evaluate_temporal(dense, sparse, 1080)


def test_temporal_refuses_an_empty_masks_file(tmp_path):
    dense = tmp_path / "dense.jsonl"
    dense.write_text("")
    sparse = tmp_path / "sparse.jsonl"
    sparse.write_text("")
    with pytest.raises(ValueError, match="no frames"):
        evaluate_temporal(dense, sparse, 1080)


def test_coverage_refuses_a_missing_processed_file(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [Box("face", 100, 100, 130, 140)]})
    (labels / "processed.txt").unlink()
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0])
    with pytest.raises(ValueError, match="processed.txt does not exist"):
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
