import csv
import json

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, Detection, write_jsonl
from vc_eval.coverage import evaluate_coverage
from vc_eval.records import read_masks
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
    (d / "processed.txt").write_text("".join(f"{fr}\n" for fr in boxes_by_frame))
    write_jsonl(d / "merged.jsonl", [Detection(f"{stem}.mp4", fr, fr * 16.7, b) for fr, bs in boxes_by_frame.items() for b in bs])
    return d


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
