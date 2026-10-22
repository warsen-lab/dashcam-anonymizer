import json

import cv2
import numpy as np
import pytest

from vc_hardneg.candidates import box_in_crop, candidate_id, crop_window, mine, native_crop, read_candidates
from vc_hardneg.config import load_config

CFG = load_config()


def test_crop_window_is_the_spec_square():
    assert crop_window((100, 200, 140, 210), 2.5, 32) == (70.0, 155.0, 100.0)
    assert crop_window((10, 10, 16, 14), 2.5, 32) == (-3.0, -4.0, 32.0)


def test_native_crop_pads_outside_the_frame():
    image = np.full((50, 60, 3), 200, np.uint8)
    crop, origin = native_crop(image, (-10.0, -5.0, 20.0), 114)
    assert origin == (-10, -5)
    assert crop.shape == (20, 20, 3)
    assert (crop[:5, :, :] == 114).all() and (crop[:, :10, :] == 114).all()
    assert (crop[5:, 10:, :] == 200).all()


def test_candidate_ids_are_stable_and_distinct():
    a = candidate_id("v.mp4", 12, "plate", (1.0, 2.0, 3.0, 4.0))
    assert a == candidate_id("v.mp4", 12, "plate", (1.04, 2.0, 3.0, 4.0))  # 0.1 px rounding, as the records
    assert a != candidate_id("v.mp4", 24, "plate", (1.0, 2.0, 3.0, 4.0))
    assert len(a) == 16


def _run(tmp_path, stem, rows, masks):
    run = tmp_path / "runs" / stem
    run.mkdir(parents=True)
    (run / "verify_crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "masks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in masks))
    return run


def _labels(tmp_path, stem, frames, teacher):
    d = tmp_path / "labels" / stem
    d.mkdir(parents=True)
    (d / "processed.txt").write_text("".join(f"{f}\n" for f in frames))
    (d / "merged.jsonl").write_text("".join(json.dumps(t) + "\n" for t in teacher))
    fr = tmp_path / "frames" / stem
    fr.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(fr / f"{f:06d}.jpg"), np.full((120, 160, 3), 90, np.uint8))


def _row(i, frame, box, cls="plate", score=0.1, src="student"):
    return {"index": i, "video": "s.mp4", "frame": frame, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
            "y2": box[3], "score": score, "src": src, "size": 64}


def _mask(frame, boxes):
    return {"video": "s.mp4", "frame": frame, "boxes": [
        {"cls": c, "x1": b[0] - 2, "y1": b[1] - 2, "x2": b[2] + 2, "y2": b[3] + 2, "track": t, "origin": "detected"}
        for c, b, t in boxes]}


def test_mine_keeps_masked_boxes_on_labelled_frames_at_most_per_track(tmp_path):
    box = (40.0, 40.0, 60.0, 50.0)
    frames = [0, 12, 24, 36, 48]
    rows = [_row(i, f, box) for i, f in enumerate(frames)]
    rows.append(_row(5, 12, (100.0, 10.0, 110.0, 20.0)))  # not masked (no track): left out
    rows.append(_row(6, 60, box))  # frame 60 is not labelled: left out
    masks = [_mask(f, [("plate", box, 7)]) for f in frames]
    run = _run(tmp_path, "s", rows, masks)
    teacher = [{"video": "s.mp4", "frame": 24, "pts_ms": 0, "cls": "plate", "x1": 41, "y1": 40, "x2": 60, "y2": 50,
                "score": 0.8, "src": "teacher_owlv2"}]
    _labels(tmp_path, "s", frames, teacher)
    out = tmp_path / "out"
    summary = mine([run], tmp_path / "labels", tmp_path / "frames", out, CFG)
    cands = read_candidates(out / "candidates.jsonl")
    assert summary.candidates == len(cands) == 3  # per_track 3 of the 5 frames of track 7
    assert summary.masked == 5
    assert [c.frame for c in cands] == [0, 24, 48]  # spread evenly along the track
    assert all(c.track == 7 and c.dump_index == c.frame // 12 for c in cands)
    assert {c.frame: c.frame_teacher for c in cands}[24] == pytest.approx(0.8)
    assert {c.frame: c.frame_teacher for c in cands}[0] == 0.0
    for c in cands:
        assert (out / c.image).is_file()
        b = box_in_crop(c, CFG)
        assert b.x1 == pytest.approx(40 - c.crop_origin[0]) and b.y1 == pytest.approx(40 - c.crop_origin[1])


def test_mine_labels_takes_the_low_band_teacher_boxes_on_processed_frames(tmp_path):
    from vc_hardneg.candidates import LOW_BANDS, mine_labels

    def t(frame, cls, box, score, src="teacher_owlv2"):
        return {"video": "s.mp4", "frame": frame, "pts_ms": 0, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
                "y2": box[3], "score": score, "src": src}

    teacher = [
        t(0, "plate", (10, 10, 30, 20), 0.15),  # low band: kept
        t(0, "plate", (40, 10, 60, 20), 0.35),  # confident: left out
        t(0, "plate", (70, 10, 90, 20), 0.05),  # below the band (not a label): left out
        t(12, "face", (10, 50, 30, 75), 0.4, "teacher_retinaface"),  # low band: kept
        t(12, "face", (40, 50, 60, 75), 0.6, "teacher_retinaface"),  # confident: left out
        t(24, "plate", (10, 10, 30, 20), 0.2),  # frame 24 is not processed: left out
    ]
    _labels(tmp_path, "s", [0, 12], teacher)
    out = tmp_path / "out"
    s = mine_labels(["s"], tmp_path / "labels", tmp_path / "frames", out, CFG)
    cands = read_candidates(out / "candidates.jsonl")
    assert (s.rows, s.candidates) == (6, 2) and len(cands) == 2
    assert [(c.frame, c.cls, c.score, c.src) for c in cands] == [(0, "plate", 0.15, "teacher_owlv2"),
                                                                  (12, "face", 0.4, "teacher_retinaface")]
    assert all(c.track == -1 and c.dump_index == -1 and c.dump_dir == "" for c in cands)
    assert all((out / c.image).is_file() for c in cands)
    assert LOW_BANDS == {"plate": (0.1, 0.3), "face": (0.3, 0.5)}
