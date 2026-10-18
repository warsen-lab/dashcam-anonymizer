from pathlib import Path

import pytest

from vc_label.contracts import (
    Box,
    Detection,
    from_yolo_lines,
    read_frames_csv,
    read_jsonl,
    to_yolo_lines,
    write_jsonl,
)


def test_box_geometry():
    b = Box("face", 10, 20, 40, 80)
    assert (b.width, b.height, b.area, b.center) == (30, 60, 1800, (25, 50))


def test_box_clipped_and_shifted():
    b = Box("plate", -5, 10, 120, 50, score=0.7, src="t")
    assert b.clipped(100, 40) == Box("plate", 0, 10, 100, 40, score=0.7, src="t")
    assert b.shifted(5, -10) == Box("plate", 0, 0, 125, 40, score=0.7, src="t")


def test_read_frames_csv(tmp_path):
    (tmp_path / "frames.csv").write_text(
        "video,frame,pts_ms,reason,motion_score\n"
        "trip.mp4,0,0.0,base,0.0000\n"
        "trip.mp4,12,200.2,motion,0.0610\n"
    )
    refs = read_frames_csv(tmp_path)
    assert [(r.video, r.frame, r.pts_ms) for r in refs] == [("trip.mp4", 0, 0.0), ("trip.mp4", 12, 200.2)]
    assert refs[1].image == tmp_path / "000012.jpg"


def test_jsonl_round_trip_and_append(tmp_path):
    path = tmp_path / "raw.jsonl"
    first = Detection("trip.mp4", 3, 50.04, Box("face", 1.04, 2, 3, 4, score=0.91234, src="teacher_yunet"))
    second = Detection("trip.mp4", 4, 66.7, Box("plate", 5, 6, 7, 8, score=0.5, src="teacher_owlv2"))
    write_jsonl(path, [first])
    write_jsonl(path, [second], append=True)
    back = list(read_jsonl(path))
    assert back[0] == Detection("trip.mp4", 3, 50.0, Box("face", 1.0, 2, 3, 4, score=0.9123, src="teacher_yunet"))
    assert back[1] == second


def test_read_jsonl_of_missing_file_is_empty(tmp_path):
    assert list(read_jsonl(tmp_path / "nope.jsonl")) == []


def test_yolo_round_trip():
    boxes = [Box("face", 100, 50, 140, 100), Box("plate", 0, 0, 1920, 1080)]
    lines = to_yolo_lines(boxes, 1920, 1080)
    assert lines[0] == "0 0.062500 0.069444 0.020833 0.046296"
    assert lines[1] == "1 0.500000 0.500000 1.000000 1.000000"
    back = from_yolo_lines(lines, 1920, 1080)
    assert [b.cls for b in back] == ["face", "plate"]
    assert back[0].x1 == pytest.approx(100, abs=0.01) and back[0].y2 == pytest.approx(100, abs=0.01)


def test_yolo_rejects_auxiliary_classes():
    with pytest.raises(ValueError, match="vehicle"):
        to_yolo_lines([Box("vehicle", 0, 0, 10, 10)], 100, 100)


def test_frame_image_paths_are_six_digit_jpgs(tmp_path):
    (tmp_path / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\nv.mp4,7,1.0,base,0\n")
    assert read_frames_csv(tmp_path)[0].image.name == "000007.jpg"
    assert isinstance(read_frames_csv(tmp_path)[0].image, Path)
