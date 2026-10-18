import csv
import json
from pathlib import Path

import pytest

from vc_label.contracts import Box, FrameRef, read_frames_csv
from vc_label.review import ensure_outside, import_review, labelme_document, write_review_queue
from vc_label.rules import Flag


def test_ensure_outside_rejects_the_frames_dir_and_anything_below_it(tmp_path):
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the frames directory"):
        ensure_outside(frames, frames / "labels")
    with pytest.raises(ValueError):
        ensure_outside(frames, frames)
    ensure_outside(frames, tmp_path / "labels" / "trip")


def test_labelme_document_points_back_to_the_frame(tmp_path):
    frames, review = tmp_path / "frames" / "trip", tmp_path / "review" / "trip"
    ref = FrameRef("trip.mp4", 7, 116.7, frames / "000007.jpg")
    face = Box("face", 10, 20, 30, 50, score=0.42, src="teacher_yunet")

    doc = labelme_document(ref, [face], [Flag("trip.mp4", 7, "low_score", face)], review, 1920, 1080)

    assert doc["imagePath"] == "../../frames/trip/000007.jpg"
    assert (doc["imageWidth"], doc["imageHeight"], doc["imageData"]) == (1920, 1080, None)
    assert doc["flags"] == {"day": False, "night": False, "rain": False, "tunnel": False, "city": False, "highway": False}
    [shape] = doc["shapes"]
    assert shape["label"] == "face" and shape["shape_type"] == "rectangle"
    assert shape["points"] == [[10, 20], [30, 50]]
    assert shape["description"] == "score=0.42 src=teacher_yunet low_score"


def test_review_queue_lists_most_flagged_frames_first(tmp_path):
    b = Box("vehicle", 1, 2, 3, 4)
    flags = [Flag("v.mp4", 5, "low_score", b), Flag("v.mp4", 9, "low_score", b), Flag("v.mp4", 9, "vehicle_without_plate", b)]
    path = tmp_path / "review_queue.csv"

    write_review_queue(path, flags)

    rows = list(csv.reader(path.open()))
    assert rows[0] == ["video", "frame", "flags", "rules", "where"]
    assert rows[1] == ["v.mp4", "9", "2", "low_score;vehicle_without_plate", "low_score@1,2,3,4 vehicle_without_plate@1,2,3,4"]
    assert rows[2][:2] == ["v.mp4", "5"]


def _frames_dir(tmp_path: Path) -> Path:
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    (frames / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\ntrip.mp4,7,116.7,base,0\n")
    return frames


def _review_json(review: Path, shapes: list[dict], flags: dict) -> None:
    review.mkdir(parents=True)
    doc = {"version": "5.4.1", "flags": flags, "shapes": shapes, "imagePath": "x.jpg", "imageData": None, "imageHeight": 100, "imageWidth": 200}
    (review / "000007.json").write_text(json.dumps(doc))


def test_import_review_writes_gold_yolo_and_scene_tags(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    shapes = [
        {"label": "plate", "points": [[120, 60], [100, 50]], "shape_type": "rectangle"},
        {"label": "face", "points": [[0, 0], [20, 10]], "shape_type": "rectangle"},
    ]
    _review_json(review, shapes, {"day": True, "city": True, "night": False})

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert (gold / "yolo" / "000007.txt").read_text().splitlines() == [
        "1 0.550000 0.550000 0.100000 0.100000",
        "0 0.050000 0.050000 0.100000 0.100000",
    ]
    assert list(csv.reader((gold / "gold_meta.csv").open())) == [
        ["image", "video", "frame", "scene_tags"],
        ["000007.jpg", "trip.mp4", "7", "city;day"],
    ]


@pytest.mark.parametrize(
    "shape, message",
    [
        ({"label": "person", "points": [[0, 0], [1, 1]], "shape_type": "rectangle"}, "unknown label"),
        ({"label": "face", "points": [[0, 0], [1, 1], [2, 0]], "shape_type": "polygon"}, "not a rectangle"),
    ],
)
def test_import_review_rejects_bad_shapes(tmp_path, shape, message):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _review_json(review, [shape], {})

    with pytest.raises(ValueError, match=message):
        import_review(review, read_frames_csv(frames), tmp_path / "gold")
