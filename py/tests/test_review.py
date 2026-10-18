import csv
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from vc_label.contracts import Box, FrameRef, read_frames_csv
from vc_label.review import (
    REVIEWED_FLAG,
    SCENE_FLAGS,
    SOURCE_KEY,
    count_without_source,
    ensure_outside,
    import_review,
    labelme_document,
    read_labelme_boxes,
    write_review_queue,
)
from vc_label.rules import Flag


def test_ensure_outside_rejects_the_frames_dir_and_anything_below_it(tmp_path):
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the frames directory"):
        ensure_outside(frames, frames / "labels")
    with pytest.raises(ValueError):
        ensure_outside(frames, frames)
    ensure_outside(frames, tmp_path / "labels" / "trip")
    ensure_outside(frames, tmp_path / "frames" / "trip2")


def _filesystem_is_case_sensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.mkdir()
    try:
        return not (directory / "caseprobe").exists()
    finally:
        probe.rmdir()


def test_ensure_outside_sees_through_case_variants_of_the_frames_dir(tmp_path):
    if _filesystem_is_case_sensitive(tmp_path):
        pytest.skip("case-sensitive filesystem: a differently cased path is a different directory")
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the frames directory"):
        ensure_outside(frames, tmp_path / "frames" / "TRIP" / "labels")
    with pytest.raises(ValueError, match="inside the frames directory"):
        ensure_outside(frames, tmp_path / "FRAMES" / "Trip")
    ensure_outside(frames, tmp_path / "FRAMES" / "trip2")


def test_labelme_document_points_back_to_the_frame(tmp_path):
    frames, review = tmp_path / "frames" / "trip", tmp_path / "review" / "trip"
    ref = FrameRef("trip.mp4", 7, 116.7, frames / "000007.jpg")
    face = Box("face", 10, 20, 30, 50, score=0.42, src="teacher_yunet")

    doc = labelme_document(ref, [face], [Flag("trip.mp4", 7, "low_score", face)], review, 1920, 1080)

    assert doc["imagePath"] == "../../frames/trip/000007.jpg"
    assert SOURCE_KEY == "vc_source"
    assert doc[SOURCE_KEY] == {"video": "trip.mp4", "frame": 7}  # survives X-AnyLabeling rewriting imagePath
    assert (doc["imageWidth"], doc["imageHeight"], doc["imageData"]) == (1920, 1080, None)
    assert doc["flags"] == {"reviewed": False, "day": False, "night": False, "rain": False, "tunnel": False, "city": False, "highway": False}
    assert list(doc["flags"]) == [REVIEWED_FLAG, *SCENE_FLAGS]  # the review checkbox comes first
    [shape] = doc["shapes"]
    assert shape["label"] == "face" and shape["shape_type"] == "rectangle"
    assert shape["points"] == [[10, 20], [30, 50]]
    assert shape["description"] == "score=0.42 src=teacher_yunet low_score"


def test_review_queue_lists_most_flagged_frames_first(tmp_path):
    b = Box("vehicle", 1, 2, 3, 4)
    flags = [Flag("v.mp4", 5, "low_score", b), Flag("v.mp4", 9, "low_score", b), Flag("v.mp4", 9, "vehicle_without_plate", b)]
    path = tmp_path / "review_queue.csv"

    write_review_queue(path, flags)

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["video", "frame", "flags", "rules", "where"]
    assert rows[1] == ["v.mp4", "9", "2", "low_score;vehicle_without_plate", "low_score@1,2,3,4 vehicle_without_plate@1,2,3,4"]
    assert rows[2][:2] == ["v.mp4", "5"]


def _frames_dir(tmp_path: Path, frames_numbers=(7,), name: str = "trip") -> Path:
    frames = tmp_path / "frames" / name
    frames.mkdir(parents=True)
    rows = ["video,frame,pts_ms,reason,motion_score"] + [f"{name}.mp4,{n},{n * 16.7:.1f},base,0" for n in frames_numbers]
    (frames / "frames.csv").write_text("\n".join(rows) + "\n")
    for n in frames_numbers:
        Image.new("RGB", (200, 100)).save(frames / f"{n:06d}.jpg")
    return frames


_FROM_FRAME = object()  # vc_source default: the frame this file is named after, of video trip.mp4


def _review_json(
    review: Path,
    shapes: list[dict],
    flags: dict,
    frame: int = 7,
    reviewed: bool = True,
    size: tuple[int, int] = (200, 100),
    image_path: str | None = None,
    vc_source: object = _FROM_FRAME,
) -> None:
    """A LabelMe file as export-review writes it; imagePath and vc_source default to the frame of `_frames_dir(review.parent)`.

    `vc_source=None` leaves the key out.
    """
    review.mkdir(parents=True, exist_ok=True)
    if image_path is None:
        image_path = os.path.relpath(review.parent / "frames" / "trip" / f"{frame:06d}.jpg", review)
    if vc_source is _FROM_FRAME:
        vc_source = {"video": "trip.mp4", "frame": frame}
    doc = {
        "version": "5.4.1",
        "flags": {REVIEWED_FLAG: reviewed, **flags},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": size[1],
        "imageWidth": size[0],
    }
    if vc_source is not None:
        doc[SOURCE_KEY] = vc_source
    (review / f"{frame:06d}.json").write_text(json.dumps(doc))


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
    with (gold / "gold_meta.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [
        ["image", "video", "frame", "reviewed", "scene_tags"],
        ["000007.jpg", "trip.mp4", "7", "true", "city;day"],
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


def test_import_review_accepts_four_point_rectangles_and_clips(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    shapes = [{"label": "face", "points": [[150, 10], [250, 10], [250, 30], [150, 30]], "shape_type": "rectangle"}]
    _review_json(review, shapes, {})

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert (gold / "yolo" / "000007.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.875000 0.200000 0.250000 0.200000",
    ]


@pytest.mark.parametrize(
    "shape, message",
    [
        ({"label": "face", "points": [[0, 0]], "shape_type": "rectangle"}, "need at least 2"),
        ({"label": "plate", "points": [[300, 10], [400, 20]], "shape_type": "rectangle"}, "empty after clipping"),
    ],
)
def test_import_review_rejects_degenerate_shapes(tmp_path, shape, message):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _review_json(review, [shape], {})

    with pytest.raises(ValueError, match=message):
        import_review(review, read_frames_csv(frames), tmp_path / "gold")


def test_export_then_import_round_trip(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    ref = FrameRef("trip.mp4", 7, 116.7, frames / "000007.jpg")
    boxes = [
        Box("face", 10, 20, 30, 50, score=0.9, src="teacher_retinaface+teacher_yunet"),
        Box("plate", 100, 60, 140, 70, score=0.6, src="teacher_owlv2"),
    ]
    review.mkdir(parents=True)
    doc = labelme_document(ref, boxes, [], review, 200, 100)
    doc["flags"][REVIEWED_FLAG] = True  # what the human does when the frame is done
    (review / "000007.json").write_text(json.dumps(doc), encoding="utf-8")

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert (gold / "yolo" / "000007.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.100000 0.350000 0.100000 0.300000",
        "1 0.600000 0.650000 0.200000 0.100000",
    ]


def test_import_review_rejects_unreviewed_frames_and_writes_nothing(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [], {}, frame=7, reviewed=True)
    _review_json(review, [], {}, frame=8, reviewed=False)

    with pytest.raises(ValueError, match=r"1 of 2 frames are not marked reviewed.*\b8\b.*--allow-unreviewed"):
        import_review(review, read_frames_csv(frames), gold)

    assert not gold.exists()


def test_import_review_lists_at_most_twenty_unreviewed_frames(tmp_path):
    numbers = tuple(range(1, 31))
    frames, review = _frames_dir(tmp_path, numbers), tmp_path / "review"
    for n in numbers:
        _review_json(review, [], {}, frame=n, reviewed=False)

    with pytest.raises(ValueError) as excinfo:
        import_review(review, read_frames_csv(frames), tmp_path / "gold")

    message = str(excinfo.value)
    assert "30 of 30 frames are not marked reviewed" in message
    assert "20" in message and "21" not in message.split("--allow")[0]  # only the first twenty are listed
    assert "10 more" in message


def test_import_review_can_pass_unreviewed_frames_through_as_pre_labels(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [], {"day": True}, frame=7, reviewed=True)
    _review_json(review, [], {"night": True}, frame=8, reviewed=False)

    assert import_review(review, read_frames_csv(frames), gold, allow_unreviewed=True) == 2

    with (gold / "gold_meta.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [
        ["image", "video", "frame", "reviewed", "scene_tags"],
        ["000007.jpg", "trip.mp4", "7", "true", "day"],
        ["000008.jpg", "trip.mp4", "8", "false", "night"],
    ]


def test_read_labelme_boxes_reports_reviewed_apart_from_scene_tags(tmp_path):
    review = tmp_path / "review"
    _review_json(review, [], {"day": True, "rain": False})

    boxes, tags, reviewed, width, height = read_labelme_boxes(review / "000007.json")

    assert (boxes, tags, reviewed, width, height) == ([], ["day"], True, 200, 100)
    assert REVIEWED_FLAG not in tags


def test_read_labelme_boxes_without_a_reviewed_flag_is_unreviewed(tmp_path):
    review = tmp_path / "review"
    _review_json(review, [], {})
    path = review / "000007.json"
    doc = json.loads(path.read_text())
    del doc["flags"][REVIEWED_FLAG]
    path.write_text(json.dumps(doc))

    assert read_labelme_boxes(path)[2] is False


def test_import_review_rejects_an_unknown_true_flag_but_ignores_a_false_one(tmp_path):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _review_json(review, [], {"snow": True})

    with pytest.raises(ValueError, match=r"000007\.json.*'snow'"):
        import_review(review, read_frames_csv(frames), tmp_path / "gold")

    _review_json(review, [], {"snow": False})
    assert import_review(review, read_frames_csv(frames), tmp_path / "gold") == 1


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


_FACE = {"label": "face", "points": [[0, 0], [20, 10]], "shape_type": "rectangle"}


def test_import_review_is_atomic_when_a_later_file_is_bad(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [_FACE], {"day": True}, frame=7)
    _review_json(review, [], {}, frame=8)
    assert import_review(review, read_frames_csv(frames), gold) == 2
    before = _snapshot(gold)
    _review_json(review, [], {"night": True}, frame=7)  # would change 000007.txt and the meta
    _review_json(review, [{"label": "person", "points": [[0, 0], [1, 1]], "shape_type": "rectangle"}], {}, frame=8)

    with pytest.raises(ValueError, match="unknown label"):
        import_review(review, read_frames_csv(frames), gold)

    assert _snapshot(gold) == before


def test_import_review_removes_labels_of_frames_no_longer_in_the_review_dir(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [_FACE], {}, frame=7)
    _review_json(review, [_FACE], {}, frame=8)
    assert import_review(review, read_frames_csv(frames), gold) == 2
    (review / "000008.json").unlink()

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert sorted(p.name for p in (gold / "yolo").iterdir()) == ["000007.txt", "classes.txt"]
    assert (gold / "yolo" / "classes.txt").read_text(encoding="utf-8") == "face\nplate\n"
    with (gold / "gold_meta.csv").open(newline="", encoding="utf-8") as fh:
        assert [row[2] for row in csv.reader(fh)] == ["frame", "7"]
    assert not (gold / "gold_meta.csv.tmp").exists()


def test_import_review_rejects_a_json_sized_differently_from_its_frame_image(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [_FACE], {}, size=(100, 200))

    with pytest.raises(ValueError, match=r"000007\.json.*100x200.*200x100"):
        import_review(review, read_frames_csv(frames), gold)

    assert not gold.exists()


def test_import_review_refuses_a_labels_dir_as_gold_dir(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "labels"
    _review_json(review, [_FACE], {})
    gold.mkdir()
    (gold / "processed.txt").write_text("7\n")

    with pytest.raises(ValueError, match="labels dir"):
        import_review(review, read_frames_csv(frames), gold)

    assert sorted(p.name for p in gold.iterdir()) == ["processed.txt"]


def _imported_gold(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A frames dir, a review dir holding one reviewed frame, and a gold dir imported from it."""
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [_FACE], {"day": True})
    assert import_review(review, read_frames_csv(frames), gold) == 1
    return frames, review, gold


def test_import_review_refuses_a_missing_or_empty_review_dir_and_leaves_gold_alone(tmp_path):
    frames, review, gold = _imported_gold(tmp_path)
    before = _snapshot(gold)
    assert before  # a gold dir with a label file and gold_meta.csv worth losing
    empty = tmp_path / "empty_review"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a LabelMe file")
    not_a_dir = tmp_path / "review.json"
    not_a_dir.write_text("{}")

    with pytest.raises(ValueError, match=r"nowhere.*not a directory"):
        import_review(tmp_path / "nowhere", read_frames_csv(frames), gold)
    with pytest.raises(ValueError, match=r"empty_review.*no \*\.json"):
        import_review(empty, read_frames_csv(frames), gold)
    with pytest.raises(ValueError, match=r"review\.json.*not a directory"):
        import_review(not_a_dir, read_frames_csv(frames), gold)

    assert _snapshot(gold) == before


def test_import_review_rejects_a_review_dir_exported_from_another_video(tmp_path):
    frames_a, frames_b = _frames_dir(tmp_path), _frames_dir(tmp_path, name="other")  # same frame number, same resolution
    review, gold = tmp_path / "review", tmp_path / "gold"
    ref_b = read_frames_csv(frames_b)[0]
    review.mkdir()
    doc = labelme_document(ref_b, [Box("face", 10, 20, 30, 50)], [], review, 200, 100)
    doc["flags"][REVIEWED_FLAG] = True
    (review / "000007.json").write_text(json.dumps(doc), encoding="utf-8")
    assert doc["imagePath"] == "../frames/other/000007.jpg"

    with pytest.raises(ValueError) as excinfo:
        import_review(review, read_frames_csv(frames_a), gold)

    message = str(excinfo.value)
    assert "000007.json" in message
    assert "other.mp4#7" in message and "trip.mp4#7" in message  # what the file says, what the importer expected
    assert not gold.exists()
    # The same file imported against the right frames dir is fine.
    assert import_review(review, read_frames_csv(frames_b), gold) == 1


def test_import_review_rejects_a_relative_image_path_into_another_frames_dir_even_with_a_matching_vc_source(tmp_path):
    frames, review, gold = _imported_gold(tmp_path)
    other = _frames_dir(tmp_path, name="other")
    before = _snapshot(gold)
    _review_json(review, [], {"night": True}, image_path=os.path.relpath(other / "000007.jpg", review))

    with pytest.raises(ValueError, match=r"000007\.json.*other"):
        import_review(review, read_frames_csv(frames), gold)

    assert _snapshot(gold) == before


def test_import_review_accepts_any_spelling_of_the_frame_image_path(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    image = frames / "000007.jpg"
    for spelled in (str(image), str(image.parent / ".." / "trip" / image.name), os.path.relpath(image, review)):
        _review_json(review, [_FACE], {}, image_path=spelled)
        assert import_review(review, read_frames_csv(frames), gold) == 1


def test_import_review_falls_back_to_the_file_name_when_image_path_does_not_resolve(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    # A review dir copied to another machine: the relative path leads nowhere, the frame image name still tells.
    _review_json(review, [], {}, frame=7, image_path="/elsewhere/trip/000007.jpg")
    _review_json(review, [], {}, frame=8, image_path="C:\\frames\\trip\\000008.jpg")
    assert import_review(review, read_frames_csv(frames), gold) == 2

    _review_json(review, [], {}, frame=8, image_path="/elsewhere/trip/000009.jpg")
    before = _snapshot(gold)
    with pytest.raises(ValueError, match=r"000008\.json.*000009\.jpg.*000008\.jpg"):
        import_review(review, read_frames_csv(frames), gold)
    assert _snapshot(gold) == before


@pytest.mark.parametrize("image_path", [None, 7, ""])
def test_import_review_rejects_a_json_without_a_usable_image_path(tmp_path, image_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [], {})
    path = review / "000007.json"
    doc = json.loads(path.read_text())
    doc["imagePath"] = image_path
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match=r"000007\.json.*imagePath"):
        import_review(review, read_frames_csv(frames), gold)

    assert not gold.exists()


@pytest.mark.parametrize("value", ["false", "true", "yes", 1, [True]])
def test_only_a_real_true_marks_a_frame_reviewed(tmp_path, value):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    _review_json(review, [_FACE], {})
    path = review / "000007.json"
    doc = json.loads(path.read_text())
    doc["flags"][REVIEWED_FLAG] = value  # a hand-edited or foreign tool's file: truthy, but not the checkbox
    path.write_text(json.dumps(doc))

    assert read_labelme_boxes(path)[2] is False
    with pytest.raises(ValueError, match=r"1 of 1 frames are not marked reviewed: 7"):
        import_review(review, read_frames_csv(frames), gold)
    assert not gold.exists()

    assert import_review(review, read_frames_csv(frames), gold, allow_unreviewed=True) == 1
    with (gold / "gold_meta.csv").open(newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh))[1][3] == "false"


def test_exported_review_dir_has_the_source_in_every_file_and_imports(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    review.mkdir()
    for ref in read_frames_csv(frames):
        doc = labelme_document(ref, [], [], review, 200, 100)
        assert doc[SOURCE_KEY] == {"video": "trip.mp4", "frame": ref.frame}
        doc["flags"][REVIEWED_FLAG] = True
        (review / f"{ref.frame:06d}.json").write_text(json.dumps(doc), encoding="utf-8")

    assert import_review(review, read_frames_csv(frames), gold) == 2


def _xanylabeling_save(frame: int = 7, source: object = _FROM_FRAME) -> dict:
    """What X-AnyLabeling 4.1.0 writes back after a GUI save of one of our files (checked on a real round trip).

    It rewrites imagePath to the bare file name, saves rectangles as 4 points, adds `checked` and per-shape
    bookkeeping, writes its own version, and keeps the image-level description and unknown top-level keys.
    """
    doc = {
        "version": "4.1.0",
        "flags": {REVIEWED_FLAG: True, "day": True, "night": False, "rain": False, "tunnel": False, "city": False, "highway": False},
        "checked": False,
        "shapes": [
            {
                "label": "face",
                "score": None,
                "points": [[146.0, 20.4], [162.9, 20.4], [162.9, 38.4], [146.0, 38.4]],
                "group_id": None,
                "description": "score=0.93 src=teacher_retinaface face_single_source",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        ],
        "imagePath": f"{frame:06d}.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 200,
        "description": f"vc_label video=trip.mp4 frame={frame}",
    }
    if source is _FROM_FRAME:
        source = {"video": "trip.mp4", "frame": frame}
    if source is not None:
        doc[SOURCE_KEY] = source
    return doc


def _write_saved(review: Path, doc: dict, frame: int = 7) -> None:
    review.mkdir(parents=True, exist_ok=True)
    (review / f"{frame:06d}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def test_import_review_accepts_a_file_as_x_anylabeling_saves_it(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    _write_saved(review, _xanylabeling_save())

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert (gold / "yolo" / "000007.txt").read_text(encoding="utf-8").splitlines() == ["0 0.772250 0.294000 0.084500 0.180000"]
    with (gold / "gold_meta.csv").open(newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh))[1] == ["000007.jpg", "trip.mp4", "7", "true", "day"]


_SAVED_WRONG_SOURCE = [
    pytest.param(None, r"000007\.json: no vc_source; re-export it with vc_label export-review", id="missing"),
    pytest.param({"video": "other.mp4", "frame": 7}, r"000007\.json.*other\.mp4#7.*trip\.mp4#7", id="other-video-same-frame-number"),
    pytest.param({"video": "trip.mp4", "frame": 8}, r"000007\.json.*trip\.mp4#8.*trip\.mp4#7", id="same-video-other-frame"),
    pytest.param({"video": "trip.mp4", "frame": "7"}, r"000007\.json.*trip\.mp4#'7'.*trip\.mp4#7", id="frame-is-a-string"),
    pytest.param({"video": "trip.mp4"}, r"000007\.json.*trip\.mp4#None.*trip\.mp4#7", id="no-frame"),
    pytest.param("trip.mp4#7", r"000007\.json.*vc_source", id="not-an-object"),
]


@pytest.mark.parametrize("source, message", _SAVED_WRONG_SOURCE)
def test_import_review_rejects_a_file_whose_vc_source_is_missing_or_wrong_and_leaves_gold_alone(tmp_path, source, message):
    frames, review, gold = _imported_gold(tmp_path)
    before = _snapshot(gold)
    # A bare imagePath names the right file (every video's frames are NNNNNN.jpg), so only vc_source can tell.
    _write_saved(review, _xanylabeling_save(source=source))

    with pytest.raises(ValueError, match=message):
        import_review(review, read_frames_csv(frames), gold)

    assert _snapshot(gold) == before


def test_import_review_with_one_file_missing_vc_source_writes_nothing(tmp_path):
    frames, review, gold = _frames_dir(tmp_path, (7, 8)), tmp_path / "review", tmp_path / "gold"
    _write_saved(review, _xanylabeling_save(frame=7))
    _write_saved(review, _xanylabeling_save(frame=8, source=None), frame=8)  # a later file without it

    with pytest.raises(ValueError, match=r"000008\.json: no vc_source"):
        import_review(review, read_frames_csv(frames), gold)

    assert not gold.exists()


def test_a_json_true_is_not_frame_one_in_vc_source(tmp_path):
    # True == 1 in Python, so frame 1 is the only frame a boolean could pass for; every other frame number differs anyway.
    frames, review, gold = _frames_dir(tmp_path, (1,)), tmp_path / "review", tmp_path / "gold"
    _write_saved(review, _xanylabeling_save(frame=1, source={"video": "trip.mp4", "frame": True}), frame=1)

    with pytest.raises(ValueError, match=r"000001\.json.*trip\.mp4#True.*trip\.mp4#1"):
        import_review(review, read_frames_csv(frames), gold)
    assert not gold.exists()

    _write_saved(review, _xanylabeling_save(frame=1), frame=1)  # the integer 1 is fine
    assert import_review(review, read_frames_csv(frames), gold) == 1


def test_vc_source_mismatch_states_both_sides_and_does_not_guess_the_cause(tmp_path):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _write_saved(review, _xanylabeling_save(source={"video": "other.mp4", "frame": 7}))

    with pytest.raises(ValueError) as excinfo:
        import_review(review, read_frames_csv(frames), tmp_path / "gold")

    message = str(excinfo.value)
    assert "000007.json" in message
    assert "other.mp4#7" in message and "trip.mp4#7" in message  # recorded, then expected
    assert "frames.csv row" in message and "does not belong" in message
    assert "exported from another video" not in message  # a hand-edited or copied file fails the same way


def test_count_without_source_counts_parsable_json_files_that_lack_the_key(tmp_path):
    review = tmp_path / "review"
    _write_saved(review, _xanylabeling_save(frame=1), frame=1)
    _write_saved(review, _xanylabeling_save(frame=2, source=None), frame=2)
    _write_saved(review, _xanylabeling_save(frame=3, source="garbage"), frame=3)  # present, if wrong: import-review reports it
    _write_saved(review, _xanylabeling_save(frame=4, source=None), frame=4)
    (review / "000005.json").write_text("{not json")  # unreadable: import-review reports it too
    (review / "review_queue.csv").write_text("video,frame\n")

    assert count_without_source(review) == (2, 5)
    assert count_without_source(tmp_path / "nowhere") == (0, 0)


@pytest.mark.parametrize("source", [None, "trip.mp4#7"], ids=["missing", "not-an-object"])
def test_import_review_points_a_file_without_a_usable_vc_source_at_the_recovery_recipe(tmp_path, source):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _write_saved(review, _xanylabeling_save(source=source))

    with pytest.raises(ValueError) as excinfo:
        import_review(review, read_frames_csv(frames), tmp_path / "gold")

    assert str(excinfo.value).endswith('; see docs/labeling-guide.md, section "没有 vc_source 的旧复核目录"')
