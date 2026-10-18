import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from vc_label.config import LabelingConfig
from vc_label.contracts import Box, Detection, read_jsonl, write_jsonl
from vc_label.pipeline import Teachers, label_frame, read_processed, run_export_review, run_fuse, run_teach

CFG = LabelingConfig()
W, H = 640, 360


class FakeFaces:
    def __init__(self, src: str) -> None:
        self.src = src

    def detect(self, bgr):
        return [Box("face", 100, 40, 120, 64, score=0.9, src=self.src)]


class FakeCoco:
    def detect(self, bgr):
        return [
            Box("vehicle", 300, 200, 500, 330, score=0.99, src="teacher_coco:car"),  # 200 px wide
            Box("vehicle", 10, 10, 20, 18, score=0.9, src="teacher_coco:car"),  # 10 px: too small
            Box("person", 90, 30, 130, 200, score=0.9, src="teacher_coco:person"),
        ]


class FakePlates:
    def __init__(self) -> None:
        self.searched: list[list[Box]] = []

    def detect(self, bgr, vehicles):
        self.searched.append(vehicles)
        return [Box("plate", v.x1 + 80, v.y2 - 30, v.x1 + 120, v.y2 - 20, score=0.6, src="teacher_owlv2") for v in vehicles]


def teachers() -> Teachers:
    return Teachers(faces=[FakeFaces("teacher_retinaface"), FakeFaces("teacher_yunet")], coco=FakeCoco(), plates=FakePlates())


def make_frames(root: Path, count: int, name: str = "trip") -> Path:
    frames = root / "frames" / name
    frames.mkdir(parents=True)
    rows = ["video,frame,pts_ms,reason,motion_score"]
    for i in range(count):
        frame = i * 12
        cv2.imwrite(str(frames / f"{frame:06d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        rows.append(f"{name}.mp4,{frame},{frame * 1000 / 60:.1f},base,0.0")
    (frames / "frames.csv").write_text("\n".join(rows) + "\n")
    return frames


def test_label_frame_only_searches_plates_on_wide_vehicles():
    t = teachers()
    raw, aux = label_frame(np.zeros((H, W, 3), np.uint8), t, CFG)
    assert [b.src for b in raw] == ["teacher_retinaface", "teacher_yunet", "teacher_owlv2"]
    assert len(aux) == 3
    assert [v.width for v in t.plates.searched[0]] == [200]


def test_run_teach_resumes_where_it_stopped(tmp_path):
    frames, labels = make_frames(tmp_path, 6), tmp_path / "labels"

    assert run_teach(frames, labels, teachers(), CFG, every=2, limit=2, log=lambda _: None) == 2
    assert read_processed(labels) == {0, 24}
    assert run_teach(frames, labels, teachers(), CFG, every=2, log=lambda _: None) == 1
    assert read_processed(labels) == {0, 24, 48}
    assert run_teach(frames, labels, teachers(), CFG, every=2, log=lambda _: None) == 0
    assert sorted({d.frame for d in read_jsonl(labels / "raw.jsonl")}) == [0, 24, 48]
    assert len(list(read_jsonl(labels / "raw.jsonl"))) == 9


def test_run_teach_drops_output_of_unfinished_frames(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    write_jsonl(labels / "raw.jsonl", [Detection("trip.mp4", 0, 0.0, Box("face", 0, 0, 1, 1, src="crashed"))])

    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    assert "crashed" not in {d.box.src for d in read_jsonl(labels / "raw.jsonl")}


def test_run_teach_leaves_complete_output_untouched(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)
    raw = labels / "raw.jsonl"
    inode, mtime, text = raw.stat().st_ino, raw.stat().st_mtime_ns, raw.read_text(encoding="utf-8")

    assert run_teach(frames, labels, teachers(), CFG, log=lambda _: None) == 0

    assert raw.stat().st_ino == inode
    assert raw.stat().st_mtime_ns == mtime  # an in-place truncate-and-rewrite would bump mtime
    assert raw.read_text(encoding="utf-8") == text
    assert not (labels / "raw.jsonl.tmp").exists()


def test_run_teach_recovers_from_a_torn_last_line(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n")
    write_jsonl(labels / "raw.jsonl", [Detection("trip.mp4", 0, 0.0, Box("face", 1, 2, 3, 4, score=0.9, src="kept"))])
    with (labels / "raw.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"video": "trip.mp4", "fr')

    assert run_teach(frames, labels, teachers(), CFG, log=lambda _: None) == 1

    assert read_processed(labels) == {0, 12}
    assert "kept" in {d.box.src for d in read_jsonl(labels / "raw.jsonl")}


def test_run_fuse_writes_merged_flags_and_yolo(tmp_path):
    frames, labels = make_frames(tmp_path, 3), tmp_path / "labels"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    summary, sequence, flags = run_fuse(frames, labels, CFG)

    assert (summary.frames, summary.boxes, summary.filled) == (3, 6, 0)
    assert [b.cls for b in sequence[0][1]] == ["face", "plate"]
    assert sequence[0][1][0].src == "teacher_retinaface+teacher_yunet"
    assert flags == []  # face from both teachers, large person has a face, wide car has a plate
    assert (labels / "yolo" / "classes.txt").read_text() == "face\nplate\n"
    assert len((labels / "yolo" / "000012.txt").read_text().splitlines()) == 2
    assert len(list(read_jsonl(labels / "merged.jsonl"))) == 6
    assert next(csv.reader((labels / "flags.csv").open()))[:3] == ["video", "frame", "rule"]


def test_run_export_review_writes_one_labelme_file_per_frame(tmp_path):
    frames, labels, review = make_frames(tmp_path, 2), tmp_path / "labels", tmp_path / "review"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    assert run_export_review(frames, labels, review, CFG) == (2, 0)

    doc = json.loads((review / "000012.json").read_text())
    assert doc["imagePath"] == "../frames/trip/000012.jpg"
    assert [s["label"] for s in doc["shapes"]] == ["face", "plate"]
    assert (review / "review_queue.csv").exists()


def _edit_review_file(path: Path) -> dict:
    doc = json.loads(path.read_text())
    doc["shapes"].append({"label": "face", "points": [[1, 2], [30, 40]], "shape_type": "rectangle", "flags": {}})
    doc["flags"]["night"] = True
    path.write_text(json.dumps(doc))
    return doc


def test_run_export_review_never_overwrites_reviewed_files_by_default(tmp_path):
    frames, labels, review = make_frames(tmp_path, 2), tmp_path / "labels", tmp_path / "review"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)
    run_export_review(frames, labels, review, CFG)
    edited = _edit_review_file(review / "000012.json")
    (review / "review_queue.csv").write_text("stale\n")

    assert run_export_review(frames, labels, review, CFG) == (0, 2)

    assert json.loads((review / "000012.json").read_text()) == edited
    assert (review / "review_queue.csv").read_text() != "stale\n"  # derived: always rewritten

    assert run_export_review(frames, labels, review, CFG, force=True) == (2, 0)

    doc = json.loads((review / "000012.json").read_text())
    assert [s["label"] for s in doc["shapes"]] == ["face", "plate"]
    assert doc["flags"]["night"] is False


def test_run_export_review_writes_only_new_frames_after_another_teach_batch(tmp_path):
    frames, labels, review = make_frames(tmp_path, 3), tmp_path / "labels", tmp_path / "review"
    run_teach(frames, labels, teachers(), CFG, limit=1, log=lambda _: None)
    assert run_export_review(frames, labels, review, CFG) == (1, 0)
    edited = _edit_review_file(review / "000000.json")
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    assert run_export_review(frames, labels, review, CFG) == (2, 1)

    assert json.loads((review / "000000.json").read_text()) == edited
    assert sorted(p.name for p in review.glob("*.json")) == ["000000.json", "000012.json", "000024.json"]


def test_run_fuse_rejects_a_labels_dir_taught_on_another_video(tmp_path):
    frames_a, labels = make_frames(tmp_path, 3), tmp_path / "labels"
    frames_b = make_frames(tmp_path, 3, name="other")
    run_teach(frames_a, labels, teachers(), CFG, log=lambda _: None)

    with pytest.raises(ValueError, match=r"labels dir .*labels does not match .*other: .*trip\.mp4"):
        run_fuse(frames_b, labels, CFG)

    assert not (labels / "merged.jsonl").exists()


def test_run_fuse_rejects_processed_frames_that_are_not_in_frames_csv(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n5\n7\n")

    with pytest.raises(ValueError, match=r"does not match .*processed\.txt.*\[5, 7\]"):
        run_fuse(frames, labels, CFG)


def test_run_teach_rejects_a_labels_dir_taught_on_another_video_and_leaves_it_alone(tmp_path):
    frames_a, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    frames_b = make_frames(tmp_path, 2, name="other")
    run_teach(frames_a, labels, teachers(), CFG, log=lambda _: None)
    before = {p.name: p.read_bytes() for p in labels.iterdir()}

    with pytest.raises(ValueError, match=r"does not match .*other"):
        run_teach(frames_b, labels, teachers(), CFG, log=lambda _: None)

    assert {p.name: p.read_bytes() for p in labels.iterdir()} == before


def test_a_bad_line_before_the_last_names_the_file_and_line(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n")
    good = '{"video": "trip.mp4", "frame": 0, "pts_ms": 0.0, "cls": "face", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.9, "src": "s"}\n'
    (labels / "raw.jsonl").write_text(good + "not json\n" + good)

    with pytest.raises(ValueError, match=r"raw\.jsonl:2"):
        run_teach(frames, labels, teachers(), CFG, log=lambda _: None)
    with pytest.raises(ValueError, match=r"raw\.jsonl:2"):
        run_fuse(frames, labels, CFG)


TEACH_PIN = {"retinaface_score": 0.3, "yunet_score": 0.5, "coco_score": 0.3, "owlv2_score": 0.1, "plate_search_min_vehicle_frac": 0.03}


def test_run_teach_pins_the_teacher_thresholds_on_first_use(tmp_path):
    frames, labels = make_frames(tmp_path, 3), tmp_path / "labels"

    run_teach(frames, labels, teachers(), CFG, limit=1, log=lambda _: None)

    assert json.loads((labels / "teach_config.json").read_text(encoding="utf-8")) == TEACH_PIN
    # Same teacher thresholds resume fine, and fuse-time settings may change between batches.
    assert run_teach(frames, labels, teachers(), replace(CFG, fuse_iou=0.6, temporal_max_gap_ms=900.0), limit=1, log=lambda _: None) == 1
    assert json.loads((labels / "teach_config.json").read_text(encoding="utf-8")) == TEACH_PIN


@pytest.mark.parametrize("change", [{"yunet_score": 0.4}, {"owlv2_score": 0.2}, {"plate_search_min_vehicle_frac": 0.05}])
def test_run_teach_rejects_changed_teacher_thresholds_and_leaves_the_labels_dir_alone(tmp_path, change):
    frames, labels = make_frames(tmp_path, 3), tmp_path / "labels"
    run_teach(frames, labels, teachers(), CFG, limit=1, log=lambda _: None)
    before = {p.name: p.read_bytes() for p in labels.iterdir()}
    (name, new), = change.items()

    with pytest.raises(ValueError, match=rf"{name} {TEACH_PIN[name]} -> {new}.*one set of teacher thresholds.*new labels dir"):
        run_teach(frames, labels, teachers(), replace(CFG, **change), log=lambda _: None)

    assert {p.name: p.read_bytes() for p in labels.iterdir()} == before


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_run_teach_rejects_a_corrupt_teacher_pin(tmp_path, content):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    (labels / "teach_config.json").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="teach_config.json"):
        run_teach(frames, labels, teachers(), CFG, log=lambda _: None)
