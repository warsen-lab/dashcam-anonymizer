import csv

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train import dataset
from vc_train.dataset import DATA_YAML, MANIFEST, build_dataset, crop_boxes, crop_windows


def test_crop_windows_at_native_scale_on_1080p():
    assert crop_windows(1920, 1080, 1.0) == [(0, 0, 1280, 1080), (640, 0, 1280, 1080)]


def test_crop_windows_small_frame_is_one_window():
    assert crop_windows(640, 360, 1.0) == [(0, 0, 640, 360)]


def test_crop_windows_on_4k_at_half_scale():
    assert crop_windows(3840, 2160, 0.5) == [
        (0, 0, 2560, 2160),
        (1280, 0, 2560, 2160),
    ]


def test_crop_windows_tiles_both_axes_when_needed():
    assert crop_windows(3000, 3000, 1.0) == [
        (0, 0, 1280, 1280),
        (1720, 0, 1280, 1280),
        (0, 1720, 1280, 1280),
        (1720, 1720, 1280, 1280),
    ]


def test_crop_windows_round_half_up_like_the_letterbox():
    assert crop_windows(100, 100, 2.0, crop=5)[0] == (0, 0, 3, 3)  # 5 / 2 = 2.5 -> 3, not 2


def test_crop_boxes_shifts_clips_and_drops_mostly_hidden_boxes():
    window = (640, 0, 1280, 1080)
    inside = Box("face", 700, 10, 720, 40)
    half = Box("plate", 600, 100, 680, 120)  # 40 of 80 px visible -> 50 %
    hidden = Box("face", 620, 200, 650, 230)  # 10 of 30 px visible -> 33 %, kept
    mostly_hidden = Box("face", 600, 300, 650, 330)  # 10 of 50 px visible -> 20 %, dropped
    out = crop_boxes([inside, half, hidden, mostly_hidden], window)
    assert [(b.cls, b.x1, b.y1, b.x2, b.y2) for b in out] == [
        ("face", 60, 10, 80, 40),
        ("plate", 0, 100, 40, 120),
        ("face", 0, 200, 10, 230),
    ]


def _segment(root, stem, frames, boxes_by_frame, size=(1920, 1080)):
    """frames: list of (frame, pts_ms). Writes frames dir + labels dir like vc_extract / vc_label fuse."""
    frames_dir = root / "frames" / stem
    labels_dir = root / "labels" / stem
    (labels_dir / "yolo").mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    width, height = size
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for frame, pts in frames:
            w.writerow([f"{stem}.mp4", frame, pts, "base", 0.0])
            cv2.imwrite(str(frames_dir / f"{frame:06d}.jpg"), np.full((height, width, 3), frame % 255, np.uint8))
            lines = to_yolo_lines(boxes_by_frame.get(frame, []), width, height)
            (labels_dir / "yolo" / f"{frame:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("".join(f"{frame}\n" for frame, _ in frames))
    return frames_dir, labels_dir


def test_build_dataset_writes_crops_full_frames_and_time_split(tmp_path):
    seg = _segment(
        tmp_path,
        "a",
        [(0, 0.0), (10, 5000.0), (20, 15000.0), (30, 30000.0)],
        {0: [Box("face", 100, 100, 120, 130)], 30: [Box("plate", 1700, 900, 1760, 920)]},
    )
    out = tmp_path / "ds"
    summary = build_dataset([seg], out, 1920, 1088)

    assert (summary.train_frames, summary.val_frames) == (2, 2)
    assert (summary.train_samples, summary.val_samples) == (6, 6)
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert names == [
        "a_000000_c0.jpg", "a_000000_c1.jpg", "a_000000_full.jpg",
        "a_000010_c0.jpg", "a_000010_c1.jpg", "a_000010_full.jpg",
    ]
    assert cv2.imread(str(out / "images" / "train" / "a_000000_c1.jpg")).shape == (1080, 1280, 3)
    # the face at x=100 is only inside the left crop
    assert (out / "labels" / "train" / "a_000000_c0.txt").read_text().startswith("0 ")
    assert (out / "labels" / "train" / "a_000000_c1.txt").read_text() == ""
    assert (out / "labels" / "train" / "a_000000_full.txt").read_text().startswith("0 ")
    # the plate at x=1700 is only inside the right crop, frame 30 is in val
    assert (out / "labels" / "val" / "a_000030_c1.txt").read_text().startswith("1 ")
    assert (out / "labels" / "val" / "a_000030_c0.txt").read_text() == ""

    yaml_text = (out / DATA_YAML).read_text()
    assert f"path: {out.resolve()}" in yaml_text
    assert "train: images/train" in yaml_text and "val: images/val" in yaml_text
    assert "0: face" in yaml_text and "1: plate" in yaml_text
    with (out / MANIFEST).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert rows[0].keys() == {"sample", "split", "video", "frame", "x0", "y0", "w", "h", "scale", "boxes"}
    assert summary.boxes == sum(int(r["boxes"]) for r in rows) == 4


def test_build_dataset_requires_fused_labels(tmp_path):
    frames_dir, labels_dir = _segment(tmp_path, "a", [(0, 0.0), (10, 200.0)], {})
    (labels_dir / "yolo" / "000010.txt").unlink()
    with pytest.raises(ValueError, match="vc_label fuse"):
        build_dataset([(frames_dir, labels_dir)], tmp_path / "ds", 1920, 1088)


def test_build_dataset_refuses_a_non_empty_output_dir(tmp_path):
    seg = _segment(tmp_path, "a", [(0, 0.0)], {})
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        build_dataset([seg], out, 1920, 1088)


def test_build_dataset_rejects_labels_of_another_video(tmp_path):
    seg_a = _segment(tmp_path, "a", [(0, 0.0)], {})
    seg_b = _segment(tmp_path, "b", [(5, 0.0)], {})
    with pytest.raises(ValueError, match="does not match"):
        build_dataset([(seg_a[0], seg_b[1])], tmp_path / "ds", 1920, 1088)


def test_failed_label_check_creates_nothing_and_a_rerun_with_the_same_out_succeeds(tmp_path):
    frames_dir, labels_dir = _segment(tmp_path, "a", [(0, 0.0), (10, 200.0)], {})
    label_file = labels_dir / "yolo" / "000010.txt"
    label_file.unlink()
    out = tmp_path / "ds"
    with pytest.raises(ValueError, match="vc_label fuse"):
        build_dataset([(frames_dir, labels_dir)], out, 1920, 1088)
    assert not out.exists()

    label_file.write_text("")
    summary = build_dataset([(frames_dir, labels_dir)], out, 1920, 1088)
    assert summary.val_frames == 2  # both frames sit inside the 20 s tail
    assert (out / "labels" / "val" / "a_000010_full.txt").exists()


def test_build_dataset_refuses_two_segments_with_the_same_video_stem(tmp_path):
    seg_one = _segment(tmp_path / "one", "a", [(0, 0.0)], {})
    seg_two = _segment(tmp_path / "two", "a", [(0, 0.0)], {})
    out = tmp_path / "ds"
    with pytest.raises(ValueError, match="'a'"):
        build_dataset([seg_one, seg_two], out, 1920, 1088)
    assert not out.exists()


@pytest.mark.parametrize("preexisting", [False, True])
def test_failed_write_leaves_the_output_dir_as_it_was(tmp_path, monkeypatch, preexisting):
    seg = _segment(tmp_path, "a", [(0, 0.0)], {})
    out = tmp_path / "ds"
    if preexisting:
        out.mkdir()
    monkeypatch.setattr(dataset.cv2, "imread", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="cannot read"):
        build_dataset([seg], out, 1920, 1088)
    if preexisting:
        assert out.is_dir() and not any(out.iterdir())
    else:
        assert not out.exists()


def test_build_dataset_4k_frame_at_half_scale_is_cut_and_shrunk_to_the_input_size(tmp_path):
    seg = _segment(tmp_path, "k", [(0, 0.0)], {0: [Box("face", 1000, 1000, 1100, 1100)]}, size=(3840, 2160))
    out = tmp_path / "ds"
    summary = build_dataset([seg], out, 1920, 1088)

    # the only frame is inside the 20 s tail, so it is validation data
    for i in (0, 1):
        assert cv2.imread(str(out / "images" / "val" / f"k_000000_c{i}.jpg")).shape == (1080, 1280, 3)
    # windows (0, 0, 2560, 2160) and (1280, 0, 2560, 2160); the box is only in the first one.
    # exact maths: cx = 1050 * 0.5 / 1280 = 0.41015625, w = 50 / 1280 = 0.0390625 (a tie), cy and h as below.
    # The input label is stored with 6 decimals on a 3840 px frame (cx 0.273438 = 1050.002 px), which moves
    # the box by ~0.002 px, so the written values are 0.410157 and 0.039063.
    assert (out / "labels" / "val" / "k_000000_c0.txt").read_text() == "0 0.410157 0.486111 0.039063 0.046296\n"
    assert (out / "labels" / "val" / "k_000000_c1.txt").read_text() == ""
    with (out / MANIFEST).open() as f:
        rows = {r["sample"]: r for r in csv.DictReader(f)}
    assert (rows["k_000000_c0.jpg"]["x0"], rows["k_000000_c0.jpg"]["y0"]) == ("0", "0")
    assert (rows["k_000000_c0.jpg"]["w"], rows["k_000000_c0.jpg"]["h"]) == ("2560", "2160")
    assert (rows["k_000000_c1.jpg"]["x0"], rows["k_000000_c1.jpg"]["y0"]) == ("1280", "0")
    assert rows["k_000000_c0.jpg"]["scale"] == rows["k_000000_c1.jpg"]["scale"] == "0.5"
    assert summary.boxes == 2  # one box in crop c0, one in the full frame


@pytest.mark.parametrize("width, height", [(100, 1088), (1920, 1080), (0, 1088), (1920, -32)])
def test_build_dataset_wants_positive_multiples_of_32_before_writing(tmp_path, width, height):
    seg = _segment(tmp_path, "a", [(0, 0.0)], {})
    out = tmp_path / "ds"
    with pytest.raises(ValueError, match="multiple of 32"):
        build_dataset([seg], out, width, height)
    assert not out.exists()
