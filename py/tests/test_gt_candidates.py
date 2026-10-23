import csv

import cv2
import numpy as np

from vc_gt.candidates import (BOXES, PROCESSED, TRACKS, FrameFinds, build_candidates, linkable, merge_duplicates,
                              read_persons, relink, tile_windows)
from vc_gt.config import load_config
from vc_gt.tracks import read_tracks
from vc_label.contracts import Box


def test_tile_windows_cover_the_frame_and_overlap_their_neighbours():
    w = tile_windows(100, 50, 2, 0.2)
    assert [(b.x1, b.y1, b.x2, b.y2) for b in w] == [(0, 0, 60, 30), (40, 0, 100, 30), (0, 20, 60, 50), (40, 20, 100, 50)]


def test_merge_duplicates_keeps_the_best_box_and_every_source():
    owl = Box("plate", 100, 100, 140, 112, 0.3, "teacher_owlv2")
    student = Box("plate", 101, 100, 141, 112, 0.6, "student")
    face = Box("face", 100, 100, 140, 112, 0.5, "teacher_yunet")  # another class: never merged
    out = merge_duplicates([owl, student, face], 0.5)
    assert [(b.cls, b.x1, b.score, b.src) for b in out] == [
        ("plate", 101, 0.6, "student+teacher_owlv2"), ("face", 100, 0.5, "teacher_yunet")]


class _Finder:
    """A plate drifting right; call n of the fake holds the plate at x = 100 + 4 n and n people."""

    def __init__(self):
        self.calls = 0

    def find(self, bgr):
        n = self.calls
        self.calls += 1
        return FrameFinds([Box("plate", 100 + 4 * n, 500, 140 + 4 * n, 512, 0.4, "teacher_owlv2")], persons=n)


def _frames(tmp_path, count=3):
    d = tmp_path / "frames" / "v"
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for k in range(count):
            w.writerow(["v.mp4", 6 * k, 100.0 * k, "base", 0.0])
            cv2.imwrite(str(d / f"{6 * k:06d}.jpg"), np.zeros((64, 96, 3), np.uint8))
    return d


def test_build_candidates_is_resumable_and_links_once_every_frame_is_done(tmp_path):
    frames, out, cfg = _frames(tmp_path), tmp_path / "gt", load_config()
    student = {12: [Box("face", 10, 10, 30, 34, 0.02)]}
    finder = _Finder()
    assert build_candidates(frames, out, finder, student, cfg, limit=2) == 2
    assert not (out / TRACKS).exists()
    assert build_candidates(frames, out, finder, student, cfg) == 1
    tracks = read_tracks(out / TRACKS)
    assert [(t.cls, t.frames) for t in tracks] == [("plate", (0, 6, 12)), ("face", (12,))]
    assert tracks[1].boxes[0].src == "student"
    assert read_persons(out) == {0: 0, 6: 1, 12: 2}
    assert build_candidates(frames, out, finder, student, cfg) == 0  # nothing left


def test_boxes_of_a_frame_a_crash_never_marked_done_are_dropped_on_resume(tmp_path):
    frames, out, cfg = _frames(tmp_path, count=2), tmp_path / "gt", load_config()
    build_candidates(frames, out, _Finder(), {}, cfg, limit=1)
    with (out / BOXES).open("a") as f:  # frame 6's boxes written, then the crash before processed.txt
        f.write('{"video": "v.mp4", "frame": 6, "pts_ms": 100.0, "cls": "plate", "x1": 1, "y1": 1, "x2": 9, '
                '"y2": 5, "score": 0.5, "src": "teacher_owlv2"}\n{"video": "v.mp4", "fra')
    build_candidates(frames, out, _Finder(), {}, cfg)
    assert (out / PROCESSED).read_text().split("\n")[:2] == ["0 0", "6 0"]
    assert [t.boxes[-1].x1 for t in read_tracks(out / TRACKS)] == [100]  # one box per frame, no stray 1..9


def test_linkable_drops_plates_of_no_plate_shape():
    cfg = load_config()  # plates at least 1.2 times as wide as tall, at most 400 px wide
    assert linkable(Box("plate", 0, 0, 60, 20), cfg)
    assert not linkable(Box("plate", 0, 0, 20, 20), cfg)  # square: a lamp, a sign
    assert not linkable(Box("plate", 0, 0, 500, 200), cfg)  # the back of a bus
    assert linkable(Box("face", 0, 0, 20, 30), cfg)  # faces are not shape-checked


def test_relink_rebuilds_tracks_from_the_boxes_without_detecting_again(tmp_path):
    frames, out, cfg = _frames(tmp_path), tmp_path / "gt", load_config()
    build_candidates(frames, out, _Finder(), {}, cfg)
    with (out / BOXES).open("a") as f:  # a square "plate" on frame 0: dropped by the shape check when linking
        f.write('{"video": "v.mp4", "frame": 0, "pts_ms": 0.0, "cls": "plate", "x1": 10, "y1": 10, "x2": 30, '
                '"y2": 30, "score": 0.9, "src": "teacher_owlv2"}\n')
    assert relink(frames, out, cfg) == 1
    assert [t.frames for t in read_tracks(out / TRACKS)] == [(0, 6, 12)]
