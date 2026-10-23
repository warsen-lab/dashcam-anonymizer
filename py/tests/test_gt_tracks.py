from vc_gt.tracks import TrackParams, link_tracks, read_tracks, write_tracks
from vc_label.contracts import Box


def plate(x, y=500, w=40, h=12, score=0.9):
    return Box("plate", x, y, x + w, y + h, score)


def test_a_box_moving_a_little_each_frame_is_one_track():
    by_frame = {f: [plate(100 + 5 * k)] for k, f in enumerate(range(0, 60, 6))}
    tracks = link_tracks(by_frame, sorted(by_frame))
    assert len(tracks) == 1 and tracks[0].frames == tuple(range(0, 60, 6))


def test_two_plates_side_by_side_stay_apart():
    by_frame = {f: [plate(100 + 3 * k), plate(900 - 3 * k)] for k, f in enumerate(range(0, 30, 6))}
    tracks = link_tracks(by_frame, sorted(by_frame))
    assert [t.boxes[0].x1 for t in tracks] == [100, 900] and all(len(t.frames) == 5 for t in tracks)


def test_a_gap_of_max_gap_frames_continues_and_one_more_splits():
    order = list(range(10))
    by_frame = {0: [plate(100)], 4: [plate(102)]}  # frames 1, 2, 3 missed: a gap of 3
    assert len(link_tracks(by_frame, order, TrackParams(max_gap=3))) == 1
    assert len(link_tracks(by_frame, order, TrackParams(max_gap=2))) == 2


def test_boxes_that_do_not_overlap_enough_never_link():
    # the 2026-10-23 pilot: linking by centre distance let tracks hop from a plate to a tail light or a barrier
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(115, w=20)]}, [0, 1])) == 2  # IoU 0.14
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(104, w=20)]}, [0, 1])) == 1  # IoU 0.67


def test_a_box_more_than_twice_as_wide_or_narrow_starts_a_new_track():
    by_frame = {0: [Box("plate", 100, 500, 300, 560, 0.9)], 1: [Box("plate", 100, 500, 199, 530, 0.9)]}  # IoU 0.25
    assert len(link_tracks(by_frame, [0, 1], TrackParams(min_iou=0.2))) == 2  # 99 px is under half of 200 px
    by_frame = {0: [Box("plate", 100, 500, 300, 560, 0.9)], 1: [Box("plate", 100, 500, 201, 530, 0.9)]}
    assert len(link_tracks(by_frame, [0, 1], TrackParams(min_iou=0.2))) == 1


def test_the_better_overlap_wins():
    by_frame = {0: [plate(100)], 1: [plate(110), plate(102)]}
    tracks = link_tracks(by_frame, [0, 1])
    assert [b.x1 for b in tracks[0].boxes] == [100, 102]


def test_classes_never_share_a_track():
    face = Box("face", 100, 500, 140, 512, 0.9)
    assert sorted(t.cls for t in link_tracks({0: [plate(100)], 1: [face]}, [0, 1])) == ["face", "plate"]


def test_ids_follow_the_first_frame_and_survive_a_round_trip(tmp_path):
    tracks = link_tracks({0: [plate(500)], 6: [plate(502), plate(100)]}, [0, 6])
    assert [(t.id, t.frames[0], t.boxes[0].x1) for t in tracks] == [(0, 0, 500), (1, 6, 100)]
    write_tracks(tmp_path / "t.jsonl", "v.mp4", tracks)
    back = read_tracks(tmp_path / "t.jsonl")
    assert back == tracks and back[0].box_at(6).x1 == 502 and back[0].box_at(3) is None
