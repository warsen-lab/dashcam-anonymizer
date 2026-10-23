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


def test_a_fast_small_plate_links_by_centre_distance():
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(125, w=20)]}, [0, 1])) == 1  # 25 px < 1.5 x 20
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(140, w=20)]}, [0, 1])) == 2  # 40 px > 30


def test_an_overlapping_box_wins_over_a_merely_close_one():
    by_frame = {0: [plate(100)], 1: [plate(130), plate(102)]}  # 102 overlaps, 130 is only close
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
