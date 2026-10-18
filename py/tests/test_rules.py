from pathlib import Path

from vc_label.config import LabelingConfig
from vc_label.contracts import Box, FrameRef
from vc_label.rules import flag_frame, temporal_fill

CFG = LabelingConfig()
W, H = 1920, 1080
BOTH = "teacher_retinaface+teacher_yunet"


def ref(frame: int, pts_ms: float) -> FrameRef:
    return FrameRef("trip.mp4", frame, pts_ms, Path(f"{frame:06d}.jpg"))


def rules(merged, aux=()):
    return sorted(f.rule for f in flag_frame(ref(1, 0.0), list(merged), list(aux), W, H, CFG))


def test_face_seen_by_both_teachers_is_not_flagged():
    assert rules([Box("face", 100, 100, 130, 140, score=0.9, src=BOTH)]) == []


def test_face_seen_by_one_teacher_is_flagged():
    assert rules([Box("face", 100, 100, 130, 140, score=0.9, src="teacher_yunet")]) == ["face_single_source"]


def test_low_score_bands_differ_per_class():
    assert rules([Box("face", 0, 0, 20, 20, score=0.4, src=BOTH)]) == ["low_score"]
    assert rules([Box("plate", 0, 0, 20, 8, score=0.2, src="teacher_owlv2")]) == ["low_score"]
    assert rules([Box("plate", 0, 0, 20, 8, score=0.4, src="teacher_owlv2")]) == []


def test_large_person_needs_a_face_in_the_head_region():
    person = Box("person", 200, 500, 300, 900, src="teacher_coco:person")  # 400 px >= 12 % of 1080
    head_face = Box("face", 240, 510, 260, 540, score=0.9, src=BOTH)
    belly_face = Box("face", 240, 800, 260, 830, score=0.9, src=BOTH)
    assert rules([], [person]) == ["person_without_face"]
    assert rules([head_face], [person]) == []
    assert rules([belly_face], [person]) == ["person_without_face"]


def test_small_person_is_not_flagged():
    assert rules([], [Box("person", 0, 0, 20, 100)]) == []


def test_large_vehicle_without_plate_is_flagged_once():
    car = Box("vehicle", 1000, 600, 1300, 800, score=0.99, src="teacher_coco:car")
    truck = Box("vehicle", 1005, 600, 1300, 805, score=0.8, src="teacher_coco:truck")
    plate = Box("plate", 1100, 750, 1160, 765, score=0.6, src="teacher_owlv2")
    assert rules([], [car, truck]) == ["vehicle_without_plate"]
    assert rules([plate], [car, truck]) == []
    assert rules([], [Box("vehicle", 0, 0, 50, 40)]) == []


def test_temporal_fill_interpolates_a_missing_box():
    prev = Box("plate", 100, 100, 160, 120, score=0.6, src="teacher_owlv2")
    nxt = Box("plate", 110, 100, 170, 120, score=0.5, src="teacher_owlv2")
    sequence = [(ref(0, 0.0), [prev]), (ref(12, 200.0), []), (ref(24, 400.0), [nxt])]

    filled, flags = temporal_fill(sequence, CFG)

    [box] = filled[1]
    assert (box.x1, box.x2, box.score, box.src) == (105, 165, 0.5, "temporal_fill")
    assert [(f.frame, f.rule) for f in flags] == [(12, "temporal_fill")]
    assert filled[0] == [prev] and filled[2] == [nxt]


def test_temporal_fill_skips_present_boxes_long_gaps_and_unmatched_boxes():
    a = Box("face", 100, 100, 130, 140, score=0.9, src=BOTH)
    far = Box("face", 900, 100, 930, 140, score=0.9, src=BOTH)
    present = [(ref(0, 0.0), [a]), (ref(12, 200.0), [a]), (ref(24, 400.0), [a])]
    long_gap = [(ref(0, 0.0), [a]), (ref(60, 1000.0), []), (ref(120, 2000.0), [a])]
    unmatched = [(ref(0, 0.0), [a]), (ref(12, 200.0), []), (ref(24, 400.0), [far])]
    for sequence in (present, long_gap, unmatched):
        _, flags = temporal_fill(sequence, CFG)
        assert flags == []


def test_temporal_fill_ignores_backward_or_equal_timestamps():
    p = Box("plate", 100, 100, 160, 120, score=0.6, src="teacher_owlv2")
    n = Box("plate", 110, 100, 170, 120, score=0.5, src="teacher_owlv2")
    equal = [(ref(0, 400.0), [p]), (ref(12, 100.0), []), (ref(24, 400.0), [n])]
    backward = [(ref(0, 400.0), [p]), (ref(12, 100.0), []), (ref(24, 300.0), [n])]
    for sequence in (equal, backward):
        filled, flags = temporal_fill(sequence, CFG)
        assert flags == []
        assert filled[1] == []


def test_filled_boxes_do_not_satisfy_or_trigger_other_rules():
    person = Box("person", 200, 500, 300, 900, src="teacher_coco:person")
    filled_face = Box("face", 240, 510, 260, 540, score=0.4, src="temporal_fill")
    assert rules([filled_face], [person]) == ["person_without_face"]

    car = Box("vehicle", 1000, 600, 1300, 800, score=0.99, src="teacher_coco:car")
    filled_plate = Box("plate", 1100, 750, 1160, 765, score=0.2, src="temporal_fill")
    assert rules([filled_plate], [car]) == ["vehicle_without_plate"]
