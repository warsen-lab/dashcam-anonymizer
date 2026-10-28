import pytest

from vc_eval.metrics import Rate
from vc_eval.objects import ClassResult, Leak, clopper_pearson, evaluate_objects, find_leaks, model_error_rate
from vc_gt.audit import AuditBox
from vc_gt.tracks import Track
from vc_hardneg.judgments import Judgment
from vc_label.contracts import Box


def test_clopper_pearson_matches_known_values():
    assert clopper_pearson(0, 3000)[0] == 0.0
    assert clopper_pearson(0, 3000)[1] == pytest.approx(0.0012289, abs=2e-6)  # 1 - 0.025 ** (1 / 3000)
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == (pytest.approx(0.187086, abs=1e-5), pytest.approx(0.812914, abs=1e-5))
    assert clopper_pearson(10, 10)[1] == 1.0 and clopper_pearson(0, 0) == (0.0, 1.0)


def plate(x, w=40):
    return Box("plate", x, 500, x + w, 512)


def test_leaks_readable_objects_and_frames_without_a_leak():
    tracks = [Track(0, "plate", (0, 6, 12), (plate(100), plate(100), plate(100))),  # real, frame 6 unmasked, readable
              Track(1, "plate", (0, 6), (plate(300), plate(300))),  # real_unreadable, all masked
              Track(2, "plate", (0,), (plate(600),)),  # not real
              Track(3, "plate", (0,), (plate(800, w=10),))]  # real_unreadable, too small to demand, unmasked
    objects = {"o0": "real_readable", "o1": "real_unreadable", "o2": "not_real", "o3": "real_unreadable"}
    frames = {"o0@6": "readable", "o3@0": "unreadable"}

    def covered(x):
        return [Box("plate", x - 5, 495, x + 45, 517)]

    masks = {0: covered(100) + covered(300), 6: covered(300), 12: covered(100)}
    r = evaluate_objects(tracks, objects, frames, masks, 1080)["plate"]
    assert r.readable == 1 and r.leaks == (Leak(0, (6,)),)
    assert (r.no_leak.hits, r.no_leak.total) == (0, 1)
    assert (r.frame_no_leak.hits, r.frame_no_leak.total) == (4, 5)  # tracks 0 and 1; track 3 is under 24 px wide
    assert r.missed_by_all is None and not r.passed


def test_an_unmasked_frame_under_must_mask_size_is_no_leak_even_if_judged_readable():
    tracks = [Track(0, "plate", (0, 6), (plate(100), plate(100, w=20)))]  # frame 6: 20 px, under 24
    r = evaluate_objects(tracks, {"o0": "real_unreadable"}, {"o0@6": "readable"}, {0: [plate(100)]}, 1080)["plate"]
    assert r.leaks == () and r.readable == 0


def test_a_frame_whose_box_holds_no_plate_is_no_leak():
    tracks = [Track(0, "plate", (0, 6), (plate(100), plate(100)))]
    r = evaluate_objects(tracks, {"o0": "real_readable"}, {"o0@6": "not_target"}, {0: [plate(100)]}, 1080)["plate"]
    assert r.leaks == () and (r.frame_no_leak.hits, r.frame_no_leak.total) == (2, 2)


def test_missed_by_all_counts_audit_boxes_no_judged_track_holds():
    tracks = [Track(0, "plate", (6,), (plate(100),)), Track(1, "plate", (6,), (plate(500),))]  # o1 never judged
    audit = [AuditBox(6, plate(102)), AuditBox(6, plate(900)), AuditBox(6, plate(502))]
    r = evaluate_objects(tracks, {"o0": "real_readable"}, {}, {6: [plate(100)]}, 1080, audit)["plate"]
    assert (r.missed_by_all.hits, r.missed_by_all.total) == (2, 3)  # 900: no track; 502: only an unjudged one


def test_passing_needs_every_bar():
    assert ClassResult("plate", 3000, (), Rate(995, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 2999, (), Rate(995, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, tuple(Leak(k, (0,)) for k in range(4)), Rate(995, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(994, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(995, 1000), Rate(2, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(995, 1000), None).passed


def test_model_error_rate_compares_people_with_the_last_model_verdict_on_the_sample():
    js = [Judgment("o1", "not_real", "llm:x"), Judgment("o1", "real_readable", "human"),
          Judgment("o2", "real_readable", "llm:x"), Judgment("o2", "real_readable", "human"),
          Judgment("o3", "not_real", "llm:x")]  # o3 not judged by people yet
    r = model_error_rate(js, {"o1", "o2", "o3"})
    assert (r.hits, r.total) == (1, 2)


def test_the_report_lists_the_table_the_error_rates_and_each_leak_with_its_crop(tmp_path):
    import cv2
    import numpy as np

    from vc_eval.objects import write_report

    image = tmp_path / "000006.jpg"
    cv2.imwrite(str(image), np.zeros((1080, 1920, 3), np.uint8))
    t = Track(0, "plate", (6,), (plate(100),))
    results = {"plate": ClassResult("plate", 1, (Leak(0, (6,)),), Rate(0, 1), None),
               "face": ClassResult("face", 0, (), Rate(), None)}
    page = write_report(results, {"问题 A": Rate(1, 50)}, tmp_path / "out", [t], {6: image}, {})
    text = page.read_text(encoding="utf-8")
    assert "不下结论（差 2999 个目标）" in text and "未抽查" in text and "问题 A：2.00%" in text
    assert "leaks/o0@000006.jpg" in text and (tmp_path / "out" / "leaks" / "o0@000006.jpg").exists()


def test_find_leaks_needs_no_question_a():
    tracks = [Track(0, "plate", (0, 6), (plate(100), plate(100))), Track(1, "plate", (6,), (plate(500, w=10),)),
              Track(2, "face", (6,), (Box("face", 10, 10, 40, 50),))]
    frames = {"o0@6": "readable", "o1@6": "readable", "o2@6": "not_target"}
    leaks = find_leaks(tracks, frames, {0: [plate(100)]}, 1080)
    assert leaks == {"plate": (Leak(0, (6,)),), "face": ()}  # o1 is under must-mask size, o2 is no face
