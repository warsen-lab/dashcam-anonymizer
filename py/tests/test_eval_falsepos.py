import json

from vc_eval.falsepos import count_false_positives


def _det(frame, box, cls="plate", src="student", confirmed=True, verified=None):
    d = {"video": "v.mp4", "frame": frame, "pts_ms": 0, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
         "y2": box[3], "score": 0.1, "src": src, "confirmed": confirmed}
    if verified is not None:
        d["verified"] = verified
    return d


def test_counts_masked_boxes_without_a_teacher_box(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n48\n")
    teacher = [
        {"video": "v.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": 10, "y1": 10, "x2": 30, "y2": 20, "score": 0.15, "src": "t"},
        {"video": "v.mp4", "frame": 48, "pts_ms": 0, "cls": "plate", "x1": 100, "y1": 100, "x2": 120, "y2": 110, "score": 0.9, "src": "t"},
    ]
    (labels / "merged.jsonl").write_text("".join(json.dumps(t) + "\n" for t in teacher))
    (labels / "excluded_refs.csv").write_text("frame,cls,x1,y1,x2,y2,reason\n48,plate,100,100,120,110,尾灯\n")
    dets = tmp_path / "d.jsonl"
    dets.write_text("".join(json.dumps(d) + "\n" for d in [
        _det(0, (11, 10, 30, 20)),                        # referenced
        _det(0, (200, 200, 220, 210)),                    # not referenced
        _det(0, (300, 300, 320, 310), confirmed=False),   # never masked: not counted
        _det(0, (400, 400, 420, 410), verified=False),    # rejected by the verifier: not counted
        _det(48, (100, 100, 120, 110), src="yunet", cls="plate"),  # on an excluded teacher box: not referenced
        _det(12, (0, 0, 5, 5)),                           # frame not labelled: not counted
    ]))
    r = count_false_positives(dets, labels)
    assert r["plate"]["masked"] == 3 and r["plate"]["unreferenced"] == 2
    assert r["plate"]["by_src"] == {"student": [2, 1], "yunet": [1, 1]}
    assert r["face"] == {"masked": 0, "unreferenced": 0, "by_src": {}}


def test_temporal_score_bands_are_the_2026_10_19_reports():
    from vc_eval.temporal import score_bin

    assert [score_bin("face", s) for s in (0.9, 0.8, 0.79, 0.5, 0.49)] == \
        ["face>=0.8", "face>=0.8", "face0.5-0.8", "face0.5-0.8", "face<0.5"]
    assert [score_bin("plate", s) for s in (0.5, 0.49, 0.2, 0.19)] == ["plate>=0.5", "plate0.2-0.5", "plate0.2-0.5", "plate<0.2"]


def test_new_misses_are_the_reference_boxes_only_the_candidate_misses(tmp_path, monkeypatch):
    import vc_eval.newmisses as nm
    from vc_eval.coverage import Miss
    from vc_label.contracts import Box

    a, b, c = (Miss(0, Box("plate", 0, 0, 10, 5), 0.0), Miss(12, Box("face", 5, 5, 9, 9), 0.2),
               Miss(24, Box("plate", 50, 50, 70, 60), 0.1))

    class Report:
        def __init__(self, misses):
            self.misses, self.uncertain_misses = misses, []

    reports = {"base": Report([a, b]), "cand": Report([a, b, c])}
    monkeypatch.setattr(nm, "evaluate_coverage", lambda frames, labels, masks, targets: reports[str(masks)])
    assert nm.new_misses(tmp_path, tmp_path, "base", "cand", None) == [c]
