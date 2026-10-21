import json

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.judgments import Judgment, append_judgments
from vc_label.contracts import Box
from vc_train.cleaning import LabelCleaning


def _cand(cid, frame, box, cls="plate", score=0.15):
    return Candidate(id=cid, video="s.mp4", stem="s", frame=frame, cls=cls, box=box, score=score, src="teacher_owlv2",
                     track=-1, dump_dir="", dump_index=-1, frame_teacher=0.0, image="", crop_origin=(0, 0))


def _setup(tmp_path):
    """merged.jsonl on frame 0: low-band plates a (judged no), b (judged yes), c (tall, never judged), d (tall,
    judged yes), a confident plate and a low-band face judged no; the labels as read back from yolo/ (rounded)."""
    lab = tmp_path / "labels" / "s"
    lab.mkdir(parents=True)

    def row(cls, x1, y1, x2, y2, score):
        return {"video": "s.mp4", "frame": 0, "pts_ms": 0, "cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": score, "src": "t"}

    rows = [row("plate", 10, 10, 30, 20, 0.15), row("plate", 40, 10, 60, 20, 0.2), row("plate", 70, 10, 80, 30, 0.25),
            row("plate", 90, 10, 100, 30, 0.25), row("plate", 120, 10, 140, 20, 0.8), row("face", 10, 50, 30, 75, 0.4)]
    (lab / "merged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_candidates(tmp_path / "c.jsonl", [_cand("a", 0, (10, 10, 30, 20)), _cand("b", 0, (40, 10, 60, 20)),
                                            _cand("d", 0, (90, 10, 100, 30), score=0.25),
                                            _cand("f", 0, (10, 50, 30, 75), cls="face", score=0.4)])
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "no", "llm:codex"), Judgment("b", "yes", "teacher"),
                                            Judgment("d", "yes", "human"), Judgment("f", "no", "llm:codex")])
    return [Box("plate", 10.02, 10, 30, 20), Box("plate", 40, 10, 60, 20), Box("plate", 70, 10, 80, 30),
            Box("plate", 90, 10, 100, 30), Box("plate", 120, 10, 140, 20), Box("face", 10, 50, 30, 75)]


def _load(tmp_path, tall):
    return LabelCleaning.load(tmp_path / "c.jsonl", tmp_path / "j.jsonl", tmp_path / "labels", ["s"],
                              drop_tall_low_plates=tall)


def test_plates_judged_not_real_are_dropped_and_faces_are_never_touched(tmp_path):
    labels = _setup(tmp_path)
    kept, dropped = _load(tmp_path, tall=False).clean("s", 0, labels)
    assert dropped == 1  # plate a; the face judged no stays
    assert [(b.cls, round(b.x1)) for b in kept] == [("plate", 40), ("plate", 70), ("plate", 90), ("plate", 120), ("face", 10)]
    assert _load(tmp_path, tall=False).clean("s", 12, labels) == (labels, 0)  # nothing known about frame 12


def test_tall_low_band_plates_are_dropped_unless_judged_real(tmp_path):
    labels = _setup(tmp_path)
    kept, dropped = _load(tmp_path, tall=True).clean("s", 0, labels)
    # a: judged no; c (10 x 20, low band, never judged): tall; d: tall but judged real; the 0.8 plate is not low band
    assert dropped == 2
    assert [(b.cls, round(b.x1)) for b in kept] == [("plate", 40), ("plate", 90), ("plate", 120), ("face", 10)]


def test_unsure_plates_stay(tmp_path):
    labels = _setup(tmp_path)
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "unsure", "human")])
    assert _load(tmp_path, tall=False).clean("s", 0, labels)[1] == 0


def test_hard_negatives_are_the_candidates_judged_not_real(tmp_path):
    from vc_train.cleaning import HardNegative, load_hard_negatives

    write_candidates(tmp_path / "c.jsonl", [_cand("b", 12, (40, 10, 60, 20)), _cand("a", 0, (10, 10, 30, 20)),
                                            _cand("c", 0, (70, 10, 90, 20))])
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "no", "llm:codex"), Judgment("b", "no", "teacher"),
                                            Judgment("c", "yes", "teacher")])
    assert load_hard_negatives(tmp_path / "c.jsonl", tmp_path / "j.jsonl") == [
        HardNegative("s", 0, Box("plate", 10, 10, 30, 20)), HardNegative("s", 12, Box("plate", 40, 10, 60, 20))]


def test_the_hard_negative_window_is_centred_and_kept_inside_the_frame():
    from vc_train.cleaning import hard_negative_window

    assert hard_negative_window(Box("plate", 900, 500, 920, 510), 1920, 1080) == (590, 185, 640, 640)
    assert hard_negative_window(Box("plate", 0, 0, 20, 10), 1920, 1080) == (0, 0, 640, 640)
    assert hard_negative_window(Box("plate", 1900, 1070, 1920, 1080), 1920, 1080) == (1280, 440, 640, 640)
    assert hard_negative_window(Box("plate", 10, 10, 20, 20), 300, 200) == (0, 0, 300, 200)
