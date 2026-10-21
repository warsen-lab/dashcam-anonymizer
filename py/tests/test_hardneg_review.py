import json
import re

import cv2
import numpy as np
import pytest

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.cli import main
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments
from vc_hardneg.pilot import STRATA, pilot_report, sample_pilot, stratum
from vc_hardneg.review import import_human, write_review_page


def _cand(cid, cls="plate", score=0.1, src="student"):
    return Candidate(id=cid, video="v.mp4", stem="v", frame=0, cls=cls, box=(110.0, 110.0, 130.0, 118.0), score=score,
                     src=src, track=1, dump_dir="d", dump_index=0, frame_teacher=0.0, image=f"crops/{cid}.png",
                     crop_origin=(100, 100))


def _crops(root, cands):
    (root / "crops").mkdir(exist_ok=True)
    for c in cands:
        cv2.imwrite(str(root / c.image), np.zeros((50, 50, 3), np.uint8))


def _page_data(text):
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S).group(1))


def test_the_page_lists_every_crop_and_carries_no_outside_resource(tmp_path):
    cands = [_cand("a"), _cand("b", cls="face", src="yunet")]
    _crops(tmp_path, cands)
    page = write_review_page(cands, tmp_path, tmp_path / "page", {"a": [Judgment("a", "no", "teacher")]}, "试判")
    text = page.read_text(encoding="utf-8")
    data = _page_data(text)
    assert [d["id"] for d in data["items"]] == ["a", "b"]
    assert data["items"][0]["verdicts"] == [{"judge": "teacher", "verdict": "no"}]
    assert data["items"][0]["box"] == pytest.approx([20.0, 20.0, 60.0, 36.0])  # percent of the 50 px crop
    assert (tmp_path / "page" / "img" / "a.png").is_file()
    assert "http://" not in text and "https://" not in text


def test_import_human_appends_known_ids_only(tmp_path):
    path = tmp_path / "j.jsonl"
    export = tmp_path / "human.json"
    export.write_text(json.dumps([{"id": "a", "verdict": "yes"}, {"id": "b", "verdict": "no"}]))
    assert import_human(path, export, {"a", "b"}) == 2
    assert [j.judge for j in read_judgments(path)] == [HUMAN, HUMAN]
    export.write_text(json.dumps([{"id": "zzz", "verdict": "yes"}]))
    with pytest.raises(ValueError):
        import_human(path, export, {"a"})
    assert len(read_judgments(path)) == 2


def test_pilot_sampling_is_stratified_and_reproducible():
    cands = [_cand(f"p{i}", score=0.05) for i in range(40)] + [_cand(f"q{i}", score=0.3) for i in range(10)] + \
            [_cand(f"y{i}", cls="face", src="yunet") for i in range(30)] + [_cand("s0", cls="face", src="student")] + \
            [_cand("both", cls="face", src="student+yunet", score=0.3)]
    a = sample_pilot(cands, 25, seed=0)
    assert a == sample_pilot(cands, 25, seed=0)
    counts = {s: sum(stratum(c) == s for c in a) for s in STRATA}
    assert counts == {"plate_lt02": 25, "plate_02_05": 10, "face_yunet": 25, "face_student": 1}
    assert stratum(cands[-1]) is None


def test_a_stratified_sample_spreads_the_total_over_the_strata_and_skips_the_pilot():
    from vc_hardneg.pilot import sample_stratified

    cands = [_cand(f"p{i}", score=0.05) for i in range(100)] + [_cand(f"q{i}", score=0.3) for i in range(100)] + \
            [_cand(f"y{i}", cls="face", src="yunet") for i in range(100)] + \
            [_cand(f"s{i}", cls="face", src="student") for i in range(10)] + \
            [_cand(f"b{i}", cls="face", src="student+yunet", score=0.3) for i in range(100)]
    picked = sample_stratified(cands, 200, seed=1, exclude={"p0", "p1"})
    assert picked == sample_stratified(cands, 200, seed=1, exclude={"p0", "p1"})
    by = {}
    for c in picked:
        by[stratum(c) or "face_both"] = by.get(stratum(c) or "face_both", 0) + 1
    assert sum(by.values()) == 200
    assert by["face_student"] == 10  # all there are; the rest goes to the other four evenly
    assert sorted(v for k, v in by.items() if k != "face_student") == [47, 47, 48, 48]
    assert not {"p0", "p1"} & {c.id for c in picked}


def test_pilot_report_gates_on_real_targets_judged_no():
    ids = [f"c{i}" for i in range(100)]
    judged = [Judgment(i, "no", "teacher") for i in ids[:60]] + [Judgment(i, "yes", "teacher") for i in ids[60:]]
    truth = [Judgment(i, "no", HUMAN) for i in ids[:59]] + [Judgment(ids[59], "yes", HUMAN)] + \
            [Judgment(i, "yes", HUMAN) for i in ids[60:]]
    r = pilot_report(ids, judged + truth)
    assert r["pipeline"]["no"] == 60 and r["pipeline"]["no_but_real"] == 1
    assert r["judges"]["teacher"]["confusion"]["yes"] == {"yes": 40, "no": 0}
    assert r["passed"] is True  # 1 / 60 = 1.7 % <= 2 %
    truth[0] = Judgment(ids[0], "yes", HUMAN)
    assert pilot_report(ids, judged + truth)["passed"] is False  # 2 / 60 = 3.3 %
    assert pilot_report(ids, judged + truth[:50])["passed"] is False  # people have not judged them all


def test_cli_review_and_import_round_trip(tmp_path):
    cands = [_cand("a"), _cand("b")]
    _crops(tmp_path, cands)
    write_candidates(tmp_path / "candidates.jsonl", cands)
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "unsure", "teacher"), Judgment("b", "no", "teacher")])
    assert main(["review", "--candidates", str(tmp_path / "candidates.jsonl"), "--judgments", str(tmp_path / "j.jsonl"),
                 "--out", str(tmp_path / "page"), "--only", "unsure"]) == 0
    data = _page_data((tmp_path / "page" / "index.html").read_text(encoding="utf-8"))
    assert [d["id"] for d in data["items"]] == ["a"]
    (tmp_path / "h.json").write_text(json.dumps([{"id": "a", "verdict": "yes"}]))
    assert main(["import-human", "--candidates", str(tmp_path / "candidates.jsonl"), "--judgments",
                 str(tmp_path / "j.jsonl"), "--file", str(tmp_path / "h.json")]) == 0
    assert read_judgments(tmp_path / "j.jsonl")[-1] == Judgment("a", "yes", HUMAN)


def test_cli_refuses_the_llm_without_the_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    write_candidates(tmp_path / "candidates.jsonl", [_cand("a")])
    code = main(["judge", "--candidates", str(tmp_path / "candidates.jsonl"), "--judgments", str(tmp_path / "j.jsonl"),
                 "--llm"])
    assert code == 2 and "VC_HARDNEG_ALLOW_LLM" in capsys.readouterr().err
    assert not (tmp_path / "j.jsonl").exists()


def test_cli_mine_labels(tmp_path):
    lab = tmp_path / "labels" / "s"
    lab.mkdir(parents=True)
    (lab / "processed.txt").write_text("0\n")
    (lab / "merged.jsonl").write_text(json.dumps({"video": "s.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": 10,
                                                  "y1": 10, "x2": 30, "y2": 20, "score": 0.2, "src": "t"}) + "\n")
    (tmp_path / "frames" / "s").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "frames" / "s" / "000000.jpg"), np.zeros((60, 80, 3), np.uint8))
    assert main(["mine-labels", "--stem", "s", "--labels-root", str(tmp_path / "labels"),
                 "--frames-root", str(tmp_path / "frames"), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "candidates.jsonl").read_text().count("\n") == 1
