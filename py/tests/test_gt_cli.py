import csv
import json

import cv2
import numpy as np

from vc_gt.audit import AUDIT, AUDIT_FRAMES, H2_SAMPLE
from vc_gt.candidates import PROCESSED, TRACKS
from vc_gt.cli import main
from vc_gt.config import load_config
from vc_gt.crops import CROPS_INDEX, append_crop_index, read_crop_index, write_object_crops
from vc_gt.judge import FRAME_VERDICTS, FRAMES, OBJECT_VERDICTS, OBJECTS
from vc_gt.tracks import Track, write_tracks
from vc_hardneg.judgments import Judgment, append_judgments, read_judgments
from vc_label.contracts import Box


def _setup(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    with (frames / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for fr in (0, 6):
            w.writerow(["v.mp4", fr, 0.0, "base", 0.0])
            cv2.imwrite(str(frames / f"{fr:06d}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    gt = tmp_path / "gt"
    gt.mkdir()
    box = Box("plate", 100, 100, 140, 112)
    track = Track(0, "plate", (0, 6), (box, box))
    write_tracks(gt / TRACKS, "v.mp4", [track])
    (gt / PROCESSED).write_text("0 0\n6 1\n")
    images = {f: frames / f"{f:06d}.jpg" for f in (0, 6)}
    append_crop_index(gt / CROPS_INDEX, "o0", write_object_crops(track, images, gt / "crops" / "objects", load_config()))
    masks = tmp_path / "masks.jsonl"  # frame 0 masks the plate, frame 6 does not
    masks.write_text(json.dumps({"video": "v.mp4", "frame": 0, "boxes": [
        {"cls": "plate", "x1": 95, "y1": 95, "x2": 145, "y2": 117, "origin": "detected"}]}) + "\n"
        + json.dumps({"video": "v.mp4", "frame": 6, "boxes": []}) + "\n")
    return frames, gt, masks


def test_judge_refuses_without_the_llm_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    frames, gt, masks = _setup(tmp_path)
    assert main(["judge", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks)]) == 2
    assert "VC_HARDNEG_ALLOW_LLM" in capsys.readouterr().err


def test_judge_frames_needs_the_redacted_frames(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VC_HARDNEG_ALLOW_LLM", "1")
    frames, gt, masks = _setup(tmp_path)
    assert main(["judge", "--stage", "frames", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks)]) == 2
    assert "--redacted-frames" in capsys.readouterr().err


def test_review_h1_then_import_people_verdicts(tmp_path):
    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "unsure", "llm:x")], verdicts=OBJECT_VERDICTS)
    assert main(["review", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks), "--set", "h1",
                 "--out", str(tmp_path / "page")]) == 0
    assert '"o0"' in (tmp_path / "page" / "objects" / "index.html").read_text(encoding="utf-8")
    export = tmp_path / "e.json"
    export.write_text(json.dumps([{"id": "o0", "verdict": "real_readable"}]))
    assert main(["import-human", "--gt", str(gt), "--file", str(export), "--kind", "objects"]) == 0
    assert read_judgments(gt / OBJECTS)[-1] == Judgment("o0", "real_readable", "human")
    export.write_text(json.dumps([{"id": "o0", "verdict": "readable"}]))
    assert main(["import-human", "--gt", str(gt), "--file", str(export), "--kind", "objects"]) == 2


def test_review_h2_keeps_its_sample(tmp_path):
    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "real_readable", "llm:x")], verdicts=OBJECT_VERDICTS)
    args = ["review", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks), "--set", "h2",
            "--out", str(tmp_path / "page")]
    assert main(args) == 0
    first = json.loads((gt / H2_SAMPLE).read_text())
    assert first["objects"] == ["o0"] and main(args) == 0 and json.loads((gt / H2_SAMPLE).read_text()) == first


def test_annotate_then_import_the_audit(tmp_path):
    frames, gt, _ = _setup(tmp_path)
    assert main(["annotate", "--gt", str(gt), "--frames", str(frames), "--out", str(tmp_path / "ann")]) == 0
    assert json.loads((gt / AUDIT_FRAMES).read_text()) == {"plate": [0, 6], "face": [6]}
    export = tmp_path / "a.json"
    export.write_text(json.dumps([{"key": "plate:0", "done": True, "boxes": [[1, 2, 30, 12]]}]))
    assert main(["import-human", "--gt", str(gt), "--file", str(export), "--kind", "audit"]) == 0
    assert json.loads((gt / AUDIT).read_text().splitlines()[0])["frame"] == 0


def test_vc_eval_objects_reports_a_leak_from_the_answer_key(tmp_path, capsys):
    from vc_eval.cli import main as eval_main

    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "real_readable", "llm:x")], verdicts=OBJECT_VERDICTS)
    append_judgments(gt / FRAMES, [Judgment("o0@6", "readable", "llm:x")], verdicts=FRAME_VERDICTS)
    assert eval_main(["objects", "--gt", str(gt), "--frames", str(frames), "--redacted-frames", str(frames),
                      "--masks", str(masks), "--out", str(tmp_path / "report")]) == 0
    assert "plate: 1 readable real objects, 1 leaked, frames without a leak 1/2" in capsys.readouterr().out
    assert (tmp_path / "report" / "leaks" / "o0@000006.jpg").exists()


def test_candidates_on_a_finished_segment_only_relinks(tmp_path, capsys):
    from vc_gt.candidates import BOXES
    from vc_gt.tracks import read_tracks

    frames, gt, _ = _setup(tmp_path)
    (gt / BOXES).write_text("".join(json.dumps({"video": "v.mp4", "frame": f, "pts_ms": 0.0, "cls": "plate", "x1": 100,
                                                "y1": 100, "x2": 140, "y2": 112, "score": 0.5, "src": "student"}) + "\n"
                                    for f in (0, 6)))
    assert main(["candidates", "--frames", str(frames), "--out", str(gt),
                 "--vc-detect", str(tmp_path / "no_vc_detect")]) == 0  # neither vc_detect nor a teacher is loaded
    assert "relinked: 1 tracks" in capsys.readouterr().out
    assert [t.frames for t in read_tracks(gt / TRACKS)] == [(0, 6)]


def test_calibrate_refuses_without_the_llm_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    _, gt, _ = _setup(tmp_path)
    assert main(["calibrate", "--gt", str(gt), "--sample", "10", "--out", str(tmp_path / "c")]) == 2
    assert "VC_HARDNEG_ALLOW_LLM" in capsys.readouterr().err and not (tmp_path / "c").exists()


def test_h1_shows_a_readable_frame_even_of_an_object_question_a_called_not_real(tmp_path):
    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "not_real", "llm:x")], verdicts=OBJECT_VERDICTS)
    append_judgments(gt / FRAMES, [Judgment("o0@6", "readable", "llm:x")], verdicts=FRAME_VERDICTS)
    append_crop_index(gt / CROPS_INDEX, "o0@6", read_crop_index(gt / CROPS_INDEX)["o0"][:1])
    assert main(["review", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks), "--set", "h1",
                 "--out", str(tmp_path / "page")]) == 0
    assert '"o0@6"' in (tmp_path / "page" / "frames" / "index.html").read_text(encoding="utf-8")


def test_vc_eval_objects_lists_leaks_of_any_track(tmp_path, capsys):
    from vc_eval.cli import main as eval_main

    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "not_real", "llm:x")], verdicts=OBJECT_VERDICTS)
    append_judgments(gt / FRAMES, [Judgment("o0@6", "readable", "human")], verdicts=FRAME_VERDICTS)
    assert eval_main(["objects", "--gt", str(gt), "--frames", str(frames), "--redacted-frames", str(frames),
                      "--masks", str(masks), "--out", str(tmp_path / "report")]) == 0
    assert "leaks found on any candidate: plate 1, face 0" in capsys.readouterr().out
    assert "o0 第 6 帧" in (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
