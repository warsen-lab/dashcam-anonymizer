import sys

import cv2
import numpy as np
import pytest

from vc_hardneg.candidates import Candidate
from vc_hardneg.config import LlmConfig, load_config
from vc_hardneg.judges import LLM_ENV, CommandJudge, FrameTeacherJudge, TeacherJudge, run_judges, verdict_from_boxes
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments, resolve
from vc_label.contracts import Box

CFG = load_config()


def _cand(cid="a", cls="plate", frame_teacher=0.0, image="crops/a.png"):
    return Candidate(id=cid, video="v.mp4", stem="v", frame=0, cls=cls, box=(110.0, 110.0, 130.0, 118.0), score=0.1,
                     src="student", track=1, dump_dir="d", dump_index=0, frame_teacher=frame_teacher, image=image,
                     crop_origin=(100, 100))


def test_resolve_keeps_the_last_and_prefers_people(tmp_path):
    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("a", "no", "teacher"), Judgment("a", "yes", HUMAN), Judgment("a", "unsure", "teacher"),
                            Judgment("b", "no", "teacher"), Judgment("b", "unsure", "llm:codex")])
    r = resolve(read_judgments(path))
    assert r["a"].verdict == "yes" and r["a"].judge == HUMAN
    assert r["b"].verdict == "unsure"
    assert resolve(read_judgments(path), include_human=False)["a"].verdict == "unsure"
    with pytest.raises(ValueError):
        append_judgments(path, [Judgment("c", "maybe", "teacher")])
    assert len(read_judgments(path)) == 5  # the refused batch wrote nothing


def test_verdict_from_boxes_uses_the_class_thresholds():
    target = Box("plate", 10, 10, 30, 18)
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.5)], CFG)[0] == "yes"
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.2)], CFG)[0] == "unsure"
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.05)], CFG)[0] == "no"
    assert verdict_from_boxes(target, [Box("plate", 200, 200, 230, 210, score=0.9)], CFG)[0] == "no"  # elsewhere
    assert verdict_from_boxes(target, [], CFG) == ("no", 0.0)


def test_frame_teacher_says_yes_only_when_confident():
    assert FrameTeacherJudge(CFG).judge(_cand(frame_teacher=0.6)).verdict == "yes"
    assert FrameTeacherJudge(CFG).judge(_cand(frame_teacher=0.4)) is None


def test_teacher_judge_upscales_and_maps_the_box_back():
    seen = {}

    def plate_detect(img):
        seen["shape"] = img.shape
        return [Box("plate", 40, 40, 120, 72, score=0.9)]  # (110..130, 110..118) - origin 100, times 4

    judge = TeacherJudge(CFG, plate_detect, lambda img: [])
    j = judge.judge(_cand(), np.zeros((50, 50, 3), np.uint8))
    assert seen["shape"] == (200, 200, 3)
    assert j.verdict == "yes" and j.judge == "teacher"
    assert judge.judge(_cand(cls="face"), np.zeros((50, 50, 3), np.uint8)).verdict == "no"


def test_the_llm_judge_refuses_to_run_without_the_switch():
    cfg = LlmConfig(name="t", command=(sys.executable, "-c", "print(1)"), timeout_s=5, prompt="p")
    with pytest.raises(PermissionError):
        CommandJudge(cfg, env={})


def test_the_llm_judge_fills_placeholders_reads_the_last_json_and_falls_back_to_unsure(tmp_path):
    echo = LlmConfig("t", (sys.executable, "-c", "import sys; print(sys.argv[1:])", "{image}", "{prompt}"), 5,
                     'is it a {cls_zh}? answer {"verdict": ...}')
    j = CommandJudge(echo, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png")
    assert j.verdict == "unsure"  # the echo holds no verdict JSON; the braces in the prompt did not break the call
    ok = LlmConfig("t", (sys.executable, "-c", 'print("thinking..."); print(\'{"verdict": "no", "reason": "tail light"}\')'),
                   5, "p")
    k = CommandJudge(ok, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png")
    assert (k.verdict, k.detail, k.judge) == ("no", "tail light", "llm:t")
    slow = LlmConfig("t", (sys.executable, "-c", "import time; time.sleep(5)"), 0.2, "p")
    s = CommandJudge(slow, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png")
    assert s.verdict == "unsure" and "timeout" in s.detail


def test_run_judges_skips_judged_and_asks_the_llm_only_when_unsure(tmp_path):
    (tmp_path / "crops").mkdir()
    cands = [_cand("a", frame_teacher=0.9, image="crops/a.png"), _cand("b", image="crops/b.png"),
             _cand("c", image="crops/c.png")]
    for c in cands:
        cv2.imwrite(str(tmp_path / c.image), np.zeros((50, 50, 3), np.uint8))
    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("c", "no", HUMAN)])
    teacher = TeacherJudge(CFG, lambda img: [Box("plate", 40, 40, 120, 72, score=0.2)], lambda img: [])  # unsure
    asked = []

    class FakeLlm:
        def judge(self, c, image_path):
            asked.append(c.id)
            return Judgment(c.id, "no", "llm:fake")

    counts = run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), teacher, FakeLlm())
    assert asked == ["b"]
    r = resolve(read_judgments(path))
    assert r["a"].judge == "frame_teacher" and r["b"].verdict == "no" and r["c"].judge == HUMAN
    assert counts["frame_teacher:yes"] == 1 and counts["teacher:unsure"] == 1 and counts["llm:fake:no"] == 1
    assert run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), teacher) == {}  # all judged: nothing to do
