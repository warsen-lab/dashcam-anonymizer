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


def test_a_box_holding_the_teachers_plate_is_real():
    # a mask over a whole bumper: the teacher's plate is small inside it, IoU 0.07, but the mask covers it
    bumper = Box("plate", 0, 0, 120, 60)
    plate = Box("plate", 50, 40, 70, 50, score=0.5)
    assert verdict_from_boxes(bumper, [plate], CFG)[0] == "yes"
    assert verdict_from_boxes(bumper, [Box("plate", 50, 40, 70, 50, score=0.2)], CFG)[0] == "unsure"


def test_a_tiny_box_is_never_judged_not_real():
    # tiny_px is the long side in the original frame; the target here is already upscaled (x4): 15 / 4 < 16 px
    tiny = Box("face", 10, 10, 25, 22)
    assert verdict_from_boxes(tiny, [], CFG, scale=4.0) == ("unsure", 0.0)
    assert verdict_from_boxes(Box("face", 10, 10, 80, 90), [], CFG, scale=4.0) == ("no", 0.0)


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


def _crop(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((50, 50, 3), np.uint8))
    return path


def test_the_llm_judge_sees_the_box_drawn_and_enlarged(tmp_path):
    seen = tmp_path / "seen.txt"
    show = LlmConfig("t", (sys.executable, "-c", f"import sys; open({str(seen)!r}, 'w').write(sys.argv[1])", "{image}"),
                     5, "p")
    CommandJudge(show, env={LLM_ENV: "1"}).judge(_cand(), _crop(tmp_path))
    image = cv2.imread(seen.read_text())
    assert image.shape[0] >= 256  # enlarged from 50 px
    s = image.shape[0] / 50
    assert image[int(10 * s), int(20 * s), 2] > 200  # red on the box's top edge, (110,110)-(130,118) in the crop at 100
    assert image[int(40 * s), int(40 * s)].max() == 0  # the rest is the crop as it was


def test_the_llm_judge_fills_placeholders_reads_the_last_json_and_falls_back_to_unsure(tmp_path):
    crop = _crop(tmp_path)
    echo = LlmConfig("t", (sys.executable, "-c", "import sys; print(sys.argv[1:])", "{image}", "{prompt}"), 5,
                     'is it a {cls_zh}? answer {"verdict": ...}')
    j = CommandJudge(echo, env={LLM_ENV: "1"}).judge(_cand(), crop)
    assert j.verdict == "unsure"  # the echo holds no verdict JSON; the braces in the prompt did not break the call
    ok = LlmConfig("t", (sys.executable, "-c", 'print("thinking..."); print(\'{"verdict": "no", "reason": "tail light"}\')'),
                   5, "p")
    k = CommandJudge(ok, env={LLM_ENV: "1"}).judge(_cand(), crop)
    assert (k.verdict, k.detail, k.judge) == ("no", "tail light", "llm:t")
    slow = LlmConfig("t", (sys.executable, "-c", "import time; time.sleep(5)"), 0.2, "p")
    s = CommandJudge(slow, env={LLM_ENV: "1"}).judge(_cand(), crop)
    assert s.verdict == "unsure" and "timeout" in s.detail


def test_run_judges_skips_judged_and_asks_the_llm_about_no_and_unsure(tmp_path):
    (tmp_path / "crops").mkdir()
    cands = [_cand("a", frame_teacher=0.9, image="crops/a.png"), _cand("b", image="crops/b.png"),
             _cand("c", image="crops/c.png"), _cand("d", cls="face", image="crops/d.png")]
    for c in cands:
        cv2.imwrite(str(tmp_path / c.image), np.zeros((50, 50, 3), np.uint8))
    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("c", "no", HUMAN)])
    # plates: unsure; faces: nothing found, a 70 x 80 px target (x4 of the 20 x 8 box... large enough): no
    teacher = TeacherJudge(CFG, lambda img: [Box("plate", 40, 40, 120, 72, score=0.2)], lambda img: [])
    asked = []

    class FakeLlm:
        def judge_batch(self, cs, paths):
            asked.extend(c.id for c in cs)
            return [Judgment(c.id, "no" if c.id == "b" else "yes", "llm:fake") for c in cs]

    counts = run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), teacher, FakeLlm())
    assert sorted(asked) == ["b", "d"]
    r = resolve(read_judgments(path))
    assert r["a"].judge == "frame_teacher" and r["c"].judge == HUMAN
    assert r["b"].verdict == "no"  # teacher unsure, LLM no
    assert r["d"].verdict == "yes"  # teacher no, LLM yes: the LLM's word stands
    assert counts["frame_teacher:yes"] == 1 and counts["teacher:unsure"] == 1 and counts["llm:fake:no"] == 1
    assert run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), teacher) == {}  # all judged: nothing to do


def _crops_for(tmp_path, cands):
    (tmp_path / "crops").mkdir(exist_ok=True)
    for c in cands:
        cv2.imwrite(str(tmp_path / c.image), np.zeros((50, 50, 3), np.uint8))
    return [tmp_path / c.image for c in cands]


def test_the_llm_judges_a_batch_on_one_numbered_grid(tmp_path):
    cands = [_cand(f"p{i}", image=f"crops/p{i}.png") for i in range(5)]
    paths = _crops_for(tmp_path, cands)
    seen = tmp_path / "seen.txt"
    # answers 1 and 3 (no, yes), skips the rest: those stay unsure
    script = (f"import sys; open({str(seen)!r}, 'w').write(sys.argv[1] + '\\n' + sys.argv[2]); "
              "print('[{\"n\": 1, \"verdict\": \"no\"}, {\"n\": 3, \"verdict\": \"yes\"}]')")
    cfg = LlmConfig("t", (sys.executable, "-c", script, "{image}", "{prompt}"), 5, "p", batch=8, batch_prompt="{count} 张{cls_zh}")
    js = CommandJudge(cfg, env={LLM_ENV: "1"}).judge_batch(cands, paths)
    assert [j.verdict for j in js] == ["no", "unsure", "yes", "unsure", "unsure"]
    assert [j.id for j in js] == [c.id for c in cands] and all(j.judge == "llm:t" for j in js)
    image_path, prompt = seen.read_text().split("\n", 1)
    assert prompt == "5 张车牌"
    grid = cv2.imread(image_path)
    assert grid.shape[1] == 4 * cfg.tile and grid.shape[0] == 2 * cfg.tile  # 5 crops: 4 per row, 2 rows


def test_a_batch_must_be_one_class(tmp_path):
    cands = [_cand("a", image="crops/a.png"), _cand("b", cls="face", image="crops/b.png")]
    cfg = LlmConfig("t", (sys.executable, "-c", "print()"), 5, "p", batch=8, batch_prompt="{count}")
    with pytest.raises(ValueError):
        CommandJudge(cfg, env={LLM_ENV: "1"}).judge_batch(cands, _crops_for(tmp_path, cands))


def test_run_judges_sends_the_llm_batches_of_one_class(tmp_path):
    cands = [_cand(f"p{i}", image=f"crops/p{i}.png") for i in range(3)] + \
            [_cand(f"f{i}", cls="face", image=f"crops/f{i}.png") for i in range(2)]
    _crops_for(tmp_path, cands)
    teacher = TeacherJudge(CFG, lambda img: [], lambda img: [])  # every one "no"
    batches = []

    class FakeLlm:
        def judge_batch(self, cs, paths):
            batches.append([c.id for c in cs])
            return [Judgment(c.id, "no", "llm:fake") for c in cs]

    run_judges(cands, tmp_path / "j.jsonl", tmp_path, FrameTeacherJudge(CFG), teacher, FakeLlm(), llm_batch=2)
    assert sorted(batches) == [["f0", "f1"], ["p0", "p1"], ["p2"]]


def test_a_rerun_asks_the_llm_about_what_an_interrupted_run_judged_but_never_asked(tmp_path):
    cands = [_cand(f"p{i}", image=f"crops/p{i}.png") for i in range(4)]
    _crops_for(tmp_path, cands)
    path = tmp_path / "j.jsonl"
    # an interrupted run: the teacher judged all four, the LLM answered p0 only; a person judged p3
    append_judgments(path, [Judgment(f"p{i}", "no", "teacher") for i in range(4)] +
                     [Judgment("p0", "yes", "llm:fake"), Judgment("p3", "no", HUMAN)])
    teacher_calls = []

    def plate_detect(img):
        teacher_calls.append(1)
        return []

    asked = []

    class FakeLlm:
        def judge_batch(self, cs, paths):
            asked.extend(c.id for c in cs)
            return [Judgment(c.id, "no", "llm:fake") for c in cs]

    counts = run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), TeacherJudge(CFG, plate_detect, lambda i: []),
                        FakeLlm(), llm_batch=4)
    assert sorted(asked) == ["p1", "p2"] and teacher_calls == []  # no teacher rerun, nobody asked twice
    assert counts == {"llm:fake:no": 2}
    # without an LLM a rerun has nothing to do
    assert run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), TeacherJudge(CFG, plate_detect, lambda i: [])) == {}
