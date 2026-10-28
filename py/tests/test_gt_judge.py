import cv2
import numpy as np
import pytest

from vc_gt.config import load_config
from vc_gt.crops import Crop
from dataclasses import replace

from vc_gt.judge import (FRAMES, OBJECT_CROPS, OBJECT_VERDICTS, OBJECTS, Item, aggregate, ask, frame_id, grid_image,
                         judge_frames, judge_objects, make_runner, object_id, parse_answers)
from vc_gt.tracks import Track
from vc_hardneg.judgments import read_judgments, resolve
from vc_label.contracts import Box


def _crop(tmp_path, name):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((120, 120, 3), 128, np.uint8))
    return Crop(p, (40, 54, 80, 66))


def test_parse_answers_takes_the_last_json_array_and_only_allowed_verdicts():
    out = ('thinking [1, 2]\n[{"n": 1, "verdict": "readable"}, {"n": 2, "verdict": "maybe"}, '
           '{"n": "3", "verdict": "unreadable"}]')
    assert parse_answers(out, ("readable", "unreadable", "unsure")) == {1: "readable"}
    assert parse_answers(None, ("readable",)) == {}


def test_grid_image_puts_an_items_crops_side_by_side(tmp_path):
    items = [Item(f"o{k}", "plate", tuple(_crop(tmp_path, f"{k}_{j}.jpg") for j in range(3))) for k in range(3)]
    out = grid_image(items, 100, tmp_path / "g.png")
    assert cv2.imread(str(out)).shape[:2] == (200, 600)  # 2 cells per row of 3 x 100 px, 2 rows


class _Runner:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, image, prompt, cls):
        self.calls.append((image.name, prompt, cls))
        return self.reply


def test_ask_batches_by_class_appends_verdicts_and_skips_what_is_judged(tmp_path):
    items = [Item("o0", "plate", (_crop(tmp_path, "a.jpg"),)), Item("o1", "plate", (_crop(tmp_path, "b.jpg"),)),
             Item("o2", "face", (_crop(tmp_path, "c.jpg"),))]
    runner = _Runner('[{"n": 1, "verdict": "not_real"}]')
    counts = ask(items, "共 {count} 组{cls_zh}", OBJECT_VERDICTS, runner, "llm:fake", tmp_path / "j.jsonl",
                 tmp_path / "llm", batch=4, tile=64)
    final = resolve(read_judgments(tmp_path / "j.jsonl"))
    assert {k: v.verdict for k, v in final.items()} == {"o0": "not_real", "o1": "unsure", "o2": "not_real"}
    assert counts == {"not_real": 2, "unsure": 1}
    assert sorted(c[1] for c in runner.calls) == ["共 1 组人脸", "共 2 组车牌"]
    assert ask(items, "{count}{cls_zh}", OBJECT_VERDICTS, runner, "llm:fake", tmp_path / "j.jsonl",
               tmp_path / "llm", batch=4, tile=64) == {}
    assert len(runner.calls) == 2


def test_question_b_may_answer_that_the_box_holds_no_plate_or_face(tmp_path):
    from vc_gt.judge import FRAME_VERDICTS

    ask([Item("o0@6", "plate", (_crop(tmp_path, "a.jpg"),))], "{count}{cls_zh}", FRAME_VERDICTS,
        _Runner('[{"n": 1, "verdict": "not_target"}]'), "llm:fake", tmp_path / "f.jsonl", tmp_path / "llm", 4, 64)
    assert read_judgments(tmp_path / "f.jsonl")[0].verdict == "not_target"


def test_a_command_that_does_not_answer_leaves_its_items_unjudged_for_a_rerun(tmp_path):
    # the 2026-10-26 run: 1,245 frames whose calls timed out were recorded "unsure" and a rerun skipped them
    items = [Item("o0", "plate", (_crop(tmp_path, "a.jpg"),))]
    counts = ask(items, "{count}{cls_zh}", OBJECT_VERDICTS, _Runner(None), "llm:fake", tmp_path / "j.jsonl",
                 tmp_path / "llm", batch=4, tile=64)
    assert counts == {"no answer": 1} and read_judgments(tmp_path / "j.jsonl") == []
    ask(items, "{count}{cls_zh}", OBJECT_VERDICTS, _Runner('[{"n": 1, "verdict": "not_real"}]'), "llm:fake",
        tmp_path / "j.jsonl", tmp_path / "llm", batch=4, tile=64)
    assert read_judgments(tmp_path / "j.jsonl")[0].verdict == "not_real"


class _PerObject:
    """Answers one crop per call (batch 1) by the object its grid belongs to: batch_o<id>#<k>.jpg."""

    def __init__(self, verdicts):
        self.verdicts, self.calls = verdicts, []

    def __call__(self, image, prompt, cls):
        self.calls.append(image.name)
        oid = image.name.removeprefix("batch_").split("#")[0].split("@")[0]
        return f'[{{"n": 1, "verdict": "{self.verdicts[oid]}"}}]'


def _one_per_call(cfg):
    return replace(cfg, llm=replace(cfg.llm, batch=1))


def test_aggregate_takes_the_strongest_real_answer_then_not_real_only_when_every_crop_says_so():
    assert aggregate(["not_real", "real_unreadable", "real_readable"]) == "real_readable"
    assert aggregate(["not_real", "real_unreadable", "unsure"]) == "real_unreadable"
    assert aggregate(["not_real", "not_real"]) == "not_real"
    assert aggregate(["not_real", "unsure"]) == "unsure"


def _segment(tmp_path, width=40, score=0.9):
    images = {}
    for f in (0, 6, 12):
        images[f] = tmp_path / f"{f:06d}.jpg"
        cv2.imwrite(str(images[f]), np.zeros((1080, 1920, 3), np.uint8))
    boxes = tuple(Box("plate", 100, 100, 100 + width, 112, score) for _ in range(3))
    return images, [Track(0, "plate", (0, 6, 12), boxes), Track(1, "plate", (0, 6, 12), boxes)]


def test_judge_objects_then_frames_asks_about_the_unmasked_frames(tmp_path):
    images, tracks = _segment(tmp_path)
    gt, cfg = tmp_path / "gt", load_config()
    gt.mkdir()
    runner = _PerObject({"o0": "real_readable", "o1": "not_real"})
    judge_objects(gt, tracks, images, _one_per_call(cfg), runner, "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / OBJECTS)).items()} == {
        object_id(0): "real_readable", object_id(1): "not_real"}
    assert len(runner.calls) == 6  # 3 crops of each object, one by one
    assert len(read_judgments(gt / OBJECT_CROPS)) == 6
    masks = {0: [Box("plate", 90, 90, 150, 120)]}  # frame 0 covered, 6 and 12 not
    runner = _Runner('[{"n": 1, "verdict": "readable"}, {"n": 2, "verdict": "unreadable"}]')
    judge_frames(gt, tracks, images, masks, cfg, runner, "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / FRAMES)).items()} == {
        frame_id(0, 6): "readable", frame_id(0, 12): "unreadable",  # o1 (not real) too: answers 3, 4 missing
        frame_id(1, 6): "unsure", frame_id(1, 12): "unsure"}
    assert (gt / "crops" / "frames" / "o0@000006.jpg").exists()


def test_judge_objects_skips_tracks_that_never_reach_must_mask_size(tmp_path):
    images, tracks = _segment(tmp_path, width=10)  # a 10 px plate on 1080p: under the 24 px must-mask width
    gt = tmp_path / "gt"
    gt.mkdir()
    runner = _Runner('[{"n": 1, "verdict": "real_readable"}]')
    assert judge_objects(gt, tracks, images, load_config(), runner, "llm:fake") == {}
    assert runner.calls == [] and not (gt / OBJECTS).exists()


def test_judge_frames_asks_only_about_unmasked_frames_at_must_mask_size(tmp_path):
    images, _ = _segment(tmp_path)
    boxes = (Box("plate", 100, 100, 140, 112, 0.9), Box("plate", 100, 100, 120, 106, 0.9),  # 40 px, then 20 px
             Box("plate", 100, 100, 140, 112, 0.9))
    track = Track(0, "plate", (0, 6, 12), boxes)
    gt = tmp_path / "gt"
    gt.mkdir()
    judge_objects(gt, [track], images, _one_per_call(load_config()), _PerObject({"o0": "real_readable"}), "llm:fake")
    judge_frames(gt, [track], images, {}, load_config(), _Runner('[{"n": 1, "verdict": "readable"}]'), "llm:fake")
    assert sorted(resolve(read_judgments(gt / FRAMES))) == [frame_id(0, 0), frame_id(0, 12)]  # not the 20 px frame 6


def test_judge_objects_skips_tracks_no_detector_scored_high_enough(tmp_path):
    images, tracks = _segment(tmp_path, score=0.29)  # judge_min_score is 0.3
    gt = tmp_path / "gt"
    gt.mkdir()
    runner = _Runner('[{"n": 1, "verdict": "real_readable"}]')
    assert judge_objects(gt, tracks, images, load_config(), runner, "llm:fake") == {}
    assert runner.calls == []


def test_grids_go_out_as_jpeg(tmp_path):
    out = grid_image([Item("o0", "plate", (_crop(tmp_path, "a.jpg"),))], 64, tmp_path / "g.jpg")
    assert out.read_bytes()[:2] == b"\xff\xd8"


def test_make_runner_refuses_without_the_llm_switch(monkeypatch):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    with pytest.raises(PermissionError):
        make_runner(load_config())


def test_an_interrupted_object_is_finished_without_asking_its_judged_crops_again(tmp_path):
    from vc_hardneg.judgments import Judgment, append_judgments

    images, tracks = _segment(tmp_path)
    gt = tmp_path / "gt"
    gt.mkdir()
    append_judgments(gt / OBJECT_CROPS, [Judgment("o0#0", "not_real", "llm:fake")], verdicts=OBJECT_VERDICTS)
    runner = _PerObject({"o0": "real_unreadable", "o1": "not_real"})
    judge_objects(gt, tracks[:1], images, _one_per_call(load_config()), runner, "llm:fake")
    assert len(runner.calls) == 2 and resolve(read_judgments(gt / OBJECTS))["o0"].verdict == "real_unreadable"


def test_question_b_is_asked_about_every_candidate_whatever_question_a_said(tmp_path):
    from vc_hardneg.judgments import Judgment, append_judgments

    images, tracks = _segment(tmp_path)
    gt = tmp_path / "gt"
    gt.mkdir()
    append_judgments(gt / OBJECTS, [Judgment("o0", "not_real", "llm:x")], verdicts=OBJECT_VERDICTS)  # o1 never asked
    runner = _PerObject({"o0": "readable", "o1": "unreadable"})
    judge_frames(gt, tracks, images, {}, _one_per_call(load_config()), runner, "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / FRAMES)).items()} == {
        frame_id(0, f): "readable" for f in (0, 6, 12)} | {frame_id(1, f): "unreadable" for f in (0, 6, 12)}


def test_question_b_skips_candidates_question_a_would_skip(tmp_path):
    images, tracks = _segment(tmp_path, score=0.29)
    gt = tmp_path / "gt"
    gt.mkdir()
    runner = _PerObject({"o0": "readable", "o1": "readable"})
    assert judge_frames(gt, tracks, images, {}, _one_per_call(load_config()), runner, "llm:fake") == {}
