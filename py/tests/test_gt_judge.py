import cv2
import numpy as np
import pytest

from vc_gt.config import load_config
from vc_gt.crops import Crop
from vc_gt.judge import (FRAMES, OBJECT_VERDICTS, OBJECTS, Item, ask, frame_id, grid_image, judge_frames,
                         judge_objects, make_runner, object_id, parse_answers)
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


def test_a_command_that_does_not_answer_leaves_its_items_unsure(tmp_path):
    ask([Item("o0", "plate", (_crop(tmp_path, "a.jpg"),))], "{count}{cls_zh}", OBJECT_VERDICTS, _Runner(None),
        "llm:fake", tmp_path / "j.jsonl", tmp_path / "llm", batch=4, tile=64)
    j = read_judgments(tmp_path / "j.jsonl")[0]
    assert (j.verdict, j.detail) == ("unsure", "no answer")


def _segment(tmp_path, width=40):
    images = {}
    for f in (0, 6, 12):
        images[f] = tmp_path / f"{f:06d}.jpg"
        cv2.imwrite(str(images[f]), np.zeros((1080, 1920, 3), np.uint8))
    boxes = tuple(Box("plate", 100, 100, 100 + width, 112) for _ in range(3))
    return images, [Track(0, "plate", (0, 6, 12), boxes), Track(1, "plate", (0, 6, 12), boxes)]


def test_judge_objects_then_frames_asks_only_about_unmasked_frames_of_real_objects(tmp_path):
    images, tracks = _segment(tmp_path)
    gt, cfg = tmp_path / "gt", load_config()
    gt.mkdir()
    judge_objects(gt, tracks, images, cfg,
                  _Runner('[{"n": 1, "verdict": "real_readable"}, {"n": 2, "verdict": "not_real"}]'), "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / OBJECTS)).items()} == {
        object_id(0): "real_readable", object_id(1): "not_real"}
    masks = {0: [Box("plate", 90, 90, 150, 120)]}  # frame 0 covered, 6 and 12 not
    runner = _Runner('[{"n": 1, "verdict": "readable"}, {"n": 2, "verdict": "unreadable"}]')
    judge_frames(gt, tracks, images, masks, cfg, runner, "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / FRAMES)).items()} == {
        frame_id(0, 6): "readable", frame_id(0, 12): "unreadable"}
    assert (gt / "crops" / "frames" / "o0@000006.jpg").exists()


def test_judge_objects_skips_tracks_that_never_reach_must_mask_size(tmp_path):
    images, tracks = _segment(tmp_path, width=10)  # a 10 px plate on 1080p: under the 24 px must-mask width
    gt = tmp_path / "gt"
    gt.mkdir()
    runner = _Runner('[{"n": 1, "verdict": "real_readable"}]')
    assert judge_objects(gt, tracks, images, load_config(), runner, "llm:fake") == {}
    assert runner.calls == [] and not (gt / OBJECTS).exists()


def test_make_runner_refuses_without_the_llm_switch(monkeypatch):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    with pytest.raises(PermissionError):
        make_runner(load_config())
