import json
from dataclasses import replace

import cv2
import numpy as np

from vc_gt.calibrate import calibrate, compare_verdicts
from vc_gt.config import load_config
from vc_gt.crops import CROPS_INDEX, Crop, append_crop_index
from vc_gt.judge import OBJECT_VERDICTS, OBJECTS
from vc_hardneg.judgments import Judgment, append_judgments


def test_compare_counts_overturns_among_answered_and_the_dangerous_ones_apart():
    people = {"o1": "real_readable", "o2": "not_real", "o3": "real_unreadable", "o4": "real_readable",
              "o5": "not_real"}
    model = {"o1": "real_readable", "o2": "real_unreadable", "o3": "not_real", "o4": "real_unreadable",
             "o5": "unsure"}
    c = compare_verdicts(model, people)
    assert (c["answered"], c["overturned"], c["unsure"]) == (4, 3, 1)
    assert c["real_called_not_real"] == 1 and c["readable_called_unreadable"] == 1


def test_calibrate_asks_about_a_seeded_sample_of_the_people_judged_objects_in_its_own_dir(tmp_path):
    gt, out = tmp_path / "gt", tmp_path / "calib"
    gt.mkdir()
    for k in range(4):
        p = tmp_path / f"c{k}.jpg"
        cv2.imwrite(str(p), np.zeros((120, 120, 3), np.uint8))
        append_crop_index(gt / CROPS_INDEX, f"o{k}", [Crop(p, (40, 54, 80, 66))])
    append_judgments(gt / OBJECTS, [Judgment(f"o{k}", "real_readable" if k % 2 else "not_real", "human")
                                    for k in range(4)] + [Judgment("o9", "not_real", "llm:x")],
                     verdicts=OBJECT_VERDICTS)
    cls = {f"o{k}": "plate" for k in range(4)}
    cfg = load_config()
    cfg = replace(cfg, llm=replace(cfg.llm, batch=1))
    calls = []

    def runner(image, prompt, c):
        calls.append(image.name)
        return '[{"n": 1, "verdict": "real_readable"}]'

    result = calibrate(gt, cls, cfg, runner, "llm:fake", sample=3, seed=1, out_dir=out)
    assert len(calls) == 3 and result["answered"] == 3
    assert result["overturned"] == sum(1 for c in calls if c.split("#")[0].removeprefix("batch_") in ("o0", "o2"))
    assert json.loads((out / "calibration.json").read_text())["answered"] == 3
    assert not (gt / "object_crops.jsonl").exists()  # the answer key's own files stay untouched
