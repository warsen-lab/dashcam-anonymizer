import json

import cv2
import numpy as np
import pytest

from vc_gt.audit import (OBJECT_CHOICES, PageItem, audit_sample, h1_frame_ids, h1_object_ids, h2_sample,
                         import_annotations, import_choices, read_audit, write_annotate_page, write_audit,
                         write_choice_page)
from vc_gt.crops import Crop
from vc_hardneg.judgments import Judgment


def test_h1_takes_unsure_objects_and_readable_or_unsure_unmasked_frames_people_did_not_judge():
    objects = {"o0": Judgment("o0", "unsure", "llm:x"), "o1": Judgment("o1", "not_real", "llm:x"),
               "o2": Judgment("o2", "unsure", "human")}
    frames = {"o0@6": Judgment("o0@6", "readable", "llm:x"), "o0@12": Judgment("o0@12", "unreadable", "llm:x"),
              "o0@18": Judgment("o0@18", "unsure", "llm:x"), "o0@24": Judgment("o0@24", "readable", "llm:x")}
    assert h1_object_ids(objects) == ["o0"]
    assert h1_frame_ids(frames, {"o0@6", "o0@12", "o0@18"}) == ["o0@18", "o0@6"]  # o0@24 is masked now


def test_h2_and_audit_samples_are_seeded_and_capped():
    ids = [f"o{k}" for k in range(100)]
    assert h2_sample(ids, 10, 1) == h2_sample(list(reversed(ids)), 10, 1) and len(h2_sample(ids, 10, 1)) == 10
    assert h2_sample(ids[:3], 10, 1) == sorted(ids[:3])
    s = audit_sample([0, 6, 12, 18], {0: 0, 6: 2, 12: 0, 18: 1}, 3, 7)
    assert len(s["plate"]) == 3 and s["face"] == [6, 18]  # face frames only where COCO saw people


def _crop(tmp_path, name):
    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros((120, 120, 3), np.uint8))
    return Crop(p, (40, 54, 80, 66))


def test_choice_page_copies_crops_and_holds_the_choices(tmp_path):
    page = write_choice_page([PageItem("o3", (_crop(tmp_path, "a.jpg"),), "plate")], tmp_path / "page", "H1 车牌",
                             OBJECT_CHOICES)
    text = page.read_text(encoding="utf-8")
    assert (tmp_path / "page" / "img" / "o3_0.jpg").exists()
    assert '"o3"' in text and "real_readable" in text and "img/o3_0.jpg" in text and "H1 车牌" in text


def test_people_can_say_a_frame_holds_no_plate_or_face():
    from vc_gt.audit import FRAME_CHOICES

    assert "not_target" in [v for v, _ in FRAME_CHOICES]


def test_import_choices_refuses_unknown_ids_and_verdicts(tmp_path):
    f = tmp_path / "e.json"
    f.write_text(json.dumps([{"id": "o1", "verdict": "readable"}]))
    assert import_choices(f, {"o1"}, ("readable", "unreadable"))[0] == Judgment("o1", "readable", "human")
    with pytest.raises(ValueError, match="unknown id"):
        import_choices(f, {"o2"}, ("readable",))
    f.write_text(json.dumps([{"id": "o1", "verdict": "real_readable"}]))
    with pytest.raises(ValueError, match="verdict"):
        import_choices(f, {"o1"}, ("readable", "unreadable"))


def test_annotations_keep_finished_frames_only_and_round_trip(tmp_path):
    img = tmp_path / "000006.jpg"
    cv2.imwrite(str(img), np.zeros((100, 200, 3), np.uint8))
    page = write_annotate_page([("plate:6", img, "plate")], tmp_path / "ann", "H3")
    assert "plate:6" in page.read_text(encoding="utf-8") and (tmp_path / "ann" / "img" / "plate_6.jpg").exists()
    export = tmp_path / "audit_export.json"
    export.write_text(json.dumps([{"key": "plate:6", "done": True, "boxes": [[10, 20, 50, 32]]},
                                  {"key": "face:12", "done": False, "boxes": [[1, 1, 5, 5]]}]))
    rows = import_annotations(export, {"plate:6": ("plate", 6), "face:12": ("face", 12)})
    assert rows == [{"key": "plate:6", "cls": "plate", "frame": 6, "boxes": [[10, 20, 50, 32]]}]
    write_audit(tmp_path / "audit.jsonl", rows)
    done, boxes = read_audit(tmp_path / "audit.jsonl")
    assert done == {"plate": {6}, "face": set()} and boxes[0].frame == 6 and boxes[0].box.x2 == 50
    with pytest.raises(ValueError, match="unknown key"):
        import_annotations(export, {"plate:6": ("plate", 6)})
