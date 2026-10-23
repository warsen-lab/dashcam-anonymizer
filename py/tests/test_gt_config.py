import json

import pytest

from vc_gt.config import DEFAULT_CONFIG, load_config


def test_the_repository_config_holds_the_spec_values():
    c = load_config()
    assert (c.owlv2_score, c.tile_grid, c.tile_overlap) == (0.05, 2, 0.2)
    assert (c.retinaface_score, c.yunet_score, c.student_score) == (0.2, 0.3, 0.01)
    assert (c.track.min_iou, c.track.max_size_ratio, c.track.max_gap) == (0.3, 2.0, 3)
    assert (c.plate_min_aspect, c.plate_max_width_px, c.judge_min_score) == (1.2, 400, 0.3)
    assert c.crops_per_object == 3
    assert (c.audit.objects, c.audit.frames, c.audit.frames_per_class) == (50, 50, 150)
    assert "{count}" in c.llm.object_prompt and "{cls_zh}" in c.llm.frame_prompt
    assert "马赛克" in c.llm.frame_prompt  # question B is asked on the redacted frame (spec §10)
    assert c.llm.command_config().command == c.llm.command


def _write(tmp_path, change):
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    change(d)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(extra=1), "keys"),
    (lambda d: d.update(owlv2_score=1.5), "owlv2_score"),
    (lambda d: d.update(tile_overlap=0.5), "tile_overlap"),
    (lambda d: d.update(crops_per_object=0), "crops_per_object"),
    (lambda d: d["llm"].update(object_prompt="no placeholders"), "object_prompt"),
    (lambda d: d["llm"].update(batch=0), "batch"),
    (lambda d: d["track"].update(max_gap=-1), "max_gap"),
    (lambda d: d["track"].update(max_size_ratio=0.9), "max_size_ratio"),
    (lambda d: d.update(judge_min_score=2), "judge_min_score"),
    (lambda d: d.update(plate_min_aspect=0), "plate_min_aspect"),
])
def test_bad_configs_are_refused_naming_the_key(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, change))
