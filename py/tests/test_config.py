import json
from dataclasses import fields, replace

import pytest

from conftest import REPO_ROOT
from vc_label.config import LabelingConfig


def test_low_score_bands_are_per_class():
    cfg = LabelingConfig()
    assert cfg.low_score_band("face") == (0.3, 0.5)
    assert cfg.low_score_band("plate") == (0.1, 0.3)


def test_from_json_overrides_and_converts_lists(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_score": 0.6, "plate_low_score": [0.1, 0.35]}))
    cfg = LabelingConfig.from_json(path)
    assert cfg.yunet_score == 0.6
    assert cfg.plate_low_score == (0.1, 0.35)
    assert cfg.retinaface_score == LabelingConfig().retinaface_score


def test_from_json_rejects_unknown_keys(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_scor": 0.6}))
    with pytest.raises(ValueError, match="yunet_scor"):
        LabelingConfig.from_json(path)


def test_repository_config_lists_every_field_and_is_valid():
    path = REPO_ROOT / "configs" / "labeling.json"  # users edit the values, so only the keys are pinned
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {f.name for f in fields(LabelingConfig)}
    LabelingConfig.from_json(path)


def test_to_json_round_trips(tmp_path):
    cfg = replace(LabelingConfig(), yunet_score=0.6, plate_low_score=(0.1, 0.35))
    path = tmp_path / "labeling.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    assert LabelingConfig.from_json(path) == cfg


@pytest.mark.parametrize(
    "field, value",
    [
        ("retinaface_score", 1.5),
        ("yunet_score", -0.1),
        ("coco_score", float("nan")),
        ("owlv2_score", "0.1"),
        ("fuse_iou", 2),
        ("temporal_match_iou", -1),
        ("temporal_present_iou", 1.01),
        ("plate_search_min_vehicle_frac", 0),
        ("min_person_height_frac", 1.5),
        ("face_region_frac", -0.2),
        ("min_vehicle_width_frac", 0.0),
        ("temporal_max_gap_ms", 0),
        ("temporal_max_gap_ms", -5.0),
        ("face_low_score", (0.5, 0.3)),  # inverted
        ("face_low_score", (0.3, 0.3)),  # empty
        ("plate_low_score", (-0.1, 0.3)),
        ("plate_low_score", (0.1, 1.2)),
        ("face_low_score", (0.3,)),
        ("face_low_score", (0.1, 0.3, 0.5)),
        ("face_low_score", 0.4),  # scalar where a band is expected
        ("plate_low_score", None),
    ],
)
def test_invalid_values_are_rejected_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        replace(LabelingConfig(), **{field: value})


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"face_low_score": (0.4, 0.5)}, "face_low_score"),  # retinaface_score 0.3 would label boxes the band never flags
        ({"retinaface_score": 0.2, "yunet_score": 0.5, "face_low_score": (0.25, 0.5)}, "face_low_score"),
        ({"yunet_score": 0.2, "face_low_score": (0.25, 0.5)}, "face_low_score"),
        ({"plate_low_score": (0.15, 0.3)}, "plate_low_score"),  # owlv2_score 0.1
    ],
)
def test_a_band_must_start_at_or_below_the_teacher_thresholds(overrides, field):
    with pytest.raises(ValueError, match=field):
        replace(LabelingConfig(), **overrides)


def test_from_json_wraps_a_scalar_band_as_a_value_error(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"face_low_score": 0.4}))
    with pytest.raises(ValueError, match="face_low_score"):
        LabelingConfig.from_json(path)


def test_from_json_rejects_a_non_object_root(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match=r"labeling\.json.*object"):
        LabelingConfig.from_json(path)


def test_from_json_reads_utf8(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text('{"yunet_score": 0.6}\n', encoding="utf-8")
    assert LabelingConfig.from_json(path).yunet_score == 0.6


def test_low_score_band_rejects_unknown_classes():
    with pytest.raises(ValueError, match="person"):
        LabelingConfig().low_score_band("person")
