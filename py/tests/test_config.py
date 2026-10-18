import json

import pytest

from conftest import REPO_ROOT
from vc_label.config import LabelingConfig


def test_low_score_bands_are_per_class():
    cfg = LabelingConfig()
    assert cfg.low_score_band("face") == (0.3, 0.5)
    assert cfg.low_score_band("plate") == (0.1, 0.3)


def test_from_json_overrides_and_converts_lists(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_score": 0.6, "plate_low_score": [0.15, 0.35]}))
    cfg = LabelingConfig.from_json(path)
    assert cfg.yunet_score == 0.6
    assert cfg.plate_low_score == (0.15, 0.35)
    assert cfg.retinaface_score == LabelingConfig().retinaface_score


def test_from_json_rejects_unknown_keys(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_scor": 0.6}))
    with pytest.raises(ValueError, match="yunet_scor"):
        LabelingConfig.from_json(path)


def test_repository_config_matches_defaults():
    assert LabelingConfig.from_json(REPO_ROOT / "configs" / "labeling.json") == LabelingConfig()
