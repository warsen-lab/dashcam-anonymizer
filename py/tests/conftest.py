from pathlib import Path

import pytest

from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FRAMES = REPO_ROOT / "data" / "frames" / "sg_drive_600s"

offline_model_env()


def pytest_collection_modifyitems(items):
    missing = [p for p in (RETINAFACE_WEIGHTS, YUNET_MODEL, SAMPLE_FRAMES / "006005.jpg") if not p.exists()]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"missing {', '.join(str(p) for p in missing)}")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def sample_frame():
    import cv2

    return cv2.imread(str(SAMPLE_FRAMES / "006005.jpg"))


@pytest.fixture(scope="session")
def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def best_match(boxes, target):
    """The box overlapping `target` most, or None."""
    from vc_label.boxes import iou

    return max(boxes, key=lambda b: iou(b, target), default=None)
