import os
from pathlib import Path

import pytest

from vc_label.paths import offline_model_env, required_model_files
from vc_train.offline import YOLO_WEIGHTS  # importing it must not import ultralytics

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FRAMES = REPO_ROOT / "data" / "frames" / "sg_drive_600s"

offline_model_env()


def pytest_collection_modifyitems(items):
    # Every model file the teachers load, not just some: a missing COCO checkpoint would make torchvision
    # download ~160 MB over the metered link instead of skipping.
    missing = [p for p in (*required_model_files(), SAMPLE_FRAMES / "006005.jpg") if not p.exists()]
    for item in items:
        if "models" in item.keywords and missing:
            item.add_marker(pytest.mark.skip(reason=f"missing {', '.join(str(p) for p in missing)}"))
        if "yolo" in item.keywords and not YOLO_WEIGHTS.exists():
            item.add_marker(pytest.mark.skip(reason=f"missing {YOLO_WEIGHTS}"))


@pytest.fixture(autouse=True)
def _restore_environ_after_train_tests(request):
    """vc_train.cli.main and ultralytics_offline_env() rewrite proxy and YOLO_* variables; keep that out of other tests."""
    if not request.module.__name__.startswith("test_train_"):
        yield
        return
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


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
