"""Model locations in the cache shared by every checkout, and the offline switches."""

from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("VC_CACHE", Path.home() / ".cache" / "video-compliance"))
MODELS = CACHE / "models"
RETINAFACE_WEIGHTS = MODELS / "Resnet50_Final.pth"
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"


def offline_model_env() -> None:
    """Points Hugging Face / torch at the shared cache and forbids downloads (metered link).

    Must run before transformers or torchvision load weights.
    """
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
