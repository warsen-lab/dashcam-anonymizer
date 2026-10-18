"""Model locations in the cache shared by every checkout, and the offline switches."""

from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("VC_CACHE", Path.home() / ".cache" / "video-compliance"))
MODELS = CACHE / "models"
RETINAFACE_WEIGHTS = MODELS / "Resnet50_Final.pth"
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"


def offline_model_env() -> None:
    """Points Hugging Face / torch at the shared cache and forbids Hugging Face downloads (metered link).

    HF_HUB_OFFLINE is forced to "1" rather than defaulted, so an exported HF_HUB_OFFLINE=0
    cannot re-enable downloads. Torch weights must be loaded from cached files: no env var
    stops torchvision from downloading a missing file.

    Must run before transformers or torchvision load weights.
    """
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ["HF_HUB_OFFLINE"] = "1"
