"""Model locations in the cache shared by every checkout, and the offline switches."""

from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("VC_CACHE", Path.home() / ".cache" / "video-compliance"))
MODELS = CACHE / "models"
RETINAFACE_WEIGHTS = MODELS / "Resnet50_Final.pth"
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
COCO_CHECKPOINT = "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
OWLV2_REPO_DIR = "models--google--owlv2-base-patch16-ensemble"
OWLV2_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


def offline_model_env() -> None:
    """Points Hugging Face / torch at the shared cache, forbids Hugging Face downloads and onnxruntime telemetry (metered link).

    HF_HUB_OFFLINE is forced to "1" rather than defaulted, so an exported HF_HUB_OFFLINE=0
    cannot re-enable downloads. Torch weights must be loaded from cached files: no env var
    stops torchvision from downloading a missing file.

    Must run before transformers or torchvision load weights.
    """
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    # onnxruntime's macOS wheel uploads usage telemetry through the system network stack, past the proxy variables;
    # it reads this when it creates its environment
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"


def required_model_files() -> list[Path]:
    """Every cached file the teachers load. Call after offline_model_env() so the cache env is set."""
    torch_home = Path(os.environ["TORCH_HOME"])
    hf_home = Path(os.environ["HF_HOME"])
    return [
        RETINAFACE_WEIGHTS,
        YUNET_MODEL,
        torch_home / "hub" / "checkpoints" / COCO_CHECKPOINT,
        hf_home / "hub" / OWLV2_REPO_DIR / "snapshots" / OWLV2_REVISION / "model.safetensors",
    ]
