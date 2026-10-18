"""Keeps ultralytics off the network (metered link): no downloads, pip installs, telemetry or settings sync."""

from __future__ import annotations

import os
from pathlib import Path

from vc_label.paths import CACHE, MODELS, offline_model_env

YOLO_WEIGHTS = MODELS / "yolov8n.pt"
BLACKHOLE = "http://127.0.0.1:9"
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def ultralytics_offline_env() -> None:
    """Must run before `import ultralytics`. Forces offline mode and points every proxy at a local black hole."""
    offline_model_env()
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    config_dir = CACHE / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    for var in _PROXY_VARS:
        os.environ[var] = BLACKHOLE
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"


def load_yolo(weights: Path):
    """ultralytics.YOLO for a local weights file; never downloads."""
    ultralytics_offline_env()
    if not Path(weights).is_file():
        raise FileNotFoundError(f"{weights} does not exist (see py/README.md: vc_train weights)")
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    if SETTINGS.get("sync"):
        SETTINGS.update({"sync": False})
    return YOLO(str(weights))
