"""Tiny end-to-end run on the CPU: dataset -> 1 epoch -> ONNX -> consistency (needs yolov8n.pt)."""

import csv
import json

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train.dataset import build_dataset

pytestmark = pytest.mark.yolo


def _tiny_segment(root, n=6, size=128):
    frames_dir, labels_dir = root / "frames" / "t", root / "labels" / "t"
    frames_dir.mkdir(parents=True)
    (labels_dir / "yolo").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i in range(n):
            img = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            x, y = 10 + 12 * i, 30
            img[y : y + 24, x : x + 24] = (0, 0, 255)
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), img)
            w.writerow(["t.mp4", i, i * 10000.0, "base", 0.0])
            lines = to_yolo_lines([Box("face", x, y, x + 24, y + 24)], size, size)
            (labels_dir / "yolo" / f"{i:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("".join(f"{i}\n" for i in range(n)))
    return frames_dir, labels_dir


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from vc_train.train import train

    root = tmp_path_factory.mktemp("yolo")
    seg = _tiny_segment(root)
    build_dataset([seg], root / "ds", 64, 64, val_tail_ms=20000.0, crop=64)
    best = train(root / "ds" / "data.yaml", root / "run", imgsz=64, epochs=1, hours=None, batch=2, device="cpu", workers=0)
    return root, seg, best


def test_offline_env_blackholes_the_network(monkeypatch):
    from vc_train.offline import BLACKHOLE, ultralytics_offline_env

    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:3128")
    monkeypatch.setenv("YOLO_OFFLINE", "0")
    ultralytics_offline_env()
    import os

    assert os.environ["HTTPS_PROXY"] == BLACKHOLE and os.environ["https_proxy"] == BLACKHOLE
    assert os.environ["YOLO_OFFLINE"] == "1" and os.environ["YOLO_AUTOINSTALL"] == "false"


def test_load_yolo_refuses_missing_weights(tmp_path):
    from vc_train.offline import load_yolo

    with pytest.raises(FileNotFoundError):
        load_yolo(tmp_path / "nope.pt")


def test_train_writes_best_weights_and_refuses_to_reuse_a_run(tiny_run):
    from vc_train.train import train

    root, _, best = tiny_run
    assert best.exists() and best.name == "best.pt"
    with pytest.raises(ValueError, match="already exists"):
        train(root / "ds" / "data.yaml", root / "run", imgsz=64, epochs=1, hours=None, batch=2, device="cpu", workers=0)


def test_export_writes_static_onnx_and_model_json(tiny_run):
    import onnxruntime as ort

    from vc_train.export import export_onnx, read_model_json

    root, _, best = tiny_run
    onnx_path = export_onnx(best, root / "model", 64, 64, manifest=root / "ds" / "manifest.csv", name="tiny")
    assert onnx_path == root / "model" / "model.onnx"
    meta = read_model_json(root / "model")
    assert meta["name"] == "tiny" and meta["classes"] == ["face", "plate"]
    assert meta["input"] == {
        "width": 64, "height": 64, "color": "RGB", "scale": 1 / 255,
        "letterbox": "top_left", "pad_value": 114,
    }
    assert meta["output_format"] == "yolov8" and len(meta["train_manifest_sha256"]) == 64
    assert isinstance(meta["metrics"]["val_map50"], float)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 64, 64]
    assert session.get_outputs()[0].shape[1] == 6


def test_pytorch_and_onnxruntime_agree(tiny_run):
    from vc_train.export import check_consistency, export_onnx

    root, (frames_dir, _), best = tiny_run
    onnx_path = export_onnx(best, root / "model_check", 64, 64)
    images = sorted(frames_dir.glob("*.jpg"))
    # threshold 0 keeps every anchor, so a score sitting on the threshold cannot make the two runtimes disagree
    report = check_consistency(best, onnx_path, images, score_threshold=0.0)
    assert report.images == len(images)
    assert report.passed(1.0), report
