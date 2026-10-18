"""Tiny end-to-end run on the CPU: dataset -> 1 epoch -> ONNX -> consistency (needs yolov8n.pt)."""

import csv
import os
import re

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train.dataset import build_dataset

# pytest applies the command line `-W error` after the ini filterwarnings, so the pyproject.toml ignores alone do not
# survive it. Marks are applied last: the same three known export warnings stay quiet, nothing else does.
EXPORT_WARNING_IGNORES = (
    "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning",
    "ignore:The feature will be removed:DeprecationWarning:torch\\.onnx\\._internal\\.torchscript_exporter\\.utils",
    "ignore:Converting a tensor to a Python boolean:torch.jit.TracerWarning",
)
pytestmark = [pytest.mark.yolo, pytest.mark.filterwarnings(*EXPORT_WARNING_IGNORES)]

OFFLINE_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    "NO_PROXY", "no_proxy", "YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_CONFIG_DIR", "ULTRALYTICS_API_KEY",
    "HF_HUB_OFFLINE", "HF_HOME", "TORCH_HOME",  # also written by offline_model_env()
)


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


class _ExportStub:
    """Stands in for ultralytics.YOLO: export() hands back a prepared ONNX file."""

    names = {0: "face", 1: "plate"}

    def __init__(self, onnx_file):
        self._onnx_file = onnx_file

    def export(self, **_):
        return str(self._onnx_file)


def test_offline_env_blackholes_the_network(monkeypatch):
    from vc_label.paths import CACHE
    from vc_train.offline import BLACKHOLE, ultralytics_offline_env

    for name in OFFLINE_ENV_VARS:
        # setenv then delenv: monkeypatch only undoes writes it has recorded, so this makes every name restorable
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:3128")
    monkeypatch.setenv("YOLO_OFFLINE", "0")
    monkeypatch.setenv("YOLO_CONFIG_DIR", "/elsewhere")
    monkeypatch.setenv("ULTRALYTICS_API_KEY", "not-a-real-key")
    ultralytics_offline_env()

    assert os.environ["HTTPS_PROXY"] == BLACKHOLE and os.environ["https_proxy"] == BLACKHOLE
    assert os.environ["YOLO_OFFLINE"] == "1" and os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["YOLO_CONFIG_DIR"] == str(CACHE / "ultralytics-config")
    assert "ULTRALYTICS_API_KEY" not in os.environ


def test_load_yolo_refuses_missing_weights(tmp_path):
    from vc_train.offline import load_yolo

    with pytest.raises(FileNotFoundError, match=re.escape("py/README.md, section vc_train")):
        load_yolo(tmp_path / "nope.pt")


def test_train_writes_best_weights_and_refuses_to_reuse_a_run(tiny_run):
    from vc_train.train import train

    root, _, best = tiny_run
    assert best.exists() and best.name == "best.pt"
    with pytest.raises(ValueError, match="already exists"):
        train(root / "ds" / "data.yaml", root / "run", imgsz=64, epochs=1, hours=None, batch=2, device="cpu", workers=0)


def test_export_writes_static_onnx_and_model_json(tiny_run):
    import onnxruntime as ort
    import torch

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
    trained_map50 = torch.load(best, map_location="cpu", weights_only=False)["train_metrics"]["metrics/mAP50(B)"]
    assert meta["metrics"]["val_map50"] == pytest.approx(trained_map50)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 64, 64]
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].shape[1] == 6
    assert session.get_outputs()[0].type == "tensor(float)"


def test_val_map50_is_the_checkpoints_metric_not_a_results_csv_guess(tiny_run, tmp_path):
    import torch

    from vc_train.export import export_onnx, read_model_json

    _, _, best = tiny_run
    run = tmp_path / "run" / "train"
    (run / "weights").mkdir(parents=True)
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    ckpt["train_metrics"]["metrics/mAP50(B)"] = 0.4321
    torch.save(ckpt, run / "weights" / "best.pt")
    # a higher value in results.csv must not leak into the report
    (run / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n2,0.9999\n", encoding="utf-8")
    export_onnx(run / "weights" / "best.pt", tmp_path / "model", 64, 64)
    assert read_model_json(tmp_path / "model")["metrics"]["val_map50"] == pytest.approx(0.4321)


def test_export_refuses_a_checkpoint_without_map50(tmp_path):
    import torch

    from vc_train.export import export_onnx

    weights = tmp_path / "best.pt"
    torch.save({"train_metrics": {"fitness": 0.1}}, weights)
    with pytest.raises(ValueError, match=re.escape(str(weights))):
        export_onnx(weights, tmp_path / "model", 64, 64)
    assert not (tmp_path / "model").exists()


def test_export_requires_input_sizes_that_are_multiples_of_32(tmp_path):
    from vc_train.export import export_onnx

    with pytest.raises(ValueError, match="multiple of 32"):
        export_onnx(tmp_path / "missing.pt", tmp_path / "model", 100, 64)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize(
    ("shape", "elem_type", "expected"),
    [
        ([1, 3, 32, 64], "FLOAT", "[1, 3, 64, 64]"),
        ([1, 3, 64, 64], "INT64", "tensor(float)"),
    ],
)
def test_export_rejects_onnx_that_is_not_static_float_nchw(tmp_path, monkeypatch, shape, elem_type, expected):
    import onnx
    import torch
    from onnx import TensorProto, helper

    from vc_train import export as export_mod

    dtype = getattr(TensorProto, elem_type)
    bad = tmp_path / "bad.onnx"
    x = helper.make_tensor_value_info("images", dtype, shape)
    y = helper.make_tensor_value_info("output0", dtype, shape)
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output0"])], "identity", [x], [y])
    # onnx 1.23 defaults to IR 14, which the pinned onnxruntime cannot load
    onnx.save(helper.make_model(graph, ir_version=11, opset_imports=[helper.make_opsetid("", 17)]), bad)
    weights = tmp_path / "best.pt"
    torch.save({"train_metrics": {"metrics/mAP50(B)": 0.5}}, weights)
    monkeypatch.setattr(export_mod, "load_yolo", lambda _weights: _ExportStub(bad))

    with pytest.raises(ValueError, match=re.escape(expected)):
        export_mod.export_onnx(weights, tmp_path / "model", 64, 64)
    assert not (tmp_path / "model").exists()


def test_consistency_needs_at_least_one_image_to_pass():
    from vc_train.export import ConsistencyReport

    assert not ConsistencyReport(images=0, boxes=0, max_diff_px=0.0, unmatched=0).passed()
    assert ConsistencyReport(images=1, boxes=0, max_diff_px=0.0, unmatched=0).passed()


def test_pytorch_and_onnxruntime_agree(tiny_run):
    from vc_train.export import check_consistency, export_onnx

    root, (frames_dir, _), best = tiny_run
    onnx_path = export_onnx(best, root / "model_check", 64, 64)
    images = sorted(frames_dir.glob("*.jpg"))
    # threshold 0 keeps every anchor, so a score sitting on the threshold cannot make the two runtimes disagree
    report = check_consistency(best, onnx_path, images, score_threshold=0.0)
    assert report.images == len(images)
    assert report.boxes > 0
    assert report.passed(1.0), report
