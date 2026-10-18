import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from vc_label.contracts import Box, to_yolo_lines
from vc_train import export as export_mod
from vc_train import train as train_mod
from vc_train.cli import main
from vc_train.export import ConsistencyReport
from vc_train.offline import YOLO_WEIGHTS

PY_DIR = Path(__file__).resolve().parents[1]


def _segment(root, stem):
    frames_dir, labels_dir = root / "frames" / stem, root / "labels" / stem
    frames_dir.mkdir(parents=True)
    (labels_dir / "yolo").mkdir(parents=True)
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i, pts in enumerate([0.0, 30000.0]):
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
            w.writerow([f"{stem}.mp4", i, pts, "base", 0.0])
            lines = to_yolo_lines([Box("face", 10, 10, 30, 40)], 1920, 1080)
            (labels_dir / "yolo" / f"{i:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("0\n1\n")


def _frames_csv(frames_dir, count):
    frames_dir.mkdir(parents=True)
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i in range(count):
            w.writerow(["clip.mp4", i, i * 1000.0, "base", 0.0])


def _trained_run(tmp_path):
    best = tmp_path / "run" / "train" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"weights")
    return best


def test_dataset_command_builds_from_stems(tmp_path, capsys):
    _segment(tmp_path, "a")
    _segment(tmp_path, "b")
    out = tmp_path / "ds"
    code = main([
        "dataset", "--stem", "a", "--stem", "b",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out),
    ])
    assert code == 0
    assert "dataset: 6 train / 6 val samples from 2 / 2 frames" in capsys.readouterr().out
    assert (out / "data.yaml").exists()


def test_dataset_command_reports_errors_with_exit_2(tmp_path, capsys):
    code = main(["dataset", "--stem", "missing", "--frames-root", str(tmp_path), "--labels-root", str(tmp_path), "--out", str(tmp_path / "ds")])
    assert code == 2
    assert capsys.readouterr().err.startswith("vc_train: ")


def test_dataset_output_inside_a_frames_dir_is_refused(tmp_path, capsys):
    _segment(tmp_path, "a")
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(tmp_path / "frames" / "a" / "ds"),
    ])
    assert code == 2


def test_dataset_refuses_a_non_empty_out_dir(tmp_path, capsys):
    _segment(tmp_path, "a")
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("previous run")
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out),
    ])
    assert code == 2
    assert "not empty" in capsys.readouterr().err
    assert (out / "keep.txt").read_text() == "previous run"


def test_export_without_weights_exits_2(tmp_path, capsys):
    assert main(["export", "--run", str(tmp_path / "run")]) == 2
    assert "best.pt" in capsys.readouterr().err


def test_train_command_passes_flags_and_prints_best_weights(tmp_path, capsys, monkeypatch):
    calls = {}

    def fake_train(data_yaml, run_dir, weights, **kwargs):
        calls.update(data=data_yaml, run=run_dir, weights=weights, **kwargs)
        return run_dir / "train" / "weights" / "best.pt"

    monkeypatch.setattr(train_mod, "train", fake_train)
    code = main([
        "train", "--data", str(tmp_path / "data.yaml"), "--run", str(tmp_path / "run"),
        "--hours", "0.25", "--epochs", "5", "--batch", "4", "--imgsz", "640", "--device", "cpu",
    ])
    assert code == 0
    assert calls["weights"] == YOLO_WEIGHTS
    assert (calls["hours"], calls["epochs"], calls["batch"], calls["imgsz"], calls["device"]) == (0.25, 5, 4, 640, "cpu")
    assert capsys.readouterr().out.startswith("train: best weights ")


def test_export_writes_onnx_and_passes_consistency_check(tmp_path, capsys, monkeypatch):
    best = _trained_run(tmp_path)
    dataset = tmp_path / "ds"
    dataset.mkdir()
    _frames_csv(tmp_path / "frames", 50)
    seen = {}

    def fake_export(weights, out_dir, in_width, in_height, manifest=None, name=None):
        seen.update(weights=weights, width=in_width, height=in_height, manifest=manifest, name=name)
        onnx = out_dir / "model.onnx"
        onnx.write_bytes(b"onnx")
        return onnx

    def fake_check(weights, onnx_path, images, **kwargs):
        seen.update(check_weights=weights, onnx=onnx_path, images=list(images))
        return ConsistencyReport(images=len(images), boxes=7, max_diff_px=0.25, unmatched=0)

    monkeypatch.setattr(export_mod, "export_onnx", fake_export)
    monkeypatch.setattr(export_mod, "check_consistency", fake_check)
    code = main([
        "export", "--run", str(tmp_path / "run"), "--dataset", str(dataset),
        "--check-frames", str(tmp_path / "frames"), "--check-count", "5",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert seen["weights"] == best
    assert (seen["width"], seen["height"]) == (1920, 1088)
    assert seen["manifest"] == dataset / "manifest.csv"
    assert seen["name"] == "run"
    # 50 frames, 5 checks: every 10th frame
    assert [p.name for p in seen["images"]] == ["000000.jpg", "000010.jpg", "000020.jpg", "000030.jpg", "000040.jpg"]
    assert "export: " in out
    assert "check: 5 images, 7 boxes, max diff 0.250 px, 0 unmatched" in out


def test_export_exits_1_when_the_consistency_check_fails(tmp_path, capsys, monkeypatch):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", 4)
    monkeypatch.setattr(export_mod, "export_onnx", lambda weights, out_dir, *a, **k: out_dir / "model.onnx")
    monkeypatch.setattr(
        export_mod, "check_consistency",
        lambda weights, onnx_path, images, **k: ConsistencyReport(images=len(images), boxes=3, max_diff_px=4.5, unmatched=2),
    )
    code = main(["export", "--run", str(tmp_path / "run"), "--check-frames", str(tmp_path / "frames")])
    assert code == 1
    assert "2 unmatched" in capsys.readouterr().out


def test_python_dash_m_runs_the_cli(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "vc_train", "--help"],
        env={"PYTHONPATH": str(PY_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "dataset" in result.stdout and "export" in result.stdout
