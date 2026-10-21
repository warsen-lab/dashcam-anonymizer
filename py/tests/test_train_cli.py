import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train import export as export_mod
from vc_train import train as train_mod
from vc_train.cli import main
from vc_train.export import ConsistencyReport, match_boxes
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
            (frames_dir / f"{i:06d}.jpg").write_bytes(b"jpeg")  # export only checks that a chosen image exists


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
    printed = capsys.readouterr().out
    assert "dataset: 6 train / 6 val samples from 2 / 2 frames" in printed
    assert "  a: 1 train / 1 val frames" in printed and "  b: 1 train / 1 val frames" in printed
    assert (out / "data.yaml").exists()


def test_dataset_command_reports_errors_with_exit_2(tmp_path, capsys):
    code = main(["dataset", "--stem", "missing", "--frames-root", str(tmp_path), "--labels-root", str(tmp_path), "--out", str(tmp_path / "ds")])
    assert code == 2
    assert capsys.readouterr().err.startswith("vc_train: ")


@pytest.mark.parametrize("what", ["labels missing", "short segment"])
def test_dataset_command_stops_on_a_segment_it_cannot_use_and_writes_nothing(tmp_path, capsys, what):
    _segment(tmp_path, "a")
    if what == "labels missing":
        shutil.rmtree(tmp_path / "labels" / "a")
    else:
        # frame 1 moves from 30 s to 10 s: the whole segment is inside the 20 s validation tail
        text = (tmp_path / "frames" / "a" / "frames.csv").read_text().replace("30000.0", "10000.0")
        (tmp_path / "frames" / "a" / "frames.csv").write_text(text)
    out = tmp_path / "ds"
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("vc_train: ") and (str(tmp_path / "labels" / "a") in err)
    assert not out.exists()


def test_dataset_output_inside_a_frames_dir_is_refused(tmp_path, capsys):
    _segment(tmp_path, "a")
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(tmp_path / "frames" / "a" / "ds"),
    ])
    assert code == 2
    assert not (tmp_path / "frames" / "a" / "ds").exists()


def test_dataset_refuses_out_under_the_frames_root_even_for_unlisted_segments(tmp_path, capsys):
    _segment(tmp_path, "a")
    (tmp_path / "frames" / "b").mkdir()  # a segment that is not listed with --stem
    out = tmp_path / "frames" / "b" / "ds"
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out),
    ])
    assert code == 2
    assert "frames directory" in capsys.readouterr().err
    assert not out.exists()


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


@pytest.mark.parametrize("flag, value", [("--width", "100"), ("--height", "1080"), ("--width", "0")])
def test_dataset_command_wants_input_sizes_that_are_multiples_of_32(tmp_path, capsys, flag, value):
    _segment(tmp_path, "a")
    out = tmp_path / "ds"
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out), flag, value,
    ])
    assert code == 2
    assert "multiple of 32" in capsys.readouterr().err
    assert not out.exists()


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
    (dataset / "manifest.csv").write_text("sample\n")
    _frames_csv(tmp_path / "frames", 50)
    seen = {}

    def fake_export(weights, out_dir, in_width, in_height, manifest=None, name=None):
        seen.update(weights=weights, width=in_width, height=in_height, manifest=manifest, name=name)
        onnx = out_dir / "model.onnx"
        onnx.write_bytes(b"onnx")
        return onnx

    def fake_check(weights, onnx_path, images, **kwargs):
        seen.update(check_weights=weights, onnx=onnx_path, images=list(images))
        return ConsistencyReport(images=len(images), boxes=7, matched=6, max_diff_px=0.25, unmatched_confident=0, unmatched_low=1)

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
    # 50 frames, 5 checks: round(i * 49 / 4). i = 2 gives 24.5, which rounds half up to 25 (Python round() would give 24)
    assert [p.name for p in seen["images"]] == ["000000.jpg", "000012.jpg", "000025.jpg", "000037.jpg", "000049.jpg"]
    assert "export: " in out
    assert "check: 5 images, 7 boxes, 6 matched, max diff 0.250 px, 0 unmatched confident, 1 unmatched low" in out


def test_export_name_flag_overrides_the_run_dir_name(tmp_path, monkeypatch):
    _trained_run(tmp_path)
    seen = {}

    def fake_export(weights, out_dir, in_width, in_height, manifest=None, name=None):
        seen.update(name=name, manifest=manifest)
        return out_dir / "model.onnx"

    monkeypatch.setattr(export_mod, "export_onnx", fake_export)
    assert main(["export", "--run", str(tmp_path / "run"), "--name", "student-r2"]) == 0
    assert seen == {"name": "student-r2", "manifest": None}


def test_export_refuses_a_dataset_without_manifest_before_writing(tmp_path, capsys, monkeypatch):
    _trained_run(tmp_path)
    (tmp_path / "ds").mkdir()
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    code = main(["export", "--run", str(tmp_path / "run"), "--dataset", str(tmp_path / "ds")])
    assert code == 2
    assert "manifest.csv" in capsys.readouterr().err
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["train"]


def test_export_refuses_a_missing_check_image_before_writing(tmp_path, capsys, monkeypatch):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", 4)
    (tmp_path / "frames" / "000002.jpg").unlink()
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    code = main(["export", "--run", str(tmp_path / "run"), "--check-frames", str(tmp_path / "frames"), "--check-count", "10"])
    assert code == 2
    assert "000002.jpg" in capsys.readouterr().err
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["train"]


def test_export_checks_only_the_chosen_images_exist(tmp_path, monkeypatch):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", 50)
    (tmp_path / "frames" / "000001.jpg").unlink()  # not one of the two chosen frames, 0 and 49
    monkeypatch.setattr(export_mod, "export_onnx", lambda weights, out_dir, *a, **k: out_dir / "model.onnx")
    monkeypatch.setattr(
        export_mod, "check_consistency",
        lambda weights, onnx_path, images, **k: ConsistencyReport(len(images), 2, 2, 0.0, 0, 0),
    )
    assert main(["export", "--run", str(tmp_path / "run"), "--check-frames", str(tmp_path / "frames"), "--check-count", "2"]) == 0


def test_export_exits_1_when_the_consistency_check_fails(tmp_path, capsys, monkeypatch):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", 4)
    monkeypatch.setattr(export_mod, "export_onnx", lambda weights, out_dir, *a, **k: out_dir / "model.onnx")
    monkeypatch.setattr(
        export_mod, "check_consistency",
        lambda weights, onnx_path, images, **k: ConsistencyReport(
            images=len(images), boxes=3, matched=1, max_diff_px=4.5, unmatched_confident=2, unmatched_low=1
        ),
    )
    code = main(["export", "--run", str(tmp_path / "run"), "--check-frames", str(tmp_path / "frames")])
    assert code == 1
    assert "2 unmatched confident, 1 unmatched low" in capsys.readouterr().out


def _sampled_check_images(tmp_path, monkeypatch, frame_count, check_count):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", frame_count)
    seen = {}

    def fake_export(weights, out_dir, *args, **kwargs):
        onnx = out_dir / "model.onnx"
        onnx.write_bytes(b"onnx")
        return onnx

    def fake_check(weights, onnx_path, images, **kwargs):
        seen["images"] = [p.name for p in images]
        return ConsistencyReport(images=len(images), boxes=1, matched=1, max_diff_px=0.0, unmatched_confident=0, unmatched_low=0)

    monkeypatch.setattr(export_mod, "export_onnx", fake_export)
    monkeypatch.setattr(export_mod, "check_consistency", fake_check)
    code = main([
        "export", "--run", str(tmp_path / "run"),
        "--check-frames", str(tmp_path / "frames"), "--check-count", str(check_count),
    ])
    assert code == 0
    return seen["images"]


@pytest.mark.parametrize(
    "frame_count, check_count, expected",
    [
        (1, 1, [0]),
        (50, 1, [24]),  # the middle frame, (n - 1) // 2
        (3, 3, [0, 1, 2]),
        (4, 20, [0, 1, 2, 3]),  # n <= count: every frame
        (50, 5, [0, 12, 25, 37, 49]),
        (50, 20, [0, 3, 5, 8, 10, 13, 15, 18, 21, 23, 26, 28, 31, 34, 36, 39, 41, 44, 46, 49]),
    ],
)
def test_export_spreads_check_frames_evenly_over_the_whole_list(tmp_path, monkeypatch, frame_count, check_count, expected):
    names = _sampled_check_images(tmp_path, monkeypatch, frame_count, check_count)
    assert names == [f"{i:06d}.jpg" for i in expected]


@pytest.mark.parametrize("check_count", [0, -3])
def test_export_rejects_a_check_count_below_one_before_writing(tmp_path, capsys, monkeypatch, check_count):
    _frames_csv(tmp_path / "frames", 4)
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    code = main([
        "export", "--run", str(tmp_path / "run"),
        "--check-frames", str(tmp_path / "frames"), "--check-count", str(check_count),
    ])
    assert code == 2
    assert "--check-count" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_export_refuses_an_empty_check_set_instead_of_passing(tmp_path, capsys, monkeypatch):
    _trained_run(tmp_path)
    _frames_csv(tmp_path / "frames", 0)
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    code = main(["export", "--run", str(tmp_path / "run"), "--check-frames", str(tmp_path / "frames")])
    assert code == 2
    assert "no frames" in capsys.readouterr().err


def test_export_refuses_a_run_dir_inside_the_check_frames_dir(tmp_path, capsys, monkeypatch):
    frames = tmp_path / "frames" / "clip"
    _frames_csv(frames, 3)
    run = frames / "run"
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    code = main(["export", "--run", str(run), "--check-frames", str(frames)])
    assert code == 2
    assert "frames directory" in capsys.readouterr().err
    assert not run.exists()


def test_train_refuses_a_run_dir_under_the_frames_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train_mod, "train", lambda *a, **k: pytest.fail("train must not run"))
    run = tmp_path / "frames" / "any_segment" / "run"
    code = main(["train", "--data", str(tmp_path / "data.yaml"), "--run", str(run), "--frames-root", str(tmp_path / "frames")])
    assert code == 2
    assert "frames directory" in capsys.readouterr().err
    assert not run.exists()


def test_export_refuses_a_run_dir_under_the_frames_root(tmp_path, capsys, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(export_mod, "export_onnx", lambda *a, **k: pytest.fail("export must not run"))
    run = tmp_path / "frames" / "any_segment" / "run"
    code = main(["export", "--run", str(run), "--weights", str(weights), "--frames-root", str(tmp_path / "frames")])
    assert code == 2
    assert "frames directory" in capsys.readouterr().err
    assert not run.exists()


def test_frames_root_defaults_to_data_frames_for_train_and_export(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_mod, "train", lambda *a, **k: pytest.fail("train must not run"))
    assert main(["train", "--data", "d.yaml", "--run", "data/frames/seg/run"]) == 2
    assert main(["export", "--run", "data/frames/seg/run", "--weights", "w.pt"]) == 2
    assert capsys.readouterr().err.count("frames directory data/frames") == 2


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


# --- offline environment (no ultralytics needed): it must be in place before anything can import ultralytics ---

class _Reached(Exception):
    """Raised by a stub standing in for the first call that could import ultralytics."""


@pytest.fixture
def online_env(monkeypatch):
    """A shell that has not been set up for offline use: YOLO_OFFLINE unset, nothing for the code under test to rely on."""
    monkeypatch.setenv("YOLO_OFFLINE", "")
    monkeypatch.delenv("YOLO_OFFLINE")


def test_export_onnx_sets_the_offline_env_before_unpickling_the_checkpoint(tmp_path, monkeypatch, online_env):
    import torch

    seen = {}

    def fake_load(*args, **kwargs):
        # unpickling a checkpoint imports ultralytics.nn.tasks, which runs its is_online() check at import time
        seen["offline"] = os.environ.get("YOLO_OFFLINE")
        raise _Reached

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(_Reached):
        export_mod.export_onnx(tmp_path / "best.pt", tmp_path / "model", 64, 64)
    assert seen["offline"] == "1"


def test_check_consistency_sets_the_offline_env_before_loading_the_model(tmp_path, monkeypatch, online_env):
    seen = {}

    def fake_load_yolo(weights):
        seen["offline"] = os.environ.get("YOLO_OFFLINE")
        raise _Reached

    monkeypatch.setattr(export_mod, "load_yolo", fake_load_yolo)
    with pytest.raises(_Reached):
        export_mod.check_consistency(tmp_path / "best.pt", tmp_path / "model.onnx", [])
    assert seen["offline"] == "1"


def test_check_consistency_sets_the_offline_env_before_importing_anything_heavy(tmp_path, monkeypatch, online_env):
    seen = {}
    real_import = __import__

    def spy_import(name, *args, **kwargs):
        if name in ("torch", "onnxruntime"):
            seen.setdefault("offline", os.environ.get("YOLO_OFFLINE"))
        return real_import(name, *args, **kwargs)

    def fake_load_yolo(weights):
        raise _Reached

    monkeypatch.setattr("builtins.__import__", spy_import)
    monkeypatch.setattr(export_mod, "load_yolo", fake_load_yolo)
    with pytest.raises(_Reached):
        export_mod.check_consistency(tmp_path / "best.pt", tmp_path / "model.onnx", [])
    assert seen["offline"] == "1"


def test_cli_main_sets_the_offline_env_first(tmp_path, online_env):
    # a failing command still ends in main(), so the env is checked after an error exit
    assert main(["export", "--run", str(tmp_path / "run")]) == 2
    assert os.environ.get("YOLO_OFFLINE") == "1"
    assert os.environ.get("YOLO_AUTOINSTALL") == "false"


def test_no_vc_train_module_imports_ultralytics_when_imported():
    code = (
        "import sys\n"
        "import vc_train.cli, vc_train.dataset, vc_train.export, vc_train.geometry, vc_train.offline, vc_train.train\n"
        "sys.exit(1 if 'ultralytics' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(PY_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


# --- consistency report (aligned with vc_eval.consistency: only confident disagreements fail) ---

CONFIDENT_SCORE = 0.3


def _box(cls="face", x=0.0, score=0.9):
    return Box(cls, x, 0.0, x + 10.0, 10.0, score=score)


def test_match_boxes_counts_matched_boxes_and_the_worst_corner_difference():
    a = [_box(x=0.0), _box("plate", x=100.0)]
    b = [_box(x=0.5), _box("plate", x=100.0)]
    assert match_boxes(a, b) == (2, 0, 0, 0.5)


def test_match_boxes_splits_unmatched_boxes_by_score_on_either_side():
    a = [_box(x=0.0, score=0.30), _box(x=200.0, score=0.29)]  # only in `a`: one at the threshold, one below it
    b = [_box(x=400.0, score=0.9), _box(x=600.0, score=0.15)]  # only in `b`
    assert match_boxes(a, b) == (0, 2, 2, 0.0)


def test_match_boxes_does_not_match_across_classes_or_below_iou_half():
    a = [_box("face", x=0.0), _box("face", x=100.0)]
    b = [_box("plate", x=0.0), _box("face", x=108.0)]  # iou of x=100 and x=108 is 2 / 18, far below 0.5
    assert match_boxes(a, b)[:3] == (0, 4, 0)


def _report(**overrides):
    values = dict(images=1, boxes=3, matched=3, max_diff_px=0.1, unmatched_confident=0, unmatched_low=0)
    return ConsistencyReport(**{**values, **overrides})


def test_consistency_report_passes_when_everything_agrees():
    assert _report().passed()


def test_consistency_report_ignores_unmatched_low_score_boxes():
    assert _report(unmatched_low=5).passed()


def test_consistency_report_fails_on_an_unmatched_confident_box():
    assert not _report(unmatched_confident=1).passed()


def test_consistency_report_fails_without_images_or_matched_boxes():
    assert not _report(images=0).passed()
    assert not _report(boxes=0, matched=0).passed()  # nothing detected: no evidence of agreement
    assert not _report(boxes=2, matched=0, unmatched_low=2).passed()


def test_consistency_report_fails_on_a_corner_difference_of_the_tolerance_or_more():
    assert not _report(max_diff_px=1.0).passed()
    assert _report(max_diff_px=0.99).passed()
    assert _report(max_diff_px=1.5).passed(tol_px=2.0)


def test_dataset_command_cleans_plates_and_adds_hard_negatives(tmp_path, capsys):
    import json

    from vc_hardneg.candidates import Candidate, write_candidates
    from vc_hardneg.judgments import Judgment, append_judgments

    _segment(tmp_path, "a")
    lab = tmp_path / "labels" / "a"
    plate = Box("plate", 900, 500, 960, 520)  # a low-band teacher plate on frame 0, judged not real
    lines = to_yolo_lines([Box("face", 10, 10, 30, 40), plate], 1920, 1080)
    (lab / "yolo" / "000000.txt").write_text("".join(f"{line}\n" for line in lines))
    (lab / "merged.jsonl").write_text(json.dumps({"video": "a.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": 900,
                                                  "y1": 500, "x2": 960, "y2": 520, "score": 0.15, "src": "t"}) + "\n")

    def cand(cid, box):
        return Candidate(id=cid, video="a.mp4", stem="a", frame=0, cls="plate", box=box, score=0.15, src="t", track=-1,
                         dump_dir="", dump_index=-1, frame_teacher=0.0, image="", crop_origin=(0, 0))

    write_candidates(tmp_path / "lc.jsonl", [cand("p", (900, 500, 960, 520))])
    append_judgments(tmp_path / "lj.jsonl", [Judgment("p", "no", "llm:codex")])
    write_candidates(tmp_path / "hc.jsonl", [cand("h", (1500, 700, 1520, 710))])
    append_judgments(tmp_path / "hj.jsonl", [Judgment("h", "no", "llm:codex")])
    code = main([
        "dataset", "--stem", "a", "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(tmp_path / "ds"), "--label-candidates", str(tmp_path / "lc.jsonl"),
        "--label-judgments", str(tmp_path / "lj.jsonl"), "--hard-negatives", str(tmp_path / "hc.jsonl"),
        "--hard-negative-judgments", str(tmp_path / "hj.jsonl"), "--hard-negative-share", "1",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "label boxes left out by the cleaning: 1" in printed and "hard-negative samples: 1 (of 1" in printed
    assert len((tmp_path / "ds" / "labels" / "train" / "a_000000_full.txt").read_text().splitlines()) == 1


def test_dataset_command_wants_both_files_of_a_pair(tmp_path, capsys):
    _segment(tmp_path, "a")
    base = ["dataset", "--stem", "a", "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
            "--out", str(tmp_path / "ds")]
    assert main([*base, "--label-candidates", str(tmp_path / "x")]) == 2
    assert main([*base, "--hard-negatives", str(tmp_path / "x")]) == 2
    assert main([*base, "--drop-tall-low-plates"]) == 2
    assert not (tmp_path / "ds").exists()
