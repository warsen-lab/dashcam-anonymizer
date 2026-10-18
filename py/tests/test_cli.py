from vc_label.cli import main
from vc_label.contracts import write_jsonl


def _frames(tmp_path):
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    (frames / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\n")
    return frames


def test_labels_inside_the_frames_dir_are_rejected(tmp_path, capsys):
    frames = _frames(tmp_path)

    assert main(["fuse", "--frames", str(frames), "--labels", str(frames / "labels")]) == 2

    assert "inside the frames directory" in capsys.readouterr().err


def test_teach_rejects_bad_every_before_loading_models(tmp_path, capsys):
    frames = _frames(tmp_path)

    assert main(["teach", "--frames", str(frames), "--labels", str(tmp_path / "labels"), "--every", "0"]) == 2

    assert "--every" in capsys.readouterr().err


def test_fuse_reports_counts(tmp_path, capsys):
    frames = _frames(tmp_path)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("")
    write_jsonl(labels / "raw.jsonl", [])

    assert main(["fuse", "--frames", str(frames), "--labels", str(labels)]) == 0

    assert "fuse: 0 frames, 0 boxes (0 temporal fills), 0 flags" in capsys.readouterr().out


def test_unknown_config_key_is_a_usage_error(tmp_path, capsys):
    frames = _frames(tmp_path)
    config = tmp_path / "bad.json"
    config.write_text('{"nope": 1}')

    assert main(["fuse", "--frames", str(frames), "--labels", str(tmp_path / "l"), "--config", str(config)]) == 2

    assert "nope" in capsys.readouterr().err


def test_teach_refuses_to_run_when_a_model_file_is_missing(tmp_path, capsys, monkeypatch):
    frames = _frames(tmp_path)
    monkeypatch.setattr("vc_label.cli.required_model_files", lambda: [tmp_path / "missing.pth"])

    assert main(["teach", "--frames", str(frames), "--labels", str(tmp_path / "labels")]) == 2

    assert "missing.pth" in capsys.readouterr().err
    assert not (tmp_path / "labels").exists()


def test_required_model_files_follow_the_cache_env(tmp_path, monkeypatch):
    import vc_label.paths

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    files = vc_label.paths.required_model_files()

    assert files[-2:] == [
        tmp_path / "torch" / "hub" / "checkpoints" / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        tmp_path
        / "hf"
        / "hub"
        / "models--google--owlv2-base-patch16-ensemble"
        / "snapshots"
        / "cfd3195ba4ea9592eec887ded089f4c08eff231d"
        / "model.safetensors",
    ]


def test_review_and_gold_inside_the_frames_dir_are_rejected(tmp_path, capsys):
    frames = _frames(tmp_path)
    f = str(frames)

    assert main(["export-review", "--frames", f, "--labels", str(tmp_path / "labels"), "--review", str(frames / "review")]) == 2
    assert "inside the frames directory" in capsys.readouterr().err

    assert main(["import-review", "--frames", f, "--review", str(tmp_path / "review"), "--gold", str(frames / "gold")]) == 2
    assert "inside the frames directory" in capsys.readouterr().err

    assert not (frames / "review").exists()
    assert not (frames / "gold").exists()


def _labelled(tmp_path):
    """A one-frame frames dir and a labels dir holding one raw face, built without running teachers."""
    import cv2
    import numpy as np

    from vc_label.contracts import Box, Detection

    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "000000.jpg"), np.full((48, 64, 3), 128, np.uint8))
    (frames / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\ntrip.mp4,0,0.0,base,0\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n")
    write_jsonl(labels / "raw.jsonl", [Detection("trip.mp4", 0, 0.0, Box("face", 10, 10, 20, 24, score=0.9, src="teacher_yunet"))])
    return frames, labels


def test_export_review_keeps_existing_files_unless_forced(tmp_path, capsys):
    frames, labels = _labelled(tmp_path)
    review = tmp_path / "review"
    argv = ["export-review", "--frames", str(frames), "--labels", str(labels), "--review", str(review)]

    assert main(argv) == 0
    assert f"export-review: 1 new LabelMe files, 0 kept in {review}" in capsys.readouterr().out

    assert main(argv) == 0
    assert f"export-review: 0 new LabelMe files, 1 kept in {review}" in capsys.readouterr().out

    assert main([*argv, "--force"]) == 0
    assert f"export-review: 1 new LabelMe files, 0 kept in {review}" in capsys.readouterr().out
