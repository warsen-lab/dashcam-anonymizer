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
