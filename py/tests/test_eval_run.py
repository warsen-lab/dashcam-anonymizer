"""run_all with a fake vc_redact / vc_detect: checks orchestration, summary and page, not the binaries."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, Detection, write_jsonl
from vc_eval.cli import main
from vc_eval.integrity import FFMPEG
from vc_eval.run import RunConfig, run_all

STEM = "seg"
W, H, N = 160, 120, 30
FACE = Box("face", 40, 30, 60, 60, score=0.9, src="student")
MASK = {"cls": "face", "x1": 35, "y1": 25, "x2": 65, "y2": 65, "origin": "detected"}
PLATE_MASK = {"cls": "plate", "x1": 95, "y1": 75, "x2": 155, "y2": 105, "origin": "detected"}
# both classes need reference targets or P1 cannot pass; the dense detector only ever sees the face


def _setup(root):
    videos = root / "videos"
    videos.mkdir()
    video = videos / f"{STEM}.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r=30:d=1", "-f", "lavfi", "-i", "sine=d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    frames = root / "frames" / STEM
    frames.mkdir(parents=True)
    labels = root / "labels" / STEM
    labels.mkdir(parents=True)
    with (frames / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for fr in range(0, N, 6):
            w.writerow([f"{STEM}.mp4", fr, fr * 33.3, "base", 0.0])
            cv2.imwrite(str(frames / f"{fr:06d}.jpg"), np.zeros((H, W, 3), np.uint8))
    done = list(range(0, N, 6))
    (labels / "processed.txt").write_text("".join(f"{fr}\n" for fr in done))
    teachers = [Box("face", 40, 30, 60, 60, src="teacher_retinaface"), Box("plate", 100, 80, 150, 100, src="teacher_owlv2")]
    write_jsonl(labels / "merged.jsonl", [Detection(f"{STEM}.mp4", fr, fr * 33.3, box) for fr in done for box in teachers])
    return video


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fake_runner(calls, masks=(MASK, PLATE_MASK), dense=(FACE,)):
    def run(cmd, check, capture_output, text):
        calls.append(cmd)
        tool = cmd[0].rsplit("/", 1)[-1]
        out = cmd[cmd.index("--out") + 1]
        if tool == "vc_detect":
            frames_dir = cmd[cmd.index("--frames") + 1]
            rows = _rows(f"{frames_dir}/frames.csv")
            write_jsonl(Path(out), [Detection(r["video"], int(r["frame"]), float(r["pts_ms"]), FACE) for r in rows])
            return
        video = Path(cmd[cmd.index("--input") + 1])
        d = Path(out) / video.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video, d / f"{video.stem}.redacted.mp4")
        with (d / "masks.jsonl").open("w") as f:
            for fr in range(N):
                f.write(json.dumps({"video": video.name, "frame": fr, "boxes": list(masks)}) + "\n")
        write_jsonl(d / "detections.jsonl", [Detection(video.name, fr, fr * 33.3, box) for fr in range(N) for box in dense])
        (d / "stats.json").write_text(json.dumps({
            "detector": "student/CPU", "encoder": "libx264", "elapsed_s": 1.5,
            "stages": {s: {"busy_s": 0.1, "items": N, "fps": 300.0} for s in ("decode", "detect", "track", "mosaic", "encode")},
        }))
    return run


def _python_dets(monkeypatch):
    import vc_eval.run as run_module

    monkeypatch.setattr(run_module, "python_detections", lambda model, refs: {r.frame: [FACE] for r in refs})


def _cfg(root, **kw):
    base = dict(
        stems=[STEM], model=root / "model", fps_values=[5, 10], out=root / "report",
        vc_redact=root / "bin" / "vc_redact", vc_detect=root / "bin" / "vc_detect", config=root / "redact.json",
        videos=root / "videos", frames_root=root / "frames", labels_root=root / "labels",
        redacted_root=root / "redacted", consistency_count=3, spot_frames=4,
    )
    return RunConfig(**{**base, **kw})


def _row(md, prefix):
    return next(line for line in md.splitlines() if line.startswith(prefix))


def test_run_all_sweeps_writes_summary_and_page(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []
    summary = run_all(_cfg(tmp_path), runner=_fake_runner(calls))

    redact_calls = [c for c in calls if c[0].endswith("vc_redact")]
    assert len(redact_calls) == 3  # dense + 2 fps
    assert "--detect-every-frame" in redact_calls[0]
    assert redact_calls[1][redact_calls[1].index("--detect-fps") + 1] == "5"
    assert redact_calls[2][redact_calls[2].index("--detect-fps") + 1] == "10"
    assert all("--overwrite" in c for c in redact_calls)
    assert summary["chosen_fps"] == 5
    run5 = summary["runs"][0]
    assert run5["stem"] == STEM and run5["fps"] == 5
    assert run5["coverage"]["face"]["hits"] == run5["coverage"]["face"]["total"] == 5
    assert run5["integrity"]["checks"]["frame_count"] is True
    assert run5["detector"] == "student/CPU" and run5["encoder"] == "libx264"
    assert summary["consistency"]["passed"] is True
    report = tmp_path / "report"
    assert json.loads((report / "summary.json").read_text())["chosen_fps"] == 5
    md = (report / "summary.md").read_text(encoding="utf-8")
    assert "代理验证结果" in md and "| seg | 5 |" in md
    page = (report / "index.html").read_text(encoding="utf-8")
    assert "seg.redacted.mp4" in page and page.count("<figure>") >= 4


def test_summary_marks_unjudgeable_p2_and_empty_rates(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))

    coverage = summary["runs"][0]["coverage"]
    assert coverage["weakest_cell"] is None and coverage["p2"] == "n/a" and coverage["passed"] is True
    assert summary["runs"][0]["temporal"]["plate"]["total"] == 0
    assert summary["runs"][0]["temporal"]["plate"]["value"] is None  # JSON must not carry NaN
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "n/a" in _row(md, "| seg | 5 |")  # the first table: coverage with its weakest heatmap cell
    assert "P2 无法判定" in _row(md, "- 选定检测频率")
    temporal = md.split("## 时间覆盖率")[1].split("##")[0]
    row = _row(temporal, "| seg | 5 |")
    assert "100.0%" in row and row.split("|")[4].strip() == "—"  # no plate target: a dash, not 0 %
    json.loads((tmp_path / "report" / "summary.json").read_text())  # strict JSON


def test_no_passing_rate_still_reports_and_builds_the_page_with_misses(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    # nothing masked; the dense run sees a face half off the right edge and one wholly outside the frame
    edge = Box("face", 150, 30, 190, 60, src="student")
    outside = Box("face", 300, 30, 340, 60, src="student")
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([], masks=(), dense=(FACE, edge, outside)))

    assert summary["chosen_fps"] is None
    assert [r["coverage"]["passed"] for r in summary["runs"]] == [False, False]
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "无满足 P1–P3 的频率" in md
    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    assert "5 fps" in page  # no chosen rate: the smallest one is shown
    assert page.count("<figure>") > 4 + 10  # 4 random frames, 10 uncovered reference targets, temporal misses
    assert (tmp_path / "report" / "img" / STEM).is_dir()


def test_choice_needs_every_stem_and_the_integrity_checks(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    import vc_eval.run as run_module

    real = run_module.check_integrity

    def broken_at_five(source, redacted):
        report = real(source, redacted)
        if "fps5" in redacted.parts:
            report.checks["av_sync"] = False
        return report

    monkeypatch.setattr(run_module, "check_integrity", broken_at_five)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert summary["chosen_fps"] == 10
    assert [r["integrity"]["passed"] for r in summary["runs"]] == [False, True]


def test_consistency_frames_are_spread_over_the_whole_segment(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []
    run_all(_cfg(tmp_path, consistency_count=2), runner=_fake_runner(calls))
    detect = next(c for c in calls if c[0].endswith("vc_detect"))
    rows = _rows(Path(detect[detect.index("--frames") + 1]) / "frames.csv")
    assert [int(r["frame"]) for r in rows] == [0, 12]  # 5 labelled frames, 2 wanted: not the first two in a row


def test_report_dir_inside_frames_dir_is_refused_before_any_run(tmp_path):
    _setup(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="inside the frames directory"):
        run_all(_cfg(tmp_path, out=tmp_path / "frames" / STEM / "report"), runner=_fake_runner(calls))
    assert calls == []


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"stems": []}, "no stems"),
        ({"fps_values": []}, "no detection rates"),
        ({"fps_values": [5, 0]}, "positive"),
        ({"fps_values": [5, 5.0]}, "twice"),
        ({"stems": [STEM, STEM]}, "twice"),
        ({"stems": ["missing"]}, "missing.mp4"),
    ],
)
def test_bad_configuration_is_refused_before_any_run(tmp_path, kw, message):
    _setup(tmp_path)
    calls = []
    with pytest.raises(ValueError, match=message):
        run_all(_cfg(tmp_path, **kw), runner=_fake_runner(calls))
    assert calls == []


def test_cli_coverage_and_errors(tmp_path, capsys):
    _setup(tmp_path)
    masks = tmp_path / "masks.jsonl"
    with masks.open("w") as f:
        for fr in range(N):
            f.write(json.dumps({"video": f"{STEM}.mp4", "frame": fr, "boxes": []}) + "\n")
    code = main(["coverage", "--frames", str(tmp_path / "frames" / STEM), "--labels", str(tmp_path / "labels" / STEM),
                 "--masks", str(masks), "--json", str(tmp_path / "c.json")])
    assert code == 1  # nothing masked
    out = capsys.readouterr().out
    assert "face 0/5" in out and "P2 n/a" in out
    assert json.loads((tmp_path / "c.json").read_text())["face"]["total"] == 5
    assert main(["coverage", "--frames", str(tmp_path / "nope"), "--labels", str(tmp_path), "--masks", str(masks)]) == 2
    assert "vc_eval:" in capsys.readouterr().err


def test_cli_coverage_passes_with_covering_masks(tmp_path, capsys):
    _setup(tmp_path)
    masks = tmp_path / "masks.jsonl"
    with masks.open("w") as f:
        for fr in range(N):
            f.write(json.dumps({"video": f"{STEM}.mp4", "frame": fr, "boxes": [MASK, PLATE_MASK]}) + "\n")
    args = ["coverage", "--frames", str(tmp_path / "frames" / STEM), "--labels", str(tmp_path / "labels" / STEM), "--masks", str(masks)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "face 5/5" in out and "plate 5/5" in out and "PASS" in out


def test_cli_temporal_and_integrity(tmp_path, capsys):
    video = _setup(tmp_path)
    dense = tmp_path / "dense.jsonl"
    write_jsonl(dense, [Detection(f"{STEM}.mp4", fr, fr * 33.3, FACE) for fr in range(N)])
    sparse = tmp_path / "sparse.jsonl"
    with sparse.open("w") as f:
        for fr in range(N):
            f.write(json.dumps({"video": f"{STEM}.mp4", "frame": fr, "boxes": [MASK]}) + "\n")
    assert main(["temporal", "--dense", str(dense), "--sparse", str(sparse), "--frame-height", str(H), "--json", str(tmp_path / "t.json")]) == 0
    assert "face 30/30" in capsys.readouterr().out
    assert json.loads((tmp_path / "t.json").read_text())["face"]["hits"] == 30
    assert main(["integrity", "--source", str(video), "--redacted", str(video), "--json", str(tmp_path / "i.json")]) == 0
    assert json.loads((tmp_path / "i.json").read_text())["checks"]["frame_count"] is True
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert main(["integrity", "--source", str(video), "--redacted", str(broken)]) == 1  # reported, not raised
    assert main(["integrity", "--source", str(broken), "--redacted", str(video)]) == 2  # unreadable source: an input error
    assert "vc_eval:" in capsys.readouterr().err


def test_cli_run_parses_arguments_and_maps_exit_codes(tmp_path, monkeypatch, capsys):
    import vc_eval.run as run_module

    seen = []
    outcome = {"chosen_fps": 10.0}

    def fake_run_all(cfg):
        seen.append(cfg)
        return outcome

    monkeypatch.setattr(run_module, "run_all", fake_run_all)
    out = tmp_path / "rep"
    argv = ["run", "--stems", "a,b", "--model", str(tmp_path / "m"), "--out", str(out), "--fps", "5,10,15"]
    assert main(argv) == 0
    cfg = seen[0]
    assert cfg.stems == ["a", "b"] and cfg.fps_values == [5.0, 10.0, 15.0] and cfg.out == out
    assert cfg.vc_redact == Path("build/cpp-release/vc_redact") and cfg.config == Path("configs/redact.json")
    printed = capsys.readouterr().out
    assert "10 fps" in printed and str(out / "index.html") in printed

    outcome["chosen_fps"] = None
    assert main(argv) == 1
    assert "no detection rate" in capsys.readouterr().out

    def failing(cfg):
        raise subprocess.CalledProcessError(3, ["vc_redact"], stderr="boom")

    monkeypatch.setattr(run_module, "run_all", failing)
    assert main(argv) == 2
    assert "vc_eval:" in capsys.readouterr().err
    assert main(["run", "--stems", "a", "--model", "m", "--out", str(out), "--fps", "five"]) == 2
    assert "vc_eval:" in capsys.readouterr().err
