"""run_all with a fake vc_redact / vc_detect: checks orchestration, summary and page, not the binaries."""

import csv
import dataclasses
import json
import os
import re
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
TEACHERS = (Box("face", 40, 30, 60, 60, src="teacher_retinaface"), Box("plate", 100, 80, 150, 100, src="teacher_owlv2"))
GRID_FRAMES = list(range(0, N, 3))  # 10 labelled frames x 10 faces and 10 plates = 100 targets per class and segment


def _grid_targets():
    """10 faces (heatmap rows 0-1) and 10 plates (rows 2-3) per frame, one per heatmap cell, so no cell reaches 30 targets."""
    faces = [Box("face", 27 * (p % 5) + 6, 30 * (p // 5) + 8, 27 * (p % 5) + 20, 30 * (p // 5) + 22, src="teacher_retinaface") for p in range(10)]
    plates = [Box("plate", 27 * (p % 5) + 4, 30 * (p // 5 + 2) + 10, 27 * (p % 5) + 24, 30 * (p // 5 + 2) + 18, src="teacher_owlv2") for p in range(10)]
    return faces + plates


def _grid_masks(missed):
    """Fake masks(stem, frame): a padded mask on every grid target except the (stem, frame, target index) triples in `missed`."""
    targets = _grid_targets()

    def masks(stem, frame):
        return [
            {"cls": t.cls, "x1": t.x1 - 2, "y1": t.y1 - 2, "x2": t.x2 + 2, "y2": t.y2 + 2, "origin": "detected"}
            for i, t in enumerate(targets)
            if (stem, frame, i) not in missed
        ]

    return masks


def _setup(root, stem=STEM, labelled=range(0, N, 6), teachers=TEACHERS):
    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    video = videos / f"{stem}.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r=30:d=1", "-f", "lavfi", "-i", "sine=d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    frames = root / "frames" / stem
    frames.mkdir(parents=True)
    labels = root / "labels" / stem
    labels.mkdir(parents=True)
    with (frames / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for fr in labelled:
            w.writerow([f"{stem}.mp4", fr, fr * 33.3, "base", 0.0])
            cv2.imwrite(str(frames / f"{fr:06d}.jpg"), np.zeros((H, W, 3), np.uint8))
    done = list(labelled)
    (labels / "processed.txt").write_text("".join(f"{fr}\n" for fr in done))
    write_jsonl(labels / "merged.jsonl", [Detection(f"{stem}.mp4", fr, fr * 33.3, box) for fr in done for box in teachers])
    return video


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fake_runner(calls, masks=(MASK, PLATE_MASK), dense=(FACE,)):
    """`masks` is a list of mask dicts for every frame, or a function (stem, frame) -> list of mask dicts."""
    masks_of = masks if callable(masks) else (lambda stem, frame: list(masks))

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
                f.write(json.dumps({"video": video.name, "frame": fr, "boxes": masks_of(video.stem, fr)}) + "\n")
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
    assert "## 速度（本机，160×120，源视频 h264）" in md  # frame size from the coverage report, codec from the source
    assert summary["segments"] == [{"stem": STEM, "width": W, "height": H, "codec": "h264"}]
    pooled = summary["pooled"]
    assert [p["fps"] for p in pooled] == [5, 10] and pooled[0]["stems"] == [STEM]
    assert pooled[0]["coverage"]["face"] == run5["coverage"]["face"]  # one segment: the pool is that segment
    assert pooled[0]["passed"] is True
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
    assert "P2 空洞满足" in _row(md, "- 选定检测频率")
    coverage_table = md.split("## 最终覆盖率")[1].split("##")[0]
    assert "空洞满足" in coverage_table and "P4" in coverage_table and "没有热力格达到 30 个目标" in coverage_table
    assert "不算通过" not in md
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
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    integrity = md.split("## 输出完整性")[1].split("##")[0]
    assert _row(integrity, "| seg | 5 |").rstrip().endswith("| av_sync: audio offset changed by 0.0 ms (limit 33.3 ms) |")  # the check name leads each detail


def _grid_run(tmp_path, monkeypatch, missed_faces=0, **kw):
    """Two segments with 100 faces and 100 plates each; the second misses `missed_faces` faces. Returns the summary."""
    for stem in ("sega", "segb"):
        _setup(tmp_path, stem, GRID_FRAMES, _grid_targets())
    _python_dets(monkeypatch)
    missed = {("segb", GRID_FRAMES[i], i) for i in range(missed_faces)}  # target i < 10 is a face
    cfg = _cfg(tmp_path, stems=["sega", "segb"], fps_values=[5, 10], **kw)
    return run_all(cfg, runner=_fake_runner([], masks=_grid_masks(missed), dense=()))


@pytest.mark.parametrize("missed_faces, chosen", [(0, 5), (1, 5), (2, 5), (3, None)])
def test_p1_is_judged_on_the_pooled_segments(tmp_path, monkeypatch, missed_faces, chosen):
    summary = _grid_run(tmp_path, monkeypatch, missed_faces)

    by_stem = {r["stem"]: r["coverage"] for r in summary["runs"] if r["fps"] == 5}
    assert (by_stem["sega"]["face"]["hits"], by_stem["sega"]["face"]["total"]) == (100, 100)
    assert (by_stem["segb"]["face"]["hits"], by_stem["segb"]["face"]["total"]) == (100 - missed_faces, 100)
    assert by_stem["segb"]["passed"] is (missed_faces <= 1)  # on its own the second segment fails from 2 misses on
    pooled = [p for p in summary["pooled"] if p["fps"] == 5][0]
    assert pooled["stems"] == ["sega", "segb"]
    assert (pooled["coverage"]["face"]["hits"], pooled["coverage"]["face"]["total"]) == (200 - missed_faces, 200)
    assert (pooled["coverage"]["plate"]["hits"], pooled["coverage"]["plate"]["total"]) == (200, 200)
    assert pooled["coverage"]["weakest_cell"] is None and pooled["coverage"]["p2"] == "n/a"  # 20 targets per cell: vacuous
    assert pooled["coverage"]["passed"] is (missed_faces <= 2)  # 198 / 200 is exactly 99 %, 197 / 200 is not
    assert summary["chosen_fps"] == chosen

    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    table = md.split("## 最终覆盖率")[1].split("##")[0]
    assert _row(table, "| 合计 | 5 |").rstrip().endswith("| 是 |" if missed_faces <= 2 else "| 否 |")
    assert "n=200" in _row(table, "| 合计 | 5 |")
    for stem in ("sega", "segb"):
        assert _row(table, f"| {stem} | 5 |").rstrip().endswith("| — |")  # only the pooled row carries the verdict
    assert "- 选定检测频率：" + ("5 fps" if chosen else "无满足") in md


def test_integrity_must_pass_for_every_segment_not_just_the_pool(tmp_path, monkeypatch):
    import vc_eval.run as run_module

    real = run_module.check_integrity

    def segb_broken_at_five(source, redacted):
        report = real(source, redacted)
        if "fps5" in redacted.parts and redacted.parent.name == "segb":
            report.checks["duration"] = False
        return report

    monkeypatch.setattr(run_module, "check_integrity", segb_broken_at_five)
    summary = _grid_run(tmp_path, monkeypatch)
    assert [p["coverage"]["passed"] for p in summary["pooled"]] == [True, True]  # coverage alone would take 5
    assert [p["passed"] for p in summary["pooled"]] == [False, True]
    assert summary["chosen_fps"] == 10


def test_segments_of_different_frame_sizes_are_refused_before_any_run(tmp_path, monkeypatch):
    _setup(tmp_path, "sega", GRID_FRAMES, _grid_targets())
    _setup(tmp_path, "segb", GRID_FRAMES, _grid_targets())
    _python_dets(monkeypatch)
    for n in GRID_FRAMES:  # segb's reference frames are 320 x 240: its coverage cannot be pooled with sega's
        cv2.imwrite(str(tmp_path / "frames" / "segb" / f"{n:06d}.jpg"), np.zeros((2 * H, 2 * W, 3), np.uint8))
    calls = []
    with pytest.raises(ValueError, match="different frame sizes"):
        run_all(_cfg(tmp_path, stems=["sega", "segb"], fps_values=[5]), runner=_fake_runner(calls, masks=_grid_masks(set()), dense=()))
    assert calls == []


def _break_labels(labels, kind):
    if kind == "stale merged":
        t = (labels / "processed.txt").stat().st_mtime - 100
        os.utime(labels / "merged.jsonl", (t, t))
    elif kind == "missing merged":
        (labels / "merged.jsonl").unlink()
    elif kind == "missing processed":
        (labels / "processed.txt").unlink()
    elif kind == "labels for other frames":
        with (labels / "processed.txt").open("a") as f:
            f.write("99\n")  # a frame number frames.csv does not have
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind, message",
    [
        ("stale merged", "older than processed.txt"),
        ("missing merged", "vc_label fuse"),
        ("missing processed", "processed.txt does not exist"),
        ("labels for other frames", "does not match"),
    ],
)
def test_bad_label_inputs_of_any_segment_are_refused_before_any_run(tmp_path, kind, message):
    _setup(tmp_path)
    _setup(tmp_path, "seg2")
    _break_labels(tmp_path / "labels" / "seg2", kind)  # the second segment: every stem is checked up front
    calls = []
    with pytest.raises(ValueError, match=message):
        run_all(_cfg(tmp_path, stems=[STEM, "seg2"]), runner=_fake_runner(calls))
    assert calls == []


def test_consistency_runs_before_the_sweeps(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []
    run_all(_cfg(tmp_path), runner=_fake_runner(calls))
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["vc_detect", "vc_redact", "vc_redact", "vc_redact"]


def test_a_failing_consistency_run_stops_before_any_sweep(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []

    def broken_detect(cmd, check, capture_output, text):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd, stderr="cannot load model")

    with pytest.raises(ValueError, match="cannot load model"):
        run_all(_cfg(tmp_path), runner=broken_detect)
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["vc_detect"]  # no vc_redact minutes were spent


def _redact_fails(calls, stderr, after=0):
    """vc_detect works; the (after+1)-th vc_redact call exits 3 with `stderr`."""
    ok = _fake_runner(calls)
    seen = []

    def run(cmd, check, capture_output, text):
        if cmd[0].endswith("vc_redact"):
            seen.append(cmd)
            if len(seen) > after:
                calls.append(cmd)
                raise subprocess.CalledProcessError(3, cmd, stderr=stderr)
        return ok(cmd, check, capture_output, text)

    return run


def test_a_failing_vc_redact_is_a_value_error_carrying_its_stderr(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    with pytest.raises(ValueError, match="boom") as info:
        run_all(_cfg(tmp_path), runner=_redact_fails([], "boom"))
    message = str(info.value)
    assert "vc_redact failed" in message and STEM in message and "exit 3" in message and "--detect-every-frame" in message
    # a later run names its own detection rate
    with pytest.raises(ValueError, match="boom") as info:
        run_all(_cfg(tmp_path), runner=_redact_fails([], "boom", after=2))
    assert "--detect-fps 10" in str(info.value)


def test_a_failing_vc_redact_reports_only_the_last_twenty_stderr_lines(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    stderr = "".join(f"log line {n}\n" for n in range(1, 31))
    with pytest.raises(ValueError) as info:
        run_all(_cfg(tmp_path), runner=_redact_fails([], stderr))
    message = str(info.value)
    assert "log line 30" in message and "log line 11" in message
    assert not re.search(r"log line (10|[1-9])\b", message)  # the first ten lines are cut


def test_cli_run_reports_a_failing_vc_redact_as_exit_2(tmp_path, monkeypatch, capsys):
    import vc_eval.run as run_module

    _setup(tmp_path)
    _python_dets(monkeypatch)
    real = run_module.run_all
    roots = dict(
        videos=tmp_path / "videos", frames_root=tmp_path / "frames", labels_root=tmp_path / "labels", redacted_root=tmp_path / "redacted",
    )
    # the CLI has no flags for the data roots or a runner: point its RunConfig at the temp tree and fake the binaries
    monkeypatch.setattr(run_module, "run_all", lambda cfg: real(dataclasses.replace(cfg, **roots), runner=_redact_fails([], "boom")))
    assert main(["run", "--stems", STEM, "--model", str(tmp_path / "model"), "--out", str(tmp_path / "report"), "--fps", "5"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vc_eval:") and "boom" in err


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
