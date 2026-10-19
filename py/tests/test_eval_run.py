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
from vc_eval import integrity
from vc_eval.integrity import FFMPEG
from vc_eval.metrics import size_bin_labels, wilson_interval
from vc_eval.run import RunConfig, run_all

STEM = "seg"
W, H, N = 160, 120, 30
FACE = Box("face", 40, 30, 60, 60, score=0.9, src="student")
MASK = {"cls": "face", "x1": 35, "y1": 25, "x2": 65, "y2": 65, "origin": "detected"}
PLATE_MASK = {"cls": "plate", "x1": 95, "y1": 75, "x2": 155, "y2": 105, "origin": "detected"}
# both classes need reference targets or P1 cannot pass; the dense detector only ever sees the face
TEACHERS = (Box("face", 40, 30, 60, 60, src="teacher_retinaface"), Box("plate", 100, 80, 150, 100, src="teacher_owlv2"))
# a teacher box the fuse step flagged (low_score) in every labelled frame: an uncertain reference, away from MASK and PLATE_MASK
UNCERTAIN = Box("face", 100, 10, 130, 50, score=0.35, src="teacher_yunet")
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


def _setup(root, stem=STEM, labelled=range(0, N, 6), teachers=TEACHERS, flagged=()):
    """Reference side of one segment; `flagged` are boxes of `teachers` that flags.csv marks low_score in every labelled frame."""
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
    with (labels / "flags.csv").open("w", newline="") as f:  # after merged.jsonl, like vc_label fuse; just the header with no flags
        w = csv.writer(f)
        w.writerow(["video", "frame", "rule", "cls", "x1", "y1", "x2", "y2", "score", "detail"])
        for fr in done:
            for b in flagged:
                w.writerow([f"{stem}.mp4", fr, "low_score", b.cls, b.x1, b.y1, b.x2, b.y2, b.score, f"score {b.score:.2f}"])
    return video


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


STAGES = ("decode", "main", "detect", "track", "mosaic", "encode")


def _fake_runner(calls, masks=(MASK, PLATE_MASK), dense=(FACE,), busy=None, stats_extra=None):
    """`masks` is a list of mask dicts for every frame, or a function (stem, frame) -> list of mask dicts.

    `busy` maps a stage to its busy seconds in stats.json (0.1 for the stages it does not name, None leaves the stage out);
    `stats_extra` adds top-level fields to stats.json."""
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
            "frames": N, "detect_every_frame": "--detect-every-frame" in cmd,
            "detector": "student/CPU", "encoder": "libx264", "elapsed_s": 1.5,
            "stages": {s: {"busy_s": b, "items": N, "fps": 300.0} for s in STAGES if (b := (busy or {}).get(s, 0.1)) is not None},
            **(stats_extra or {}),
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
    assert "P2 空洞满足" in _row(md, "- 候选检测频率")
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
    assert "| 分辨率 |" in integrity and _row(integrity, "| seg | 10 |").count("✓") == 6
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
    assert "- 候选检测频率：" + ("5 fps（P1–P3 通过；P4 待你看完 index.html 确认）" if chosen else "无满足 P1–P3") in md


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
    elif kind == "missing flags":
        (labels / "flags.csv").unlink()
    elif kind == "stale flags":
        t = (labels / "processed.txt").stat().st_mtime - 100
        os.utime(labels / "flags.csv", (t, t))
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
        ("stale flags", r"flags\.csv is older than processed\.txt"),
        ("missing flags", r"flags\.csv does not exist; run vc_label fuse"),
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


def _section(page, title):
    """The part of the spot-check page under the <h2> `title`."""
    return page.split(f"<h2>{title}</h2>")[1].split("<h2>")[0]


def test_uncertain_misses_never_gate_but_are_reported_and_shown_on_their_own(tmp_path, monkeypatch):
    _setup(tmp_path, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))

    assert summary["chosen_fps"] == 5  # the confident tier is fully masked; the uncovered flagged face does not gate
    coverage = summary["runs"][0]["coverage"]
    assert (coverage["face"]["hits"], coverage["face"]["total"]) == (5, 5)  # P1 counts the confident face only
    assert (coverage["uncertain"]["face"]["hits"], coverage["uncertain"]["face"]["total"]) == (0, 5)
    assert coverage["uncertain"]["plate"]["total"] == 0 and coverage["uncertain"]["plate"]["value"] is None
    assert coverage["passed"] is True and summary["pooled"][0]["coverage"]["uncertain"] == coverage["uncertain"]

    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "P1 / P2 只按高置信参照" in md
    low = md.split("## 低置信参照")[1].split("##")[0]
    assert "只报告" in low and "不设门槛" in low
    row = _row(low, "| seg | 5 |")
    assert "0.0%" in row and "n=5" in row and row.split("|")[4].strip() == "—"  # no flagged plate: a dash
    assert "n=5" in _row(low, "| 合计 | 5 |")
    table = md.split("## 最终覆盖率")[1].split("##")[0]
    assert "100.0%" in _row(table, "| seg | 5 |") and _row(table, "| 合计 | 5 |").rstrip().endswith("| 是 |")

    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    titles = re.findall(r"<h2>(.*?)</h2>", page)
    assert titles.index("低置信参照里没遮住的目标") == titles.index("未遮住的参照目标") + 1  # right after the confident misses
    assert "<figure>" not in _section(page, "未遮住的参照目标")  # no confident miss
    uncertain = _section(page, "低置信参照里没遮住的目标")
    assert uncertain.count("<figure>") == 5 and "误检" in uncertain and "模糊" in uncertain  # the note: false positives and real plates
    assert "帧 12 face 覆盖 0%" in uncertain and "分数 0.35" in uncertain
    assert len(list((tmp_path / "report" / "img" / STEM).glob("uncertain_*.jpg"))) == 5


def test_pooled_uncertain_rates_add_up_over_the_segments(tmp_path, monkeypatch):
    for stem in ("sega", "segb"):
        _setup(tmp_path, stem, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))
    _python_dets(monkeypatch)
    covers_flagged = {"cls": "face", "x1": 95, "y1": 5, "x2": 135, "y2": 55, "origin": "detected"}
    masks = lambda stem, frame: [MASK, PLATE_MASK] + ([covers_flagged] if stem == "sega" else [])  # noqa: E731
    summary = run_all(_cfg(tmp_path, stems=["sega", "segb"], fps_values=[5]), runner=_fake_runner([], masks=masks))

    by_stem = {r["stem"]: r["coverage"]["uncertain"]["face"] for r in summary["runs"]}
    assert (by_stem["sega"]["hits"], by_stem["sega"]["total"]) == (5, 5)
    assert (by_stem["segb"]["hits"], by_stem["segb"]["total"]) == (0, 5)
    pooled = summary["pooled"][0]["coverage"]["uncertain"]["face"]
    assert (pooled["hits"], pooled["total"]) == (5, 10)
    assert summary["chosen_fps"] == 5
    low = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("## 低置信参照")[1].split("##")[0]
    assert "n=10" in _row(low, "| 合计 | 5 |")
    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    assert _section(page, "低置信参照里没遮住的目标").count("<figure>") == 5  # only segb's misses


def test_uncertain_tiles_are_capped_at_sixty_and_spread_over_the_segment(tmp_path, monkeypatch):
    flagged = tuple(Box("face", 5 + 50 * k, 2, 25 + 50 * k, 22, score=0.3, src="teacher_yunet") for k in range(3))  # clear of both masks
    _setup(tmp_path, labelled=range(N), teachers=(*TEACHERS, *flagged), flagged=flagged)  # 30 frames x 3 = 90 uncertain misses
    _python_dets(monkeypatch)
    run_all(_cfg(tmp_path, fps_values=[5]), runner=_fake_runner([]))

    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    uncertain = _section(page, "低置信参照里没遮住的目标")
    assert uncertain.count("<figure>") == 60
    frames = [int(n) for n in re.findall(r"帧 (\d+) ", uncertain)]
    assert min(frames) == 0 and max(frames) >= 28  # spread over the whole segment, not the first 20 frames
    assert len(list((tmp_path / "report" / "img" / STEM).glob("uncertain_*.jpg"))) == 60


def test_each_video_is_probed_once_per_run_and_the_spot_check_reuses_the_frame_count(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    real = subprocess.run
    probed = []

    def spy(cmd, *args, **kwargs):
        if cmd[0] == integrity.FFPROBE:  # `-count_frames` decodes the whole file
            probed.append(cmd[-1])
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(integrity.subprocess, "run", spy)
    summary = run_all(_cfg(tmp_path, fps_values=[5, 10, 15]), runner=_fake_runner([]))
    assert len(probed) == len(set(probed)) == 4  # the source, and the redacted video of each of the 3 rates
    assert [r["integrity"]["redacted_frames"] for r in summary["runs"]] == [N, N, N]


def test_the_full_resolution_redacted_frames_are_deleted_once_the_tiles_are_written(tmp_path, monkeypatch):
    _setup(tmp_path, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))
    _python_dets(monkeypatch)
    run_all(_cfg(tmp_path), runner=_fake_runner([], dense=(FACE, Box("face", 100, 10, 130, 50, src="student"))))
    img = tmp_path / "report" / "img" / STEM
    assert not (img / "redacted").exists()
    assert list(img.glob("random_*.jpg")) and list(img.glob("uncertain_*.jpg")) and list(img.glob("temporal_*.jpg"))
    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    assert all((tmp_path / "report" / src).is_file() for src in re.findall(r'<img src="([^"]+)"', page))


def test_the_extracted_frames_are_deleted_even_when_a_tile_fails(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    import vc_eval.run as run_module

    def broken(*args, **kwargs):
        raise ValueError("cannot write tile")

    monkeypatch.setattr(run_module, "thumbnail", broken)
    with pytest.raises(ValueError, match="cannot write tile"):
        run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert not (tmp_path / "report" / "img" / STEM / "redacted").exists()
    assert (tmp_path / "report" / "summary.md").is_file()  # the numbers were written first


def _coverage_section(md):
    return md.split("## 最终覆盖率")[1].split("\n## ")[0]


def test_a_passing_class_with_too_few_targets_gets_a_sample_size_warning_under_the_pooled_row(tmp_path, monkeypatch):
    _setup(tmp_path)  # 5 faces and 5 plates, all masked: 100 % of 5 passes P1 but proves little
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert summary["chosen_fps"] == 5
    lo = f"{wilson_interval(5, 5)[0]:.1%}"
    lines = _coverage_section((tmp_path / "report" / "summary.md").read_text(encoding="utf-8")).splitlines()
    for cls in ("人脸", "车牌"):
        warning = f"- {cls} n=5：即使全部遮住，95% 区间下限只有 {lo}，样本不足以证明 ≥ 99%"
        assert warning in lines
    pooled = next(n for n, line in enumerate(lines) if line.startswith("| 合计 | 5 |"))
    assert lines[pooled + 1] == "" and lines[pooled + 2].startswith("- 人脸 n=5：") and lines[pooled + 3].startswith("- 车牌 n=5：")
    pooled10 = next(n for n, line in enumerate(lines) if line.startswith("| 合计 | 10 |"))
    assert lines[pooled10 + 2].startswith("- 人脸 n=5：")  # every detection rate has its own pooled row with its own warning
    chosen = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("- 候选检测频率")[1].splitlines()[0]
    assert "样本量不足以证明" in chosen and "人脸" in chosen and "车牌" in chosen


def test_the_warning_says_what_was_measured_when_not_everything_was_covered(tmp_path, monkeypatch):
    summary = _grid_run(tmp_path, monkeypatch, missed_faces=2)  # 198 / 200 faces: exactly 99 %, a pass
    assert summary["chosen_fps"] == 5
    lo = f"{wilson_interval(198, 200)[0]:.1%}"
    lines = _coverage_section((tmp_path / "report" / "summary.md").read_text(encoding="utf-8")).splitlines()
    assert f"- 人脸 n=200：实测 198/200，95% 区间下限只有 {lo}，样本不足以证明 ≥ 99%" in lines
    assert f"- 车牌 n=200：即使全部遮住，95% 区间下限只有 {f'{wilson_interval(200, 200)[0]:.1%}'}，样本不足以证明 ≥ 99%" in lines


def test_no_sample_size_warning_for_a_class_that_fails_or_has_no_targets(tmp_path, monkeypatch):
    summary = _grid_run(tmp_path, monkeypatch, missed_faces=3)  # 197 / 200: P1 fails, so there is nothing to caveat
    assert summary["chosen_fps"] is None
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "- 人脸 n=" not in md and "- 车牌 n=200：" in md  # the plates still pass with too few targets
    assert "样本量不足以证明" not in md.split("- 候选检测频率")[1].splitlines()[0]


def test_the_candidate_line_and_the_cli_do_not_claim_more_than_p1_to_p3(tmp_path, monkeypatch, capsys):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    import vc_eval.run as run_module

    real = run_module.run_all
    roots = dict(videos=tmp_path / "videos", frames_root=tmp_path / "frames", labels_root=tmp_path / "labels", redacted_root=tmp_path / "redacted")
    monkeypatch.setattr(run_module, "run_all", lambda cfg: real(dataclasses.replace(cfg, **roots), runner=_fake_runner([])))
    assert main(["run", "--stems", STEM, "--model", str(tmp_path / "model"), "--out", str(tmp_path / "report"), "--fps", "5,10"]) == 0
    out = capsys.readouterr().out
    assert "run: candidate 5 fps (P1-P3 passed; P4: review index.html)" in out and "chosen" not in out
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "- 候选检测频率：5 fps（P1–P3 通过；P4 待你看完 index.html 确认）" in md and "选定" not in md
    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    assert "候选检测频率 5 fps" in page and "P4" in page


def test_the_summary_says_pixel_level_masking_is_not_what_these_numbers_measure(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    run_all(_cfg(tmp_path), runner=_fake_runner([]))
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    line = next(line for line in md.splitlines() if "masks.jsonl" in line and "像素" in line)
    assert "C++" in line and "P4" in line and "不是" in line


def test_the_summary_lists_how_much_of_the_reference_is_confident_and_how_much_is_gated_out(tmp_path, monkeypatch):
    _setup(tmp_path, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))  # 5 confident + 5 uncertain faces, 5 confident plates
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert summary["reference"] == {
        "face": {"confident": 5, "uncertain": 5, "uncertain_share": 0.5},
        "plate": {"confident": 5, "uncertain": 0, "uncertain_share": 0.0},
    }
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    tiers = md.split("## 参照分层")[1].split("\n## ")[0]
    assert "| 人脸 | 5（50.0%） | 5（50.0%） | 10 |" in tiers and "| 车牌 | 5（100.0%） | 0（0.0%） | 5 |" in tiers


def test_reference_tiers_are_pooled_over_the_segments_and_a_class_without_targets_has_no_share(tmp_path, monkeypatch):
    for stem in ("sega", "segb"):
        _setup(tmp_path, stem, teachers=(TEACHERS[0], UNCERTAIN), flagged=(UNCERTAIN,))  # no plates at all
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path, stems=["sega", "segb"], fps_values=[5]), runner=_fake_runner([]))
    assert summary["reference"]["face"] == {"confident": 10, "uncertain": 10, "uncertain_share": 0.5}
    assert summary["reference"]["plate"] == {"confident": 0, "uncertain": 0, "uncertain_share": None}
    tiers = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("## 参照分层")[1].split("\n## ")[0]
    assert "| 车牌 | 0 | 0 | 0 |" in tiers


def test_every_miss_section_says_how_many_there_are_and_how_many_are_shown(tmp_path, monkeypatch):
    flagged = tuple(Box("face", 5 + 50 * k, 2, 25 + 50 * k, 22, score=0.3, src="teacher_yunet") for k in range(3))
    _setup(tmp_path, labelled=range(N), teachers=(*TEACHERS, *flagged), flagged=flagged)  # 90 uncertain misses
    _python_dets(monkeypatch)
    edge = Box("face", 5, 90, 25, 110, src="student")  # the dense run also sees a face nothing masks: 30 temporal misses
    summary = run_all(_cfg(tmp_path, fps_values=[5]), runner=_fake_runner([], dense=(FACE, edge)))

    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    assert "seg：共 0 个，均匀抽取展示 0 个" in _section(page, "未遮住的参照目标")
    assert "seg：共 90 个，均匀抽取展示 60 个" in _section(page, "低置信参照里没遮住的目标")
    assert "seg：共 30 个，均匀抽取展示 30 个" in _section(page, "逐帧检测有、稀疏打码漏掉")
    assert summary["runs"][0]["misses"] == {"confident": 0, "uncertain": 90, "temporal": 30}
    assert summary["pooled"][0]["misses"] == {"confident": 0, "uncertain": 90, "temporal": 30}
    assert summary["spot_check"] == {
        "fps": 5,
        "stems": {STEM: {"confident": {"total": 0, "shown": 0}, "uncertain": {"total": 90, "shown": 60}, "temporal": {"total": 30, "shown": 30}}},
    }
    assert len(list((tmp_path / "report" / "img" / STEM).glob("uncertain_*.jpg"))) == 60


def test_miss_notes_list_every_segment_with_its_own_counts(tmp_path, monkeypatch):
    for stem in ("sega", "segb"):
        _setup(tmp_path, stem, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))
    _python_dets(monkeypatch)
    covers_flagged = {"cls": "face", "x1": 95, "y1": 5, "x2": 135, "y2": 55, "origin": "detected"}
    masks = lambda stem, frame: [MASK, PLATE_MASK] + ([covers_flagged] if stem == "sega" else [])  # noqa: E731
    summary = run_all(_cfg(tmp_path, stems=["sega", "segb"], fps_values=[5]), runner=_fake_runner([], masks=masks))
    page = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    note = _section(page, "低置信参照里没遮住的目标")
    assert "sega：共 0 个，均匀抽取展示 0 个" in note and "segb：共 5 个，均匀抽取展示 5 个" in note
    assert summary["spot_check"]["stems"]["segb"]["uncertain"] == {"total": 5, "shown": 5}


def test_summary_json_carries_the_full_heatmap_of_every_coverage_entry(tmp_path, monkeypatch):
    summary = _grid_run(tmp_path, monkeypatch, missed_faces=1)
    for entry in (summary["runs"][0]["coverage"], summary["pooled"][0]["coverage"]):
        face = entry["heatmap"]["face"]
        assert len(face) == 4 and all(len(row) == 6 for row in face)
        assert all(set(cell) == {"hits", "total", "value", "ci"} for row in face for cell in row)
    face = summary["pooled"][0]["coverage"]["heatmap"]["face"]
    assert [[cell["total"] for cell in row] for row in face] == [[20] * 5 + [0], [20] * 5 + [0], [0] * 6, [0] * 6]
    assert sum(cell["hits"] for row in face for cell in row) == 199
    assert face[0][0]["value"] is not None and face[3][5]["value"] is None  # an empty cell is null, never NaN
    plate = summary["pooled"][0]["coverage"]["heatmap"]["plate"]
    assert [[cell["total"] for cell in row] for row in plate] == [[0] * 6, [0] * 6, [20] * 5 + [0], [20] * 5 + [0]]
    json.loads((tmp_path / "report" / "summary.json").read_text())


def test_summary_md_details_the_candidate_rate_with_bins_fill_best_effort_and_the_heatmap(tmp_path, monkeypatch):
    _grid_run(tmp_path, monkeypatch, missed_faces=1)
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    detail = md.split("## 细分")[1].split("\n## ")[0]
    assert detail.splitlines()[0].startswith("（候选 5 fps")
    assert "尺寸档" in detail and "时间补洞" in detail and "尽力遮挡" in detail and "热力图" in detail
    for label in size_bin_labels(H):
        assert f"| {label} |" in detail  # one row per size bin
    assert _row(detail, f"| {size_bin_labels(H)[-1]} |").count("n=200") == 2  # the 14 px faces and 20 px plates all sit in the largest bin
    grid = [line for line in detail.splitlines() if line.startswith("| r")]
    assert len(grid) == 8  # 4 rows for the faces, 4 for the plates
    assert grid[0] == "| r0 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | — |"
    assert grid[2] == "| r2 | — | — | — | — | — | — |" and grid[4] == "| r0 | — | — | — | — | — | — |" and grid[6].startswith("| r2 | 20/20 |")
    assert "| 行 | c0 | c1 | c2 | c3 | c4 | c5 |" in detail


def test_summary_md_details_the_smallest_rate_when_none_passes(tmp_path, monkeypatch):
    _grid_run(tmp_path, monkeypatch, missed_faces=3)  # P1 fails at every rate
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert md.split("## 细分")[1].splitlines()[0].startswith("（最小的 5 fps，没有频率通过 P1–P3")


def test_the_viewing_metrics_of_every_run_go_into_the_summary(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    viewing = summary["runs"][0]["viewing"]
    # MASK and PLATE_MASK on every frame of the 160 x 120 picture: 99 + 144 of the 1200 raster cells
    assert viewing["area"] == {
        "mean": pytest.approx(0.2025), "p95": pytest.approx(0.2025), "p99": pytest.approx(0.2025), "max": pytest.approx(0.2025),
    }
    assert viewing["flicker"] == {"pairs": N - 1, "over": 0, "share": 0.0}
    assert viewing["boxes"] == {"median": 2.0, "p95": 2.0}
    assert viewing["multi_box_frames"] is None  # the fake masks carry no track ids
    assert summary["pooled"][0]["viewing"]["frames"] == N
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    section = md.split("## 观感")[1].split("\n## ")[0]
    assert "| seg | 5 | 20.25% | 20.25% | 20.25% | 20.25% | 0.0%（0/29） | 2 | 2 | — |" in section
    assert "| 合计 | 10 | 20.25% |" in section
    assert "1/4" in section and "1%" in section


def test_the_summary_keeps_the_area_on_screen_apart_from_the_area_of_the_boxes(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    # with a block the mosaic pixelates the box snapped to the block grid: x 32..80, y 16..80 and x 80..160, y 64..112
    masks = (dict(MASK, block=16), dict(PLATE_MASK, block=16))
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([], masks=masks))
    for viewing in (summary["runs"][0]["viewing"], summary["pooled"][0]["viewing"]):
        assert viewing["area"]["mean"] == pytest.approx((192 + 240) / 1200)
        assert viewing["box_area"] == {
            "mean": pytest.approx(0.2025), "p95": pytest.approx(0.2025), "p99": pytest.approx(0.2025), "max": pytest.approx(0.2025),
        }
    section = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("## 观感")[1].split("\n## ")[0]
    assert "| seg | 5 | 36.00% | 36.00% | 36.00% | 36.00% | 0.0%（0/29） | 2 | 2 | — |" in section
    assert "| seg | 5 | 20.25% | 20.25% | 20.25% | 20.25% |" in section.split("框并集", 1)[1]
    assert "对齐" in section and "旧基线" in section


SPEED_HEADER = "| 段 | fps | 准备 | 解码 | 主线程 | 编码 | 检测（关键帧） | 补框 | 打码 | 总耗时 | 吞吐 |"


def _speed(tmp_path) -> str:
    return (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("## 速度")[1].split("\n## ")[0]


def test_speed_table_measures_the_parallel_stages_against_elapsed_time_and_splits_the_main_thread(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    busy = {"main": 1.2, "detect": 0.9, "track": 0.1, "mosaic": 0.2}
    run_all(_cfg(tmp_path), runner=_fake_runner([], busy=busy, stats_extra={"setup_s": 0.3}))
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    speed = _speed(tmp_path)
    assert md.count("## 速度（本机，160×120，源视频 h264）") == 1
    assert SPEED_HEADER in speed
    cells = _row(speed, "| seg | 5 |").split("|")[3:-1]
    # decode / main / encode run in parallel: share of the run's elapsed 1.5 s; detect / track / mosaic: share of the main thread
    assert cells == [
        " 0.3 s ", " 0.1 s（7%） ", " 1.2 s（80%） ", " 0.1 s（7%） ",
        " 0.9 s（75%） ", " 0.1 s（8%） ", " 0.2 s（17%） ", " 1.5 s ", " 20 fps ",
    ]
    slowest = next(line for line in speed.splitlines() if "估计瓶颈" in line)
    assert "主线程" in slowest and "2.4 s" in slowest and "80%" in slowest  # both detection rates together
    note = speed.split(SPEED_HEADER)[0]
    assert "并行" in note and "串行" in note and "关键帧" in note and "准备" in note


def test_speed_table_adds_up_the_main_thread_for_stats_without_it(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    run_all(_cfg(tmp_path), runner=_fake_runner([], busy={"main": None, "detect": 0.6}))
    cells = _row(_speed(tmp_path), "| seg | 5 |").split("|")[3:-1]
    assert cells[:4] == [" — ", " 0.1 s（7%） ", " 0.8 s（53%） ", " 0.1 s（7%） "]  # main = 0.6 + 0.1 + 0.1, no setup_s
    assert cells[4] == " 0.6 s（75%） "


def test_speed_bottleneck_pools_busy_over_elapsed_across_old_and_new_stats(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    # 5 fps: new stats, main 1.2 of 1.5 s; 10 fps: old stats without main (0.3 + 0.1 + 0.1), encode 2.4 of 3.0 s
    per_rate = {
        5: ({"main": 1.2}, {"setup_s": 0.3}),
        10: ({"main": None, "detect": 0.3, "encode": 2.4}, {"elapsed_s": 3.0}),
    }
    fake = {fps: _fake_runner([], busy=b, stats_extra=x) for fps, (b, x) in per_rate.items()}

    def runner(cmd, check, capture_output, text):
        fps = int(cmd[cmd.index("--detect-fps") + 1]) if "--detect-fps" in cmd else 5
        fake[fps](cmd, check, capture_output, text)

    run_all(_cfg(tmp_path), runner=runner)
    speed = _speed(tmp_path)
    assert _row(speed, "| seg | 10 |").split("|")[3:7] == [" — ", " 0.1 s（3%） ", " 0.5 s（17%） ", " 2.4 s（80%） "]
    slowest = next(line for line in speed.splitlines() if "估计瓶颈" in line)
    # pooled: encode 2.5 s of 4.5 s (56%) beats main 1.7 s (38%); the mean of the per-run shares would pick main (48% vs 43%)
    assert "编码" in slowest and "2.5 s" in slowest and "56%" in slowest


def test_speed_table_dashes_the_main_thread_of_old_stats_missing_a_stage(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    run_all(_cfg(tmp_path), runner=_fake_runner([], busy={"main": None, "track": None}, stats_extra={"elapsed_s": None}))
    speed = _speed(tmp_path)
    cells = _row(speed, "| seg | 5 |").split("|")[3:-1]
    assert cells[2] == " — " and cells[5] == " — " and cells[-2:] == [" — ", " — "]
    assert "估计瓶颈" not in speed  # no run it can measure


def test_speed_table_survives_a_run_without_stage_timings(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    fake = _fake_runner([])

    def no_stages(cmd, check, capture_output, text):
        fake(cmd, check, capture_output, text)
        if cmd[0].endswith("vc_redact"):
            stats = Path(cmd[cmd.index("--out") + 1]) / STEM / "stats.json"
            stats.write_text(json.dumps({**json.loads(stats.read_text()), "stages": {}}))

    run_all(_cfg(tmp_path), runner=no_stages)
    speed = _speed(tmp_path)
    assert _row(speed, "| seg | 5 |").split("|")[3:-1] == [" — "] * 7 + [" 1.5 s ", " 20 fps "]
    assert "估计瓶颈" not in speed


def test_consistency_runs_before_the_sweeps(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []
    run_all(_cfg(tmp_path), runner=_fake_runner(calls))
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["vc_detect", "vc_redact", "vc_redact", "vc_redact"]


def test_consistency_feeds_both_runtimes_the_same_re_encoded_frames(tmp_path, monkeypatch):
    import vc_eval.run as run_module

    _setup(tmp_path)
    seen = {}
    monkeypatch.setattr(run_module, "python_detections", lambda model, refs: seen.setdefault("py", refs) and {r.frame: [FACE] for r in refs})
    calls = []
    fake = _fake_runner(calls)

    def runner(cmd, check, capture_output, text):
        if cmd[0].endswith("vc_detect"):
            frames_dir = Path(cmd[cmd.index("--frames") + 1])
            seen["cpp"] = sorted(p.resolve() for p in frames_dir.glob("*.jpg"))
        fake(cmd, check, capture_output, text)

    run_all(_cfg(tmp_path), runner=runner)
    py_images = sorted(r.image.resolve() for r in seen["py"])
    assert py_images == seen["cpp"] and py_images
    assert all(p.parent == (tmp_path / "report" / "consistency" / "same_pixels").resolve() for p in py_images)


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
    # a later run names its own detection rate; the first, failed run left files in the report dir
    with pytest.raises(ValueError, match="boom") as info:
        run_all(_cfg(tmp_path, overwrite=True), runner=_redact_fails([], "boom", after=2))
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


def test_a_dense_run_that_did_not_detect_every_frame_cannot_be_the_temporal_reference(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    fake = _fake_runner([])

    def sparse_in_disguise(cmd, check, capture_output, text):
        fake(cmd, check, capture_output, text)
        if "--detect-every-frame" in cmd:
            stats = Path(cmd[cmd.index("--out") + 1]) / STEM / "stats.json"
            stats.write_text(json.dumps({**json.loads(stats.read_text()), "detect_every_frame": False}))

    with pytest.raises(ValueError, match="detect_every_frame"):
        run_all(_cfg(tmp_path), runner=sparse_in_disguise)


def test_consistency_line_separates_the_gated_difference_from_the_one_over_all_pairs(tmp_path, monkeypatch):
    _setup(tmp_path)
    import vc_eval.run as run_module

    low = Box("plate", 50, 50, 70, 60, score=0.2, src="student")
    moved = Box("plate", 53, 50, 73, 60, score=0.2, src="student")
    monkeypatch.setattr(run_module, "python_detections", lambda model, refs: {r.frame: [FACE, moved] for r in refs})
    fake = _fake_runner([])

    def detect_with_a_low_plate(cmd, check, capture_output, text):
        fake(cmd, check, capture_output, text)
        if cmd[0].endswith("vc_detect"):
            out = Path(cmd[cmd.index("--out") + 1])
            rows = _rows(f"{cmd[cmd.index('--frames') + 1]}/frames.csv")
            write_jsonl(out, [Detection(r["video"], int(r["frame"]), float(r["pts_ms"]), b) for r in rows for b in (FACE, low)])

    summary = run_all(_cfg(tmp_path), runner=detect_with_a_low_plate)
    c = summary["consistency"]
    assert c["max_diff_px"] == 0.0 and c["max_diff_all_px"] == pytest.approx(3.0) and c["matched_confident"] == 3 and c["passed"] is True
    line = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8").split("## C++ / Python 一致性")[1]
    assert "分数 ≥ 0.3 的配对最大坐标差 0.000 px" in line and "所有配对最大坐标差 3.000 px" in line


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
    "where",
    [
        ("out", "frames", "unlisted", "report"),  # the frames dir of a stem that --stems does not list
        ("out", "frames", "report"),  # straight under the frames root, in no stem's directory
        ("redacted_root", "frames", "unlisted", "redacted"),
        ("redacted_root", "frames", "redacted"),
    ],
)
def test_report_and_redacted_dirs_anywhere_under_the_frames_root_are_refused(tmp_path, where):
    _setup(tmp_path)
    (tmp_path / "frames" / "unlisted").mkdir()  # vc_extract --overwrite would delete this whole directory
    field, *parts = where
    calls = []
    with pytest.raises(ValueError, match="inside the frames directory"):
        run_all(_cfg(tmp_path, **{field: tmp_path.joinpath(*parts)}), runner=_fake_runner(calls))
    assert calls == [] and not (tmp_path / "frames" / "report").exists() and not (tmp_path / "frames" / "unlisted" / "report").exists()


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"stems": []}, "no stems"),
        ({"fps_values": []}, "no detection rates"),
        ({"fps_values": [5, 0]}, "from 5 to 60"),
        ({"fps_values": [5, -1]}, "from 5 to 60"),
        ({"fps_values": [5, 4.9]}, "from 5 to 60"),
        ({"fps_values": [5, float("nan")]}, "from 5 to 60"),
        ({"fps_values": [5, float("inf")]}, "from 5 to 60"),
        ({"fps_values": [5, 61]}, "from 5 to 60"),
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


def test_a_non_empty_report_dir_is_refused_unless_overwriting(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    report = tmp_path / "report"
    report.mkdir()
    (report / "old.txt").write_text("a previous report")
    calls = []
    with pytest.raises(ValueError, match="not empty") as info:
        run_all(_cfg(tmp_path), runner=_fake_runner(calls))
    assert "--overwrite" in str(info.value) and calls == [] and (report / "old.txt").read_text() == "a previous report"

    run_all(_cfg(tmp_path, overwrite=True), runner=_fake_runner(calls))  # cleared first, then the run writes its own files
    assert not (report / "old.txt").exists() and (report / "summary.json").is_file()


def test_an_empty_or_missing_report_dir_needs_no_overwrite(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    (tmp_path / "report").mkdir()
    run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert (tmp_path / "report" / "index.html").is_file()


def test_a_report_path_that_is_a_file_is_refused(tmp_path):
    _setup(tmp_path)
    (tmp_path / "report").write_text("not a directory")
    calls = []
    with pytest.raises(ValueError, match="not a directory"):
        run_all(_cfg(tmp_path, overwrite=True), runner=_fake_runner(calls))
    assert calls == [] and (tmp_path / "report").read_text() == "not a directory"


@pytest.mark.parametrize("inputs", ["frames", "labels", "videos", "model", "redacted"])
def test_overwrite_never_clears_a_directory_holding_the_inputs(tmp_path, inputs):
    _setup(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "redacted" / "dense").mkdir(parents=True)
    keep = {"frames": tmp_path / "frames" / STEM / "frames.csv", "labels": tmp_path / "labels" / STEM / "merged.jsonl",
            "videos": tmp_path / "videos" / f"{STEM}.mp4", "model": tmp_path / "model", "redacted": tmp_path / "redacted"}[inputs]
    keep.parent.mkdir(exist_ok=True)
    out = {"frames": tmp_path, "labels": tmp_path / "labels", "videos": tmp_path / "videos", "model": tmp_path / "model", "redacted": tmp_path / "redacted" / "dense"}[inputs]
    calls = []
    with pytest.raises(ValueError, match="overlaps"):
        run_all(_cfg(tmp_path, out=out, overwrite=True), runner=_fake_runner(calls))
    assert calls == [] and keep.exists()


def test_a_bad_input_leaves_the_old_report_in_place_even_when_overwriting(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    _break_labels(tmp_path / "labels" / STEM, "missing flags")
    report = tmp_path / "report"
    report.mkdir()
    (report / "old.txt").write_text("a previous report")
    with pytest.raises(ValueError, match="flags.csv"):
        run_all(_cfg(tmp_path, overwrite=True), runner=_fake_runner([]))
    assert (report / "old.txt").exists()


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


def test_cli_coverage_reports_the_low_confidence_tier_without_judging_it(tmp_path, capsys):
    _setup(tmp_path, teachers=(*TEACHERS, UNCERTAIN), flagged=(UNCERTAIN,))
    masks = tmp_path / "masks.jsonl"
    with masks.open("w") as f:
        for fr in range(N):
            f.write(json.dumps({"video": f"{STEM}.mp4", "frame": fr, "boxes": [MASK, PLATE_MASK]}) + "\n")
    args = ["coverage", "--frames", str(tmp_path / "frames" / STEM), "--labels", str(tmp_path / "labels" / STEM), "--masks", str(masks)]
    assert main([*args, "--json", str(tmp_path / "c.json")]) == 0  # the uncovered flagged face is not a failure
    out = capsys.readouterr().out
    assert "PASS" in out and "face 5/5" in out.split("low-confidence")[0]
    assert "face 0/5" in out.split("low-confidence")[1]
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["face"]["total"] == 5 and data["uncertain"]["face"]["total"] == 5 and data["uncertain"]["face"]["hits"] == 0


def test_cli_temporal_and_integrity(tmp_path, capsys):
    video = _setup(tmp_path)
    dense = tmp_path / "dense" / "detections.jsonl"
    dense.parent.mkdir()
    write_jsonl(dense, [Detection(f"{STEM}.mp4", fr, fr * 33.3, FACE) for fr in range(N)])
    (dense.parent / "stats.json").write_text(json.dumps({"frames": N, "detect_every_frame": True}))
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
    assert cfg.overwrite is False
    assert cfg.vc_redact == Path("build/cpp-release/vc_redact") and cfg.config == Path("configs/redact.json")
    assert cfg.redacted_root == Path("data/redacted")
    printed = capsys.readouterr().out
    assert "10 fps" in printed and str(out / "index.html") in printed

    assert main([*argv, "--overwrite"]) == 0 and seen[-1].overwrite is True
    # a second config gets its own videos, or its run would overwrite the ones the first report links to
    assert main([*argv, "--redacted-root", str(tmp_path / "tuned")]) == 0 and seen[-1].redacted_root == tmp_path / "tuned"
    capsys.readouterr()

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


FACE_MODEL_JSON = {
    "name": "yunet-2023mar", "classes": ["face"], "output_format": "yunet",
    "input": {"color": "BGR", "scale": 1.0, "letterbox": "top_left", "pad_value": 0, "pad_multiple": 32},
}


def _face_model(directory, meta=None, onnx=b"onnx"):
    """A face model directory as vc_train yunet writes it; only vc_redact reads the files, run_all checks them."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(onnx)
    (directory / "model.json").write_text(json.dumps(FACE_MODEL_JSON if meta is None else meta))
    return directory


def test_a_face_model_goes_to_every_vc_redact_run_and_is_named_in_the_summary(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    face = _face_model(tmp_path / "yunet")
    calls = []
    summary = run_all(_cfg(tmp_path, face_model=face), runner=_fake_runner(calls))

    redact = [c for c in calls if c[0].endswith("vc_redact")]
    assert len(redact) == 3  # dense + 2 fps
    for command in redact:
        assert command[command.index("--face-model") + 1] == str(face)
    # the C++ / Python consistency check compares the student alone
    assert all("--face-model" not in c for c in calls if c[0].endswith("vc_detect"))
    assert summary["face_model"] == str(face)
    assert json.loads((tmp_path / "report" / "summary.json").read_text())["face_model"] == str(face)
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert f"人脸模型：{face}" in md


def test_without_a_face_model_vc_redact_runs_as_before_and_the_summary_says_so(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    calls = []
    summary = run_all(_cfg(tmp_path), runner=_fake_runner(calls))
    assert all("--face-model" not in c for c in calls)
    assert summary["face_model"] is None
    assert "人脸模型：无" in (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")


def test_a_missing_face_model_is_refused_before_any_run(tmp_path):
    _setup(tmp_path)
    calls = []
    with pytest.raises(ValueError, match=r"yunet has no model\.onnx / model\.json; write the face model with python -m vc_train yunet"):
        run_all(_cfg(tmp_path, face_model=tmp_path / "yunet"), runner=_fake_runner(calls))
    assert calls == []


@pytest.mark.parametrize("name", ["model.onnx", "model.json"])
def test_a_face_model_with_a_file_missing_is_refused_before_any_run(tmp_path, name):
    _setup(tmp_path)
    face = _face_model(tmp_path / "yunet")
    (face / name).unlink()
    calls = []
    with pytest.raises(ValueError, match=rf"yunet has no {name.replace('.', r'\.')};"):
        run_all(_cfg(tmp_path, face_model=face), runner=_fake_runner(calls))
    assert calls == []


@pytest.mark.parametrize(
    "case, meta, onnx, message",
    [
        ("empty onnx", None, b"", r"model\.onnx is empty"),
        ("json array", [], b"onnx", r"model\.json must hold a JSON object"),
        ("json scalar", "yunet", b"onnx", r"model\.json must hold a JSON object"),
        ("student format", {**FACE_MODEL_JSON, "output_format": "yolov8"}, b"onnx", r"output_format must be \"yunet\""),
        ("no format", {"classes": ["face"]}, b"onnx", r"output_format must be \"yunet\""),
        ("plate class", {**FACE_MODEL_JSON, "classes": ["plate"]}, b"onnx", r"classes must be \[\"face\"\]"),
        ("two classes", {**FACE_MODEL_JSON, "classes": ["face", "plate"]}, b"onnx", r"classes must be \[\"face\"\]"),
        ("no classes", {"output_format": "yunet"}, b"onnx", r"classes must be \[\"face\"\]"),
    ],
)
def test_an_unusable_face_model_is_refused_before_any_run(tmp_path, case, meta, onnx, message):
    _setup(tmp_path)
    face = _face_model(tmp_path / "yunet", meta, onnx)
    calls = []
    with pytest.raises(ValueError, match=message):
        run_all(_cfg(tmp_path, face_model=face), runner=_fake_runner(calls))
    assert calls == []


@pytest.mark.parametrize("content", [b"{", b"", b"\xff\xfe"])
def test_a_face_model_json_that_is_not_json_is_refused_before_any_run(tmp_path, content):
    _setup(tmp_path)
    face = _face_model(tmp_path / "yunet")
    (face / "model.json").write_bytes(content)
    calls = []
    with pytest.raises(ValueError, match=r"model\.json is not valid JSON"):
        run_all(_cfg(tmp_path, face_model=face), runner=_fake_runner(calls))
    assert calls == []


@pytest.mark.parametrize("face", ["missing", "empty onnx", "student format"])
def test_a_bad_face_model_leaves_the_previous_report_alone_under_overwrite(tmp_path, face):
    _setup(tmp_path)
    old = tmp_path / "report"
    old.mkdir()
    (old / "summary.json").write_text("previous report")
    (old / "sub").mkdir()
    (old / "sub" / "page.html").write_text("previous page")
    directory = tmp_path / "yunet"  # "missing" leaves it absent
    if face == "empty onnx":
        _face_model(directory, onnx=b"")
    elif face == "student format":
        _face_model(directory, {**FACE_MODEL_JSON, "output_format": "yolov8"})
    calls = []
    with pytest.raises(ValueError, match="yunet"):
        run_all(_cfg(tmp_path, face_model=directory, overwrite=True), runner=_fake_runner(calls))
    assert calls == []
    assert (old / "summary.json").read_text() == "previous report" and (old / "sub" / "page.html").read_text() == "previous page"


def test_overwrite_never_clears_the_face_model(tmp_path):
    _setup(tmp_path)
    face = tmp_path / "report" / "yunet"
    face.mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match="overlaps"):
        run_all(_cfg(tmp_path, face_model=face, overwrite=True), runner=_fake_runner(calls))
    assert calls == [] and face.is_dir()


def test_cli_run_passes_the_face_model(tmp_path, monkeypatch):
    import vc_eval.run as run_module

    seen = []
    monkeypatch.setattr(run_module, "run_all", lambda cfg: seen.append(cfg) or {"chosen_fps": 10.0})
    argv = ["run", "--stems", "a", "--model", str(tmp_path / "m"), "--out", str(tmp_path / "rep")]
    assert main(argv) == 0 and seen[-1].face_model is None
    assert main([*argv, "--face-model", str(tmp_path / "yunet")]) == 0
    assert seen[-1].face_model == tmp_path / "yunet"
