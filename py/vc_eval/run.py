"""Detection-rate sweep over held-out segments, summary tables and the spot-check page (spec §6.3–§6.5)."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from vc_label.contracts import LABEL_CLASSES, read_frames_csv
from vc_label.review import ensure_outside
from vc_eval.consistency import compare_detections, python_detections, run_vc_detect
from vc_eval.coverage import CoverageReport, evaluate_coverage, load_label_inputs, pool
from vc_eval.integrity import check_integrity, probe, probe_cache
from vc_eval.metrics import Rate
from vc_eval.report import summary_md
from vc_eval.spotcheck import Item, Section, extract_frames, miss_tile, thumbnail, write_spotcheck
from vc_eval.temporal import TemporalReport, evaluate_temporal

MAX_MISS_TILES = 60
MAX_UNCERTAIN_TILES = 60
MAX_TEMPORAL_TILES = 30
STDERR_TAIL_LINES = 20
MIN_DETECT_FPS = 5.0


@dataclass
class RunConfig:
    stems: list[str]
    model: Path
    fps_values: list[float]
    out: Path
    vc_redact: Path
    vc_detect: Path
    config: Path
    videos: Path = Path("data/videos")
    frames_root: Path = Path("data/frames")
    labels_root: Path = Path("data/labels")
    redacted_root: Path = Path("data/redacted")
    consistency_count: int = 20
    spot_frames: int = 30
    seed: int = 0
    overwrite: bool = False  # clear a non-empty `out` first instead of refusing it


def _same_or_below(path: Path, parent: Path) -> bool:
    """True when `path` is `parent` or lies anywhere under it (spelling-proof: APFS is case-insensitive)."""
    target, root = path.resolve(), parent.resolve()
    if target == root or root in target.parents:
        return True
    if parent.exists():
        return any(c.exists() and os.path.samefile(c, parent) for c in (target, *target.parents))
    return False


def _validate_out(cfg: RunConfig) -> None:
    """`out` is where the report goes and, with --overwrite, what gets cleared: it must not touch the data the run reads."""
    for base in (cfg.out, cfg.redacted_root):
        ensure_outside(cfg.frames_root, base)  # anywhere under the frames root, not only below the listed stems
        for stem in cfg.stems:
            ensure_outside(cfg.frames_root / stem, base)  # a stem directory may be a symlink to elsewhere
    reads = [cfg.frames_root, cfg.labels_root, cfg.videos, cfg.model, cfg.vc_redact, cfg.vc_detect, cfg.config, cfg.redacted_root]
    for path in reads:
        if _same_or_below(path, cfg.out):
            raise ValueError(f"{cfg.out} overlaps {path}; --out must be a directory of its own (--overwrite clears it)")
    for run_dir in (cfg.redacted_root / "dense", *(_fps_dir(cfg, fps) for fps in cfg.fps_values)):
        if _same_or_below(cfg.out, run_dir):
            raise ValueError(f"{cfg.out} overlaps {run_dir}, where vc_redact writes; --out must be a directory of its own")
    if cfg.out.exists():
        if not cfg.out.is_dir():
            raise ValueError(f"{cfg.out} is not a directory")
        if not cfg.overwrite and any(cfg.out.iterdir()):
            raise ValueError(f"{cfg.out} is not empty; pass --overwrite to clear it first, or use another --out")


def _clear(directory: Path) -> None:
    """Empties `directory` (links are removed, never followed) and keeps the directory itself."""
    for child in directory.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def _validate(cfg: RunConfig) -> None:
    """Refuses a bad configuration before the first vc_redact run, which takes minutes."""
    if not cfg.stems:
        raise ValueError("no stems to evaluate")
    if len(set(cfg.stems)) != len(cfg.stems):
        raise ValueError("a stem is listed twice")
    if not cfg.fps_values:
        raise ValueError("no detection rates to sweep")
    if not all(math.isfinite(fps) and fps >= MIN_DETECT_FPS for fps in cfg.fps_values):
        raise ValueError(f"detection rates must be finite numbers of at least {MIN_DETECT_FPS:g} fps")
    if len({float(fps) for fps in cfg.fps_values}) != len(cfg.fps_values):
        raise ValueError("a detection rate is listed twice")
    if cfg.consistency_count < 1 or cfg.spot_frames < 0:
        raise ValueError("consistency_count must be at least 1 and spot_frames at least 0")
    _validate_out(cfg)
    for stem in cfg.stems:
        for needed in (cfg.videos / f"{stem}.mp4", cfg.frames_root / stem / "frames.csv"):
            if not needed.is_file():
                raise ValueError(f"{needed} does not exist")


def _check_inputs(cfg: RunConfig) -> None:
    """The reference-side checks evaluate_coverage makes, for every stem, before the first vc_redact run."""
    sizes = {}
    for stem in cfg.stems:
        labels = load_label_inputs(cfg.frames_root / stem, cfg.labels_root / stem)
        sizes[stem] = (labels.width, labels.height)
    if len(set(sizes.values())) > 1:
        shown = ", ".join(f"{stem} {w}x{h}" for stem, (w, h) in sizes.items())
        raise ValueError(f"the segments have different frame sizes ({shown}); their coverage cannot be pooled")


def _fps_dir(cfg: RunConfig, fps: float) -> Path:
    return cfg.redacted_root / f"fps{fps:g}"


def _redact(cfg: RunConfig, stem: str, out: Path, extra: list[str], runner) -> None:
    command = [
        str(cfg.vc_redact), "--input", str(cfg.videos / f"{stem}.mp4"), "--out", str(out), "--model", str(cfg.model),
        "--config", str(cfg.config), *extra, "--overwrite",
    ]
    try:
        runner(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-STDERR_TAIL_LINES:])
        raise ValueError(f"vc_redact failed on {stem} ({' '.join(extra)}, exit {exc.returncode}): {tail}") from exc


def rate_json(rate: Rate) -> dict:
    lo, hi = rate.interval
    return {"hits": rate.hits, "total": rate.total, "value": None if rate.total == 0 else rate.value, "ci": [lo, hi]}


def coverage_json(c: CoverageReport) -> dict:
    out = {cls: rate_json(c.classes[cls].rate) for cls in LABEL_CLASSES}
    weakest = c.weakest_cell()
    out["weakest_cell"] = None if weakest is None else {"cls": weakest[0], "row": weakest[1], "col": weakest[2], **rate_json(weakest[3])}
    # with no heatmap cell holding enough targets P2 is vacuously satisfied ("n/a"); only the user's spot check guards that
    out["p2"] = "n/a" if weakest is None else ("pass" if c.heatmap_passed() else "fail")
    out["temporal_fill"] = {cls: rate_json(c.classes[cls].temporal_fill) for cls in LABEL_CLASSES}
    out["best_effort"] = {cls: rate_json(c.classes[cls].best_effort) for cls in LABEL_CLASSES}
    # the rates above are the confident tier; the flagged (low-confidence) reference boxes are reported here and never gate
    out["uncertain"] = {cls: rate_json(c.classes[cls].uncertain) for cls in LABEL_CLASSES}
    out["bins"] = {cls: {k: rate_json(v) for k, v in c.classes[cls].bins.items()} for cls in LABEL_CLASSES}
    # the whole grid (rows top to bottom, columns left to right), not just the weakest cell: the cells under 30 targets are the ones P2 skips
    out["heatmap"] = {cls: [[rate_json(rate) for rate in row] for row in c.classes[cls].heatmap] for cls in LABEL_CLASSES}
    out["passed"] = c.passed()
    return out


def _evaluate(cfg: RunConfig, stem: str, fps: float) -> tuple[dict, CoverageReport, TemporalReport]:
    run_dir = _fps_dir(cfg, fps) / stem
    coverage = evaluate_coverage(cfg.frames_root / stem, cfg.labels_root / stem, run_dir / "masks.jsonl")
    temporal = evaluate_temporal(cfg.redacted_root / "dense" / stem / "detections.jsonl", run_dir / "masks.jsonl", coverage.height)
    integrity = check_integrity(cfg.videos / f"{stem}.mp4", run_dir / f"{stem}.redacted.mp4")
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    entry = {
        "stem": stem,
        "fps": fps,
        "coverage": coverage_json(coverage),
        "temporal": {cls: rate_json(temporal.classes[cls]) for cls in LABEL_CLASSES},
        "integrity": {
            "checks": integrity.checks, "details": integrity.details, "passed": integrity.passed,
            "redacted_frames": integrity.redacted_frames,
        },
        "stray": {"max": coverage.stray_max, "mean": coverage.stray_mean, "p95": coverage.stray_p95},
        "misses": {"confident": len(coverage.misses), "uncertain": len(coverage.uncertain_misses), "temporal": len(temporal.misses)},
        "stages": stats.get("stages", {}),
        "elapsed_s": stats.get("elapsed_s"),
        "setup_s": stats.get("setup_s"),
        "frames": stats.get("frames"),
        "detector": stats.get("detector", ""),
        "encoder": stats.get("encoder", ""),
    }
    return entry, coverage, temporal


def _spread(items: list, limit: int) -> list:
    """At most `limit` items, evenly spaced over the list (a run of misses is mostly one target in neighbouring frames)."""
    if len(items) <= limit:
        return list(items)
    return [items[i * len(items) // limit] for i in range(limit)]


def _consistency(cfg: RunConfig, runner) -> dict:
    refs = _spread(read_frames_csv(cfg.frames_root / cfg.stems[0]), cfg.consistency_count)
    cpp = run_vc_detect(cfg.vc_detect, cfg.model, refs, cfg.out / "consistency", runner=runner)
    report = compare_detections(cpp, python_detections(cfg.model, refs))
    return {**asdict(report), "passed": report.passed()}


def _pooled_entry(stems: list[str], fps: float, runs: list[dict], reports: dict[tuple[str, float], tuple[CoverageReport, TemporalReport]]) -> dict:
    """P1 / P2 on the coverage of all segments together (one final recall per class); P3 must hold for every segment."""
    pooled = pool([reports[(stem, fps)][0] for stem in stems])
    coverage = coverage_json(pooled)
    integrity_passed = all(r["integrity"]["passed"] for r in runs if r["fps"] == fps)
    return {
        "fps": fps,
        "stems": list(stems),
        "coverage": coverage,
        "stray": {"max": pooled.stray_max, "mean": pooled.stray_mean, "p95": pooled.stray_p95},
        "misses": {kind: sum(r["misses"][kind] for r in runs if r["fps"] == fps) for kind in ("confident", "uncertain", "temporal")},
        "integrity_passed": integrity_passed,
        "passed": coverage["passed"] and integrity_passed,
    }


def _choose(pooled: list[dict]) -> float | None:
    """The smallest detection rate whose pooled coverage passes while every segment's integrity does."""
    return min((entry["fps"] for entry in pooled if entry["passed"]), default=None)


def _href(path: str | Path) -> str:
    return quote(Path(path).as_posix())


def _reference_tiers(coverage: dict) -> dict:
    """Per class, how many must-mask reference targets are confident (gate P1 / P2) and how many uncertain (report only).

    The split is a property of the reference, so any detection rate's pooled coverage gives the same counts.
    """
    tiers = {}
    for cls in LABEL_CLASSES:
        confident, uncertain = coverage[cls]["total"], coverage["uncertain"][cls]["total"]
        share = None if confident + uncertain == 0 else uncertain / (confident + uncertain)
        tiers[cls] = {"confident": confident, "uncertain": uncertain, "uncertain_share": share}
    return tiers


def _spot_counts(reports: dict[str, tuple[CoverageReport, TemporalReport]]) -> dict[str, dict]:
    """Per stem and kind of miss: how many there are and how many of them the spot-check page shows."""
    counted = lambda items, limit: {"total": len(items), "shown": len(_spread(items, limit))}  # noqa: E731
    return {
        stem: {
            "confident": counted(coverage.misses, MAX_MISS_TILES),
            "uncertain": counted(coverage.uncertain_misses, MAX_UNCERTAIN_TILES),
            "temporal": counted(tempo.misses, MAX_TEMPORAL_TILES),
        }
        for stem, (coverage, tempo) in reports.items()
    }


def _counted(note: str, counts: dict[str, dict], kind: str) -> str:
    """The section note followed by, per stem, how many misses there are and how many are shown (an even spread)."""
    per_stem = "；".join(f"{stem}：共 {c[kind]['total']} 个，均匀抽取展示 {c[kind]['shown']} 个" for stem, c in counts.items())
    return f"{note} {per_stem}。"


def _spot_sections(
    cfg: RunConfig,
    fps: float,
    reports: dict[str, tuple[CoverageReport, TemporalReport]],
    frame_counts: dict[str, int | None],
    counts: dict[str, dict],
) -> list[Section]:
    """`frame_counts` holds the decoded frame count of each stem's redacted video, which integrity already counted;
    `counts` is _spot_counts of the same reports."""
    img = cfg.out / "img"
    rng = random.Random(cfg.seed)
    misses, uncertain, temporal, random_items, links = [], [], [], [], []
    for stem, (coverage, tempo) in reports.items():
        video = _fps_dir(cfg, fps) / stem / f"{stem}.redacted.mp4"
        links.append((f"{stem} 打码视频（{fps:g} fps 检测）", _href(os.path.relpath(video, cfg.out))))
        chosen_misses = _spread(coverage.misses, MAX_MISS_TILES)
        chosen_uncertain = _spread(coverage.uncertain_misses, MAX_UNCERTAIN_TILES)
        chosen_temporal = _spread(tempo.misses, MAX_TEMPORAL_TILES)
        frames = frame_counts[stem] or 0
        pick = sorted(rng.sample(range(frames), min(cfg.spot_frames, frames)))
        wanted = [m.frame for m in chosen_misses] + [m.frame for m in chosen_uncertain] + [m.frame for m in chosen_temporal] + pick
        scratch = img / stem / "redacted"  # full-resolution frames: needed only to cut the tiles, then deleted
        try:
            decoded = extract_frames(video, wanted, scratch)
            for n, m in enumerate(chosen_misses):
                original = cfg.frames_root / stem / f"{m.frame:06d}.jpg"
                box = m.box.clipped(coverage.width, coverage.height)  # miss_tile needs a box at least partly inside the image
                tile = miss_tile(original, decoded[m.frame], box, img / stem / f"miss_{m.frame:06d}_{n}.jpg")
                misses.append(Item(_href(tile.relative_to(cfg.out)), f"{stem} 帧 {m.frame} {m.box.cls} 覆盖 {m.covered:.0%}（左原帧，右打码）"))
            for n, m in enumerate(chosen_uncertain):
                original = cfg.frames_root / stem / f"{m.frame:06d}.jpg"
                box = m.box.clipped(coverage.width, coverage.height)
                tile = miss_tile(original, decoded[m.frame], box, img / stem / f"uncertain_{m.frame:06d}_{n}.jpg")
                caption = f"{stem} 帧 {m.frame} {m.box.cls} 覆盖 {m.covered:.0%}，老师分数 {m.box.score:.2f}（左原帧，右打码）"
                uncertain.append(Item(_href(tile.relative_to(cfg.out)), caption))
            for n, m in enumerate(chosen_temporal):
                box = m.box.clipped(coverage.width, coverage.height)
                tile = miss_tile(decoded[m.frame], None, box, img / stem / f"temporal_{m.frame:06d}_{n}.jpg")
                temporal.append(Item(_href(tile.relative_to(cfg.out)), f"{stem} 帧 {m.frame} {m.box.cls} 覆盖 {m.covered:.0%}"))
            for frame in pick:
                thumb = thumbnail(decoded[frame], img / stem / f"random_{frame:06d}.jpg")
                random_items.append(Item(_href(thumb.relative_to(cfg.out)), f"{stem} 帧 {frame}"))
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)
    return [
        Section("打码视频", "请完整看一遍，留意是否有露出的人脸或车牌。", [], links),
        Section(
            "未遮住的参照目标",
            _counted("老师标注里必须遮住、但打码框覆盖不足 80% 的高置信目标（P1 / P2 按这一层判定）。", counts, "confident"),
            misses,
        ),
        Section(
            "低置信参照里没遮住的目标",
            _counted(
                "这些是老师自己拿不准的参照框（分数偏低，或只有一个人脸老师检出），打码框覆盖不足 80%。其中不少是老师误检"
                "（路障、标志、贴纸、装饰物），也有一部分是真的模糊车牌，请逐个用眼睛判断该不该遮；它们不计入 P1 / P2。",
                counts,
                "uncertain",
            ),
            uncertain,
        ),
        Section(
            "逐帧检测有、稀疏打码漏掉",
            _counted("同一模型逐帧检测到、但关键帧 + 补框没盖住的框（打码后画面）。", counts, "temporal"),
            temporal,
        ),
        Section("随机打码帧", "随机抽取的打码后画面，检查漏打与误打。", random_items),
    ]


def run_all(cfg: RunConfig, runner=subprocess.run) -> dict:
    _validate(cfg)
    _check_inputs(cfg)
    if cfg.out.exists() and cfg.overwrite:
        _clear(cfg.out)  # only after every input has been checked: a bad input must not cost the previous report
    cfg.out.mkdir(parents=True, exist_ok=True)
    consistency = _consistency(cfg, runner)  # a missing vc_detect or a bad model fails here, not after minutes of sweeps
    with probe_cache():  # `ffprobe -count_frames` decodes a whole file: once per file is enough
        return _sweep_and_report(cfg, runner, consistency)


def _sweep_and_report(cfg: RunConfig, runner, consistency: dict) -> dict:
    runs: list[dict] = []
    segments: list[dict] = []
    reports: dict[tuple[str, float], tuple[CoverageReport, TemporalReport]] = {}
    for stem in cfg.stems:
        _redact(cfg, stem, cfg.redacted_root / "dense", ["--detect-every-frame"], runner)
        for fps in cfg.fps_values:
            _redact(cfg, stem, _fps_dir(cfg, fps), ["--detect-fps", f"{fps:g}"], runner)
        for fps in cfg.fps_values:
            entry, coverage, temporal = _evaluate(cfg, stem, fps)
            runs.append(entry)
            reports[(stem, fps)] = (coverage, temporal)
        coverage = reports[(stem, cfg.fps_values[0])][0]
        codec = probe(cfg.videos / f"{stem}.mp4").video.codec
        segments.append({"stem": stem, "width": coverage.width, "height": coverage.height, "codec": codec})
    pooled = [_pooled_entry(cfg.stems, fps, runs, reports) for fps in sorted(cfg.fps_values)]
    chosen = _choose(pooled)
    spot_fps = chosen if chosen is not None else min(cfg.fps_values)
    spot_reports = {stem: reports[(stem, spot_fps)] for stem in cfg.stems}
    counts = _spot_counts(spot_reports)
    summary = {
        "model": str(cfg.model),
        "chosen_fps": chosen,
        "segments": segments,
        "reference": _reference_tiers(pooled[0]["coverage"]),
        "runs": runs,
        "pooled": pooled,
        "spot_check": {"fps": spot_fps, "stems": counts},
        "consistency": consistency,
    }
    # the numbers first: a failure while building the page must not lose them
    (cfg.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (cfg.out / "summary.md").write_text(summary_md(summary), encoding="utf-8")
    frame_counts = {r["stem"]: r["integrity"]["redacted_frames"] for r in runs if r["fps"] == spot_fps}
    sections = _spot_sections(cfg, spot_fps, spot_reports, frame_counts, counts)
    intro = (
        f"候选检测频率 {spot_fps:g} fps（P1–P3 通过；P4 由你看完本页后确认）。"
        if chosen is not None
        else f"没有检测频率通过 P1–P3，下面是扫描里最小的 {spot_fps:g} fps。"
    )
    write_spotcheck(cfg.out, "打码抽查", f"{intro}参照为老师融合标注（代理），老师漏标的目标只能靠人眼发现。", sections)
    return summary
