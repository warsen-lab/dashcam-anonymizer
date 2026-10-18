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
from vc_eval.integrity import check_integrity, probe
from vc_eval.metrics import HEATMAP_MIN_TARGETS, Rate
from vc_eval.spotcheck import Item, Section, extract_frames, miss_tile, thumbnail, write_spotcheck
from vc_eval.temporal import TemporalReport, evaluate_temporal

MAX_MISS_TILES = 60
MAX_UNCERTAIN_TILES = 60
MAX_TEMPORAL_TILES = 30
STDERR_TAIL_LINES = 20
MIN_DETECT_FPS = 5.0
STAGES = ("decode", "detect", "track", "mosaic", "encode")
DASH = "—"


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
        "integrity": {"checks": integrity.checks, "details": integrity.details, "passed": integrity.passed},
        "stray": {"max": coverage.stray_max, "mean": coverage.stray_mean, "p95": coverage.stray_p95},
        "stages": stats.get("stages", {}),
        "elapsed_s": stats.get("elapsed_s"),
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
        "integrity_passed": integrity_passed,
        "passed": coverage["passed"] and integrity_passed,
    }


def _choose(pooled: list[dict]) -> float | None:
    """The smallest detection rate whose pooled coverage passes while every segment's integrity does."""
    return min((entry["fps"] for entry in pooled if entry["passed"]), default=None)


def _href(path: str | Path) -> str:
    return quote(Path(path).as_posix())


def _spot_sections(cfg: RunConfig, fps: float, reports: dict[str, tuple[CoverageReport, TemporalReport]]) -> list[Section]:
    img = cfg.out / "img"
    rng = random.Random(cfg.seed)
    misses, uncertain, temporal, random_items, links = [], [], [], [], []
    for stem, (coverage, tempo) in reports.items():
        video = _fps_dir(cfg, fps) / stem / f"{stem}.redacted.mp4"
        links.append((f"{stem} 打码视频（{fps:g} fps 检测）", _href(os.path.relpath(video, cfg.out))))
        chosen_misses = _spread(coverage.misses, MAX_MISS_TILES)
        chosen_uncertain = _spread(coverage.uncertain_misses, MAX_UNCERTAIN_TILES)
        chosen_temporal = _spread(tempo.misses, MAX_TEMPORAL_TILES)
        frames = probe(video).video.frames or 0
        pick = sorted(rng.sample(range(frames), min(cfg.spot_frames, frames)))
        wanted = [m.frame for m in chosen_misses] + [m.frame for m in chosen_uncertain] + [m.frame for m in chosen_temporal] + pick
        decoded = extract_frames(video, wanted, img / stem / "redacted")
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
    return [
        Section("打码视频", "请完整看一遍，留意是否有露出的人脸或车牌。", [], links),
        Section("未遮住的参照目标", "老师标注里必须遮住、但打码框覆盖不足 80% 的高置信目标（P1 / P2 按这一层判定）。", misses),
        Section(
            "低置信参照里没遮住的目标",
            "这些是老师自己拿不准的参照框（分数偏低，或只有一个人脸老师检出），打码框覆盖不足 80%。其中不少是老师误检"
            "（路障、标志、贴纸、装饰物），也有一部分是真的模糊车牌，请逐个用眼睛判断该不该遮；它们不计入 P1 / P2。",
            uncertain,
        ),
        Section("逐帧检测有、稀疏打码漏掉", "同一模型逐帧检测到、但关键帧 + 补框没盖住的框（打码后画面）。", temporal),
        Section("随机打码帧", "随机抽取的打码后画面，检查漏打与误打。", random_items),
    ]


def _pct(rate: dict) -> str:
    if not rate["total"]:
        return DASH
    lo, hi = rate["ci"]
    return f"{rate['value']:.1%} ({lo * 100:.1f}–{hi:.1%})，n={rate['total']}"


def _weakest(coverage: dict) -> str:
    w = coverage["weakest_cell"]
    if w is None:
        return f"n/a（无热力格达到 {HEATMAP_MIN_TARGETS} 个目标）"
    return f"{w['cls']} r{w['row']}c{w['col']} {w['value']:.1%}（n={w['total']}）"


def _number(value, fmt: str) -> str:
    return DASH if value is None else format(value, fmt)


def _failed_details(integrity: dict) -> str:
    failed = [f"{name}: {integrity['details'].get(name, 'failed')}" for name, ok in integrity["checks"].items() if not ok]
    return "；".join(failed).replace("|", "/")


def _segment_text(segments: list[dict]) -> str:
    """Frame size and source codec of the segments, once if they agree, per segment otherwise."""
    kinds = {(s["width"], s["height"], s["codec"]) for s in segments}
    text = lambda s: f"{s['width']}×{s['height']}，源视频 {s['codec']}"  # noqa: E731
    if len(kinds) == 1:
        return text(segments[0])
    return "；".join(f"{s['stem']} {text(s)}" for s in segments)


def _summary_md(summary: dict) -> str:
    runs = summary["runs"]
    pooled = summary["pooled"]
    first = runs[0] if runs else {}
    chosen = summary["chosen_fps"]
    mark = lambda ok: "✓" if ok else "✗"  # noqa: E731
    if chosen is None:
        chosen_line = "- 选定检测频率：无满足 P1–P3 的频率"
    else:
        unjudged = any(p["fps"] == chosen and p["coverage"]["p2"] == "n/a" for p in pooled)
        note = f"（没有热力格达到 {HEATMAP_MIN_TARGETS} 个目标，P2 空洞满足；空间上的薄弱点只能靠 P4 抽查把关）"
        chosen_line = f"- 选定检测频率：{chosen:g} fps" + (note if unjudged else "")
    lines = [
        "# 代理验证结果",
        "",
        f"- 模型：{summary['model']}；检测后端：{first.get('detector', '')}；编码器：{first.get('encoder', '')}",
        chosen_line,
        "- 本结果以老师融合标注为参照（代理验收），不是人工金标准验收。",
        "- 参照分两层：P1 / P2 只按高置信参照（没被 `low_score` / `face_single_source` 标记的框）判定；低置信参照单列，只报告，不设门槛。",
        "",
        "## 最终覆盖率（P1 / P2，高置信参照）",
        "",
        "| 段 | fps | 人脸（95% 区间） | 车牌（95% 区间） | 最弱热力格 | 通过 |",
        "|---|---|---|---|---|---|",
    ]
    for fps in sorted({r["fps"] for r in runs}):
        for r in (r for r in runs if r["fps"] == fps):
            c = r["coverage"]
            lines.append(f"| {r['stem']} | {fps:g} | {_pct(c['face'])} | {_pct(c['plate'])} | {_weakest(c)} | — |")
        p = next(p for p in pooled if p["fps"] == fps)
        c = p["coverage"]
        lines.append(f"| 合计 | {fps:g} | {_pct(c['face'])} | {_pct(c['plate'])} | {_weakest(c)} | {'是' if c['passed'] else '否'} |")
    lines += ["", "合计行是各段目标数相加后的覆盖率，P1 / P2 以它为准，“通过”一栏只在合计行；各段行仅供参考。"]
    if any(r["coverage"]["p2"] == "n/a" for r in runs) or any(p["coverage"]["p2"] == "n/a" for p in pooled):
        lines += [
            "",
            f"n/a：没有热力格达到 {HEATMAP_MIN_TARGETS} 个目标，所以 P2 空洞满足（不阻止选定）；空间上的薄弱点只能靠用户的抽查（P4）把关。",
        ]
    lines += [
        "",
        "## 低置信参照（只报告，不设门槛）",
        "",
        "被 `low_score` / `face_single_source` 标记的参照框，只算必须遮住的。其中不少是老师误检（路障、标志、贴纸、装饰物），"
        "也有一部分是真的模糊车牌，所以不进 P1 / P2；没遮住的在抽查页的“低置信参照里没遮住的目标”一节，请用眼睛判断。",
        "",
        "| 段 | fps | 人脸（95% 区间） | 车牌（95% 区间） |",
        "|---|---|---|---|",
    ]
    for fps in sorted({r["fps"] for r in runs}):
        for r in (r for r in runs if r["fps"] == fps):
            u = r["coverage"]["uncertain"]
            lines.append(f"| {r['stem']} | {fps:g} | {_pct(u['face'])} | {_pct(u['plate'])} |")
        u = next(p for p in pooled if p["fps"] == fps)["coverage"]["uncertain"]
        lines.append(f"| 合计 | {fps:g} | {_pct(u['face'])} | {_pct(u['plate'])} |")
    lines += ["", "## 时间覆盖率", "", "| 段 | fps | 人脸 | 车牌 |", "|---|---|---|---|"]
    lines += [f"| {r['stem']} | {r['fps']:g} | {_pct(r['temporal']['face'])} | {_pct(r['temporal']['plate'])} |" for r in runs]
    lines += ["", "## 输出完整性（P3）", "", "| 段 | fps | 可解码 | 帧数 | 分辨率 | 时长 | 音频 | 音画同步 | 未通过项 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        i = r["integrity"]
        k = i["checks"]
        cells = " | ".join(mark(k.get(name)) for name in ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync"))
        lines.append(f"| {r['stem']} | {r['fps']:g} | {cells} | {_failed_details(i)} |")
    lines += ["", "## 误打面积（只报告）", "", "| 段 | fps | 最大 | 均值 | 95 分位 |", "|---|---|---|---|---|"]
    lines += [f"| {r['stem']} | {r['fps']:g} | {r['stray']['max']:.2%} | {r['stray']['mean']:.2%} | {r['stray']['p95']:.2%} |" for r in runs]
    lines += [
        "", f"## 速度（本机，{_segment_text(summary['segments'])}）", "",
        "各阶段为吞吐（帧/秒），总耗时为整段秒数；后端见开头的检测后端与编码器。", "",
        "| 段 | fps | 解码 | 检测 | 补框 | 打码 | 编码 | 总耗时 |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        cells = " | ".join(_number(r["stages"].get(s, {}).get("fps"), ".0f") for s in STAGES)
        lines.append(f"| {r['stem']} | {r['fps']:g} | {cells} | {_number(r['elapsed_s'], '.1f')} s |")
    c = summary["consistency"]
    lines += [
        "",
        "## C++ / Python 一致性",
        "",
        f"{c['images']} 张图、{c['boxes']} 个框，配上 {c['matched']} 个，最大坐标差 {c['max_diff_px']:.3f} px，"
        f"未配上（分数 ≥ 0.3）{c['unmatched_confident']} 个、（< 0.3）{c['unmatched_low']} 个：{'通过' if c['passed'] else '未通过'}。",
        "",
    ]
    return "\n".join(lines)


def run_all(cfg: RunConfig, runner=subprocess.run) -> dict:
    _validate(cfg)
    _check_inputs(cfg)
    if cfg.out.exists() and cfg.overwrite:
        _clear(cfg.out)  # only after every input has been checked: a bad input must not cost the previous report
    cfg.out.mkdir(parents=True, exist_ok=True)
    consistency = _consistency(cfg, runner)  # a missing vc_detect or a bad model fails here, not after minutes of sweeps
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
    summary = {"model": str(cfg.model), "chosen_fps": chosen, "segments": segments, "runs": runs, "pooled": pooled, "consistency": consistency}
    # the numbers first: a failure while building the page must not lose them
    (cfg.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (cfg.out / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    spot_fps = chosen if chosen is not None else min(cfg.fps_values)
    sections = _spot_sections(cfg, spot_fps, {stem: reports[(stem, spot_fps)] for stem in cfg.stems})
    write_spotcheck(
        cfg.out,
        "打码抽查",
        f"检测频率 {spot_fps:g} fps。参照为老师融合标注（代理），老师漏标的目标只能靠人眼发现。",
        sections,
    )
    return summary
