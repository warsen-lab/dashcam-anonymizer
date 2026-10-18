"""summary.md of a run: what was measured, what the numbers prove and what they leave to the user's spot check."""

from __future__ import annotations

from vc_label.contracts import LABEL_CLASSES
from vc_eval.metrics import CLASS_PASS, HEATMAP_MIN_TARGETS

DASH = "—"
CLS_NAME = {"face": "人脸", "plate": "车牌"}
PARALLEL_STAGES = ("decode", "main", "encode")  # the three pipeline threads
MAIN_STAGES = ("detect", "track", "mosaic")  # run one after another on the main thread
STAGE_NAME = {"decode": "解码", "main": "主线程", "encode": "编码", "detect": "检测（关键帧）", "track": "补框", "mosaic": "打码"}
INTEGRITY_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync")


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


def _pooled_at(summary: dict, fps: float) -> dict:
    return next(p for p in summary["pooled"] if p["fps"] == fps)


def spot_fps(summary: dict) -> float:
    """The rate the details and the spot-check page show: the candidate, else the smallest rate swept."""
    chosen = summary["chosen_fps"]
    return chosen if chosen is not None else min(p["fps"] for p in summary["pooled"])


def _thin_samples(coverage: dict) -> list[tuple[str, dict]]:
    """Classes that pass P1 although their sample cannot prove it: the lower end of the 95 % interval is under the bar."""
    return [
        (cls, coverage[cls])
        for cls in LABEL_CLASSES
        if coverage[cls]["total"] and coverage[cls]["value"] >= CLASS_PASS and coverage[cls]["ci"][0] < CLASS_PASS
    ]


def _thin_sample_line(cls: str, rate: dict) -> str:
    lead = "即使全部遮住" if rate["hits"] == rate["total"] else f"实测 {rate['hits']}/{rate['total']}"
    return f"- {CLS_NAME[cls]} n={rate['total']}：{lead}，95% 区间下限只有 {rate['ci'][0]:.1%}，样本不足以证明 ≥ {CLASS_PASS:.0%}"


def _candidate_line(summary: dict) -> str:
    chosen = summary["chosen_fps"]
    if chosen is None:
        return "- 候选检测频率：无满足 P1–P3 的频率"
    coverage = _pooled_at(summary, chosen)["coverage"]
    line = f"- 候选检测频率：{chosen:g} fps（P1–P3 通过；P4 待你看完 index.html 确认）"
    if coverage["p2"] == "n/a":
        line += f"（没有热力格达到 {HEATMAP_MIN_TARGETS} 个目标，P2 空洞满足；空间上的薄弱点只能靠 P4 抽查把关）"
    thin = _thin_samples(coverage)
    if thin:
        names = "、".join(CLS_NAME[cls] for cls, _ in thin)
        line += f"；注意：{names}的目标数太少，样本量不足以证明 ≥ {CLASS_PASS:.0%}（见覆盖率表下的提醒）"
    return line


def _head_lines(summary: dict) -> list[str]:
    first = summary["runs"][0] if summary["runs"] else {}
    return [
        "# 代理验证结果",
        "",
        f"- 模型：{summary['model']}；检测后端：{first.get('detector', '')}；编码器：{first.get('encoder', '')}",
        _candidate_line(summary),
        "- 本结果以老师融合标注为参照（代理验收），不是人工金标准验收。",
        "- 覆盖率读的是 `masks.jsonl`（`vc_redact` 声明要打码的框），不是打码后视频的像素：马赛克有没有真的打在画面上，"
        "不是这份报告测的，只由 C++ 单元测试和 P4（你看 index.html）验证。",
        "- 参照分两层：P1 / P2 只按高置信参照（没被 `low_score` / `face_single_source` 标记的框）判定；低置信参照单列，只报告，不设门槛。",
    ]


def _coverage_lines(summary: dict) -> list[str]:
    runs, pooled = summary["runs"], summary["pooled"]
    header = ["| 段 | fps | 人脸（95% 区间） | 车牌（95% 区间） | 最弱热力格 | 通过 |", "|---|---|---|---|---|---|"]
    lines = ["", "## 最终覆盖率（P1 / P2，高置信参照）"]
    for fps in sorted({r["fps"] for r in runs}):
        lines += ["", *header]
        for r in (r for r in runs if r["fps"] == fps):
            c = r["coverage"]
            lines.append(f"| {r['stem']} | {fps:g} | {_pct(c['face'])} | {_pct(c['plate'])} | {_weakest(c)} | — |")
        c = _pooled_at(summary, fps)["coverage"]
        lines.append(f"| 合计 | {fps:g} | {_pct(c['face'])} | {_pct(c['plate'])} | {_weakest(c)} | {'是' if c['passed'] else '否'} |")
        thin = [_thin_sample_line(cls, rate) for cls, rate in _thin_samples(c)]
        if thin:
            lines += ["", *thin]
    lines += ["", "合计行是各段目标数相加后的覆盖率，P1 / P2 以它为准，“通过”一栏只在合计行；各段行仅供参考。"]
    if any(r["coverage"]["p2"] == "n/a" for r in runs) or any(p["coverage"]["p2"] == "n/a" for p in pooled):
        lines += [
            "",
            f"n/a：没有热力格达到 {HEATMAP_MIN_TARGETS} 个目标，所以 P2 空洞满足（不阻止它成为候选）；空间上的薄弱点只能靠用户的抽查（P4）把关。",
        ]
    return lines


def _tier_lines(summary: dict) -> list[str]:
    lines = [
        "",
        "## 参照分层（必须遮住的目标数，各段合计）",
        "",
        "| 类别 | 高置信（计入 P1 / P2） | 低置信（只报告） | 合计 |",
        "|---|---|---|---|",
    ]
    for cls in LABEL_CLASSES:
        t = summary["reference"][cls]
        total = t["confident"] + t["uncertain"]
        if total:
            share = lambda n: f"{n}（{n / total:.1%}）"  # noqa: E731
            lines.append(f"| {CLS_NAME[cls]} | {share(t['confident'])} | {share(t['uncertain'])} | {total} |")
        else:
            lines.append(f"| {CLS_NAME[cls]} | 0 | 0 | 0 |")
    lines += [
        "",
        "低置信的目标不进 P1 / P2：它们占的比例越大，P1 / P2 代表的参照就越少，这一部分是否遮住只能靠 index.html 的抽查来判断。",
    ]
    return lines


def _uncertain_lines(summary: dict) -> list[str]:
    runs, pooled = summary["runs"], summary["pooled"]
    lines = [
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
        u = _pooled_at(summary, fps)["coverage"]["uncertain"]
        lines.append(f"| 合计 | {fps:g} | {_pct(u['face'])} | {_pct(u['plate'])} |")
    return lines


def _heatmap_lines(heatmap: dict) -> list[str]:
    lines = [
        "**热力图**（行 r0 起自上而下，列 c0 起自左向右；格内是 命中/目标数，— 表示该格没有目标；目标数 ≥ "
        f"{HEATMAP_MIN_TARGETS} 的格子才参与 P2）",
    ]
    for cls in LABEL_CLASSES:
        rows = heatmap[cls]
        lines += ["", CLS_NAME[cls], "", "| 行 | " + " | ".join(f"c{c}" for c in range(len(rows[0]))) + " |", "|---" * (len(rows[0]) + 1) + "|"]
        for r, row in enumerate(rows):
            cells = " | ".join(DASH if cell["total"] == 0 else f"{cell['hits']}/{cell['total']}" for cell in row)
            lines.append(f"| r{r} | {cells} |")
    return lines


def _detail_lines(summary: dict) -> list[str]:
    fps = spot_fps(summary)
    c = _pooled_at(summary, fps)["coverage"]
    which = f"候选 {fps:g} fps" if summary["chosen_fps"] is not None else f"最小的 {fps:g} fps，没有频率通过 P1–P3"
    lines = [
        "",
        f"## 细分（{which}，各段合计，高置信参照）",
        "",
        "**尺寸档**（像素；人脸按高、车牌按宽，只含必须遮住的目标）",
        "",
        "| 尺寸档 | 人脸 | 车牌 |",
        "|---|---|---|",
    ]
    lines += [f"| {label} | {_pct(c['bins']['face'][label])} | {_pct(c['bins']['plate'][label])} |" for label in c["bins"]["face"]]
    lines += [
        "",
        "| 项目 | 人脸 | 车牌 |",
        "|---|---|---|",
        f"| 时间补洞框（老师靠前后帧补出来的框，高置信） | {_pct(c['temporal_fill']['face'])} | {_pct(c['temporal_fill']['plate'])} |",
        f"| 尽力遮挡（目标太小，不要求遮住；两层合计） | {_pct(c['best_effort']['face'])} | {_pct(c['best_effort']['plate'])} |",
        "",
        *_heatmap_lines(c["heatmap"]),
    ]
    return lines


def _temporal_lines(runs: list[dict]) -> list[str]:
    lines = [
        "",
        "## 时间覆盖率",
        "",
        "以同一模型逐帧检测的框为参照，衡量稀疏检测加补框比逐帧检测少盖了多少；它不说明模型本身漏了多少。",
        "",
        "| 段 | fps | 人脸 | 车牌 |",
        "|---|---|---|---|",
    ]
    return lines + [f"| {r['stem']} | {r['fps']:g} | {_pct(r['temporal']['face'])} | {_pct(r['temporal']['plate'])} |" for r in runs]


def _integrity_lines(runs: list[dict]) -> list[str]:
    mark = lambda ok: "✓" if ok else "✗"  # noqa: E731
    lines = [
        "", "## 输出完整性（P3）", "",
        "| 段 | fps | 可解码 | 帧数 | 分辨率 | 时长 | 音频 | 音画同步 | 未通过项 |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        i = r["integrity"]
        cells = " | ".join(mark(i["checks"].get(name)) for name in INTEGRITY_CHECKS)
        lines.append(f"| {r['stem']} | {r['fps']:g} | {cells} | {_failed_details(i)} |")
    lines += ["", "## 误打面积（只报告）", "", "| 段 | fps | 最大 | 均值 | 95 分位 |", "|---|---|---|---|---|"]
    return lines + [f"| {r['stem']} | {r['fps']:g} | {r['stray']['max']:.2%} | {r['stray']['mean']:.2%} | {r['stray']['p95']:.2%} |" for r in runs]


def _busy(stages: dict, stage: str) -> float | None:
    """Busy seconds of a stage; stats.json without a main entry (older vc_redact) adds up the main thread's stages."""
    if stage == "main" and "main" not in stages:
        parts = [stages.get(s, {}).get("busy_s") for s in MAIN_STAGES]
        return sum(parts) if all(p is not None for p in parts) else None
    return stages.get(stage, {}).get("busy_s")


def _share(busy: float | None, whole: float | None) -> str:
    if busy is None:
        return DASH
    return f"{busy:.1f} s（{busy / whole:.0%}）" if whole else f"{busy:.1f} s（{DASH}）"


def _speed_row(r: dict) -> tuple[str, dict[str, float | None]]:
    elapsed = r.get("elapsed_s")
    busy = {s: _busy(r["stages"], s) for s in (*PARALLEL_STAGES, *MAIN_STAGES)}
    setup = r.get("setup_s")
    frames = r.get("frames")
    cells = [
        DASH if setup is None else f"{setup:.1f} s",
        *(_share(busy[s], elapsed) for s in PARALLEL_STAGES),
        *(_share(busy[s], busy["main"]) for s in MAIN_STAGES),
        DASH if elapsed is None else f"{elapsed:.1f} s",
        f"{frames / elapsed:.0f} fps" if frames and elapsed else DASH,
    ]
    return f"| {r['stem']} | {r['fps']:g} | {' | '.join(cells)} |", busy


def _speed_lines(summary: dict) -> list[str]:
    lines = [
        "",
        f"## 速度（本机，{_segment_text(summary['segments'])}）",
        "",
        "解码、主线程、编码是流水线里并行运行的三个线程，括号里是忙碌秒数占该次总耗时的比例，最接近 100% 的就是瓶颈。"
        "检测、补框、打码在主线程里串行，括号里是占主线程忙碌时间的比例；“检测（关键帧）”只统计做过检测的关键帧，不是每一帧。"
        "准备是第一帧进入流水线之前的时间（打开输入、加载模型和 CoreML 编译；编码器在编码线程收到第一帧时才打开，算在编码里），算在总耗时里；吞吐是帧数除以总耗时。"
        "后端见开头的检测后端与编码器。",
        "",
        "| 段 | fps | 准备 | " + " | ".join(STAGE_NAME[s] for s in (*PARALLEL_STAGES, *MAIN_STAGES)) + " | 总耗时 | 吞吐 |",
        "|---|---|---|" + "---|" * (len(PARALLEL_STAGES) + len(MAIN_STAGES) + 2),
    ]
    totals = dict.fromkeys(PARALLEL_STAGES, 0.0)
    elapsed_total = 0.0
    for r in summary["runs"]:
        line, busy = _speed_row(r)
        lines.append(line)
        if r.get("elapsed_s") and all(busy[s] is not None for s in PARALLEL_STAGES):
            elapsed_total += r["elapsed_s"]
            for s in PARALLEL_STAGES:
                totals[s] += busy[s]
    if elapsed_total > 0:
        slowest = max(PARALLEL_STAGES, key=totals.get)
        lines += [
            "",
            f"估计瓶颈：{STAGE_NAME[slowest]}（有完整计时的运行合计忙碌 {totals[slowest]:.1f} s，为这些运行合计总耗时的 {totals[slowest] / elapsed_total:.0%}）。",
        ]
    return lines


def _consistency_lines(c: dict) -> list[str]:
    return [
        "",
        "## C++ / Python 一致性",
        "",
        f"{c['images']} 张图、{c['boxes']} 个框，配上 {c['matched']} 个（其中分数 ≥ 0.3 的 {c['matched_confident']} 个）。"
        f"分数 ≥ 0.3 的配对最大坐标差 {c['max_diff_px']:.3f} px（门槛 < 1 px，判定只看这一项）；"
        f"所有配对最大坐标差 {c['max_diff_all_px']:.3f} px（含低分框，只报告，贴着阈值的低分框在两个运行时之间本来就会有出入）；"
        f"未配上（分数 ≥ 0.3）{c['unmatched_confident']} 个、（< 0.3）{c['unmatched_low']} 个：{'通过' if c['passed'] else '未通过'}。",
        "",
    ]


def summary_md(summary: dict) -> str:
    runs = summary["runs"]
    lines = [
        *_head_lines(summary),
        *_coverage_lines(summary),
        *_tier_lines(summary),
        *_uncertain_lines(summary),
        *_detail_lines(summary),
        *_temporal_lines(runs),
        *_integrity_lines(runs),
        *_speed_lines(summary),
        *_consistency_lines(summary["consistency"]),
    ]
    return "\n".join(lines)
