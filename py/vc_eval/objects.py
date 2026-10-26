"""Per-object and per-frame leaks against the vc_gt answer key (spec 2026-10-22 §4, §10): an object leaks on a sampled
frame where the masks leave its must-mask-size box uncovered and the redacted frame still shows it readable; the bars
are 99.9 % without a leak over at least 3,000 readable real objects, 99.5 % of must-mask-size frames without a leak, at
most 0.1 % of the people's audit boxes outside every judged track."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vc_eval.metrics import Rate, covered_fraction, must_mask
from vc_gt.audit import AuditBox
from vc_gt.crops import leak_frames, write_crop
from vc_gt.judge import REAL, frame_id, object_id
from vc_gt.tracks import Track
from vc_hardneg.judgments import HUMAN, Judgment, resolve
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box

ALPHA = 0.05
NEEDED_OBJECTS = 3000
NO_LEAK_PASS = 0.999
FRAME_NO_LEAK_PASS = 0.995
MISSED_BY_ALL_MAX = 0.001
AUDIT_IOU = 0.3
NEIGHBOURS = 5  # original-rate frames either side of a leak whose cover the report shows


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), 0 < p < 1, summed in log space."""
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(n + 1)
    return min(1.0, sum(math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(k + 1)))


def _root(f) -> float:
    """The p in (0, 1) where the increasing function f crosses 0, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """The exact two-sided 1 - alpha interval of a binomial proportion k / n."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _root(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    hi = 1.0 if k == n else _root(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return (lo, hi)


@dataclass(frozen=True)
class Leak:
    track: int
    frames: tuple[int, ...]  # sampled frames where the object was readable and not masked


@dataclass(frozen=True)
class ClassResult:
    cls: str
    readable: int  # real objects readable on at least one frame
    leaks: tuple[Leak, ...]
    frame_no_leak: Rate  # must-mask-size frames of real objects without a leak
    missed_by_all: Rate | None  # hits: audit boxes no track holds; None without an audit

    @property
    def no_leak(self) -> Rate:
        return Rate(self.readable - len(self.leaks), self.readable)

    @property
    def passed(self) -> bool:
        return (self.readable >= NEEDED_OBJECTS and self.no_leak.value >= NO_LEAK_PASS
                and self.frame_no_leak.value >= FRAME_NO_LEAK_PASS and self.missed_by_all is not None
                and self.missed_by_all.total > 0 and self.missed_by_all.value <= MISSED_BY_ALL_MAX)


def _held(a: AuditBox, tracks: list[Track]) -> bool:
    """Whether some of these tracks of the audit box's class has a box on its frame overlapping it by AUDIT_IOU."""
    for t in tracks:
        b = t.box_at(a.frame) if t.cls == a.box.cls else None
        if b is not None and iou(b, a.box) >= AUDIT_IOU:
            return True
    return False


def evaluate_objects(tracks: list[Track], objects: dict[str, str], frames: dict[str, str],
                     masks_by_frame: dict[int, list[Box]], frame_height: int,
                     audit: list[AuditBox] | None = None) -> dict[str, ClassResult]:
    """objects: object id -> final verdict of question A; frames: frame id -> final verdict of question B."""
    out = {}
    for cls in LABEL_CLASSES:
        readable, leaks, no_leak = 0, [], Rate()
        for t in (t for t in tracks if t.cls == cls and objects.get(object_id(t.id)) in REAL):
            leaked = tuple(f for f in leak_frames(t, masks_by_frame, frame_height)
                           if frames.get(frame_id(t.id, f)) == "readable")
            for f, b in zip(t.frames, t.boxes):
                if must_mask(b, frame_height):
                    no_leak = no_leak.add(f not in leaked)
            if objects[object_id(t.id)] == "real_readable" or leaked:
                readable += 1
            if leaked:
                leaks.append(Leak(t.id, leaked))
        missed = None
        if audit is not None:
            mine = [a for a in audit if a.box.cls == cls]
            judged = [t for t in tracks if object_id(t.id) in objects]  # an unjudged track holds nothing (spec §8)
            missed = Rate(sum(not _held(a, judged) for a in mine), len(mine))
        out[cls] = ClassResult(cls, readable, tuple(leaks), no_leak, missed)
    return out


def find_leaks(tracks: list[Track], frames: dict[str, str], masks_by_frame: dict[int, list[Box]],
               frame_height: int) -> dict[str, tuple[Leak, ...]]:
    """Per class, every track with a must-mask-size unmasked frame judged readable, whatever question A said (spec §12:
    the leak finder)."""
    out = {}
    for cls in LABEL_CLASSES:
        found = []
        for t in (t for t in tracks if t.cls == cls):
            leaked = tuple(f for f in leak_frames(t, masks_by_frame, frame_height)
                           if frames.get(frame_id(t.id, f)) == "readable")
            if leaked:
                found.append(Leak(t.id, leaked))
        out[cls] = tuple(found)
    return out


def model_error_rate(judgments: Iterable[Judgment], sample_ids: set[str]) -> Rate:
    """Over the sampled ids people judged: how often their verdict differs from the last model verdict."""
    judgments = list(judgments)
    model = resolve(judgments, include_human=False)
    people = {j.id: j for j in judgments if j.judge == HUMAN}
    hits = total = 0
    for i in sorted(sample_ids):
        if i in people and i in model and model[i].judge != HUMAN:
            total += 1
            hits += people[i].verdict != model[i].verdict
    return Rate(hits, total)


def _pct(rate: Rate) -> str:
    if not rate.total:
        return "—"
    lo, hi = clopper_pearson(rate.hits, rate.total)
    return f"{rate.value:.2%}（{lo:.2%}–{hi:.2%}），n={rate.total}"


def write_report(results: dict[str, ClassResult], error_rates: dict[str, Rate], out_dir: Path, tracks: list[Track],
                 images: dict[int, Path], masks_by_frame: dict[int, list[Box]],
                 found: dict[str, tuple[Leak, ...]] | None = None) -> Path:
    """out_dir/summary.md with every leak first (`found`, the leak finder's, else the table's), then the table and the
    model error rates; a crop per leaked frame (from `images`, the redacted frames: what a viewer sees) in
    out_dir/leaks/."""
    import cv2

    out_dir = Path(out_dir)
    (out_dir / "leaks").mkdir(parents=True, exist_ok=True)
    by_id = {t.id: t for t in tracks}
    leaks = found if found is not None else {cls: r.leaks for cls, r in results.items()}
    lines = ["# 泄露查找（在打码后的画面上判断）", "",
             "人工确认过的为准；任何候选轨迹都查，不看问题 A 的结论。", ""]
    for cls, ls in leaks.items():
        for leak in ls:
            t = by_id[leak.track]
            for f in leak.frames:
                box = t.box_at(f)
                name = f"o{t.id}@{f:06d}.jpg"
                write_crop(cv2.imread(str(images[f])), box, out_dir / "leaks" / name, 3.0, 96)
                around = ", ".join(f"{g}:{covered_fraction(box, masks_by_frame.get(g, [])):.0%}"
                                   for g in range(f - NEIGHBOURS, f + NEIGHBOURS + 1))
                lines.append(f"- {cls} o{t.id} 第 {f} 帧：![](leaks/{name}) 前后原始帧遮盖比例 {around}")
    lines += ["", "## 按目标与按帧比例（仅供参考：分母依赖问题 A，问题 A 未过校准，见设计 §11、§12）", "",
              "| 类别 | 看得清的真目标 | 泄露 | 按目标不泄露率（95% 区间） | 按帧不泄露率（95% 区间） | 大家一起漏 | 达标 |",
             "|---|---|---|---|---|---|---|"]
    for cls, r in results.items():
        missed = _pct(r.missed_by_all) if r.missed_by_all is not None else "未抽查"
        high = r.missed_by_all is not None and r.missed_by_all.total and r.missed_by_all.value > MISSED_BY_ALL_MAX
        flag = " ⚠ 高于 0.1%，主指标会被高估" if high else ""
        verdict = ("是" if r.passed else "否") if r.readable >= NEEDED_OBJECTS else \
            f"不下结论（差 {NEEDED_OBJECTS - r.readable} 个目标）"
        lines.append(f"| {cls} | {r.readable} | {len(r.leaks)} | {_pct(r.no_leak)} | {_pct(r.frame_no_leak)} | "
                     f"{missed}{flag} | {verdict} |")
    lines += ["", "## 模型出错率（H2 抽核里被人工推翻的比例）", ""]
    lines += [f"- {name}：{_pct(rate)}" for name, rate in error_rates.items()]
    page = out_dir / "summary.md"
    page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return page
