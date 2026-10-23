# 99.9% Test Set and Per-Object Metric — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an answer key for new held-out dashcam videos (every plate and face as a tracked object, judged real/readable by an LLM and checked by a person) and a `vc_eval objects` report that measures per-object leaks and per-frame recall against it, as specified in `docs/superpowers/specs/2026-10-22-gt-test-set-design.md`.

**Architecture:** A new package `py/vc_gt` finds every plausible plate and face on 10 fps frames (teachers at low thresholds + 2×2 OWLv2 tiles + the student), links them into tracks, crops them, asks the LLM through the existing `vc_hardneg` command judge, and writes local review pages for people. `py/vc_eval/objects.py` turns tracks + judgments + a vc_redact `masks.jsonl` into the metrics, Clopper-Pearson intervals and a report.

**Tech Stack:** Python 3.12, OpenCV, NumPy, torch/transformers (teachers, already cached), pytest; the C++ `vc_detect` / `vc_extract` / `vc_redact` binaries in `build/cpp-release/`.

## Global Constraints

- Reply to the user in Simplified Chinese.
- Never run full test suites on this machine: run only the test files named in each task (`cd py && ../.venv/bin/python -m pytest -q tests/<file>`).
- Metered network: nothing may reach the network unless the user approved it. LLM calls only run with `VC_HARDNEG_ALLOW_LLM=1` (the existing `CommandJudge` refuses otherwise). Model weights load from the cache only (`vc_label.paths.offline_model_env()`).
- Test videos never enter training: no stem under `data/gt/` or `data/videos/test-v2/` may be passed to `vc_train dataset --stem`.
- Kill processes only by PID or `pkill -x <exact name>`.
- Thresholds from the spec, verbatim: OWLv2 0.05 with 2×2 tiles overlapping 20 %; RetinaFace 0.2; YuNet 0.3; student 0.01; link IoU ≥ 0.3 or centre distance < 1.5 box widths, gap ≤ 3 sampled frames; 3 crops per object; audit 50 + 50 verdicts and 150 frames per class; pass = ≥ 3,000 readable real objects, no-leak ≥ 99.9 %, per-frame recall ≥ 99.5 %, missed-by-all ≤ 0.1 %; 95 % Clopper-Pearson intervals.
- Code style: match the surrounding code (module docstrings citing the spec, frozen dataclasses, `from __future__ import annotations`, short comments only where they explain a why).

## File Structure

| File | Responsibility |
|---|---|
| `py/vc_hardneg/judgments.py` (modify) | `append_judgments` takes the allowed verdicts |
| `py/vc_hardneg/judges.py` (modify) | `CommandJudge.run` becomes public so vc_gt can call the LLM |
| `py/vc_gt/__init__.py`, `__main__.py` | package, `python -m vc_gt` |
| `py/vc_gt/tracks.py` | `Track`, `TrackParams`, `link_tracks`, `write_tracks`, `read_tracks` |
| `py/vc_gt/config.py`, `configs/gt.json` | `GtConfig` and its loader |
| `py/vc_gt/candidates.py` | `tile_windows`, `merge_duplicates`, `TeacherFinder`, `build_candidates` |
| `py/vc_gt/crops.py` | crop windows, sharpness, frame picking, unmasked frames, crop index |
| `py/vc_gt/judge.py` | verdict sets, ids, grid image, `ask`, `judge_objects`, `judge_frames`, `make_runner` |
| `py/vc_gt/audit.py` | H1/H2 selection, audit frame sample, review page, annotate page, imports |
| `py/vc_gt/cli.py` | `candidates`, `judge`, `review`, `annotate`, `import-human` |
| `py/vc_eval/objects.py` | Clopper-Pearson, `evaluate_objects`, model error rate, report |
| `py/vc_eval/cli.py` (modify) | `vc_eval objects` |
| `py/README.md` (modify) | how to build a test set |
| `py/tests/test_gt_*.py`, `py/tests/test_eval_objects.py` | tests |

Answer-key layout per segment, `data/gt/<stem>/` (not in git): `boxes.jsonl`, `processed.txt` (`<frame> <persons>` per line), `tracks.jsonl`, `crops/objects/`, `crops/frames/`, `crops.jsonl` (crop index), `llm/` (grids sent to the LLM), `objects.jsonl`, `frames.jsonl` (judgments), `h2_sample.json`, `audit_frames.json`, `audit.jsonl`.

---

### Task 1: Let vc_gt reuse the vc_hardneg judgment file and LLM command

**Files:**
- Modify: `py/vc_hardneg/judgments.py`
- Modify: `py/vc_hardneg/judges.py:131-192`
- Test: `py/tests/test_hardneg_judges.py`

**Interfaces:**
- Produces: `append_judgments(path: Path, items: Iterable[Judgment], verdicts: tuple[str, ...] = VERDICTS) -> None`; `CommandJudge.run(image: Path, prompt: str, cls: str) -> tuple[str | None, str]` (stdout or None, and why).

- [ ] **Step 1: Write the failing tests** (append to `py/tests/test_hardneg_judges.py`)

```python
def test_append_judgments_takes_other_verdict_sets(tmp_path):
    from vc_hardneg.judgments import Judgment, append_judgments, read_judgments

    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("o1", "real_readable", "llm:x")], verdicts=("real_readable", "unsure"))
    assert read_judgments(path)[0].verdict == "real_readable"
    with pytest.raises(ValueError, match="real_readable"):
        append_judgments(path, [Judgment("o2", "real_readable", "llm:x")])  # the default set is yes / no / unsure
    assert len(read_judgments(path)) == 1


def test_command_judge_run_is_public_and_fills_the_command(tmp_path):
    from vc_hardneg.config import LlmConfig
    from vc_hardneg.judges import CommandJudge

    cfg = LlmConfig("echo", (sys.executable, "-c", "import sys; print(sys.argv[1:])", "{image}", "{cls_zh}", "{prompt}"),
                    10.0, "")
    out, why = CommandJudge(cfg, env={"VC_HARDNEG_ALLOW_LLM": "1"}).run(tmp_path / "a.png", "问题", "plate")
    assert why == "exit 0" and str(tmp_path / "a.png") in out and "车牌" in out and "问题" in out
```

If `sys` or `pytest` is not imported at the top of the test file yet, add `import sys` / `import pytest` there.

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_hardneg_judges.py -k "other_verdict_sets or run_is_public"`
Expected: 2 failed (`unexpected keyword argument 'verdicts'`, `has no attribute 'run'`).

- [ ] **Step 3: Implement**

In `py/vc_hardneg/judgments.py` replace `append_judgments` with:

```python
def append_judgments(path: Path, items: Iterable[Judgment], verdicts: tuple[str, ...] = VERDICTS) -> None:
    """Appends every item, or none when one has a verdict outside `verdicts` (ValueError). vc_gt passes its own
    verdict sets (spec 2026-10-22 §2)."""
    items = list(items)
    for j in items:
        if j.verdict not in verdicts:
            raise ValueError(f"verdict must be one of {verdicts}, got {j.verdict!r} for {j.id}")
    with Path(path).open("a", encoding="utf-8") as f:
        for j in items:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")
```

In `py/vc_hardneg/judges.py` rename the method `_run` to `run` (definition at line 131 and its two callers in `judge` and `judge_batch`), and change its docstring to:

```python
    def run(self, image: Path, prompt: str, cls: str) -> tuple[str | None, str]:
        """The command's stdout for this image and prompt, or None and why it did not answer (also used by vc_gt)."""
```

Then check nothing else calls `_run`: `grep -rn "\._run(" vc_hardneg tests` must print nothing.

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_hardneg_judges.py tests/test_hardneg_review.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add py/vc_hardneg/judgments.py py/vc_hardneg/judges.py py/tests/test_hardneg_judges.py
git commit -m "refactor: vc_hardneg judgments take other verdict sets and CommandJudge.run is public"
```

---

### Task 2: Link candidate boxes into tracks

**Files:**
- Create: `py/vc_gt/__init__.py`, `py/vc_gt/tracks.py`
- Test: `py/tests/test_gt_tracks.py`

**Interfaces:**
- Produces: `TrackParams(min_iou=0.3, max_center_dist=1.5, max_gap=3)`; `Track(id: int, cls: str, frames: tuple[int, ...], boxes: tuple[Box, ...])` with `box_at(frame) -> Box | None`; `link_tracks(by_frame: dict[int, list[Box]], order: list[int], p: TrackParams = TrackParams()) -> list[Track]`; `write_tracks(path: Path, video: str, tracks: list[Track]) -> None`; `read_tracks(path: Path) -> list[Track]`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_tracks.py`

```python
from vc_gt.tracks import TrackParams, link_tracks, read_tracks, write_tracks
from vc_label.contracts import Box


def plate(x, y=500, w=40, h=12, score=0.9):
    return Box("plate", x, y, x + w, y + h, score)


def test_a_box_moving_a_little_each_frame_is_one_track():
    by_frame = {f: [plate(100 + 5 * k)] for k, f in enumerate(range(0, 60, 6))}
    tracks = link_tracks(by_frame, sorted(by_frame))
    assert len(tracks) == 1 and tracks[0].frames == tuple(range(0, 60, 6))


def test_two_plates_side_by_side_stay_apart():
    by_frame = {f: [plate(100 + 3 * k), plate(900 - 3 * k)] for k, f in enumerate(range(0, 30, 6))}
    tracks = link_tracks(by_frame, sorted(by_frame))
    assert [t.boxes[0].x1 for t in tracks] == [100, 900] and all(len(t.frames) == 5 for t in tracks)


def test_a_gap_of_max_gap_frames_continues_and_one_more_splits():
    order = list(range(10))
    by_frame = {0: [plate(100)], 4: [plate(102)]}  # frames 1, 2, 3 missed: a gap of 3
    assert len(link_tracks(by_frame, order, TrackParams(max_gap=3))) == 1
    assert len(link_tracks(by_frame, order, TrackParams(max_gap=2))) == 2


def test_a_fast_small_plate_links_by_centre_distance():
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(125, w=20)]}, [0, 1])) == 1  # 25 px < 1.5 x 20
    assert len(link_tracks({0: [plate(100, w=20)], 1: [plate(140, w=20)]}, [0, 1])) == 2  # 40 px > 30


def test_an_overlapping_box_wins_over_a_merely_close_one():
    by_frame = {0: [plate(100)], 1: [plate(130), plate(102)]}  # 102 overlaps, 130 is only close
    tracks = link_tracks(by_frame, [0, 1])
    assert [b.x1 for b in tracks[0].boxes] == [100, 102]


def test_classes_never_share_a_track():
    face = Box("face", 100, 500, 140, 512, 0.9)
    assert sorted(t.cls for t in link_tracks({0: [plate(100)], 1: [face]}, [0, 1])) == ["face", "plate"]


def test_ids_follow_the_first_frame_and_survive_a_round_trip(tmp_path):
    tracks = link_tracks({0: [plate(500)], 6: [plate(502), plate(100)]}, [0, 6])
    assert [(t.id, t.frames[0], t.boxes[0].x1) for t in tracks] == [(0, 0, 500), (1, 6, 100)]
    write_tracks(tmp_path / "t.jsonl", "v.mp4", tracks)
    back = read_tracks(tmp_path / "t.jsonl")
    assert back == tracks and back[0].box_at(6).x1 == 502 and back[0].box_at(3) is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_tracks.py`
Expected: collection error, `No module named 'vc_gt'`.

- [ ] **Step 3: Implement**

`py/vc_gt/__init__.py`:

```python
"""The 99.9 % answer key: every plate and face of a held-out video as a judged object (spec 2026-10-22)."""
```

`py/vc_gt/tracks.py`:

```python
"""Candidate boxes linked into objects (spec 2026-10-22 §2 step 2, §3): one track per plate or face over time."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from vc_label.boxes import iou
from vc_label.contracts import Box


@dataclass(frozen=True)
class TrackParams:
    min_iou: float = 0.3
    max_center_dist: float = 1.5  # in widths of the wider of the two boxes
    max_gap: int = 3  # sampled frames a track may go without a box and still continue


@dataclass(frozen=True)
class Track:
    id: int
    cls: str
    frames: tuple[int, ...]  # increasing
    boxes: tuple[Box, ...]  # one per frame

    def box_at(self, frame: int) -> Box | None:
        return self.boxes[self.frames.index(frame)] if frame in self.frames else None


def _match(a: Box, b: Box, p: TrackParams) -> float | None:
    """How well b continues a track whose last box is a (higher is better), None when it cannot. Any overlap match
    (> 1) beats any distance match (<= 1)."""
    overlap = iou(a, b)
    if overlap >= p.min_iou:
        return 1.0 + overlap
    (ax, ay), (bx, by) = a.center, b.center
    limit = p.max_center_dist * max(a.width, b.width)
    dist = math.hypot(ax - bx, ay - by)
    return 1.0 - dist / limit if limit > 0 and dist < limit else None


def link_tracks(by_frame: dict[int, list[Box]], order: list[int], p: TrackParams = TrackParams()) -> list[Track]:
    """Links per-frame boxes into tracks of one class each. At every frame of `order` the best (track, box) pairs are
    taken greedily; a box no track takes starts a track; a track that misses more than p.max_gap frames of `order` in
    a row ends. Ids count from 0 by first frame, then left edge."""
    open_tracks: list[dict] = []
    ended: list[dict] = []
    for i, frame in enumerate(order):
        alive = []
        for t in open_tracks:
            (ended if i - t["last"] - 1 > p.max_gap else alive).append(t)
        open_tracks = alive
        boxes = by_frame.get(frame, [])
        pairs = [(score, ti, bi) for ti, t in enumerate(open_tracks) for bi, b in enumerate(boxes)
                 if b.cls == t["cls"] and (score := _match(t["boxes"][-1], b, p)) is not None]
        taken_t: set[int] = set()
        taken_b: set[int] = set()
        for _, ti, bi in sorted(pairs, key=lambda x: -x[0]):
            if ti in taken_t or bi in taken_b:
                continue
            taken_t.add(ti)
            taken_b.add(bi)
            t = open_tracks[ti]
            t["frames"].append(frame)
            t["boxes"].append(boxes[bi])
            t["last"] = i
        open_tracks += [{"cls": b.cls, "frames": [frame], "boxes": [b], "last": i}
                        for bi, b in enumerate(boxes) if bi not in taken_b]
    every = sorted(ended + open_tracks, key=lambda t: (t["frames"][0], t["boxes"][0].x1, t["cls"]))
    return [Track(k, t["cls"], tuple(t["frames"]), tuple(t["boxes"])) for k, t in enumerate(every)]


def write_tracks(path: Path, video: str, tracks: list[Track]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for t in tracks:
            boxes = [[round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(b.score, 4), b.src]
                     for b in t.boxes]
            f.write(json.dumps({"id": t.id, "cls": t.cls, "video": video, "frames": list(t.frames), "boxes": boxes},
                               ensure_ascii=False) + "\n")


def read_tracks(path: Path) -> list[Track]:
    tracks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                boxes = tuple(Box(r["cls"], *map(float, b[:5]), src=str(b[5])) for b in r["boxes"])
                tracks.append(Track(int(r["id"]), r["cls"], tuple(int(x) for x in r["frames"]), boxes))
    return tracks
```

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_tracks.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add py/vc_gt/__init__.py py/vc_gt/tracks.py py/tests/test_gt_tracks.py
git commit -m "feat: vc_gt links candidate boxes into tracks"
```

---

### Task 3: configs/gt.json and its loader

**Files:**
- Create: `configs/gt.json`, `py/vc_gt/config.py`
- Test: `py/tests/test_gt_config.py`

**Interfaces:**
- Consumes: `TrackParams` (Task 2); `vc_hardneg.config.LlmConfig`.
- Produces: `GtConfig` (fields below), `AuditConfig(objects, frames, frames_per_class, seed)`, `GtLlm(name, command, timeout_s, parallel, batch, tile, object_prompt, frame_prompt)` with `command_config() -> LlmConfig`; `load_config(path: Path = DEFAULT_CONFIG) -> GtConfig`; `DEFAULT_CONFIG`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_config.py`

```python
import json

import pytest

from vc_gt.config import DEFAULT_CONFIG, load_config


def test_the_repository_config_holds_the_spec_values():
    c = load_config()
    assert (c.owlv2_score, c.tile_grid, c.tile_overlap) == (0.05, 2, 0.2)
    assert (c.retinaface_score, c.yunet_score, c.student_score) == (0.2, 0.3, 0.01)
    assert (c.track.min_iou, c.track.max_center_dist, c.track.max_gap) == (0.3, 1.5, 3)
    assert c.crops_per_object == 3
    assert (c.audit.objects, c.audit.frames, c.audit.frames_per_class) == (50, 50, 150)
    assert "{count}" in c.llm.object_prompt and "{cls_zh}" in c.llm.frame_prompt
    assert c.llm.command_config().command == c.llm.command


def _write(tmp_path, change):
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    change(d)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(extra=1), "keys"),
    (lambda d: d.update(owlv2_score=1.5), "owlv2_score"),
    (lambda d: d.update(tile_overlap=0.5), "tile_overlap"),
    (lambda d: d.update(crops_per_object=0), "crops_per_object"),
    (lambda d: d["llm"].update(object_prompt="no placeholders"), "object_prompt"),
    (lambda d: d["llm"].update(batch=0), "batch"),
    (lambda d: d["track"].update(max_gap=-1), "max_gap"),
])
def test_bad_configs_are_refused_naming_the_key(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, change))
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_config.py`
Expected: collection error, `No module named 'vc_gt.config'`.

- [ ] **Step 3: Implement**

`configs/gt.json`:

```json
{
  "owlv2_score": 0.05,
  "tile_grid": 2,
  "tile_overlap": 0.2,
  "retinaface_score": 0.2,
  "yunet_score": 0.3,
  "coco_score": 0.3,
  "student_score": 0.01,
  "merge_iou": 0.5,
  "track": {"min_iou": 0.3, "max_center_dist": 1.5, "max_gap": 3},
  "crops_per_object": 3,
  "crop_factor": 3.0,
  "min_crop_px": 96,
  "audit": {"objects": 50, "frames": 50, "frames_per_class": 150, "seed": 20261022},
  "llm": {
    "name": "codex",
    "command": ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-c",
                "model_reasoning_effort=\"low\"", "--image={image}", "{prompt}"],
    "timeout_s": 180,
    "parallel": 4,
    "batch": 4,
    "tile": 384,
    "object_prompt": "这张图由 {count} 组小图拼成，每组左上角有黄色编号 1 到 {count}；同一组的几张小图是行车记录仪画面里同一个目标在不同时刻的截图，红框是目标。对每一组判断：红框里是不是真实的{cls_zh}；如果是，其中最清楚的一张，暂停并放大后能不能认出（车牌：能读出大部分字符；人脸：认识这个人的人能认出是谁）。只回答一个 JSON 数组，每组一项：[{\"n\": 编号, \"verdict\": \"real_readable\" | \"real_unreadable\" | \"not_real\" | \"unsure\"}]。拿不准就答 unsure。",
    "frame_prompt": "这张图由 {count} 张小图拼成，每张左上角有黄色编号 1 到 {count}；每张来自行车记录仪画面，红框里是一块真实的{cls_zh}。对每张判断：暂停并放大这张画面，能不能认出这个{cls_zh}（车牌：能读出大部分字符；人脸：认识这个人的人能认出是谁）。只回答一个 JSON 数组，每张一项：[{\"n\": 编号, \"verdict\": \"readable\" | \"unreadable\" | \"unsure\"}]。拿不准就答 unsure。"
  }
}
```

`py/vc_gt/config.py`:

```python
"""configs/gt.json: candidate search, linking, crops, the LLM's two questions and the audit sample
(spec 2026-10-22 §3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vc_gt.tracks import TrackParams
from vc_hardneg.config import LlmConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gt.json"  # the repository's, wherever this runs
_KEYS = {"owlv2_score", "tile_grid", "tile_overlap", "retinaface_score", "yunet_score", "coco_score", "student_score",
         "merge_iou", "track", "crops_per_object", "crop_factor", "min_crop_px", "audit", "llm"}
_TRACK_KEYS = {"min_iou", "max_center_dist", "max_gap"}
_AUDIT_KEYS = {"objects", "frames", "frames_per_class", "seed"}
_LLM_KEYS = {"name", "command", "timeout_s", "parallel", "batch", "tile", "object_prompt", "frame_prompt"}


@dataclass(frozen=True)
class AuditConfig:
    objects: int  # H2: model verdicts on objects people re-judge
    frames: int  # H2: model verdicts on unmasked frames people re-judge
    frames_per_class: int  # H3: whole frames people box every plate (face) on
    seed: int


@dataclass(frozen=True)
class GtLlm:
    name: str
    command: tuple[str, ...]  # argv; {image}, {cls}, {cls_zh} and {prompt} are filled in
    timeout_s: float
    parallel: int
    batch: int  # objects (frames) per call, on one numbered grid
    tile: int  # a crop's side on the grid, px
    object_prompt: str  # question A; {count}, {cls_zh} filled in
    frame_prompt: str  # question B

    def command_config(self) -> LlmConfig:
        """The settings vc_hardneg's CommandJudge runs the command with."""
        return LlmConfig(self.name, self.command, self.timeout_s, prompt="")


@dataclass(frozen=True)
class GtConfig:
    owlv2_score: float
    tile_grid: int
    tile_overlap: float
    retinaface_score: float
    yunet_score: float
    coco_score: float
    student_score: float
    merge_iou: float  # boxes of one class overlapping this much on a frame are one candidate
    track: TrackParams
    crops_per_object: int
    crop_factor: float  # a crop's side in target sizes
    min_crop_px: float
    audit: AuditConfig
    llm: GtLlm


def _check(ok: bool, path: Path, what: str) -> None:
    if not ok:
        raise ValueError(f"{path}: {what}")


def _keys(d: object, keys: set[str], path: Path, where: str) -> dict:
    _check(isinstance(d, dict) and set(d) == keys, path, f"{where} keys must be exactly {sorted(keys)}")
    return d


def load_config(path: Path = DEFAULT_CONFIG) -> GtConfig:
    """Raises ValueError naming the key for unknown or missing keys and values out of range."""
    path = Path(path)
    d = _keys(json.loads(path.read_text(encoding="utf-8")), _KEYS, path, "top-level")
    t = _keys(d["track"], _TRACK_KEYS, path, "track")
    a = _keys(d["audit"], _AUDIT_KEYS, path, "audit")
    m = _keys(d["llm"], _LLM_KEYS, path, "llm")
    for k in ("owlv2_score", "retinaface_score", "yunet_score", "coco_score", "student_score", "merge_iou"):
        _check(0 <= d[k] <= 1, path, f"{k} must be in [0, 1]")
    _check(int(d["tile_grid"]) >= 1, path, "tile_grid must be >= 1")
    _check(0 <= d["tile_overlap"] < 0.5, path, "tile_overlap must be in [0, 0.5)")
    _check(int(d["crops_per_object"]) >= 1, path, "crops_per_object must be >= 1")
    _check(d["crop_factor"] >= 1 and d["min_crop_px"] >= 16, path, "crop_factor must be >= 1 and min_crop_px >= 16")
    _check(0 <= t["min_iou"] <= 1 and t["max_center_dist"] > 0, path, "track.min_iou in [0, 1], max_center_dist > 0")
    _check(int(t["max_gap"]) >= 0, path, "track.max_gap must be >= 0")
    _check(all(int(a[k]) >= 0 for k in ("objects", "frames", "frames_per_class")), path, "audit sizes must be >= 0")
    _check(int(m["parallel"]) >= 1, path, "llm.parallel must be >= 1")
    _check(int(m["batch"]) >= 1, path, "llm.batch must be >= 1")
    _check(int(m["tile"]) >= 64, path, "llm.tile must be >= 64")
    for k in ("object_prompt", "frame_prompt"):
        _check("{count}" in m[k] and "{cls_zh}" in m[k], path, f"llm.{k} must hold {{count}} and {{cls_zh}}")
    return GtConfig(
        owlv2_score=float(d["owlv2_score"]), tile_grid=int(d["tile_grid"]), tile_overlap=float(d["tile_overlap"]),
        retinaface_score=float(d["retinaface_score"]), yunet_score=float(d["yunet_score"]),
        coco_score=float(d["coco_score"]), student_score=float(d["student_score"]), merge_iou=float(d["merge_iou"]),
        track=TrackParams(float(t["min_iou"]), float(t["max_center_dist"]), int(t["max_gap"])),
        crops_per_object=int(d["crops_per_object"]), crop_factor=float(d["crop_factor"]),
        min_crop_px=float(d["min_crop_px"]),
        audit=AuditConfig(int(a["objects"]), int(a["frames"]), int(a["frames_per_class"]), int(a["seed"])),
        llm=GtLlm(str(m["name"]), tuple(str(x) for x in m["command"]), float(m["timeout_s"]), int(m["parallel"]),
                  int(m["batch"]), int(m["tile"]), str(m["object_prompt"]), str(m["frame_prompt"])),
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_config.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add configs/gt.json py/vc_gt/config.py py/tests/test_gt_config.py
git commit -m "feat: configs/gt.json holds the answer key's thresholds, prompts and audit sizes"
```

---

### Task 4: Find every candidate and build tracks

**Files:**
- Create: `py/vc_gt/candidates.py`
- Test: `py/tests/test_gt_candidates.py`

**Interfaces:**
- Consumes: `GtConfig`, `load_config` (Task 3); `link_tracks`, `write_tracks` (Task 2); `vc_label.contracts` (`Box`, `Detection`, `read_frames_csv`, `read_jsonl`, `write_jsonl`, `LABEL_CLASSES`).
- Produces: `FrameFinds(boxes: list[Box], persons: int)`; `Finder` protocol with `find(bgr) -> FrameFinds`; `tile_windows(width, height, grid, overlap) -> list[Box]`; `merge_duplicates(boxes, merge_iou) -> list[Box]`; `TeacherFinder(cfg: GtConfig, device: str = "auto")`; `build_candidates(frames_dir: Path, out_dir: Path, finder, student: dict[int, list[Box]], cfg: GtConfig, limit: int | None = None) -> int`; `read_persons(out_dir: Path) -> dict[int, int]`; constants `BOXES = "boxes.jsonl"`, `PROCESSED = "processed.txt"`, `TRACKS = "tracks.jsonl"`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_candidates.py`

```python
import csv

import cv2
import numpy as np

from vc_gt.candidates import (BOXES, PROCESSED, TRACKS, FrameFinds, build_candidates, merge_duplicates,
                              read_persons, tile_windows)
from vc_gt.config import load_config
from vc_gt.tracks import read_tracks
from vc_label.contracts import Box


def test_tile_windows_cover_the_frame_and_overlap_their_neighbours():
    w = tile_windows(100, 50, 2, 0.2)
    assert [(b.x1, b.y1, b.x2, b.y2) for b in w] == [(0, 0, 60, 30), (40, 0, 100, 30), (0, 20, 60, 50), (40, 20, 100, 50)]


def test_merge_duplicates_keeps_the_best_box_and_every_source():
    owl = Box("plate", 100, 100, 140, 112, 0.3, "teacher_owlv2")
    student = Box("plate", 101, 100, 141, 112, 0.6, "student")
    face = Box("face", 100, 100, 140, 112, 0.5, "teacher_yunet")  # another class: never merged
    out = merge_duplicates([owl, student, face], 0.5)
    assert [(b.cls, b.x1, b.score, b.src) for b in out] == [
        ("plate", 101, 0.6, "student+teacher_owlv2"), ("face", 100, 0.5, "teacher_yunet")]


class _Finder:
    """A plate drifting right; frame n of the fake holds the plate at x = 100 + 4 n."""

    def __init__(self):
        self.calls = 0

    def find(self, bgr):
        n = self.calls
        self.calls += 1
        return FrameFinds([Box("plate", 100 + 4 * n, 500, 140 + 4 * n, 512, 0.4, "teacher_owlv2")], persons=n)


def _frames(tmp_path, count=3):
    d = tmp_path / "frames" / "v"
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for k in range(count):
            w.writerow(["v.mp4", 6 * k, 100.0 * k, "base", 0.0])
            cv2.imwrite(str(d / f"{6 * k:06d}.jpg"), np.zeros((64, 96, 3), np.uint8))
    return d


def test_build_candidates_is_resumable_and_links_once_every_frame_is_done(tmp_path):
    frames, out, cfg = _frames(tmp_path), tmp_path / "gt", load_config()
    student = {12: [Box("face", 10, 10, 30, 34, 0.02)]}
    finder = _Finder()
    assert build_candidates(frames, out, finder, student, cfg, limit=2) == 2
    assert not (out / TRACKS).exists()
    assert build_candidates(frames, out, finder, student, cfg) == 1
    tracks = read_tracks(out / TRACKS)
    assert [(t.cls, t.frames) for t in tracks] == [("plate", (0, 6, 12)), ("face", (12,))]
    assert tracks[1].boxes[0].src == "student"
    assert read_persons(out) == {0: 0, 6: 1, 12: 2}
    assert build_candidates(frames, out, finder, student, cfg) == 0  # nothing left


def test_boxes_of_a_frame_a_crash_never_marked_done_are_dropped_on_resume(tmp_path):
    frames, out, cfg = _frames(tmp_path, count=2), tmp_path / "gt", load_config()
    build_candidates(frames, out, _Finder(), {}, cfg, limit=1)
    with (out / BOXES).open("a") as f:  # frame 6's boxes written, then the crash before processed.txt
        f.write('{"video": "v.mp4", "frame": 6, "pts_ms": 100.0, "cls": "plate", "x1": 1, "y1": 1, "x2": 9, '
                '"y2": 5, "score": 0.5, "src": "teacher_owlv2"}\n{"video": "v.mp4", "fra')
    build_candidates(frames, out, _Finder(), {}, cfg)
    assert (out / PROCESSED).read_text().split("\n")[:2] == ["0 0", "6 0"]
    assert [t.boxes[-1].x1 for t in read_tracks(out / TRACKS)] == [100]  # one box per frame, no stray 1..9
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_candidates.py`
Expected: collection error, `No module named 'vc_gt.candidates'`.

- [ ] **Step 3: Implement** — `py/vc_gt/candidates.py`

```python
"""Every plausible plate and face on every sampled frame, linked into tracks (spec 2026-10-22 §2 step 2): the teachers
at low thresholds, OWLv2 also on 2 x 2 tiles of the whole frame for small far plates, and the student's boxes."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vc_gt.config import GtConfig
from vc_gt.tracks import link_tracks, write_tracks
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box, Detection, read_frames_csv, read_jsonl, write_jsonl

BOXES = "boxes.jsonl"
PROCESSED = "processed.txt"  # "<frame> <persons>" per finished frame, written after its boxes
TRACKS = "tracks.jsonl"


@dataclass(frozen=True)
class FrameFinds:
    boxes: list[Box]  # faces and plates
    persons: int  # COCO person boxes: the face audit samples frames with people (spec §2 step 5)


class Finder(Protocol):
    def find(self, bgr: np.ndarray) -> FrameFinds: ...


def tile_windows(width: int, height: int, grid: int, overlap: float) -> list[Box]:
    """grid x grid windows over the frame, each grown by `overlap` of a tile into its neighbours, inside the frame."""
    tw, th = width / grid, height / grid
    return [Box("vehicle", max(0.0, c * tw - overlap * tw), max(0.0, r * th - overlap * th),
                min(float(width), (c + 1) * tw + overlap * tw), min(float(height), (r + 1) * th + overlap * th))
            for r in range(grid) for c in range(grid)]


def merge_duplicates(boxes: list[Box], merge_iou: float) -> list[Box]:
    """Per class, a box overlapping a higher-scoring kept box by merge_iou or more folds into it: the kept box stays,
    its src lists every detector that found it (sorted, '+'-joined)."""
    kept: list[tuple[Box, set[str]]] = []
    for b in sorted(boxes, key=lambda x: -x.score):
        home = next((srcs for k, srcs in kept if k.cls == b.cls and iou(k, b) >= merge_iou), None)
        if home is None:
            kept.append((b, {b.src}))
        else:
            home.add(b.src)
    return [replace(k, src="+".join(sorted(s for s in srcs if s))) for k, srcs in kept]


class TeacherFinder:
    """RetinaFace and YuNet for faces, COCO for vehicles and people, OWLv2 in every vehicle and on the tiles; cached
    weights only."""

    def __init__(self, cfg: GtConfig, device: str = "auto") -> None:
        from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env, required_model_files

        offline_model_env()
        missing = [p for p in required_model_files() if not p.exists()]
        if missing:
            raise FileNotFoundError("missing model files: " + ", ".join(str(p) for p in missing))
        import torch

        from vc_label.teachers.coco import CocoTeacher
        from vc_label.teachers.owlv2 import Owlv2PlateTeacher
        from vc_label.teachers.retinaface import RetinaFaceTeacher
        from vc_label.teachers.yunet import YuNetTeacher

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        dev = torch.device(device)
        self.cfg = cfg
        self.faces = [RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), dev, score_threshold=cfg.retinaface_score),
                      YuNetTeacher(str(YUNET_MODEL), score_threshold=cfg.yunet_score)]
        self.coco = CocoTeacher(dev, score_threshold=cfg.coco_score)
        self.owl = Owlv2PlateTeacher(dev, score_threshold=cfg.owlv2_score)

    def find(self, bgr: np.ndarray) -> FrameFinds:
        height, width = bgr.shape[:2]
        faces = [b for t in self.faces for b in t.detect(bgr)]
        aux = self.coco.detect(bgr)
        windows = [a for a in aux if a.cls == "vehicle"] + tile_windows(width, height, self.cfg.tile_grid,
                                                                         self.cfg.tile_overlap)
        return FrameFinds(faces + self.owl.detect(bgr, windows), sum(a.cls == "person" for a in aux))


def read_persons(out_dir: Path) -> dict[int, int]:
    """frame -> COCO person boxes, for every finished frame."""
    path = Path(out_dir) / PROCESSED
    if not path.exists():
        return {}
    return {int(f): int(p) for f, p in (line.split() for line in path.read_text().splitlines() if line.strip())}


def _drop_unfinished(out_dir: Path, done: set[int]) -> None:
    """Keeps only the boxes of finished frames; a torn last line (a crash mid-append) goes too."""
    path = out_dir / BOXES
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line)["frame"] in done:
                keep.append(line)
        except (ValueError, KeyError):
            continue
    path.write_text("".join(f"{line}\n" for line in keep), encoding="utf-8")


def build_candidates(frames_dir: Path, out_dir: Path, finder: Finder, student: dict[int, list[Box]], cfg: GtConfig,
                     limit: int | None = None) -> int:
    """Runs `finder` on every frame of frames.csv not finished yet (at most `limit` in this call), merges its boxes
    with the student's for that frame and appends them to boxes.jsonl, then marks the frame in processed.txt. Once
    every frame is finished, links them into tracks.jsonl. Returns the frames finished in this call."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = read_frames_csv(frames_dir)
    done = set(read_persons(out_dir))
    _drop_unfinished(out_dir, done)
    count = 0
    for ref in refs:
        if ref.frame in done:
            continue
        if limit is not None and count >= limit:
            break
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise FileNotFoundError(f"{ref.image} cannot be read")
        finds = finder.find(bgr)
        theirs = [replace(b, src="student") for b in student.get(ref.frame, []) if b.cls in LABEL_CLASSES]
        merged = merge_duplicates([b for b in finds.boxes if b.cls in LABEL_CLASSES] + theirs, cfg.merge_iou)
        write_jsonl(out_dir / BOXES, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in merged], append=True)
        with (out_dir / PROCESSED).open("a", encoding="utf-8") as f:
            f.write(f"{ref.frame} {finds.persons}\n")
        done.add(ref.frame)
        count += 1
    if refs and all(r.frame in done for r in refs):
        by_frame: dict[int, list[Box]] = defaultdict(list)
        for d in read_jsonl(out_dir / BOXES):
            by_frame[d.frame].append(d.box)
        write_tracks(out_dir / TRACKS, refs[0].video, link_tracks(by_frame, [r.frame for r in refs], cfg.track))
    return count
```

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_candidates.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add py/vc_gt/candidates.py py/tests/test_gt_candidates.py
git commit -m "feat: vc_gt finds every candidate plate and face on a segment and links them into tracks"
```

---

### Task 5: Crops, frame picking and unmasked frames

**Files:**
- Create: `py/vc_gt/crops.py`
- Test: `py/tests/test_gt_crops.py`

**Interfaces:**
- Consumes: `Track` (Task 2); `GtConfig` (Task 3); `vc_eval.metrics.covered_fraction`, `COVERED_FRACTION`.
- Produces: `Crop(path: Path, box: tuple[float, float, float, float])`; `crop_window(box, width, height, factor, min_px) -> tuple[int, int, int, int]`; `write_crop(bgr, box, path, factor, min_px) -> Crop`; `sharpness(bgr, box) -> float`; `pick_frames(track, sharp: dict[int, float], k: int) -> list[int]`; `unmasked_frames(track, masks_by_frame: dict[int, list[Box]]) -> list[int]`; `write_object_crops(track, images: dict[int, Path], out_dir, cfg) -> list[Crop]`; `write_frame_crop(track, frame, images, out_dir, cfg) -> Crop`; `append_crop_index(path, item_id, crops)`; `read_crop_index(path) -> dict[str, tuple[Crop, ...]]`; constant `CROPS_INDEX = "crops.jsonl"`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_crops.py`

```python
import cv2
import numpy as np

from vc_gt.config import load_config
from vc_gt.crops import (Crop, append_crop_index, crop_window, pick_frames, read_crop_index, sharpness,
                         unmasked_frames, write_crop, write_object_crops)
from vc_gt.tracks import Track
from vc_label.contracts import Box


def track(widths, frames=None):
    frames = tuple(frames or range(0, 6 * len(widths), 6))
    return Track(0, "plate", frames, tuple(Box("plate", 100, 100, 100 + w, 110) for w in widths))


def test_crop_window_is_a_square_around_the_box_inside_the_frame():
    assert crop_window(Box("plate", 100, 100, 140, 112), 1920, 1080, 3.0, 96) == (60, 46, 180, 166)
    assert crop_window(Box("plate", 0, 0, 10, 10), 1920, 1080, 3.0, 96) == (0, 0, 53, 53)  # min_px, clipped


def test_write_crop_gives_the_box_in_crop_pixels(tmp_path):
    crop = write_crop(np.zeros((1080, 1920, 3), np.uint8), Box("plate", 100, 100, 140, 112), tmp_path / "a.jpg", 3.0, 96)
    assert crop.box == (40, 54, 80, 66) and cv2.imread(str(crop.path)).shape[:2] == (120, 120)


def test_sharpness_ranks_detail_over_flat():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
    assert sharpness(noisy, Box("plate", 0, 0, 50, 50)) > sharpness(np.full((50, 50, 3), 90, np.uint8), Box("plate", 0, 0, 50, 50))


def test_pick_frames_takes_largest_sharpest_and_middle_in_time_order():
    t = track([20, 30, 50, 40, 10])  # frames 0, 6, 12, 18, 24; largest at 12, middle at 12
    assert pick_frames(t, {24: 9.0}, 3) == [12, 18, 24]  # largest 12 (also the middle), sharpest 24, next largest 18
    assert pick_frames(track([20]), {}, 3) == [0]


def test_unmasked_frames_are_those_under_80_percent_cover():
    t = track([40, 40, 40])
    masks = {0: [Box("plate", 90, 95, 150, 115)], 6: [Box("plate", 100, 100, 120, 110)]}  # 100 %, 50 %, none
    assert unmasked_frames(t, masks) == [6, 12]


def test_object_crops_and_the_crop_index_round_trip(tmp_path):
    images = {}
    for f in (0, 6, 12):
        images[f] = tmp_path / f"{f:06d}.jpg"
        cv2.imwrite(str(images[f]), np.zeros((300, 400, 3), np.uint8))
    crops = write_object_crops(track([20, 30, 40]), images, tmp_path / "crops", load_config())
    assert [c.path.name for c in crops] == ["o0_000000.jpg", "o0_000006.jpg", "o0_000012.jpg"]
    append_crop_index(tmp_path / "crops.jsonl", "o0", crops)
    assert read_crop_index(tmp_path / "crops.jsonl") == {"o0": tuple(crops)}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_crops.py`
Expected: collection error, `No module named 'vc_gt.crops'`.

- [ ] **Step 3: Implement** — `py/vc_gt/crops.py`

```python
"""Crops of tracked objects for the LLM and the review pages (spec 2026-10-22 §2 steps 4-5, §3): which frames show an
object best, and on which frames vc_redact left it unmasked."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_eval.metrics import COVERED_FRACTION, covered_fraction
from vc_gt.config import GtConfig
from vc_gt.tracks import Track
from vc_label.contracts import Box

CROPS_INDEX = "crops.jsonl"
SHARPNESS_CANDIDATES = 5  # sharpness is measured on this many of the largest boxes only (each costs a frame read)


@dataclass(frozen=True)
class Crop:
    path: Path
    box: tuple[float, float, float, float]  # the target in the crop's pixels


def crop_window(box: Box, width: int, height: int, factor: float, min_px: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): a square of factor x the box's long side (at least min_px) around its centre, inside the frame."""
    side = max(min_px, factor * max(box.width, box.height))
    cx, cy = box.center
    return (int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2))),
            int(min(width, round(cx + side / 2))), int(min(height, round(cy + side / 2))))


def write_crop(bgr: np.ndarray, box: Box, path: Path, factor: float, min_px: float) -> Crop:
    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = crop_window(box, width, height, factor, min_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr[y0:y1, x0:x1]):
        raise ValueError(f"cannot write {path}")
    return Crop(path, (box.x1 - x0, box.y1 - y0, box.x2 - x0, box.y2 - y0))


def sharpness(bgr: np.ndarray, box: Box) -> float:
    """Variance of the Laplacian inside the box: higher is sharper."""
    patch = bgr[max(0, int(box.y1)) : math.ceil(box.y2), max(0, int(box.x1)) : math.ceil(box.x2)]
    if patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pick_frames(track: Track, sharp: dict[int, float], k: int) -> list[int]:
    """Up to k of the track's frames, in time order: the largest box, the sharpest (by `sharp`, missing = 0), the
    middle frame, then the next largest until there are k."""
    by_area = sorted(track.frames, key=lambda f: (-track.box_at(f).area, f))
    sharpest = max(track.frames, key=lambda f: (sharp.get(f, 0.0), -f))
    picked: list[int] = []
    for f in [by_area[0], sharpest, track.frames[len(track.frames) // 2], *by_area]:
        if f not in picked:
            picked.append(f)
        if len(picked) == k:
            break
    return sorted(picked)


def unmasked_frames(track: Track, masks_by_frame: dict[int, list[Box]]) -> list[int]:
    """The track's frames where the masks cover less than COVERED_FRACTION of its box (vc_eval's covered test)."""
    return [f for f, b in zip(track.frames, track.boxes)
            if covered_fraction(b, masks_by_frame.get(f, [])) < COVERED_FRACTION]


def _read(images: dict[int, Path], frame: int) -> np.ndarray:
    bgr = cv2.imread(str(images[frame]))
    if bgr is None:
        raise FileNotFoundError(f"{images[frame]} cannot be read")
    return bgr


def write_object_crops(track: Track, images: dict[int, Path], out_dir: Path, cfg: GtConfig) -> list[Crop]:
    """Question A's crops: cfg.crops_per_object picked frames, out_dir/o<id>_<frame>.jpg."""
    largest = sorted(track.frames, key=lambda f: -track.box_at(f).area)[:SHARPNESS_CANDIDATES]
    sharp = {f: sharpness(_read(images, f), track.box_at(f)) for f in largest}
    return [write_crop(_read(images, f), track.box_at(f), Path(out_dir) / f"o{track.id}_{f:06d}.jpg",
                       cfg.crop_factor, cfg.min_crop_px)
            for f in pick_frames(track, sharp, cfg.crops_per_object)]


def write_frame_crop(track: Track, frame: int, images: dict[int, Path], out_dir: Path, cfg: GtConfig) -> Crop:
    """Question B's crop of one unmasked frame: out_dir/o<id>@<frame>.jpg."""
    return write_crop(_read(images, frame), track.box_at(frame), Path(out_dir) / f"o{track.id}@{frame:06d}.jpg",
                      cfg.crop_factor, cfg.min_crop_px)


def append_crop_index(path: Path, item_id: str, crops: list[Crop] | tuple[Crop, ...]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item_id, "crops": [{"path": str(c.path), "box": list(c.box)} for c in crops]}) + "\n")


def read_crop_index(path: Path) -> dict[str, tuple[Crop, ...]]:
    """item id -> its crops; a later line for an id replaces an earlier one."""
    path = Path(path)
    out: dict[str, tuple[Crop, ...]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = tuple(Crop(Path(c["path"]), tuple(float(v) for v in c["box"])) for c in r["crops"])
    return out
```

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_crops.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add py/vc_gt/crops.py py/tests/test_gt_crops.py
git commit -m "feat: vc_gt crops each object at its best frames and finds the frames vc_redact left unmasked"
```

---

### Task 6: Ask the LLM questions A and B

**Files:**
- Create: `py/vc_gt/judge.py`
- Test: `py/tests/test_gt_judge.py`

**Interfaces:**
- Consumes: `Crop`, `write_object_crops`, `write_frame_crop`, `unmasked_frames`, `append_crop_index`, `read_crop_index`, `CROPS_INDEX` (Task 5); `Track` (Task 2); `GtConfig` (Task 3); `vc_hardneg.judgments` (`Judgment`, `append_judgments`, `read_judgments`, `resolve`); `vc_hardneg.judges` (`CommandJudge`, `CLS_ZH`, `_fill`).
- Produces: `OBJECT_VERDICTS`, `FRAME_VERDICTS`, `REAL`, `OBJECTS = "objects.jsonl"`, `FRAMES = "frames.jsonl"`; `object_id(track_id) -> str`; `frame_id(track_id, frame) -> str`; `Item(id, cls, crops)`; `Runner = Callable[[Path, str, str], str | None]`; `parse_answers(out, allowed) -> dict[int, str]`; `grid_image(items, tile, out) -> Path`; `ask(items, prompt, allowed, runner, judge, judgments, work_dir, batch, tile, parallel=1) -> Counter`; `judge_objects(gt_dir, tracks, images, cfg, runner, judge) -> Counter`; `judge_frames(gt_dir, tracks, images, masks_by_frame, cfg, runner, judge) -> Counter`; `make_runner(cfg) -> tuple[Runner, str]`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_judge.py`

```python
import cv2
import numpy as np
import pytest

from vc_gt.config import load_config
from vc_gt.crops import Crop
from vc_gt.judge import (FRAMES, OBJECT_VERDICTS, OBJECTS, Item, ask, frame_id, grid_image, judge_frames,
                         judge_objects, make_runner, object_id, parse_answers)
from vc_gt.tracks import Track
from vc_hardneg.judgments import read_judgments, resolve
from vc_label.contracts import Box


def _crop(tmp_path, name):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((120, 120, 3), 128, np.uint8))
    return Crop(p, (40, 54, 80, 66))


def test_parse_answers_takes_the_last_json_array_and_only_allowed_verdicts():
    out = 'thinking [1, 2]\n[{"n": 1, "verdict": "readable"}, {"n": 2, "verdict": "maybe"}, {"n": "3", "verdict": "unreadable"}]'
    assert parse_answers(out, ("readable", "unreadable", "unsure")) == {1: "readable"}
    assert parse_answers(None, ("readable",)) == {}


def test_grid_image_puts_an_items_crops_side_by_side(tmp_path):
    items = [Item(f"o{k}", "plate", tuple(_crop(tmp_path, f"{k}_{j}.jpg") for j in range(3))) for k in range(3)]
    out = grid_image(items, 100, tmp_path / "g.png")
    assert cv2.imread(str(out)).shape[:2] == (200, 600)  # 2 cells per row of 3 x 100 px, 2 rows


class _Runner:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, image, prompt, cls):
        self.calls.append((image.name, prompt, cls))
        return self.reply


def test_ask_batches_by_class_appends_verdicts_and_skips_what_is_judged(tmp_path):
    items = [Item("o0", "plate", (_crop(tmp_path, "a.jpg"),)), Item("o1", "plate", (_crop(tmp_path, "b.jpg"),)),
             Item("o2", "face", (_crop(tmp_path, "c.jpg"),))]
    runner = _Runner('[{"n": 1, "verdict": "not_real"}]')
    counts = ask(items, "共 {count} 组{cls_zh}", OBJECT_VERDICTS, runner, "llm:fake", tmp_path / "j.jsonl",
                 tmp_path / "llm", batch=4, tile=64)
    final = resolve(read_judgments(tmp_path / "j.jsonl"))
    assert {k: v.verdict for k, v in final.items()} == {"o0": "not_real", "o1": "unsure", "o2": "not_real"}
    assert counts == {"not_real": 2, "unsure": 1}
    assert sorted(c[1] for c in runner.calls) == ["共 1 组人脸", "共 2 组车牌"]
    assert ask(items, "{count}{cls_zh}", OBJECT_VERDICTS, runner, "llm:fake", tmp_path / "j.jsonl",
               tmp_path / "llm", batch=4, tile=64) == {}
    assert len(runner.calls) == 2


def test_a_command_that_does_not_answer_leaves_its_items_unsure(tmp_path):
    ask([Item("o0", "plate", (_crop(tmp_path, "a.jpg"),))], "{count}{cls_zh}", OBJECT_VERDICTS, _Runner(None),
        "llm:fake", tmp_path / "j.jsonl", tmp_path / "llm", batch=4, tile=64)
    j = read_judgments(tmp_path / "j.jsonl")[0]
    assert (j.verdict, j.detail) == ("unsure", "no answer")


def _segment(tmp_path):
    images = {}
    for f in (0, 6, 12):
        images[f] = tmp_path / f"{f:06d}.jpg"
        cv2.imwrite(str(images[f]), np.zeros((300, 400, 3), np.uint8))
    boxes = tuple(Box("plate", 100, 100, 140, 112) for _ in range(3))
    return images, [Track(0, "plate", (0, 6, 12), boxes), Track(1, "plate", (0, 6, 12), boxes)]


def test_judge_objects_then_frames_asks_only_about_unmasked_frames_of_real_objects(tmp_path):
    images, tracks = _segment(tmp_path)
    gt, cfg = tmp_path / "gt", load_config()
    gt.mkdir()
    judge_objects(gt, tracks, images, cfg, _Runner('[{"n": 1, "verdict": "real_readable"}, {"n": 2, "verdict": "not_real"}]'),
                  "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / OBJECTS)).items()} == {
        object_id(0): "real_readable", object_id(1): "not_real"}
    masks = {0: [Box("plate", 90, 90, 150, 120)]}  # frame 0 covered, 6 and 12 not
    runner = _Runner('[{"n": 1, "verdict": "readable"}, {"n": 2, "verdict": "unreadable"}]')
    judge_frames(gt, tracks, images, masks, cfg, runner, "llm:fake")
    assert {k: v.verdict for k, v in resolve(read_judgments(gt / FRAMES)).items()} == {
        frame_id(0, 6): "readable", frame_id(0, 12): "unreadable"}
    assert (gt / "crops" / "frames" / "o0@000006.jpg").exists()


def test_make_runner_refuses_without_the_llm_switch(monkeypatch):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    with pytest.raises(PermissionError):
        make_runner(load_config())
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_judge.py`
Expected: collection error, `No module named 'vc_gt.judge'`.

- [ ] **Step 3: Implement** — `py/vc_gt/judge.py`

```python
"""The LLM's two questions (spec 2026-10-22 §2 step 4): A, per object, real or not and readable at its best; B, per
frame vc_redact left a real object unmasked, readable there or not. Verdicts go to objects.jsonl and frames.jsonl in
vc_hardneg's judgment format, so people's verdicts (judge "human") override them."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_gt.config import GtConfig
from vc_gt.crops import (CROPS_INDEX, Crop, append_crop_index, read_crop_index, unmasked_frames, write_frame_crop,
                         write_object_crops)
from vc_gt.tracks import Track
from vc_hardneg.judges import CLS_ZH, CommandJudge, _fill
from vc_hardneg.judgments import Judgment, append_judgments, read_judgments, resolve
from vc_label.contracts import Box

OBJECT_VERDICTS = ("real_readable", "real_unreadable", "not_real", "unsure")
FRAME_VERDICTS = ("readable", "unreadable", "unsure")
REAL = ("real_readable", "real_unreadable")
OBJECTS = "objects.jsonl"
FRAMES = "frames.jsonl"
Runner = Callable[[Path, str, str], "str | None"]  # (grid image, prompt, class) -> the command's stdout


def object_id(track_id: int) -> str:
    return f"o{track_id}"


def frame_id(track_id: int, frame: int) -> str:
    return f"o{track_id}@{frame}"


@dataclass(frozen=True)
class Item:
    id: str
    cls: str
    crops: tuple[Crop, ...]


def parse_answers(out: str | None, allowed: tuple[str, ...]) -> dict[int, str]:
    """{n: verdict} from the last JSON array of objects in the output; rows with another verdict are left out."""
    for text in reversed(re.findall(r"\[[^\[\]]*\]", out or "")):
        try:
            rows = json.loads(text)
        except ValueError:
            continue
        if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
            return {r["n"]: r["verdict"] for r in rows if isinstance(r.get("n"), int) and r.get("verdict") in allowed}
    return {}


def grid_image(items: list[Item], tile: int, out: Path) -> Path:
    """Every item in one cell, its crops side by side, each fitted into tile x tile with the target in red and the
    item's number (1, 2, ...) in yellow at the cell's top left; two cells per row when items have several crops,
    else four."""
    per = max(len(it.crops) for it in items)
    cols = min(len(items), 2 if per > 1 else 4)
    rows = (len(items) + cols - 1) // cols
    cell = per * tile
    grid = np.full((rows * tile, cols * cell, 3), 40, np.uint8)
    for k, it in enumerate(items):
        r, q = divmod(k, cols)
        for j, crop in enumerate(it.crops):
            img = cv2.imread(str(crop.path))
            if img is None:
                raise FileNotFoundError(f"{crop.path} cannot be read")
            s = (tile - 4) / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            x1, y1, x2, y2 = crop.box
            cv2.rectangle(img, (round(x1 * s), round(y1 * s)), (round(x2 * s), round(y2 * s)), (0, 0, 255), 2)
            oy, ox = r * tile + 2, q * cell + j * tile + 2
            grid[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
        for color, width in (((0, 0, 0), 5), ((0, 255, 255), 2)):
            cv2.putText(grid, str(k + 1), (q * cell + 8, r * tile + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, width)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), grid):
        raise ValueError(f"cannot write {out}")
    return out


def ask(items: list[Item], prompt: str, allowed: tuple[str, ...], runner: Runner, judge: str, judgments: Path,
        work_dir: Path, batch: int, tile: int, parallel: int = 1) -> Counter:
    """Asks about every item without a judgment in `judgments` yet: batches of up to `batch` items of one class on a
    numbered grid (written to work_dir), `parallel` calls at once. Each batch's verdicts are appended as it comes, so an
    interrupted run keeps them; an item the answer leaves out is "unsure". Returns counts per verdict."""
    known = resolve(read_judgments(judgments))
    todo = [it for it in items if it.id not in known]
    batches = [same[i : i + batch] for cls in sorted({it.cls for it in todo})
               for same in [[it for it in todo if it.cls == cls]] for i in range(0, len(same), batch)]

    def one(b: list[Item]) -> list[Judgment]:
        cls = b[0].cls
        image = grid_image(b, tile, Path(work_dir) / f"batch_{b[0].id}.png")
        out = runner(image, _fill(prompt, {"{count}": str(len(b)), "{cls_zh}": CLS_ZH[cls], "{cls}": cls}), cls)
        answers = parse_answers(out, allowed)
        detail = f"grid of {len(b)}" if answers else ("no answer" if out is None else "no verdicts in the output")
        return [Judgment(it.id, answers.get(k + 1, "unsure"), judge, None, detail) for k, it in enumerate(b)]

    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for got in pool.map(one, batches):
            append_judgments(judgments, got, verdicts=allowed)
            counts.update(j.verdict for j in got)
    return counts


def _cached(index: dict[str, tuple[Crop, ...]], gt_dir: Path, item_id: str, make) -> tuple[Crop, ...]:
    if item_id in index:
        return index[item_id]
    crops = tuple(make())
    append_crop_index(gt_dir / CROPS_INDEX, item_id, crops)
    return crops


def judge_objects(gt_dir: Path, tracks: list[Track], images: dict[int, Path], cfg: GtConfig, runner: Runner,
                  judge: str) -> Counter:
    """Question A about every track not judged yet."""
    gt_dir = Path(gt_dir)
    known = resolve(read_judgments(gt_dir / OBJECTS))
    index = read_crop_index(gt_dir / CROPS_INDEX)
    items = [Item(object_id(t.id), t.cls,
                  _cached(index, gt_dir, object_id(t.id),
                          lambda t=t: write_object_crops(t, images, gt_dir / "crops" / "objects", cfg)))
             for t in tracks if object_id(t.id) not in known]
    return ask(items, cfg.llm.object_prompt, OBJECT_VERDICTS, runner, judge, gt_dir / OBJECTS, gt_dir / "llm",
               cfg.llm.batch, cfg.llm.tile, cfg.llm.parallel)


def judge_frames(gt_dir: Path, tracks: list[Track], images: dict[int, Path], masks_by_frame: dict[int, list[Box]],
                 cfg: GtConfig, runner: Runner, judge: str) -> Counter:
    """Question B about every unmasked frame of every track judged real (by the latest verdict) not judged yet."""
    gt_dir = Path(gt_dir)
    objects = resolve(read_judgments(gt_dir / OBJECTS))
    known = resolve(read_judgments(gt_dir / FRAMES))
    index = read_crop_index(gt_dir / CROPS_INDEX)
    items = []
    for t in tracks:
        j = objects.get(object_id(t.id))
        if j is None or j.verdict not in REAL:
            continue
        for f in unmasked_frames(t, masks_by_frame):
            fid = frame_id(t.id, f)
            if fid not in known:
                crops = _cached(index, gt_dir, fid,
                                lambda t=t, f=f: [write_frame_crop(t, f, images, gt_dir / "crops" / "frames", cfg)])
                items.append(Item(fid, t.cls, crops))
    return ask(items, cfg.llm.frame_prompt, FRAME_VERDICTS, runner, judge, gt_dir / FRAMES, gt_dir / "llm",
               cfg.llm.batch, cfg.llm.tile, cfg.llm.parallel)


def make_runner(cfg: GtConfig) -> tuple[Runner, str]:
    """(the runner, its judge name "llm:<name>"); raises PermissionError without VC_HARDNEG_ALLOW_LLM=1."""
    command = CommandJudge(cfg.llm.command_config())
    return (lambda image, prompt, cls: command.run(image, prompt, cls)[0]), command.name
```

Addendum (2026-10-23, spec §8): `judge_objects` only asks about tracks whose box reaches `vc_eval.metrics.must_mask` size on at least one frame (frame height read from the first image); smaller tracks get no question A and therefore never count. Add `test_judge_objects_skips_tracks_that_never_reach_must_mask_size` (a 10 px wide plate track gets no judgment).

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_judge.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add py/vc_gt/judge.py py/tests/test_gt_judge.py
git commit -m "feat: vc_gt asks the LLM whether each object is real and readable, and each unmasked frame readable"
```

---

### Task 7: Review pages, audit sample and imports

**Files:**
- Create: `py/vc_gt/audit.py`
- Test: `py/tests/test_gt_audit.py`

**Interfaces:**
- Consumes: `Crop` (Task 5); `OBJECT_VERDICTS`, `FRAME_VERDICTS` (Task 6); `vc_hardneg.judgments` (`HUMAN`, `Judgment`).
- Produces: `PageItem(id, crops, note)`; `OBJECT_CHOICES`, `FRAME_CHOICES`; `h1_object_ids(objects) -> list[str]`; `h1_frame_ids(frames, unmasked_ids) -> list[str]`; `h2_sample(model_ids, n, seed) -> list[str]`; `audit_sample(frames, persons, per_class, seed) -> dict[str, list[int]]`; `write_choice_page(items, out_dir, title, choices) -> Path`; `import_choices(export_file, known_ids, allowed) -> list[Judgment]`; `write_annotate_page(frames, out_dir, title) -> Path`; `AuditBox(frame: int, box: Box)`; `import_annotations(export_file, keys) -> list[dict]`; `write_audit(path, rows)`; `read_audit(path) -> tuple[dict[str, set[int]], list[AuditBox]]`; constants `H2_SAMPLE = "h2_sample.json"`, `AUDIT_FRAMES = "audit_frames.json"`, `AUDIT = "audit.jsonl"`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_audit.py`

```python
import json

import cv2
import numpy as np
import pytest

from vc_gt.audit import (FRAME_CHOICES, OBJECT_CHOICES, PageItem, audit_sample, h1_frame_ids, h1_object_ids,
                         h2_sample, import_annotations, import_choices, read_audit, write_annotate_page,
                         write_audit, write_choice_page)
from vc_gt.crops import Crop
from vc_hardneg.judgments import Judgment


def test_h1_takes_unsure_objects_and_readable_or_unsure_unmasked_frames_people_did_not_judge():
    objects = {"o0": Judgment("o0", "unsure", "llm:x"), "o1": Judgment("o1", "not_real", "llm:x"),
               "o2": Judgment("o2", "unsure", "human")}
    frames = {"o0@6": Judgment("o0@6", "readable", "llm:x"), "o0@12": Judgment("o0@12", "unreadable", "llm:x"),
              "o0@18": Judgment("o0@18", "unsure", "llm:x"), "o0@24": Judgment("o0@24", "readable", "llm:x")}
    assert h1_object_ids(objects) == ["o0"]
    assert h1_frame_ids(frames, {"o0@6", "o0@12", "o0@18"}) == ["o0@18", "o0@6"]  # o0@24 is masked now


def test_h2_and_audit_samples_are_seeded_and_capped():
    ids = [f"o{k}" for k in range(100)]
    assert h2_sample(ids, 10, 1) == h2_sample(list(reversed(ids)), 10, 1) and len(h2_sample(ids, 10, 1)) == 10
    assert h2_sample(ids[:3], 10, 1) == sorted(ids[:3])
    s = audit_sample([0, 6, 12, 18], {0: 0, 6: 2, 12: 0, 18: 1}, 3, 7)
    assert len(s["plate"]) == 3 and s["face"] == [6, 18]  # face frames only where COCO saw people


def _crop(tmp_path, name):
    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros((120, 120, 3), np.uint8))
    return Crop(p, (40, 54, 80, 66))


def test_choice_page_copies_crops_and_holds_the_choices(tmp_path):
    page = write_choice_page([PageItem("o3", (_crop(tmp_path, "a.jpg"),), "plate")], tmp_path / "page", "H1 车牌",
                             OBJECT_CHOICES)
    text = page.read_text(encoding="utf-8")
    assert (tmp_path / "page" / "img" / "o3_0.jpg").exists()
    assert '"o3"' in text and "real_readable" in text and "img/o3_0.jpg" in text and "H1 车牌" in text


def test_import_choices_refuses_unknown_ids_and_verdicts(tmp_path):
    f = tmp_path / "e.json"
    f.write_text(json.dumps([{"id": "o1", "verdict": "readable"}]))
    assert import_choices(f, {"o1"}, ("readable", "unreadable"))[0] == Judgment("o1", "readable", "human")
    with pytest.raises(ValueError, match="unknown id"):
        import_choices(f, {"o2"}, ("readable",))
    f.write_text(json.dumps([{"id": "o1", "verdict": "real_readable"}]))
    with pytest.raises(ValueError, match="verdict"):
        import_choices(f, {"o1"}, ("readable", "unreadable"))


def test_annotations_keep_finished_frames_only_and_round_trip(tmp_path):
    img = tmp_path / "000006.jpg"
    cv2.imwrite(str(img), np.zeros((100, 200, 3), np.uint8))
    page = write_annotate_page([("plate:6", img, "plate")], tmp_path / "ann", "H3")
    assert "plate:6" in page.read_text(encoding="utf-8") and (tmp_path / "ann" / "img" / "plate_6.jpg").exists()
    export = tmp_path / "audit_export.json"
    export.write_text(json.dumps([{"key": "plate:6", "done": True, "boxes": [[10, 20, 50, 32]]},
                                  {"key": "face:12", "done": False, "boxes": [[1, 1, 5, 5]]}]))
    rows = import_annotations(export, {"plate:6": ("plate", 6), "face:12": ("face", 12)})
    assert rows == [{"key": "plate:6", "cls": "plate", "frame": 6, "boxes": [[10, 20, 50, 32]]}]
    write_audit(tmp_path / "audit.jsonl", rows)
    done, boxes = read_audit(tmp_path / "audit.jsonl")
    assert done == {"plate": {6}, "face": set()} and boxes[0].frame == 6 and boxes[0].box.x2 == 50
    with pytest.raises(ValueError, match="unknown key"):
        import_annotations(export, {"plate:6": ("plate", 6)})
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_audit.py`
Expected: collection error, `No module named 'vc_gt.audit'`.

- [ ] **Step 3: Implement** — `py/vc_gt/audit.py`

```python
"""People's part (spec 2026-10-22 §2 step 5): H1 every leak and every unsure verdict, H2 a seeded sample of model
verdicts, H3 whole frames boxed by hand. Local pages with no external resources; results leave them as downloaded
JSON files that the import functions read back."""

from __future__ import annotations

import html
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from vc_gt.crops import Crop
from vc_hardneg.judgments import HUMAN, Judgment
from vc_label.contracts import Box

H2_SAMPLE = "h2_sample.json"
AUDIT_FRAMES = "audit_frames.json"
AUDIT = "audit.jsonl"
OBJECT_CHOICES = [("real_readable", "真的，能认出"), ("real_unreadable", "真的，认不出"), ("not_real", "不是")]
FRAME_CHOICES = [("readable", "能认出"), ("unreadable", "认不出")]


@dataclass(frozen=True)
class PageItem:
    id: str
    crops: tuple[Crop, ...]
    note: str


@dataclass(frozen=True)
class AuditBox:
    frame: int
    box: Box


def h1_object_ids(objects: dict[str, Judgment]) -> list[str]:
    """Objects whose final verdict is a model's "unsure"."""
    return sorted(i for i, j in objects.items() if j.verdict == "unsure" and j.judge != HUMAN)


def h1_frame_ids(frames: dict[str, Judgment], unmasked_ids: set[str]) -> list[str]:
    """Frames still unmasked whose final verdict is a model's "readable" (a leak) or "unsure"."""
    return sorted(i for i in unmasked_ids if i in frames and frames[i].judge != HUMAN
                  and frames[i].verdict in ("readable", "unsure"))


def h2_sample(model_ids: list[str], n: int, seed: int) -> list[str]:
    """n of the ids (all when fewer), the same for the same ids and seed whatever their order."""
    ids = sorted(model_ids)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def audit_sample(frames: list[int], persons: dict[int, int], per_class: int, seed: int) -> dict[str, list[int]]:
    """H3 frames: for plates any frame, for faces only frames where COCO saw a person."""
    rng = random.Random(seed)
    ordered = sorted(frames)
    people = [f for f in ordered if persons.get(f, 0) > 0]
    return {"plate": sorted(rng.sample(ordered, min(per_class, len(ordered)))),
            "face": sorted(rng.sample(people, min(per_class, len(people))))}


_STYLE = """:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--on:#1f6feb;--line:#dedbd3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#22221f;--ink:#ecebe6;--muted:#9a988f;--line:#3a3934}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,"PingFang SC",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.meta{color:var(--muted);font-size:12px;margin:6px 0}
button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.on{border-color:var(--on);color:var(--on);font-weight:600}
button:focus-visible{outline:2px solid var(--on);outline-offset:2px}
.pic{position:relative}.pic img{display:block;width:100%;height:auto;image-rendering:pixelated}
.box{position:absolute;border:2px solid #ff3b30;pointer-events:none}"""

_CHOICE_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>__STYLE__
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}.card.done{border-color:var(--on)}
.pics{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.row{display:flex;gap:6px;flex-wrap:wrap}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span><button id="export">导出结果（__EXPORT__）</button>
<span class="meta">红框是目标。"能认出"指暂停并放大后：车牌能读出大部分字符，人脸认识的人能认出是谁。</span></header>
<main id="grid"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_gt:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const progress = () => { document.getElementById('progress').textContent = Object.keys(state).length + ' / ' + data.items.length; };
const grid = document.getElementById('grid');
for (const it of data.items) {
  const card = document.createElement('div'); card.className = 'card';
  const pics = document.createElement('div'); pics.className = 'pics';
  for (const p of it.crops) {
    const pic = document.createElement('div'); pic.className = 'pic';
    const img = document.createElement('img'); img.loading = 'lazy'; img.alt = ''; img.src = p.img;
    const b = document.createElement('div'); b.className = 'box';
    b.style.left = p.box[0] + '%'; b.style.top = p.box[1] + '%';
    b.style.width = (p.box[2] - p.box[0]) + '%'; b.style.height = (p.box[3] - p.box[1]) + '%';
    pic.append(img, b); pics.append(pic);
  }
  const meta = document.createElement('div'); meta.className = 'meta'; meta.textContent = it.id + ' ' + it.note;
  const row = document.createElement('div'); row.className = 'row';
  const mark = () => { for (const b of row.children) b.classList.toggle('on', state[it.id] === b.dataset.value);
                       card.classList.toggle('done', !!state[it.id]); };
  for (const [value, label] of data.choices) {
    const btn = document.createElement('button'); btn.textContent = label; btn.dataset.value = value;
    btn.onclick = () => { state[it.id] = value; mark(); save(); progress(); };
    row.append(btn);
  }
  card.append(pics, meta, row); grid.append(card); mark();
}
progress();
document.getElementById('export').onclick = () => {
  const out = Object.entries(state).map(([id, verdict]) => ({id, verdict}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = data.export; a.click();
};
</script></body></html>
"""

_ANNOTATE_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>__STYLE__
main{padding:16px;display:flex;flex-direction:column;gap:24px}
.frame{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}.frame.done{border-color:var(--on)}
.pic{cursor:crosshair;user-select:none;touch-action:none}.row{display:flex;gap:6px;margin-top:6px;align-items:center}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span><button id="export">导出结果（audit_export.json）</button>
<span class="meta">在每帧上拖拽框出每一个暂停放大后能认出的目标（只框标题里那一类），框完点"本帧标完"；没有就直接点"本帧标完"。</span></header>
<main id="list"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_gt:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const progress = () => { document.getElementById('progress').textContent =
  Object.values(state).filter(s => s.done).length + ' / ' + data.frames.length + ' 帧标完'; };
const list = document.getElementById('list');
for (const fr of data.frames) {
  const s = state[fr.key] || (state[fr.key] = {done: false, boxes: []});
  const card = document.createElement('div'); card.className = 'frame';
  const pic = document.createElement('div'); pic.className = 'pic';
  const img = document.createElement('img'); img.loading = 'lazy'; img.alt = ''; img.src = fr.img;
  pic.append(img);
  const meta = document.createElement('div'); meta.className = 'meta';
  const draw = () => {
    for (const el of pic.querySelectorAll('.box')) el.remove();
    for (const [x1, y1, x2, y2] of s.boxes) {
      const b = document.createElement('div'); b.className = 'box';
      b.style.left = 100 * x1 / fr.w + '%'; b.style.top = 100 * y1 / fr.h + '%';
      b.style.width = 100 * (x2 - x1) / fr.w + '%'; b.style.height = 100 * (y2 - y1) / fr.h + '%';
      pic.append(b);
    }
    meta.textContent = fr.key + '：框出 ' + s.boxes.length + ' 个' + fr.cls_zh;
    card.classList.toggle('done', s.done); doneBtn.classList.toggle('on', s.done);
  };
  const at = (e) => { const r = img.getBoundingClientRect();
    return [Math.max(0, Math.min(fr.w, (e.clientX - r.left) / r.width * fr.w)),
            Math.max(0, Math.min(fr.h, (e.clientY - r.top) / r.height * fr.h))]; };
  let start = null;
  pic.onpointerdown = (e) => { start = at(e); pic.setPointerCapture(e.pointerId); };
  pic.onpointerup = (e) => { if (!start) return; const end = at(e);
    const b = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])].map(Math.round);
    start = null; if (b[2] - b[0] >= 3 && b[3] - b[1] >= 3) { s.boxes.push(b); save(); draw(); } };
  const row = document.createElement('div'); row.className = 'row';
  const undo = document.createElement('button'); undo.textContent = '撤销上一个框';
  undo.onclick = () => { s.boxes.pop(); save(); draw(); };
  const doneBtn = document.createElement('button'); doneBtn.textContent = '本帧标完';
  doneBtn.onclick = () => { s.done = !s.done; save(); draw(); progress(); };
  row.append(undo, doneBtn);
  card.append(pic, meta, row); list.append(card); draw();
}
progress();
document.getElementById('export').onclick = () => {
  const out = data.frames.map(fr => ({key: fr.key, done: state[fr.key].done, boxes: state[fr.key].boxes}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'audit_export.json'; a.click();
};
</script></body></html>
"""


def _page(template: str, title: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    return (template.replace("__STYLE__", _STYLE).replace("__TITLE__", html.escape(title))
            .replace("__EXPORT__", html.escape(data.get("export", ""))).replace("__DATA__", payload))


def write_choice_page(items: list[PageItem], out_dir: Path, title: str, choices: list[tuple[str, str]],
                      export: str = "human_judgments.json") -> Path:
    """out_dir/index.html: every item's crops (copied to out_dir/img) with the target boxed, one button per choice."""
    import cv2

    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        crops = []
        for j, c in enumerate(it.crops):
            name = f"{it.id.replace('@', '_')}_{j}.jpg"
            shutil.copyfile(c.path, out_dir / "img" / name)
            h, w = cv2.imread(str(c.path)).shape[:2]
            x1, y1, x2, y2 = c.box
            crops.append({"img": f"img/{name}", "box": [100 * x1 / w, 100 * y1 / h, 100 * x2 / w, 100 * y2 / h]})
        rows.append({"id": it.id, "note": it.note, "crops": crops})
    page = out_dir / "index.html"
    page.write_text(_page(_CHOICE_PAGE, title, {"title": title, "export": export, "choices": choices, "items": rows}),
                    encoding="utf-8")
    return page


def import_choices(export_file: Path, known_ids: set[str], allowed: tuple[str, ...]) -> list[Judgment]:
    """A choice page's export as human judgments. Raises ValueError for an unknown id or a verdict outside `allowed`."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{export_file} is not a list of {{id, verdict}}")
    out = []
    for r in rows:
        if r.get("id") not in known_ids:
            raise ValueError(f"{export_file}: unknown id {r.get('id')!r}")
        if r.get("verdict") not in allowed:
            raise ValueError(f"{export_file}: verdict of {r['id']} must be one of {allowed}")
        out.append(Judgment(r["id"], r["verdict"], HUMAN))
    return out


def write_annotate_page(frames: list[tuple[str, Path, str]], out_dir: Path, title: str) -> Path:
    """out_dir/index.html for H3: (key, frame image, class to box) per frame, the images copied to out_dir/img."""
    from vc_label.review import image_size

    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for key, image, cls in frames:
        name = key.replace(":", "_") + ".jpg"
        shutil.copyfile(image, out_dir / "img" / name)
        w, h = image_size(image)
        rows.append({"key": key, "img": f"img/{name}", "w": w, "h": h, "cls": cls,
                     "cls_zh": {"plate": "车牌", "face": "人脸"}[cls]})
    page = out_dir / "index.html"
    page.write_text(_page(_ANNOTATE_PAGE, title, {"title": title, "frames": rows}), encoding="utf-8")
    return page


def import_annotations(export_file: Path, keys: dict[str, tuple[str, int]]) -> list[dict]:
    """The finished frames of an annotate page's export, as audit rows {key, cls, frame, boxes}. Raises ValueError for
    a key the sample does not hold."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r.get("key") not in keys:
            raise ValueError(f"{export_file}: unknown key {r.get('key')!r}")
        if r.get("done"):
            cls, frame = keys[r["key"]]
            out.append({"key": r["key"], "cls": cls, "frame": frame, "boxes": [[float(v) for v in b] for b in r["boxes"]]})
    return out


def write_audit(path: Path, rows: list[dict]) -> None:
    """Replaces audit.jsonl with these rows (one finished frame each)."""
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_audit(path: Path) -> tuple[dict[str, set[int]], list[AuditBox]]:
    """({class: finished frames}, every box people drew on them)."""
    done: dict[str, set[int]] = {"plate": set(), "face": set()}
    boxes: list[AuditBox] = []
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["cls"]].add(int(r["frame"]))
                boxes += [AuditBox(int(r["frame"]), Box(r["cls"], *b)) for b in r["boxes"]]
    return done, boxes
```

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_audit.py`
Expected: 5 passed.

- [ ] **Step 5: Open both pages once in the browser pane** (local file, no network) to check they render and work: build a page from the test fixtures into the scratchpad with a short Python snippet, open it with `mcp__Claude_Browser__preview_start` (`url: file://...`), click a choice, draw one box, press export. Fix anything broken, rerun Step 4.

- [ ] **Step 6: Commit**

```bash
git add py/vc_gt/audit.py py/tests/test_gt_audit.py
git commit -m "feat: vc_gt review and frame-annotation pages, H1/H2/H3 samples and their imports"
```

---

### Task 8: `vc_eval objects`: per-object leaks, per-frame recall, intervals, report

**Files:**
- Create: `py/vc_eval/objects.py`
- Modify: `py/vc_eval/cli.py` (parser, handler, module docstring)
- Test: `py/tests/test_eval_objects.py`

**Interfaces:**
- Consumes: `Track`, `read_tracks` (Task 2); `unmasked_frames`, `write_crop` (Task 5); `object_id`, `frame_id`, `REAL`, `OBJECTS`, `FRAMES` (Task 6); `AuditBox`, `read_audit`, `AUDIT`, `H2_SAMPLE` (Task 7); `vc_eval.metrics` (`Rate`, `must_mask`, `covered_fraction`, `COVERED_FRACTION`); `vc_eval.records.read_masks`; `vc_hardneg.judgments`.
- Produces: `clopper_pearson(k, n, alpha=0.05) -> tuple[float, float]`; `Leak(track, frames)`; `ClassResult(cls, readable, leaks, frame_recall, missed_by_all)` with `no_leak -> Rate` and `passed -> bool`; `evaluate_objects(tracks, objects, frames, masks_by_frame, frame_height, audit=None) -> dict[str, ClassResult]`; `model_error_rate(judgments, sample_ids) -> Rate`; `write_report(results, error_rates, out_dir, tracks, images, masks_by_frame) -> Path`.

- [ ] **Step 1: Write the failing tests** — `py/tests/test_eval_objects.py`

```python
import pytest

from vc_eval.metrics import Rate
from vc_eval.objects import ClassResult, Leak, clopper_pearson, evaluate_objects, model_error_rate
from vc_gt.audit import AuditBox
from vc_gt.tracks import Track
from vc_hardneg.judgments import Judgment
from vc_label.contracts import Box


def test_clopper_pearson_matches_known_values():
    assert clopper_pearson(0, 3000)[0] == 0.0
    assert clopper_pearson(0, 3000)[1] == pytest.approx(0.0012289, abs=2e-6)  # 1 - 0.025 ** (1 / 3000)
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == (pytest.approx(0.187086, abs=1e-5), pytest.approx(0.812914, abs=1e-5))
    assert clopper_pearson(10, 10)[1] == 1.0 and clopper_pearson(0, 0) == (0.0, 1.0)


def plate(x, w=40):
    return Box("plate", x, 500, x + w, 512)


def test_leaks_readable_objects_and_frame_recall():
    tracks = [Track(0, "plate", (0, 6, 12), (plate(100), plate(100), plate(100))),  # real, frame 6 unmasked, readable
              Track(1, "plate", (0, 6), (plate(300), plate(300))),  # real_unreadable, all masked
              Track(2, "plate", (0,), (plate(600),)),  # not real
              Track(3, "plate", (0,), (plate(800, w=10),))]  # real_unreadable, too small to demand, unmasked
    objects = {"o0": "real_readable", "o1": "real_unreadable", "o2": "not_real", "o3": "real_unreadable"}
    frames = {"o0@6": "readable", "o3@0": "unreadable"}
    covered = lambda x: [Box("plate", x - 5, 495, x + 45, 517)]
    masks = {0: covered(100) + covered(300), 6: covered(300), 12: covered(100)}
    r = evaluate_objects(tracks, objects, frames, masks, 1080)["plate"]
    assert r.readable == 1 and r.leaks == (Leak(0, (6,)),)
    assert (r.no_leak.hits, r.no_leak.total) == (0, 1)
    assert (r.frame_recall.hits, r.frame_recall.total) == (4, 5)  # tracks 0 and 1; track 3 is under 24 px wide
    assert r.missed_by_all is None and not r.passed


def test_missed_by_all_counts_audit_boxes_no_track_holds():
    tracks = [Track(0, "plate", (6,), (plate(100),))]
    audit = [AuditBox(6, plate(102)), AuditBox(6, plate(900))]
    r = evaluate_objects(tracks, {"o0": "real_readable"}, {}, {6: [plate(100)]}, 1080, audit)["plate"]
    assert (r.missed_by_all.hits, r.missed_by_all.total) == (1, 2)


def test_passing_needs_every_bar():
    good = ClassResult("plate", 3000, (), Rate(995, 1000), Rate(0, 1000))
    assert good.passed
    assert not ClassResult("plate", 2999, (), Rate(995, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, tuple(Leak(k, (0,)) for k in range(4)), Rate(995, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(994, 1000), Rate(0, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(995, 1000), Rate(2, 1000)).passed
    assert not ClassResult("plate", 3000, (), Rate(995, 1000), None).passed


def test_model_error_rate_compares_people_with_the_last_model_verdict_on_the_sample():
    js = [Judgment("o1", "not_real", "llm:x"), Judgment("o1", "real_readable", "human"),
          Judgment("o2", "real_readable", "llm:x"), Judgment("o2", "real_readable", "human"),
          Judgment("o3", "not_real", "llm:x")]  # o3 not judged by people yet
    r = model_error_rate(js, {"o1", "o2", "o3"})
    assert (r.hits, r.total) == (1, 2)
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_eval_objects.py`
Expected: collection error, `No module named 'vc_eval.objects'`.

- [ ] **Step 3: Implement** — `py/vc_eval/objects.py`

```python
"""Per-object leaks and per-frame recall against the vc_gt answer key (spec 2026-10-22 §4): an object leaks when a
sampled frame shows it readable and the masks leave it uncovered; the bars are 99.9 % without a leak over at least
3,000 readable real objects, 99.5 % per-frame recall, at most 0.1 % of the people's audit boxes outside every track."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vc_eval.metrics import COVERED_FRACTION, Rate, covered_fraction, must_mask
from vc_gt.audit import AuditBox
from vc_gt.crops import unmasked_frames, write_crop
from vc_gt.judge import REAL, frame_id, object_id
from vc_gt.tracks import Track
from vc_hardneg.judgments import HUMAN, Judgment, resolve
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box

ALPHA = 0.05
NEEDED_OBJECTS = 3000
NO_LEAK_PASS = 0.999
FRAME_RECALL_PASS = 0.995
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
    frame_recall: Rate  # must-mask boxes of real objects on sampled frames, covered
    missed_by_all: Rate | None  # hits: audit boxes no track holds; None without an audit

    @property
    def no_leak(self) -> Rate:
        return Rate(self.readable - len(self.leaks), self.readable)

    @property
    def passed(self) -> bool:
        return (self.readable >= NEEDED_OBJECTS and self.no_leak.value >= NO_LEAK_PASS
                and self.frame_recall.value >= FRAME_RECALL_PASS and self.missed_by_all is not None
                and self.missed_by_all.total > 0 and self.missed_by_all.value <= MISSED_BY_ALL_MAX)


def evaluate_objects(tracks: list[Track], objects: dict[str, str], frames: dict[str, str],
                     masks_by_frame: dict[int, list[Box]], frame_height: int,
                     audit: list[AuditBox] | None = None) -> dict[str, ClassResult]:
    """objects: object id -> final verdict of question A; frames: frame id -> final verdict of question B."""
    out = {}
    for cls in LABEL_CLASSES:
        readable, leaks, recall = 0, [], Rate()
        for t in (t for t in tracks if t.cls == cls and objects.get(object_id(t.id)) in REAL):
            for f, b in zip(t.frames, t.boxes):
                if must_mask(b, frame_height):
                    recall = recall.add(covered_fraction(b, masks_by_frame.get(f, [])) >= COVERED_FRACTION)
            leaked = tuple(f for f in unmasked_frames(t, masks_by_frame) if frames.get(frame_id(t.id, f)) == "readable")
            if objects[object_id(t.id)] == "real_readable" or leaked:
                readable += 1
            if leaked:
                leaks.append(Leak(t.id, leaked))
        missed = None
        if audit is not None:
            mine = [a for a in audit if a.box.cls == cls]
            missed = Rate(sum(not any(t.cls == cls and (b := t.box_at(a.frame)) is not None and iou(b, a.box) >= AUDIT_IOU
                                      for t in tracks) for a in mine), len(mine))
        out[cls] = ClassResult(cls, readable, tuple(leaks), recall, missed)
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
                 images: dict[int, Path], masks_by_frame: dict[int, list[Box]]) -> Path:
    """out_dir/summary.md with the table, the model error rates and every leak; a crop per leaked frame in
    out_dir/leaks/."""
    import cv2

    out_dir = Path(out_dir)
    (out_dir / "leaks").mkdir(parents=True, exist_ok=True)
    by_id = {t.id: t for t in tracks}
    lines = ["# 按目标泄露与按帧召回", "", "| 类别 | 看得清的真目标 | 泄露 | 不泄露率（95% 区间） | 按帧召回（95% 区间） | 大家一起漏 | 达标 |",
             "|---|---|---|---|---|---|---|"]
    for cls, r in results.items():
        missed = _pct(r.missed_by_all) if r.missed_by_all is not None else "未抽查"
        flag = " ⚠ 高于 0.1%，主指标会被高估" if r.missed_by_all is not None and r.missed_by_all.value > MISSED_BY_ALL_MAX else ""
        verdict = "是" if r.passed else ("否" if r.readable >= NEEDED_OBJECTS else f"不下结论（差 {NEEDED_OBJECTS - r.readable} 个目标）")
        lines.append(f"| {cls} | {r.readable} | {len(r.leaks)} | {_pct(r.no_leak)} | {_pct(r.frame_recall)} | {missed}{flag} | {verdict} |")
    lines += ["", "## 模型出错率（H2 抽核里被人工推翻的比例）", ""]
    lines += [f"- {name}：{_pct(rate)}" for name, rate in error_rates.items()]
    lines += ["", "## 泄露目标", ""]
    for cls, r in results.items():
        for leak in r.leaks:
            t = by_id[leak.track]
            for f in leak.frames:
                box = t.box_at(f)
                name = f"o{t.id}@{f:06d}.jpg"
                write_crop(cv2.imread(str(images[f])), box, out_dir / "leaks" / name, 3.0, 96)
                around = ", ".join(f"{g}:{covered_fraction(box, masks_by_frame.get(g, [])):.0%}"
                                   for g in range(f - NEIGHBOURS, f + NEIGHBOURS + 1))
                lines.append(f"- {cls} o{t.id} 第 {f} 帧：![](leaks/{name}) 前后原始帧遮盖比例 {around}")
    page = out_dir / "summary.md"
    page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return page
```

In `py/vc_eval/cli.py`:
- add `| objects` to the module docstring's command list and `spec 2026-10-22 §4` to its spec list;
- after the `newmisses` parser add:

```python
    o = sub.add_parser("objects", help="per-object leaks and per-frame recall against a vc_gt answer key")
    o.add_argument("--gt", type=Path, required=True, help="data/gt/<stem>")
    o.add_argument("--frames", type=Path, required=True, help="the 10 fps frames the answer key was built on")
    o.add_argument("--masks", type=Path, required=True, help="masks.jsonl of the vc_redact run under test")
    o.add_argument("--out", type=Path, required=True)
```

- add the handler before `main`:

```python
def _objects(args) -> int:
    import json as _json

    from vc_eval.objects import evaluate_objects, model_error_rate, write_report
    from vc_eval.records import read_masks
    from vc_gt.audit import AUDIT, H2_SAMPLE, read_audit
    from vc_gt.candidates import TRACKS
    from vc_gt.judge import FRAMES, OBJECTS
    from vc_gt.tracks import read_tracks
    from vc_hardneg.judgments import read_judgments, resolve
    from vc_label.contracts import read_frames_csv
    from vc_label.review import image_size

    tracks = read_tracks(args.gt / TRACKS)
    refs = read_frames_csv(args.frames)
    images = {r.frame: r.image for r in refs}
    _, masks = read_masks(args.masks)
    objects_j, frames_j = read_judgments(args.gt / OBJECTS), read_judgments(args.gt / FRAMES)
    objects = {i: j.verdict for i, j in resolve(objects_j).items()}
    frames = {i: j.verdict for i, j in resolve(frames_j).items()}
    audit = read_audit(args.gt / AUDIT)[1] if (args.gt / AUDIT).exists() else None
    results = evaluate_objects(tracks, objects, frames, masks, image_size(refs[0].image)[1], audit)
    sample = _json.loads((args.gt / H2_SAMPLE).read_text()) if (args.gt / H2_SAMPLE).exists() else {}
    errors = {"问题 A（真不真、认不认得出）": model_error_rate(objects_j, set(sample.get("objects", []))),
              "问题 B（未打码帧认不认得出）": model_error_rate(frames_j, set(sample.get("frames", [])))}
    page = write_report(results, errors, args.out, tracks, images, masks)
    for cls, r in results.items():
        print(f"{cls}: {r.readable} readable real objects, {len(r.leaks)} leaked, frame recall "
              f"{r.frame_recall.hits}/{r.frame_recall.total}, passed {r.passed}")
    print(f"report: {page}")
    return 0
```

- register it: add `"objects": _objects` to the `handlers` dict in `main`.

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_eval_objects.py tests/test_eval_metrics.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add py/vc_eval/objects.py py/vc_eval/cli.py py/tests/test_eval_objects.py
git commit -m "feat: vc_eval objects measures per-object leaks and per-frame recall against a vc_gt answer key"
```

---

### Task 9: `python -m vc_gt` and the README

**Files:**
- Create: `py/vc_gt/cli.py`, `py/vc_gt/__main__.py`
- Modify: `py/README.md` (new section "测试集（vc_gt）")
- Test: `py/tests/test_gt_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7; `vc_eval.consistency.run_vc_detect`; `vc_eval.records.read_masks`.
- Produces: `main(argv: list[str] | None = None) -> int` with commands `candidates`, `judge`, `review`, `annotate`, `import-human`; exit 0 on success, 2 on a usage or input error (message on stderr).

- [ ] **Step 1: Write the failing tests** — `py/tests/test_gt_cli.py`

```python
import csv
import json

import cv2
import numpy as np

from vc_gt.audit import AUDIT, AUDIT_FRAMES, H2_SAMPLE
from vc_gt.candidates import PROCESSED, TRACKS
from vc_gt.cli import main
from vc_gt.config import load_config
from vc_gt.crops import CROPS_INDEX, append_crop_index, write_object_crops
from vc_gt.judge import OBJECTS
from vc_gt.tracks import Track, write_tracks
from vc_hardneg.judgments import Judgment, append_judgments, read_judgments
from vc_label.contracts import Box


def _setup(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    with (frames / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for fr in (0, 6):
            w.writerow(["v.mp4", fr, 0.0, "base", 0.0])
            cv2.imwrite(str(frames / f"{fr:06d}.jpg"), np.zeros((300, 400, 3), np.uint8))
    gt = tmp_path / "gt"
    gt.mkdir()
    box = Box("plate", 100, 100, 140, 112)
    track = Track(0, "plate", (0, 6), (box, box))
    write_tracks(gt / TRACKS, "v.mp4", [track])
    (gt / PROCESSED).write_text("0 0\n6 1\n")
    images = {f: frames / f"{f:06d}.jpg" for f in (0, 6)}
    append_crop_index(gt / CROPS_INDEX, "o0", write_object_crops(track, images, gt / "crops" / "objects", load_config()))
    masks = tmp_path / "masks.jsonl"
    masks.write_text(json.dumps({"video": "v.mp4", "frame": 0, "boxes": []}) + "\n")
    return frames, gt, masks


def test_judge_refuses_without_the_llm_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VC_HARDNEG_ALLOW_LLM", raising=False)
    frames, gt, masks = _setup(tmp_path)
    assert main(["judge", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks)]) == 2
    assert "VC_HARDNEG_ALLOW_LLM" in capsys.readouterr().err


def test_review_h1_then_import_people_verdicts(tmp_path):
    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "unsure", "llm:x")],
                     verdicts=("real_readable", "real_unreadable", "not_real", "unsure"))
    assert main(["review", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks), "--set", "h1",
                 "--out", str(tmp_path / "page")]) == 0
    assert '"o0"' in (tmp_path / "page" / "objects" / "index.html").read_text(encoding="utf-8")
    export = tmp_path / "e.json"
    export.write_text(json.dumps([{"id": "o0", "verdict": "real_readable"}]))
    assert main(["import-human", "--gt", str(gt), "--file", str(export), "--kind", "objects"]) == 0
    assert read_judgments(gt / OBJECTS)[-1] == Judgment("o0", "real_readable", "human")


def test_review_h2_keeps_its_sample(tmp_path):
    frames, gt, masks = _setup(tmp_path)
    append_judgments(gt / OBJECTS, [Judgment("o0", "real_readable", "llm:x")],
                     verdicts=("real_readable", "real_unreadable", "not_real", "unsure"))
    args = ["review", "--gt", str(gt), "--frames", str(frames), "--masks", str(masks), "--set", "h2",
            "--out", str(tmp_path / "page")]
    assert main(args) == 0
    first = json.loads((gt / H2_SAMPLE).read_text())
    assert first["objects"] == ["o0"] and main(args) == 0 and json.loads((gt / H2_SAMPLE).read_text()) == first


def test_annotate_then_import_the_audit(tmp_path):
    frames, gt, _ = _setup(tmp_path)
    assert main(["annotate", "--gt", str(gt), "--frames", str(frames), "--out", str(tmp_path / "ann")]) == 0
    sample = json.loads((gt / AUDIT_FRAMES).read_text())
    assert sample == {"plate": [0, 6], "face": [6]}
    export = tmp_path / "a.json"
    export.write_text(json.dumps([{"key": "plate:0", "done": True, "boxes": [[1, 2, 30, 12]]}]))
    assert main(["import-human", "--gt", str(gt), "--file", str(export), "--kind", "audit"]) == 0
    assert json.loads((gt / AUDIT).read_text().splitlines()[0])["frame"] == 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_cli.py`
Expected: collection error, `No module named 'vc_gt.cli'`.

- [ ] **Step 3: Implement**

`py/vc_gt/__main__.py`:

```python
import sys

from vc_gt.cli import main

sys.exit(main())
```

`py/vc_gt/cli.py`:

```python
"""python -m vc_gt candidates | judge | review | annotate | import-human (spec 2026-10-22 §2): build the answer key of
one held-out segment in data/gt/<stem>/; `python -m vc_eval objects` measures a vc_redact run against it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from vc_gt.audit import (AUDIT, AUDIT_FRAMES, FRAME_CHOICES, H2_SAMPLE, OBJECT_CHOICES, PageItem, audit_sample,
                         h1_frame_ids, h1_object_ids, h2_sample, import_annotations, import_choices, write_annotate_page,
                         write_audit, write_choice_page)
from vc_gt.candidates import TRACKS, read_persons
from vc_gt.config import DEFAULT_CONFIG, load_config
from vc_gt.crops import CROPS_INDEX, read_crop_index, unmasked_frames
from vc_gt.judge import (FRAME_VERDICTS, FRAMES, OBJECT_VERDICTS, OBJECTS, REAL, frame_id, judge_frames, judge_objects,
                         make_runner, object_id)
from vc_gt.tracks import read_tracks
from vc_hardneg.judgments import HUMAN, append_judgments, read_judgments, resolve
from vc_label.contracts import read_frames_csv

REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_gt")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates", help="every plausible plate and face, linked into tracks (slow, resumable)")
    c.add_argument("--frames", type=Path, required=True, help="10 fps frames of one segment (vc_extract)")
    c.add_argument("--out", type=Path, required=True, help="data/gt/<stem>")
    c.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    c.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    c.add_argument("--vc-detect", type=Path, default=REPO / "build/cpp-release/vc_detect")
    c.add_argument("--student-model", type=Path, default=REPO / "data/models/r3")
    c.add_argument("--limit", type=int, help="process at most N new frames in this run")
    j = sub.add_parser("judge", help="ask the LLM questions A and B (needs VC_HARDNEG_ALLOW_LLM=1)")
    j.add_argument("--gt", type=Path, required=True)
    j.add_argument("--frames", type=Path, required=True)
    j.add_argument("--masks", type=Path, required=True, help="masks.jsonl of the vc_redact run under test")
    j.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    j.add_argument("--stage", choices=["objects", "frames", "both"], default="both")
    r = sub.add_parser("review", help="local pages for people: h1 (leaks and unsure) or h2 (sample of model verdicts)")
    r.add_argument("--gt", type=Path, required=True)
    r.add_argument("--frames", type=Path, required=True)
    r.add_argument("--masks", type=Path, required=True)
    r.add_argument("--set", choices=["h1", "h2"], required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = sub.add_parser("annotate", help="H3: a page to box every readable plate (face) on sampled whole frames")
    a.add_argument("--gt", type=Path, required=True)
    a.add_argument("--frames", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    i = sub.add_parser("import-human", help="append a page's export as people's verdicts (or the audit boxes)")
    i.add_argument("--gt", type=Path, required=True)
    i.add_argument("--file", type=Path, required=True)
    i.add_argument("--kind", choices=["objects", "frames", "audit"], required=True)
    return p


def _candidates(args) -> int:
    from vc_eval.consistency import run_vc_detect
    from vc_gt.candidates import TeacherFinder, build_candidates

    cfg = load_config(args.config)
    refs = read_frames_csv(args.frames)
    student = run_vc_detect(args.vc_detect, args.student_model, refs, args.out / "student", score=cfg.student_score)
    n = build_candidates(args.frames, args.out, TeacherFinder(cfg, args.device), student, cfg, args.limit)
    done = (args.out / TRACKS).exists()
    print(f"candidates: {n} frames processed in this run; tracks {'written' if done else 'not yet (frames left)'}")
    return 0


def _judge(args) -> int:
    from vc_eval.records import read_masks

    cfg = load_config(args.config)
    runner, name = make_runner(cfg)
    tracks = read_tracks(args.gt / TRACKS)
    images = {r.frame: r.image for r in read_frames_csv(args.frames)}
    if args.stage in ("objects", "both"):
        print(f"question A: {dict(judge_objects(args.gt, tracks, images, cfg, runner, name))}")
    if args.stage in ("frames", "both"):
        _, masks = read_masks(args.masks)
        print(f"question B: {dict(judge_frames(args.gt, tracks, images, masks, cfg, runner, name))}")
    return 0


def _unmasked_ids(gt: Path, masks_path: Path, tracks) -> set[str]:
    from vc_eval.records import read_masks

    _, masks = read_masks(masks_path)
    objects = resolve(read_judgments(gt / OBJECTS))
    return {frame_id(t.id, f) for t in tracks if (j := objects.get(object_id(t.id))) is not None and j.verdict in REAL
            for f in unmasked_frames(t, masks)}


def _review(args) -> int:
    cfg = load_config(args.config)
    tracks = read_tracks(args.gt / TRACKS)
    cls_of = {object_id(t.id): t.cls for t in tracks}
    objects = resolve(read_judgments(args.gt / OBJECTS))
    frames = resolve(read_judgments(args.gt / FRAMES))
    index = read_crop_index(args.gt / CROPS_INDEX)
    if args.set == "h1":
        object_ids, frame_ids = h1_object_ids(objects), h1_frame_ids(frames, _unmasked_ids(args.gt, args.masks, tracks))
    else:
        path = args.gt / H2_SAMPLE
        if not path.exists():
            model = lambda js: [i for i, j in resolve(js, include_human=False).items() if j.judge != HUMAN]
            path.write_text(json.dumps({
                "objects": h2_sample(model(read_judgments(args.gt / OBJECTS)), cfg.audit.objects, cfg.audit.seed),
                "frames": h2_sample(model(read_judgments(args.gt / FRAMES)), cfg.audit.frames, cfg.audit.seed)}))
        sample = json.loads(path.read_text())
        object_ids, frame_ids = sample["objects"], sample["frames"]
    for kind, ids, choices, judged in (("objects", object_ids, OBJECT_CHOICES, objects),
                                       ("frames", frame_ids, FRAME_CHOICES, frames)):
        items = [PageItem(i, index[i], f"{cls_of[i.split('@')[0]]} 模型：{judged[i].verdict}" if i in judged else "")
                 for i in ids if i in index]
        page = write_choice_page(items, args.out / kind, f"{args.set.upper()} {kind} {args.gt.name}", choices)
        print(f"{kind}: {len(items)} to judge, {page}")
    return 0


def _annotate(args) -> int:
    cfg = load_config(args.config)
    path = args.gt / AUDIT_FRAMES
    if not path.exists():
        persons = read_persons(args.gt)
        path.write_text(json.dumps(audit_sample(sorted(persons), persons, cfg.audit.frames_per_class, cfg.audit.seed)))
    sample = json.loads(path.read_text())
    images = {r.frame: r.image for r in read_frames_csv(args.frames)}
    frames = [(f"{cls}:{f}", images[f], cls) for cls in ("plate", "face") for f in sample[cls]]
    page = write_annotate_page(frames, args.out, f"H3 {args.gt.name}")
    print(f"annotate: {len(frames)} frames, {page}")
    return 0


def _import(args) -> int:
    if args.kind == "audit":
        sample = json.loads((args.gt / AUDIT_FRAMES).read_text())
        keys = {f"{cls}:{f}": (cls, f) for cls in sample for f in sample[cls]}
        rows = import_annotations(args.file, keys)
        write_audit(args.gt / AUDIT, rows)
        print(f"audit: {len(rows)} finished frames, {sum(len(r['boxes']) for r in rows)} boxes")
        return 0
    tracks = read_tracks(args.gt / TRACKS)
    if args.kind == "objects":
        known = {object_id(t.id) for t in tracks}
        allowed, path = tuple(v for v, _ in OBJECT_CHOICES), args.gt / OBJECTS
        verdicts = OBJECT_VERDICTS
    else:
        known = {frame_id(t.id, f) for t in tracks for f in t.frames}
        allowed, path = tuple(v for v, _ in FRAME_CHOICES), args.gt / FRAMES
        verdicts = FRAME_VERDICTS
    items = import_choices(args.file, known, allowed)
    append_judgments(path, items, verdicts=verdicts)
    print(f"import-human: {len(items)} {args.kind} verdicts")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"candidates": _candidates, "judge": _judge, "review": _review, "annotate": _annotate,
                "import-human": _import}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError, PermissionError, subprocess.CalledProcessError) as e:
        print(f"vc_gt: {e}", file=sys.stderr)
        return 2
```

`PermissionError` is a subclass of `OSError`; it is listed for the reader.

`py/README.md` — add this section at the end:

````markdown
## 测试集（vc_gt）

按目标衡量泄露的标准答案（设计：`docs/superpowers/specs/2026-10-22-gt-test-set-design.md`）。每段一个目录 `data/gt/<段>/`，不进 git，**这些段永远不进 `vc_train dataset --stem`**。

```bash
build/cpp-release/vc_extract --input data/videos/test-v2/<段>.mp4 --out data/gt-frames --fps 10 --max-fps 10
PYTHONPATH=py .venv/bin/python -m vc_gt candidates --frames data/gt-frames/<段> --out data/gt/<段>
scripts/redact.sh data/redacted/gt data/videos/test-v2/<段>.mp4
VC_HARDNEG_ALLOW_LLM=1 PYTHONPATH=py .venv/bin/python -m vc_gt judge --gt data/gt/<段> --frames data/gt-frames/<段> \
  --masks data/redacted/gt/<段>/masks.jsonl
PYTHONPATH=py .venv/bin/python -m vc_gt review --set h1 ...   # 再 --set h2；网页导出后 import-human --kind objects|frames
PYTHONPATH=py .venv/bin/python -m vc_gt annotate --gt data/gt/<段> --frames data/gt-frames/<段> --out <页面目录>
PYTHONPATH=py .venv/bin/python -m vc_gt import-human --gt data/gt/<段> --file ~/Downloads/audit_export.json --kind audit
PYTHONPATH=py .venv/bin/python -m vc_eval objects --gt data/gt/<段> --frames data/gt-frames/<段> \
  --masks data/redacted/gt/<段>/masks.jsonl --out data/reports/gt/<段>
```

`judge` 会把截图发给 `configs/gt.json` 里的命令（默认 Codex），只有设置 `VC_HARDNEG_ALLOW_LLM=1` 才会运行，走计费流量。`candidates` 最慢（OWLv2 分块），可以 `--limit` 分批跑，中断后重跑会接着做。换打码设置只需重跑 vc_redact、`judge --stage frames`（只问新出现的未打码帧）和 `vc_eval objects`。
````

- [ ] **Step 4: Run the tests**

Run: `cd py && ../.venv/bin/python -m pytest -q tests/test_gt_cli.py tests/test_gt_audit.py tests/test_gt_judge.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add py/vc_gt/cli.py py/vc_gt/__main__.py py/tests/test_gt_cli.py py/README.md
git commit -m "feat: python -m vc_gt builds a segment's answer key step by step"
```

---

### Task 10: Pilot on sg_drive_60s (stops for the user twice)

This task runs the real pipeline; it writes no code unless the pilot shows a bug (then fix it test-first in the module concerned and commit before going on).

- [ ] **Step 1: Extract 10 fps frames**

```bash
build/cpp-release/vc_extract --input data/videos/sg_drive_60s.mp4 --out data/gt-frames --fps 10 --max-fps 10
```

Expected: `data/gt-frames/sg_drive_60s/frames.csv` with about 1,200 rows (the video is 120 s).

- [ ] **Step 2: Candidates in the background** (OWLv2 on vehicles + 4 tiles per frame: expect tens of minutes; do not run it while a training job runs)

```bash
PYTHONPATH=py .venv/bin/python -m vc_gt candidates --frames data/gt-frames/sg_drive_60s --out data/gt/sg_drive_60s \
  > data/gt/sg_drive_60s.candidates.log 2>&1
```

Expected: `tracks.jsonl` written. Report the number of tracks per class (`grep -c '"cls": "plate"' data/gt/sg_drive_60s/tracks.jsonl`, same for face).

- [ ] **Step 3: The system's masks**: reuse `data/redacted/demo-r3-pw16/sg_drive_60s/masks.jsonl` (r3 + 16 px, the default). It holds every original frame, so frame numbers match the extract.

- [ ] **Step 4: STOP — ask the user before any LLM call.** Tell them how many object grids (tracks / 4) and an upper bound on frame grids the pilot will send, and that it uses the metered link. Only on their explicit yes:

```bash
VC_HARDNEG_ALLOW_LLM=1 PYTHONPATH=py .venv/bin/python -m vc_gt judge --gt data/gt/sg_drive_60s \
  --frames data/gt-frames/sg_drive_60s --masks data/redacted/demo-r3-pw16/sg_drive_60s/masks.jsonl
```

- [ ] **Step 5: H2 pages for the user** (pilot size: the default 50 + 50)

```bash
PYTHONPATH=py .venv/bin/python -m vc_gt review --gt data/gt/sg_drive_60s --frames data/gt-frames/sg_drive_60s \
  --masks data/redacted/demo-r3-pw16/sg_drive_60s/masks.jsonl --set h2 --out data/review/gt-h2-sg_drive_60s
```

- [ ] **Step 6: STOP — the user judges both H2 pages** and exports. Import each export:

```bash
PYTHONPATH=py .venv/bin/python -m vc_gt import-human --gt data/gt/sg_drive_60s --file ~/Downloads/human_judgments.json --kind objects
PYTHONPATH=py .venv/bin/python -m vc_gt import-human --gt data/gt/sg_drive_60s --file "~/Downloads/human_judgments (1).json" --kind frames
```

(use the file names the browser actually gave the downloads).

- [ ] **Step 7: Pilot report**

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval objects --gt data/gt/sg_drive_60s --frames data/gt-frames/sg_drive_60s \
  --masks data/redacted/demo-r3-pw16/sg_drive_60s/masks.jsonl --out data/reports/gt/sg_drive_60s
```

Show the user the summary: tracks, readable real objects, leaks with their crops, frame recall, and the two model error rates. The spec's gate: if either H2 overturn rate is above 2 %, propose prompt or judging changes before any new video; otherwise the pipeline is ready for the user's new videos (one segment at a time, stopping after each).
````
