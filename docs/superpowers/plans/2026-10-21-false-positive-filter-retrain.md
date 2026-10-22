# 减少误打：重训学生模型 r4（方案 B）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用洗过的老师标签和方案 A 判定出的误检反例重训学生模型 r4，在召回线不动的前提下减少误打，并和 r3 做端到端对比。

**Architecture:** 三步。
1. **洗标签**：新命令 `vc_hardneg mine-labels` 把训练段里老师的低分框挖成候选，用方案 A 的判定流程（本地老师放大复看 + Codex 一次判 4 张）判定。
2. **建数据集**：`vc_train dataset` 跳过被判"不是"的标签框（按你选定的策略，也可以去掉没判的低分框），并在误检位置额外裁出"反例样本"，样本里保留该区域的其余（清洗后的）标签。
3. **训练与评测**：从 r3 接着训 r4，导出后用 `vc_eval run` / `falsepos` / `newmisses` 和 r3 对比。

**Tech Stack:** Python 3.12、ultralytics YOLOv8n（MPS）、OpenCV、pytest；C++ `vc_redact`（不改）。

**设计文档：** `docs/superpowers/specs/2026-10-21-false-positive-filter-design.md` §5、§6。方案 A 的结论见本计划"背景"。

## 背景（方案 A 的结论，2026-10-21）

- 复核模型（裁图 + 小 CNN / MobileNetV3）分不清"又小又糊的真车牌"和尾灯、标志：在任何有效的门槛下都会拦掉真车牌，已停用（`data/models/verifier-v1-plates-rejected`）。复核的代码保留，不给模型时行为不变。
- **可复用的数据**：`data/hardneg/v1/` 下 4,100 张判定过的学生误检候选，含 1,658 张"不是"（判定流程经 100 张试判验证：判"不是"的里面真目标 0/41），以及 185 张人工结论。
- **训练标签的噪声很大**：10 个训练段里，老师低分车牌标签（OWLv2 0.1–0.3）有 14,957 个，占车牌标签的 69%；低分人脸标签（0.3–0.5）只有 378 个。尾灯、路牌这类误检就是从这里学来的。

## 与设计文档 §5 的两处细化

- **反例样本带标签**：设计里写"作为没有标签的背景图加入"。但误检附近常有真车牌，整张背景图会把它们也教成"不是"。改为：以误检为中心裁 640×640 的原尺度样本，**保留该区域内其余清洗后的标签**，只是误检本身没有标签。
- **洗标签先抽样再定策略**：低分车牌标签有约 1.5 万个，全判需要 Codex 约 4 小时。先抽 1,000 个判定、人工抽核 100 个，测出这批标签里真车牌的比例，再由用户在三种策略里选（见任务 3）。

## Global Constraints

- 召回硬门槛：三个留出段（`sg_drive_1200s`、`2400s`、`60s`）合计人脸 ≥ 95%、车牌 ≥ 98%；基线（r3）遮住、r4 没遮住的真目标为 0；逐帧参照各分数段覆盖比 r3 低不超过 1 个百分点。
- 误打目标：对不上老师框的打码，人脸至少减少 50%，车牌至少减少 30%（未达到也出报告，由用户定）。
- 速度：r4 与 r3 同结构（YOLOv8n、1920×1088），吞吐差异应在噪声内；10 档在 `sg_drive_2400s` 上慢不超过 10%。
- 训练只用 10 个训练段：`sg_drive_300s`、`600s`、`900s`、`1500s`、`1800s`、`2100s`、`2700s`、`3000s`、`3300s`、`3600s`；不用 `sg_full.mp4`。
- 低分档：车牌 [0.1, 0.3)、人脸 [0.3, 0.5)，取自 `configs/labeling.json` 的 `plate_low_score` / `face_low_score`。
- 判"拿不准"的标签框一律**保留**。
- 反例样本约占数据集样本数的 10%（`--hard-negative-share`，默认 0.10）。
- `data/labels/<段>/yolo/` 原件不改，清洗只发生在建数据集时。
- r4 替换 r3 的条件：召回不降且误打更少；否则保留 r3。是否替换由用户决定。
- 本机只跑与改动相关的测试；联网仅限已获批的 Codex 调用（`VC_HARDNEG_ALLOW_LLM=1`）；训练（约 2 小时）、批量判定开始前告知用户，放后台跑。
- 提交信息格式 `<type>: <description>`，结尾 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`；分支 `feat/fp-filter`。

## 文件结构

| 文件 | 职责 |
|---|---|
| `py/vc_hardneg/candidates.py` | 抽出裁图写盘的公共函数；新增 `mine_labels`：从老师标签挖低分框候选 |
| `py/vc_hardneg/cli.py` | 新子命令 `mine-labels` |
| `py/vc_train/cleaning.py`（新） | 读判定结果，决定每个标签框保留还是去掉；读反例 |
| `py/vc_train/dataset.py` | `build_dataset` 接收清洗与反例，写反例样本，汇总去掉多少框 |
| `py/vc_train/cli.py` | `dataset` 子命令的新参数 |
| `py/tests/test_hardneg_candidates.py`、`py/tests/test_train_cleaning.py`（新）、`py/tests/test_train_dataset.py` | 测试 |
| `docs/reports/fp-filter-2026-10-21.md`（新） | 方案 A 结论 + 方案 B 验收报告 |

## 命令

- Python 测试：在 `py/` 下 `../.venv/bin/python -m pytest tests/<文件> -q`
- 模块运行：在仓库根目录 `PYTHONPATH=py .venv/bin/python -m <模块> ...`

---

### Task 1: 从老师标签挖低分框候选（`vc_hardneg mine-labels`）

**Files:**
- Modify: `py/vc_hardneg/candidates.py`, `py/vc_hardneg/cli.py`
- Test: `py/tests/test_hardneg_candidates.py`

**Interfaces:**
- Consumes: `Candidate`, `candidate_id`, `crop_window`, `native_crop`, `write_candidates`, `HardnegConfig`（已有）。
- Produces:
  - `write_crops(cands: list[Candidate], frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> list[Candidate]`：写 `out_dir/crops/<id>.png`，返回填好 `image`、`crop_origin` 的候选，按 (stem, frame, id) 排序。`mine` 改为调用它。
  - `LOW_BANDS = {"plate": (0.1, 0.3), "face": (0.3, 0.5)}`。
  - `mine_labels(stems: list[str], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig, bands: dict[str, tuple[float, float]] = LOW_BANDS) -> MineSummary`：每个 `merged.jsonl` 框，若在 `processed.txt` 的帧上、类别在 `bands` 里、分数在 [lo, hi)、帧图存在，就成为候选。字段约定：`track=-1`、`dump_dir=""`、`dump_index=-1`、`frame_teacher=0.0`、`src` 为老师的 `src`。写 `out_dir/candidates.jsonl`。`MineSummary.rows` 为读到的老师框数，`masked` 为落在档内的框数，`candidates` 为候选数（等于 `masked`）。
  - CLI：`python -m vc_hardneg mine-labels --stem <stem>（可重复） --labels-root --frames-root --out --config`。

- [ ] **Step 1: 写失败的测试**

在 `py/tests/test_hardneg_candidates.py` 末尾加：

```python
def test_mine_labels_takes_the_low_band_teacher_boxes_on_processed_frames(tmp_path):
    from vc_hardneg.candidates import LOW_BANDS, mine_labels

    t = lambda frame, cls, box, score, src="teacher_owlv2": {  # noqa: E731
        "video": "s.mp4", "frame": frame, "pts_ms": 0, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
        "y2": box[3], "score": score, "src": src}
    teacher = [
        t(0, "plate", (10, 10, 30, 20), 0.15),  # low band: kept
        t(0, "plate", (40, 10, 60, 20), 0.35),  # confident: left out
        t(0, "plate", (70, 10, 90, 20), 0.05),  # below the band (not a label): left out
        t(12, "face", (10, 50, 30, 75), 0.4, "teacher_retinaface"),  # low band: kept
        t(12, "face", (40, 50, 60, 75), 0.6, "teacher_retinaface"),  # confident: left out
        t(24, "plate", (10, 10, 30, 20), 0.2),  # frame 24 is not processed: left out
    ]
    _labels(tmp_path, "s", [0, 12], teacher)
    out = tmp_path / "out"
    s = mine_labels(["s"], tmp_path / "labels", tmp_path / "frames", out, CFG)
    cands = read_candidates(out / "candidates.jsonl")
    assert (s.rows, s.candidates) == (6, 2) and len(cands) == 2
    assert [(c.frame, c.cls, c.score, c.src) for c in cands] == [(0, "plate", 0.15, "teacher_owlv2"),
                                                                  (12, "face", 0.4, "teacher_retinaface")]
    assert all(c.track == -1 and c.dump_index == -1 and c.dump_dir == "" for c in cands)
    assert all((out / c.image).is_file() for c in cands)
    assert LOW_BANDS == {"plate": (0.1, 0.3), "face": (0.3, 0.5)}
```

在 `py/tests/test_hardneg_review.py` 末尾加：

```python
def test_cli_mine_labels(tmp_path):
    import cv2
    import numpy as np

    lab = tmp_path / "labels" / "s"
    lab.mkdir(parents=True)
    (lab / "processed.txt").write_text("0\n")
    (lab / "merged.jsonl").write_text(json.dumps({"video": "s.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": 10,
                                                  "y1": 10, "x2": 30, "y2": 20, "score": 0.2, "src": "t"}) + "\n")
    (tmp_path / "frames" / "s").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "frames" / "s" / "000000.jpg"), np.zeros((60, 80, 3), np.uint8))
    assert main(["mine-labels", "--stem", "s", "--labels-root", str(tmp_path / "labels"),
                 "--frames-root", str(tmp_path / "frames"), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "candidates.jsonl").read_text().count("\n") == 1
```

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_candidates.py tests/test_hardneg_review.py -q`
Expected: 两个新测试 FAIL（`cannot import name 'mine_labels'`、`invalid choice: 'mine-labels'`）。

- [ ] **Step 2: 写实现**

`py/vc_hardneg/candidates.py`：

1. 在 `CROPS_DIR = "crops"` 下加：

```python
LOW_BANDS = {"plate": (0.1, 0.3), "face": (0.3, 0.5)}  # configs/labeling.json plate_low_score / face_low_score
```

2. 把 `mine` 末尾写小图的循环抽成公共函数（放在 `mine` 之前），`mine` 改为调用它：

```python
def write_crops(cands: list[Candidate], frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> list[Candidate]:
    """Each candidate's native-resolution crop to out_dir/crops/<id>.png; the candidates with image and crop_origin
    filled in, ordered by stem, frame and id."""
    (out_dir / CROPS_DIR).mkdir(parents=True, exist_ok=True)
    final: list[Candidate] = []
    images: dict[tuple[str, int], np.ndarray] = {}
    for c in sorted(cands, key=lambda c: (c.stem, c.frame, c.id)):
        key = (c.stem, c.frame)
        if key not in images:
            images.clear()  # candidates come frame by frame: keep one image
            path = frames_root / c.stem / f"{c.frame:06d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"{path} cannot be read")
            images[key] = image
        crop, origin = native_crop(images[key], crop_window(c.box, cfg.crop_factor, cfg.min_crop_px), cfg.pad_value)
        rel = f"{CROPS_DIR}/{c.id}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        final.append(replace(c, image=rel, crop_origin=origin))
    return final
```

   `mine` 里从 `(out_dir / CROPS_DIR).mkdir(...)` 到 `write_candidates(...)` 之前的那段循环，换成：

```python
    final = write_crops(picked, frames_root, out_dir, cfg)
```

3. 在文件末尾加：

```python
def mine_labels(stems: list[str], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig,
                bands: dict[str, tuple[float, float]] = LOW_BANDS) -> MineSummary:
    """The teachers' own boxes in a score band, as candidates (spec 2026-10-21 §5.1): judging them cleans the labels
    the student learns from. Only boxes on processed frames whose image is there."""
    found: list[Candidate] = []
    rows_read = 0
    for stem in stems:
        labels = labels_root / stem
        processed = {int(v) for v in (labels / "processed.txt").read_text().split()}
        for d in read_jsonl(labels / "merged.jsonl"):
            rows_read += 1
            b = d.box
            lo, hi = bands.get(b.cls, (1.0, 0.0))
            if d.frame not in processed or not lo <= b.score < hi:
                continue
            if not (frames_root / stem / f"{d.frame:06d}.jpg").is_file():
                continue
            coords = (b.x1, b.y1, b.x2, b.y2)
            found.append(Candidate(
                id=candidate_id(d.video, d.frame, b.cls, coords), video=d.video, stem=stem, frame=d.frame, cls=b.cls,
                box=coords, score=b.score, src=b.src, track=-1, dump_dir="", dump_index=-1, frame_teacher=0.0,
                image="", crop_origin=(0, 0),
            ))
    final = write_crops(found, frames_root, out_dir, cfg)
    write_candidates(out_dir / CANDIDATES, final)
    return MineSummary(rows=rows_read, masked=len(found), candidates=len(final))
```

`py/vc_hardneg/cli.py`：
- `_parser()` 里 `mine` 子命令之后加：

```python
    ml = sub.add_parser("mine-labels", help="candidates from the teachers' low-score label boxes (label cleaning)")
    ml.add_argument("--stem", action="append", required=True, help="a training segment; repeat")
    ml.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    ml.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    ml.add_argument("--out", type=Path, required=True)
    ml.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
```

- 加处理函数并加入分派表 `"mine-labels": _mine_labels`：

```python
def _mine_labels(args) -> int:
    from vc_hardneg.candidates import mine_labels

    s = mine_labels(args.stem, args.labels_root, args.frames_root, args.out, load_config(args.config))
    print(f"mine-labels: {s.rows} teacher boxes, {s.candidates} in the low-score bands; {args.out / CANDIDATES}")
    return 0
```

- [ ] **Step 3: 运行测试，确认通过**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_candidates.py tests/test_hardneg_review.py tests/test_hardneg_judges.py -q`
Expected: 全部 passed。

- [ ] **Step 4: 提交**

```bash
git add py/vc_hardneg/candidates.py py/vc_hardneg/cli.py py/tests/test_hardneg_candidates.py py/tests/test_hardneg_review.py
git commit -m "feat: vc_hardneg mine-labels turns the teachers' low-score boxes into candidates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 抽样判定低分标签并人工抽核（操作步骤，含用户关卡）

**产出**：`data/hardneg/labels-v1/`（不进仓库）。

- [ ] **Step 1: 挖候选**

```bash
PYTHONPATH=py .venv/bin/python -m vc_hardneg mine-labels $(for s in 300 600 900 1500 1800 2100 2700 3000 3300 3600; do echo --stem sg_drive_${s}s; done) \
  --labels-root data/labels --frames-root data/frames --out data/hardneg/labels-v1
```

Expected：约 15,300 个候选（车牌约 14,950，人脸约 380）。

- [ ] **Step 2: 抽 1,000 个判定**

`sample_stratified` 按 `_sample_key` 分层，而老师框的分层只有车牌两档和人脸。先告知用户：Codex 约 250 次调用、约 20 分钟。

```bash
VC_HARDNEG_ALLOW_LLM=1 PYTHONPATH=py .venv/bin/python -m vc_hardneg judge --llm --sample 1000 --seed 0 \
  --candidates data/hardneg/labels-v1/candidates.jsonl --judgments data/hardneg/labels-v1/judgments.jsonl
```

- [ ] **Step 3: 人工抽核 100 个**

从这 1,000 个里按最终判定分层抽 100 个（"不是"60、"是"20、"拿不准"20，不足则取全部），生成复核页：

```bash
PYTHONPATH=py .venv/bin/python -I - <<'EOF'
import random, sys
sys.path.insert(0, "py")
from vc_hardneg.judgments import read_judgments, resolve
D = "data/hardneg/labels-v1"
r = resolve(read_judgments(f"{D}/judgments.jsonl"))
rng = random.Random(0)
ids = []
for verdict, n in (("no", 60), ("yes", 20), ("unsure", 20)):
    pool = sorted(i for i, j in r.items() if j.verdict == verdict)
    ids += rng.sample(pool, min(n, len(pool)))
open(f"{D}/spot_ids.txt", "w").write("".join(i + "\n" for i in ids))
print(len(ids))
EOF
PYTHONPATH=py .venv/bin/python -m vc_hardneg review --candidates data/hardneg/labels-v1/candidates.jsonl \
  --judgments data/hardneg/labels-v1/judgments.jsonl --out data/hardneg/labels-v1/spot --only all --ids data/hardneg/labels-v1/spot_ids.txt
open data/hardneg/labels-v1/spot/index.html
```

停下，请用户逐张点"是 / 不是"并导出。读回：

```bash
PYTHONPATH=py .venv/bin/python -m vc_hardneg import-human --candidates data/hardneg/labels-v1/candidates.jsonl \
  --judgments data/hardneg/labels-v1/judgments.jsonl --file <导出文件>
```

- [ ] **Step 4: 统计并请用户定清洗策略（用户关卡）**

```bash
PYTHONPATH=py .venv/bin/python -I - <<'EOF'
import sys
sys.path.insert(0, "py")
from collections import Counter
from vc_hardneg.candidates import read_candidates
from vc_hardneg.judgments import HUMAN, read_judgments, resolve
D = "data/hardneg/labels-v1"
c = {x.id: x for x in read_candidates(f"{D}/candidates.jsonl")}
J = read_judgments(f"{D}/judgments.jsonl")
model = resolve(J, include_human=False)
human = {j.id: j.verdict for j in J if j.judge == HUMAN}
print("model verdicts by class:", Counter((c[i].cls, j.verdict) for i, j in model.items()))
for v in ("no", "yes", "unsure"):
    seen = [i for i in human if model[i].verdict == v]
    print(f"model {v}: people looked at {len(seen)}, real {sum(human[i] == 'yes' for i in seen)}")
EOF
```

据此算出：
- **判"不是"的错误率**：判"不是"的里面真车牌的比例。若超过 2%，按方案 A 的经验调门槛或改用逐张判，然后重判。
- **低分档整体的真车牌比例**：用三类判定的人工结果按占比加权估算。

把这两个数报给用户，请用户在三种策略里选：
- **A1 只去判过"不是"的**：只去掉判为"不是"的标签框，其余照留。最保守，洗得最少。
- **A2 全档放量判定**：剩下约 1.4 万个低分框全部判定（Codex 约 3,500 次调用、约 3–4 小时），再按 A1 去掉"不是"。
- **A3 去掉没判"是"的整档**：低分档里只保留判为"是"的，其余（含没判的）都去掉。洗得最狠；若人工抽核显示低分档里真车牌不少，会伤远处小车牌的召回。

用户选定后才进入任务 3、4。若选 A2，先执行：

```bash
VC_HARDNEG_ALLOW_LLM=1 PYTHONPATH=py .venv/bin/python -m vc_hardneg judge --llm \
  --candidates data/hardneg/labels-v1/candidates.jsonl --judgments data/hardneg/labels-v1/judgments.jsonl
```

---

### Task 3: 建数据集时清洗标签（`vc_train/cleaning.py` + `dataset.py`）

> **执行时的调整（2026-10-21，用户选定）**：两次人工抽核（共 233 张）的结果——
> - 车牌判"不是"的误判率 1/133（两张镀铬饰条框经用户确认不算车牌）；
> - 人脸判"不是"的误判率 8/27，**人脸标签不清洗**；
> - 低分车牌里约 25–35% 是真车牌，所以 A3（整档去掉）不做。
>
> 用户选定"形状规则 + 放量判定"：低分档里**高大于宽**的车牌框直接去掉（老师高分车牌里这种形状只占 1.1%，人工判为假标签的里面占 35%），其余低分车牌全部判定，去掉判"不是"的；判过"是"的永远保留。下面的接口因此改为：`LabelCleaning(no, yes, low_boxes, drop_tall_low_plates)`，只清洗车牌；命令行参数 `--drop-tall-low-plates` 取代 `--drop-unjudged-low`。

**Files:**
- Create: `py/vc_train/cleaning.py`, `py/tests/test_train_cleaning.py`
- Modify: `py/vc_train/dataset.py`, `py/vc_train/cli.py`, `py/tests/test_train_dataset.py`

**Interfaces:**
- Consumes: `read_candidates`、`read_judgments`、`resolve`（vc_hardneg）；`LOW_BANDS`（Task 1）；`iou`、`read_jsonl`、`Box`（vc_label）。
- Produces:
  - `LabelCleaning(no: dict[tuple[str, int], list[Box]], yes: dict[tuple[str, int], list[Box]], drop_unjudged_low: bool, low_boxes: dict[tuple[str, int], list[Box]])`。
  - `LabelCleaning.load(candidates: Path, judgments: Path, labels_root: Path, stems: list[str], drop_unjudged_low: bool) -> LabelCleaning`：`no` / `yes` 来自判定（按 stem、frame 归组）；`low_boxes` 来自各段 `merged.jsonl` 的低分档框。
  - `LabelCleaning.clean(stem: str, frame: int, boxes: list[Box]) -> tuple[list[Box], int]`：返回保留的框和去掉的框数。规则：与某个"不是"框同类且 IoU ≥ 0.9 的去掉；`drop_unjudged_low` 时，与某个低分档框同类、IoU ≥ 0.9、却不和任何"是"框 IoU ≥ 0.9 的也去掉。
  - `build_dataset(..., cleaning: LabelCleaning | None = None, hard_negatives: list[HardNegative] = (), hard_negative_share: float = 0.10)`（`HardNegative` 见 Task 4；本任务先只加 `cleaning`）。
  - `DatasetSummary` 新字段 `labels_dropped: int = 0`。
  - CLI：`vc_train dataset --label-candidates <file> --label-judgments <file> [--drop-unjudged-low]`，两个文件要么都给、要么都不给。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_train_cleaning.py`:

```python
import json

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.judgments import Judgment, append_judgments
from vc_label.contracts import Box
from vc_train.cleaning import LabelCleaning


def _cand(cid, frame, box, cls="plate", score=0.15):
    return Candidate(id=cid, video="s.mp4", stem="s", frame=frame, cls=cls, box=box, score=score, src="teacher_owlv2",
                     track=-1, dump_dir="", dump_index=-1, frame_teacher=0.0, image="", crop_origin=(0, 0))


def _setup(tmp_path):
    lab = tmp_path / "labels" / "s"
    lab.mkdir(parents=True)
    rows = [  # merged.jsonl: three low-band plates and one confident plate on frame 0
        {"video": "s.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": x, "y1": 10, "x2": x + 20, "y2": 20,
         "score": s, "src": "teacher_owlv2"} for x, s in ((10, 0.15), (40, 0.2), (70, 0.25), (100, 0.8))]
    (lab / "merged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_candidates(tmp_path / "c.jsonl", [_cand("a", 0, (10, 10, 30, 20)), _cand("b", 0, (40, 10, 60, 20))])
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "no", "llm:codex"), Judgment("b", "yes", "teacher")])
    labels = [Box("plate", x, 10, x + 20, 20) for x in (10.02, 40, 70, 100)]  # as read back from yolo/ (rounded)
    return labels


def test_boxes_judged_not_real_are_dropped(tmp_path):
    labels = _setup(tmp_path)
    clean = LabelCleaning.load(tmp_path / "c.jsonl", tmp_path / "j.jsonl", tmp_path / "labels", ["s"], drop_unjudged_low=False)
    kept, dropped = clean.clean("s", 0, labels)
    assert dropped == 1 and [round(b.x1) for b in kept] == [40, 70, 100]
    assert clean.clean("s", 12, labels) == (labels, 0)  # nothing judged on frame 12


def test_drop_unjudged_low_keeps_only_the_low_boxes_judged_real(tmp_path):
    labels = _setup(tmp_path)
    clean = LabelCleaning.load(tmp_path / "c.jsonl", tmp_path / "j.jsonl", tmp_path / "labels", ["s"], drop_unjudged_low=True)
    kept, dropped = clean.clean("s", 0, labels)
    # a: judged no; c (x 70, 0.25): low band, never judged; b: judged yes; the 0.8 plate is not in the band
    assert dropped == 2 and [round(b.x1) for b in kept] == [40, 100]


def test_unsure_boxes_stay(tmp_path):
    labels = _setup(tmp_path)
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "unsure", "human")])
    clean = LabelCleaning.load(tmp_path / "c.jsonl", tmp_path / "j.jsonl", tmp_path / "labels", ["s"], drop_unjudged_low=False)
    assert clean.clean("s", 0, labels)[1] == 0
```

在 `py/tests/test_train_dataset.py` 加一个用例。先读该文件，找到现有的、构造一个小数据集的测试辅助函数（构造 frames 目录、`frames.csv`、labels 目录、`processed.txt`、`yolo/*.txt` 的那个），照它的写法：

```python
def test_cleaning_drops_label_boxes_from_every_sample_and_counts_them(tmp_path):
    # Build a one-segment dataset whose frame 0 has two plate labels with the existing helper, then:
    class DropFirst:
        def clean(self, stem, frame, boxes):
            return boxes[1:], 1

    summary = build_dataset(segments, tmp_path / "ds", 1920, 1088, val_tail_ms=0.0, cleaning=DropFirst())
    assert summary.labels_dropped == summary.train_frames + summary.val_frames  # one per labelled frame
    full = (tmp_path / "ds" / "labels").rglob("*_full.txt")
    assert all(len(p.read_text().splitlines()) == 1 for p in full)
```

（`segments` 用现有辅助函数造；若现有测试用的是别的帧尺寸或 `val_tail_ms`，照搬它的值。）

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_cleaning.py tests/test_train_dataset.py -q`
Expected: FAIL（`No module named 'vc_train.cleaning'`、`unexpected keyword argument 'cleaning'`）。

- [ ] **Step 2: 写 `cleaning.py`**

```python
"""Label cleaning for the student (spec 2026-10-21 §5.1): teacher label boxes judged not real are left out of the
dataset; with drop_unjudged_low also the low-band boxes no judge called real. labels/<stem>/yolo is never changed."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vc_hardneg.candidates import LOW_BANDS, read_candidates
from vc_hardneg.judgments import read_judgments, resolve
from vc_label.boxes import iou
from vc_label.contracts import Box, read_jsonl

SAME_BOX_IOU = 0.9  # a label box read back from yolo/ (rounded to 6 decimals) and its merged.jsonl box


def _matches(box: Box, others: list[Box]) -> bool:
    return any(o.cls == box.cls and iou(o, box) >= SAME_BOX_IOU for o in others)


@dataclass
class LabelCleaning:
    no: dict[tuple[str, int], list[Box]]
    yes: dict[tuple[str, int], list[Box]]
    drop_unjudged_low: bool = False
    low_boxes: dict[tuple[str, int], list[Box]] = field(default_factory=dict)

    @staticmethod
    def load(candidates: Path, judgments: Path, labels_root: Path, stems: list[str],
             drop_unjudged_low: bool) -> LabelCleaning:
        final = resolve(read_judgments(judgments))
        no: dict[tuple[str, int], list[Box]] = defaultdict(list)
        yes: dict[tuple[str, int], list[Box]] = defaultdict(list)
        for c in read_candidates(candidates):
            j = final.get(c.id)
            if j is None or j.verdict == "unsure":
                continue
            (no if j.verdict == "no" else yes)[(c.stem, c.frame)].append(Box(c.cls, *c.box))
        low: dict[tuple[str, int], list[Box]] = defaultdict(list)
        if drop_unjudged_low:
            for stem in stems:
                for d in read_jsonl(labels_root / stem / "merged.jsonl"):
                    lo, hi = LOW_BANDS.get(d.box.cls, (1.0, 0.0))
                    if lo <= d.box.score < hi:
                        low[(stem, d.frame)].append(d.box)
        return LabelCleaning(dict(no), dict(yes), drop_unjudged_low, dict(low))

    def clean(self, stem: str, frame: int, boxes: list[Box]) -> tuple[list[Box], int]:
        key = (stem, frame)
        no, yes, low = self.no.get(key, []), self.yes.get(key, []), self.low_boxes.get(key, [])
        kept = []
        for b in boxes:
            if _matches(b, no):
                continue
            if self.drop_unjudged_low and _matches(b, low) and not _matches(b, yes):
                continue
            kept.append(b)
        return kept, len(boxes) - len(kept)
```

- [ ] **Step 3: 接进 `dataset.py` 与命令行**

`py/vc_train/dataset.py`：
- `DatasetSummary` 加字段 `labels_dropped: int = 0  # label boxes the cleaning left out`。
- `_plan(segments, val_tail_ms)` 加参数 `cleaning=None`，并返回 `(frames, dropped)`。在读出 `boxes = from_yolo_lines(...)` 之后加：

```python
            if cleaning is not None:
                boxes, n = cleaning.clean(stem, ref.frame, boxes)
                dropped += n
```

  函数开头加 `dropped = 0`，结尾改为 `return frames, dropped`。
- `build_dataset` 加参数 `cleaning=None`，`frames = _plan(...)` 改为 `frames, dropped = _plan(segments, val_tail_ms, cleaning)`，返回前 `summary = _write_dataset(...)`，再 `return replace(summary, labels_dropped=dropped)`。（文件头部已 `from dataclasses import dataclass, replace`。）
- `build_dataset` 的 docstring 末尾加一句：`cleaning (vc_train.cleaning.LabelCleaning) leaves label boxes out of every sample of their frame.`

`py/vc_train/cli.py` 的 `dataset` 子命令加：

```python
    d.add_argument("--label-candidates", type=Path, default=None, help="vc_hardneg mine-labels candidates.jsonl")
    d.add_argument("--label-judgments", type=Path, default=None, help="their judgments.jsonl")
    d.add_argument("--drop-unjudged-low", action="store_true",
                   help="also leave out the low-band label boxes no judge called real (strategy A3)")
```

`_dataset` 里，在 `build_dataset` 之前加：

```python
    cleaning = None
    if (args.label_candidates is None) != (args.label_judgments is None):
        raise ValueError("--label-candidates and --label-judgments go together")
    if args.label_candidates is not None:
        from vc_train.cleaning import LabelCleaning

        cleaning = LabelCleaning.load(args.label_candidates, args.label_judgments, args.labels_root, args.stem,
                                      args.drop_unjudged_low)
```

`build_dataset(...)` 调用加 `cleaning=cleaning`；打印行后加 `if s.labels_dropped: print(f"  label boxes left out by the cleaning: {s.labels_dropped}")`。

- [ ] **Step 4: 运行测试**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_cleaning.py tests/test_train_dataset.py tests/test_train_cli.py -q`
Expected: 全部 passed。

- [ ] **Step 5: 提交**

```bash
git add py/vc_train/cleaning.py py/vc_train/dataset.py py/vc_train/cli.py py/tests/test_train_cleaning.py py/tests/test_train_dataset.py
git commit -m "feat: vc_train dataset leaves out the label boxes judged not real" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 反例样本（以误检为中心的原尺度裁图）

**Files:**
- Modify: `py/vc_train/cleaning.py`, `py/vc_train/dataset.py`, `py/vc_train/cli.py`
- Test: `py/tests/test_train_cleaning.py`, `py/tests/test_train_dataset.py`

**Interfaces:**
- Produces:
  - `HardNegative(stem: str, frame: int, box: Box)`（frozen dataclass，放在 `cleaning.py`）。
  - `load_hard_negatives(candidates: Path, judgments: Path) -> list[HardNegative]`：最终判定为"不是"的候选，按 id 排序。
  - `hard_negative_window(box: Box, width: int, height: int, side: int = 640) -> tuple[int, int, int, int]`：以框中心为中心的 side×side 窗口，平移到画面内（画面比 side 小时取整幅）；返回 (x0, y0, w, h)。
  - `build_dataset(..., hard_negatives: list[HardNegative] = (), hard_negative_share: float = 0.10)`：对落在数据集帧上的反例，按 id 顺序取前 `round(share × 普通样本数)` 个。每个写一张 `<stem>_<frame:06d>_hn<k>.jpg`（原尺度，不缩放），标签是该帧清洗后、在窗口内可见 ≥ 30% 的框（`crop_boxes`），split 跟随该帧；manifest 照常写一行。`DatasetSummary` 新字段 `hard_negatives: int = 0`。
  - CLI：`vc_train dataset --hard-negatives <candidates.jsonl> --hard-negative-judgments <judgments.jsonl> [--hard-negative-share 0.10]`。

- [ ] **Step 1: 写失败的测试**

在 `py/tests/test_train_cleaning.py` 加：

```python
def test_hard_negatives_are_the_candidates_judged_not_real(tmp_path):
    from vc_train.cleaning import HardNegative, load_hard_negatives

    write_candidates(tmp_path / "c.jsonl", [_cand("b", 12, (40, 10, 60, 20)), _cand("a", 0, (10, 10, 30, 20)),
                                            _cand("c", 0, (70, 10, 90, 20))])
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "no", "llm:codex"), Judgment("b", "no", "teacher"),
                                            Judgment("c", "yes", "teacher")])
    assert load_hard_negatives(tmp_path / "c.jsonl", tmp_path / "j.jsonl") == [
        HardNegative("s", 0, Box("plate", 10, 10, 30, 20)), HardNegative("s", 12, Box("plate", 40, 10, 60, 20))]


def test_the_hard_negative_window_is_centred_and_kept_inside_the_frame():
    from vc_train.cleaning import hard_negative_window

    assert hard_negative_window(Box("plate", 900, 500, 920, 510), 1920, 1080) == (590, 185, 640, 640)
    assert hard_negative_window(Box("plate", 0, 0, 20, 10), 1920, 1080) == (0, 0, 640, 640)
    assert hard_negative_window(Box("plate", 1900, 1070, 1920, 1080), 1920, 1080) == (1280, 440, 640, 640)
    assert hard_negative_window(Box("plate", 10, 10, 20, 20), 300, 200) == (0, 0, 300, 200)
```

在 `py/tests/test_train_dataset.py` 加（同样照现有辅助函数造一个段，帧 0 上有一个车牌标签）：

```python
def test_hard_negatives_add_centred_native_crops_with_the_labels_inside(tmp_path):
    from vc_label.contracts import Box
    from vc_train.cleaning import HardNegative

    # segments: one segment, frame 0 labelled (use the existing helper)
    hn = [HardNegative(STEM, 0, Box("plate", 900, 500, 920, 510))]
    s = build_dataset(segments, tmp_path / "ds", 1920, 1088, val_tail_ms=0.0, hard_negatives=hn, hard_negative_share=1.0)
    assert s.hard_negatives == 1
    images = list((tmp_path / "ds" / "images").rglob("*_hn0.jpg"))
    assert len(images) == 1
    import cv2

    assert cv2.imread(str(images[0])).shape[:2] == (640, 640)
    rows = (tmp_path / "ds" / "manifest.csv").read_text().splitlines()
    assert any("_hn0.jpg" in r for r in rows)
```

（`STEM` 换成现有辅助函数用的段名；`hard_negative_share=1.0` 让唯一的反例一定入选。）

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_cleaning.py tests/test_train_dataset.py -q`
Expected: 新测试 FAIL。

- [ ] **Step 2: 写实现**

`py/vc_train/cleaning.py` 末尾加：

```python
@dataclass(frozen=True)
class HardNegative:
    """A student false positive on a labelled frame (vc_hardneg, judged "no"): its region becomes an extra sample."""

    stem: str
    frame: int
    box: Box


def load_hard_negatives(candidates: Path, judgments: Path) -> list[HardNegative]:
    final = resolve(read_judgments(judgments))
    out = [HardNegative(c.stem, c.frame, Box(c.cls, *c.box)) for c in sorted(read_candidates(candidates), key=lambda c: c.id)
           if final.get(c.id) is not None and final[c.id].verdict == "no"]
    return out


def hard_negative_window(box: Box, width: int, height: int, side: int = 640) -> tuple[int, int, int, int]:
    """side x side frame pixels centred on the box, shifted inside the frame (the whole frame when it is smaller)."""
    w, h = min(side, width), min(side, height)
    cx, cy = box.center
    x0 = int(min(max(round(cx - w / 2), 0), width - w))
    y0 = int(min(max(round(cy - h / 2), 0), height - h))
    return x0, y0, w, h
```

`py/vc_train/dataset.py`：
- `DatasetSummary` 加 `hard_negatives: int = 0`。
- `_write_dataset` 加参数 `hard_negatives`，在写完普通样本、排序 `rows` 之前加：

```python
    by_frame = {(f.stem, f.ref.frame): f for f in frames}
    usable = [h for h in hard_negatives if (h.stem, h.frame) in by_frame]
    taken = usable[: round(hard_negative_share * len(rows))]
    counter: dict[tuple[str, int], int] = {}
    for h in taken:
        frame = by_frame[(h.stem, h.frame)]
        image = cv2.imread(str(frame.ref.image))
        if image is None:
            raise ValueError(f"cannot read {frame.ref.image}")
        x0, y0, w, h_ = hard_negative_window(h.box, frame.width, frame.height)
        k = counter.get((h.stem, h.frame), 0)
        counter[(h.stem, h.frame)] = k + 1
        name = f"{frame.stem}_{frame.ref.frame:06d}_hn{k}"
        part_boxes = crop_boxes(frame.boxes, (x0, y0, w, h_))
        _write_sample(out_dir, frame.split, name, image[y0 : y0 + h_, x0 : x0 + w], part_boxes)
        rows.append(_manifest_row(name, frame, x0, y0, w, h_, 1.0, len(part_boxes)))
```

  并在文件头部 `from vc_train.geometry import ...` 之后加 `from vc_train.cleaning import hard_negative_window`。
- `_write_dataset` 的签名改为 `(out_dir, segments, frames, in_width, in_height, crop, hard_negatives=(), hard_negative_share=0.10)`，返回的 `DatasetSummary(...)` 加 `hard_negatives=len(taken)`。
- `build_dataset` 加参数 `hard_negatives=(), hard_negative_share: float = 0.10`，透传给 `_write_dataset`；`hard_negative_share` 不在 [0, 1] 时抛 `ValueError`（在 `_check_out_dir` 之前检查）。

`py/vc_train/cli.py` 的 `dataset` 子命令加：

```python
    d.add_argument("--hard-negatives", type=Path, default=None, help="vc_hardneg candidates.jsonl of student false positives")
    d.add_argument("--hard-negative-judgments", type=Path, default=None, help="their judgments.jsonl")
    d.add_argument("--hard-negative-share", type=float, default=0.10, help="extra samples, as a share of the others")
```

`_dataset` 里：

```python
    hard = []
    if (args.hard_negatives is None) != (args.hard_negative_judgments is None):
        raise ValueError("--hard-negatives and --hard-negative-judgments go together")
    if args.hard_negatives is not None:
        from vc_train.cleaning import load_hard_negatives

        hard = load_hard_negatives(args.hard_negatives, args.hard_negative_judgments)
```

`build_dataset(...)` 调用加 `hard_negatives=hard, hard_negative_share=args.hard_negative_share`；打印 `if s.hard_negatives: print(f"  hard-negative samples: {s.hard_negatives}")`。

- [ ] **Step 3: 运行测试**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_cleaning.py tests/test_train_dataset.py tests/test_train_cli.py -q`
Expected: 全部 passed。

- [ ] **Step 4: 提交**

```bash
git add py/vc_train/cleaning.py py/vc_train/dataset.py py/vc_train/cli.py py/tests/test_train_cleaning.py py/tests/test_train_dataset.py
git commit -m "feat: vc_train dataset adds native crops around judged student false positives" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 建 r4 数据集并训练（操作步骤）

- [ ] **Step 1: 建数据集**

按用户在任务 2 选的策略（A1 / A2 不加 `--drop-unjudged-low`，A3 加）：

```bash
PYTHONPATH=py .venv/bin/python -m vc_train dataset $(for s in 300 600 900 1500 1800 2100 2700 3000 3300 3600; do echo --stem sg_drive_${s}s; done) \
  --out data/datasets/r4 \
  --label-candidates data/hardneg/labels-v1/candidates.jsonl --label-judgments data/hardneg/labels-v1/judgments.jsonl \
  --hard-negatives data/hardneg/v1/candidates.jsonl --hard-negative-judgments data/hardneg/v1/judgments.jsonl
```

Expected：样本数约为 r3 的 5,871 再加约 10%，并打印去掉的标签框数和反例样本数。把这三个数报给用户。

- [ ] **Step 2: 训练（约 2 小时，后台）**

先告知用户：MPS 训练约 2 小时，期间本机较忙。配方同 r3：从上一版权重接着训，`time 2.0`。

```bash
PYTHONPATH=py .venv/bin/python -m vc_train train --data data/datasets/r4/data.yaml --run data/models/r4 \
  --weights data/models/r3/train/weights/best.pt --hours 2 --imgsz 1280 --batch 8 --device mps > data/models/r4-train.log 2>&1
```

- [ ] **Step 3: 导出并做一致性检查**

```bash
PYTHONPATH=py .venv/bin/python -m vc_train export --run data/models/r4 --name r4 --dataset data/datasets/r4 \
  --check-frames data/frames/sg_drive_1200s --check-count 20
```

Expected：`check: ... passed`（导出命令返回 0）。`data/models/r4/model.onnx` 和 `model.json` 生成，`model.json` 记录训练清单的 sha256。

---

### Task 6: r4 对 r3 的端到端验收、报告与文档

- [ ] **Step 1: 跑 r4（后台，约 15 分钟）**

r3 的基线就是 `data/reports/fp-thr0`（复核门槛 0，打码等同 r3，已跑完）。只需跑 r4：

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval run --stems sg_drive_1200s,sg_drive_2400s,sg_drive_60s --model data/models/r4 \
  --face-model data/models/yunet --fps 10 --recall-target face=0.95,plate=0.98 --config configs/redact.json \
  --out data/reports/fp-r4 --redacted-root data/redacted/fp-r4 --overwrite > data/reports/fp-r4.log 2>&1
```

- [ ] **Step 2: 误打计数、新漏核查、逐帧分段**

```bash
for s in sg_drive_1200s sg_drive_2400s sg_drive_60s; do for v in fp-thr0 fp-r4; do
  PYTHONPATH=py .venv/bin/python -m vc_eval falsepos --labels data/labels/$s \
    --detections data/redacted/$v/fps10/$s/detections.jsonl --json data/reports/$v/falsepos_$s.json
done
PYTHONPATH=py .venv/bin/python -m vc_eval newmisses --frames data/frames/$s --labels data/labels/$s \
  --baseline data/redacted/fp-thr0/fps10/$s/masks.jsonl --candidate data/redacted/fp-r4/fps10/$s/masks.jsonl \
  --out data/reports/fp-r4/newmisses/$s --recall-target face=0.95,plate=0.98
done
```

逐张看 `data/reports/fp-r4/newmisses/<段>/` 的裁图，记下真目标还是老师误检。**有一个真车牌或真人脸就算验收不过。**

`vc_eval run` 的逐帧参照是用 r4 自己重跑的，和 r3 的不是同一份。要可比，用 **r3 的逐帧参照**（`data/redacted/fp-thr0/dense`）分别算两者的分段覆盖：

```bash
PYTHONPATH=py .venv/bin/python -I - <<'EOF'
import sys
sys.path.insert(0, "py")
from pathlib import Path
from vc_eval.temporal import evaluate_temporal
for stem in ("sg_drive_1200s", "sg_drive_2400s", "sg_drive_60s"):
    dense = Path(f"data/redacted/fp-thr0/dense/{stem}/detections.jsonl")
    for run in ("fp-thr0", "fp-r4"):
        r = evaluate_temporal(dense, Path(f"data/redacted/{run}/fps10/{stem}/masks.jsonl"), 1080)
        print(stem, run, {k: f"{v.hits}/{v.total}" for k, v in sorted(r.bins.items())})
EOF
```

按 Global Constraints 逐项核对：
- 召回：`pooled[0].coverage`。
- 逐帧分段：每段的 `temporal_bins` 和 `fp-thr0` 比。
- 误打降幅：Step 2 的 JSON。
- 观感：`viewing`。
- 速度：`sg_drive_2400s` 的 `stats.json`。

- [ ] **Step 3: 写报告**

`docs/reports/fp-filter-2026-10-21.md`，格式参照 `docs/reports/stable-masks-2026-10-19.md`：
1. 结论；
2. 方案 A：试判过程（14.8% → 0/41）、复核模型的指标与门槛扫描、停用原因；
3. 方案 B：标签清洗策略与抽核数字、数据集变化、训练记录；
4. r3 / r4 验收表；
5. 新漏核查；
6. 已知问题。

- [ ] **Step 4: 更新文档**

- `py/README.md`：
  - 新增 `vc_hardneg` 一节：`mine`、`mine-labels`、`judge`（`--sample`、`--llm`、`VC_HARDNEG_ALLOW_LLM`）、`review`、`import-human`、`pilot`、`pilot-report`；`configs/hardneg.json` 各键；
  - `vc_train dataset` 的新参数；
  - `vc_train verifier`（含 `--arch`、`--class`）；
  - `vc_eval falsepos`、`newmisses`、`run --verifier-model`。
- `cpp/README.md`：`vc_redact` 的 `--verifier-model`、`--dump-verify-crops`、`--dump-verify-every`；配置表加 4 个 `verify_*` 键；`detections.jsonl` 的 `verify` / `verified`；`stats.json` 的 `verifier`、`verify`、`stages.verify`。写明复核模型目前**没有可用的版本**（方案 A 的结论）。
- `docs/USAGE.md`：`scripts/redact.sh` 的 `VC_VERIFIER`（目录不存在时不复核）。

- [ ] **Step 5: 提交并交给用户决定**

```bash
git add docs/reports/fp-filter-2026-10-21.md py/README.md cpp/README.md docs/USAGE.md
git commit -m "docs: false-positive report (verifier stopped, student r4) and usage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

把验收表和新漏核查结果报给用户，由用户决定：
- 是否把默认模型换成 r4（`scripts/redact.sh` 的 `VC_MODEL` 默认值、`vc_eval run` 的示例）；
- 是否把 `feat/fp-filter` 合并到 main。合并后按全局约定把本地 checkout 更新到最新。
