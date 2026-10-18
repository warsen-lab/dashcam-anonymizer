# Python 老师集成预标注与人工复核 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现规格 §5 的标注模块 `vc_label`：四个老师模型（RetinaFace、YuNet、COCO Faster R-CNN、OWLv2）对抽帧结果做预标注，加权融合后用规则标出疑似错误，导出给 X-AnyLabeling 人工复核，再把复核结果转成 YOLO 格式的金标准。所有标注文件都放在抽帧目录之外。

**Architecture:** Python 包 `py/vc_label/`。纯逻辑模块（数据契约、框运算、配置、规则、复核格式转换）与模型模块（`teachers/`）分开，前者用小样例做单测，后者用样例帧 `data/frames/sg_drive_600s/006005.jpg` 做真实模型回归。流程拆成 `teach`（慢、可断点续跑，只写老师原始输出）和 `fuse` / `export-review` / `import-review`（快，每次从原始输出重新计算）。

**Tech Stack:** Python 3.14、PyTorch 2.14（MPS）、torchvision 0.29、OpenCV 5.0、transformers 5.20、Pillow、pytest。

**规格：** `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md`（§3.2 数据契约、§5 标注模块、§9 测试）。前一份计划：`docs/superpowers/plans/2026-10-17-cpp-vc-extract.md`（已合并）。

**计划路线（第 2 份，共 5 份）：** 1 C++ 与 vc_extract（已完成）→ **2 本计划** → 3 训练与 ONNX 导出 → 4 vc_redact → 5 评测与验收。

## Global Constraints

- Python 虚拟环境固定为仓库根目录的 `.venv`（本机已建好），依赖版本固定在 `py/requirements.txt`；**不得新增依赖**，确需新增要先停下来问用户（计费流量）。
- **不联网**：所有模型权重已缓存在 `~/.cache/video-compliance/`（所有 checkout 共享），代码通过 `vc_label.paths.offline_model_env()` 设置 `HF_HUB_OFFLINE=1`、`HF_HOME`、`TORCH_HOME`。
- **只跑指定的测试文件**：`.venv/bin/pytest -q py/tests/<file>.py`，从仓库根目录执行；不得运行不带路径的 `pytest`。
- **标注文件绝不放在 `data/frames/<stem>/` 里面**：`vc_extract --overwrite` 会整个删除那个目录。命令行对 `--labels` / `--review` / `--gold` 做强制检查。
- 老师模型批量运行每批不超过 15 分钟；本计划的样例运行只取每 10 帧 1 帧（59 帧，约 2.5 分钟）。
- 老师模型只用 MIT / Apache-2.0 / BSD 许可的实现与权重。
- 坐标一律为原图像素坐标 `(x1, y1, x2, y2)`；检测记录的 JSON 字段固定为规格 §3.2 的 `video, frame, pts_ms, cls, x1, y1, x2, y2, score, src`；YOLO 类别号 `face=0`、`plate=1`。
- 提交信息用 conventional commits，不加署名行。在分支 `feat/py-labeling` 上工作。

### 与规格的偏差（原型阶段决定，交付时告知用户）

1. **不实现切块（规格 §5.4）**：RetinaFace 整帧推理在 1080p 上实测 0.3 秒/帧，不需要切块备用方案。4K 若出现内存或耗时问题再补。
2. **阈值按原型实测调整**：YuNet 用 0.5（0.3 时楼面、招牌误检很多）；OWLv2 用 0.1（真车牌 0.55–0.75，误检 0.10–0.14）。“低分框进复核”的区间按类别分开：人脸 0.3–0.5，车牌 0.1–0.3。全部可在 `configs/labeling.json` 调整。
3. **OWLv2 只在宽度 ≥ 画面 3% 的车辆上找车牌**：必须遮住的车牌（4K 下宽 ≥ 48px，即画面 1.25%）所在的车辆至少约占画面 4%，远处小车上的车牌不需要遮。这把车多场景的耗时从约 6.6 秒/帧降到 2 秒左右。
4. **按视频划分数据集**（规格 §5.5）放到计划 3 实现（训练时才用得到）；本计划只在标注规范里写明规则。
5. **X-AnyLabeling 的安装不在本计划内**：它是 GPL 图形工具，安装需另行征得同意；本计划只负责与它交换的 LabelMe JSON 文件。
6. 老师原始输出与融合结果分别存为 `raw.jsonl` / `merged.jsonl`，记录格式与规格 §3.2 的 `detections.jsonl` 完全相同。

## 原型实测数据（2026-10-17，sg_drive_600s，1080p）

| 项目 | 结果 |
|---|---|
| RetinaFace 整帧 | 0.3 秒/帧；近处行人脸（12–33px）得分 0.9–1.0 |
| YuNet | 0.04 秒/帧；与 RetinaFace 互为补充，但误检分数可达 0.73 |
| COCO Faster R-CNN v2 | 0.3 秒/帧 |
| OWLv2（车辆裁图） | 每批约 0.45 秒；样例帧 006005 红车车牌 0.545 |
| teach 端到端 | 约 2.4 秒/帧（每 30 帧取 1 帧，20 帧 53 秒） |

## 文件结构

```
configs/labeling.json              LabelingConfig 默认值（与代码默认值一致，测试保证）
docs/labeling-guide.md             类别定义、目录约定、复核流程、金标准规则
py/
  pyproject.toml                   pytest 配置（pythonpath、models 标记）
  requirements.txt                 固定版本的依赖
  README.md                        环境、权重下载与校验、命令用法
  vc_label/
    __init__.py
    __main__.py                    python -m vc_label
    paths.py                       模型路径、离线环境变量
    contracts.py                   Box、FrameRef、Detection、frames.csv / jsonl / YOLO 读写
    boxes.py                       IoU、NMS、加权框融合
    config.py                      LabelingConfig（JSON 覆盖）
    rules.py                       规则标记、时间补洞
    review.py                      抽帧目录防护、LabelMe 导出/导入、复核队列
    pipeline.py                    teach / fuse / export-review 流程
    cli.py                         命令行
    teachers/
      __init__.py
      retinaface_net.py            RetinaFace-R50 网络（移植自 biubug6，MIT）
      retinaface.py                先验框、解码、RetinaFaceTeacher
      yunet.py                     YuNetTeacher
      coco.py                      CocoTeacher（行人、车辆）
      owlv2.py                     Owlv2PlateTeacher（车辆裁图内找车牌）
  tests/
    conftest.py                    路径常量、离线环境、models 用例跳过、共享夹具
    test_contracts.py  test_boxes.py  test_config.py  test_rules.py  test_review.py
    test_face_teachers.py  test_plate_teachers.py  test_pipeline.py  test_cli.py
```

运行时数据（全部在被 git 忽略的 `data/` 下，且在抽帧目录之外）：

```
data/labels/<stem>/   raw.jsonl  aux.jsonl  processed.txt  merged.jsonl  flags.csv  yolo/
data/review/<stem>/   <帧号>.json（LabelMe）  review_queue.csv
data/gold/<stem>/     yolo/  gold_meta.csv
```

---

## 开工前准备（由主会话执行）

本机在写计划时已经装好依赖、下好权重。执行前只做核对，**不联网**。

- [ ] **Step 1: 核对依赖版本**

Run: `.venv/bin/pip freeze | grep -iE "^(torch|torchvision|opencv-python-headless|transformers|numpy|pillow|pytest)=="`
Expected（7 行）：`numpy==2.5.4`、`opencv-python-headless==5.0.0.93`、`pillow==13.0.0`、`pytest==9.1.1`、`torch==2.14.1`、`torchvision==0.29.1`、`transformers==5.20.1`

- [ ] **Step 2: 核对权重**

Run:
```bash
shasum -a 256 ~/.cache/video-compliance/models/Resnet50_Final.pth ~/.cache/video-compliance/models/face_detection_yunet_2023mar.onnx ~/.cache/video-compliance/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
```
Expected：
```
6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d  …/Resnet50_Final.pth
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  …/face_detection_yunet_2023mar.onnx
dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf  …/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
```
以及 `ls ~/.cache/video-compliance/hf/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d/model.safetensors` 存在。任何一项不符：停下来告诉用户，不要自行下载。

- [ ] **Step 3: 核对样例帧并建分支**

```bash
ls data/frames/sg_drive_600s/006005.jpg data/frames/sg_drive_600s/frames.csv
git checkout -b feat/py-labeling
```

---

### Task 1: 包骨架与数据契约

**Files:**
- Create: `py/pyproject.toml`
- Create: `py/requirements.txt`
- Create: `py/vc_label/__init__.py`
- Create: `py/vc_label/paths.py`
- Create: `py/vc_label/contracts.py`
- Create: `py/tests/conftest.py`
- Test: `py/tests/test_contracts.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `vc_label.paths`：`CACHE: Path`、`MODELS: Path`、`RETINAFACE_WEIGHTS: Path`、`YUNET_MODEL: Path`、`offline_model_env() -> None`
  - `vc_label.contracts`：`LABEL_CLASSES = ("face", "plate")`；`@dataclass(frozen=True) Box(cls: str, x1, y1, x2, y2: float, score: float = 1.0, src: str = "")`，属性 `width`、`height`、`area`、`center`，方法 `clipped(width, height) -> Box`、`shifted(dx, dy) -> Box`；`@dataclass(frozen=True) FrameRef(video: str, frame: int, pts_ms: float, image: Path)`；`read_frames_csv(frames_dir: Path) -> list[FrameRef]`；`@dataclass(frozen=True) Detection(video, frame, pts_ms, box)`，`to_record() -> dict`、`from_record(dict) -> Detection`；`write_jsonl(path, detections, append=False)`；`read_jsonl(path) -> Iterator[Detection]`（文件不存在时为空）；`to_yolo_lines(boxes, width, height) -> list[str]`（非标注类别抛 `ValueError`）；`from_yolo_lines(lines, width, height, src="") -> list[Box]`
  - `py/tests/conftest.py`：`REPO_ROOT`、`SAMPLE_FRAMES`；`models` 标记的用例在权重或样例帧缺失时自动跳过

- [ ] **Step 1: 写配置文件和失败的测试**

`py/pyproject.toml`：

```toml
[project]
name = "vc-label"
version = "0.1.0"
description = "Teacher pre-labelling, rule flags and human review round trip for dashcam redaction"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
    "models: needs the cached teacher weights and the sg_drive_600s sample frames",
]
```

`py/requirements.txt`：

```
numpy==2.5.4
opencv-python-headless==5.0.0.93
pillow==13.0.0
pytest==9.1.1
torch==2.14.1
torchvision==0.29.1
transformers==5.20.1
```

`py/vc_label/__init__.py`：

```python
"""Teacher pre-labelling, rule flags and human review round trip (spec §5)."""
```

`py/tests/conftest.py`：

```python
from pathlib import Path

import pytest

from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FRAMES = REPO_ROOT / "data" / "frames" / "sg_drive_600s"

offline_model_env()


def pytest_collection_modifyitems(items):
    missing = [p for p in (RETINAFACE_WEIGHTS, YUNET_MODEL, SAMPLE_FRAMES / "006005.jpg") if not p.exists()]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"missing {', '.join(str(p) for p in missing)}")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)
```

`py/tests/test_contracts.py`：

```python
from pathlib import Path

import pytest

from vc_label.contracts import (
    Box,
    Detection,
    from_yolo_lines,
    read_frames_csv,
    read_jsonl,
    to_yolo_lines,
    write_jsonl,
)


def test_box_geometry():
    b = Box("face", 10, 20, 40, 80)
    assert (b.width, b.height, b.area, b.center) == (30, 60, 1800, (25, 50))


def test_box_clipped_and_shifted():
    b = Box("plate", -5, 10, 120, 50, score=0.7, src="t")
    assert b.clipped(100, 40) == Box("plate", 0, 10, 100, 40, score=0.7, src="t")
    assert b.shifted(5, -10) == Box("plate", 0, 0, 125, 40, score=0.7, src="t")


def test_read_frames_csv(tmp_path):
    (tmp_path / "frames.csv").write_text(
        "video,frame,pts_ms,reason,motion_score\n"
        "trip.mp4,0,0.0,base,0.0000\n"
        "trip.mp4,12,200.2,motion,0.0610\n"
    )
    refs = read_frames_csv(tmp_path)
    assert [(r.video, r.frame, r.pts_ms) for r in refs] == [("trip.mp4", 0, 0.0), ("trip.mp4", 12, 200.2)]
    assert refs[1].image == tmp_path / "000012.jpg"


def test_jsonl_round_trip_and_append(tmp_path):
    path = tmp_path / "raw.jsonl"
    first = Detection("trip.mp4", 3, 50.04, Box("face", 1.04, 2, 3, 4, score=0.91234, src="teacher_yunet"))
    second = Detection("trip.mp4", 4, 66.7, Box("plate", 5, 6, 7, 8, score=0.5, src="teacher_owlv2"))
    write_jsonl(path, [first])
    write_jsonl(path, [second], append=True)
    back = list(read_jsonl(path))
    assert back[0] == Detection("trip.mp4", 3, 50.0, Box("face", 1.0, 2, 3, 4, score=0.9123, src="teacher_yunet"))
    assert back[1] == second


def test_read_jsonl_of_missing_file_is_empty(tmp_path):
    assert list(read_jsonl(tmp_path / "nope.jsonl")) == []


def test_yolo_round_trip():
    boxes = [Box("face", 100, 50, 140, 100), Box("plate", 0, 0, 1920, 1080)]
    lines = to_yolo_lines(boxes, 1920, 1080)
    assert lines[0] == "0 0.062500 0.069444 0.020833 0.046296"
    assert lines[1] == "1 0.500000 0.500000 1.000000 1.000000"
    back = from_yolo_lines(lines, 1920, 1080)
    assert [b.cls for b in back] == ["face", "plate"]
    assert back[0].x1 == pytest.approx(100, abs=0.01) and back[0].y2 == pytest.approx(100, abs=0.01)


def test_yolo_rejects_auxiliary_classes():
    with pytest.raises(ValueError, match="vehicle"):
        to_yolo_lines([Box("vehicle", 0, 0, 10, 10)], 100, 100)


def test_frame_image_paths_are_six_digit_jpgs(tmp_path):
    (tmp_path / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\nv.mp4,7,1.0,base,0\n")
    assert read_frames_csv(tmp_path)[0].image.name == "000007.jpg"
    assert isinstance(read_frames_csv(tmp_path)[0].image, Path)
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_contracts.py`
Expected: FAIL，收集阶段报 `ModuleNotFoundError: No module named 'vc_label.paths'`

- [ ] **Step 3: 写实现**

`py/vc_label/paths.py`：

```python
"""Model locations in the cache shared by every checkout, and the offline switches."""

from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("VC_CACHE", Path.home() / ".cache" / "video-compliance"))
MODELS = CACHE / "models"
RETINAFACE_WEIGHTS = MODELS / "Resnet50_Final.pth"
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"


def offline_model_env() -> None:
    """Points Hugging Face / torch at the shared cache and forbids downloads (metered link).

    Must run before transformers or torchvision load weights.
    """
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
```

`py/vc_label/contracts.py`：

```python
"""Shared data contracts (spec §3.2): boxes in original pixel coordinates."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

LABEL_CLASSES = ("face", "plate")


@dataclass(frozen=True)
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    src: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, width: int, height: int) -> Box:
        return replace(
            self,
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
        )

    def shifted(self, dx: float, dy: float) -> Box:
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


@dataclass(frozen=True)
class FrameRef:
    video: str
    frame: int
    pts_ms: float
    image: Path


def read_frames_csv(frames_dir: Path) -> list[FrameRef]:
    """Rows of <frames_dir>/frames.csv (written by vc_extract), in file order."""
    with (frames_dir / "frames.csv").open(newline="") as f:
        return [
            FrameRef(
                video=row["video"],
                frame=int(row["frame"]),
                pts_ms=float(row["pts_ms"]),
                image=frames_dir / f"{int(row['frame']):06d}.jpg",
            )
            for row in csv.DictReader(f)
        ]


@dataclass(frozen=True)
class Detection:
    video: str
    frame: int
    pts_ms: float
    box: Box

    def to_record(self) -> dict:
        b = self.box
        return {
            "video": self.video,
            "frame": self.frame,
            "pts_ms": round(self.pts_ms, 1),
            "cls": b.cls,
            "x1": round(b.x1, 1),
            "y1": round(b.y1, 1),
            "x2": round(b.x2, 1),
            "y2": round(b.y2, 1),
            "score": round(b.score, 4),
            "src": b.src,
        }

    @staticmethod
    def from_record(record: dict) -> Detection:
        box = Box(
            record["cls"],
            float(record["x1"]),
            float(record["y1"]),
            float(record["x2"]),
            float(record["y2"]),
            score=float(record["score"]),
            src=record["src"],
        )
        return Detection(record["video"], int(record["frame"]), float(record["pts_ms"]), box)


def write_jsonl(path: Path, detections: Iterable[Detection], append: bool = False) -> None:
    with path.open("a" if append else "w") as f:
        for d in detections:
            f.write(json.dumps(d.to_record(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Detection]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield Detection.from_record(json.loads(line))


def to_yolo_lines(boxes: Iterable[Box], width: int, height: int) -> list[str]:
    """YOLO txt rows `<class> <cx> <cy> <w> <h>` normalised to [0, 1], label classes only."""
    lines = []
    for b in boxes:
        if b.cls not in LABEL_CLASSES:
            raise ValueError(f"not a label class: {b.cls}")
        cx, cy = b.center
        lines.append(
            f"{LABEL_CLASSES.index(b.cls)} {cx / width:.6f} {cy / height:.6f} "
            f"{b.width / width:.6f} {b.height / height:.6f}"
        )
    return lines


def from_yolo_lines(lines: Iterable[str], width: int, height: int, src: str = "") -> list[Box]:
    boxes = []
    for line in lines:
        if not line.strip():
            continue
        cls_id, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append(
            Box(LABEL_CLASSES[int(cls_id)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, src=src)
        )
    return boxes
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_contracts.py`
Expected: `8 passed`，无警告

- [ ] **Step 5: 提交**

```bash
git add py/pyproject.toml py/requirements.txt py/vc_label/__init__.py py/vc_label/paths.py py/vc_label/contracts.py py/tests/conftest.py py/tests/test_contracts.py
git commit -m "feat: add vc_label package skeleton and data contracts"
```

---

### Task 2: 框运算与加权融合

**Files:**
- Create: `py/vc_label/boxes.py`
- Test: `py/tests/test_boxes.py`

**Interfaces:**
- Consumes: `Box`（Task 1）
- Produces（`vc_label.boxes`）：`SOURCE_SEPARATOR = "+"`；`intersection_area(a, b) -> float`；`iou(a, b) -> float`；`contains_point(box, x, y) -> bool`（含边界）；`sources(box) -> set[str]`（按 `+` 拆分 `src`）；`nms(boxes, iou_threshold) -> list[Box]`（按类别、分数从高到低贪心保留）；`fuse(boxes, iou_threshold) -> list[Box]`（按类别的加权框融合：坐标按分数加权平均，分数取成员最大值，`src` 为成员来源排序后用 `+` 连接）

- [ ] **Step 1: 写失败的测试**

`py/tests/test_boxes.py`：

```python
import pytest

from vc_label.boxes import contains_point, fuse, iou, nms, sources
from vc_label.contracts import Box


def test_iou():
    a = Box("face", 0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, Box("face", 20, 20, 30, 30)) == 0.0
    assert iou(a, Box("face", 5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_contains_point():
    assert contains_point(Box("vehicle", 0, 0, 10, 10), 10, 5)
    assert not contains_point(Box("vehicle", 0, 0, 10, 10), 10.1, 5)


def test_nms_keeps_best_box_per_class():
    boxes = [
        Box("vehicle", 0, 0, 100, 100, score=0.6),
        Box("vehicle", 2, 2, 100, 100, score=0.9),
        Box("person", 0, 0, 100, 100, score=0.5),
    ]
    kept = nms(boxes, 0.6)
    assert [(b.cls, b.score) for b in kept] == [("vehicle", 0.9), ("person", 0.5)]


def test_fuse_merges_overlapping_sources_with_score_weighted_coordinates():
    retina = Box("face", 0, 0, 10, 10, score=0.9, src="teacher_retinaface")
    yunet = Box("face", 1, 1, 11, 11, score=0.6, src="teacher_yunet")

    [fused] = fuse([yunet, retina], 0.5)

    assert fused.x1 == pytest.approx((0 * 0.9 + 1 * 0.6) / 1.5)
    assert fused.x2 == pytest.approx((10 * 0.9 + 11 * 0.6) / 1.5)
    assert fused.score == 0.9
    assert fused.src == "teacher_retinaface+teacher_yunet"
    assert sources(fused) == {"teacher_retinaface", "teacher_yunet"}


def test_fuse_keeps_classes_and_distant_boxes_apart():
    boxes = [
        Box("face", 0, 0, 10, 10, score=0.9, src="a"),
        Box("plate", 0, 0, 10, 10, score=0.8, src="b"),
        Box("face", 50, 50, 60, 60, score=0.7, src="a"),
    ]
    fused = fuse(boxes, 0.5)
    assert sorted((b.cls, b.x1) for b in fused) == [("face", 0), ("face", 50), ("plate", 0)]


def test_fuse_of_nothing_is_nothing():
    assert fuse([], 0.5) == []
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_boxes.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.boxes'`

- [ ] **Step 3: 写实现**

`py/vc_label/boxes.py`：

```python
"""Box geometry: IoU, NMS and weighted box fusion across teacher sources."""

from __future__ import annotations

from collections import defaultdict

from vc_label.contracts import Box

SOURCE_SEPARATOR = "+"


def intersection_area(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def contains_point(box: Box, x: float, y: float) -> bool:
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def sources(box: Box) -> set[str]:
    """Teacher sources that contributed to a (possibly fused) box."""
    return set(box.src.split(SOURCE_SEPARATOR)) if box.src else set()


def nms(boxes: list[Box], iou_threshold: float) -> list[Box]:
    """Greedy per-class non-maximum suppression, highest score first."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: x.score, reverse=True):
        if all(k.cls != b.cls or iou(k, b) < iou_threshold for k in kept):
            kept.append(b)
    return kept


def _weighted(members: list[Box]) -> Box:
    weights = [m.score for m in members]
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * len(members), float(len(members))
    coords = [
        sum(w * getattr(m, k) for w, m in zip(weights, members)) / total
        for k in ("x1", "y1", "x2", "y2")
    ]
    return Box(
        members[0].cls,
        *coords,
        score=max(m.score for m in members),
        src=SOURCE_SEPARATOR.join(sorted(set().union(*(sources(m) for m in members)))),
    )


def fuse(boxes: list[Box], iou_threshold: float) -> list[Box]:
    """Weighted box fusion per class.

    Boxes are visited by descending score; each joins the first cluster whose fused box overlaps it
    by at least `iou_threshold`, otherwise it starts a new cluster. A fused box has score-weighted
    coordinates, the maximum member score (recall first: no down-weighting of single-source boxes)
    and the union of member sources joined with '+'.
    """
    by_class: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        by_class[b.cls].append(b)
    fused: list[Box] = []
    for cls in sorted(by_class):
        clusters: list[list[Box]] = []
        centers: list[Box] = []
        for b in sorted(by_class[cls], key=lambda x: x.score, reverse=True):
            best = max(range(len(centers)), key=lambda i: iou(centers[i], b), default=None)
            if best is not None and iou(centers[best], b) >= iou_threshold:
                clusters[best].append(b)
                centers[best] = _weighted(clusters[best])
            else:
                clusters.append([b])
                centers.append(_weighted([b]))
        fused.extend(centers)
    return fused
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_boxes.py`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/boxes.py py/tests/test_boxes.py
git commit -m "feat: add box IoU, NMS and weighted box fusion"
```

---

### Task 3: 标注配置

**Files:**
- Create: `py/vc_label/config.py`
- Create: `configs/labeling.json`
- Test: `py/tests/test_config.py`

**Interfaces:**
- Consumes: `REPO_ROOT`（`py/tests/conftest.py`，Task 1）
- Produces（`vc_label.config`）：`@dataclass(frozen=True) LabelingConfig`，字段与默认值见下方代码；方法 `low_score_band(cls) -> tuple[float, float]`、`LabelingConfig.from_json(path) -> LabelingConfig`（未知键抛 `ValueError`，JSON 数组转 tuple）、`to_json() -> str`

- [ ] **Step 1: 写失败的测试**

`py/tests/test_config.py`：

```python
import json

import pytest

from conftest import REPO_ROOT
from vc_label.config import LabelingConfig


def test_low_score_bands_are_per_class():
    cfg = LabelingConfig()
    assert cfg.low_score_band("face") == (0.3, 0.5)
    assert cfg.low_score_band("plate") == (0.1, 0.3)


def test_from_json_overrides_and_converts_lists(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_score": 0.6, "plate_low_score": [0.15, 0.35]}))
    cfg = LabelingConfig.from_json(path)
    assert cfg.yunet_score == 0.6
    assert cfg.plate_low_score == (0.15, 0.35)
    assert cfg.retinaface_score == LabelingConfig().retinaface_score


def test_from_json_rejects_unknown_keys(tmp_path):
    path = tmp_path / "labeling.json"
    path.write_text(json.dumps({"yunet_scor": 0.6}))
    with pytest.raises(ValueError, match="yunet_scor"):
        LabelingConfig.from_json(path)


def test_repository_config_matches_defaults():
    assert LabelingConfig.from_json(REPO_ROOT / "configs" / "labeling.json") == LabelingConfig()
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_config.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.config'`

- [ ] **Step 3: 写实现和配置文件**

`py/vc_label/config.py`：

```python
"""Labeling thresholds. Defaults come from the 2026-10-17 prototype on sg_drive_600s (1080p60)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class LabelingConfig:
    # Teacher score thresholds (recall first: low, review catches the noise).
    retinaface_score: float = 0.3
    yunet_score: float = 0.5
    coco_score: float = 0.3
    owlv2_score: float = 0.1
    # OWLv2 only searches vehicles at least this wide (fraction of image width): a must-mask
    # plate (>= 48 px at 4K, 1.25 % of the width) sits on a vehicle >= ~4 % of the width.
    plate_search_min_vehicle_frac: float = 0.03
    fuse_iou: float = 0.5
    # Boxes scoring inside [low, high) for their class go to the review queue.
    face_low_score: tuple[float, float] = (0.3, 0.5)
    plate_low_score: tuple[float, float] = (0.1, 0.3)
    # "Should have but none": large persons need a face in their top part, large vehicles a plate.
    min_person_height_frac: float = 0.12
    face_region_frac: float = 0.35
    min_vehicle_width_frac: float = 0.05
    # Temporal hole filling between neighbouring labelled frames.
    temporal_max_gap_ms: float = 500.0
    temporal_match_iou: float = 0.3
    temporal_present_iou: float = 0.1

    def low_score_band(self, cls: str) -> tuple[float, float]:
        return {"face": self.face_low_score, "plate": self.plate_low_score}[cls]

    @staticmethod
    def from_json(path: Path) -> LabelingConfig:
        """Defaults overridden by the keys in a JSON object; unknown keys are an error."""
        overrides = json.loads(path.read_text())
        known = {f.name for f in fields(LabelingConfig)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown labeling config keys in {path}: {sorted(unknown)}")
        tuples = {k: tuple(v) for k, v in overrides.items() if isinstance(v, list)}
        return replace(LabelingConfig(), **{**overrides, **tuples})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"
```

`configs/labeling.json`（即 `LabelingConfig().to_json()` 的输出）：

```json
{
  "retinaface_score": 0.3,
  "yunet_score": 0.5,
  "coco_score": 0.3,
  "owlv2_score": 0.1,
  "plate_search_min_vehicle_frac": 0.03,
  "fuse_iou": 0.5,
  "face_low_score": [
    0.3,
    0.5
  ],
  "plate_low_score": [
    0.1,
    0.3
  ],
  "min_person_height_frac": 0.12,
  "face_region_frac": 0.35,
  "min_vehicle_width_frac": 0.05,
  "temporal_max_gap_ms": 500.0,
  "temporal_match_iou": 0.3,
  "temporal_present_iou": 0.1
}
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_config.py`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/config.py configs/labeling.json py/tests/test_config.py
git commit -m "feat: add labeling thresholds config"
```

---

### Task 4: 规则标记与时间补洞

**Files:**
- Create: `py/vc_label/rules.py`
- Test: `py/tests/test_rules.py`

**Interfaces:**
- Consumes: `Box`、`FrameRef`（Task 1）；`contains_point`、`iou`、`nms`、`sources`（Task 2）；`LabelingConfig`（Task 3）
- Produces（`vc_label.rules`）：`FACE_SOURCES = frozenset({"teacher_retinaface", "teacher_yunet"})`、`TEMPORAL_SOURCE = "temporal_fill"`、`AUX_DEDUP_IOU = 0.6`；`@dataclass(frozen=True) Flag(video: str, frame: int, rule: str, box: Box, detail: str = "")`；`flag_frame(ref, merged, aux, width, height, cfg) -> list[Flag]`；`temporal_fill(sequence: list[tuple[FrameRef, list[Box]]], cfg) -> tuple[list[list[Box]], list[Flag]]`
- 规则名（写进 flags.csv 与复核队列）：`face_single_source`、`low_score`、`person_without_face`、`vehicle_without_plate`、`temporal_fill`

规则约定：
- `face_single_source`：融合后的人脸框，来源没有同时包含 RetinaFace 和 YuNet。
- `low_score`：分数落在该类别的低分区间 `[low, high)`。
- `person_without_face`：去重后的行人框高度 ≥ `min_person_height_frac × 图高`，其上部 `face_region_frac` 区域内没有任何人脸中心点。
- `vehicle_without_plate`：去重后（COCO 常把同一辆车同时报成 car 和 truck，IoU ≥ 0.6 只留一个）的车辆框宽度 ≥ `min_vehicle_width_frac × 图宽`，框内没有任何车牌中心点。
- `temporal_fill`：相邻三帧（时间间隔都不超过 `temporal_max_gap_ms`）中，前后两帧有同类且 IoU ≥ `temporal_match_iou` 的框，而当前帧没有与插值框 IoU ≥ `temporal_present_iou` 的同类框时，补一个线性插值框（分数取前后较小值，来源 `temporal_fill`），并为它记一条标记。补出来的框不参与其他规则。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_rules.py`：

```python
from pathlib import Path

from vc_label.config import LabelingConfig
from vc_label.contracts import Box, FrameRef
from vc_label.rules import flag_frame, temporal_fill

CFG = LabelingConfig()
W, H = 1920, 1080
BOTH = "teacher_retinaface+teacher_yunet"


def ref(frame: int, pts_ms: float) -> FrameRef:
    return FrameRef("trip.mp4", frame, pts_ms, Path(f"{frame:06d}.jpg"))


def rules(merged, aux=()):
    return sorted(f.rule for f in flag_frame(ref(1, 0.0), list(merged), list(aux), W, H, CFG))


def test_face_seen_by_both_teachers_is_not_flagged():
    assert rules([Box("face", 100, 100, 130, 140, score=0.9, src=BOTH)]) == []


def test_face_seen_by_one_teacher_is_flagged():
    assert rules([Box("face", 100, 100, 130, 140, score=0.9, src="teacher_yunet")]) == ["face_single_source"]


def test_low_score_bands_differ_per_class():
    assert rules([Box("face", 0, 0, 20, 20, score=0.4, src=BOTH)]) == ["low_score"]
    assert rules([Box("plate", 0, 0, 20, 8, score=0.2, src="teacher_owlv2")]) == ["low_score"]
    assert rules([Box("plate", 0, 0, 20, 8, score=0.4, src="teacher_owlv2")]) == []


def test_large_person_needs_a_face_in_the_head_region():
    person = Box("person", 200, 500, 300, 900, src="teacher_coco:person")  # 400 px >= 12 % of 1080
    head_face = Box("face", 240, 510, 260, 540, score=0.9, src=BOTH)
    belly_face = Box("face", 240, 800, 260, 830, score=0.9, src=BOTH)
    assert rules([], [person]) == ["person_without_face"]
    assert rules([head_face], [person]) == []
    assert rules([belly_face], [person]) == ["person_without_face"]


def test_small_person_is_not_flagged():
    assert rules([], [Box("person", 0, 0, 20, 100)]) == []


def test_large_vehicle_without_plate_is_flagged_once():
    car = Box("vehicle", 1000, 600, 1300, 800, score=0.99, src="teacher_coco:car")
    truck = Box("vehicle", 1005, 600, 1300, 805, score=0.8, src="teacher_coco:truck")
    plate = Box("plate", 1100, 750, 1160, 765, score=0.6, src="teacher_owlv2")
    assert rules([], [car, truck]) == ["vehicle_without_plate"]
    assert rules([plate], [car, truck]) == []
    assert rules([], [Box("vehicle", 0, 0, 50, 40)]) == []


def test_temporal_fill_interpolates_a_missing_box():
    prev = Box("plate", 100, 100, 160, 120, score=0.6, src="teacher_owlv2")
    nxt = Box("plate", 110, 100, 170, 120, score=0.5, src="teacher_owlv2")
    sequence = [(ref(0, 0.0), [prev]), (ref(12, 200.0), []), (ref(24, 400.0), [nxt])]

    filled, flags = temporal_fill(sequence, CFG)

    [box] = filled[1]
    assert (box.x1, box.x2, box.score, box.src) == (105, 165, 0.5, "temporal_fill")
    assert [(f.frame, f.rule) for f in flags] == [(12, "temporal_fill")]
    assert filled[0] == [prev] and filled[2] == [nxt]


def test_temporal_fill_skips_present_boxes_long_gaps_and_unmatched_boxes():
    a = Box("face", 100, 100, 130, 140, score=0.9, src=BOTH)
    far = Box("face", 900, 100, 930, 140, score=0.9, src=BOTH)
    present = [(ref(0, 0.0), [a]), (ref(12, 200.0), [a]), (ref(24, 400.0), [a])]
    long_gap = [(ref(0, 0.0), [a]), (ref(60, 1000.0), []), (ref(120, 2000.0), [a])]
    unmatched = [(ref(0, 0.0), [a]), (ref(12, 200.0), []), (ref(24, 400.0), [far])]
    for sequence in (present, long_gap, unmatched):
        _, flags = temporal_fill(sequence, CFG)
        assert flags == []
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_rules.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.rules'`

- [ ] **Step 3: 写实现**

`py/vc_label/rules.py`：

```python
"""Rule-based flags that route likely label errors to human review (spec §5.3, no LLMs)."""

from __future__ import annotations

from dataclasses import dataclass

from vc_label.boxes import contains_point, iou, nms, sources
from vc_label.config import LabelingConfig
from vc_label.contracts import Box, FrameRef

FACE_SOURCES = frozenset({"teacher_retinaface", "teacher_yunet"})
TEMPORAL_SOURCE = "temporal_fill"
AUX_DEDUP_IOU = 0.6


@dataclass(frozen=True)
class Flag:
    video: str
    frame: int
    rule: str
    box: Box
    detail: str = ""


def flag_frame(
    ref: FrameRef, merged: list[Box], aux: list[Box], width: int, height: int, cfg: LabelingConfig
) -> list[Flag]:
    """Per-frame rules: single-source faces, low scores, persons without faces, vehicles without plates."""
    flags: list[Flag] = []
    for b in merged:
        if b.src == TEMPORAL_SOURCE:
            continue
        if b.cls == "face" and not FACE_SOURCES <= sources(b):
            flags.append(Flag(ref.video, ref.frame, "face_single_source", b, f"only {b.src}"))
        low, high = cfg.low_score_band(b.cls)
        if low <= b.score < high:
            flags.append(Flag(ref.video, ref.frame, "low_score", b, f"score {b.score:.2f}"))
    faces = [b for b in merged if b.cls == "face"]
    plates = [b for b in merged if b.cls == "plate"]
    # COCO often reports one vehicle as both car and truck; flag it once.
    aux = nms(aux, AUX_DEDUP_IOU)
    for person in (a for a in aux if a.cls == "person"):
        if person.height < cfg.min_person_height_frac * height:
            continue
        region = Box("person", person.x1, person.y1, person.x2, person.y1 + cfg.face_region_frac * person.height)
        if not any(contains_point(region, *f.center) for f in faces):
            flags.append(Flag(ref.video, ref.frame, "person_without_face", person))
    for vehicle in (a for a in aux if a.cls == "vehicle"):
        if vehicle.width < cfg.min_vehicle_width_frac * width:
            continue
        if not any(contains_point(vehicle, *p.center) for p in plates):
            flags.append(Flag(ref.video, ref.frame, "vehicle_without_plate", vehicle))
    return flags


def _interpolate(a: Box, b: Box, t: float) -> tuple[float, float, float, float]:
    return tuple(getattr(a, k) + (getattr(b, k) - getattr(a, k)) * t for k in ("x1", "y1", "x2", "y2"))


def temporal_fill(
    sequence: list[tuple[FrameRef, list[Box]]], cfg: LabelingConfig
) -> tuple[list[list[Box]], list[Flag]]:
    """Adds a box at frame t when a matching box exists at t-1 and t+1 but nothing similar at t.

    `sequence` is ordered by time. Neighbours more than `temporal_max_gap_ms` apart are not used.
    A match pairs boxes of the same class with IoU >= `temporal_match_iou`; the filled box is the
    linear interpolation at t's timestamp, scored min(prev, next) with src "temporal_fill".
    Returns the per-frame box lists (originals plus fills) and one flag per fill.
    """
    out = [list(boxes) for _, boxes in sequence]
    flags: list[Flag] = []
    for i in range(1, len(sequence) - 1):
        (prev_ref, prev), (ref, _), (next_ref, nxt) = sequence[i - 1], sequence[i], sequence[i + 1]
        gap = cfg.temporal_max_gap_ms
        if ref.pts_ms - prev_ref.pts_ms > gap or next_ref.pts_ms - ref.pts_ms > gap:
            continue
        t = (ref.pts_ms - prev_ref.pts_ms) / (next_ref.pts_ms - prev_ref.pts_ms)
        for p in (b for b in prev if b.src != TEMPORAL_SOURCE):
            match = max(
                (n for n in nxt if n.cls == p.cls and n.src != TEMPORAL_SOURCE),
                key=lambda n: iou(p, n),
                default=None,
            )
            if match is None or iou(p, match) < cfg.temporal_match_iou:
                continue
            filled = Box(p.cls, *_interpolate(p, match, t), score=min(p.score, match.score), src=TEMPORAL_SOURCE)
            if any(c.cls == p.cls and iou(c, filled) >= cfg.temporal_present_iou for c in out[i]):
                continue
            out[i].append(filled)
            flags.append(Flag(ref.video, ref.frame, "temporal_fill", filled, f"between frames {prev_ref.frame} and {next_ref.frame}"))
    return out, flags
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_rules.py`
Expected: `8 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/rules.py py/tests/test_rules.py
git commit -m "feat: add rule-based review flags and temporal hole filling"
```

---

### Task 5: 人工复核往返（LabelMe）与抽帧目录防护

**Files:**
- Create: `py/vc_label/review.py`
- Test: `py/tests/test_review.py`

**Interfaces:**
- Consumes: `Box`、`FrameRef`、`LABEL_CLASSES`、`to_yolo_lines`、`read_frames_csv`（Task 1）；`Flag`（Task 4）
- Produces（`vc_label.review`）：`SCENE_FLAGS = ("day", "night", "rain", "tunnel", "city", "highway")`、`LABELME_VERSION = "5.4.1"`；`ensure_outside(frames_dir, path) -> None`（`path` 等于或位于抽帧目录内时抛 `ValueError`，消息含 `inside the frames directory`）；`labelme_document(ref, boxes, flags, review_dir, width, height) -> dict`；`write_review_queue(path, flags) -> None`（列 `video,frame,flags,rules,where`，标记多的帧在前）；`read_labelme_boxes(path) -> tuple[list[Box], list[str], int, int]`；`import_review(review_dir, refs, gold_dir) -> int`（写 `gold_dir/yolo/<帧号>.txt` 与 `gold_dir/gold_meta.csv`）

LabelMe 约定：每帧一个 `<帧号:06d>.json`；`imagePath` 是从复核目录指向抽帧图片的相对路径；每个框的 `description` 为 `score=… src=… <规则名…>`；图片级 `flags` 是六个场景标记，人工复核时勾选。导入时只接受 `face` / `plate` 两种标签的矩形，其他一律报错，不静默丢弃。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_review.py`：

```python
import csv
import json
from pathlib import Path

import pytest

from vc_label.contracts import Box, FrameRef, read_frames_csv
from vc_label.review import ensure_outside, import_review, labelme_document, write_review_queue
from vc_label.rules import Flag


def test_ensure_outside_rejects_the_frames_dir_and_anything_below_it(tmp_path):
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the frames directory"):
        ensure_outside(frames, frames / "labels")
    with pytest.raises(ValueError):
        ensure_outside(frames, frames)
    ensure_outside(frames, tmp_path / "labels" / "trip")


def test_labelme_document_points_back_to_the_frame(tmp_path):
    frames, review = tmp_path / "frames" / "trip", tmp_path / "review" / "trip"
    ref = FrameRef("trip.mp4", 7, 116.7, frames / "000007.jpg")
    face = Box("face", 10, 20, 30, 50, score=0.42, src="teacher_yunet")

    doc = labelme_document(ref, [face], [Flag("trip.mp4", 7, "low_score", face)], review, 1920, 1080)

    assert doc["imagePath"] == "../../frames/trip/000007.jpg"
    assert (doc["imageWidth"], doc["imageHeight"], doc["imageData"]) == (1920, 1080, None)
    assert doc["flags"] == {"day": False, "night": False, "rain": False, "tunnel": False, "city": False, "highway": False}
    [shape] = doc["shapes"]
    assert shape["label"] == "face" and shape["shape_type"] == "rectangle"
    assert shape["points"] == [[10, 20], [30, 50]]
    assert shape["description"] == "score=0.42 src=teacher_yunet low_score"


def test_review_queue_lists_most_flagged_frames_first(tmp_path):
    b = Box("vehicle", 1, 2, 3, 4)
    flags = [Flag("v.mp4", 5, "low_score", b), Flag("v.mp4", 9, "low_score", b), Flag("v.mp4", 9, "vehicle_without_plate", b)]
    path = tmp_path / "review_queue.csv"

    write_review_queue(path, flags)

    rows = list(csv.reader(path.open()))
    assert rows[0] == ["video", "frame", "flags", "rules", "where"]
    assert rows[1] == ["v.mp4", "9", "2", "low_score;vehicle_without_plate", "low_score@1,2,3,4 vehicle_without_plate@1,2,3,4"]
    assert rows[2][:2] == ["v.mp4", "5"]


def _frames_dir(tmp_path: Path) -> Path:
    frames = tmp_path / "frames" / "trip"
    frames.mkdir(parents=True)
    (frames / "frames.csv").write_text("video,frame,pts_ms,reason,motion_score\ntrip.mp4,7,116.7,base,0\n")
    return frames


def _review_json(review: Path, shapes: list[dict], flags: dict) -> None:
    review.mkdir(parents=True)
    doc = {"version": "5.4.1", "flags": flags, "shapes": shapes, "imagePath": "x.jpg", "imageData": None, "imageHeight": 100, "imageWidth": 200}
    (review / "000007.json").write_text(json.dumps(doc))


def test_import_review_writes_gold_yolo_and_scene_tags(tmp_path):
    frames, review, gold = _frames_dir(tmp_path), tmp_path / "review", tmp_path / "gold"
    shapes = [
        {"label": "plate", "points": [[120, 60], [100, 50]], "shape_type": "rectangle"},
        {"label": "face", "points": [[0, 0], [20, 10]], "shape_type": "rectangle"},
    ]
    _review_json(review, shapes, {"day": True, "city": True, "night": False})

    assert import_review(review, read_frames_csv(frames), gold) == 1

    assert (gold / "yolo" / "000007.txt").read_text().splitlines() == [
        "1 0.550000 0.550000 0.100000 0.100000",
        "0 0.050000 0.050000 0.100000 0.100000",
    ]
    assert list(csv.reader((gold / "gold_meta.csv").open())) == [
        ["image", "video", "frame", "scene_tags"],
        ["000007.jpg", "trip.mp4", "7", "city;day"],
    ]


@pytest.mark.parametrize(
    "shape, message",
    [
        ({"label": "person", "points": [[0, 0], [1, 1]], "shape_type": "rectangle"}, "unknown label"),
        ({"label": "face", "points": [[0, 0], [1, 1], [2, 0]], "shape_type": "polygon"}, "not a rectangle"),
    ],
)
def test_import_review_rejects_bad_shapes(tmp_path, shape, message):
    frames, review = _frames_dir(tmp_path), tmp_path / "review"
    _review_json(review, [shape], {})

    with pytest.raises(ValueError, match=message):
        import_review(review, read_frames_csv(frames), tmp_path / "gold")
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_review.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.review'`

- [ ] **Step 3: 写实现**

`py/vc_label/review.py`：

```python
"""Human review round trip through X-AnyLabeling (LabelMe JSON), kept outside the frames dir."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from vc_label.contracts import LABEL_CLASSES, Box, FrameRef, to_yolo_lines
from vc_label.rules import Flag

SCENE_FLAGS = ("day", "night", "rain", "tunnel", "city", "highway")
LABELME_VERSION = "5.4.1"


def ensure_outside(frames_dir: Path, path: Path) -> None:
    """Refuses any output inside the frames dir: vc_extract --overwrite deletes that whole dir."""
    frames = frames_dir.resolve()
    target = path.resolve()
    if target == frames or frames in target.parents:
        raise ValueError(f"{path} is inside the frames directory {frames_dir}; write labels elsewhere")


def labelme_document(ref: FrameRef, boxes: list[Box], flags: list[Flag], review_dir: Path, width: int, height: int) -> dict:
    notes: dict[Box, list[str]] = defaultdict(list)
    for f in flags:
        notes[f.box].append(f.rule)
    shapes = [
        {
            "label": b.cls,
            "points": [[round(b.x1, 1), round(b.y1, 1)], [round(b.x2, 1), round(b.y2, 1)]],
            "group_id": None,
            "description": " ".join([f"score={b.score:.2f}", f"src={b.src}", *notes.get(b, [])]),
            "shape_type": "rectangle",
            "flags": {},
        }
        for b in boxes
    ]
    return {
        "version": LABELME_VERSION,
        "flags": {name: False for name in SCENE_FLAGS},
        "shapes": shapes,
        "imagePath": os.path.relpath(ref.image.resolve(), review_dir.resolve()),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def write_review_queue(path: Path, flags: list[Flag]) -> None:
    """One row per flagged frame, most flags first; `where` lists rule@x1,y1,x2,y2 per flag."""
    per_frame: dict[tuple[str, int], list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[(f.video, f.frame)].append(f)
    rows = sorted(per_frame.items(), key=lambda item: (-len(item[1]), item[0][1]))
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "frame", "flags", "rules", "where"])
        for (video, frame), frame_flags in rows:
            where = " ".join(
                f"{f.rule}@{f.box.x1:.0f},{f.box.y1:.0f},{f.box.x2:.0f},{f.box.y2:.0f}" for f in frame_flags
            )
            rules = ";".join(sorted({f.rule for f in frame_flags}))
            writer.writerow([video, frame, len(frame_flags), rules, where])


def read_labelme_boxes(path: Path) -> tuple[list[Box], list[str], int, int]:
    """(boxes, true scene flags, width, height). Non-rectangles or unknown labels are errors."""
    doc = json.loads(path.read_text())
    boxes = []
    for shape in doc.get("shapes", []):
        label = shape.get("label")
        if label not in LABEL_CLASSES:
            raise ValueError(f"{path}: unknown label {label!r} (expected one of {LABEL_CLASSES})")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"{path}: shape {label!r} is {shape.get('shape_type')!r}, not a rectangle")
        (xa, ya), (xb, yb) = shape["points"]
        boxes.append(Box(label, min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), src="human"))
    tags = sorted(name for name, on in doc.get("flags", {}).items() if on)
    return boxes, tags, int(doc["imageWidth"]), int(doc["imageHeight"])


def import_review(review_dir: Path, refs: list[FrameRef], gold_dir: Path) -> int:
    """Converts reviewed LabelMe JSONs to gold YOLO labels plus gold_meta.csv; returns frame count."""
    by_name = {f"{r.frame:06d}": r for r in refs}
    yolo_dir = gold_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(review_dir.glob("*.json")):
        ref = by_name.get(path.stem)
        if ref is None:
            raise ValueError(f"{path}: no frame {path.stem} in frames.csv")
        boxes, tags, width, height = read_labelme_boxes(path)
        (yolo_dir / f"{path.stem}.txt").write_text("".join(f"{line}\n" for line in to_yolo_lines(boxes, width, height)))
        rows.append([ref.image.name, ref.video, ref.frame, ";".join(tags)])
    with (gold_dir / "gold_meta.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "video", "frame", "scene_tags"])
        writer.writerows(rows)
    return len(rows)
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_review.py`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/review.py py/tests/test_review.py
git commit -m "feat: add LabelMe review round trip and frames-dir guard"
```

---

### Task 6: 人脸老师（RetinaFace、YuNet）

**Files:**
- Create: `py/vc_label/teachers/__init__.py`
- Create: `py/vc_label/teachers/retinaface_net.py`
- Create: `py/vc_label/teachers/retinaface.py`
- Create: `py/vc_label/teachers/yunet.py`
- Modify: `py/tests/conftest.py`（文件末尾追加共享夹具）
- Test: `py/tests/test_face_teachers.py`

**Interfaces:**
- Consumes: `Box`（Task 1）；`RETINAFACE_WEIGHTS`、`YUNET_MODEL`（Task 1 `paths`）；`iou`（Task 2）
- Produces：
  - `vc_label.teachers.retinaface_net`：`RetinaFaceNet`（`forward(x) -> (deltas [N, A, 4], face_prob [N, A])`）、`load_retinaface(weights_path, device) -> RetinaFaceNet`（`weights_only=True`、严格加载、去掉 `module.` 前缀）
  - `vc_label.teachers.retinaface`：`SOURCE = "teacher_retinaface"`；`prior_boxes(height, width) -> np.ndarray [A, 4]`（`cx, cy, w, h` 归一化，顺序与网络输出一致）；`decode(deltas, priors, width, height) -> np.ndarray [A, 4]`（像素 `x1, y1, x2, y2`）；`RetinaFaceTeacher(weights_path: str, device, score_threshold=0.3, nms_iou=0.4)`，`detect(bgr) -> list[Box]`
  - `vc_label.teachers.yunet`：`SOURCE = "teacher_yunet"`；`YuNetTeacher(model_path: str, score_threshold=0.3, nms_iou=0.3)`，`detect(bgr) -> list[Box]`（命令行用配置里的 0.5）
  - `py/tests/conftest.py` 追加：会话级夹具 `sample_frame`（样例帧 006005 的 BGR 数组）、`device`（MPS 可用则 MPS，否则 CPU），函数 `best_match(boxes, target) -> Box | None`

RetinaFace 要点：输入 BGR 减均值 `(104, 117, 123)`，不缩放、整帧推理；三层特征步长 8/16/32，每格两种尺寸 `(16,32)`、`(64,128)`、`(256,512)`；特征图大小取 `ceil(H/step)`，与 ResNet 的实际输出一致（1080 高时第二层为 68 行）。

- [ ] **Step 1: 追加共享夹具，写失败的测试**

在 `py/tests/conftest.py` 末尾追加：

```python


@pytest.fixture(scope="session")
def sample_frame():
    import cv2

    return cv2.imread(str(SAMPLE_FRAMES / "006005.jpg"))


@pytest.fixture(scope="session")
def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def best_match(boxes, target):
    """The box overlapping `target` most, or None."""
    from vc_label.boxes import iou

    return max(boxes, key=lambda b: iou(b, target), default=None)
```

`py/tests/test_face_teachers.py`：

```python
import numpy as np
import pytest

from conftest import best_match
from vc_label.boxes import iou
from vc_label.contracts import Box
from vc_label.teachers.retinaface import decode, prior_boxes

# Frame 006005 of the sg_drive_600s sample: a man facing the camera on the left.
SAMPLE_FACE = Box("face", 254, 704, 281, 737)


def test_prior_boxes_follow_the_network_output_order():
    priors = prior_boxes(64, 64)
    assert priors.shape == (8 * 8 * 2 + 4 * 4 * 2 + 2 * 2 * 2, 4)
    np.testing.assert_allclose(priors[0], [4 / 64, 4 / 64, 16 / 64, 16 / 64])
    np.testing.assert_allclose(priors[1], [4 / 64, 4 / 64, 32 / 64, 32 / 64])
    np.testing.assert_allclose(priors[2], [12 / 64, 4 / 64, 16 / 64, 16 / 64])


def test_prior_boxes_match_odd_feature_map_sizes():
    # 1080 / 16 = 67.5 -> 68 rows, as the ResNet strides produce.
    assert prior_boxes(1080, 1920).shape == (135 * 240 * 2 + 68 * 120 * 2 + 34 * 60 * 2, 4)


def test_decode_of_zero_deltas_returns_the_prior_boxes():
    priors = np.array([[0.5, 0.5, 0.25, 0.5]], np.float32)
    np.testing.assert_allclose(decode(np.zeros((1, 4), np.float32), priors, 200, 100), [[75, 25, 125, 75]])


@pytest.mark.models
def test_retinaface_finds_the_sample_face(sample_frame, device):
    from vc_label.paths import RETINAFACE_WEIGHTS
    from vc_label.teachers.retinaface import RetinaFaceTeacher

    best = best_match(RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device).detect(sample_frame), SAMPLE_FACE)
    assert best is not None and iou(best, SAMPLE_FACE) >= 0.5 and best.score >= 0.9
    assert best.src == "teacher_retinaface"


@pytest.mark.models
def test_yunet_finds_the_sample_face(sample_frame):
    from vc_label.paths import YUNET_MODEL
    from vc_label.teachers.yunet import YuNetTeacher

    best = best_match(YuNetTeacher(str(YUNET_MODEL)).detect(sample_frame), SAMPLE_FACE)
    assert best is not None and iou(best, SAMPLE_FACE) >= 0.5 and best.score >= 0.8
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_face_teachers.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.teachers'`

- [ ] **Step 3: 写实现**

`py/vc_label/teachers/__init__.py`：

```python
"""Teacher models for pre-labelling (MIT / Apache-2.0 / BSD licensed only)."""
```

`py/vc_label/teachers/retinaface_net.py`：

```python
"""RetinaFace (ResNet-50) network, ported from biubug6/Pytorch_Retinaface (MIT License).

Only the layers needed to load `Resnet50_Final.pth` strictly and run inference are kept.
Copyright (c) 2019 biubug6. Licensed under the MIT License.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

_ANCHORS_PER_CELL = 2
_CHANNELS = 256


def _conv_bn(inp: int, oup: int, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.BatchNorm2d(oup))


def _conv_bn_1x1(inp: int, oup: int, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class _SSH(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3X3 = _conv_bn_no_relu(channels, channels // 2)
        self.conv5X5_1 = _conv_bn(channels, channels // 4)
        self.conv5X5_2 = _conv_bn_no_relu(channels // 4, channels // 4)
        self.conv7X7_2 = _conv_bn(channels // 4, channels // 4)
        self.conv7x7_3 = _conv_bn_no_relu(channels // 4, channels // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv5x5_1 = self.conv5X5_1(x)
        out = torch.cat(
            [
                self.conv3X3(x),
                self.conv5X5_2(conv5x5_1),
                self.conv7x7_3(self.conv7X7_2(conv5x5_1)),
            ],
            dim=1,
        )
        return F.relu(out)


class _FPN(nn.Module):
    def __init__(self, in_channels: list[int], channels: int) -> None:
        super().__init__()
        self.output1 = _conv_bn_1x1(in_channels[0], channels)
        self.output2 = _conv_bn_1x1(in_channels[1], channels)
        self.output3 = _conv_bn_1x1(in_channels[2], channels)
        self.merge1 = _conv_bn(channels, channels)
        self.merge2 = _conv_bn(channels, channels)

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = list(features.values())
        out3 = self.output3(c5)
        out2 = self.output2(c4) + F.interpolate(out3, size=c4.shape[2:], mode="nearest")
        out2 = self.merge2(out2)
        out1 = self.output1(c3) + F.interpolate(out2, size=c3.shape[2:], mode="nearest")
        out1 = self.merge1(out1)
        return [out1, out2, out3]


class _Head(nn.Module):
    def __init__(self, values_per_anchor: int) -> None:
        super().__init__()
        self.values = values_per_anchor
        self.conv1x1 = nn.Conv2d(_CHANNELS, _ANCHORS_PER_CELL * values_per_anchor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.values)


class RetinaFaceNet(nn.Module):
    """Returns (box deltas [N, A, 4], face probability [N, A]) for A priors."""

    def __init__(self) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(
            resnet50(weights=None), {"layer2": 1, "layer3": 2, "layer4": 3}
        )
        self.fpn = _FPN([512, 1024, 2048], _CHANNELS)
        self.ssh1 = _SSH(_CHANNELS)
        self.ssh2 = _SSH(_CHANNELS)
        self.ssh3 = _SSH(_CHANNELS)
        self.ClassHead = nn.ModuleList(_Head(2) for _ in range(3))
        self.BboxHead = nn.ModuleList(_Head(4) for _ in range(3))
        # Unused at inference, kept so the released checkpoint loads strictly.
        self.LandmarkHead = nn.ModuleList(_Head(10) for _ in range(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = self.fpn(self.body(x))
        features = [self.ssh1(pyramid[0]), self.ssh2(pyramid[1]), self.ssh3(pyramid[2])]
        deltas = torch.cat([head(f) for head, f in zip(self.BboxHead, features)], dim=1)
        logits = torch.cat([head(f) for head, f in zip(self.ClassHead, features)], dim=1)
        return deltas, F.softmax(logits, dim=-1)[..., 1]


def load_retinaface(weights_path: str, device: torch.device) -> RetinaFaceNet:
    """Loads Resnet50_Final.pth (keys prefixed with 'module.') with weights_only=True."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    net = RetinaFaceNet()
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)
```

`py/vc_label/teachers/retinaface.py`：

```python
"""RetinaFace-R50 face teacher: full-frame inference, no tiling seams."""

from __future__ import annotations

import math

import numpy as np
import torch
from torchvision.ops import nms

from vc_label.contracts import Box
from vc_label.teachers.retinaface_net import load_retinaface

SOURCE = "teacher_retinaface"
_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_STEPS = (8, 16, 32)
_VARIANCES = (0.1, 0.2)
_BGR_MEAN = (104.0, 117.0, 123.0)


def prior_boxes(height: int, width: int) -> np.ndarray:
    """Anchors as (cx, cy, w, h) normalised to the image, in network output order."""
    rows = []
    for step, sizes in zip(_STEPS, _MIN_SIZES):
        fh, fw = math.ceil(height / step), math.ceil(width / step)
        cy, cx = np.meshgrid(
            (np.arange(fh) + 0.5) * step / height,
            (np.arange(fw) + 0.5) * step / width,
            indexing="ij",
        )
        for_sizes = [
            np.stack(
                [cx, cy, np.full_like(cx, s / width), np.full_like(cy, s / height)], axis=-1
            )
            for s in sizes
        ]
        # Per cell the anchors of each size are adjacent: (fh, fw, len(sizes), 4).
        rows.append(np.stack(for_sizes, axis=2).reshape(-1, 4))
    return np.concatenate(rows, axis=0).astype(np.float32)


def decode(deltas: np.ndarray, priors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Box deltas -> absolute (x1, y1, x2, y2) pixel boxes."""
    centers = priors[:, :2] + deltas[:, :2] * _VARIANCES[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(deltas[:, 2:] * _VARIANCES[1])
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    return boxes * np.array([width, height, width, height], dtype=np.float32)


class RetinaFaceTeacher:
    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        score_threshold: float = 0.3,
        nms_iou: float = 0.4,
    ) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.net = load_retinaface(weights_path, device)

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        tensor = torch.from_numpy(
            (bgr.astype(np.float32) - np.array(_BGR_MEAN, dtype=np.float32)).transpose(2, 0, 1)
        )
        deltas, scores = self.net(tensor.unsqueeze(0).to(self.device))
        deltas = deltas[0].float().cpu().numpy()
        scores = scores[0].float().cpu().numpy()
        keep = scores >= self.score_threshold
        boxes = decode(deltas[keep], prior_boxes(height, width)[keep], width, height)
        scores = scores[keep]
        if len(scores) == 0:
            return []
        kept = nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.nms_iou).numpy()
        return [
            Box("face", *map(float, boxes[i]), score=float(scores[i]), src=SOURCE).clipped(
                width, height
            )
            for i in kept
        ]
```

`py/vc_label/teachers/yunet.py`：

```python
"""YuNet face teacher (OpenCV Zoo, MIT) — the second opinion next to RetinaFace."""

from __future__ import annotations

import cv2
import numpy as np

from vc_label.contracts import Box

SOURCE = "teacher_yunet"
_TOP_K = 5000


class YuNetTeacher:
    def __init__(self, model_path: str, score_threshold: float = 0.3, nms_iou: float = 0.3) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_iou, _TOP_K
        )

    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        if faces is None:
            return []
        # Each row: x, y, w, h, 5 landmark pairs, score.
        return [
            Box("face", float(x), float(y), float(x + w), float(y + h), score=float(row[-1]),
                src=SOURCE).clipped(width, height)
            for row in faces
            for x, y, w, h in [row[:4]]
        ]
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_face_teachers.py`
Expected: `5 passed`（含两个真实模型用例，约 5 秒；OpenCV 可能在 stderr 打印一行 `setPreferableTarget` 提示，不影响结果）

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/teachers/__init__.py py/vc_label/teachers/retinaface_net.py py/vc_label/teachers/retinaface.py py/vc_label/teachers/yunet.py py/tests/conftest.py py/tests/test_face_teachers.py
git commit -m "feat: add RetinaFace and YuNet face teachers"
```

---

### Task 7: 车辆与车牌老师（COCO、OWLv2）

**Files:**
- Create: `py/vc_label/teachers/coco.py`
- Create: `py/vc_label/teachers/owlv2.py`
- Test: `py/tests/test_plate_teachers.py`

**Interfaces:**
- Consumes: `Box`（Task 1）；`iou`（Task 2）；conftest 夹具 `sample_frame`、`device`、`best_match`（Task 6）
- Produces：
  - `vc_label.teachers.coco`：`SOURCE = "teacher_coco"`；`CocoTeacher(device, score_threshold=0.3)`，`detect(bgr) -> list[Box]`，只输出 `cls="person"` 或 `cls="vehicle"`，`src` 形如 `teacher_coco:car`（COCO 类别 car/motorcycle/bus/truck 归为 vehicle）
  - `vc_label.teachers.owlv2`：`SOURCE = "teacher_owlv2"`、`MODEL_ID = "google/owlv2-base-patch16-ensemble"`、`MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"`、`QUERIES = ("a photo of a license plate",)`；`crop_window(vehicle, width, height, margin=0.1) -> tuple[int, int, int, int]`；`Owlv2PlateTeacher(device, score_threshold=0.1, batch_size=8)`，`detect(bgr, vehicles) -> list[Box]`（坐标已换回整图）

OWLv2 要点：transformers 5 的后处理函数是 `post_process_grounded_object_detection`；OWLv2 会把每张裁图在右下补成正方形，所以 `target_sizes` 必须是补齐后的边长 `(s, s)`，不是裁图原尺寸。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_plate_teachers.py`：

```python
import pytest

from conftest import best_match
from vc_label.boxes import iou
from vc_label.contracts import Box
from vc_label.teachers.owlv2 import crop_window

# Frame 006005 of the sg_drive_600s sample: the plate of the red car in the right lane.
SAMPLE_PLATE = Box("plate", 1189, 793, 1245, 808)


def test_crop_window_grows_and_stays_inside_the_image():
    assert crop_window(Box("vehicle", 100, 100, 200, 150), 1000, 1000) == (90, 95, 210, 155)
    assert crop_window(Box("vehicle", 0, 0, 100, 50), 105, 1000) == (0, 0, 105, 55)


@pytest.mark.models
def test_coco_and_owlv2_find_the_sample_car_and_plate(sample_frame, device):
    from vc_label.teachers.coco import CocoTeacher
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher

    aux = CocoTeacher(device).detect(sample_frame)
    car = next(a for a in aux if a.cls == "vehicle" and a.src == "teacher_coco:car" and a.width > 150)
    plates = Owlv2PlateTeacher(device).detect(sample_frame, [car])
    best = best_match(plates, SAMPLE_PLATE)
    assert best is not None and iou(best, SAMPLE_PLATE) >= 0.5 and best.score >= 0.3
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_plate_teachers.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.teachers.owlv2'`

- [ ] **Step 3: 写实现**

`py/vc_label/teachers/coco.py`：

```python
"""COCO Faster R-CNN v2 (torchvision, BSD): persons and vehicles as auxiliary boxes."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from vc_label.contracts import Box

SOURCE = "teacher_coco"
# COCO category ids used by torchvision detection models.
_PERSON = {1: "person"}
_VEHICLES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}


class CocoTeacher:
    """Returns Box(cls="person" | "vehicle"); src carries the COCO name, e.g. "teacher_coco:bus"."""

    def __init__(self, device: torch.device, score_threshold: float = 0.3) -> None:
        self.device = device
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_score_thresh=score_threshold,
        ).eval().to(device)

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1)
        out = self.model([rgb.float().div(255).to(self.device)])[0]
        boxes = []
        for (x1, y1, x2, y2), label, score in zip(
            out["boxes"].cpu().tolist(), out["labels"].cpu().tolist(), out["scores"].cpu().tolist()
        ):
            if label in _PERSON:
                cls, name = "person", _PERSON[label]
            elif label in _VEHICLES:
                cls, name = "vehicle", _VEHICLES[label]
            else:
                continue
            boxes.append(
                Box(cls, x1, y1, x2, y2, score=score, src=f"{SOURCE}:{name}").clipped(width, height)
            )
        return boxes
```

`py/vc_label/teachers/owlv2.py`：

```python
"""OWLv2 (Google, Apache-2.0) plate teacher: open-vocabulary search inside vehicle crops."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vc_label.contracts import Box

SOURCE = "teacher_owlv2"
MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
QUERIES = ("a photo of a license plate",)
_CROP_MARGIN = 0.1


def crop_window(vehicle: Box, width: int, height: int, margin: float = _CROP_MARGIN) -> tuple[int, int, int, int]:
    """Integer crop (x0, y0, x1, y1) around a vehicle, grown by `margin` per side, inside the image."""
    dx, dy = vehicle.width * margin, vehicle.height * margin
    x0 = max(0, int(vehicle.x1 - dx))
    y0 = max(0, int(vehicle.y1 - dy))
    x1 = min(width, int(round(vehicle.x2 + dx)))
    y1 = min(height, int(round(vehicle.y2 + dy)))
    return x0, y0, x1, y1


class Owlv2PlateTeacher:
    def __init__(self, device: torch.device, score_threshold: float = 0.1, batch_size: int = 8) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = self.model.eval().to(device)

    def detect(self, bgr: np.ndarray, vehicles: list[Box]) -> list[Box]:
        height, width = bgr.shape[:2]
        windows = [crop_window(v, width, height) for v in vehicles]
        windows = [w for w in windows if w[2] - w[0] >= 8 and w[3] - w[1] >= 8]
        plates: list[Box] = []
        for start in range(0, len(windows), self.batch_size):
            plates.extend(self._detect_batch(bgr, windows[start : start + self.batch_size]))
        return plates

    @torch.inference_mode()
    def _detect_batch(self, bgr: np.ndarray, windows: list[tuple[int, int, int, int]]) -> list[Box]:
        crops = [Image.fromarray(bgr[y0:y1, x0:x1, ::-1]) for x0, y0, x1, y1 in windows]
        inputs = self.processor(
            text=[list(QUERIES)] * len(crops), images=crops, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # OWLv2 pads each crop to a square at the bottom/right; boxes are relative to that square.
        sides = [max(c.width, c.height) for c in crops]
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(s, s) for s in sides]
        )
        plates = []
        for (x0, y0, x1, y1), result in zip(windows, results):
            for (bx1, by1, bx2, by2), score in zip(result["boxes"].tolist(), result["scores"].tolist()):
                box = Box("plate", bx1, by1, bx2, by2, score=score, src=SOURCE)
                plates.append(box.clipped(x1 - x0, y1 - y0).shifted(x0, y0))
        return plates
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_plate_teachers.py`
Expected: `2 passed`（真实模型用例数秒；不得出现联网请求，`HF_HUB_OFFLINE=1` 由 conftest 设置）

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/teachers/coco.py py/vc_label/teachers/owlv2.py py/tests/test_plate_teachers.py
git commit -m "feat: add COCO vehicle and OWLv2 plate teachers"
```

---

### Task 8: 标注流程（teach / fuse / export-review）

**Files:**
- Create: `py/vc_label/pipeline.py`
- Test: `py/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1–5 的全部公开接口；老师对象只通过下面两个协议调用（测试里用假对象替代真实模型）
- Produces（`vc_label.pipeline`）：常量 `RAW = "raw.jsonl"`、`AUX = "aux.jsonl"`、`PROCESSED = "processed.txt"`、`MERGED = "merged.jsonl"`、`FLAGS = "flags.csv"`；协议 `FrameDetector.detect(bgr) -> list[Box]`、`PlateDetector.detect(bgr, vehicles) -> list[Box]`；`@dataclass Teachers(faces: list[FrameDetector], coco: FrameDetector, plates: PlateDetector)`；`label_frame(bgr, teachers, cfg) -> tuple[list[Box], list[Box]]`；`read_processed(labels_dir) -> set[int]`；`run_teach(frames_dir, labels_dir, teachers, cfg, every=1, limit=None, log=print) -> int`；`@dataclass(frozen=True) FuseSummary(frames, boxes, filled, flags: int)`；`run_fuse(frames_dir, labels_dir, cfg) -> tuple[FuseSummary, list[tuple[FrameRef, list[Box]]], list[Flag]]`；`run_export_review(frames_dir, labels_dir, review_dir, cfg) -> int`

流程约定：
- `run_teach`：按 frames.csv 顺序每 `every` 帧取一帧，跳过 `processed.txt` 里已有的帧，最多处理 `limit` 帧。每帧先把检测追加进 `raw.jsonl` / `aux.jsonl`，再把帧号追加进 `processed.txt`。启动时先删掉“已写检测但没记进 processed”的帧（上次中途崩溃留下的），保证续跑不重复。每 10 帧和最后一帧打印速度与剩余时间。
- `run_fuse`：对每个已处理帧做加权融合 → 时间补洞 → 规则标记，写 `merged.jsonl`、`flags.csv`（列 `video,frame,rule,cls,x1,y1,x2,y2,score,detail`）和 `yolo/<帧号>.txt` + `yolo/classes.txt`。图片尺寸用 Pillow 只读文件头。
- `run_export_review`：先跑一次 `run_fuse`，再每帧写一个 LabelMe JSON，加 `review_queue.csv`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_pipeline.py`：

```python
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vc_label.config import LabelingConfig
from vc_label.contracts import Box, Detection, read_jsonl, write_jsonl
from vc_label.pipeline import Teachers, label_frame, read_processed, run_export_review, run_fuse, run_teach

CFG = LabelingConfig()
W, H = 640, 360


class FakeFaces:
    def __init__(self, src: str) -> None:
        self.src = src

    def detect(self, bgr):
        return [Box("face", 100, 40, 120, 64, score=0.9, src=self.src)]


class FakeCoco:
    def detect(self, bgr):
        return [
            Box("vehicle", 300, 200, 500, 330, score=0.99, src="teacher_coco:car"),  # 200 px wide
            Box("vehicle", 10, 10, 20, 18, score=0.9, src="teacher_coco:car"),  # 10 px: too small
            Box("person", 90, 30, 130, 200, score=0.9, src="teacher_coco:person"),
        ]


class FakePlates:
    def __init__(self) -> None:
        self.searched: list[list[Box]] = []

    def detect(self, bgr, vehicles):
        self.searched.append(vehicles)
        return [Box("plate", v.x1 + 80, v.y2 - 30, v.x1 + 120, v.y2 - 20, score=0.6, src="teacher_owlv2") for v in vehicles]


def teachers() -> Teachers:
    return Teachers(faces=[FakeFaces("teacher_retinaface"), FakeFaces("teacher_yunet")], coco=FakeCoco(), plates=FakePlates())


def make_frames(root: Path, count: int) -> Path:
    frames = root / "frames" / "trip"
    frames.mkdir(parents=True)
    rows = ["video,frame,pts_ms,reason,motion_score"]
    for i in range(count):
        frame = i * 12
        cv2.imwrite(str(frames / f"{frame:06d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        rows.append(f"trip.mp4,{frame},{frame * 1000 / 60:.1f},base,0.0")
    (frames / "frames.csv").write_text("\n".join(rows) + "\n")
    return frames


def test_label_frame_only_searches_plates_on_wide_vehicles():
    t = teachers()
    raw, aux = label_frame(np.zeros((H, W, 3), np.uint8), t, CFG)
    assert [b.src for b in raw] == ["teacher_retinaface", "teacher_yunet", "teacher_owlv2"]
    assert len(aux) == 3
    assert [v.width for v in t.plates.searched[0]] == [200]


def test_run_teach_resumes_where_it_stopped(tmp_path):
    frames, labels = make_frames(tmp_path, 6), tmp_path / "labels"

    assert run_teach(frames, labels, teachers(), CFG, every=2, limit=2, log=lambda _: None) == 2
    assert read_processed(labels) == {0, 24}
    assert run_teach(frames, labels, teachers(), CFG, every=2, log=lambda _: None) == 1
    assert read_processed(labels) == {0, 24, 48}
    assert run_teach(frames, labels, teachers(), CFG, every=2, log=lambda _: None) == 0
    assert sorted({d.frame for d in read_jsonl(labels / "raw.jsonl")}) == [0, 24, 48]
    assert len(list(read_jsonl(labels / "raw.jsonl"))) == 9


def test_run_teach_drops_output_of_unfinished_frames(tmp_path):
    frames, labels = make_frames(tmp_path, 2), tmp_path / "labels"
    labels.mkdir()
    write_jsonl(labels / "raw.jsonl", [Detection("trip.mp4", 0, 0.0, Box("face", 0, 0, 1, 1, src="crashed"))])

    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    assert "crashed" not in {d.box.src for d in read_jsonl(labels / "raw.jsonl")}


def test_run_fuse_writes_merged_flags_and_yolo(tmp_path):
    frames, labels = make_frames(tmp_path, 3), tmp_path / "labels"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    summary, sequence, flags = run_fuse(frames, labels, CFG)

    assert (summary.frames, summary.boxes, summary.filled) == (3, 6, 0)
    assert [b.cls for b in sequence[0][1]] == ["face", "plate"]
    assert sequence[0][1][0].src == "teacher_retinaface+teacher_yunet"
    assert flags == []  # face from both teachers, large person has a face, wide car has a plate
    assert (labels / "yolo" / "classes.txt").read_text() == "face\nplate\n"
    assert len((labels / "yolo" / "000012.txt").read_text().splitlines()) == 2
    assert len(list(read_jsonl(labels / "merged.jsonl"))) == 6
    assert next(csv.reader((labels / "flags.csv").open()))[:3] == ["video", "frame", "rule"]


def test_run_export_review_writes_one_labelme_file_per_frame(tmp_path):
    frames, labels, review = make_frames(tmp_path, 2), tmp_path / "labels", tmp_path / "review"
    run_teach(frames, labels, teachers(), CFG, log=lambda _: None)

    assert run_export_review(frames, labels, review, CFG) == 2

    doc = json.loads((review / "000012.json").read_text())
    assert doc["imagePath"] == "../frames/trip/000012.jpg"
    assert [s["label"] for s in doc["shapes"]] == ["face", "plate"]
    assert (review / "review_queue.csv").exists()
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_pipeline.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.pipeline'`

- [ ] **Step 3: 写实现**

`py/vc_label/pipeline.py`：

```python
"""teach (expensive, resumable) and fuse/export (cheap, recomputed from raw teacher output)."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from vc_label.boxes import fuse
from vc_label.config import LabelingConfig
from vc_label.contracts import LABEL_CLASSES, Box, Detection, FrameRef, read_frames_csv, read_jsonl, to_yolo_lines, write_jsonl
from vc_label.review import labelme_document, write_review_queue
from vc_label.rules import Flag, flag_frame, temporal_fill

RAW = "raw.jsonl"
AUX = "aux.jsonl"
PROCESSED = "processed.txt"
MERGED = "merged.jsonl"
FLAGS = "flags.csv"


class FrameDetector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Box]: ...


class PlateDetector(Protocol):
    def detect(self, bgr: np.ndarray, vehicles: list[Box]) -> list[Box]: ...


@dataclass
class Teachers:
    faces: list[FrameDetector]
    coco: FrameDetector
    plates: PlateDetector


def label_frame(bgr: np.ndarray, teachers: Teachers, cfg: LabelingConfig) -> tuple[list[Box], list[Box]]:
    """(raw face/plate boxes from every teacher, auxiliary person/vehicle boxes)."""
    raw = [b for teacher in teachers.faces for b in teacher.detect(bgr)]
    aux = teachers.coco.detect(bgr)
    min_width = cfg.plate_search_min_vehicle_frac * bgr.shape[1]
    vehicles = [a for a in aux if a.cls == "vehicle" and a.width >= min_width]
    raw += teachers.plates.detect(bgr, vehicles)
    return raw, aux


def read_processed(labels_dir: Path) -> set[int]:
    path = labels_dir / PROCESSED
    return {int(line) for line in path.read_text().split()} if path.exists() else set()


def _drop_unfinished(labels_dir: Path, done: set[int]) -> None:
    """Removes detections of frames a crashed run wrote but never marked processed."""
    for name in (RAW, AUX):
        path = labels_dir / name
        kept = [d for d in read_jsonl(path) if d.frame in done]
        write_jsonl(path, kept)


def run_teach(
    frames_dir: Path,
    labels_dir: Path,
    teachers: Teachers,
    cfg: LabelingConfig,
    every: int = 1,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Labels every `every`-th frame of frames.csv not yet processed, at most `limit`; resumable."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    done = read_processed(labels_dir)
    _drop_unfinished(labels_dir, done)
    todo = [r for r in read_frames_csv(frames_dir)[::every] if r.frame not in done][:limit]
    started = time.monotonic()
    for i, ref in enumerate(todo, start=1):
        bgr = cv2.imread(str(ref.image))
        if bgr is None:
            raise FileNotFoundError(f"cannot read {ref.image}")
        raw, aux = label_frame(bgr, teachers, cfg)
        write_jsonl(labels_dir / RAW, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in raw], append=True)
        write_jsonl(labels_dir / AUX, [Detection(ref.video, ref.frame, ref.pts_ms, b) for b in aux], append=True)
        with (labels_dir / PROCESSED).open("a") as f:
            f.write(f"{ref.frame}\n")
        if i % 10 == 0 or i == len(todo):
            per_frame = (time.monotonic() - started) / i
            log(f"teach: {i}/{len(todo)} frames, {per_frame:.2f} s/frame, ~{per_frame * (len(todo) - i) / 60:.1f} min left")
    return len(todo)


@dataclass(frozen=True)
class FuseSummary:
    frames: int
    boxes: int
    filled: int
    flags: int


def _by_frame(path: Path) -> dict[int, list[Box]]:
    grouped: dict[int, list[Box]] = defaultdict(list)
    for d in read_jsonl(path):
        grouped[d.frame].append(d.box)
    return grouped


def _image_size(ref: FrameRef) -> tuple[int, int]:
    with Image.open(ref.image) as img:
        return img.size


def run_fuse(frames_dir: Path, labels_dir: Path, cfg: LabelingConfig) -> tuple[FuseSummary, list[tuple[FrameRef, list[Box]]], list[Flag]]:
    """Fuses raw teacher boxes, fills temporal holes, applies rules, writes merged/flags/YOLO."""
    done = read_processed(labels_dir)
    refs = [r for r in read_frames_csv(frames_dir) if r.frame in done]
    raw, aux = _by_frame(labels_dir / RAW), _by_frame(labels_dir / AUX)
    fused = [(r, fuse(raw.get(r.frame, []), cfg.fuse_iou)) for r in refs]
    filled_lists, flags = temporal_fill(fused, cfg)
    sequence = [(ref, boxes) for (ref, _), boxes in zip(fused, filled_lists)]
    yolo_dir = labels_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / "classes.txt").write_text("".join(f"{c}\n" for c in LABEL_CLASSES))
    for ref, boxes in sequence:
        width, height = _image_size(ref)
        flags += flag_frame(ref, boxes, aux.get(ref.frame, []), width, height, cfg)
        (yolo_dir / f"{ref.frame:06d}.txt").write_text("".join(f"{line}\n" for line in to_yolo_lines(boxes, width, height)))
    write_jsonl(labels_dir / MERGED, [Detection(r.video, r.frame, r.pts_ms, b) for r, boxes in sequence for b in boxes])
    with (labels_dir / FLAGS).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "frame", "rule", "cls", "x1", "y1", "x2", "y2", "score", "detail"])
        for f in sorted(flags, key=lambda f: (f.frame, f.rule)):
            b = f.box
            writer.writerow([f.video, f.frame, f.rule, b.cls, round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(b.score, 4), f.detail])
    summary = FuseSummary(
        frames=len(sequence),
        boxes=sum(len(b) for _, b in sequence),
        filled=sum(1 for f in flags if f.rule == "temporal_fill"),
        flags=len(flags),
    )
    return summary, sequence, flags


def run_export_review(frames_dir: Path, labels_dir: Path, review_dir: Path, cfg: LabelingConfig) -> int:
    """Writes one LabelMe JSON per labelled frame plus review_queue.csv; returns the frame count."""
    _, sequence, flags = run_fuse(frames_dir, labels_dir, cfg)
    review_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[f.frame].append(f)
    for ref, boxes in sequence:
        width, height = _image_size(ref)
        doc = labelme_document(ref, boxes, per_frame.get(ref.frame, []), review_dir, width, height)
        (review_dir / f"{ref.frame:06d}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    write_review_queue(review_dir / "review_queue.csv", flags)
    return len(sequence)
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_pipeline.py`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_label/pipeline.py py/tests/test_pipeline.py
git commit -m "feat: add resumable teach, fuse and review export pipeline"
```

---

### Task 9: 命令行、说明文档与样例运行

**Files:**
- Create: `py/vc_label/cli.py`
- Create: `py/vc_label/__main__.py`
- Create: `py/README.md`
- Create: `docs/labeling-guide.md`
- Test: `py/tests/test_cli.py`

**Interfaces:**
- Consumes: `LabelingConfig`（Task 3）；`read_frames_csv`（Task 1）；`RETINAFACE_WEIGHTS`、`YUNET_MODEL`、`offline_model_env`（Task 1）；`ensure_outside`、`import_review`（Task 5）；四个老师类（Task 6–7）；`Teachers`、`run_teach`、`run_fuse`、`run_export_review`（Task 8）
- Produces：`vc_label.cli.build_parser() -> argparse.ArgumentParser`、`vc_label.cli.main(argv=None) -> int`（0 成功；`ValueError` / `FileNotFoundError` 打印 `vc_label: <消息>` 到 stderr 并返回 2）；`python -m vc_label`

命令（从仓库根目录运行，需要 `PYTHONPATH=py`）：

```
vc_label teach         --frames DIR --labels DIR [--config JSON] [--every N] [--limit N] [--device auto|mps|cpu]
vc_label fuse          --frames DIR --labels DIR [--config JSON]
vc_label export-review --frames DIR --labels DIR --review DIR [--config JSON]
vc_label import-review --frames DIR --review DIR --gold DIR
```

`--labels`、`--review`、`--gold` 都先经过 `ensure_outside` 检查；`teach` 在加载模型前先校验 `--every` / `--limit` ≥ 1；模型只在 `teach` 时加载。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_cli.py`：

```python
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
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest -q py/tests/test_cli.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_label.cli'`

- [ ] **Step 3: 写实现**

`py/vc_label/cli.py`：

```python
"""Command line: python -m vc_label {teach,fuse,export-review,import-review} ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vc_label.config import LabelingConfig
from vc_label.contracts import read_frames_csv
from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env
from vc_label.review import ensure_outside, import_review


def _load_teachers(cfg: LabelingConfig, device_name: str):
    offline_model_env()
    import torch

    from vc_label.pipeline import Teachers
    from vc_label.teachers.coco import CocoTeacher
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher
    from vc_label.teachers.retinaface import RetinaFaceTeacher
    from vc_label.teachers.yunet import YuNetTeacher

    for path in (RETINAFACE_WEIGHTS, YUNET_MODEL):
        if not path.exists():
            raise FileNotFoundError(f"missing model file {path}; see py/README.md")
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    return Teachers(
        faces=[
            RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device, score_threshold=cfg.retinaface_score),
            YuNetTeacher(str(YUNET_MODEL), score_threshold=cfg.yunet_score),
        ],
        coco=CocoTeacher(device, score_threshold=cfg.coco_score),
        plates=Owlv2PlateTeacher(device, score_threshold=cfg.owlv2_score),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vc_label", description="Teacher pre-labelling and review round trip.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--frames", type=Path, required=True, help="vc_extract output dir (with frames.csv)")
        p.add_argument("--labels", type=Path, required=True, help="label dir, outside --frames")
        p.add_argument("--config", type=Path, help="JSON overrides for LabelingConfig")

    teach = sub.add_parser("teach", help="run the teacher models (slow, resumable)")
    common(teach)
    teach.add_argument("--every", type=int, default=1, help="label every N-th frame of frames.csv")
    teach.add_argument("--limit", type=int, help="label at most N new frames in this run")
    teach.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    common(sub.add_parser("fuse", help="fuse teacher boxes, apply rules, write YOLO labels"))

    export = sub.add_parser("export-review", help="write LabelMe JSON for X-AnyLabeling")
    common(export)
    export.add_argument("--review", type=Path, required=True, help="review dir, outside --frames")

    imp = sub.add_parser("import-review", help="reviewed LabelMe JSON -> gold YOLO labels")
    imp.add_argument("--frames", type=Path, required=True)
    imp.add_argument("--review", type=Path, required=True)
    imp.add_argument("--gold", type=Path, required=True, help="gold label dir, outside --frames")
    return parser


def _run(args: argparse.Namespace) -> str:
    for name in ("labels", "review", "gold"):
        if getattr(args, name, None) is not None:
            ensure_outside(args.frames, getattr(args, name))
    if args.command == "import-review":
        count = import_review(args.review, read_frames_csv(args.frames), args.gold)
        return f"import-review: {count} gold frames in {args.gold}"

    cfg = LabelingConfig.from_json(args.config) if args.config else LabelingConfig()
    if args.command == "teach":
        from vc_label.pipeline import run_teach

        if args.every < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("--every and --limit must be >= 1")
        count = run_teach(args.frames, args.labels, _load_teachers(cfg, args.device), cfg, args.every, args.limit)
        return f"teach: labelled {count} new frames into {args.labels}"
    if args.command == "fuse":
        from vc_label.pipeline import run_fuse

        summary, _, _ = run_fuse(args.frames, args.labels, cfg)
        return f"fuse: {summary.frames} frames, {summary.boxes} boxes ({summary.filled} temporal fills), {summary.flags} flags"
    from vc_label.pipeline import run_export_review

    count = run_export_review(args.frames, args.labels, args.review, cfg)
    return f"export-review: {count} LabelMe files in {args.review}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(_run(args))
    except (ValueError, FileNotFoundError) as e:
        print(f"vc_label: {e}", file=sys.stderr)
        return 2
    return 0
```

`py/vc_label/__main__.py`：

```python
import sys

from vc_label.cli import main

sys.exit(main())
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest -q py/tests/test_cli.py`
Expected: `4 passed`

- [ ] **Step 5: 写说明文档**

`py/README.md`：

````markdown
# Python 模块：老师预标注与人工复核（vc_label）

设计见 `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §5，标注规范与复核流程见 `docs/labeling-guide.md`。

## 环境

仓库根目录的 `.venv`，Python ≥ 3.12（本机 3.14）。依赖版本固定在 `py/requirements.txt`，安装约下载 260MB，**联网前先确认**：

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r py/requirements.txt
```

## 模型权重

一次性下载约 870MB，放在所有 checkout 共享的 `~/.cache/video-compliance/`。**联网前先确认**。

| 模型 | 用途 | 许可 | 位置 | sha256 |
|---|---|---|---|---|
| RetinaFace-R50 | 人脸（主） | MIT | `models/Resnet50_Final.pth` | `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d` |
| YuNet 2023mar | 人脸（第二意见） | MIT | `models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Faster R-CNN v2 COCO | 行人、车辆 | BSD | `torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` |
| OWLv2 base ensemble | 车辆裁图内找车牌 | Apache-2.0 | `hf/`（版本 `cfd3195ba4ea9592eec887ded089f4c08eff231d`） | `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` |

下载命令：

```bash
mkdir -p ~/.cache/video-compliance/models && curl -L -o ~/.cache/video-compliance/models/Resnet50_Final.pth https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth
```

```bash
curl -L -o ~/.cache/video-compliance/models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

```bash
TORCH_HOME=~/.cache/video-compliance/torch .venv/bin/python -c "import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')"
```

```bash
HF_HOME=~/.cache/video-compliance/hf .venv/bin/python -c "from transformers import Owlv2ForObjectDetection as M, Owlv2Processor as P; r='cfd3195ba4ea9592eec887ded089f4c08eff231d'; P.from_pretrained('google/owlv2-base-patch16-ensemble', revision=r); M.from_pretrained('google/owlv2-base-patch16-ensemble', revision=r)"
```

下载后用 `shasum -a 256` 核对上表。运行时 `vc_label` 会设置 `HF_HUB_OFFLINE=1`，不会再联网。

## 用法

在仓库根目录运行。标注目录必须在抽帧目录之外，否则命令直接拒绝。

```bash
PYTHONPATH=py .venv/bin/python -m vc_label teach --frames data/frames/<stem> --labels data/labels/<stem> --limit 300
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_label export-review --frames data/frames/<stem> --labels data/labels/<stem> --review data/review/<stem>
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_label import-review --frames data/frames/<stem> --review data/review/<stem> --gold data/gold/<stem>
```

- `teach`：跑四个老师模型，可中断、可续跑。1080p 约 2.5 秒/帧，用 `--limit` 让每批不超过 15 分钟；`--every N` 只标每 N 帧中的一帧。
- `fuse`：融合、时间补洞、规则标记，写 `merged.jsonl`、`flags.csv`、`yolo/`。很快，可以反复跑（比如改了 `configs/labeling.json` 之后）。
- `export-review`：先做一次 `fuse`，再为每帧写 LabelMe JSON 和 `review_queue.csv`，供 X-AnyLabeling 复核。
- `import-review`：把复核后的 JSON 转成金标准（`yolo/` + `gold_meta.csv`）。

`data/labels/<stem>/` 里的文件：

| 文件 | 内容 |
|---|---|
| `raw.jsonl` | 每个老师模型的原始人脸/车牌框（规格 §3.2 格式，`src` 标明模型） |
| `aux.jsonl` | COCO 的行人和车辆框（规则用） |
| `processed.txt` | 已完成的帧号（续跑依据） |
| `merged.jsonl` | 融合 + 时间补洞后的框（多个来源用 `+` 连接） |
| `flags.csv` | 规则标记，每条带框坐标 |
| `yolo/` | YOLO 格式预标注，`classes.txt` 为 `face`、`plate` |

## 测试

按文件运行，不要跑不带路径的 `pytest`：

```bash
.venv/bin/pytest -q py/tests/test_contracts.py py/tests/test_boxes.py py/tests/test_config.py py/tests/test_rules.py
```

```bash
.venv/bin/pytest -q py/tests/test_review.py py/tests/test_pipeline.py py/tests/test_cli.py
```

```bash
.venv/bin/pytest -q py/tests/test_face_teachers.py py/tests/test_plate_teachers.py
```

带 `models` 标记的用例需要上面的权重和样例帧 `data/frames/sg_drive_600s/006005.jpg`，缺失时自动跳过。
````

`docs/labeling-guide.md`：

```markdown
# 标注规范与复核流程

配合 `vc_label`（`py/README.md`）使用。规格见 `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §5。

## 类别

| 类别 | 标什么 | 不标什么 | 框的范围 |
|---|---|---|---|
| `face` | 能看到五官区域的人脸：正脸、侧脸、挡风玻璃后的司机、广告牌和海报上的人脸（召回优先，一律标） | 只看到后脑勺 | 从额头到下巴，左右贴脸颊 |
| `plate` | 车辆号牌：前牌、后牌、摩托车牌，部分遮挡也标 | 外形像车牌但不是号牌的牌子（路牌、广告） | 贴牌照边框 |

“必须遮住”的尺寸（验收只统计这些）：4K 下人脸高 ≥ 24px、车牌宽 ≥ 48px；1080p 下减半（12px、24px）。更小的也尽量标，但不计入考核。

## 目录约定

| 目录 | 内容 | 谁写 |
|---|---|---|
| `data/frames/<stem>/` | `vc_extract` 抽出的帧和 `frames.csv` | 只由 `vc_extract` 写 |
| `data/labels/<stem>/` | 老师输出、融合结果、规则标记、YOLO 预标注 | `vc_label teach/fuse` |
| `data/review/<stem>/` | 复核用的 LabelMe JSON、`review_queue.csv` | `vc_label export-review`，人工复核 |
| `data/gold/<stem>/` | 金标准 YOLO 标注和 `gold_meta.csv` | `vc_label import-review` |

**任何标注都不能放进 `data/frames/<stem>/`**：重新抽帧时 `vc_extract --overwrite` 会把这个目录整个删掉。`vc_label` 会拒绝把输出写进去；在 X-AnyLabeling 里也要把保存目录设到 `data/review/<stem>/`，不能用默认的“保存在图片旁边”。

## 流程

1. **预标注**：`vc_label teach`，每批不超过 15 分钟（1080p 约 2.5 秒/帧，`--limit 300` 约 12 分钟），可以分多批续跑。
2. **导出复核文件**：`vc_label export-review`。
3. **人工复核**（X-AnyLabeling，需另行安装）：
   - 打开图片目录 `data/frames/<stem>/`，把标注保存目录设为 `data/review/<stem>/`，即可看到预标注框。
   - 标签只用 `face`、`plate`，只画矩形。
   - 每个框的描述里有 `score=…`、`src=…` 和被标记的规则名，方便判断。
   - 勾选图片级场景标记：`day`、`night`、`rain`、`tunnel`、`city`、`highway`。
   - 删除误检，补画漏检，调整框到上面的规范。
   - **训练用视频**：至少复核 `review_queue.csv` 里列出的帧（被标记多的排在前面，`where` 列给出被标记框的位置）。
   - **金标准视频**：逐帧复核每一帧，不能只看被标记的帧。
4. **导入**：`vc_label import-review`，生成 `data/gold/<stem>/`。

## 规则标记

| 规则 | 含义 | 常见情况 |
|---|---|---|
| `face_single_source` | 只有一个人脸模型检出 | YuNet 在楼面、招牌上的误检；或 RetinaFace 漏掉的小脸 |
| `low_score` | 分数落在低分区间（人脸 0.3–0.5，车牌 0.1–0.3） | 远处小目标或误检 |
| `person_without_face` | 较高的行人头部区域没有人脸 | 背对镜头（正常）；或漏检 |
| `vehicle_without_plate` | 较宽的车辆上没有车牌 | 侧面看不到车牌（正常）；或漏检 |
| `temporal_fill` | 前后帧都有、本帧没有，已自动补框 | 确认补出来的框位置是否正确 |

阈值都在 `configs/labeling.json`，第一轮复核后可以按误报和漏报情况调整，再跑 `vc_label fuse` 重新生成。

## 金标准规则（规格 §5.5）

- **按视频划分**：同一段视频的帧只能属于训练、验证、金标准中的一个。金标准视频完全不参与训练。
- **规模按目标数定**：每类“必须遮住”的目标至少 500 个，帧数不够就继续加。
- **分层覆盖**：白天/夜间、城区/高速、雨天、隧道都要有。
```

- [ ] **Step 6: 样例运行（约 3 分钟，只取每 10 帧 1 帧）**

```bash
PYTHONPATH=py .venv/bin/python -m vc_label teach --frames data/frames/sg_drive_600s --labels data/labels/sg_drive_600s --every 10
```

Expected：每 10 帧打印一次进度，最后一行 `teach: labelled 59 new frames into data/labels/sg_drive_600s`（frames.csv 有 581 行，每 10 行取 1 行），约 2.5 秒/帧。

```bash
PYTHONPATH=py .venv/bin/python -m vc_label export-review --frames data/frames/sg_drive_600s --labels data/labels/sg_drive_600s --review data/review/sg_drive_600s
```

Expected：`export-review: 59 LabelMe files in data/review/sg_drive_600s`；`ls data/labels/sg_drive_600s/yolo | wc -l` 为 60（59 帧 + `classes.txt`）。

```bash
PYTHONPATH=py .venv/bin/python -m vc_label fuse --frames data/frames/sg_drive_600s --labels data/frames/sg_drive_600s/labels; echo "exit=$?"
```

Expected：`vc_label: … is inside the frames directory …`，`exit=2`，且没有创建 `data/frames/sg_drive_600s/labels`。

把 `fuse` 输出的框数、标记数和 `cut -d, -f3 data/labels/sg_drive_600s/flags.csv | sort | uniq -c` 的各规则计数写进报告。

- [ ] **Step 7: 提交**

```bash
git add py/vc_label/cli.py py/vc_label/__main__.py py/tests/test_cli.py py/README.md docs/labeling-guide.md
git commit -m "feat: add vc_label command line, docs and labeling guide"
```

---

## 收尾

- [ ] **按文件复跑全部用例**（三组，合计 48 个）：

```bash
.venv/bin/pytest -q py/tests/test_contracts.py py/tests/test_boxes.py py/tests/test_config.py py/tests/test_rules.py
```

```bash
.venv/bin/pytest -q py/tests/test_review.py py/tests/test_pipeline.py py/tests/test_cli.py
```

```bash
.venv/bin/pytest -q py/tests/test_face_teachers.py py/tests/test_plate_teachers.py
```

Expected：`26 passed`、`15 passed`、`7 passed`。

- [ ] **交付时告知用户**：本计划“与规格的偏差”六条；样例运行的耗时、框数和各规则标记数；人工复核需要安装 X-AnyLabeling（另行征得同意）。
