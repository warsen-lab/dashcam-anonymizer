# 学生模型训练与 ONNX 导出（vc_train）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Python 包 `vc_train`：把老师标注（`labels/<stem>/yolo`）做成按原尺度裁块的 YOLO 数据集，离线训练 YOLOv8n（face、plate 两类），导出静态输入、图内不含 NMS 的 ONNX 与 `model.json`，并检查 PyTorch 与 onnxruntime 输出一致。

**Architecture:** `py/vc_train/` 与已有的 `py/vc_label/` 并列，复用其数据契约（`Box`、`read_frames_csv`、`from_yolo_lines`、`nms`）。`geometry.py` 定义训练、导出、C++、评测共用的推理几何（左上角对齐的 letterbox、YOLOv8 输出解码）；`dataset.py` 是纯文件处理；`offline.py` 在导入 ultralytics 之前切断一切联网；`train.py` / `export.py` 是 ultralytics 的薄封装；`cli.py` 提供 `dataset` / `train` / `export` 三个子命令。

**Tech Stack:** Python 3.14、ultralytics 8.4.172（`--no-deps` 安装）、PyTorch 2.14（MPS）、onnx 1.23.1、onnxruntime 1.30.0、OpenCV 5.0（headless）、numpy、pytest。

**规格：** `docs/superpowers/specs/2026-10-18-redaction-proxy-validation-design.md` §4（主设计 `2026-10-17-dashcam-redaction-mvp-design.md` §6）。

**计划路线：** 1 vc_extract（已完成）→ 2 vc_label（已完成）→ **3 本计划** → 4 vc_redact（并行）→ 5 代理评测。

## Global Constraints

- Python 虚拟环境为仓库根目录的 `.venv`；本计划需要的包已安装（ultralytics 8.4.172、onnx 1.23.1、onnxruntime 1.30.0 及其依赖），**不得再安装或升级任何包**，不得联网。ultralytics 用 `--no-deps` 安装，因为它声明依赖 `opencv-python`，会与已装的 `opencv-python-headless` 冲突。
- 权重固定在共享缓存 `~/.cache/video-compliance/models/yolov8n.pt`（sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`）；文件缺失时报错，绝不让 ultralytics 去下载。
- 导入 ultralytics 之前必须调用 `vc_train.offline.ultralytics_offline_env()`：`YOLO_OFFLINE=1`、`YOLO_AUTOINSTALL=false`、`YOLO_CONFIG_DIR=~/.cache/video-compliance/ultralytics-config`，所有代理变量指向本机黑洞 `http://127.0.0.1:9`；导入后关闭 ultralytics 设置同步（`sync=False`）。
- **只跑指定的测试文件**，从仓库根目录执行：`.venv/bin/pytest -q py/tests/<file>.py`；不得运行不带路径的 `pytest`，不得运行与本计划无关的测试文件。
- 推理几何（§4.1）：缩放比 `r = min(W_in / W, H_in / H)`；缩放后尺寸 `round(W·r) × round(H·r)`（四舍五入取半进位，与 C++ `std::lround` 一致；不用 Python 的银行家舍入）；图像放在**左上角**，其余像素填 114；通道顺序 RGB，像素除以 255；模型输出坐标除以 `r` 回到原图。
- YOLOv8 ONNX 输出形状 `[1, 4 + nc, N]`：前 4 行为输入像素坐标下的 `cx, cy, w, h`，其后每类一行分数（已过 sigmoid）；每个锚点取分数最高的类别（不做多标签）。
- 类别固定 `face=0`、`plate=1`（`vc_label.contracts.LABEL_CLASSES`）。
- 裁块：裁窗 `w = min(W, round(1280 / s))`、`h = min(H, round(1280 / s))`，`s` 为推理缩放比；x 方向位置 `{0, W − w}`、y 方向 `{0, H − h}`（去重）；框裁到窗内后可见面积 < 30% 的丢弃。另加一张整帧样本。
- 验证集：每段 `pts_ms ≥ 段内最大 pts_ms − 20000` 的帧。
- 训练默认：`imgsz=1280`、`device=mps`、`time=0.5`（小时）、`batch=8`、`plots=False`、`seed=0`。
- 导出：ONNX opset 17、静态输入、`nms=False`、`simplify=False`、`dynamic=False`；一致性检查阈值 0.15、NMS IoU 0.5、坐标差 < 1px。
- 标注、数据集、模型都不得写进 `data/frames/<stem>/`。
- 提交信息用 conventional commits，不加署名行。在分支 `feat/vc-train` 上工作（worktree `.claude/worktrees/vc-train`）。

## 文件结构

```
py/
  pyproject.toml                 新增 yolo 标记
  requirements-train.txt         训练 / 导出依赖的固定版本（说明 --no-deps）
  README.md                      新增 vc_train 一节
  vc_train/
    __init__.py
    __main__.py                  python -m vc_train
    geometry.py                  letterbox、decode_yolov8
    dataset.py                   裁窗、裁框、数据集构建
    offline.py                   ultralytics 离线环境、YOLO_WEIGHTS、load_yolo
    train.py                     train()
    export.py                    export_onnx()、model.json、check_consistency()
    cli.py                       dataset / train / export 子命令
  tests/
    conftest.py                  新增 yolo 标记的跳过逻辑
    test_train_geometry.py
    test_train_dataset.py
    test_train_yolo.py           小模型 CPU 训练 + 导出 + 一致性（yolo 标记）
```

---

### Task 1: 推理几何（letterbox 与 YOLOv8 输出解码）

**Files:**
- Create: `py/vc_train/__init__.py`
- Create: `py/vc_train/geometry.py`
- Test: `py/tests/test_train_geometry.py`

**Interfaces:**
- Consumes: `vc_label.contracts.Box(cls, x1, y1, x2, y2, score=1.0, src="")`；`vc_label.boxes.nms(boxes: list[Box], iou_threshold: float) -> list[Box]`（按类贪心 NMS，分数高者优先）。
- Produces:
  - `PAD_VALUE = 114`
  - `letterbox_scale(width: int, height: int, in_width: int, in_height: int) -> float`
  - `letterbox(bgr: np.ndarray, in_width: int, in_height: int) -> tuple[np.ndarray, float]`：返回 `(float32 数组 [1, 3, in_height, in_width]，RGB，0–1；r)`
  - `decode_yolov8(pred: np.ndarray, r: float, classes: tuple[str, ...] | list[str], frame_width: int, frame_height: int, score_threshold: float, iou_threshold: float, src: str = "student") -> list[Box]`：`pred` 形状 `[1, 4+nc, N]` 或 `[4+nc, N]`；返回原图坐标、裁到画面内、按类 NMS 后的框。

- [ ] **Step 1: Write the failing test**

`py/vc_train/__init__.py`：

```python
"""Student model training and ONNX export for dashcam redaction (proxy-validation spec §4)."""
```

`py/tests/test_train_geometry.py`：

```python
import numpy as np
import pytest

from vc_train.geometry import PAD_VALUE, decode_yolov8, letterbox, letterbox_scale


def test_letterbox_scale_fits_the_limiting_side():
    assert letterbox_scale(1920, 1080, 1920, 1088) == 1.0
    assert letterbox_scale(3840, 2160, 1920, 1088) == 0.5
    assert letterbox_scale(100, 50, 64, 64) == pytest.approx(0.64)


def test_letterbox_places_image_top_left_in_rgb_and_pads_with_114():
    bgr = np.zeros((50, 100, 3), np.uint8)
    bgr[:, :] = (255, 0, 0)  # pure blue in BGR
    arr, r = letterbox(bgr, 64, 64)
    assert r == pytest.approx(0.64)
    assert arr.shape == (1, 3, 64, 64) and arr.dtype == np.float32
    # resized content is 64 x 32 (round(100*0.64), round(50*0.64)) at the top left
    assert np.allclose(arr[0, 2, :32, :], 1.0)  # blue lands in RGB channel 2
    assert np.allclose(arr[0, 0, :32, :], 0.0)
    assert np.allclose(arr[0, :, 32:, :], PAD_VALUE / 255.0)


def test_letterbox_without_resize_keeps_pixels_and_pads_bottom_rows():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    arr, r = letterbox(bgr, 1920, 1088)
    assert r == 1.0
    assert np.array_equal((arr[0, :, :1080, :] * 255).round().astype(np.uint8), bgr[:, :, ::-1].transpose(2, 0, 1))
    assert np.allclose(arr[0, :, 1080:, :], PAD_VALUE / 255.0)


def _pred(columns):
    """columns: list of (cx, cy, w, h, face_score, plate_score) -> [1, 6, N]."""
    return np.array(columns, np.float32).T[None]


def test_decode_maps_back_by_r_thresholds_and_suppresses_per_class():
    pred = _pred(
        [
            (32, 16, 8, 4, 0.9, 0.0),    # face, kept
            (100, 100, 10, 10, 0.1, 0.05),  # below threshold
            (33, 16, 8, 4, 0.6, 0.0),    # overlaps the first face -> suppressed
            (32, 16, 8, 4, 0.0, 0.5),    # same place, but plate -> kept
        ]
    )
    boxes = decode_yolov8(pred, 0.5, ("face", "plate"), 1000, 1000, 0.15, 0.5)
    faces = [b for b in boxes if b.cls == "face"]
    plates = [b for b in boxes if b.cls == "plate"]
    assert len(faces) == 1 and len(plates) == 1
    f = faces[0]
    assert (f.x1, f.y1, f.x2, f.y2) == pytest.approx((56, 28, 72, 36))
    assert f.score == pytest.approx(0.9) and f.src == "student"


def test_decode_clips_to_the_frame_and_accepts_2d_input():
    pred = _pred([(2, 2, 10, 10, 0.8, 0.0)])[0]
    (b,) = decode_yolov8(pred, 1.0, ("face", "plate"), 20, 20, 0.15, 0.5)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0, 0, 7, 7))


def test_decode_rejects_a_class_count_mismatch():
    with pytest.raises(ValueError, match="classes"):
        decode_yolov8(_pred([(1, 1, 1, 1, 0.9, 0.1)]), 1.0, ("face",), 10, 10, 0.15, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q py/tests/test_train_geometry.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'vc_train.geometry'`

- [ ] **Step 3: Write minimal implementation**

`py/vc_train/geometry.py`：

```python
"""Inference geometry shared by training, export, the C++ detector and evaluation (spec §4.1).

The frame is scaled by r = min(W_in / W, H_in / H), placed at the top-left corner of a
W_in x H_in canvas filled with 114, converted to RGB and divided by 255. Model outputs are in
canvas pixels; dividing by r maps them back to the frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vc_label.boxes import nms
from vc_label.contracts import Box

PAD_VALUE = 114


def letterbox_scale(width: int, height: int, in_width: int, in_height: int) -> float:
    return min(in_width / width, in_height / height)


def letterbox(bgr: np.ndarray, in_width: int, in_height: int) -> tuple[np.ndarray, float]:
    """BGR uint8 image -> ([1, 3, in_height, in_width] float32 RGB in [0, 1], r)."""
    height, width = bgr.shape[:2]
    r = letterbox_scale(width, height, in_width, in_height)
    # half up, like C++ std::lround (Python's round() would round 360.5 to 360)
    new_w = min(in_width, math.floor(width * r + 0.5))
    new_h = min(in_height, math.floor(height * r + 0.5))
    resized = bgr if (new_w, new_h) == (width, height) else cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((in_height, in_width, 3), PAD_VALUE, np.uint8)
    canvas[:new_h, :new_w] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None], dtype=np.float32) / 255.0, r


def decode_yolov8(
    pred: np.ndarray,
    r: float,
    classes: tuple[str, ...] | list[str],
    frame_width: int,
    frame_height: int,
    score_threshold: float,
    iou_threshold: float,
    src: str = "student",
) -> list[Box]:
    """Raw YOLOv8 output -> boxes in frame pixels after thresholding and per-class NMS."""
    p = pred[0] if pred.ndim == 3 else pred
    if p.shape[0] != 4 + len(classes):
        raise ValueError(f"output has {p.shape[0] - 4} classes, model.json lists {len(classes)} classes")
    scores = p[4:]
    cls_ids = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    keep = np.nonzero(conf >= score_threshold)[0]
    cx, cy, w, h = (p[i, keep] for i in range(4))
    x1 = np.clip((cx - w / 2) / r, 0, frame_width)
    y1 = np.clip((cy - h / 2) / r, 0, frame_height)
    x2 = np.clip((cx + w / 2) / r, 0, frame_width)
    y2 = np.clip((cy + h / 2) / r, 0, frame_height)
    boxes = [
        Box(classes[int(cls_ids[j])], float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), score=float(conf[j]), src=src)
        for i, j in enumerate(keep)
    ]
    return nms(boxes, iou_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q py/tests/test_train_geometry.py`
Expected: `6 passed`，无 warnings。

- [ ] **Step 5: Commit**

```bash
git add py/vc_train/__init__.py py/vc_train/geometry.py py/tests/test_train_geometry.py
git commit -m "feat: add shared letterbox and YOLOv8 decode geometry"
```

---

### Task 2: 数据集构建（原尺度裁块 + 整帧，按时间切验证集）

**Files:**
- Create: `py/vc_train/dataset.py`
- Test: `py/tests/test_train_dataset.py`

**Interfaces:**
- Consumes: `vc_label.contracts.read_frames_csv(frames_dir) -> list[FrameRef]`（`FrameRef(video, frame, pts_ms, image)`）、`from_yolo_lines(lines, width, height, src="") -> list[Box]`、`to_yolo_lines(boxes, width, height) -> list[str]`、`LABEL_CLASSES`；`vc_label.pipeline.read_processed(labels_dir) -> set[int]`、`check_labels_match_frames(frames_dir, labels_dir, refs)`（不匹配时抛 `ValueError`）；`vc_label.review.image_size(path) -> (width, height)`；Task 1 的 `letterbox_scale`。
- Produces:
  - `Window = tuple[int, int, int, int]`（x0, y0, w, h）
  - `crop_windows(width: int, height: int, scale: float, crop: int = 1280) -> list[Window]`
  - `crop_boxes(boxes: list[Box], window: Window, min_visible: float = 0.3) -> list[Box]`：返回窗口坐标下（已平移、已裁剪）的框
  - `@dataclass(frozen=True) DatasetSummary(train_samples: int, val_samples: int, train_frames: int, val_frames: int, boxes: int)`
  - `build_dataset(segments: list[tuple[Path, Path]], out_dir: Path, in_width: int, in_height: int, val_tail_ms: float = 20000.0, crop: int = 1280) -> DatasetSummary`：写出 `images/{train,val}/`、`labels/{train,val}/`、`data.yaml`、`manifest.csv`；`out_dir` 已存在且非空时抛 `ValueError`
  - 常量 `MANIFEST = "manifest.csv"`、`DATA_YAML = "data.yaml"`

- [ ] **Step 1: Write the failing test**

`py/tests/test_train_dataset.py`：

```python
import csv

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train.dataset import DATA_YAML, MANIFEST, build_dataset, crop_boxes, crop_windows


def test_crop_windows_at_native_scale_on_1080p():
    assert crop_windows(1920, 1080, 1.0) == [(0, 0, 1280, 1080), (640, 0, 1280, 1080)]


def test_crop_windows_small_frame_is_one_window():
    assert crop_windows(640, 360, 1.0) == [(0, 0, 640, 360)]


def test_crop_windows_on_4k_at_half_scale():
    assert crop_windows(3840, 2160, 0.5) == [
        (0, 0, 2560, 2160),
        (1280, 0, 2560, 2160),
    ]


def test_crop_windows_tiles_both_axes_when_needed():
    assert crop_windows(3000, 3000, 1.0) == [
        (0, 0, 1280, 1280),
        (1720, 0, 1280, 1280),
        (0, 1720, 1280, 1280),
        (1720, 1720, 1280, 1280),
    ]


def test_crop_boxes_shifts_clips_and_drops_mostly_hidden_boxes():
    window = (640, 0, 1280, 1080)
    inside = Box("face", 700, 10, 720, 40)
    half = Box("plate", 600, 100, 680, 120)  # 40 of 80 px visible -> 50 %
    hidden = Box("face", 620, 200, 650, 230)  # 10 of 30 px visible -> 33 %, kept
    mostly_hidden = Box("face", 600, 300, 650, 330)  # 10 of 50 px visible -> 20 %, dropped
    out = crop_boxes([inside, half, hidden, mostly_hidden], window)
    assert [(b.cls, b.x1, b.y1, b.x2, b.y2) for b in out] == [
        ("face", 60, 10, 80, 40),
        ("plate", 0, 100, 40, 120),
        ("face", 0, 200, 10, 230),
    ]


def _segment(root, stem, frames, boxes_by_frame, size=(1920, 1080)):
    """frames: list of (frame, pts_ms). Writes frames dir + labels dir like vc_extract / vc_label fuse."""
    frames_dir = root / "frames" / stem
    labels_dir = root / "labels" / stem
    (labels_dir / "yolo").mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    width, height = size
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for frame, pts in frames:
            w.writerow([f"{stem}.mp4", frame, pts, "base", 0.0])
            cv2.imwrite(str(frames_dir / f"{frame:06d}.jpg"), np.full((height, width, 3), frame % 255, np.uint8))
            lines = to_yolo_lines(boxes_by_frame.get(frame, []), width, height)
            (labels_dir / "yolo" / f"{frame:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("".join(f"{frame}\n" for frame, _ in frames))
    return frames_dir, labels_dir


def test_build_dataset_writes_crops_full_frames_and_time_split(tmp_path):
    seg = _segment(
        tmp_path,
        "a",
        [(0, 0.0), (10, 5000.0), (20, 15000.0), (30, 30000.0)],
        {0: [Box("face", 100, 100, 120, 130)], 30: [Box("plate", 1700, 900, 1760, 920)]},
    )
    out = tmp_path / "ds"
    summary = build_dataset([seg], out, 1920, 1088)

    assert (summary.train_frames, summary.val_frames) == (2, 2)
    assert (summary.train_samples, summary.val_samples) == (6, 6)
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert names == [
        "a_000000_c0.jpg", "a_000000_c1.jpg", "a_000000_full.jpg",
        "a_000010_c0.jpg", "a_000010_c1.jpg", "a_000010_full.jpg",
    ]
    assert cv2.imread(str(out / "images" / "train" / "a_000000_c1.jpg")).shape == (1080, 1280, 3)
    # the face at x=100 is only inside the left crop
    assert (out / "labels" / "train" / "a_000000_c0.txt").read_text().startswith("0 ")
    assert (out / "labels" / "train" / "a_000000_c1.txt").read_text() == ""
    assert (out / "labels" / "train" / "a_000000_full.txt").read_text().startswith("0 ")
    # the plate at x=1700 is only inside the right crop, frame 30 is in val
    assert (out / "labels" / "val" / "a_000030_c1.txt").read_text().startswith("1 ")
    assert (out / "labels" / "val" / "a_000030_c0.txt").read_text() == ""

    yaml_text = (out / DATA_YAML).read_text()
    assert f"path: {out.resolve()}" in yaml_text
    assert "train: images/train" in yaml_text and "val: images/val" in yaml_text
    assert "0: face" in yaml_text and "1: plate" in yaml_text
    with (out / MANIFEST).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert rows[0].keys() == {"sample", "split", "video", "frame", "x0", "y0", "w", "h", "scale", "boxes"}
    assert summary.boxes == sum(int(r["boxes"]) for r in rows) == 4


def test_build_dataset_requires_fused_labels(tmp_path):
    frames_dir, labels_dir = _segment(tmp_path, "a", [(0, 0.0), (10, 200.0)], {})
    (labels_dir / "yolo" / "000010.txt").unlink()
    with pytest.raises(ValueError, match="vc_label fuse"):
        build_dataset([(frames_dir, labels_dir)], tmp_path / "ds", 1920, 1088)


def test_build_dataset_refuses_a_non_empty_output_dir(tmp_path):
    seg = _segment(tmp_path, "a", [(0, 0.0)], {})
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        build_dataset([seg], out, 1920, 1088)


def test_build_dataset_rejects_labels_of_another_video(tmp_path):
    seg_a = _segment(tmp_path, "a", [(0, 0.0)], {})
    seg_b = _segment(tmp_path, "b", [(5, 0.0)], {})
    with pytest.raises(ValueError, match="does not match"):
        build_dataset([(seg_a[0], seg_b[1])], tmp_path / "ds", 1920, 1088)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q py/tests/test_train_dataset.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'vc_train.dataset'`

- [ ] **Step 3: Write minimal implementation**

`py/vc_train/dataset.py`：

```python
"""YOLO dataset from teacher labels: native-scale crops plus one full frame per labelled frame (spec §4.2)."""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from vc_label.contracts import LABEL_CLASSES, Box, from_yolo_lines, read_frames_csv, to_yolo_lines
from vc_label.pipeline import check_labels_match_frames, read_processed
from vc_label.review import image_size
from vc_train.geometry import letterbox_scale

Window = tuple[int, int, int, int]  # x0, y0, w, h in frame pixels
MANIFEST = "manifest.csv"
DATA_YAML = "data.yaml"
_MANIFEST_FIELDS = ["sample", "split", "video", "frame", "x0", "y0", "w", "h", "scale", "boxes"]


@dataclass(frozen=True)
class DatasetSummary:
    train_samples: int
    val_samples: int
    train_frames: int
    val_frames: int
    boxes: int


def _positions(size: int, window: int) -> list[int]:
    return sorted({0, size - window})


def crop_windows(width: int, height: int, scale: float, crop: int = 1280) -> list[Window]:
    """Windows that become crop x crop samples at `scale`; together they contain every box of width <= W - w."""
    w = min(width, round(crop / scale))
    h = min(height, round(crop / scale))
    return [(x0, y0, w, h) for y0 in _positions(height, h) for x0 in _positions(width, w)]


def crop_boxes(boxes: list[Box], window: Window, min_visible: float = 0.3) -> list[Box]:
    """Boxes in window coordinates; a box with less than `min_visible` of its area inside is dropped."""
    x0, y0, w, h = window
    out = []
    for b in boxes:
        moved = b.shifted(-x0, -y0)
        clipped = moved.clipped(w, h)
        if b.area > 0 and clipped.area / b.area >= min_visible:
            out.append(clipped)
    return out


def _scaled(box: Box, s: float) -> Box:
    return replace(box, x1=box.x1 * s, y1=box.y1 * s, x2=box.x2 * s, y2=box.y2 * s)


def _write_sample(out_dir: Path, split: str, name: str, image, boxes: list[Box]) -> None:
    height, width = image.shape[:2]
    cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = to_yolo_lines(boxes, width, height)
    (out_dir / "labels" / split / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines))


def _link_full_frame(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"dataset dir {out_dir} is not empty; choose a new --out")
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path) -> None:
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(LABEL_CLASSES))
    (out_dir / DATA_YAML).write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}"
    )


def build_dataset(
    segments: list[tuple[Path, Path]],
    out_dir: Path,
    in_width: int,
    in_height: int,
    val_tail_ms: float = 20000.0,
    crop: int = 1280,
) -> DatasetSummary:
    """segments: (frames_dir, labels_dir) pairs; only frames in processed.txt are used."""
    _prepare_out_dir(out_dir)
    counts = {"train_samples": 0, "val_samples": 0, "train_frames": 0, "val_frames": 0, "boxes": 0}
    rows = []
    for frames_dir, labels_dir in segments:
        refs = read_frames_csv(frames_dir)
        check_labels_match_frames(frames_dir, labels_dir, refs)
        done = read_processed(labels_dir)
        used = [r for r in refs if r.frame in done]
        if not used:
            continue
        val_from = max(r.pts_ms for r in used) - val_tail_ms
        for ref in used:
            label_file = labels_dir / "yolo" / f"{ref.frame:06d}.txt"
            if not label_file.exists():
                raise ValueError(f"{label_file} is missing; run vc_label fuse for {labels_dir} first")
            width, height = image_size(ref.image)
            boxes = from_yolo_lines(label_file.read_text().splitlines(), width, height)
            split = "val" if ref.pts_ms >= val_from else "train"
            stem = Path(ref.video).stem
            s = letterbox_scale(width, height, in_width, in_height)
            image = cv2.imread(str(ref.image))
            if image is None:
                raise ValueError(f"cannot read {ref.image}")
            samples = []
            for i, (x0, y0, w, h) in enumerate(crop_windows(width, height, s, crop)):
                part = image[y0 : y0 + h, x0 : x0 + w]
                part_boxes = crop_boxes(boxes, (x0, y0, w, h))
                if s != 1.0:
                    part = cv2.resize(part, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                    part_boxes = [_scaled(b, s) for b in part_boxes]
                name = f"{stem}_{ref.frame:06d}_c{i}"
                _write_sample(out_dir, split, name, part, part_boxes)
                samples.append((name, x0, y0, w, h, s, len(part_boxes)))
            full = f"{stem}_{ref.frame:06d}_full"
            _link_full_frame(ref.image, out_dir / "images" / split / f"{full}.jpg")
            lines = to_yolo_lines(boxes, width, height)
            (out_dir / "labels" / split / f"{full}.txt").write_text("".join(f"{line}\n" for line in lines))
            samples.append((full, 0, 0, width, height, 1.0, len(boxes)))
            for name, x0, y0, w, h, scale, n in samples:
                rows.append([f"{name}.jpg", split, ref.video, ref.frame, x0, y0, w, h, scale, n])
                counts["boxes"] += n
            counts[f"{split}_samples"] += len(samples)
            counts[f"{split}_frames"] += 1
    rows.sort(key=lambda r: (r[1] != "train", r[0]))
    with (out_dir / MANIFEST).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MANIFEST_FIELDS)
        writer.writerows(rows)
    _write_data_yaml(out_dir)
    return DatasetSummary(**counts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q py/tests/test_train_dataset.py`
Expected: `9 passed`，无 warnings。

- [ ] **Step 5: Commit**

```bash
git add py/vc_train/dataset.py py/tests/test_train_dataset.py
git commit -m "feat: build native-scale crop dataset from teacher labels"
```

---

### Task 3: 离线 ultralytics 封装、训练与导出、一致性检查

**Files:**
- Create: `py/vc_train/offline.py`
- Create: `py/vc_train/train.py`
- Create: `py/vc_train/export.py`
- Modify: `py/pyproject.toml`（新增 `yolo` 标记）
- Modify: `py/tests/conftest.py`（`yolo` 标记在权重缺失时跳过）
- Create: `py/requirements-train.txt`
- Test: `py/tests/test_train_yolo.py`

**Interfaces:**
- Consumes: `vc_label.paths.CACHE`、`MODELS`、`offline_model_env()`；Task 1 的 `letterbox`、`decode_yolov8`；Task 2 的 `build_dataset`、`MANIFEST`；`vc_label.contracts.read_frames_csv`、`LABEL_CLASSES`；`vc_label.boxes.iou(a, b) -> float`。
- Produces:
  - `offline.YOLO_WEIGHTS: Path`（`MODELS / "yolov8n.pt"`）、`offline.BLACKHOLE = "http://127.0.0.1:9"`
  - `offline.ultralytics_offline_env() -> None`
  - `offline.load_yolo(weights: Path)`：返回 `ultralytics.YOLO`；文件不存在抛 `FileNotFoundError`
  - `train.train(data_yaml: Path, run_dir: Path, weights: Path = YOLO_WEIGHTS, *, imgsz: int = 1280, epochs: int = 300, hours: float | None = 0.5, batch: int = 8, device: str = "mps", workers: int = 4) -> Path`：返回 `run_dir / "train" / "weights" / "best.pt"`；`run_dir/train` 已存在时抛 `ValueError`
  - `export.export_onnx(weights: Path, out_dir: Path, in_width: int, in_height: int, manifest: Path | None = None, name: str | None = None) -> Path`：写 `out_dir/model.onnx` 与 `out_dir/model.json`，返回 `model.onnx` 路径
  - `export.read_model_json(out_dir: Path) -> dict`
  - `@dataclass(frozen=True) export.ConsistencyReport(images: int, boxes: int, max_diff_px: float, unmatched: int)`，属性 `passed(tol_px: float = 1.0) -> bool`
  - `export.check_consistency(weights: Path, onnx_path: Path, images: list[Path], score_threshold: float = 0.15, iou_threshold: float = 0.5) -> ConsistencyReport`

- [ ] **Step 1: Write the failing test**

`py/pyproject.toml` 的 `markers` 增加一行：

```toml
markers = [
    "models: needs the cached teacher weights and the sg_drive_600s sample frames",
    "yolo: needs the cached YOLOv8n COCO weights; trains a tiny model on the CPU (about a minute)",
]
```

`py/tests/conftest.py` 的 `pytest_collection_modifyitems` 改为同时处理两个标记（保留原有 `models` 逻辑）：

```python
def pytest_collection_modifyitems(items):
    # Every model file the teachers load, not just some: a missing COCO checkpoint would make torchvision
    # download ~160 MB over the metered link instead of skipping.
    missing = [p for p in (*required_model_files(), SAMPLE_FRAMES / "006005.jpg") if not p.exists()]
    yolo_weights = MODELS / "yolov8n.pt"
    for item in items:
        if "models" in item.keywords and missing:
            item.add_marker(pytest.mark.skip(reason=f"missing {', '.join(str(p) for p in missing)}"))
        if "yolo" in item.keywords and not yolo_weights.exists():
            item.add_marker(pytest.mark.skip(reason=f"missing {yolo_weights}"))
```

并把导入改为 `from vc_label.paths import MODELS, offline_model_env, required_model_files`。

`py/tests/test_train_yolo.py`：

```python
"""Tiny end-to-end run on the CPU: dataset -> 1 epoch -> ONNX -> consistency (needs yolov8n.pt)."""

import csv
import json

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box, to_yolo_lines
from vc_train.dataset import build_dataset

pytestmark = pytest.mark.yolo


def _tiny_segment(root, n=6, size=128):
    frames_dir, labels_dir = root / "frames" / "t", root / "labels" / "t"
    frames_dir.mkdir(parents=True)
    (labels_dir / "yolo").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i in range(n):
            img = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            x, y = 10 + 12 * i, 30
            img[y : y + 24, x : x + 24] = (0, 0, 255)
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), img)
            w.writerow(["t.mp4", i, i * 10000.0, "base", 0.0])
            lines = to_yolo_lines([Box("face", x, y, x + 24, y + 24)], size, size)
            (labels_dir / "yolo" / f"{i:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("".join(f"{i}\n" for i in range(n)))
    return frames_dir, labels_dir


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from vc_train.train import train

    root = tmp_path_factory.mktemp("yolo")
    seg = _tiny_segment(root)
    build_dataset([seg], root / "ds", 64, 64, val_tail_ms=20000.0, crop=64)
    best = train(root / "ds" / "data.yaml", root / "run", imgsz=64, epochs=1, hours=None, batch=2, device="cpu", workers=0)
    return root, seg, best


def test_offline_env_blackholes_the_network(monkeypatch):
    from vc_train.offline import BLACKHOLE, ultralytics_offline_env

    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:3128")
    monkeypatch.setenv("YOLO_OFFLINE", "0")
    ultralytics_offline_env()
    import os

    assert os.environ["HTTPS_PROXY"] == BLACKHOLE and os.environ["https_proxy"] == BLACKHOLE
    assert os.environ["YOLO_OFFLINE"] == "1" and os.environ["YOLO_AUTOINSTALL"] == "false"


def test_load_yolo_refuses_missing_weights(tmp_path):
    from vc_train.offline import load_yolo

    with pytest.raises(FileNotFoundError):
        load_yolo(tmp_path / "nope.pt")


def test_train_writes_best_weights_and_refuses_to_reuse_a_run(tiny_run):
    from vc_train.train import train

    root, _, best = tiny_run
    assert best.exists() and best.name == "best.pt"
    with pytest.raises(ValueError, match="already exists"):
        train(root / "ds" / "data.yaml", root / "run", imgsz=64, epochs=1, hours=None, batch=2, device="cpu", workers=0)


def test_export_writes_static_onnx_and_model_json(tiny_run):
    import onnxruntime as ort

    from vc_train.export import export_onnx, read_model_json

    root, _, best = tiny_run
    onnx_path = export_onnx(best, root / "model", 64, 64, manifest=root / "ds" / "manifest.csv", name="tiny")
    assert onnx_path == root / "model" / "model.onnx"
    meta = read_model_json(root / "model")
    assert meta["name"] == "tiny" and meta["classes"] == ["face", "plate"]
    assert meta["input"] == {
        "width": 64, "height": 64, "color": "RGB", "scale": 1 / 255,
        "letterbox": "top_left", "pad_value": 114,
    }
    assert meta["output_format"] == "yolov8" and len(meta["train_manifest_sha256"]) == 64
    assert isinstance(meta["metrics"]["val_map50"], float)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 64, 64]
    assert session.get_outputs()[0].shape[1] == 6


def test_pytorch_and_onnxruntime_agree(tiny_run):
    from vc_train.export import check_consistency, export_onnx

    root, (frames_dir, _), best = tiny_run
    onnx_path = export_onnx(best, root / "model_check", 64, 64)
    images = sorted(frames_dir.glob("*.jpg"))
    # threshold 0 keeps every anchor, so a score sitting on the threshold cannot make the two runtimes disagree
    report = check_consistency(best, onnx_path, images, score_threshold=0.0)
    assert report.images == len(images)
    assert report.passed(1.0), report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q py/tests/test_train_yolo.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'vc_train.offline'`（或 `vc_train.train`）

- [ ] **Step 3: Write minimal implementation**

`py/vc_train/offline.py`：

```python
"""Keeps ultralytics off the network (metered link): no downloads, pip installs, telemetry or settings sync."""

from __future__ import annotations

import os
from pathlib import Path

from vc_label.paths import CACHE, MODELS, offline_model_env

YOLO_WEIGHTS = MODELS / "yolov8n.pt"
BLACKHOLE = "http://127.0.0.1:9"
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def ultralytics_offline_env() -> None:
    """Must run before `import ultralytics`. Forces offline mode and points every proxy at a local black hole."""
    offline_model_env()
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    config_dir = CACHE / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    for var in _PROXY_VARS:
        os.environ[var] = BLACKHOLE
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"


def load_yolo(weights: Path):
    """ultralytics.YOLO for a local weights file; never downloads."""
    ultralytics_offline_env()
    if not Path(weights).is_file():
        raise FileNotFoundError(f"{weights} does not exist (see py/README.md: vc_train weights)")
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    if SETTINGS.get("sync"):
        SETTINGS.update({"sync": False})
    return YOLO(str(weights))
```

`py/vc_train/train.py`：

```python
"""Fine-tunes YOLOv8n on a vc_train dataset, offline (spec §4.3)."""

from __future__ import annotations

from pathlib import Path

from vc_train.offline import YOLO_WEIGHTS, load_yolo


def train(
    data_yaml: Path,
    run_dir: Path,
    weights: Path = YOLO_WEIGHTS,
    *,
    imgsz: int = 1280,
    epochs: int = 300,
    hours: float | None = 0.5,
    batch: int = 8,
    device: str = "mps",
    workers: int = 4,
) -> Path:
    """Trains into run_dir/train; `hours` caps wall time (ultralytics `time`), epochs is the upper bound."""
    run_dir = Path(run_dir).resolve()
    if (run_dir / "train").exists():
        raise ValueError(f"{run_dir / 'train'} already exists; choose a new --run")
    model = load_yolo(weights)
    args = dict(
        data=str(Path(data_yaml).resolve()),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=str(run_dir),
        name="train",
        exist_ok=False,
        plots=False,
        seed=0,
        verbose=False,
    )
    if hours is not None:
        args["time"] = hours
    model.train(**args)
    best = run_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished without {best}")
    return best
```

`py/vc_train/export.py`：

```python
"""ONNX export with model.json, and the PyTorch vs onnxruntime consistency check (spec §4.4)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_label.boxes import iou
from vc_train.geometry import PAD_VALUE, decode_yolov8, letterbox
from vc_train.offline import load_yolo

MODEL_ONNX = "model.onnx"
MODEL_JSON = "model.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_map50(weights: Path) -> float:
    results = weights.parent.parent / "results.csv"
    if not results.exists():
        return 0.0
    with results.open(newline="") as f:
        rows = list(csv.DictReader(f))
    values = [float(v) for r in rows for k, v in r.items() if k.strip() == "metrics/mAP50(B)" and v.strip()]
    return max(values, default=0.0)


def export_onnx(
    weights: Path,
    out_dir: Path,
    in_width: int,
    in_height: int,
    manifest: Path | None = None,
    name: str | None = None,
) -> Path:
    """Static [1, 3, in_height, in_width] ONNX without NMS, plus model.json, into out_dir."""
    model = load_yolo(weights)
    exported = Path(
        model.export(format="onnx", imgsz=(in_height, in_width), opset=17, simplify=False, dynamic=False, nms=False, half=False, device="cpu")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / MODEL_ONNX
    shutil.move(str(exported), target)
    classes = [model.names[i] for i in sorted(model.names)]
    meta = {
        "name": name or out_dir.name,
        "classes": classes,
        "input": {"width": in_width, "height": in_height, "color": "RGB", "scale": 1 / 255, "letterbox": "top_left", "pad_value": PAD_VALUE},
        "output_format": "yolov8",
        "train_manifest_sha256": _sha256(manifest) if manifest else "",
        "metrics": {"val_map50": float(_best_map50(Path(weights)))},
    }
    (out_dir / MODEL_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return target


def read_model_json(out_dir: Path) -> dict:
    return json.loads((out_dir / MODEL_JSON).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ConsistencyReport:
    images: int
    boxes: int
    max_diff_px: float
    unmatched: int

    def passed(self, tol_px: float = 1.0) -> bool:
        return self.unmatched == 0 and self.max_diff_px < tol_px


def check_consistency(
    weights: Path,
    onnx_path: Path,
    images: list[Path],
    score_threshold: float = 0.15,
    iou_threshold: float = 0.5,
) -> ConsistencyReport:
    """Runs the same preprocessed tensors through PyTorch (CPU) and onnxruntime and compares decoded boxes."""
    import onnxruntime as ort
    import torch

    net = load_yolo(weights).model.float().eval()
    classes = [net.names[i] for i in sorted(net.names)]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    in_height, in_width = inp.shape[2], inp.shape[3]
    total, unmatched, worst = 0, 0, 0.0
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cannot read {path}")
        height, width = bgr.shape[:2]
        arr, r = letterbox(bgr, in_width, in_height)
        with torch.no_grad():
            out = net(torch.from_numpy(arr))
        torch_pred = (out[0] if isinstance(out, (list, tuple)) else out).numpy()
        ort_pred = session.run(None, {inp.name: arr})[0]
        a = decode_yolov8(torch_pred, r, classes, width, height, score_threshold, iou_threshold)
        b = decode_yolov8(ort_pred, r, classes, width, height, score_threshold, iou_threshold)
        total += len(a)
        remaining = list(b)
        for box in a:
            same = [o for o in remaining if o.cls == box.cls]
            match = max(same, key=lambda o: iou(o, box), default=None)
            if match is None or iou(match, box) < 0.5:
                unmatched += 1
                continue
            remaining.remove(match)
            worst = max(worst, *(abs(p - q) for p, q in zip((box.x1, box.y1, box.x2, box.y2), (match.x1, match.y1, match.x2, match.y2))))
        unmatched += len(remaining)
    return ConsistencyReport(images=len(images), boxes=total, max_diff_px=worst, unmatched=unmatched)
```

`py/requirements-train.txt`：

```text
# Training / export extras on top of requirements.txt (proxy-validation spec §4).
# ultralytics declares opencv-python, which clashes with opencv-python-headless, so install it without deps:
#   .venv/bin/pip install -r py/requirements-train.txt
#   .venv/bin/pip install --no-deps ultralytics==8.4.172
matplotlib==3.11.2
nvidia-ml-py==13.615.71
onnx==1.23.1
onnxruntime==1.30.0
polars==2.0.0
psutil==7.2.2
requests==2.34.2
ultralytics-platform==0.1.90
ultralytics-thop==2.2.2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q py/tests/test_train_yolo.py`
Expected: `5 passed`（约 1 分钟，CPU）。若 ultralytics 在导出或训练时尝试联网（日志中出现下载、`pip install`、`requirements` 自动更新），必须改为离线方式解决，不得放开网络；若 `export` 因版本检查只打印警告，可以接受，在报告中注明。

Also run: `.venv/bin/pytest -q py/tests/test_train_geometry.py py/tests/test_train_dataset.py`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add py/vc_train/offline.py py/vc_train/train.py py/vc_train/export.py py/pyproject.toml py/tests/conftest.py py/requirements-train.txt py/tests/test_train_yolo.py
git commit -m "feat: train and export YOLOv8n offline with a consistency check"
```

---

### Task 4: 命令行与文档

**Files:**
- Create: `py/vc_train/cli.py`
- Create: `py/vc_train/__main__.py`
- Modify: `py/README.md`（新增 “vc_train” 一节）
- Test: `py/tests/test_train_cli.py`

**Interfaces:**
- Consumes: Task 2 `build_dataset`、`DatasetSummary`、`MANIFEST`；Task 3 `train`、`export_onnx`、`check_consistency`、`ConsistencyReport`；`vc_label.contracts.read_frames_csv`；`vc_label.review.ensure_outside(frames_dir: Path, path: Path) -> None`（`path` 位于抽帧目录内时抛 `ValueError`）。
- Produces: `main(argv: list[str] | None = None) -> int`；子命令：
  - `dataset --stem <stem> [--stem …] [--frames-root data/frames] [--labels-root data/labels] --out <dir> [--width 1920] [--height 1088] [--val-tail-ms 20000]`：打印 `dataset: {train_samples} train / {val_samples} val samples from {train_frames} / {val_frames} frames, {boxes} boxes in {out}`
  - `train --data <data.yaml> --run <dir> [--weights PATH] [--hours 0.5] [--epochs 300] [--batch 8] [--imgsz 1280] [--device mps]`：打印 `train: best weights {path}`
  - `export --run <dir> [--weights PATH，默认 <run>/train/weights/best.pt] [--dataset <dir>] [--width 1920] [--height 1088] [--check-frames <frames dir>] [--check-count 20]`：写 `<run>/model.onnx`、`<run>/model.json`；有 `--check-frames` 时对 frames.csv 中均匀间隔的 `--check-count` 帧做一致性检查，打印 `check: {images} images, {boxes} boxes, max diff {max_diff_px:.3f} px, {unmatched} unmatched`，不通过返回 1
  - 退出码：0 成功；1 一致性检查失败；2 `ValueError` / `OSError` / `FileNotFoundError`（打印 `vc_train: <message>` 到 stderr）

- [ ] **Step 1: Write the failing test**

`py/tests/test_train_cli.py`：

```python
import csv

import cv2
import numpy as np

from vc_label.contracts import Box, to_yolo_lines
from vc_train.cli import main


def _segment(root, stem):
    frames_dir, labels_dir = root / "frames" / stem, root / "labels" / stem
    frames_dir.mkdir(parents=True)
    (labels_dir / "yolo").mkdir(parents=True)
    with (frames_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "pts_ms", "reason", "motion_score"])
        for i, pts in enumerate([0.0, 30000.0]):
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
            w.writerow([f"{stem}.mp4", i, pts, "base", 0.0])
            lines = to_yolo_lines([Box("face", 10, 10, 30, 40)], 1920, 1080)
            (labels_dir / "yolo" / f"{i:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
    (labels_dir / "processed.txt").write_text("0\n1\n")


def test_dataset_command_builds_from_stems(tmp_path, capsys):
    _segment(tmp_path, "a")
    _segment(tmp_path, "b")
    out = tmp_path / "ds"
    code = main([
        "dataset", "--stem", "a", "--stem", "b",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(out),
    ])
    assert code == 0
    assert "dataset: 6 train / 6 val samples from 2 / 2 frames" in capsys.readouterr().out
    assert (out / "data.yaml").exists()


def test_dataset_command_reports_errors_with_exit_2(tmp_path, capsys):
    code = main(["dataset", "--stem", "missing", "--frames-root", str(tmp_path), "--labels-root", str(tmp_path), "--out", str(tmp_path / "ds")])
    assert code == 2
    assert capsys.readouterr().err.startswith("vc_train: ")


def test_dataset_output_inside_a_frames_dir_is_refused(tmp_path, capsys):
    _segment(tmp_path, "a")
    code = main([
        "dataset", "--stem", "a",
        "--frames-root", str(tmp_path / "frames"), "--labels-root", str(tmp_path / "labels"),
        "--out", str(tmp_path / "frames" / "a" / "ds"),
    ])
    assert code == 2


def test_export_without_weights_exits_2(tmp_path, capsys):
    assert main(["export", "--run", str(tmp_path / "run")]) == 2
    assert "best.pt" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q py/tests/test_train_cli.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'vc_train.cli'`

- [ ] **Step 3: Write minimal implementation**

先阅读 `py/vc_label/review.py` 中 `ensure_outside` 的签名与 `py/vc_label/cli.py` 的错误处理写法，保持一致。

`py/vc_train/__main__.py`：

```python
import sys

from vc_train.cli import main

sys.exit(main())
```

`py/vc_train/cli.py`：

```python
"""python -m vc_train dataset | train | export (proxy-validation spec §4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vc_label.contracts import read_frames_csv
from vc_label.review import ensure_outside


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_train")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("dataset", help="crop dataset from teacher labels (labels/<stem>/yolo)")
    d.add_argument("--stem", action="append", required=True, help="video stem; repeat for several segments")
    d.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    d.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--width", type=int, default=1920, help="model input width")
    d.add_argument("--height", type=int, default=1088, help="model input height")
    d.add_argument("--val-tail-ms", type=float, default=20000.0)

    t = sub.add_parser("train", help="fine-tune YOLOv8n offline")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--weights", type=Path, default=None, help="default: cached yolov8n.pt")
    t.add_argument("--hours", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--device", default="mps")

    e = sub.add_parser("export", help="ONNX + model.json, optional consistency check")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--weights", type=Path, default=None, help="default: <run>/train/weights/best.pt")
    e.add_argument("--dataset", type=Path, default=None, help="dataset dir whose manifest.csv is hashed into model.json")
    e.add_argument("--width", type=int, default=1920)
    e.add_argument("--height", type=int, default=1088)
    e.add_argument("--check-frames", type=Path, default=None, help="frames dir (frames.csv) for the consistency check")
    e.add_argument("--check-count", type=int, default=20)
    return p


def _dataset(args) -> int:
    from vc_train.dataset import build_dataset

    segments = []
    for stem in args.stem:
        frames_dir = args.frames_root / stem
        ensure_outside(frames_dir, args.out)
        segments.append((frames_dir, args.labels_root / stem))
    s = build_dataset(segments, args.out, args.width, args.height, args.val_tail_ms)
    print(
        f"dataset: {s.train_samples} train / {s.val_samples} val samples from "
        f"{s.train_frames} / {s.val_frames} frames, {s.boxes} boxes in {args.out}"
    )
    return 0


def _train(args) -> int:
    from vc_train.offline import YOLO_WEIGHTS
    from vc_train.train import train

    best = train(
        args.data, args.run, args.weights or YOLO_WEIGHTS,
        imgsz=args.imgsz, epochs=args.epochs, hours=args.hours, batch=args.batch, device=args.device,
    )
    print(f"train: best weights {best}")
    return 0


def _export(args) -> int:
    weights = args.weights or args.run / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} does not exist; train first or pass --weights")
    from vc_train.export import check_consistency, export_onnx

    manifest = args.dataset / "manifest.csv" if args.dataset else None
    onnx_path = export_onnx(weights, args.run, args.width, args.height, manifest=manifest, name=args.run.name)
    print(f"export: {onnx_path}")
    if args.check_frames is None:
        return 0
    refs = read_frames_csv(args.check_frames)
    step = max(1, len(refs) // max(1, args.check_count))
    images = [r.image for r in refs[::step][: args.check_count]]
    report = check_consistency(weights, onnx_path, images)
    print(f"check: {report.images} images, {report.boxes} boxes, max diff {report.max_diff_px:.3f} px, {report.unmatched} unmatched")
    return 0 if report.passed() else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"dataset": _dataset, "train": _train, "export": _export}[args.command](args)
    except (ValueError, OSError) as e:  # FileNotFoundError is an OSError
        print(f"vc_train: {e}", file=sys.stderr)
        return 2
```

`py/README.md` 末尾新增一节（按现有 README 的语气与格式）：

````markdown
## vc_train：学生模型训练与导出

依赖：在 `requirements.txt` 之外再装 `requirements-train.txt`，ultralytics 用 `--no-deps` 单独安装（它声明的 `opencv-python` 会与 `opencv-python-headless` 冲突）。权重 `~/.cache/video-compliance/models/yolov8n.pt` 来自 ultralytics/assets v8.4.0 发布页，sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`。运行时不联网：导入 ultralytics 前设置 `YOLO_OFFLINE=1`、`YOLO_AUTOINSTALL=false` 并把代理指向本机黑洞 `127.0.0.1:9`。

```bash
# 1. 数据集：每张已标注帧 → 两个 1280×1080 原尺度裁块 + 一张整帧；每段最后 20 秒作验证集
PYTHONPATH=py .venv/bin/python -m vc_train dataset --stem sg_drive_300s --stem sg_drive_600s \
    --stem sg_drive_1500s --stem sg_drive_2700s --out data/datasets/r1
# 2. 训练（MPS，30 分钟封顶）
PYTHONPATH=py .venv/bin/python -m vc_train train --data data/datasets/r1/data.yaml --run data/models/r1
# 3. 导出 ONNX（1×3×1088×1920，图内无 NMS）+ model.json，并与 PyTorch 比对 20 张留出帧
PYTHONPATH=py .venv/bin/python -m vc_train export --run data/models/r1 --dataset data/datasets/r1 \
    --check-frames data/frames/sg_drive_2100s
```

测试：`.venv/bin/pytest -q py/tests/test_train_geometry.py py/tests/test_train_dataset.py py/tests/test_train_cli.py`；
`.venv/bin/pytest -q py/tests/test_train_yolo.py`（在 CPU 上训一个 64px 小模型，约 1 分钟，缺权重时跳过）。
````

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q py/tests/test_train_cli.py`
Expected: `4 passed`

Run: `.venv/bin/pytest -q py/tests/test_train_geometry.py py/tests/test_train_dataset.py py/tests/test_train_cli.py`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add py/vc_train/cli.py py/vc_train/__main__.py py/README.md py/tests/test_train_cli.py
git commit -m "feat: add vc_train command line and usage docs"
```
