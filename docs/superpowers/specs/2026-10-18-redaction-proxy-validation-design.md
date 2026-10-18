# 视频打码 MVP：代理验证阶段设计

- 日期：2026-10-18
- 状态：设计已逐节确认；用户授权自主推进至 MVP 完成
- 上位文档：`docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md`（下称“主设计”）。本文只写与主设计不同或补充的部分，未提及之处沿用主设计。

---

## 1. 背景与目标

用户决定**暂不做人工标注**，先跑通“视频打马赛克”的 MVP：用生产架构（YOLOv8n 学生模型 → ONNX → C++ `vc_redact`）产出打码视频，并在没有人工金标准的条件下量化精度。

本阶段产出：

1. 训练好的学生模型 `model.onnx` + `model.json`；
2. C++ `vc_redact`：输入视频，输出打码视频（音频保留）、`detections.jsonl`、`masks.jsonl`、分阶段耗时；
3. 代理评测报告 + 本地抽查页 + 本机速度报告。

主设计 §8.4 的正式验收（A1–A4 基于人工金标准）**推迟到有金标准之后**；本阶段报告一律注明“代理验收”。

## 2. 与主设计的差异

| 项 | 主设计 | 本阶段 |
|---|---|---|
| 素材 | ≥ 15 段真实 4K 行车记录仪视频 | 用户下载的新加坡驾车视频（YouTube，1080p60 H.264，63.8 分钟）剪 6 段 |
| 精度参照 | 人工金标准 | 留出段上老师融合标注（代理）+ 人眼抽查 |
| 阶段 3（人工确认、金标准） | 需要 | 跳过 |
| 训练标签 | 人工确认后的标注 | 老师 `teach → fuse` 导出的 `labels/<stem>/yolo` |
| 推理尺度 | 4K 上 0.67 / 0.5 | 1080p 上 1.0（输入 1920×1088） |
| 训练裁块 | 4K 上 1920×1920 → 1280 | 原尺度 1280×1080 裁块 + 整帧 |
| 必须遮住阈值 | 人脸高 ≥ 24px、车牌宽 ≥ 48px（4K） | 按帧高 / 2160 等比缩放：1080p 为 12px / 24px |
| 难例回流 | 一轮 | 不做（需要人工确认） |
| `--keep-streams` | 有 | 不做（样片无数据流，无法验证）；数据流仍在日志中列出并丢弃 |
| 参数扫描 | 检测频率 × 推理尺度 | 只扫检测频率 |
| 验收 | A1–A4 | §6.6 代理通过条件 |

## 3. 数据

### 3.1 分段

用 `ffmpeg -ss <起点> -t 120 -map 0:v:0 -map 0:a:0 -c copy` 无损剪切（`sg_drive_600s` 是早先剪的，没有音轨，沿用）。命名 `sg_drive_<起点秒>s`。

| 段 | 起点（秒） | 用途 | 老师标注抽样 |
|---|---|---|---|
| sg_drive_300s | 300 | 训练 | 每 3 帧 1 帧 |
| sg_drive_600s | 600 | 训练（另有早先每 10 帧的 59 帧） | 每 3 帧 1 帧 |
| sg_drive_1500s | 1500 | 训练 | 每 3 帧 1 帧 |
| sg_drive_2700s | 2700 | 训练 | 每 3 帧 1 帧 |
| sg_drive_2100s | 2100 | 留出 | 每 4 帧 1 帧 |
| sg_drive_3300s | 3300 | 留出 | 每 4 帧 1 帧 |

抽帧用 `vc_extract` 默认配置（基础 5fps，运动时最高 15fps，`--hwaccel none`），每段约 580 帧。老师标注约 780 + 290 帧，按 2.4 s/帧约 45 分钟，`nice` 后台跑；标注后 `fuse`，不导出复核。

### 3.2 帧号约定

`frames.csv` 的 `frame` 与 `masks.jsonl` / `detections.jsonl` 的 `frame` 都是 `VideoReader` 的解码序号（从 0 起，已应用 MP4 编辑列表，丢弃的预滚帧不计）。评测按 `(video, frame)` 对齐两边。

### 3.3 代理参照与尺寸规则

- 参照框：留出段 `labels/<stem>/merged.jsonl` 中 `face` / `plate` 的全部融合框（含 `temporal_fill`，报告中单列其数量）。
- 必须遮住：人脸高 ≥ 24 × H/2160 px，车牌宽 ≥ 48 × H/2160 px（H 为帧高；1080p 为 12 / 24px）。更小的目标单列“尽力遮挡”，不计入考核。
- 遮住：参照框面积的 ≥ 80% 被该帧打码框的并集覆盖。
- 尺寸档（以 4K 档位按 H/2160 缩放；1080p）：12–24 / 24–48 / 48–96 / > 96 px（人脸按高，车牌按宽）。

## 4. 训练（计划 3，Python 包 `py/vc_train`）

### 4.1 推理几何（训练、导出、C++、评测共用）

- 模型输入固定 W_in × H_in（本阶段 1920 × 1088）。
- 缩放比 `r = min(W_in / W, H_in / H)`；缩放后的图像放在**左上角**，其余像素填 114；RGB，像素 / 255。
- 模型输出坐标除以 `r` 即回到原图坐标。
- 这些参数写在 `model.json`（见 §4.4），C++ 不写死。

### 4.2 数据集构建 `vc_train dataset`

- 输入：训练段的 `frames/<stem>/` 与 `labels/<stem>/`；只用 `processed.txt` 中的帧，无框的帧作为负样本保留。
- 推理尺度 `s = r`（1080p 为 1.0）。裁窗大小 `w = min(W, 1280 / s)`、`h = min(H, 1280 / s)`；裁窗位置在 x 方向取 `{0, W − w}`，y 方向取 `{0, H − h}`（去重）。1080p 即左右两个 1280×1080 裁块（x = 0、640），宽 ≤ 640px 的框至少完整落入其中一块。
- 每个裁块：框裁到窗内，可见面积 < 30% 的丢弃；按 `s` 缩放写出。
- 另加 1 张整帧样本（ultralytics 以 imgsz 1280 读入时缩到 0.67，让模型见过完整大目标）。
- 验证集：每个训练段 `pts_ms ≥ 段末 − 20000` 的帧（时间连续切分）。只用于看训练曲线。
- 产物：`data/datasets/<run>/{images,labels}/{train,val}/`、`data.yaml`（`names: {0: face, 1: plate}`）、`manifest.csv`（样本 → 源帧、裁窗）。

### 4.3 训练 `vc_train train`

- ultralytics YOLOv8n，从本地缓存 `~/.cache/video-compliance/models/yolov8n.pt`（COCO）微调；`imgsz=1280`、`device=mps`、`time=0.5`（30 分钟封顶）、`plots=False`、`seed=0`，增强用默认。
- **离线防护**：进程内设置 `YOLO_OFFLINE=1`、`YOLO_AUTOINSTALL=false`，关闭设置同步，并把 `HTTP(S)_PROXY` / `ALL_PROXY` 指向 `http://127.0.0.1:9`（本机黑洞，这是保证）。权重文件不存在时直接报错，不让 ultralytics 去下载。
- 到点后若验证曲线仍明显上升，再加训一轮（用户已授权）。

### 4.4 导出 `vc_train export`

- ONNX opset 17，静态输入 1×3×1088×1920，图内不含 NMS，`simplify=False`（避免 onnxslim 依赖）。
- `model.json`：

```json
{
  "name": "yolov8n-face-plate-r1",
  "classes": ["face", "plate"],
  "input": {"width": 1920, "height": 1088, "color": "RGB", "scale": 0.00392156862745098,
            "letterbox": "top_left", "pad_value": 114},
  "output_format": "yolov8",
  "train_manifest_sha256": "…",
  "metrics": {"val_map50": 0.0}
}
```

- 一致性检查：抽 20 张留出帧，用同一套 §4.1 预处理分别跑 PyTorch 与 Python onnxruntime，阈值 0.15 + 按类 NMS 后逐框配对，坐标差 < 1px。

## 5. C++ `vc_redact`（计划 4）

### 5.1 命令行

```
vc_redact --input <video> --out <dir> (--model <dir> | --fake-detections <jsonl>)
          [--config configs/redact.json] [--detect-fps <n>] [--detect-every-frame]
          [--ep coreml|cpu] [--hwaccel auto|none|videotoolbox] [--overwrite]
```

产物在 `<dir>/<stem>/`：`<stem>.redacted.mp4`、`detections.jsonl`、`masks.jsonl`（**每帧一行**，无框时 `boxes` 为空）、`stats.json`。退出码同 `vc_extract`：0 成功、1 处理错误、2 参数错误、3 输出已存在。

### 5.2 组件

- **VideoReader 扩展**：可登记“透传流”（音频），其包经回调交给写端；数据流（GPS 等）列入日志后丢弃。默认软件解码（实测快于 VideoToolbox）。
- **FrameSampler**：复用。`detect_fps`（默认 5）为基础频率、`max_detect_fps`（默认 15）为运动时上限；`--detect-every-frame` 时每帧都是关键帧。
- **Detector**：接口 `detect(const VideoFrame&) → vector<Box>`（原图坐标）。
  - `OnnxYoloDetector`：onnxruntime C++，优先 CoreML EP，失败退 CPU 并记录；libswscale 一步完成缩放 + 转 RGB24，按 §4.1 填充；输出 `[1, 4+nc, N]`（cx, cy, w, h, 各类分数）→ 阈值 → 按类 NMS → 除以 `r`。`model.json` 与 ONNX 输入形状、类别数不符时拒绝启动。
  - `FakeDetector`：读 `detections.jsonl` 契约格式，按帧号返回框，供测试。
- **BoxTimeline**：按主设计 §7.2；插值按 PTS，PTS 不严格递增时退回按帧序号。
- **Mosaic**：在 yuv420p 与 NV12 上直接像素化（Y、U、V 分别取块均值），块边长 = max(16px, ⌈框长边 / 8⌉)，块网格从框左上角起。
- **VideoWriter**：与原片同编码格式（H.264 → `h264_videotoolbox`，HEVC → `hevc_videotoolbox`），目标码率取原片视频码率；VideoToolbox 打不开时退 `libx264` / `libx265` 并记录。PTS 透传、音频拷贝；写临时文件，成功后原子重命名，失败删除半成品。
- **线程**：解码线程 → 有界队列 → 主线程（关键帧选择、检测、补框缓冲、打码）→ 有界队列 → 编码封装线程。缓冲帧用 `av_frame_clone` 共享引用。
- **配置** `configs/redact.json`：检测频率、阈值、NMS IoU、关联距离、外扩比例与位移外扩系数、最小框尺寸、马赛克块参数；用 nlohmann/json 解析。
- **`vc_detect`**（调试工具）：对一批 JPEG 跑 `OnnxYoloDetector`，输出 `detections.jsonl`，用于 C++ 与 Python 一致性检查。

### 5.3 测试

GoogleTest，全部用 ffmpeg 现场生成的小视频与 `FakeDetector`：NMS、坐标映射、BoxTimeline（配对、插值、延长、外扩、PTS 回退）、Mosaic（框内每块方差为 0、框外不变，两种像素格式）、端到端（移动色块处均已打码、帧数一致、音频保留、临时文件在失败时被删除）。真实模型到位后加一条 `OnnxYoloDetector` 冒烟测试（模型缺失时跳过）。

## 6. 代理评测（计划 5，Python 包 `py/vc_eval`）

### 6.1 最终覆盖率

`vc_redact` 以默认配置处理两段留出视频，`masks.jsonl` 与留出帧上的参照框按 §3.3 逐帧比对。参照帧由 `vc_extract` 选出，大多不是 `vc_redact` 的关键帧，因此该数字已包含插值损失，直接作为“最终召回”。输出：按类覆盖率 + 95% Wilson 区间、尺寸档、6×4 空间热力图、`temporal_fill` 参照框单列。

### 6.2 时间覆盖率

同一视频再以 `--detect-every-frame` 运行，取其 `origin = detected` 的必须遮住框为参照，计算稀疏关键帧 + 补框的打码框对它的覆盖率（同 §3.3 的 80% 规则）。不依赖老师标注，覆盖全部帧。

### 6.3 检测频率扫描

`detect_fps ∈ {5, 10, 15, 30}` 各跑一次 §6.1 与 §6.2，列表给出覆盖率与各阶段耗时；选满足 §6.6 且开销最小的频率写回 `configs/redact.json`。

### 6.4 输出完整性

ffprobe：输出与原片帧数、时长一致（±1 帧）；音频流存在且编码与时长一致（±1 帧时长）；视频与音频起点差不超过 1 帧。

### 6.5 误打面积、抽查页与速度

- 误打面积：打码框并集中与任何参照框都不相交的面积占画面比例，逐帧报告最大值与分布；**只报告不设门槛**（老师漏标会被算成误打）。
- 抽查页：`data/reports/<run>/index.html`，纯本地静态文件、不加载外部资源：未遮住的参照目标（裁图 + 原帧位置）、逐帧检测有而稀疏打码漏掉的帧、随机 30 张打码帧、两段打码视频链接。
- 速度：两段留出视频各阶段 fps 与占比，按“单段耗时 = 最慢阶段、段间并行”估算生产需求；注明是 1080p H.264、CoreML/CPU 实际后端。
- 汇总写入 `docs/reports/proxy-validation-<运行日期>.md`（只含数字与结论，不含图片）。

### 6.6 代理通过条件

| # | 条件 |
|---|---|
| P1 | 最终覆盖率（§6.1）人脸、车牌各自 ≥ 99%（同时报告 95% 区间） |
| P2 | 热力图中样本 ≥ 30 的格子覆盖率均 ≥ 97% |
| P3 | 输出完整性（§6.4）全部满足 |
| P4 | 抽查页与打码视频中未见明显露出的人脸或车牌（用户确认） |

未通过时依次调整：外扩比例与检测频率 → 检测阈值 → 加训；每次调整记入报告。

## 7. 实施顺序与负载

| 步骤 | 内容 | 依赖 |
|---|---|---|
| 数据 | 剪段、抽帧、老师标注（后台） | 无 |
| 依赖 | 下载 ultralytics 及依赖、YOLOv8n 权重、onnx、onnxruntime（Python）、onnxruntime C++ 预编译包、nlohmann/json | 无 |
| 计划 3 | `vc_train`：数据集、训练、导出、一致性检查 | 训练段标注 |
| 计划 4 | `vc_redact`、`vc_detect` 及测试 | 无（真实模型冒烟测试待计划 3） |
| 计划 5 | `vc_eval` 与报告 | 计划 3、4，留出段标注 |

计划 3 与计划 4 并行开发。重负载任务（老师标注、训练）同一时间只跑一个；测试只跑与改动相关的文件。

## 8. 风险

| 风险 | 应对 |
|---|---|
| 学生学自老师、又用老师评测，数字反映“像不像老师” | 抽查页人眼兜底；正式验收待人工金标准 |
| YouTube 二次压缩的 1080p 不代表真实 4K 行车记录仪 | 速度与小目标数字只作参考；拿到真机素材后重测 |
| MPS 训练 30 分钟不充分 | 看曲线决定是否加训 |
| CoreML EP 不支持部分算子或形状 | 静态输入；退 CPU 并在报告中写明实际后端 |
| ultralytics 运行时联网 | 环境变量关闭 + 本机黑洞代理 |
| AGPL | 流水线只依赖 ONNX 契约，生产前可换模型 |
