# 行车记录视频脱敏（人脸 + 车牌马赛克）

把行车记录仪视频里的**人脸和车牌**打上马赛克，输出脱敏后的 H.264 视频，以及每帧打了哪些框的记录。原则是**宁可多打，不可漏打**：检测器出错时整段失败，不输出没打全的视频。

- 检测：自己训练的 YOLO 学生模型（人脸 + 车牌），人脸再叠加 OpenCV Zoo 的 YuNet，两者取并集。
- 跟踪与打码：关键帧检测，其间按轨迹插值；马赛克网格固定在画面坐标上，车牌只打在框内。
- 平台：日常在 **macOS + Apple Silicon** 上跑（CoreML 检测、VideoToolbox 编码）；**Linux + NVIDIA** 的本机部分已合入，GPU 部分还没有在真机上验证（见 [部署文档](docs/DEPLOYMENT.md)）。

## 目录结构

| 目录 | 内容 |
|---|---|
| `cpp/` | C++ 工具（FFmpeg + ONNX Runtime）：`vc_extract`、`vc_detect`、`vc_redact` 及 GoogleTest 测试，见 [cpp/README.md](cpp/README.md) |
| `py/` | Python 包：`vc_label`、`vc_train`、`vc_eval`，见 [py/README.md](py/README.md) |
| `configs/` | `redact.json`（`vc_redact` 的参数，内置默认值与它相同）、`labeling.json`（打标参数） |
| `scripts/` | `redact.sh`（批量脱敏）、`batch_json.py`（写批量汇总）、`setup_linux.sh`（Linux 安装与构建）、`review_ui.sh`（打开人工复核工具） |
| `docker/` | Linux + NVIDIA 生产镜像：`Dockerfile`、`build.sh`、`entrypoint.sh` |
| `docs/` | 使用、部署、标注规范、设计与验证报告（见下方“文档”） |
| `data/` | 视频、抽帧、标注、模型和输出。**含个人信息，不进仓库**（`.gitignore` 已排除），换机器时自己拷贝 |

## 组件

**C++ 工具**（`cpp/`，构建产物在 `build/<目录>/` 下）：

| 工具 | 管什么 |
|---|---|
| `vc_extract` | 从视频里抽帧（基础频率 + 运动加密，或 `--clip` 片段模式），写 JPEG 和 `frames.csv`，供标注和训练 |
| `vc_detect` | 用 ONNX 学生模型批量检测 `vc_extract` 抽出的图片，写 `detections.jsonl`，用来核对 C++ 与 Python 推理是否一致 |
| `vc_redact` | 生产工具：解码 → 关键帧检测 → 跟踪补框 → 马赛克 → 编码，输出脱敏视频和 `detections.jsonl`、`masks.jsonl`、`stats.json` |

**Python 包**（`py/`，只有打标、训练、评测时才需要 Python 环境；只打码视频不需要）：

| 包 | 管什么 |
|---|---|
| `vc_label` | 老师模型（RetinaFace、YuNet、Faster R-CNN、OWLv2）预标注、融合、规则标记疑似漏标，导出给人工复核 |
| `vc_train` | 由标注生成数据集、微调 YOLOv8n 学生模型、导出 `model.onnx` + `model.json`；`vc_train yunet` 把 YuNet 转成 `vc_redact --face-model` 能用的格式 |
| `vc_eval` | 代理评测：检测频率扫描（`run`）、对老师标注的覆盖率（`coverage`）、关键帧之间的时间覆盖率（`temporal`）、观感（`viewing`：打码面积、跳变）、输出完整性（`integrity`：帧数、音频、H.264 Level） |

## 最快上手（macOS）

前提：`brew install cmake ffmpeg`，onnxruntime 与 nlohmann/json 已解压到 `~/.cache/video-compliance/third_party`（命令与校验和见 [cpp/README.md](cpp/README.md) 的“依赖”，会联网约 43 MB，只需一次），模型目录 `data/models/r3/` 和 `data/models/yunet/` 已就位。从零开始的完整步骤见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

1. 编译 Release（不编测试，不需要 GoogleTest）：

   ```bash
   cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF
   ```

   ```bash
   cmake --build build/cpp-release -j 4
   ```

2. 跑一段视频（也可以给目录，处理其中所有 `.mp4` / `.mov` / `.mkv`）：

   ```bash
   scripts/redact.sh ~/redacted ~/dashcam/trip01.mp4
   ```

3. 看输出：`~/redacted/trip01/trip01.redacted.mp4` 是打码后的视频；同目录的 `stats.json` 里 `detector` 应为 `student/CoreML + yunet/CoreML`；`~/redacted/batch.json` 是整批汇总。各文件的含义见 [docs/USAGE.md](docs/USAGE.md) §2“输出”。

还没有模型、只想确认编译没问题，可以用 `--fake-detections` 回放一个框做冒烟（几秒钟），命令见 [cpp/README.md](cpp/README.md) `vc_redact` 一节末尾。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 部署：macOS 从零编译与放模型，Linux + NVIDIA 裸机与 Docker，哪些还没在 GPU 上验证 |
| [docs/USAGE.md](docs/USAGE.md) | 日常使用：单个 / 批量、输出文件、检查结果、4K 注意事项、调参表、已知局限 |
| [cpp/README.md](cpp/README.md) | C++ 依赖、构建、按模块过滤的测试，三个工具的全部参数、输出格式和行为细节 |
| [py/README.md](py/README.md) | Python 环境、老师模型权重、打标、训练与导出、评测子命令 |
| [docs/labeling-guide.md](docs/labeling-guide.md) | 标注规范与人工复核流程 |
| [docs/superpowers/specs/](docs/superpowers/specs/) | 设计文档：MVP 主设计、代理验证、稳定打码与速度、Linux + NVIDIA 部署 |
| [docs/superpowers/plans/](docs/superpowers/plans/) | 各阶段的实施计划 |
| [docs/reports/](docs/reports/) | 验证报告：代理验证（2026-10-18 / 19）、YuNet、稳定打码标定 |

## 现状

- 只在 1080p H.264 行车视频上验证过；4K、HEVC、夜间、雨天没测。验收数字和残余漏洞见 [docs/USAGE.md](docs/USAGE.md) §5 与 [docs/reports/stable-masks-2026-10-19.md](docs/reports/stable-masks-2026-10-19.md)。
- Linux + NVIDIA 的 TensorRT / CUDA 推理、全程 GPU 流水线还在另一份计划里，见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 的“未在 GPU 机器上验证”。
- 许可：学生模型基于 YOLO（AGPL），Linux 上自编的 FFmpeg 带 libx264（GPL），生产许可待定，见 [Linux 部署设计](docs/superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md) §8。
