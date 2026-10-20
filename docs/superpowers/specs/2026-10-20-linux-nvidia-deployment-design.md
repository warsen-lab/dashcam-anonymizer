# Linux + NVIDIA 部署：全程 GPU 的批量脱敏

日期：2026-10-20。前置：`docs/superpowers/specs/2026-10-19-stable-masks-and-speed-design.md`（当前的补框、确认、马赛克规则与验收线）、`docs/reports/stable-masks-2026-10-19.md`（macOS 上的验收结果）。

## 1. 背景、目标与范围

### 目标

- 批量处理行车记录仪视频（2 分钟一段，4K60，H.264 或 HEVC）：一台 Linux + NVIDIA 机器上，**合计 1 小时素材在 10 分钟内处理完**，即合计约 360 帧/秒的 4K。按批量吞吐算，不要求单个长文件在 10 分钟内出结果。
- **输出一律 H.264**（MP4，音频照旧原样拷贝），macOS 也改成这样，两个平台输出一致。
- 帧从解码到编码都留在显存里（"全程 GPU"），见 §4。

### 用户已定的取舍

- 硬件还没定，消费级和数据中心卡都要考虑；系统和 CUDA 也还没装，由本设计给出推荐和固定版本。
- 交付两种形式：Docker 镜像（生产）和裸机安装脚本（开发机）。
- 按全程 GPU 设计；三个子项目一次设计完，依次实现。
- 暂时没有 Linux + NVIDIA 开发机：实现时先做本机（Mac）能验证的部分，GPU 部分等有机器再做和验收。

### 推荐硬件（按 NVIDIA 公开规格，以实测为准）

| 卡 | 定位 | 编 / 解码单元 | 说明 |
|---|---|---|---|
| L4 | 生产首选 | 2 NVENC / 4 NVDEC，24 GB | 云上便宜，功耗低 |
| L40S | 大批量 | 3 NVENC / 3 NVDEC，48 GB | |
| RTX 4090 | 消费级 | 2 NVENC / 1 NVDEC，24 GB | 驱动限制同时编码会话数（约 8 路），并发要设上限 |
| A100 / H100 | **不可用** | 没有 NVENC | |

软件基线：Ubuntu 22.04 / 24.04，NVIDIA 驱动 ≥ 550，CUDA 12；onnxruntime 1.30.0（与 macOS 同版本）的 Linux GPU 包，带 CUDA 与 TensorRT 执行提供者；cuDNN、TensorRT 的版本按 onnxruntime 1.30.0 的发布说明取，写进 Dockerfile 与安装脚本并锁定。

### 不在本轮范围

AMD / Intel GPU、Windows、Kubernetes 与多机调度、实时流、单个长文件切段并行。

## 2. 构建与交付

- **CMake：** 新开关 `VC_WITH_CUDA`（默认 `OFF`；macOS 与本机测试都走 `OFF`）。`ON` 时启用 CUDA 语言编译核函数，`CMAKE_CUDA_ARCHITECTURES` 默认 `86;89`（A10 / 3090、L4 / L40S / 4090）。onnxruntime 仍从 `~/.cache/video-compliance/third_party`（Docker 里 `/opt`）找，按平台选包。`OFF` 时不链接任何 CUDA 库，所有 GPU 代码由宏隔开。
- **FFmpeg：** 项目要求 ≥ 8，Ubuntu 自带的太旧，两种交付方式都从源码编固定版本的 FFmpeg 8：打开 NVDEC / NVENC（nv-codec-headers，MIT），另编 libx264 作为没有 GPU 编码器时的退路。libx264 是 GPL，会使 FFmpeg 成为 GPL，与 YOLO 的 AGPL 一起记入"生产许可待定"。
- **Docker 镜像（生产）：**
  - 两阶段：构建阶段基于 `nvidia/cuda:12.x-devel-ubuntu22.04`，运行阶段基于对应的 `runtime` 镜像加 cuDNN、TensorRT 运行库；基础镜像按 digest 锁定。
  - 镜像里有 `vc_redact`、`scripts/redact.sh`、`scripts/bench_gpu.sh`、`scripts/accept_gpu.sh`，以及验收用的精简 Python 环境（`vc_eval`）。
  - 模型和缓存不进镜像：模型目录挂到 `/models`；`VC_CACHE` 指向挂载卷 `/cache`（TensorRT 引擎缓存）。
  - 用法：`docker run --gpus all -v <素材>:/in -v <输出>:/out -v <模型>:/models -v <缓存>:/cache vc-redact /in /out`。
- **裸机脚本（开发机）：** `scripts/setup_linux.sh`，支持 Ubuntu 22.04 / 24.04：检查驱动版本，装 apt 依赖，下载并按 sha256 校验 onnxruntime GPU 包与 TensorRT，编 FFmpeg，编 `vc_redact`（`VC_WITH_CUDA=ON`）。会联网下载，**默认拒绝运行**，显式设 `VC_ALLOW_DOWNLOAD=1` 才放行；下载缓存放在所有 checkout 共享的 `~/.cache/video-compliance/`，已有且校验通过就不再下载。
- **本机能验证的：** `VC_WITH_CUDA=OFF` 的构建和测试照常通过；Dockerfile 用 `hadolint`、脚本用 `shellcheck` 静态检查（本机没有就跳过并在报告里注明）。镜像本身在目标机或云上构建。

## 3. 子项目 1：Linux 基础（帧在内存里）

- **检测后端：** `--ep` 取值 `auto | tensorrt | cuda | coreml | cpu`，默认 `auto`。
  - `auto` 的尝试顺序：Linux 上 TensorRT → CUDA → CPU；macOS 上 CoreML → CPU。回退时 stderr 打一行原因。
  - 显式指定的后端用不了时，沿用现在 `coreml` 的做法：stderr 提示后退到 CPU。
  - 实际用的后端写进 `stats.json` 的 `detector`，例如 `student/TensorRT + yunet/TensorRT`。
- **TensorRT：** 开 FP16；引擎缓存 `$VC_CACHE/trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`，按模型和输入尺寸分目录（与 CoreML 缓存同一做法，含"带缓存失败就不带缓存再试一次"）。YuNet 的输入随视频分辨率变，每种分辨率首次运行时编译一次，`setup_s` 会偏大。
- **解码：** `--hwaccel` 增加 `cuda`（NVDEC，解完拷回内存，后续流程不变）；Linux 上默认 `auto`，NVDEC 打不开退回软件解码，stderr 一行。
- **编码：** 一律 H.264，编码器按顺序尝试 `h264_nvenc` →（macOS）`h264_videotoolbox` → `libx264`，实际用的写在 `stats.json` 的 `encoder`。
  - **码率：** H.264 原片沿用原码率；HEVC（及其他非 H.264）原片乘 `output_bitrate_factor_hevc`（新配置键，默认 1.5，范围 [1, 4]）；读不到码率时仍按 8 Mb/s × 系数。
  - **4K60：** H.264 需要 Level 5.2，由编码器按分辨率和帧率自动选；`vc_eval integrity` 增加检查：输出是 H.264，Level 不低于该分辨率与帧率所需。
  - 原片像素格式是 10 位或 4:2:2 时，转成 8 位 4:2:0 再编码（H.264 NVENC 不支持 10 位），stderr 一行。
- **一致性：** 同一段视频在 CPU 和 GPU 后端上各跑一次，检测框坐标差、召回差都要在门槛内；门槛到目标机实测后定，先按 FP16 预留 1 px。此项进验收脚本。
- **本机能验证的：** HEVC 原片在 macOS 上也输出 H.264、码率按系数；`--ep` / `--hwaccel` 新取值的解析；`auto` 回退顺序（用假后端测）；新配置键的解析与范围；`integrity` 的 H.264 与 Level 检查。

## 4. 子项目 2：全程 GPU

- **数据流：** NVDEC 解出的帧留在显存（FFmpeg 的 CUDA 硬件帧，NV12），解码 → 主线程 → 编码三个线程之间只传帧引用；NVENC 直接用同一个硬件帧上下文编码，中间不拷回内存。
- **检测前处理：** CUDA 核函数在显存里生成检测器输入：
  - 学生模型：NV12 → RGB 浮点，缩放、右下补边到 1920×1088（与现在 CPU 路径的缩放比例、补边值一致）。
  - YuNet：NV12 → BGR 浮点（0–255），原分辨率，右下补 0 到 32 的倍数。
  - 输入用 onnxruntime 的 IoBinding 绑定显存（CUDA 内存），不经过内存。检测器输出很小，拷回内存后沿用现有的解码和 NMS 代码。
- **马赛克：** CUDA 核函数直接在显存里的 NV12 上打码，规则与 CPU 版（`cpp/src/mask/mosaic.cpp`）完全一致：网格锚定画面、块大小档位、大块先打、边缘不完整的块按实际像素取均值、色度按子采样换算、取整方式相同。要求与 CPU 版**逐像素相同**。"框 → 要打哪些块、每块的像素范围"写成 CPU 与 GPU 共用的纯函数（`__host__ __device__`），CPU 版也改用它，在本机就能测。
- **显存预算：** 一帧 4K NV12 约 12 MB。GPU 路径把"缓冲满就强制关键帧"的上限从 120 帧降到 max(2 × 关键帧步长, 24) 帧（10 档时 24 帧，约 300 MB），NVDEC 帧池按这个上限加上解码、编码队列深度分配。这是对"120 帧强制关键帧"的一处改动，**只在 GPU 路径上生效**；内存帧路径不变。
- **退路：** GPU 路径只接 8 位 4:2:0 原片。10 位、4:2:2、NVDEC 不支持的编码，或 CUDA 设备打不开时，整段改走子项目 1 的内存帧路径，stderr 一行原因，`stats.json` 记 `pipeline: "cpu-frames"` 或 `"gpu"`。
- **正确性（目标机上）：**
  1. 马赛克核函数与 CPU 版在同一批帧、同一批框上逐像素比对；
  2. 前处理张量与 CPU 版比对，差值在门槛内（缩放插值不同会有小差别，门槛实测后定）；
  3. 同一段视频走 GPU 路径与内存帧路径，检测框、召回和观感指标的差值在门槛内。

  三项都进 GPU 测试（`VC_WITH_CUDA=ON` 时编进 `vc_tests`）和验收脚本。
- **本机能验证的：** 共用纯函数（网格、块、像素坐标映射）的单元测试；CPU 马赛克改用纯函数后现有测试全部照过；`VC_WITH_CUDA=OFF` 构建不受影响。

## 5. 子项目 3：批量、多卡、调参、压测与验收

- **批量：** `scripts/redact.sh` 增加：
  - `VC_JOBS`：每块 GPU 同时跑几个 `vc_redact` 进程，默认 4；
  - `VC_GPUS`：用哪几块卡（如 `0,1`），视频按顺序轮流分配，用 `CUDA_VISIBLE_DEVICES` 绑卡；没有 GPU（macOS）时 `VC_JOBS` 仍可用，表示同时跑几个进程，默认 1。
  - 现有规则不变：单段失败不影响其他段，最后汇总失败列表，退出码 1；同名输出拒绝覆盖。
  - 输出目录下新增 `batch.json`：每段的耗时、fps、后端、编码器、路径（gpu / cpu-frames），以及合计吞吐。
- **调参项**（写进 `docs/USAGE.md` 新增的 Linux 一节）：`VC_JOBS`；`detect_fps` 档位；`nvenc_preset`（新配置键，默认 `p4`，`p1` 最快）；TensorRT FP16 开关（新配置键 `trt_fp16`，默认 `true`）；onnxruntime 的 CPU 线程数（新配置键 `ort_intra_threads`，多进程时调小，避免两个会话互相抢 CPU）；消费级卡的并发编码会话不超过驱动上限。
- **压测脚本** `scripts/bench_gpu.sh`：对一批 2 分钟片段，按并发 1、2、4、6、8 依次跑；同时用 `nvidia-smi` 采样 GPU、编码单元、解码单元的占用和显存；输出各并发下的合计 fps，推荐 `VC_JOBS`，写 `bench.json` 与 `bench.md`。
- **验收脚本** `scripts/accept_gpu.sh`：在目标机上一条命令跑完，输出 `accept.md`：
  1. 输出完整性（含 H.264、Level）；
  2. GPU 路径与内存帧路径的一致性（§4 三项）、TensorRT 与 CPU 后端的一致性（§3）；
  3. 三段留出视频（`sg_drive_1200s`、`sg_drive_2400s`、`sg_drive_60s`）的召回与观感，验收线同 macOS：人脸 ≥ 95%、车牌 ≥ 98%（排除参照误检后）、全片面积 p99 ≤ 10%、最大 ≤ 30%、跳变 ≤ 5%、同轨多框帧 0；
  4. 1080p 速度；
  5. 4K 合计吞吐 ≥ 360 fps（即 1 小时素材 10 分钟内）。
- **4K 素材：** 现有测试视频都是 1080p。4K 速度验收需要用户提供若干段真实的 4K60 行车片段；没有时用 1080p 片段放大到 4K 的替代素材，只测速度，报告里标明。
- **本机能验证的：** 批量脚本的并发、轮流分卡、失败汇总、`batch.json`（用假的 `vc_redact` 测）；三个脚本的 `shellcheck`。

## 6. 实现顺序与验证

| 阶段 | 内容 | 在哪验证 |
|---|---|---|
| 1a | `VC_WITH_CUDA` 开关骨架、H.264 统一输出与码率系数、`--ep` / `--hwaccel` 新取值与 `auto` 回退逻辑、`integrity` 检查 | 本机 |
| 1b | Dockerfile、`setup_linux.sh` | 本机静态检查；目标机构建 |
| 1c | CUDA / TensorRT 执行提供者、TensorRT 缓存、NVDEC（拷回内存）、NVENC | 目标机 |
| 2a | 马赛克网格纯函数（CPU 版改用它） | 本机 |
| 2b | CUDA 前处理、IoBinding、CUDA 马赛克、硬件帧贯通、显存预算、退路 | 目标机 |
| 3a | `redact.sh` 并发与分卡、`batch.json` | 本机 |
| 3b | `bench_gpu.sh`、`accept_gpu.sh`、目标机压测与验收、调参文档 | 目标机 |

没有 GPU 机器时，先做 1a、1b（静态部分）、2a、3a；1c、2b、3b 等有机器（自有或租用）再做。所有 GPU 代码只在 `VC_WITH_CUDA=ON` 时编译，`OFF` 的行为与现在逐字节相同（输出编码改成 H.264 除外）。

## 7. 测试

- 本机：上表"本机"各项的单元测试；只跑过滤后的 gtest 与指定 pytest 文件（与现有规则相同）。
- 目标机：`VC_WITH_CUDA=ON` 的 `vc_tests` 过滤组（新增 `Gpu*` 套件：马赛克逐像素、前处理张量、端到端一致性）、`bench_gpu.sh`、`accept_gpu.sh`。

## 8. 已定的取舍与待定项

- **已定：** 输出一律 H.264；HEVC 原片码率乘 1.5；GPU 路径只接 8 位 4:2:0；GPU 路径的强制关键帧上限降到 max(2 × 步长, 24)；libx264 作为退路保留（GPL）。
- **待定（不阻塞设计）：** 生产许可（YOLO AGPL、libx264 GPL）；目标机型号与云厂商；4K60 真实素材；一致性门槛的具体数值（目标机实测后定）。
