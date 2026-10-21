# 部署

这份文档讲**怎么把 `vc_redact` 装到一台机器上跑起来**：macOS（Apple Silicon）从零编译，Linux + NVIDIA 的裸机安装与 Docker 镜像。装好之后的日常用法（批量、检查结果、4K、调参、已知局限）见 [USAGE.md](USAGE.md)；每个参数和行为的细节见 [cpp/README.md](../cpp/README.md)。

只打码视频**不需要 Python 环境**（`scripts/redact.sh` 只用系统 `python3` 的标准库写 `batch.json`）；生成 YuNet 人脸模型、训练、评测才需要，见 [py/README.md](../py/README.md)。

> 联网提醒：下面凡是会下载的步骤都标了出来（onnxruntime、FFmpeg 源码、apt 包、Docker 基础镜像）。在计费流量上先确认再做；下载都放在所有 checkout 共享的 `~/.cache/video-compliance/` 下，只需一次。

## 1. macOS（Apple Silicon，CoreML + VideoToolbox）

日常使用的平台，全部在本机验证过。检测走 CoreML，编码走 VideoToolbox（`h264_videotoolbox`，打不开退回 libx264），解码默认软解（1080p H.264 上比 VideoToolbox 快）。

### 1.1 依赖

| 组件 | 怎么装 |
|---|---|
| 编译器 | Xcode Command Line Tools（`xcode-select --install`），要支持 C++20 |
| CMake ≥ 3.24、FFmpeg ≥ 8 | `brew install cmake ffmpeg`。CMake 通过 `brew --prefix ffmpeg` 找 FFmpeg；FFmpeg 低于 8（libswscale < 9）时配置阶段直接报错 |
| onnxruntime 1.30.0（`onnxruntime-osx-arm64-1.30.0`，含 CoreML EP）+ nlohmann/json 3.12.0 | **联网约 43 MB**。解压到 `~/.cache/video-compliance/third_party/`，带 sha256 校验的完整命令见 [cpp/README.md](../cpp/README.md) 的“依赖”。别的位置用 `-DVC_THIRD_PARTY_DIR=<目录>` 指定；缺了配置阶段会报错并指向这里 |
| GoogleTest | 只有编测试时才要：`brew install googletest` |

### 1.2 编译 Release

```bash
cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF
```

```bash
cmake --build build/cpp-release -j 4
```

- 产物：`build/cpp-release/vc_redact`、`vc_extract`、`vc_detect`。`build/cpp-release/vc_redact` 正是 `scripts/redact.sh` 默认找的位置。
- `-DVC_BUILD_TESTS=OFF` 不编测试、不需要 GoogleTest。要编测试就去掉这个选项（默认 `ON`），按 [cpp/README.md](../cpp/README.md) “测试”一节分组过滤运行，不要跑不带过滤的全量。
- `VC_WITH_CUDA` 在 macOS 上保持默认 `OFF`。
- 编好后 `build/cpp-release/vc_redact --help` 列出全部参数和退出码。

### 1.3 放模型

模型不在 git 里（`data/` 整个不进仓库），每个目录只需要 `model.onnx` 和 `model.json` 两个文件：

| 目录 | 是什么 | 怎么来 |
|---|---|---|
| `data/models/r3/` | 学生模型（YOLOv8n，人脸 + 车牌，`model.onnx` 约 12 MB） | 从已有机器整个目录拷过来（`train/` 子目录不需要）；或按 [py/README.md](../py/README.md) “vc_train”重新训练并 `vc_train export` |
| `data/models/yunet/` | YuNet 人脸模型（约 230 KB，MIT） | 拷过来；或在有 Python 环境的机器上 `PYTHONPATH=py .venv/bin/python -m vc_train yunet --out data/models/yunet`，源文件 `~/.cache/video-compliance/models/face_detection_yunet_2023mar.onnx`（来源与 sha256 见 [py/README.md](../py/README.md) “模型权重”），不联网 |

`scripts/redact.sh` 启动时检查这两个目录都有非空的 `model.onnx` + `model.json`，缺了直接报错（退出码 2）。

### 1.4 冒烟与第一次运行

1. 不需要模型的冒烟（几秒钟，确认编译和编码器可用）：命令见 [cpp/README.md](../cpp/README.md) `vc_redact` 一节末尾（`--fake-detections` 回放一个框）。注意那里用的是 `build/cpp/` 下的二进制，换成 `build/cpp-release/` 即可。
2. 真跑一段：

   ```bash
   scripts/redact.sh ~/redacted ~/dashcam/trip01.mp4
   ```

3. 看 `~/redacted/trip01/stats.json`：`detector` 应为 `student/CoreML + yunet/CoreML`，`encoder` 应为 `h264_videotoolbox`。第一次用某个模型（YuNet 是每个新分辨率）时 CoreML 要编译，`setup_s` 偏大，之后走缓存。

### 1.5 常用环境变量

`scripts/redact.sh` 读的环境变量（全部取值以脚本开头的注释为准）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `VC_REDACT` | `build/cpp-release/vc_redact` | 用哪个 `vc_redact` 二进制 |
| `VC_MODEL` | `data/models/r3` | 学生模型目录 |
| `VC_FACE_MODEL` | `data/models/yunet` | YuNet 目录；设为空（`VC_FACE_MODEL=`）只用学生模型 |
| `VC_DETECT_FPS` | 不传，即配置里的 10 | 每秒检测几次，5–60 |
| `VC_REDACT_ARGS` | 空 | 追加给 `vc_redact` 的参数，按空白切分，例如 `"--hwaccel videotoolbox"`、`"--overwrite"`、`"--config my.json"` |
| `VC_JOBS` | 1（设了 `VC_GPUS` 时每卡 4） | 同时跑几个 `vc_redact`，1–64 |
| `VC_GPUS` | 不设 | 只在 Linux + NVIDIA 上有意义，见 §2.5 |
| `VC_PYTHON` | `python3` | 写 `batch.json` 用的解释器（只用标准库） |
| `VC_SANDBOX` | `1` | macOS 上用 `sandbox-exec` 禁止 `vc_redact` 出网；`0` 不套沙箱。没有 `sandbox-exec` 的系统（Linux）上不起作用 |

`vc_redact` 自己读的：

| 变量 | 默认 | 作用 |
|---|---|---|
| `VC_CACHE` | `~/.cache/video-compliance` | 缓存根目录：CoreML 编译缓存在 `coreml-cache/<model.onnx 的 sha256>/`，TensorRT 引擎缓存（GPU 计划里用上）在 `trt-cache/`。缓存损坏时的处理见 [cpp/README.md](../cpp/README.md) “CoreML 编译缓存” |

onnxruntime 的遥测由 `vc_redact` 在创建运行环境前关闭（`ORT_DISABLE_TELEMETRY=1` 并调用 `DisableTelemetryEvents()`），不需要自己设。

## 2. Linux + NVIDIA

设计见 [2026-10-20-linux-nvidia-deployment-design.md](superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md)，本机部分的实施计划见 [2026-10-20-linux-nvidia-local.md](superpowers/plans/2026-10-20-linux-nvidia-local.md)。目标是一台 Linux + NVIDIA 机器上合计 1 小时 4K60 素材 10 分钟内处理完。

### 2.1 验证状态（先看这一节）

main 上合并的是**不需要 GPU 就能写和测的部分**。还没有 GPU 真机，下面右栏的东西**未在 GPU 机器上验证**：

| 已在本机（macOS）实现并测过 | 未在 GPU 机器上验证 |
|---|---|
| CMake 开关 `-DVC_WITH_CUDA=ON` 的配置逻辑（找不到 nvcc 时报错） | 用 CUDA 12 的 nvcc 真正编译；`ON` 目前只多出 CUDA 语言和宏 `VC_WITH_CUDA`，还没有核函数 |
| `--ep auto\|tensorrt\|cuda\|coreml\|cpu` 的解析与回退顺序（Linux：TensorRT → CUDA → CPU） | **TensorRT / CUDA 会话还没接上，现在总是报不可用并退到 CPU**（stderr 一行原因），所以 Linux 上 `stats.json` 的 `detector` 目前会是 `student/CPU + yunet/CPU` |
| `--hwaccel cuda`（NVDEC）与 `h264_nvenc` 编码器的代码路径、回退逻辑 | NVDEC 解码、NVENC 编码实测；NVENC 恒定画质的 `cq` 19 是暂定值，要在目标机上按 SSIM 标定 |
| 输出一律 H.264、`rate_control`、Level 指定（两个平台共用的部分） | NVENC 下的画质与文件大小 |
| `scripts/redact.sh` 的 `VC_JOBS`、`VC_GPUS` 调度和 `batch.json` | 多卡实际吞吐、推荐的 `VC_JOBS` |
| `scripts/setup_linux.sh`、`docker/` 的脚本逻辑（拒绝条件、校验和检查） | **两者都没在目标机上跑过**；几项下载的 sha256、基础镜像 digest、TensorRT 版本都还空着，空着时脚本拒绝运行（见 §2.2） |

还要 GPU 机器的部分（另一份计划）：TensorRT / CUDA 会话与引擎缓存、帧全程留在显存（CUDA 前处理与马赛克）、压测与验收脚本，以及填好上面空着的校验和与版本。在那之前，Linux 上能跑通的是“CPU 检测 + 软件或 NVDEC/NVENC 编解码”，速度远达不到目标。

### 2.2 前提

- Ubuntu 22.04 或 24.04，x86_64（onnxruntime GPU 包只有 x64），NVIDIA 驱动 ≥ 550（`nvidia-smi` 要能用；nv-codec-headers 12.2 的 NVENC / NVDEC 需要它）。
- **固定的下载要先填好 sha256。** 不联网地列出当前状态：

  ```bash
  scripts/setup_linux.sh --check-pins
  ```

  每行 `ok …` 或 `missing …`，有 `missing` 时退出码 1。目前空着的是 onnxruntime GPU 包、nv-codec-headers、Kitware CMake、两个 cuda-keyring；要在目标机上按发布方给的校验和填进脚本的 `PINS`。空着时 `setup_linux.sh` 和 `docker/build.sh` 都拒绝运行。
- `TENSORRT_VERSION`：onnxruntime 1.30.0 需要的 TensorRT 10 运行库的 apt 版本（形如 `10.x.y.z-1+cuda12.8`），也要在目标机上查实。

### 2.3 裸机（开发机）

```bash
VC_ALLOW_DOWNLOAD=1 TENSORRT_VERSION=<apt 版本> scripts/setup_linux.sh
```

**会联网**，不设 `VC_ALLOW_DOWNLOAD=1` 直接拒绝。它依次：

1. 检查系统和驱动版本；
2. 用 sudo 装 apt 包：编译工具、libx264 / libx265 / GoogleTest，以及 NVIDIA 源里的 `cuda-toolkit-12-8`、cuDNN 9、固定到 `TENSORRT_VERSION` 的 TensorRT 运行库；Ubuntu 22.04 的 cmake 太旧，改用缓存里的 Kitware CMake 3.31.6；
3. 把 onnxruntime GPU 包（`onnxruntime-linux-x64-gpu-1.30.0`）和 nlohmann/json 解压到 `~/.cache/video-compliance/third_party/`；
4. 从源码编 FFmpeg 8.1.2（NVENC / NVDEC、libx264、libx265）装到 `~/.cache/video-compliance/linux/ffmpeg-8.1.2/`；已经装好就跳过。**不要用 apt 的 FFmpeg 开发包**，Ubuntu 22.04 / 24.04 的版本是 4.4 / 6.1，不够；
5. 以 `-DVC_WITH_CUDA=ON -DVC_BUILD_TESTS=ON` 构建到 `build/cpp-linux/`。

下载放在 `~/.cache/video-compliance/downloads/`，校验通过的文件不会重复下载。可覆盖的位置：`VC_CACHE`、`VC_FFMPEG_PREFIX`、`VC_THIRD_PARTY_DIR`、`VC_BUILD_DIR`（含义见脚本开头的注释）。

跑起来：`redact.sh` 默认找 `build/cpp-release/vc_redact`，Linux 上要指向 `build/cpp-linux`：

```bash
VC_REDACT=build/cpp-linux/vc_redact scripts/redact.sh ~/redacted ~/dashcam/
```

Linux 上 `vc_redact` 的默认值是 `--ep auto`（TensorRT → CUDA → CPU）、`--hwaccel auto`（试 NVDEC，打不开退回软解并在 stderr 写一行）、`--encoder auto`（`h264_nvenc` → libx264）。Linux 上没有 `sandbox-exec`，`redact.sh` 不会禁止出网（`vc_redact` 本身不联网）。

测试同 macOS：`build/cpp-linux/tests/vc_tests` 按 [cpp/README.md](../cpp/README.md) 的六组过滤运行。

### 2.4 Docker（生产）

镜像分两段：构建段在 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` 上跑 `setup_linux.sh --docker-deps`（下载与编 FFmpeg，不含 libx265）和 `--docker-build`（只编 `vc_redact`，不编测试）；运行段在 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` 上装 TensorRT 运行库、`python3`、`python3-numpy`，带上 `redact.sh`、`batch_json.py`、`configs/` 和 `vc_eval`。改 C++ 源码只会重编 `vc_redact`，不会重新下载、重编 FFmpeg（需要 BuildKit）。

**构建**（会联网拉基础镜像和固定的包；digest 用 `docker buildx imagetools inspect <镜像>` 查）：

```bash
VC_ALLOW_DOWNLOAD=1 CUDA_DEVEL_DIGEST=sha256:<…> CUDA_RUNTIME_DIGEST=sha256:<…> TENSORRT_VERSION=<apt 版本> docker/build.sh
```

可以再给一个参数作镜像标签，默认 `vc-redact`。两个 digest 必须是 `sha256:` 加 64 位十六进制，`TENSORRT_VERSION` 不能空，`setup_linux.sh --check-pins` 必须全部 `ok`，否则拒绝。构建上下文按 `.dockerignore` 排除 `data/`、`build/`、`.venv/`、`.git/` 等，视频和模型不会进镜像。

**运行**（需要 NVIDIA container toolkit）：

```bash
docker run --gpus all -v <素材目录>:/in -v <输出目录>:/out -v <模型目录>:/models -v <缓存目录>:/cache vc-redact /in /out
```

- 参数顺序是**输入在前、输出在后**（`entrypoint.sh` 转成 `redact.sh /out /in`），且必须恰好两个，否则打印用法、退出码 2。
- `/models` 下要有 `r3/` 和 `yunet/`（例如挂 `data/models`）；`/cache` 放 TensorRT 引擎缓存（`trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`），挂到宿主机上才能跨容器复用。
- 镜像里已设 `VC_REDACT=/opt/vc/bin/vc_redact`、`VC_MODEL=/models/r3`、`VC_FACE_MODEL=/models/yunet`、`VC_CACHE=/cache`、`VC_SANDBOX=0`，以及 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`：没有 `video`，container toolkit 不挂 NVENC / NVDEC 的库，编解码会悄悄退回软件。
- 其余 `redact.sh` 变量用 `-e` 传，例如 `-e VC_JOBS=4 -e VC_GPUS=0,1 -e VC_DETECT_FPS=15`。
- 自定义配置：镜像里的 `vc_redact` 默认用内置默认值（与 `configs/redact.json` 相同）。要改，把文件挂进去再用 `VC_REDACT_ARGS` 传，例如 `-v $PWD/my.json:/cfg/redact.json:ro -e VC_REDACT_ARGS="--config /cfg/redact.json"`。
- `docker stop` 发 SIGTERM：`redact.sh` 是 PID 1，立刻停掉在跑的 `vc_redact`，写出已知部分的 `batch.json`，退出码 143；被打断的视频留着 `.partial`，重跑要加 `-e VC_REDACT_ARGS=--overwrite`。

### 2.5 并行与多卡

- `VC_GPUS=0,1`：视频按顺序轮流分到各卡，每个 `vc_redact` 只看到自己那块（设 `CUDA_VISIBLE_DEVICES`）；也可以写 GPU UUID。不设时 `CUDA_VISIBLE_DEVICES` 保持原样。
- `VC_JOBS`：设了 `VC_GPUS` 时是**每块卡**同时几个（默认 4），否则是总共几个（默认 1）。消费级卡（如 RTX 4090）的驱动限制同时编码的会话数（约 8 路），每卡的 `VC_JOBS` 不要超过它。
- 多个进程时把配置里的 `ort_intra_threads` 调小（例如 CPU 核数 ÷ 同时运行的进程数），免得 onnxruntime 会话互相抢 CPU。
- 并行时每个视频的输出在它跑完后整段打印，不会交错；`batch.json` 的 `throughput_fps` 是整批吞吐。实测数字和推荐值等 GPU 压测跑完再填（见 [USAGE.md](USAGE.md) §7）。

### 2.6 Linux 相关的配置项

`configs/redact.json` 里只在 Linux + NVIDIA 上起作用（或主要为它加）的键：`rate_control`、`output_bitrate_factor_hevc`、`nvenc_preset`、`trt_fp16`、`ort_intra_threads`。默认值和调法见 [USAGE.md](USAGE.md) §4 与 §7“调参（Linux）”；其中 `trt_fp16` 要等 TensorRT 会话接上（GPU 计划）才真正生效。

## 3. 配置文件

`vc_redact --config <json>` 读的参数文件，格式就是仓库里的 [configs/redact.json](../configs/redact.json)；省略的键取内置默认值（与该文件相同），未知键、类型不对或越界的值报错（退出码 2）。`scripts/redact.sh` 不传 `--config`，用的就是内置默认值；要换就 `VC_REDACT_ARGS="--config my.json"`。

各键的含义和调法不在这里重复：

- 调参表（调大 / 调小的效果）：[USAGE.md](USAGE.md) §4；
- 跟踪、补框、马赛克、人脸模型相关键的精确行为：[cpp/README.md](../cpp/README.md) `vc_redact` 一节的“要点”与“人脸模型”；
- 默认值从哪来：[docs/reports/proxy-validation-2026-10-18.md](reports/proxy-validation-2026-10-18.md)（检测频率、分数阈值）、[docs/reports/stable-masks-2026-10-19.md](reports/stable-masks-2026-10-19.md)（外扩、收缩、确认门槛）。

## 4. 输出文件

每个视频写到 `<输出目录>/<视频名>/`（视频名是文件名去掉扩展名；不同目录下的同名视频会撞到同一个输出目录，`redact.sh` 把后一个记为失败、退出码 3）：

| 文件 | 内容 |
|---|---|
| `<视频名>.redacted.mp4` | 打码后的视频，一律 H.264 / MP4，默认恒定画质（文件通常比原片大，1080p 行车记录约 4 倍）；音频原样拷贝，旋转、色彩标签随原片。运行中写成 `.partial`，成功才改名 |
| `detections.jsonl` | 关键帧上检测器的原始输出，每框一行，含来源 `src` 和是否确认 `confirmed` |
| `masks.jsonl` | 每帧一行，列出实际打码的框：坐标、轨迹 `track`、来源 `origin`、马赛克块大小 `block`、是否向外对齐网格 `snap`（车牌为 `false`） |
| `stats.json` | 帧数、关键帧数、检测与打码数量、检测后端 `detector`、编码器 `encoder`、`rate_control`、各阶段耗时 `stages`、总耗时、生效的完整配置 |

`redact.sh` 另在输出目录根下写 `batch.json`（每次覆盖）：每个视频的退出状态、GPU，成功的还有帧数、耗时、`fps`、`detector`、`encoder` 等，以及整批的 `wall_s`、`throughput_fps`。

字段的完整说明见 [cpp/README.md](../cpp/README.md) `vc_redact` 一节（`detections.jsonl` 的格式另见 [主设计](superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md) §3.2），怎么用它们检查结果见 [USAGE.md](USAGE.md) §2“检查结果”。

## 5. 退出码

`vc_redact`：0 成功，1 处理失败，2 参数错误（含非法的 `--config`），3 输出目录已存在（加 `--overwrite` 覆盖上一次 `vc_redact` 的输出，规则见 [cpp/README.md](../cpp/README.md)）。

`scripts/redact.sh`：0 全部成功；1 有视频失败（其余照常处理，最后列出失败的文件）；2 参数或环境错误（找不到二进制、模型、`VC_JOBS` 越界等）；130 / 143 被 SIGINT / SIGTERM 打断。
