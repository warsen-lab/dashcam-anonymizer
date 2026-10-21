# Linux + NVIDIA 部署：本机可验证部分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按已批准的设计 `docs/superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md`（下称"本设计"）实现开发机（Apple Silicon Mac，没有 NVIDIA GPU）上能验证的四个阶段：1a（`VC_WITH_CUDA` 开关骨架、输出一律 H.264 与码率系数、`--ep` / `--hwaccel` 新取值与 `auto` 回退逻辑、TensorRT 缓存目录、新配置键、`vc_eval integrity` 的 H.264 与 Level 检查）、1b 的静态部分（`docker/Dockerfile`、`scripts/setup_linux.sh`）、2a（马赛克网格纯函数，CPU 马赛克改用它且逐字节不变）、3a（`scripts/redact.sh` 的 `VC_JOBS` / `VC_GPUS` 与 `batch.json`）。

**Architecture:** 不加新的进程或线程模型。`cpp/src/common/platform.h`（新）给出编译平台和是否带 CUDA，`--ep auto`、`--hwaccel auto` 与 `vc_redact` 的默认值按它取；`detect/ort_session.*` 把"按顺序尝试执行提供者、跳过的写一行 stderr"抽成可用假后端测试的 `tryProviders()`，TensorRT / CUDA 在本计划里一律报不可用（真正的会话设置在 GPU 计划 1c）；`io/video_writer.*` 只编 H.264（`h264_nvenc` → `h264_videotoolbox`（仅 macOS）→ `libx264`），按 H.264 Table A-1 给编码器指定 Level，输入一律转成 8 位 4:2:0；`mask/mosaic_grid.h`（新）是 `__host__ __device__` 纯函数（没有 CUDA 时宏为空），`mosaic.cpp` 改用它；`scripts/redact.sh` 改成按槽位并行、按 GPU 轮流分配，结果交给 `scripts/batch_json.py`（新，只用标准库）写 `batch.json`；Linux 交付是 `scripts/setup_linux.sh`（固定版本与 sha256 表）和两阶段 `docker/Dockerfile`，本机只做静态检查。

**基线：** 本计划按 `main` 的 0f2aa85 写成（其中车牌的马赛克已改成不向外对齐网格、只打在框内，`MosaicRect` 与 `MaskBox` 多了 `snap`）。开工时 `main` 若又有新提交，先对照本计划里"把 … 换成"的原文，有出入就以当前代码为准、按同样的意图改。

**Tech Stack:** C++20、CMake ≥ 3.24、FFmpeg 8（本机 8.1.2）、onnxruntime 1.30.0、nlohmann/json 3.12.0、GoogleTest；Python 3.14、pytest（仓库根目录 `.venv`）；bash（本机 `/bin/bash` 是 3.2，脚本要在 3.2 和 Ubuntu 的 bash 5 上都能跑）；shellcheck（本机已装）；hadolint（本机没装，跳过）。

**GPU 部分另起计划。** 本设计 §6 的 1c、2b、3b 要在 Linux + NVIDIA 机器上做和验收，等有机器再写计划，本计划不含它们的任务。它们将覆盖：

- **1c**：TensorRT（FP16，`trt_fp16`）与 CUDA 执行提供者的会话设置（替换本计划 `gpuProviderUnavailableReason()` 里"stage 1c"那一支），用本计划的 `tensorRtCacheDirectory()` 建引擎缓存（含"带缓存失败就不带缓存再试一次"）；NVDEC 解码拷回内存、NVENC 编码在目标机上的实测（本计划已写好通用代码，见"已定的做法"）；`setup_linux.sh` / Dockerfile 里待填的 sha256、基础镜像 digest、`TENSORRT_VERSION` 在目标机上查实并提交，镜像与裸机构建；`VC_WITH_CUDA=ON` 的全部过滤测试组在目标机上跑通；TensorRT 与 CPU 后端检测框的一致性门槛。
- **2b**：CUDA 前处理核函数（学生 NV12 → RGB 浮点 1920×1088、YuNet NV12 → BGR 浮点补到 32 的倍数）、IoBinding、用本计划 `mosaic_grid.h` 的 CUDA 马赛克核函数（与 CPU 逐像素相同）、NVDEC → NVENC 硬件帧贯通、GPU 路径的强制关键帧上限 max(2 × 步长, 24)、`pipeline: "gpu"` 与退路；`Gpu*` 测试套件。
- **3b**：`scripts/bench_gpu.sh`、`scripts/accept_gpu.sh`（并把它们加进 Dockerfile 的运行阶段）、目标机压测、调参文档的实测数字、验收（4K 合计 ≥ 360 fps）。

## Global Constraints

- **输出一律 H.264**（本设计 §1、§3）：MP4，音频照旧原样拷贝，macOS 也改成这样。编码器按顺序尝试 `h264_nvenc` →（macOS）`h264_videotoolbox` → `libx264`，实际用的写在 `stats.json` 的 `encoder`。H.264 原片沿用原码率；HEVC（及其他非 H.264）原片乘 `output_bitrate_factor_hevc`（默认 1.5，范围 [1, 4]）；读不到码率时仍按 8 Mb/s × 系数。原片是 10 位或 4:2:2 时转成 8 位 4:2:0 再编码，stderr 一行。4K60 需要 Level 5.2，`vc_eval integrity` 检查输出是 H.264、Level 不低于其分辨率与帧率所需。
- **`--ep`**：`auto | tensorrt | cuda | coreml | cpu`，默认 `auto`；`auto` 在 Linux 上 TensorRT → CUDA → CPU，macOS 上 CoreML → CPU，回退时 stderr 一行原因；显式指定的后端用不了时 stderr 提示后退到 CPU（沿用现在 `coreml` 的做法）；本构建没编进的提供者报不可用、试下一个。TensorRT 引擎缓存目录 `$VC_CACHE/trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`。
- **`--hwaccel`** 增加 `cuda`；Linux 上默认 `auto`（先试 NVDEC，打不开退回软件解码，stderr 一行）；macOS 上 `cuda` 报不可用、退回软件解码并在 stderr 说明，macOS 默认仍是 `none`。
- **新配置键**：`nvenc_preset`（默认 `p4`，`p1`–`p7`）、`trt_fp16`（默认 `true`）、`ort_intra_threads`（默认 0 = onnxruntime 自己定；> 0 设 SessionOptions 的 intra-op 线程数，macOS 上也生效）、`output_bitrate_factor_hevc`（见上）。`configs/redact.json` 与内置默认值逐键相等（已有测试保证）。
- **CMake**：`VC_WITH_CUDA` 默认 `OFF`；`ON` 时启用 CUDA 语言，`CMAKE_CUDA_ARCHITECTURES` 默认 `86;89`，定义编译宏 `VC_WITH_CUDA`。`OFF` 时不链接任何 CUDA 库，除"输出改成 H.264"以外行为与现在逐字节相同（`masks.jsonl`、`detections.jsonl` 逐字节相同；H.264 原片用 `--encoder software` 时解码后的每帧也相同）。
- **Linux 交付**：Ubuntu 22.04 / 24.04，NVIDIA 驱动 ≥ 550，CUDA 12，onnxruntime 1.30.0 的 Linux GPU 包，FFmpeg 8 从源码编（NVENC / NVDEC + libx264）；`setup_linux.sh` 不设 `VC_ALLOW_DOWNLOAD=1` 拒绝运行，下载按固定 sha256 表校验，缓存在所有 checkout 共享的 `~/.cache/video-compliance/`；Docker 两阶段（`nvidia/cuda:12.x` devel → runtime），基础镜像按 digest 锁定。**不许编造 sha256 或 digest**：不知道的留空（脚本拒绝空值）或做成必填的构建参数，由目标机计划查实后填写。
- **批量**：`VC_JOBS`（列了 GPU 时每块卡同时几个 `vc_redact`，默认 4；没列 GPU 时表示同时几个进程，默认 1），`VC_GPUS`（如 `0,1`，按视频顺序轮流分配，用 `CUDA_VISIBLE_DEVICES` 绑卡），输出目录下写 `batch.json`（每段的耗时、fps、后端、编码器、路径与退出状态，以及合计吞吐）。单段失败不影响其他段，最后汇总失败列表，退出码 1；同名输出拒绝覆盖。
- **语言**：计划正文简体中文；代码、标识符、命令、提交信息用英文。
- C++20，`-Wall -Wextra -Wpedantic` 零警告（本机 Apple clang 对"位置初始化漏掉一个没有默认初始化器的成员"会报 `-Wmissing-field-initializers`，所以新结构体的每个成员都写默认初始化器，例如 `std::string formatNote{};`）。Debug 测试构建在 `build/cpp`：`cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug`；Release 在 `build/cpp-release`：`cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF`。
- **完全不联网。** 每条构建、测试、运行命令都包在沙箱里：下文的命令块先定义 `SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')`，再用 `"${SB[@]}" …`。onnxruntime 遥测保持关闭（`cpp/src/detect/ort_session.cpp` 已关，不要动）。不安装、不升级任何包；`setup_linux.sh` 和 `docker/build.sh` 在本机只跑到它们的拒绝分支。
- **只跑过滤后的 gtest**（`./build/cpp/tests/vc_tests --gtest_filter=…`），不得运行不带过滤器的 `vc_tests` 或 `ctest`。`cpp/README.md` 的六组过滤命令在本计划里扩成包含新套件 `BuildConfig`（第一组）、`ExecutionProvider`（第四组）、`MosaicGrid`（第五组），六组之并仍等于全部测试。
- **只跑本任务改动的 pytest 文件**，在 `py/` 目录里：`/Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/<file>.py -p no:cacheprovider -q`（worktree 里没有 `.venv`，所以用主 checkout 的绝对路径）。不得运行不带路径的 pytest。
- **提交：** conventional commits，不加 Co-Authored-By 或任何署名行；不 push；绝不 `git stash`。
- **本机一次只跑一个重任务**：构建、`vc_redact`、`vc_eval` 不并行，也不和别的会话的重任务重叠。实现任务里不在真实视频上跑 `vc_redact`；只有 Task 10（控制者的本机验收）用 `data/` 里的视频。
- **工作位置：** 全部任务在 worktree 分支 `feat/linux-local` 上做（superpowers:using-git-worktrees），下文的相对路径都相对 worktree 根目录。`data/` 不在 git 里、只在主 checkout：Task 10 用绝对路径 `MAIN=/Users/warsen/Projects/video-compliance-process` 读它，输出写在 worktree 的 `build/accept/`（`build/` 已被 `.gitignore` 忽略）。
- **没提到的行为一律不变：** 补框、确认、马赛克规则与默认值（2026-10-19 标定）；"缓冲满 120 帧强制出一个关键帧"（GPU 路径的改动属于 2b）；`--encoder auto|software` 的含义（`software` 只用 libx264）；`vc_detect` 的 `--ep` 默认仍是 `cpu`；`vc_extract` 的 `--hwaccel` 默认仍是 `auto`；输出先写 `.partial`、失败删光本次文件的规则；`redact.sh` 现有的环境变量与沙箱规则。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `cpp/CMakeLists.txt` | 改（Task 1） | `VC_WITH_CUDA` 开关（`check_language(CUDA)`、`CMAKE_CUDA_ARCHITECTURES` 默认 `86;89`、宏 `VC_WITH_CUDA=1`）；警告选项只给 C++ 源文件 |
| `cpp/cmake/ThirdParty.cmake` | 改（Task 1） | onnxruntime 包按平台选目录：macOS `onnxruntime-osx-arm64-1.30.0`，其他 `onnxruntime-linux-x64-gpu-1.30.0` |
| `cpp/src/common/platform.h` | 新建（Task 1） | `Platform`、`buildPlatform()`、`builtWithCuda()` |
| `cpp/src/common/host_device.h` | 新建（Task 1） | `VC_HOST_DEVICE`：nvcc 下 `__host__ __device__`，否则为空 |
| `cpp/tests/build_config_test.cpp` | 新建（Task 1） | `BuildConfig.*` |
| `cpp/apps/vc_redact/redact_config.h`, `.cpp`, `configs/redact.json` | 改（Task 2） | `output_bitrate_factor_hevc`、`nvenc_preset`、`trt_fp16`、`ort_intra_threads` |
| `cpp/tests/redact_config_test.cpp` | 改（Task 2） | 新键的默认值、读取、范围与消息 |
| `cpp/src/io/video_writer.h`, `.cpp` | 改（Task 3） | 只编 H.264：`h264EncoderCandidates()`、`h264OutputBitRate()`、`h264LevelIdc()`、`h264FormatNote()`；VideoToolbox / NVENC 的 Level、NVENC 的 preset；输入转 8 位 4:2:0 |
| `cpp/apps/vc_redact/redact_job.h`, `.cpp` | 改（Task 3、4、5） | 码率系数、`stats.json` 的 `output_bit_rate` 与 `pipeline`、10 位 / 4:2:2 的 note（Task 3）；`--ep` 默认 `auto`、会话设置传给检测器（Task 4）；`defaultRedactHwAccel()`（Task 5） |
| `cpp/tests/test_media.h`, `.cpp` | 改（Task 3） | `yuv422Video()`；`VideoPresentation` 加 `codec`、`level`、`pixelFormat` |
| `cpp/tests/video_writer_test.cpp`, `redact_job_test.cpp` | 改（Task 3） | H.264 输出、4:2:0、Level、码率、note |
| `cpp/src/detect/execution_provider.h` | 改（Task 4） | `ExecutionProvider` 五个取值、`providerLabel()`、`providerOrder()`、`SessionSettings`、`tensorRtCacheDirectory()` |
| `cpp/src/detect/ort_session.h`, `.cpp` | 改（Task 4） | `tryProviders()`、`gpuProviderUnavailableReason()`、`SessionCaches`、新的 `createOrtSession()` |
| `cpp/src/detect/onnx_yolo_detector.*`, `yunet_detector.*` | 改（Task 4） | 构造函数多一个 `const SessionSettings& = {}`；缓存目录改成按需计算 |
| `cpp/tests/execution_provider_test.cpp` | 新建（Task 4） | `ExecutionProvider.*`（假后端测回退顺序与 stderr 行、TensorRT 缓存目录） |
| `cpp/apps/vc_redact/redact_cli.cpp`, `cpp/apps/vc_detect/detect_cli.cpp` | 改（Task 4、5） | 帮助文本 |
| `cpp/tests/onnx_yolo_detector_test.cpp`, `detect_cli_test.cpp`, `redact_cli_test.cpp` | 改（Task 4、5） | 新取值；原来"cuda 被拒"的用例换成真正未知的名字 |
| `cpp/src/io/video_reader.h`, `.cpp` | 改（Task 5） | `HwAccel::Cuda`、`HwDecodePlan` / `hwDecodePlan()`、CUDA 设备与硬件帧下载 |
| `cpp/apps/vc_extract/extract_cli.cpp`, `cpp/tests/video_reader_test.cpp`, `extract_cli_test.cpp` | 改（Task 5） | 帮助文本与测试 |
| `py/vc_eval/integrity.py` | 改（Task 6） | `level` 字段、`H264_LEVELS`、`required_h264_level()`、检查项 `h264` / `h264_level` |
| `py/vc_eval/report.py`, `py/vc_eval/cli.py` | 改（Task 6） | 完整性表多两列；子命令帮助 |
| `py/tests/test_eval_integrity.py`, `py/tests/test_eval_run.py` | 改（Task 6） | 新检查项 |
| `cpp/src/mask/mosaic_grid.h` | 新建（Task 7） | `PixelRect`（从 `mosaic.h` 搬来）与 `grid::` 纯函数 |
| `cpp/src/mask/mosaic.h`, `.cpp` | 改（Task 7） | 改用 `grid::maskCells()` / `cellRect()` / `roundedMean()` |
| `cpp/tests/mosaic_grid_test.cpp` | 新建（Task 7） | `MosaicGrid.*`，含与旧实现逐字节比对 |
| `scripts/redact.sh` | 改（Task 8） | 并行槽位、按 GPU 轮流、同名预先拒绝、调用 `batch_json.py` |
| `scripts/batch_json.py` | 新建（Task 8） | 写 `<out>/batch.json`（只用标准库） |
| `py/tests/test_redact_script.py` | 新建（Task 8） | 用假 `vc_redact` 测 `redact.sh` |
| `scripts/setup_linux.sh` | 新建（Task 9） | 裸机 / Docker 构建阶段的安装与构建；`--check-pins` |
| `docker/Dockerfile`, `docker/build.sh`, `docker/entrypoint.sh`, `.dockerignore` | 新建（Task 9） | 两阶段镜像；digest 必填的构建入口；`/in /out` 参数顺序；构建上下文排除 `data/` |
| `py/tests/test_linux_deploy.py` | 新建（Task 9） | 拒绝分支、固定表、Dockerfile 的静态检查，shellcheck（hadolint 有则跑） |
| `cpp/README.md`, `docs/USAGE.md`, `py/README.md` | 改（各任务顺带） | 新开关、新取值、H.264 输出、新配置键、批量、Linux 一节 |

任务顺序：Task 1 → 2 → 3 → 4 → 5（C++，后一个依赖前一个的头文件或测试夹具），Task 6（Python，独立），Task 7（C++，只依赖 Task 1 的 `host_device.h`），Task 8（脚本），Task 9（依赖 Task 8 的 `scripts/batch_json.py`，Dockerfile 要拷它），Task 10 最后。

---

### Task 1: `VC_WITH_CUDA` 开关骨架与平台判断

**Files:**
- Create: `cpp/src/common/platform.h`, `cpp/src/common/host_device.h`
- Create: `cpp/tests/build_config_test.cpp`
- Modify: `cpp/CMakeLists.txt`（`set(CMAKE_CXX_STANDARD 20)` 之后到 `add_compile_options` 那几行）
- Modify: `cpp/cmake/ThirdParty.cmake`（`VC_ONNXRUNTIME_ROOT` 的默认值、末尾 RPATH 注释）
- Modify: `cpp/tests/CMakeLists.txt`（源文件列表、`target_compile_definitions`）
- Modify: `cpp/README.md`（"依赖"的 onnxruntime 一条、"构建"、"测试"第一组过滤命令）

**Interfaces:**
- Consumes: 无。
- Produces（Task 3、4、5、7 使用）:
  - `enum class Platform { Linux, MacOS, Other };`（`common/platform.h`，命名空间 `vc`）
  - `constexpr Platform buildPlatform();` —— `__APPLE__` → `MacOS`，`__linux__` → `Linux`，其余 `Other`。
  - `constexpr bool builtWithCuda();` —— 定义了宏 `VC_WITH_CUDA` 时为 true。
  - `VC_HOST_DEVICE`（`common/host_device.h`）—— `__CUDACC__` 下是 `__host__ __device__`，否则为空。
  - CMake：`option(VC_WITH_CUDA … OFF)`；`ON` 时 `add_compile_definitions(VC_WITH_CUDA=1)`；测试目标有编译宏 `VC_TEST_EXPECT_CUDA`（`ON` 为 1，`OFF` 为 0）。

- [ ] **Step 1: 写失败的测试**

新建 `cpp/tests/build_config_test.cpp`：

```cpp
#include <gtest/gtest.h>

#include "common/host_device.h"
#include "common/platform.h"

namespace vc {
namespace {

VC_HOST_DEVICE constexpr int twice(int value) { return 2 * value; }

// Without nvcc the macro is empty: a VC_HOST_DEVICE function is an ordinary (here constexpr) host function.
static_assert(twice(21) == 42);

TEST(BuildConfig, KnowsThePlatformItWasBuiltFor) {
#if defined(__APPLE__)
  EXPECT_EQ(buildPlatform(), Platform::MacOS);
#elif defined(__linux__)
  EXPECT_EQ(buildPlatform(), Platform::Linux);
#else
  EXPECT_EQ(buildPlatform(), Platform::Other);
#endif
}

// VC_TEST_EXPECT_CUDA is 1 exactly when CMake was configured with -DVC_WITH_CUDA=ON.
TEST(BuildConfig, KnowsWhetherCudaIsBuiltIn) { EXPECT_EQ(builtWithCuda(), VC_TEST_EXPECT_CUDA == 1); }

}  // namespace
}  // namespace vc
```

把 `cpp/tests/CMakeLists.txt` 里的：

```cmake
  box_timeline_test.cpp
  detect_rate_test.cpp
```

换成：

```cmake
  box_timeline_test.cpp
  build_config_test.cpp
  detect_rate_test.cpp
```

把 `cpp/tests/CMakeLists.txt` 里的：

```cmake
  VC_REPO_DIR="${PROJECT_SOURCE_DIR}/.."
)
```

换成：

```cmake
  VC_REPO_DIR="${PROJECT_SOURCE_DIR}/.."
  VC_TEST_EXPECT_CUDA=$<BOOL:${VC_WITH_CUDA}>
)
```

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`build_config_test.cpp:3:10: fatal error: 'common/host_device.h' file not found`。

- [ ] **Step 3: 平台头文件与 CMake 开关**

新建 `cpp/src/common/platform.h`：

```cpp
#pragma once

// What this build runs on and what it was built with (spec 2026-10-20 §2-§3): the execution provider order of
// --ep auto, the decoder of --hwaccel auto and vc_redact's defaults depend on it.

namespace vc {

enum class Platform { Linux, MacOS, Other };

constexpr Platform buildPlatform() {
#if defined(__APPLE__)
  return Platform::MacOS;
#elif defined(__linux__)
  return Platform::Linux;
#else
  return Platform::Other;
#endif
}

// True when CMake configured the build with -DVC_WITH_CUDA=ON.
constexpr bool builtWithCuda() {
#if defined(VC_WITH_CUDA)
  return true;
#else
  return false;
#endif
}

}  // namespace vc
```

新建 `cpp/src/common/host_device.h`：

```cpp
#pragma once

// Marks a function that CUDA kernels call as well as host code (spec 2026-10-20 §4: the mosaic grid is shared by
// the CPU and the GPU): __host__ __device__ under nvcc, nothing for the host compiler, so a VC_WITH_CUDA=OFF build
// compiles the same code as before.
#if defined(__CUDACC__)
#define VC_HOST_DEVICE __host__ __device__
#else
#define VC_HOST_DEVICE
#endif
```

把 `cpp/CMakeLists.txt` 里的：

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall -Wextra -Wpedantic)
```

换成：

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The GPU code paths (spec 2026-10-20): CUDA kernels for NVIDIA cards. OFF (the default, macOS and every local
# test) compiles and links no CUDA at all; the GPU code is behind #ifdef VC_WITH_CUDA.
option(VC_WITH_CUDA "Build the CUDA code paths for NVIDIA GPUs (needs the CUDA 12 toolkit)" OFF)
if(VC_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "VC_WITH_CUDA=ON needs the CUDA toolkit (nvcc), which CMake did not find: install CUDA 12 "
                        "(scripts/setup_linux.sh) or configure with -DVC_WITH_CUDA=OFF")
  endif()
  # 86: A10 / RTX 3090, 89: L4 / L40S / RTX 4090 (spec 2026-10-20 §2).
  set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures to compile the kernels for")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(VC_WITH_CUDA=1)
  message(STATUS "VC_WITH_CUDA: ON (CUDA architectures ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# C++ only: nvcc takes host-compiler warnings through -Xcompiler, which the CUDA sources set themselves.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
```

把 `cpp/cmake/ThirdParty.cmake` 里的：

```cmake
set(VC_ONNXRUNTIME_ROOT "${VC_THIRD_PARTY_DIR}/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "Unpacked ONNX Runtime package (include/, lib/)")
```

换成：

```cmake
# The package for the platform: macOS arm64 (CoreML EP), Linux x64 GPU (CUDA and TensorRT EPs, CPU as well).
if(APPLE)
  set(_vc_onnxruntime_package "onnxruntime-osx-arm64-1.30.0")
else()
  set(_vc_onnxruntime_package "onnxruntime-linux-x64-gpu-1.30.0")
endif()
set(VC_ONNXRUNTIME_ROOT "${VC_THIRD_PARTY_DIR}/${_vc_onnxruntime_package}"
    CACHE PATH "Unpacked ONNX Runtime package (include/, lib/)")
```

把 `cpp/cmake/ThirdParty.cmake` 里的：

```cmake
# libonnxruntime's install name is @rpath/libonnxruntime.1.dylib.
list(APPEND CMAKE_BUILD_RPATH "${VC_ONNXRUNTIME_ROOT}/lib")
```

换成：

```cmake
# libonnxruntime's install name is @rpath/libonnxruntime.1.dylib (on Linux the binaries find libonnxruntime.so.1
# through the same RPATH entry).
list(APPEND CMAKE_BUILD_RPATH "${VC_ONNXRUNTIME_ROOT}/lib")
```

- [ ] **Step 4: 构建并跑测试，确认 `OFF` 的编译选项没变、`ON` 找不到 nvcc 时报错清楚**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*:BuildConfig.*'
grep -c '"command"' build/cpp/compile_commands.json
grep -c -- '-Wall -Wextra -Wpedantic' build/cpp/compile_commands.json
"${SB[@]}" cmake -S cpp -B build/cpp-cuda-check -DVC_WITH_CUDA=ON -DVC_BUILD_TESTS=OFF; echo "configure exit $?"
rm -rf build/cpp-cuda-check
```

预期：警告计数 `0`；`[  PASSED  ] 18 tests.`（第一组原有 16 个加 `BuildConfig` 2 个）；两个 `grep -c` 打印同一个数（每个编译命令都还带着 `-Wall -Wextra -Wpedantic`，与改动前的顺序相同）；`ON` 的配置打印 `VC_WITH_CUDA=ON needs the CUDA toolkit (nvcc), which CMake did not find` 并以 `configure exit 1` 结束。

- [ ] **Step 5: 文档**

把 `cpp/README.md` 里的：

```markdown
- onnxruntime 1.30.0 官方预编译包（macOS arm64，含 CoreML EP）与 nlohmann/json 3.12.0 单头文件。
```

换成：

```markdown
- onnxruntime 1.30.0 官方预编译包（macOS arm64，含 CoreML EP；在 Linux 上 CMake 默认找 x64 GPU 包 `onnxruntime-linux-x64-gpu-1.30.0`，含 CUDA 与 TensorRT EP）与 nlohmann/json 3.12.0 单头文件。
```

把 `cpp/README.md` 里的：

```markdown
不指定 `CMAKE_BUILD_TYPE` 时，单配置生成器（Makefile、Ninja）默认按 Release 构建。
```

换成：

```markdown
不指定 `CMAKE_BUILD_TYPE` 时，单配置生成器（Makefile、Ninja）默认按 Release 构建。

`-DVC_WITH_CUDA=ON` 打开 NVIDIA GPU 的代码路径（设计见 `docs/superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md`）：启用 CUDA 语言，`CMAKE_CUDA_ARCHITECTURES` 默认 `86;89`（A10 / RTX 3090，L4 / L40S / RTX 4090），并定义宏 `VC_WITH_CUDA`；找不到 nvcc 时配置阶段直接报错。默认 `OFF`：不编译、不链接任何 CUDA，macOS 和本机测试都用它。目前 `ON` 只多出 CUDA 语言和这个宏，核函数在 GPU 计划里实现。
```

把 `cpp/README.md` 里的：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
```

换成：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*:BuildConfig.*'
```

- [ ] **Step 6: 提交**

```bash
git add cpp/CMakeLists.txt cpp/cmake/ThirdParty.cmake cpp/src/common/platform.h cpp/src/common/host_device.h \
    cpp/tests/build_config_test.cpp cpp/tests/CMakeLists.txt cpp/README.md
git commit -m "build: add the VC_WITH_CUDA option and platform helpers"
```

### Task 2: 新配置键（`output_bitrate_factor_hevc`、`nvenc_preset`、`trt_fp16`、`ort_intra_threads`）

**Files:**
- Modify: `cpp/apps/vc_redact/redact_config.h`（includes、`RedactConfig` 末尾、`parseRedactConfig` 的注释）
- Modify: `cpp/apps/vc_redact/redact_config.cpp`（`kKeys`、整数设置、`stringSetting`、校验、读取、序列化）
- Modify: `configs/redact.json`
- Test: `cpp/tests/redact_config_test.cpp`
- Modify: `docs/USAGE.md`（§4 调参表）、`cpp/README.md`（"要点"）

**Interfaces:**
- Consumes: 无。
- Produces（Task 3、4 使用）:
  - `RedactConfig::outputBitrateFactorHevc`（`double`，默认 1.5，[1, 4]）
  - `RedactConfig::nvencPreset`（`std::string`，默认 `"p4"`，`kNvencPresets` 之一）
  - `RedactConfig::trtFp16`（`bool`，默认 `true`）
  - `RedactConfig::ortIntraThreads`（`int`，默认 0，[0, 1024]）
  - `inline constexpr std::array<std::string_view, 7> kNvencPresets = {"p1", …, "p7"};`
  - JSON 键 `output_bitrate_factor_hevc`、`nvenc_preset`、`trt_fp16`、`ort_intra_threads`，按这个顺序排在 `hold_lost_ghosts` 之后（`redactConfigToJson` 与 `configs/redact.json`）。

- [ ] **Step 1: 写失败的测试**

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts"};
```

换成：

```cpp
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts",    "output_bitrate_factor_hevc", "nvenc_preset",
    "trt_fp16",            "ort_intra_threads"};
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
  EXPECT_EQ(c.timeline.holdLostGhosts, d.timeline.holdLostGhosts);
}
```

换成：

```cpp
  EXPECT_EQ(c.timeline.holdLostGhosts, d.timeline.holdLostGhosts);
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, d.outputBitrateFactorHevc);
  EXPECT_EQ(c.nvencPreset, d.nvencPreset);
  EXPECT_EQ(c.trtFp16, d.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, d.ortIntraThreads);
}
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
  c.timeline.holdLostGhosts = true;
  return c;
}
```

换成：

```cpp
  c.timeline.holdLostGhosts = true;
  c.outputBitrateFactorHevc = 2.25;
  c.nvencPreset = "p1";
  c.trtFp16 = false;
  c.ortIntraThreads = 3;
  return c;
}
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
      R"({"large_face_area_fraction": 1.0000001})", R"({"large_face_min_score": 1.0000001})",
  };
```

换成：

```cpp
      R"({"large_face_area_fraction": 1.0000001})", R"({"large_face_min_score": 1.0000001})",
      // The Linux settings (spec 2026-10-20): a bitrate factor in [1, 4], a preset p1-p7, a boolean, a thread count.
      R"({"output_bitrate_factor_hevc": 0.99})",    R"({"output_bitrate_factor_hevc": 4.01})",
      R"({"output_bitrate_factor_hevc": "1.5"})",   R"({"output_bitrate_factor_hevc": 1e400})",
      R"({"output_bitrate_factor_hevc": null})",
      R"({"nvenc_preset": "p0"})",                  R"({"nvenc_preset": "p8"})",
      R"({"nvenc_preset": "P4"})",                  R"({"nvenc_preset": "fast"})",
      R"({"nvenc_preset": 4})",                     R"({"nvenc_preset": null})",
      R"({"trt_fp16": 1})",                         R"({"trt_fp16": "true"})",
      R"({"trt_fp16": null})",
      R"({"ort_intra_threads": -1})",               R"({"ort_intra_threads": 1025})",
      R"({"ort_intra_threads": 2.5})",              R"({"ort_intra_threads": "4"})",
      R"({"ort_intra_threads": 4294967297})",       R"({"ort_intra_threads": null})",
  };
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
      {"large_face_min_score", [](RedactConfig& c, double v) { c.largeFaceMinScore = static_cast<float>(v); }},
  };
```

换成：

```cpp
      {"large_face_min_score", [](RedactConfig& c, double v) { c.largeFaceMinScore = static_cast<float>(v); }},
      {"output_bitrate_factor_hevc", [](RedactConfig& c, double v) { c.outputBitrateFactorHevc = v; }},
  };
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 19u);
```

换成：

```cpp
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 23u);
```

把 `cpp/tests/redact_config_test.cpp` 里的：

```cpp
TEST(RedactConfig, DumpParsesBackToTheSameConfig) {
```

换成：

```cpp
// spec 2026-10-20 §3, §5: an HEVC source's H.264 gets 1.5 times its bitrate, NVENC runs preset p4, TensorRT in FP16,
// and onnxruntime keeps its own thread count.
TEST(RedactConfig, LinuxSettingsDefaults) {
  const RedactConfig c;
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, 1.5);
  EXPECT_EQ(c.nvencPreset, "p4");
  EXPECT_TRUE(c.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, 0);
  const nlohmann::ordered_json j = redactConfigToJson(c);
  EXPECT_EQ(j.at("output_bitrate_factor_hevc").dump(), "1.5");
  EXPECT_EQ(j.at("nvenc_preset").dump(), "\"p4\"");
  EXPECT_EQ(j.at("trt_fp16").dump(), "true");
  EXPECT_EQ(j.at("ort_intra_threads").dump(), "0");
}

TEST(RedactConfig, ReadsTheLinuxSettings) {
  const RedactConfig c = parseRedactConfig(
      R"({"output_bitrate_factor_hevc": 2, "nvenc_preset": "p7", "trt_fp16": false, "ort_intra_threads": 2})", "t");
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, 2.0);
  EXPECT_EQ(c.nvencPreset, "p7");
  EXPECT_FALSE(c.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, 2);
  for (const char* preset : {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}) {
    EXPECT_EQ(parseRedactConfig(std::string(R"({"nvenc_preset": ")") + preset + "\"}", "t").nvencPreset, preset);
  }
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"output_bitrate_factor_hevc": 1})", "t").outputBitrateFactorHevc, 1.0);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"output_bitrate_factor_hevc": 4})", "t").outputBitrateFactorHevc, 4.0);
  EXPECT_EQ(parseRedactConfig(R"({"ort_intra_threads": 1024})", "t").ortIntraThreads, 1024);
}

// The messages name the key and the allowed values.
TEST(RedactConfig, LinuxSettingMessagesGiveTheRule) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"({"output_bitrate_factor_hevc": 5})", "cfg: output_bitrate_factor_hevc must be in [1, 4]"},
      {R"({"nvenc_preset": "p9"})", "cfg: nvenc_preset must be one of p1 ... p7, got 'p9'"},
      {R"({"nvenc_preset": 4})", "cfg: nvenc_preset must be a string"},
      {R"({"trt_fp16": 1})", "cfg: trt_fp16 must be true or false"},
      {R"({"ort_intra_threads": -1})", "cfg: ort_intra_threads must be an integer in [0, 1024]"},
      {R"({"mosaic_min_block_px": 0})", "cfg: mosaic_min_block_px must be an integer in [1, 4096]"},
  };
  for (const auto& [text, message] : cases) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), message);
    }
  }
}

TEST(RedactConfig, ValidateChecksTheLinuxSettings) {
  RedactConfig c;
  c.nvencPreset = "p0";
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.ortIntraThreads = 1025;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.outputBitrateFactorHevc = 0.5;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
}

TEST(RedactConfig, DumpParsesBackToTheSameConfig) {
```

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`redact_config_test.cpp` 报 `no member named 'outputBitrateFactorHevc' in 'vc::RedactConfig'`（以及 `nvencPreset`、`trtFp16`、`ortIntraThreads`）。

- [ ] **Step 3: 实现**

把 `cpp/apps/vc_redact/redact_config.h` 里的：

```cpp
#include <filesystem>
#include <string>
#include <vector>
```

换成：

```cpp
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
```

把 `cpp/apps/vc_redact/redact_config.h` 里的：

```cpp
  double largeFaceAreaFraction = 0.05;
  float largeFaceMinScore = 0.6f;
};
```

换成：

```cpp
  double largeFaceAreaFraction = 0.05;
  float largeFaceMinScore = 0.6f;
  // Linux deployment (spec 2026-10-20 §3, §5). The output is always H.264: a source in another codec (HEVC, say) is
  // encoded at its own bitrate times outputBitrateFactorHevc, in [1, 4]; an H.264 source keeps its bitrate.
  double outputBitrateFactorHevc = 1.5;
  // h264_nvenc's preset, "p1" (fastest) to "p7" (best); only used when NVENC encodes.
  std::string nvencPreset = "p4";
  // The TensorRT execution provider in FP16 (used once TensorRT runs: stage 1c of the Linux plan).
  bool trtFp16 = true;
  // onnxruntime's intra-op CPU threads per session, in [0, 1024]: 0 keeps onnxruntime's default, more sets it
  // (smaller when several vc_redact processes share a machine). Applies on every platform.
  int ortIntraThreads = 0;
};

// The values nvenc_preset takes.
inline constexpr std::array<std::string_view, 7> kNvencPresets = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"};
```

把 `cpp/apps/vc_redact/redact_config.h` 里的：

```cpp
// Keys may be omitted; unknown keys, the removed max_detect_fps and motion_threshold, wrong types and
// out-of-range values throw VcError("<origin>: …"). detect_fps is in [5, 60]; shrink_per_frame is in (0, 1],
// max_mask_area_ratio >= 1; mosaic_min_block_px and mosaic_blocks_per_long_side must be JSON integers in
// [1, 4096]; every other value must be a finite number except hold_lost_ghosts, which must be a JSON boolean.
// face_score_threshold, face_confirm_score, plate_confirm_score and large_face_min_score are in [0, 1],
// face_nms_iou and large_face_area_fraction in (0, 1].
```

换成：

```cpp
// Keys may be omitted; unknown keys, the removed max_detect_fps and motion_threshold, wrong types and
// out-of-range values throw VcError("<origin>: …"). detect_fps is in [5, 60]; shrink_per_frame is in (0, 1],
// max_mask_area_ratio >= 1; mosaic_min_block_px and mosaic_blocks_per_long_side must be JSON integers in
// [1, 4096], ort_intra_threads a JSON integer in [0, 1024]; hold_lost_ghosts and trt_fp16 must be JSON booleans,
// nvenc_preset one of the strings "p1" to "p7"; every other value must be a finite number.
// face_score_threshold, face_confirm_score, plate_confirm_score and large_face_min_score are in [0, 1],
// face_nms_iou and large_face_area_fraction in (0, 1], output_bitrate_factor_hevc in [1, 4].
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
constexpr std::array<std::string_view, 19> kKeys = {
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "shrink_per_frame",
    "max_mask_area_ratio", "mosaic_min_block_px",    "mosaic_blocks_per_long_side",
    "face_score_threshold", "face_nms_iou",       "face_confirm_score",
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts"};
```

换成：

```cpp
constexpr std::array<std::string_view, 23> kKeys = {
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "shrink_per_frame",
    "max_mask_area_ratio", "mosaic_min_block_px",    "mosaic_blocks_per_long_side",
    "face_score_threshold", "face_nms_iou",       "face_confirm_score",
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts",    "output_bitrate_factor_hevc", "nvenc_preset",
    "trt_fp16",            "ort_intra_threads"};
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
// Mosaic block settings are small positive integers; anything larger is a typo, not a block size.
constexpr double kMaxMosaicSetting = 4096;
```

换成：

```cpp
// Mosaic block settings are small positive integers; anything larger is a typo, not a block size.
constexpr int kMaxMosaicSetting = 4096;
// More onnxruntime threads than this is a typo, not a thread count.
constexpr int kMaxOrtIntraThreads = 1024;
constexpr double kMinBitrateFactor = 1.0;
constexpr double kMaxBitrateFactor = 4.0;
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
// The setting as an int: `key` must be a JSON integer in [1, kMaxMosaicSetting] (a fraction or an
// out-of-range number must not be truncated or overflow into a different value); absent: `fallback`.
int mosaicSetting(const nlohmann::json& j, const char* key, int fallback, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  const std::string rule = origin + ": " + key + " must be an integer in [1, 4096]";
  if (!it->is_number_integer()) throw VcError(rule);
  const double value = it->get<double>();  // exact for every value that can pass the range test
  if (!(value >= 1 && value <= kMaxMosaicSetting)) throw VcError(rule);
  return static_cast<int>(value);
}
```

换成：

```cpp
// The setting as an int: `key` must be a JSON integer in [low, high] (a fraction or an out-of-range number must
// not be truncated or overflow into a different value); absent: `fallback`.
int integerSetting(const nlohmann::json& j, const char* key, int fallback, int low, int high,
                   const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  const std::string rule =
      origin + ": " + key + " must be an integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
  if (!it->is_number_integer()) throw VcError(rule);
  const double value = it->get<double>();  // exact for every value that can pass the range test
  if (!(value >= low && value <= high)) throw VcError(rule);
  return static_cast<int>(value);
}
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
  if (!it->is_boolean()) throw VcError(origin + ": " + key + " must be true or false");
  return it->get<bool>();
}
```

换成：

```cpp
  if (!it->is_boolean()) throw VcError(origin + ": " + key + " must be true or false");
  return it->get<bool>();
}

// The setting as a string: `key` must be a JSON string; absent: `fallback`.
std::string stringSetting(const nlohmann::json& j, const char* key, const std::string& fallback,
                          const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_string()) throw VcError(origin + ": " + key + " must be a string");
  return it->get<std::string>();
}
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
  require(std::isfinite(c.largeFaceMinScore), "large_face_min_score must be finite");
```

换成：

```cpp
  require(std::isfinite(c.largeFaceMinScore), "large_face_min_score must be finite");
  require(std::isfinite(c.outputBitrateFactorHevc), "output_bitrate_factor_hevc must be finite");
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
  require(c.face.topK >= 1, "the face model's top_k must be >= 1");
}
```

换成：

```cpp
  require(c.face.topK >= 1, "the face model's top_k must be >= 1");
  require(c.outputBitrateFactorHevc >= kMinBitrateFactor && c.outputBitrateFactorHevc <= kMaxBitrateFactor,
          "output_bitrate_factor_hevc must be in [1, 4]");
  require(std::find(kNvencPresets.begin(), kNvencPresets.end(), c.nvencPreset) != kNvencPresets.end(),
          "nvenc_preset must be one of p1 ... p7, got '" + c.nvencPreset + "'");
  require(c.ortIntraThreads >= 0 && c.ortIntraThreads <= kMaxOrtIntraThreads, "ort_intra_threads must be in [0, 1024]");
}
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
  c.mosaic.minBlockPx = mosaicSetting(j, "mosaic_min_block_px", c.mosaic.minBlockPx, origin);
  c.mosaic.blocksPerLongSide = mosaicSetting(j, "mosaic_blocks_per_long_side", c.mosaic.blocksPerLongSide, origin);
```

换成：

```cpp
  c.mosaic.minBlockPx = integerSetting(j, "mosaic_min_block_px", c.mosaic.minBlockPx, 1, kMaxMosaicSetting, origin);
  c.mosaic.blocksPerLongSide =
      integerSetting(j, "mosaic_blocks_per_long_side", c.mosaic.blocksPerLongSide, 1, kMaxMosaicSetting, origin);
  c.outputBitrateFactorHevc = numberSetting(j, "output_bitrate_factor_hevc", c.outputBitrateFactorHevc, origin);
  c.nvencPreset = stringSetting(j, "nvenc_preset", c.nvencPreset, origin);
  c.trtFp16 = boolSetting(j, "trt_fp16", c.trtFp16, origin);
  c.ortIntraThreads = integerSetting(j, "ort_intra_threads", c.ortIntraThreads, 0, kMaxOrtIntraThreads, origin);
```

把 `cpp/apps/vc_redact/redact_config.cpp` 里的：

```cpp
  j["hold_lost_ghosts"] = c.timeline.holdLostGhosts;
  return j;
```

换成：

```cpp
  j["hold_lost_ghosts"] = c.timeline.holdLostGhosts;
  j["output_bitrate_factor_hevc"] = c.outputBitrateFactorHevc;
  j["nvenc_preset"] = c.nvencPreset;
  j["trt_fp16"] = c.trtFp16;
  j["ort_intra_threads"] = c.ortIntraThreads;
  return j;
```

把 `configs/redact.json` 里的：

```json
  "hold_lost_ghosts": false
}
```

换成：

```json
  "hold_lost_ghosts": false,
  "output_bitrate_factor_hevc": 1.5,
  "nvenc_preset": "p4",
  "trt_fp16": true,
  "ort_intra_threads": 0
}
```

（`ortIntraThreads` 在 Task 4 接到 onnxruntime 会话，`outputBitrateFactorHevc` 与 `nvencPreset` 在 Task 3 接到编码器；本任务只有解析、校验和序列化。）

- [ ] **Step 4: 跑测试**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:BoundedQueue.*:RedactConfig.*:Records.*'
```

预期：`0`；`[  PASSED  ] 131 tests.`（`RedactConfig` 27 个，其中 `RepositoryFileMatchesTheDefaults` 证明 `configs/redact.json` 与默认值逐键相等）。

- [ ] **Step 5: 文档**

把 `docs/USAGE.md` 里的：

```markdown
| `hold_lost_ghosts` | false | 同一个目标连续两个关键帧都漏检时，第二个间隔是否仍在它最后停留的位置打码（JSON 布尔，不能写 0 / 1）。`true` 把没有框的空档从两个检测间隔缩到一个，代价是消失的目标（包括误检）多留一个间隔的码；`false` 时那个间隔不打码 |
```

换成：

```markdown
| `hold_lost_ghosts` | false | 同一个目标连续两个关键帧都漏检时，第二个间隔是否仍在它最后停留的位置打码（JSON 布尔，不能写 0 / 1）。`true` 把没有框的空档从两个检测间隔缩到一个，代价是消失的目标（包括误检）多留一个间隔的码；`false` 时那个间隔不打码 |
| `output_bitrate_factor_hevc` | 1.5 | 输出一律 H.264：H.264 原片沿用原码率，HEVC 等其他编码的原片按原码率乘这个系数（同样的画质 H.264 要更高的码率），范围 [1, 4]；读不到原码率时按 8 Mb/s 乘系数。调大画质好、文件大 |
| `nvenc_preset` | `p4` | NVENC 编码预设（Linux + NVIDIA），`p1` 最快、`p7` 画质最好；只在用 `h264_nvenc` 编码时起作用 |
| `trt_fp16` | true | TensorRT 用 FP16（Linux + NVIDIA，GPU 计划里接上；JSON 布尔） |
| `ort_intra_threads` | 0 | onnxruntime 每个会话的 CPU 线程数，0 由 onnxruntime 自己定；同时跑多个 `vc_redact`（`VC_JOBS`）时调小，免得互相抢 CPU。macOS 上也生效，范围 [0, 1024] |
```

把 `cpp/README.md` 里的：

```markdown
- **检测器出错整段失败**：
```

换成：

```markdown
- **Linux 部署用的配置键**（设计 2026-10-20）：`output_bitrate_factor_hevc`（默认 1.5，[1, 4]，非 H.264 原片的码率系数）、`nvenc_preset`（默认 `p4`，`p1`–`p7`）、`trt_fp16`（默认 `true`）、`ort_intra_threads`（默认 0 = onnxruntime 自己定，> 0 设每个会话的 intra-op 线程数，macOS 上也生效）；写进 `stats.json` 的 `config`，含义见 `docs/USAGE.md` §4。
- **检测器出错整段失败**：
```

- [ ] **Step 6: 提交**

```bash
git add cpp/apps/vc_redact/redact_config.h cpp/apps/vc_redact/redact_config.cpp configs/redact.json \
    cpp/tests/redact_config_test.cpp docs/USAGE.md cpp/README.md
git commit -m "feat: add the Linux deployment settings to the redact config"
```

### Task 3: 输出一律 H.264（编码器顺序、码率系数、Level、8 位 4:2:0）

**Files:**
- Modify: `cpp/src/io/video_writer.h`（`WriterConfig`、新函数声明、`encoderName` 注释）
- Modify: `cpp/src/io/video_writer.cpp`（includes 与常量、`encoderFormat`、`setEncoderOptions`、新函数、`openEncoder`、去掉 hvc1 标签）
- Modify: `cpp/apps/vc_redact/redact_job.h`（`RedactStats`、`redactWarnings` 注释）
- Modify: `cpp/apps/vc_redact/redact_job.cpp`（`writerConfig`、`runPipeline`、`writeStats`、`redactWarnings`、`runRedact`）
- Modify: `cpp/tests/test_media.h`, `cpp/tests/test_media.cpp`（`yuv422Video()`、`VideoPresentation` 三个字段）
- Test: `cpp/tests/video_writer_test.cpp`, `cpp/tests/redact_job_test.cpp`
- Modify: `cpp/README.md`（"编码格式跟随原片"一条、`stats.json` 一条）、`docs/USAGE.md`（输出表、§3 画质一条）

**Interfaces:**
- Consumes: `Platform`、`buildPlatform()`（Task 1）；`RedactConfig::outputBitrateFactorHevc`、`RedactConfig::nvencPreset`（Task 2）。
- Produces（Task 10 与 GPU 计划使用）:
  - `std::vector<std::string> h264EncoderCandidates(bool preferHardware, Platform platform);` —— 有硬件偏好时 `h264_nvenc`（非 macOS）、`h264_videotoolbox`（仅 macOS），最后总是 `libx264`。
  - `int64_t h264OutputBitRate(bool sourceIsH264, int64_t sourceBitRate, double nonH264Factor);` —— 原码率（≤ 0 时 8 Mb/s），非 H.264 乘系数并 `llround`。
  - `int h264LevelIdc(int width, int height, double fps);` —— Table A-1 的 level_idc（4K60 → 52），超出 6.2 返回 0。
  - `std::optional<std::string> h264FormatNote(AVPixelFormat source);`
  - `WriterConfig`：删掉 `sourceCodec`，加 `std::string nvencPreset = "p4";`。
  - `RedactStats::outputBitRate`（`int64_t`）、`RedactStats::formatNote`（`std::string`）、`RedactStats::pipeline`（`"cpu-frames"`）；`stats.json` 新字段 `output_bit_rate`、`pipeline`（Task 8 的 `batch.json` 读 `pipeline`）。
  - 测试夹具：`test::yuv422Video()`；`test::VideoPresentation::codec`（`AVCodecID`）、`level`（`int`）、`pixelFormat`（`int`）。

- [ ] **Step 1: 测试夹具与失败的测试**

把 `cpp/tests/test_media.h` 里的：

```cpp
// 1 s, 640x360 @ 30 fps 10-bit HEVC (30 frames).
std::filesystem::path tenBitVideo();
```

换成：

```cpp
// 1 s, 640x360 @ 30 fps 10-bit HEVC (30 frames).
std::filesystem::path tenBitVideo();
// 1 s, 640x360 @ 30 fps 8-bit 4:2:2 H.264 (High 4:2:2, 30 frames).
std::filesystem::path yuv422Video();
```

把 `cpp/tests/test_media.h` 里的：

```cpp
  double rotationDegrees = 0;  // av_display_rotation_get of the matrix, 0 without one
  uint32_t codecTag = 0;
};
```

换成：

```cpp
  double rotationDegrees = 0;  // av_display_rotation_get of the matrix, 0 without one
  uint32_t codecTag = 0;
  AVCodecID codec = AV_CODEC_ID_NONE;
  int level = 0;          // the codec parameters' level (H.264: level_idc, 52 for 5.2)
  int pixelFormat = -1;   // the codec parameters' format (an AVPixelFormat)
};
```

把 `cpp/tests/test_media.cpp` 里的：

```cpp
fs::path audioOnlyFile() {
```

换成：

```cpp
fs::path yuv422Video() {
  return makeMedia("yuv422.mp4", "-f lavfi -i testsrc2=s=640x360:r=30:d=1 -c:v libx264 -pix_fmt yuv422p");
}

fs::path audioOnlyFile() {
```

把 `cpp/tests/test_media.cpp` 里的：

```cpp
  p.codecTag = par->codec_tag;
```

换成：

```cpp
  p.codecTag = par->codec_tag;
  p.codec = par->codec_id;
  p.level = par->level;
  p.pixelFormat = par->format;
```

把 `cpp/tests/video_writer_test.cpp` 里的：

```cpp
  config.frameRate = stream->avg_frame_rate;
  config.sourceCodec = reader.info().codecName;
  config.bitRate = stream->codecpar->bit_rate;
```

换成：

```cpp
  config.frameRate = stream->avg_frame_rate;
  config.bitRate = stream->codecpar->bit_rate;
```

把 `cpp/tests/video_writer_test.cpp` 里的：

```cpp
TEST(VideoWriter, KeepsHevcAsHevc) {
  const fs::path out = test::tmpDir() / "writer_x265.mp4";
  EXPECT_EQ(reencode(test::tenBitVideo(), out, false), "libx265");
  EXPECT_EQ(VideoReader(out.string(), HwAccel::None).info().codecName, "hevc");
  EXPECT_EQ(framePts(out).size(), 30u);
}
```

换成：

```cpp
// spec 2026-10-20 §3: the output is H.264 whatever the source, here a 10-bit HEVC one.
TEST(VideoWriter, EncodesAnHevcSourceAsH264) {
  for (const bool hardware : {false, true}) {
    const fs::path out = test::tmpDir() / "writer_hevc_to_h264.mp4";
    fs::remove(out);
    const std::string encoder = reencode(test::tenBitVideo(), out, hardware);
    EXPECT_TRUE(encoder == "libx264" || (hardware && encoder == "h264_videotoolbox")) << encoder;
    const test::VideoPresentation got = test::videoPresentation(out);
    EXPECT_EQ(got.codec, AV_CODEC_ID_H264) << encoder;
    EXPECT_EQ(got.pixelFormat, AV_PIX_FMT_YUV420P) << encoder;
    EXPECT_EQ(framePts(out).size(), 30u) << encoder;
  }
}

// An 8-bit 4:2:2 source is encoded as 4:2:0 too: libx264 would otherwise write High 4:2:2, which hardware
// decoders and NVENC do not handle.
TEST(VideoWriter, Encodes422As420) {
  ASSERT_EQ(test::videoPresentation(test::yuv422Video()).pixelFormat, AV_PIX_FMT_YUV422P);
  for (const bool hardware : {false, true}) {
    const fs::path out = test::tmpDir() / "writer_422.mp4";
    fs::remove(out);
    const std::string encoder = reencode(test::yuv422Video(), out, hardware);
    const test::VideoPresentation got = test::videoPresentation(out);
    EXPECT_EQ(got.codec, AV_CODEC_ID_H264) << encoder;
    EXPECT_EQ(got.pixelFormat, AV_PIX_FMT_YUV420P) << encoder;
    EXPECT_EQ(framePts(out).size(), 30u) << encoder;
  }
}

// h264_videotoolbox left alone writes level 4.0 for 1080p at 59.94 fps and 5.1 for 4K60, below what those need
// (4.2, 5.2); the writer asks every encoder for a sufficient level.
TEST(VideoWriter, WritesALevelTheFrameSizeAndRateNeed) {
  struct Case {
    int width;
    int height;
    AVRational rate;
    int level;
  };
  for (const Case& c : {Case{1920, 1080, AVRational{60000, 1001}, 42}, Case{3840, 2160, AVRational{60, 1}, 52}}) {
    for (const bool hardware : {false, true}) {
      WriterConfig config;
      config.path = test::tmpDir() / "writer_level.mp4";
      fs::remove(config.path);
      config.width = c.width;
      config.height = c.height;
      config.timeBase = AVRational{c.rate.den, c.rate.num};
      config.frameRate = c.rate;
      config.bitRate = 20'000'000;
      config.preferHardware = hardware;
      std::string encoder;
      {
        VideoWriter writer(config, {});
        encoder = writer.encoderName();
        for (int i = 0; i < 3; ++i) {
          av::FramePtr frame = test::makeFrame(c.width, c.height, AV_PIX_FMT_YUV420P, 100, 128, 128);
          frame->pts = i;
          writer.writeVideo(*frame);
        }
        writer.finish();
      }
      const test::VideoPresentation got = test::videoPresentation(config.path);
      EXPECT_EQ(got.codec, AV_CODEC_ID_H264) << encoder;
      EXPECT_GE(got.level, c.level) << encoder << " " << c.width << "x" << c.height;
    }
  }
}

TEST(VideoWriter, ListsTheH264EncodersInOrder) {
  using Names = std::vector<std::string>;
  EXPECT_EQ(h264EncoderCandidates(true, Platform::Linux), (Names{"h264_nvenc", "libx264"}));
  EXPECT_EQ(h264EncoderCandidates(true, Platform::MacOS), (Names{"h264_videotoolbox", "libx264"}));
  EXPECT_EQ(h264EncoderCandidates(true, Platform::Other), (Names{"h264_nvenc", "libx264"}));
  for (const Platform p : {Platform::Linux, Platform::MacOS, Platform::Other}) {
    EXPECT_EQ(h264EncoderCandidates(false, p), (Names{"libx264"}));
  }
}

// H.264 sources keep their bitrate; others get it times the factor; an unknown one counts as 8 Mb/s.
TEST(VideoWriter, ChoosesTheOutputBitRate) {
  EXPECT_EQ(h264OutputBitRate(true, 6'200'390, 1.5), 6'200'390);
  EXPECT_EQ(h264OutputBitRate(false, 6'200'390, 1.5), 9'300'585);
  EXPECT_EQ(h264OutputBitRate(false, 1'000'001, 1.5), 1'500'002);  // 1500001.5 rounds away from zero
  EXPECT_EQ(h264OutputBitRate(false, 20'000'000, 4.0), 80'000'000);
  EXPECT_EQ(h264OutputBitRate(true, 0, 1.5), 8'000'000);
  EXPECT_EQ(h264OutputBitRate(false, 0, 1.5), 12'000'000);
  EXPECT_EQ(h264OutputBitRate(false, -1, 2.0), 16'000'000);
}

TEST(VideoWriter, KnowsTheH264LevelOfASizeAndRate) {
  EXPECT_EQ(h264LevelIdc(3840, 2160, 60), 52);
  EXPECT_EQ(h264LevelIdc(3840, 2160, 60000.0 / 1001), 52);
  EXPECT_EQ(h264LevelIdc(3840, 2160, 30), 51);
  EXPECT_EQ(h264LevelIdc(1920, 1080, 60000.0 / 1001), 42);
  EXPECT_EQ(h264LevelIdc(1920, 1080, 30), 40);
  EXPECT_EQ(h264LevelIdc(1280, 720, 60), 32);
  EXPECT_EQ(h264LevelIdc(1280, 720, 30), 31);
  EXPECT_EQ(h264LevelIdc(160, 120, 30), 11);
  EXPECT_EQ(h264LevelIdc(7680, 4320, 60), 61);
  EXPECT_EQ(h264LevelIdc(7680, 4320, 120), 62);
  EXPECT_EQ(h264LevelIdc(7680, 4320, 240), 0);
}

TEST(VideoWriter, NotesSourcesThatAreNot8Bit420) {
  EXPECT_FALSE(h264FormatNote(AV_PIX_FMT_YUV420P).has_value());
  EXPECT_FALSE(h264FormatNote(AV_PIX_FMT_YUVJ420P).has_value());
  EXPECT_FALSE(h264FormatNote(AV_PIX_FMT_NV12).has_value());
  EXPECT_FALSE(h264FormatNote(AV_PIX_FMT_NONE).has_value());
  EXPECT_EQ(h264FormatNote(AV_PIX_FMT_YUV420P10LE),
            "the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0");
  EXPECT_EQ(h264FormatNote(AV_PIX_FMT_YUV422P), "the source video is yuv422p (8-bit 4:2:2); the H.264 output is 8-bit 4:2:0");
  EXPECT_EQ(h264FormatNote(AV_PIX_FMT_YUV422P10LE),
            "the source video is yuv422p10le (10-bit 4:2:2); the H.264 output is 8-bit 4:2:0");
  EXPECT_EQ(h264FormatNote(AV_PIX_FMT_YUV444P), "the source video is yuv444p (8-bit 4:4:4); the H.264 output is 8-bit 4:2:0");
}
```

把 `cpp/tests/video_writer_test.cpp` 里的（HEVC 输出没有了，`hvc1` 标签也就没有了）：

```cpp
TEST(VideoWriter, TagsHevcAsHvc1) {
  const fs::path out = test::tmpDir() / "writer_hvc1.mp4";
  EXPECT_EQ(reencode(test::tenBitVideo(), out, false), "libx265");
  EXPECT_EQ(test::videoPresentation(out).codecTag, static_cast<uint32_t>(MKTAG('h', 'v', 'c', '1')));
}

TEST(VideoWriter, KeepsTheSampleAspectRatio) {
```

换成：

```cpp
TEST(VideoWriter, KeepsTheSampleAspectRatio) {
```

把 `cpp/tests/video_writer_test.cpp` 里的（libx264 收 NV12，换格式的场景改用 4:2:2 输入）：

```cpp
  config.pixelFormat = AV_PIX_FMT_NV12;  // libx265 takes no NV12: converted to yuv420p
  config.sourceCodec = "hevc";
  config.colorRange = AVCOL_RANGE_JPEG;
```

换成：

```cpp
  config.pixelFormat = AV_PIX_FMT_YUV422P;  // 4:2:2: converted to 4:2:0 for the encoder
  config.colorRange = AVCOL_RANGE_JPEG;
```

把 `cpp/tests/video_writer_test.cpp` 里的：

```cpp
      av::FramePtr frame = test::makeFrame(128, 128, AV_PIX_FMT_NV12, 235, 128, 128);
```

换成：

```cpp
      av::FramePtr frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV422P, 235, 128, 128);
```

把 `cpp/tests/redact_job_test.cpp` 里的：

```cpp
// detect, track and mosaic run one after another on the processing thread: together they are the
// "main" stage, which runs in parallel with decode and encode. setup_s is the time before the first
// frame reaches it.
```

换成：

```cpp
// The source video stream's bitrate as the container states it.
int64_t sourceBitRate(const fs::path& video) {
  VideoReader reader(video.string(), HwAccel::None);
  return reader.videoStream()->codecpar->bit_rate;
}

nlohmann::json statsOf(const RedactStats& s) {
  std::ifstream statsFile(s.outputDir / "stats.json");
  return nlohmann::json::parse(statsFile);
}

// spec 2026-10-20 §3: a 10-bit HEVC source comes out as 8-bit 4:2:0 H.264 at 1.5 times its bitrate, masked like any
// other, with a note on the conversion.
TEST(RedactJob, AnHevcSourceComesOutAsH264AtTheFactorsBitrate) {
  RedactOptions o = baseOptions("redact_hevc");
  o.input = test::tenBitVideo();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 30);
  EXPECT_EQ(s.maskedFrames, 30);  // the fake face box is masked on every frame, as on an H.264 source
  EXPECT_EQ(s.encoder, "libx264");
  const test::VideoPresentation out = test::videoPresentation(s.video);
  EXPECT_EQ(out.codec, AV_CODEC_ID_H264);
  EXPECT_EQ(out.pixelFormat, AV_PIX_FMT_YUV420P);
  const int64_t source = sourceBitRate(o.input);
  ASSERT_GT(source, 0);
  EXPECT_EQ(s.outputBitRate, std::llround(static_cast<double>(source) * 1.5));
  const nlohmann::json stats = statsOf(s);
  EXPECT_EQ(stats.at("output_bit_rate").get<int64_t>(), s.outputBitRate);
  EXPECT_EQ(stats.at("encoder").get<std::string>(), "libx264");
  EXPECT_EQ(stats.at("pipeline").get<std::string>(), "cpu-frames");
  EXPECT_DOUBLE_EQ(stats.at("config").at("output_bitrate_factor_hevc").get<double>(), 1.5);
  EXPECT_EQ(s.formatNote, "the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0");
  const std::vector<std::string> warnings = redactWarnings(s);
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0], "note: " + s.formatNote);
  EXPECT_EQ(decodeAll(s.video).size(), 30u);
}

TEST(RedactJob, TheBitrateFactorComesFromTheConfig) {
  RedactOptions o = baseOptions("redact_hevc_factor");
  o.input = test::tenBitVideo();
  o.config.outputBitrateFactorHevc = 2.0;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.outputBitRate, sourceBitRate(o.input) * 2);
}

// An H.264 source keeps its bitrate and needs no note.
TEST(RedactJob, AnH264SourceKeepsItsBitrate) {
  const RedactOptions o = baseOptions("redact_h264_rate");
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.outputBitRate, sourceBitRate(o.input));
  EXPECT_TRUE(s.formatNote.empty());
  EXPECT_EQ(test::videoPresentation(s.video).codec, AV_CODEC_ID_H264);
}

TEST(RedactJob, A422SourceIsEncodedAs420WithANote) {
  RedactOptions o = baseOptions("redact_422");
  o.input = test::yuv422Video();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 30);
  EXPECT_EQ(test::videoPresentation(s.video).pixelFormat, AV_PIX_FMT_YUV420P);
  EXPECT_EQ(s.formatNote, "the source video is yuv422p (8-bit 4:2:2); the H.264 output is 8-bit 4:2:0");
}

// detect, track and mosaic run one after another on the processing thread: together they are the
// "main" stage, which runs in parallel with decode and encode. setup_s is the time before the first
// frame reaches it.
```

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`video_writer_test.cpp` 报 `use of undeclared identifier 'h264EncoderCandidates'`（以及 `h264OutputBitRate`、`h264LevelIdc`、`h264FormatNote`、`Platform`），`redact_job_test.cpp` 报 `no member named 'outputBitRate' in 'vc::RedactStats'`。

- [ ] **Step 3: 实现 `VideoWriter` 的 H.264 输出**

把 `cpp/src/io/video_writer.h` 里的：

```cpp
#include <string>
#include <vector>

extern "C" {
```

换成：

```cpp
#include <string>
#include <vector>

#include "common/platform.h"

extern "C" {
```

把 `cpp/src/io/video_writer.h` 里的：

```cpp
struct WriterConfig {
  std::filesystem::path path;  // created as MP4 whatever the extension (so ".partial" works)
  int width = 0;
  int height = 0;
  AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;  // of the frames passed to writeVideo(); converted when
                                                   // the encoder does not take it
  AVRational timeBase{1, 1000};                    // of the frames' pts
  AVRational frameRate{0, 1};                      // 0/1 when unknown (the encoder assumes 30 fps)
  std::string sourceCodec = "h264";                // "hevc" stays HEVC, anything else becomes H.264
  int64_t bitRate = 0;                             // <= 0: 8 Mb/s
  bool preferHardware = true;                      // VideoToolbox first, then libx264 / libx265
```

换成：

```cpp
// The output is always H.264 (spec 2026-10-20 §3), whatever the source codec.
struct WriterConfig {
  std::filesystem::path path;  // created as MP4 whatever the extension (so ".partial" works)
  int width = 0;
  int height = 0;
  AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;  // of the frames passed to writeVideo(); anything but 8-bit
                                                   // 4:2:0 is converted to it (H.264 encoders all take that)
  AVRational timeBase{1, 1000};                    // of the frames' pts
  AVRational frameRate{0, 1};                      // 0/1 when unknown (the encoder assumes 30 fps)
  int64_t bitRate = 0;                             // <= 0: 8 Mb/s (h264OutputBitRate gives vc_redact's)
  bool preferHardware = true;                      // h264EncoderCandidates
  std::string nvencPreset = "p4";                  // h264_nvenc's preset (configs/redact.json nvenc_preset)
```

把 `cpp/src/io/video_writer.h` 里的：

```cpp
// Throws VcError unless the encoder returned exactly one packet per frame it was sent: a dropped or
// extra frame would put the output out of step with the source and masks.jsonl.
void checkEncodedPackets(int64_t frames, int64_t packets);
```

换成：

```cpp
// Throws VcError unless the encoder returned exactly one packet per frame it was sent: a dropped or
// extra frame would put the output out of step with the source and masks.jsonl.
void checkEncodedPackets(int64_t frames, int64_t packets);

// The H.264 encoders to try, in order (spec 2026-10-20 §3): with preferHardware h264_nvenc (not on macOS), then
// h264_videotoolbox (macOS only), then libx264; without it libx264 alone.
std::vector<std::string> h264EncoderCandidates(bool preferHardware, Platform platform);

// The output bitrate: the source's (8 Mb/s when it is unknown, i.e. <= 0), times nonH264Factor unless the source is
// H.264 (configs/redact.json output_bitrate_factor_hevc), rounded to whole bits per second.
int64_t h264OutputBitRate(bool sourceIsH264, int64_t sourceBitRate, double nonH264Factor);

// level_idc (52 for level 5.2) of the lowest H.264 level whose frame size and macroblock rate (ITU-T H.264 Table
// A-1) hold width x height at fps: 3840x2160 at 60 fps needs 52, 1920x1080 at 59.94 fps 42. 0 beyond level 6.2.
// The same table as vc_eval's required_h264_level (py/vc_eval/integrity.py).
int h264LevelIdc(int width, int height, double fps);

// What vc_redact notes on stderr when the source is not 8-bit 4:2:0 (spec 2026-10-20 §3: it is encoded as 8-bit
// 4:2:0), e.g. "the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0"; nullopt for an
// 8-bit 4:2:0 or unknown format.
std::optional<std::string> h264FormatNote(AVPixelFormat source);
```

把 `cpp/src/io/video_writer.h` 里的：

```cpp
  // e.g. "h264_videotoolbox" or "libx264".
  const std::string& encoderName() const;
```

换成：

```cpp
  // e.g. "h264_nvenc", "h264_videotoolbox" or "libx264".
  const std::string& encoderName() const;
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {
namespace {

constexpr int64_t kDefaultBitRate = 8'000'000;
```

换成：

```cpp
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

constexpr int64_t kDefaultBitRate = 8'000'000;
constexpr int kMacroblockPx = 16;
// h264_videotoolbox's level option starts at 3.0 (1.3 is Baseline only) and ends at 5.2.
constexpr int kVideoToolboxMinLevel = 30;
constexpr int kVideoToolboxMaxLevel = 52;

// ITU-T H.264 Table A-1: level_idc, MaxMBPS (macroblocks per second), MaxFS (macroblocks per frame). Level 1b has
// the limits of level 1 and is left out.
struct H264Level {
  int idc = 0;
  double maxMacroblocksPerSecond = 0;
  int64_t maxFrameMacroblocks = 0;
};
constexpr std::array<H264Level, 19> kH264Levels = {{
    {10, 1485, 99},       {11, 3000, 396},       {12, 6000, 396},       {13, 11880, 396},     {20, 11880, 396},
    {21, 19800, 792},     {22, 20250, 1620},     {30, 40500, 1620},     {31, 108000, 3600},   {32, 216000, 5120},
    {40, 245760, 8192},   {41, 245760, 8192},    {42, 522240, 8704},    {50, 589824, 22080},  {51, 983040, 36864},
    {52, 2073600, 36864}, {60, 4177920, 139264}, {61, 8355840, 139264}, {62, 16711680, 139264},
}};

bool is8Bit420(AVPixelFormat format) {
  return format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUVJ420P || format == AV_PIX_FMT_NV12;
}
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
// `wanted` when the encoder takes it, else yuv420p, else the encoder's first format (libx265 has no NV12).
AVPixelFormat encoderFormat(const AVCodecContext* ctx, const AVCodec* codec, AVPixelFormat wanted) {
  const void* configs = nullptr;
  int count = 0;
  if (avcodec_get_supported_config(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0 ||
      configs == nullptr || count == 0) {
    return wanted;  // the encoder accepts any format
  }
  const auto* formats = static_cast<const AVPixelFormat*>(configs);
  for (int i = 0; i < count; ++i) {
    if (formats[i] == wanted) return wanted;
  }
  for (int i = 0; i < count; ++i) {
    if (formats[i] == AV_PIX_FMT_YUV420P) return AV_PIX_FMT_YUV420P;
  }
  return formats[0];
}

std::vector<std::string> encoderCandidates(const WriterConfig& config) {
  const bool hevc = config.sourceCodec == "hevc";
  std::vector<std::string> names;
  if (config.preferHardware) names.emplace_back(hevc ? "hevc_videotoolbox" : "h264_videotoolbox");
  names.emplace_back(hevc ? "libx265" : "libx264");
  return names;
}
```

换成：

```cpp
// The 8-bit 4:2:0 format the encoder gets: `wanted` when it is one and the encoder takes it, else NV12, else
// yuv420p; AV_PIX_FMT_NONE when the encoder takes neither.
AVPixelFormat encoderFormat(const AVCodecContext* ctx, const AVCodec* codec, AVPixelFormat wanted) {
  const void* configs = nullptr;
  int count = 0;
  if (avcodec_get_supported_config(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0 ||
      configs == nullptr || count == 0) {
    return is8Bit420(wanted) ? wanted : AV_PIX_FMT_YUV420P;  // the encoder accepts any format
  }
  const auto* formats = static_cast<const AVPixelFormat*>(configs);
  const auto takes = [&](AVPixelFormat f) { return std::find(formats, formats + count, f) != formats + count; };
  if (is8Bit420(wanted) && takes(wanted)) return wanted;
  if (takes(AV_PIX_FMT_NV12)) return AV_PIX_FMT_NV12;
  if (takes(AV_PIX_FMT_YUV420P)) return AV_PIX_FMT_YUV420P;
  return AV_PIX_FMT_NONE;
}

// Encoder-specific options; the reason it cannot be used, or an empty string. h264_videotoolbox picks levels below
// what the frame rate needs (4.0 for 1080p at 59.94 fps, 5.1 for 4K60), so High profile and the level are set;
// h264_nvenc gets the preset and the level; libx264 chooses a sufficient level by itself.
std::string setEncoderOptions(AVCodecContext& ctx, const std::string& name, const WriterConfig& config, int level) {
  const auto failed = [&](int ret, const char* what) { return std::string(what) + ": " + avErrorText(ret); };
  if (name == "h264_videotoolbox") {
    if (const int ret = av_opt_set(ctx.priv_data, "profile", "high", 0); ret < 0) return failed(ret, "profile high");
    const int vtLevel =
        std::clamp(level > 0 ? level : kVideoToolboxMaxLevel, kVideoToolboxMinLevel, kVideoToolboxMaxLevel);
    if (const int ret = av_opt_set_int(ctx.priv_data, "level", vtLevel, 0); ret < 0) return failed(ret, "level");
  } else if (name == "h264_nvenc") {
    if (const int ret = av_opt_set(ctx.priv_data, "preset", config.nvencPreset.c_str(), 0); ret < 0) {
      return failed(ret, "preset");
    }
    if (level > 0) {
      if (const int ret = av_opt_set_int(ctx.priv_data, "level", level, 0); ret < 0) return failed(ret, "level");
    }
  }
  return {};
}
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
bool validRatio(AVRational r) { return r.num > 0 && r.den > 0; }

}  // namespace
```

换成：

```cpp
bool validRatio(AVRational r) { return r.num > 0 && r.den > 0; }

}  // namespace

std::vector<std::string> h264EncoderCandidates(bool preferHardware, Platform platform) {
  std::vector<std::string> names;
  if (preferHardware) {
    if (platform != Platform::MacOS) names.emplace_back("h264_nvenc");
    if (platform == Platform::MacOS) names.emplace_back("h264_videotoolbox");
  }
  names.emplace_back("libx264");
  return names;
}

int64_t h264OutputBitRate(bool sourceIsH264, int64_t sourceBitRate, double nonH264Factor) {
  const int64_t base = sourceBitRate > 0 ? sourceBitRate : kDefaultBitRate;
  if (sourceIsH264) return base;
  return std::llround(static_cast<double>(base) * nonH264Factor);
}

int h264LevelIdc(int width, int height, double fps) {
  const int64_t frameMacroblocks = static_cast<int64_t>((width + kMacroblockPx - 1) / kMacroblockPx) *
                                   static_cast<int64_t>((height + kMacroblockPx - 1) / kMacroblockPx);
  for (const H264Level& level : kH264Levels) {
    if (frameMacroblocks <= level.maxFrameMacroblocks &&
        static_cast<double>(frameMacroblocks) * fps <= level.maxMacroblocksPerSecond) {
      return level.idc;
    }
  }
  return 0;
}

std::optional<std::string> h264FormatNote(AVPixelFormat source) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(source);
  if (desc == nullptr || is8Bit420(source)) return std::nullopt;
  std::string layout;
  if ((desc->flags & AV_PIX_FMT_FLAG_RGB) != 0) {
    layout = "RGB";
  } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
    layout = "4:2:0";
  } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
    layout = "4:2:2";
  } else if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
    layout = "4:4:4";
  } else {
    layout = "subsampled " + std::to_string(1 << desc->log2_chroma_w) + "x" + std::to_string(1 << desc->log2_chroma_h);
  }
  return std::string("the source video is ") + desc->name + " (" + std::to_string(desc->comp[0].depth) + "-bit " +
         layout + "); the H.264 output is 8-bit 4:2:0";
}
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
    std::vector<std::pair<std::string, std::string>> failed;  // encoder, reason
    for (const std::string& name : encoderCandidates(config)) {
```

换成：

```cpp
    const int level = h264LevelIdc(config.width, config.height, av_q2d(rate));
    std::vector<std::pair<std::string, std::string>> failed;  // encoder, reason
    for (const std::string& name : h264EncoderCandidates(config.preferHardware, buildPlatform())) {
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
      ctx->pix_fmt = encoderFormat(ctx.get(), codec, config.pixelFormat);
```

换成：

```cpp
      ctx->pix_fmt = encoderFormat(ctx.get(), codec, config.pixelFormat);
      if (ctx->pix_fmt == AV_PIX_FMT_NONE) {
        failed.emplace_back(name, "takes no 8-bit 4:2:0 input");
        continue;
      }
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
      if ((output->oformat->flags & AVFMT_GLOBALHEADER) != 0) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      const int ret = avcodec_open2(ctx.get(), codec, nullptr);
```

换成：

```cpp
      if ((output->oformat->flags & AVFMT_GLOBALHEADER) != 0) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      if (const std::string why = setEncoderOptions(*ctx, name, config, level); !why.empty()) {
        failed.emplace_back(name, why);
        continue;
      }
      const int ret = avcodec_open2(ctx.get(), codec, nullptr);
```

把 `cpp/src/io/video_writer.cpp` 里的：

```cpp
  checkAv(avcodec_parameters_from_context(s.videoStream->codecpar, s.encoder.get()), "copy encoder parameters");
  if (s.encoder->codec_id == AV_CODEC_ID_HEVC) s.videoStream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
```

换成：

```cpp
  checkAv(avcodec_parameters_from_context(s.videoStream->codecpar, s.encoder.get()), "copy encoder parameters");
```

（10 位原片在 `VideoReader` 里已经转成 NV12（`toCpu8Bit`），所以编码器收到的已是 8 位 4:2:0；4:2:2 / 4:4:4 原片由 `writeVideo` 现有的 `sws_scale_frame` 分支转成 `encoderFormat` 选的格式，色彩范围照旧带过去。）

- [ ] **Step 4: `vc_redact` 用上码率系数并记下 note**

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
  std::string encoder;   // e.g. "h264_videotoolbox", "libx264"
```

换成：

```cpp
  std::string encoder;   // e.g. "h264_nvenc", "h264_videotoolbox", "libx264"; the output is always H.264
  int64_t outputBitRate = 0;  // the encoder's target: h264OutputBitRate of the source (spec 2026-10-20 §3)
  // Set when the source is not 8-bit 4:2:0 and is converted (h264FormatNote); redactWarnings prints it.
  std::string formatNote{};
  // "cpu-frames": frames go through main memory between decode, detection, mosaic and encode (the GPU path of
  // spec 2026-10-20 §4, "gpu", comes with stage 2b).
  std::string pipeline = "cpu-frames";
```

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
// Lines for stderr after a successful run (empty when there is nothing to warn about): a note when detect_fps
// is above the video's frame rate (so every frame is a keyframe), and a warning for decoder errors, i.e.
// corrupt packets that were skipped.
```

换成：

```cpp
// Lines for stderr after a successful run (empty when there is nothing to warn about): a note when detect_fps
// is above the video's frame rate (so every frame is a keyframe), a note when the source was not 8-bit 4:2:0
// (formatNote), and a warning for decoder errors, i.e. corrupt packets that were skipped.
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
// Everything the encode thread needs to know about the input video, copied by value here on the
// main thread, so that it never reads the demuxer's streams (the decode thread is using them).
WriterConfig writerConfig(const VideoReader& reader, const fs::path& path, bool preferHardware) {
  const AVStream* stream = reader.videoStream();
  WriterConfig c;
  c.path = path;
  c.width = reader.info().width;
  c.height = reader.info().height;
  c.timeBase = stream->time_base;
  c.frameRate = stream->avg_frame_rate;
  c.sourceCodec = reader.info().codecName;
  c.bitRate = stream->codecpar->bit_rate;
  c.preferHardware = preferHardware;
  copyPresentation(*stream, c);  // aspect ratio, colour tags and rotation stay as in the input
  return c;
}
```

换成：

```cpp
// Everything the encode thread needs to know about the input video, copied by value here on the
// main thread, so that it never reads the demuxer's streams (the decode thread is using them). The output is
// H.264 at the source's bitrate, times output_bitrate_factor_hevc for any other codec (spec 2026-10-20 §3).
WriterConfig writerConfig(const VideoReader& reader, const fs::path& path, const RedactOptions& options) {
  const AVStream* stream = reader.videoStream();
  WriterConfig c;
  c.path = path;
  c.width = reader.info().width;
  c.height = reader.info().height;
  c.timeBase = stream->time_base;
  c.frameRate = stream->avg_frame_rate;
  c.bitRate = h264OutputBitRate(stream->codecpar->codec_id == AV_CODEC_ID_H264, stream->codecpar->bit_rate,
                                options.config.outputBitrateFactorHevc);
  c.preferHardware = options.preferHardwareEncoder;
  c.nvencPreset = options.config.nvencPreset;
  copyPresentation(*stream, c);  // aspect ratio, colour tags and rotation stay as in the input
  return c;
}
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
  const WriterConfig config = writerConfig(reader, partial, options.preferHardwareEncoder);
```

换成：

```cpp
  const WriterConfig config = writerConfig(reader, partial, options);
  stats.outputBitRate = config.bitRate;
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
  j["encoder"] = s.encoder;
```

换成：

```cpp
  j["encoder"] = s.encoder;
  j["output_bit_rate"] = s.outputBitRate;
  j["pipeline"] = s.pipeline;
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
    warnings.push_back(formatted("note: detect_fps %g is above the video's %.2f fps, so every frame is a keyframe",
                                 stats.detectFpsRequested, stats.videoFps));
  }
```

换成：

```cpp
    warnings.push_back(formatted("note: detect_fps %g is above the video's %.2f fps, so every frame is a keyframe",
                                 stats.detectFpsRequested, stats.videoFps));
  }
  if (!stats.formatNote.empty()) warnings.push_back("note: " + stats.formatNote);
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
  stats.detectFpsAboveVideoRate = schedule.aboveVideoRate;
```

换成：

```cpp
  stats.detectFpsAboveVideoRate = schedule.aboveVideoRate;
  // 10-bit and 4:2:2 sources are encoded as 8-bit 4:2:0 (spec 2026-10-20 §3); read here, before the decode thread
  // owns the reader.
  stats.formatNote = h264FormatNote(static_cast<AVPixelFormat>(reader.videoStream()->codecpar->format)).value_or("");
```

- [ ] **Step 5: 跑测试**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*:VideoWriter.*:FrameFileName.*'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```

预期：`0`；第二组 `[  PASSED  ] 45 tests.`（`VideoWriter` 25 个，`WritesALevelTheFrameSizeAndRateNeed` 在本机走 `h264_videotoolbox` 和 `libx264` 各一遍，Level 分别不低于 4.2 / 5.2）；第六组 `[  PASSED  ] 81 tests.`。

- [ ] **Step 6: 文档**

把 `cpp/README.md` 里的：

```markdown
- **编码格式跟随原片**：HEVC 仍是 HEVC，其他一律输出 H.264，容器固定为 MP4，码率取原片视频流的码率（读不到时 8 Mb/s）。VideoToolbox 不可用（例如 H.264 宽或高超过 4096）时退到 libx264 / libx265，stderr 会有一行 `vc: video encoder h264_videotoolbox could not be opened (原因); using libx264`，实际用的编码器写在统计行 `encoder`。编码器交回的包数必须与送进去的帧数一致，否则整次运行失败。
```

换成：

```markdown
- **输出一律 H.264**（设计 2026-10-20 §3，macOS 与 Linux 相同），容器固定为 MP4。编码器按顺序尝试 `h264_nvenc`（不在 macOS 上）→ `h264_videotoolbox`（只在 macOS 上）→ `libx264`，`--encoder software` 只用 libx264；前面的打不开（例如 VideoToolbox 的 H.264 宽或高超过 4096）时 stderr 一行 `vc: video encoder h264_videotoolbox could not be opened (原因); using libx264`，实际用的写在统计行和 `stats.json` 的 `encoder`。码率：H.264 原片取原片视频流的码率，HEVC 等其他编码乘 `output_bitrate_factor_hevc`（默认 1.5），读不到原码率时按 8 Mb/s（再乘系数）；实际目标码率写在 `stats.json` 的 `output_bit_rate`。Level 按 H.264 Table A-1 由分辨率和帧率定（1080p 59.94 fps 要 4.2，4K60 要 5.2）：VideoToolbox 自己选的偏低（这两种分别给 4.0、5.1），所以给它指定 High profile 和这个 Level（它的 Level 选项只有 3.0–5.2）；NVENC 也指定 Level，并用 `nvenc_preset`；libx264 自己选的就够。编码器的输入一律是 8 位 4:2:0（NV12 或 yuv420p）：10 位原片在解码时转成 NV12，4:2:2 / 4:4:4 原片在送进编码器前转换；原片不是 8 位 4:2:0 时，运行结束 stderr 多一行 `vc_redact: note: the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0`（退出码仍为 0）。编码器交回的包数必须与送进去的帧数一致，否则整次运行失败。
```

把 `cpp/README.md` 里的：

```markdown
实际使用的检测后端和编码器、
```

换成：

```markdown
实际使用的检测后端和编码器、编码器的目标码率 `output_bit_rate`、流水线 `pipeline`（目前总是 `cpu-frames`：帧在内存里流转；全程 GPU 的 `gpu` 在 GPU 计划里）、
```

把 `docs/USAGE.md` 里的：

```markdown
| `<视频名>.redacted.mp4` | 打码后的视频。编码跟随原片（HEVC → HEVC，其他 → H.264），码率取原片的，音频原样复制，旋转、色彩信息照搬 |
```

换成：

```markdown
| `<视频名>.redacted.mp4` | 打码后的视频，一律 H.264（macOS 与 Linux 相同）：H.264 原片沿用原码率，HEVC 等其他编码按原码率乘 `output_bitrate_factor_hevc`（默认 1.5）；10 位、4:2:2 原片转成 8 位 4:2:0（运行结束 stderr 一行 note）。音频原样复制，旋转、色彩信息照搬 |
```

把 `docs/USAGE.md` 里的：

```markdown
| `stats.json` | 帧数、关键帧数、各阶段耗时和吞吐、检测后端、编码器 |
```

换成：

```markdown
| `stats.json` | 帧数、关键帧数、各阶段耗时和吞吐、检测后端、编码器、目标码率 `output_bit_rate`、流水线 `pipeline` |
```

把 `docs/USAGE.md` 里的：

```markdown
- **画质和大小：** 输出码率等于原片码率，4K HEVC 输出仍是 HEVC。
```

换成：

```markdown
- **画质和大小：** 输出一律 H.264：H.264 原片码率不变，4K HEVC 原片按 1.5 倍码率编成 H.264（`output_bitrate_factor_hevc`）；Level 按分辨率和帧率指定（4K60 为 5.2），`vc_eval integrity` 会检查。
```

- [ ] **Step 7: 提交**

```bash
git add cpp/src/io/video_writer.h cpp/src/io/video_writer.cpp cpp/apps/vc_redact/redact_job.h \
    cpp/apps/vc_redact/redact_job.cpp cpp/tests/test_media.h cpp/tests/test_media.cpp \
    cpp/tests/video_writer_test.cpp cpp/tests/redact_job_test.cpp cpp/README.md docs/USAGE.md
git commit -m "feat: encode every output as H.264 at a level that fits"
```

### Task 4: `--ep auto | tensorrt | cuda | coreml | cpu`、回退顺序、TensorRT 缓存目录、`ort_intra_threads`

**Files:**
- Modify（整个换掉）: `cpp/src/detect/execution_provider.h`
- Modify: `cpp/src/detect/ort_session.h`, `cpp/src/detect/ort_session.cpp`
- Modify: `cpp/src/detect/onnx_yolo_detector.h`, `.cpp`, `cpp/src/detect/yunet_detector.h`, `.cpp`
- Modify: `cpp/apps/vc_redact/redact_job.h`（`provider` 默认值、`detector` 注释）、`cpp/apps/vc_redact/redact_job.cpp`（`makeDetector`）
- Modify: `cpp/apps/vc_redact/redact_cli.cpp`, `cpp/apps/vc_detect/detect_cli.cpp`（帮助文本）
- Create: `cpp/tests/execution_provider_test.cpp`
- Modify: `cpp/tests/CMakeLists.txt`, `cpp/tests/onnx_yolo_detector_test.cpp`, `cpp/tests/detect_cli_test.cpp`, `cpp/tests/redact_cli_test.cpp`
- Modify: `cpp/README.md`（`--ep` 两行、`detector` 一条、TensorRT 缓存、第四组过滤命令）

**Interfaces:**
- Consumes: `Platform`、`buildPlatform()`、`builtWithCuda()`（Task 1）；`RedactConfig::ortIntraThreads`、`RedactConfig::trtFp16`（Task 2）。
- Produces（GPU 计划 1c 使用）:
  - `enum class ExecutionProvider { Auto, TensorRT, Cuda, CoreML, Cpu };`
  - `ExecutionProvider parseExecutionProvider(const std::string&);`（`"auto"|"tensorrt"|"cuda"|"coreml"|"cpu"`）
  - `std::string providerLabel(ExecutionProvider);` —— `"TensorRT"`、`"CUDA"`、`"CoreML"`、`"CPU"`、`"auto"`。
  - `std::vector<ExecutionProvider> providerOrder(ExecutionProvider requested, Platform platform);`
  - `struct SessionSettings { int intraOpThreads = 0; bool tensorRtFp16 = true; };`
  - `std::optional<std::filesystem::path> tensorRtCacheDirectory(const std::filesystem::path& onnx, int width, int height);`
  - `ort_session.h`（`src/detect` 内部）：`struct ProviderAttempt { bool ok = false; bool attached = false; std::string error{}; };`、`using ProviderTry = std::function<ProviderAttempt(ExecutionProvider)>;`、`using NoteSink = std::function<void(const std::string&)>;`、`ExecutionProvider tryProviders(const std::vector<ExecutionProvider>& order, const ProviderTry& attempt, const NoteSink& note);`、`std::string gpuProviderUnavailableReason(ExecutionProvider, const std::vector<std::string>& available, bool withCuda);`、`using CacheDirectory = std::function<std::optional<std::filesystem::path>()>;`、`struct SessionCaches { CacheDirectory coreMl{}; CacheDirectory tensorRt{}; };`、`CreatedSession createOrtSession(const std::filesystem::path& onnx, ExecutionProvider provider, const SessionCaches& caches, const SessionSettings& settings, const SessionConfigurer& configure = {});`
  - `OnnxYoloDetector(modelDir, provider, params, frameWidth, frameHeight, const SessionSettings& settings = {})`，`YuNetDetector(modelDir, provider, params, frameWidth, frameHeight, const SessionSettings& settings = {})`。
  - `RedactOptions::provider` 默认 `ExecutionProvider::Auto`。

- [ ] **Step 1: 写失败的测试**

新建 `cpp/tests/execution_provider_test.cpp`：

```cpp
#include "detect/execution_provider.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "detect/ort_session.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;
using P = ExecutionProvider;
using Order = std::vector<ExecutionProvider>;

// Sets (or, with nullptr, unsets) an environment variable for the lifetime of the object.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_;
};

// A fake provider set: the providers in `working` succeed, the others fail, unattached unless in `attached`.
struct FakeProviders {
  Order working;
  Order attached{};
  Order calls{};
  std::vector<std::string> notes{};

  ExecutionProvider run(const Order& order) {
    return tryProviders(
        order,
        [this](ExecutionProvider p) {
          calls.push_back(p);
          const auto in = [p](const Order& list) { return std::find(list.begin(), list.end(), p) != list.end(); };
          return ProviderAttempt{in(working), in(working) || in(attached), "no " + providerLabel(p)};
        },
        [this](const std::string& line) { notes.push_back(line); });
  }
};

TEST(ExecutionProvider, ParsesEveryName) {
  EXPECT_EQ(parseExecutionProvider("auto"), P::Auto);
  EXPECT_EQ(parseExecutionProvider("tensorrt"), P::TensorRT);
  EXPECT_EQ(parseExecutionProvider("cuda"), P::Cuda);
  EXPECT_EQ(parseExecutionProvider("coreml"), P::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), P::Cpu);
  for (const char* bad : {"", "CPU", "trt", "rocm", "gpu"}) {
    try {
      parseExecutionProvider(bad);
      ADD_FAILURE() << "accepted '" << bad << "'";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "unknown execution provider '" + std::string(bad) +
                                           "' (auto|tensorrt|cuda|coreml|cpu)");
    }
  }
}

TEST(ExecutionProvider, LabelsAreTheStatsNames) {
  EXPECT_EQ(providerLabel(P::TensorRT), "TensorRT");
  EXPECT_EQ(providerLabel(P::Cuda), "CUDA");
  EXPECT_EQ(providerLabel(P::CoreML), "CoreML");
  EXPECT_EQ(providerLabel(P::Cpu), "CPU");
  EXPECT_EQ(providerLabel(P::Auto), "auto");
}

// spec 2026-10-20 §3: auto is TensorRT, CUDA, CPU on Linux and CoreML, CPU on macOS; an explicit provider falls
// back to the CPU.
TEST(ExecutionProvider, OrderPerPlatform) {
  EXPECT_EQ(providerOrder(P::Auto, Platform::Linux), (Order{P::TensorRT, P::Cuda, P::Cpu}));
  EXPECT_EQ(providerOrder(P::Auto, Platform::MacOS), (Order{P::CoreML, P::Cpu}));
  EXPECT_EQ(providerOrder(P::Auto, Platform::Other), (Order{P::Cpu}));
  for (const Platform platform : {Platform::Linux, Platform::MacOS, Platform::Other}) {
    EXPECT_EQ(providerOrder(P::TensorRT, platform), (Order{P::TensorRT, P::Cpu}));
    EXPECT_EQ(providerOrder(P::Cuda, platform), (Order{P::Cuda, P::Cpu}));
    EXPECT_EQ(providerOrder(P::CoreML, platform), (Order{P::CoreML, P::Cpu}));
    EXPECT_EQ(providerOrder(P::Cpu, platform), (Order{P::Cpu}));
  }
}

TEST(ExecutionProvider, TheFirstWorkingProviderWinsWithoutANote) {
  FakeProviders fake{{P::TensorRT, P::Cuda, P::Cpu}};
  EXPECT_EQ(fake.run({P::TensorRT, P::Cuda, P::Cpu}), P::TensorRT);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT}));
  EXPECT_TRUE(fake.notes.empty());
}

TEST(ExecutionProvider, LinuxAutoFallsFromTensorRtToCuda) {
  FakeProviders fake{{P::Cuda, P::Cpu}};
  EXPECT_EQ(fake.run(providerOrder(P::Auto, Platform::Linux)), P::Cuda);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cuda}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); trying CUDA"}));
}

TEST(ExecutionProvider, LinuxAutoEndsOnTheCpu) {
  FakeProviders fake{{P::Cpu}, {P::Cuda}};
  EXPECT_EQ(fake.run(providerOrder(P::Auto, Platform::Linux)), P::Cpu);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cuda, P::Cpu}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); trying CUDA",
                                                  "vc: CUDA session failed (no CUDA); using CPU"}));
}

// The macOS lines are the ones vc_redact printed before --ep auto existed.
TEST(ExecutionProvider, MacAutoFallsFromCoreMlToTheCpu) {
  FakeProviders unavailable{{P::Cpu}};
  EXPECT_EQ(unavailable.run(providerOrder(P::Auto, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(unavailable.notes, (std::vector<std::string>{"vc: CoreML execution provider unavailable (no CoreML); using CPU"}));
  FakeProviders failing{{P::Cpu}, {P::CoreML}};
  EXPECT_EQ(failing.run(providerOrder(P::Auto, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(failing.notes, (std::vector<std::string>{"vc: CoreML session failed (no CoreML); using CPU"}));
}

TEST(ExecutionProvider, AnExplicitProviderFallsBackToTheCpu) {
  FakeProviders fake{{P::Cpu}};
  EXPECT_EQ(fake.run(providerOrder(P::TensorRT, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cpu}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); using CPU"}));
}

TEST(ExecutionProvider, ACpuFailureIsAnErrorAndTheOrderMustEndOnTheCpu) {
  FakeProviders none{{}};
  EXPECT_THROW(none.run({P::CoreML, P::Cpu}), VcError);
  FakeProviders fake{{P::Cuda}};
  EXPECT_THROW(fake.run({P::Cuda}), std::invalid_argument);
  EXPECT_THROW(fake.run({}), std::invalid_argument);
  EXPECT_TRUE(fake.calls.empty());
}

// TensorRT and CUDA cannot run a session in this build; the reason says which piece is missing.
TEST(ExecutionProvider, SaysWhyAGpuProviderIsUnavailable) {
  const std::vector<std::string> mac = {"CoreMLExecutionProvider", "CPUExecutionProvider"};
  const std::vector<std::string> gpu = {"TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"};
  EXPECT_EQ(gpuProviderUnavailableReason(P::TensorRT, mac, false), "TensorrtExecutionProvider is not in this onnxruntime build");
  EXPECT_EQ(gpuProviderUnavailableReason(P::Cuda, mac, true), "CUDAExecutionProvider is not in this onnxruntime build");
  EXPECT_EQ(gpuProviderUnavailableReason(P::Cuda, gpu, false), "vc_redact was built without VC_WITH_CUDA");
  EXPECT_EQ(gpuProviderUnavailableReason(P::TensorRT, gpu, true),
            "the TensorRT / CUDA session set-up comes with stage 1c of the Linux plan");
}

// spec 2026-10-20 §3: $VC_CACHE/trt-cache/<sha256 of model.onnx>/<width>x<height>/, like the CoreML cache.
TEST(ExecutionProvider, KeysTheTensorRtCacheByModelAndInputSize) {
  const fs::path onnx = fs::path(VC_TEST_DATA_DIR) / "const_yolo" / "model.onnx";
  // `shasum -a 256 tests/data/const_yolo/model.onnx`
  const std::string sha = "cf8f5bf0e958cf75b422ac5b6991c6c4c6f50c389e389eb5ec69262076b35d51";
  const fs::path root = test::tmpDir() / "trt_cache_root";
  {
    const ScopedEnv cache("VC_CACHE", root.c_str());
    EXPECT_EQ(tensorRtCacheDirectory(onnx, 1920, 1088), root / "trt-cache" / sha / "1920x1088");
    EXPECT_EQ(tensorRtCacheDirectory(onnx, 3840, 2176), root / "trt-cache" / sha / "3840x2176");
    EXPECT_FALSE(fs::exists(root));  // nothing is created
    EXPECT_THROW(tensorRtCacheDirectory(test::tmpDir() / "no_such_model.onnx", 64, 64), VcError);
  }
  const ScopedEnv home("HOME", "/nonexistent-home-for-test");
  const ScopedEnv noCache("VC_CACHE", nullptr);
  EXPECT_EQ(tensorRtCacheDirectory(onnx, 64, 64),
            fs::path("/nonexistent-home-for-test") / ".cache" / "video-compliance" / "trt-cache" / sha / "64x64");
  const ScopedEnv noHome("HOME", nullptr);
  EXPECT_FALSE(tensorRtCacheDirectory(onnx, 64, 64).has_value());
}

}  // namespace
}  // namespace vc
```

把 `cpp/tests/CMakeLists.txt` 里的：

```cmake
  detect_images_test.cpp
```

换成：

```cmake
  detect_images_test.cpp
  execution_provider_test.cpp
```

把 `cpp/tests/onnx_yolo_detector_test.cpp` 里的：

```cpp
TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_THROW(parseExecutionProvider("cuda"), std::invalid_argument);
}
```

换成：

```cpp
TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_EQ(parseExecutionProvider("cuda"), ExecutionProvider::Cuda);
  EXPECT_THROW(parseExecutionProvider("rocm"), std::invalid_argument);
}

// ort_intra_threads > 0 sets the session's intra-op thread count; the boxes stay the same.
TEST(OnnxYoloDetector, RunsWithAFixedIntraOpThreadCount) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128, SessionSettings{1, true});
  EXPECT_EQ(detector.backend(), "CPU");
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
}

// An explicit GPU provider is unavailable on this build: one line on stderr, then the CPU.
TEST(OnnxYoloDetector, AnUnavailableGpuProviderFallsBackToTheCpu) {
  for (const ExecutionProvider provider : {ExecutionProvider::TensorRT, ExecutionProvider::Cuda}) {
    testing::internal::CaptureStderr();
    OnnxYoloDetector detector(constModel(), provider, DecodeParams{}, 128, 128);
    const std::string log = testing::internal::GetCapturedStderr();
    if (detector.backend() != "CPU") GTEST_SKIP() << "this build runs " << detector.backend();
    EXPECT_NE(log.find("vc: " + providerLabel(provider) + " execution provider unavailable ("), std::string::npos) << log;
    EXPECT_NE(log.find("); using CPU"), std::string::npos) << log;
    const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
    expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
  }
}

// Auto is CoreML then the CPU on macOS: the backend is one of the two, never a GPU provider.
TEST(OnnxYoloDetector, AutoUsesCoreMlOrTheCpuOnMacOs) {
  if (buildPlatform() != Platform::MacOS) GTEST_SKIP() << "macOS only";
  const ScopedEnv cache("VC_CACHE", freshCacheRoot("coreml_cache_auto").c_str());
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Auto, DecodeParams{}, 128, 128);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
}
```

把 `cpp/tests/detect_cli_test.cpp` 里的：

```cpp
  expectUsageError(with({"--ep", "cuda"}), "unknown execution provider 'cuda'");
```

换成：

```cpp
  expectUsageError(with({"--ep", "rocm"}), "unknown execution provider 'rocm'");
```

把 `cpp/tests/redact_cli_test.cpp` 里的：

```cpp
  const RedactOptions o = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--model", "m"});
  EXPECT_EQ(o.provider, ExecutionProvider::CoreML);
```

换成：

```cpp
  const RedactOptions o = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--model", "m"});
  EXPECT_EQ(o.provider, ExecutionProvider::Auto);
```

把 `cpp/tests/redact_cli_test.cpp` 里的：

```cpp
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", "cuda"},
```

换成：

```cpp
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", "rocm"},
```

把 `cpp/tests/redact_cli_test.cpp` 里的：

```cpp
TEST(RedactCli, DetectFpsOverridesTheDefault) {
```

换成：

```cpp
TEST(RedactCli, ParsesEveryExecutionProvider) {
  const std::vector<std::pair<std::string, ExecutionProvider>> cases = {
      {"auto", ExecutionProvider::Auto},     {"tensorrt", ExecutionProvider::TensorRT},
      {"cuda", ExecutionProvider::Cuda},     {"coreml", ExecutionProvider::CoreML},
      {"cpu", ExecutionProvider::Cpu}};
  for (const auto& [name, provider] : cases) {
    EXPECT_EQ(parseRedactArgs({"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", name}).provider, provider)
        << name;
  }
  EXPECT_THROW(parseRedactArgs({"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", "rocm"}), UsageError);
}

TEST(RedactCli, DetectFpsOverridesTheDefault) {
```

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`execution_provider_test.cpp` 报 `no member named 'Auto' in 'vc::ExecutionProvider'`、`use of undeclared identifier 'tryProviders'`（以及 `providerOrder`、`providerLabel`、`tensorRtCacheDirectory`），`onnx_yolo_detector_test.cpp` 报 `use of undeclared identifier 'SessionSettings'`（以及 `buildPlatform`）。

- [ ] **Step 3: 执行提供者的取值、顺序与缓存目录**

把 `cpp/src/detect/execution_provider.h` 整个换成：

```cpp
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/platform.h"

namespace vc {

// --ep (spec 2026-10-20 §3). Auto tries TensorRT, CUDA, then the CPU on Linux and CoreML, then the CPU on macOS;
// an explicit provider that cannot be used falls back to the CPU.
enum class ExecutionProvider { Auto, TensorRT, Cuda, CoreML, Cpu };

// "auto" | "tensorrt" | "cuda" | "coreml" | "cpu"; throws std::invalid_argument.
ExecutionProvider parseExecutionProvider(const std::string& text);

// The name a detector's backend() and stats.json's detector use: "TensorRT", "CUDA", "CoreML", "CPU" ("auto" for
// Auto, which never runs a session).
std::string providerLabel(ExecutionProvider provider);

// The providers createOrtSession tries for `requested`, in order, always ending with the CPU: Auto is TensorRT,
// CUDA, CPU on Linux, CoreML, CPU on macOS and the CPU alone elsewhere; any other provider is tried before the
// CPU (the CPU alone for Cpu).
std::vector<ExecutionProvider> providerOrder(ExecutionProvider requested, Platform platform);

// Settings every onnxruntime session of a run shares (configs/redact.json).
struct SessionSettings {
  int intraOpThreads = 0;    // ort_intra_threads: 0 keeps onnxruntime's default, more sets the intra-op threads
  bool tensorRtFp16 = true;  // trt_fp16, for the TensorRT provider (stage 1c of the Linux plan)
};

// Where CoreML keeps the model it compiles from `onnx`: $VC_CACHE/coreml-cache/<sha256 of the file>,
// VC_CACHE defaulting to ~/.cache/video-compliance (shared by all checkouts). Keyed by the content,
// so a changed model is compiled afresh. nullopt when neither VC_CACHE nor HOME is set. Throws
// VcError when the file cannot be read.
std::optional<std::filesystem::path> coreMlCacheDirectory(const std::filesystem::path& onnx);

// Where TensorRT keeps the engines it builds from `onnx` for a width x height input (spec 2026-10-20 §3):
// $VC_CACHE/trt-cache/<sha256 of the file>/<width>x<height>, VC_CACHE defaulting as for coreMlCacheDirectory. An
// engine is fixed to its model and input size, so each gets its own directory. nullopt / VcError as
// coreMlCacheDirectory. Creates nothing.
std::optional<std::filesystem::path> tensorRtCacheDirectory(const std::filesystem::path& onnx, int width, int height);

}  // namespace vc
```

把 `cpp/src/detect/ort_session.h` 里的：

```cpp
// What createOrtSession returns. (Not named OrtSession: onnxruntime's C API has a type of that name.)
struct CreatedSession {
  Ort::Session session{nullptr};
  std::string backend = "CPU";  // "CoreML" when the session runs with the CoreML provider registered
};

// A session for `onnx` with all graph optimisations. CoreML first when asked for, compiling into
// `coreMlCache` (created; compiling into a temporary directory when it is nullopt or cannot be
// created, with a note on stderr for the latter). When that session fails with the cache it is
// created once more without it, leaving the shared directory alone (another process may be writing
// it), with one line on stderr; only when that fails too, or when the provider cannot be attached, does
// the CPU provider take over (also with a line on stderr). Throws Ort::Exception when the CPU session
// fails.
CreatedSession createOrtSession(const std::filesystem::path& onnx, ExecutionProvider provider,
                                const std::optional<std::filesystem::path>& coreMlCache,
                                const SessionConfigurer& configure = {});
```

换成：

```cpp
// What createOrtSession returns. (Not named OrtSession: onnxruntime's C API has a type of that name.)
struct CreatedSession {
  Ort::Session session{nullptr};
  std::string backend = "CPU";  // providerLabel of the provider the session runs with, e.g. "CoreML"
};

// The outcome of trying one execution provider.
struct ProviderAttempt {
  bool ok = false;        // a session exists on it
  bool attached = false;  // the provider could be registered (its session may still have failed)
  std::string error{};    // why not, otherwise
};
using ProviderTry = std::function<ProviderAttempt(ExecutionProvider)>;
using NoteSink = std::function<void(const std::string&)>;

// Tries `order` front to back with `attempt` until one is ok and returns it. For each provider passed over, `note`
// gets one line naming the next: "vc: <P> execution provider unavailable (<error>); trying <next>" when it could not
// be attached, "vc: <P> session failed (<error>); ..." when its session failed, "using <next>" instead of "trying
// <next>" when the next is the last. `order` must end with Cpu (std::invalid_argument otherwise); a CPU attempt
// that throws propagates, one that reports failure throws VcError.
ExecutionProvider tryProviders(const std::vector<ExecutionProvider>& order, const ProviderTry& attempt,
                               const NoteSink& note);

// Why the TensorRT or CUDA provider cannot run a session in this build: not in this onnxruntime (its
// GetAvailableProviders() is `available`), vc_redact built without VC_WITH_CUDA, or (with it) the GPU session
// set-up of stage 1c not being in this build yet.
std::string gpuProviderUnavailableReason(ExecutionProvider provider, const std::vector<std::string>& available,
                                         bool withCuda);

// The caches of the providers that compile the model, computed only when that provider is tried (hashing the
// model takes a moment, and an unreadable file only matters then).
using CacheDirectory = std::function<std::optional<std::filesystem::path>()>;
struct SessionCaches {
  CacheDirectory coreMl{};    // CoreML's compile cache (coreMlCacheDirectory, or per input size for YuNet)
  CacheDirectory tensorRt{};  // TensorRT's engine cache (tensorRtCacheDirectory), used from stage 1c on
};

// A session for `onnx` with all graph optimisations and `settings`, on the first provider of
// providerOrder(provider, buildPlatform()) that works (tryProviders, notes on stderr). CoreML compiles into
// caches.coreMl() (created; compiling into a temporary directory when there is none or it cannot be created, with
// a note on stderr for the latter); when that session fails with the cache it is created once more without it,
// leaving the shared directory alone (another process may be writing it), with one line on stderr, before the next
// provider is tried. TensorRT and CUDA are unavailable in this build (gpuProviderUnavailableReason). Throws
// Ort::Exception when the CPU session fails.
CreatedSession createOrtSession(const std::filesystem::path& onnx, ExecutionProvider provider,
                                const SessionCaches& caches, const SessionSettings& settings,
                                const SessionConfigurer& configure = {});
```

把 `cpp/src/detect/ort_session.cpp` 里的：

```cpp
#include <array>
#include <cstdio>
```

换成：

```cpp
#include <algorithm>
#include <array>
#include <cstdio>
```

把 `cpp/src/detect/ort_session.cpp` 里的：

```cpp
Ort::SessionOptions makeOptions(const SessionConfigurer& configure) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (configure) configure(options);
  return options;
}
```

换成：

```cpp
Ort::SessionOptions makeOptions(const SessionSettings& settings, const SessionConfigurer& configure) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (settings.intraOpThreads > 0) options.SetIntraOpNumThreads(settings.intraOpThreads);
  if (configure) configure(options);
  return options;
}
```

把 `cpp/src/detect/ort_session.cpp` 里的：

```cpp
CoreMlAttempt attemptCoreMl(const fs::path& onnx, const std::optional<fs::path>& cache,
                            const SessionConfigurer& configure) {
  CoreMlAttempt attempt;
  Ort::SessionOptions options = makeOptions(configure);
```

换成：

```cpp
CoreMlAttempt attemptCoreMl(const fs::path& onnx, const std::optional<fs::path>& cache,
                            const SessionSettings& settings, const SessionConfigurer& configure) {
  CoreMlAttempt attempt;
  Ort::SessionOptions options = makeOptions(settings, configure);
```

把 `cpp/src/detect/ort_session.cpp` 里的：

```cpp
  return attempt;
}

}  // namespace
```

换成：

```cpp
  return attempt;
}

// CoreML with the compile cache, and once more without it when the session fails with it.
CoreMlAttempt attemptCoreMlWithRetry(const fs::path& onnx, const std::optional<fs::path>& requestedCache,
                                     const SessionSettings& settings, const SessionConfigurer& configure) {
  const std::optional<fs::path> cache = usableCoreMlCache(requestedCache);
  CoreMlAttempt coreMl = attemptCoreMl(onnx, cache, settings, configure);
  if (coreMl.attached && !coreMl.ok && cache) {
    // The cache is shared: processes that compile at the same time race on it ("Error copying
    // compiled model to cache path"), and a compile that died half-way leaves a model that cannot
    // be loaded. Another process may be writing the directory, so it is left alone; compile
    // without it once before giving the CoreML provider up.
    std::fprintf(stderr,
                 "vc: CoreML session failed with the compile cache %s (%s); retrying without the cache "
                 "(delete %s to rebuild the cache)\n",
                 cache->c_str(), coreMl.error.c_str(), cache->c_str());
    coreMl = attemptCoreMl(onnx, std::nullopt, settings, configure);
  }
  return coreMl;
}

// $VC_CACHE, or ~/.cache/video-compliance; nullopt when neither VC_CACHE nor HOME is set.
std::optional<fs::path> cacheRoot() {
  if (const auto cache = envValue("VC_CACHE")) return fs::path(*cache);
  if (const auto home = envValue("HOME")) return fs::path(*home) / ".cache" / "video-compliance";
  return std::nullopt;
}

const char* ortProviderName(ExecutionProvider provider) {
  return provider == ExecutionProvider::TensorRT ? "TensorrtExecutionProvider" : "CUDAExecutionProvider";
}

}  // namespace
```

把 `cpp/src/detect/ort_session.cpp` 里的：

```cpp
std::optional<fs::path> coreMlCacheDirectory(const fs::path& onnx) {
  std::optional<fs::path> root;
  if (const auto cache = envValue("VC_CACHE")) {
    root = fs::path(*cache);
  } else if (const auto home = envValue("HOME")) {
    root = fs::path(*home) / ".cache" / "video-compliance";
  }
  if (!root) return std::nullopt;
  return *root / "coreml-cache" / sha256Hex(onnx);
}

ExecutionProvider parseExecutionProvider(const std::string& text) {
  if (text == "coreml") return ExecutionProvider::CoreML;
  if (text == "cpu") return ExecutionProvider::Cpu;
  throw std::invalid_argument("unknown execution provider '" + text + "' (coreml|cpu)");
}

CreatedSession createOrtSession(const fs::path& onnx, ExecutionProvider provider,
                                const std::optional<fs::path>& coreMlCache, const SessionConfigurer& configure) {
  CreatedSession result;
  if (provider == ExecutionProvider::CoreML) {
    const std::optional<fs::path> cache = usableCoreMlCache(coreMlCache);
    CoreMlAttempt coreMl = attemptCoreMl(onnx, cache, configure);
    if (coreMl.attached && !coreMl.ok && cache) {
      // The cache is shared: processes that compile at the same time race on it ("Error copying
      // compiled model to cache path"), and a compile that died half-way leaves a model that cannot
      // be loaded. Another process may be writing the directory, so it is left alone; compile
      // without it once before giving the CoreML provider up.
      std::fprintf(stderr,
                   "vc: CoreML session failed with the compile cache %s (%s); retrying without the cache "
                   "(delete %s to rebuild the cache)\n",
                   cache->c_str(), coreMl.error.c_str(), cache->c_str());
      coreMl = attemptCoreMl(onnx, std::nullopt, configure);
    }
    if (coreMl.ok) {
      result.session = std::move(coreMl.session);
      result.backend = "CoreML";
      return result;
    }
    if (coreMl.attached) {
      std::fprintf(stderr, "vc: CoreML session failed (%s); using CPU\n", coreMl.error.c_str());
    } else {
      std::fprintf(stderr, "vc: CoreML execution provider unavailable (%s); using CPU\n", coreMl.error.c_str());
    }
  }
  const Ort::SessionOptions cpuOptions = makeOptions(configure);
  result.session = Ort::Session(ortEnv(), onnx.c_str(), cpuOptions);
  result.backend = "CPU";
  return result;
}
```

换成：

```cpp
std::optional<fs::path> coreMlCacheDirectory(const fs::path& onnx) {
  const std::optional<fs::path> root = cacheRoot();
  if (!root) return std::nullopt;
  return *root / "coreml-cache" / sha256Hex(onnx);
}

std::optional<fs::path> tensorRtCacheDirectory(const fs::path& onnx, int width, int height) {
  const std::optional<fs::path> root = cacheRoot();
  if (!root) return std::nullopt;
  return *root / "trt-cache" / sha256Hex(onnx) / (std::to_string(width) + "x" + std::to_string(height));
}

ExecutionProvider parseExecutionProvider(const std::string& text) {
  if (text == "auto") return ExecutionProvider::Auto;
  if (text == "tensorrt") return ExecutionProvider::TensorRT;
  if (text == "cuda") return ExecutionProvider::Cuda;
  if (text == "coreml") return ExecutionProvider::CoreML;
  if (text == "cpu") return ExecutionProvider::Cpu;
  throw std::invalid_argument("unknown execution provider '" + text + "' (auto|tensorrt|cuda|coreml|cpu)");
}

std::string providerLabel(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider::Auto: return "auto";
    case ExecutionProvider::TensorRT: return "TensorRT";
    case ExecutionProvider::Cuda: return "CUDA";
    case ExecutionProvider::CoreML: return "CoreML";
    case ExecutionProvider::Cpu: return "CPU";
  }
  return "unknown";
}

std::vector<ExecutionProvider> providerOrder(ExecutionProvider requested, Platform platform) {
  using P = ExecutionProvider;
  switch (requested) {
    case P::Auto:
      if (platform == Platform::Linux) return {P::TensorRT, P::Cuda, P::Cpu};
      if (platform == Platform::MacOS) return {P::CoreML, P::Cpu};
      return {P::Cpu};
    case P::Cpu:
      return {P::Cpu};
    default:
      return {requested, P::Cpu};
  }
}

ExecutionProvider tryProviders(const std::vector<ExecutionProvider>& order, const ProviderTry& attempt,
                               const NoteSink& note) {
  if (order.empty() || order.back() != ExecutionProvider::Cpu) {
    throw std::invalid_argument("tryProviders: the providers must end with the CPU");
  }
  for (size_t i = 0;; ++i) {
    const ProviderAttempt tried = attempt(order[i]);
    if (tried.ok) return order[i];
    if (i + 1 == order.size()) throw VcError("the CPU session failed: " + tried.error);
    const std::string what = tried.attached ? " session failed (" : " execution provider unavailable (";
    const std::string next = (i + 2 == order.size() ? "using " : "trying ") + providerLabel(order[i + 1]);
    note("vc: " + providerLabel(order[i]) + what + tried.error + "); " + next);
  }
}

std::string gpuProviderUnavailableReason(ExecutionProvider provider, const std::vector<std::string>& available,
                                         bool withCuda) {
  const std::string name = ortProviderName(provider);
  if (std::find(available.begin(), available.end(), name) == available.end()) {
    return name + " is not in this onnxruntime build";
  }
  if (!withCuda) return "vc_redact was built without VC_WITH_CUDA";
  return "the TensorRT / CUDA session set-up comes with stage 1c of the Linux plan";
}

CreatedSession createOrtSession(const fs::path& onnx, ExecutionProvider provider, const SessionCaches& caches,
                                const SessionSettings& settings, const SessionConfigurer& configure) {
  CreatedSession result;
  const ProviderTry attempt = [&](ExecutionProvider p) -> ProviderAttempt {
    switch (p) {
      case ExecutionProvider::CoreML: {
        CoreMlAttempt coreMl =
            attemptCoreMlWithRetry(onnx, caches.coreMl ? caches.coreMl() : std::nullopt, settings, configure);
        if (coreMl.ok) result.session = std::move(coreMl.session);
        return ProviderAttempt{coreMl.ok, coreMl.attached, coreMl.error};
      }
      case ExecutionProvider::TensorRT:
      case ExecutionProvider::Cuda:
        return ProviderAttempt{false, false,
                               gpuProviderUnavailableReason(p, Ort::GetAvailableProviders(), builtWithCuda())};
      case ExecutionProvider::Cpu: {
        const Ort::SessionOptions options = makeOptions(settings, configure);
        result.session = Ort::Session(ortEnv(), onnx.c_str(), options);  // Ort::Exception: nothing left to try
        return ProviderAttempt{true, true, ""};
      }
      case ExecutionProvider::Auto:
        break;
    }
    throw std::logic_error("createOrtSession: auto is an order of providers, not one to try");
  };
  const NoteSink note = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
  const ExecutionProvider used = tryProviders(providerOrder(provider, buildPlatform()), attempt, note);
  result.backend = providerLabel(used);
  return result;
}
```

（macOS 上 `auto` 与 `coreml` 走的是同一串尝试：CoreML（带缓存失败就不带缓存再试一次），再 CPU；stderr 的两种行与改动前一字不差，见 `MacAutoFallsFromCoreMlToTheCpu`。）

- [ ] **Step 4: 检测器接上会话设置与按需的缓存目录**

把 `cpp/src/detect/onnx_yolo_detector.h` 里的：

```cpp
// backend() "CoreML" means the CoreML execution provider was registered for the session; onnxruntime
// may still run nodes the provider does not support on the CPU. The model must be float32.
```

换成：

```cpp
// backend() names the execution provider the session was created with (providerLabel: "CoreML", "CPU", ...);
// onnxruntime may still run nodes the provider does not support on the CPU. The model must be float32.
```

把 `cpp/src/detect/onnx_yolo_detector.h` 里的：

```cpp
  // Frames passed to detect() must be frameWidth x frameHeight. CoreML keeps its compiled model in
```

换成：

```cpp
  // Frames passed to detect() must be frameWidth x frameHeight. The providers of providerOrder(provider) are tried
  // in turn (createOrtSession), with `settings` for every session. CoreML keeps its compiled model in
```

把 `cpp/src/detect/onnx_yolo_detector.h` 里的：

```cpp
                   const DecodeParams& params, int frameWidth, int frameHeight);
```

换成：

```cpp
                   const DecodeParams& params, int frameWidth, int frameHeight, const SessionSettings& settings = {});
```

把 `cpp/src/detect/onnx_yolo_detector.cpp` 里的：

```cpp
                                   const DecodeParams& params, int frameWidth, int frameHeight)
```

换成：

```cpp
                                   const DecodeParams& params, int frameWidth, int frameHeight,
                                   const SessionSettings& settings)
```

把 `cpp/src/detect/onnx_yolo_detector.cpp` 里的：

```cpp
    const std::optional<fs::path> cache =
        provider == ExecutionProvider::CoreML ? coreMlCacheDirectory(onnx) : std::nullopt;
    CreatedSession created = createOrtSession(onnx, provider, cache);
```

换成：

```cpp
    const int inputWidth = s.info.inputWidth;
    const int inputHeight = s.info.inputHeight;
    const SessionCaches caches{[&] { return coreMlCacheDirectory(onnx); },
                               [&] { return tensorRtCacheDirectory(onnx, inputWidth, inputHeight); }};
    CreatedSession created = createOrtSession(onnx, provider, caches, settings);
```

把 `cpp/src/detect/yunet_detector.h` 里的：

```cpp
                int frameWidth, int frameHeight);
```

换成：

```cpp
                int frameWidth, int frameHeight, const SessionSettings& settings = {});
```

把 `cpp/src/detect/yunet_detector.h` 里的：

```cpp
  // Of the current session ("CoreML" or "CPU").
```

换成：

```cpp
  // Of the current session (providerLabel, e.g. "CoreML" or "CPU").
```

把 `cpp/src/detect/yunet_detector.cpp` 里的：

```cpp
  ExecutionProvider provider = ExecutionProvider::Cpu;
  YuNetParams params;
```

换成：

```cpp
  ExecutionProvider provider = ExecutionProvider::Cpu;
  SessionSettings settings{};
  YuNetParams params;
```

把 `cpp/src/detect/yunet_detector.cpp` 里的：

```cpp
      const std::optional<fs::path> cache =
          provider == ExecutionProvider::CoreML ? yunetCoreMlCacheDirectory(onnx, pw, ph) : std::nullopt;
      created = createOrtSession(onnx, provider, cache, fixSize);
```

换成：

```cpp
      const SessionCaches caches{[&] { return yunetCoreMlCacheDirectory(onnx, pw, ph); },
                                 [&] { return tensorRtCacheDirectory(onnx, pw, ph); }};
      created = createOrtSession(onnx, provider, caches, settings, fixSize);
```

把 `cpp/src/detect/yunet_detector.cpp` 里的：

```cpp
                             int frameWidth, int frameHeight)
```

换成：

```cpp
                             int frameWidth, int frameHeight, const SessionSettings& settings)
```

把 `cpp/src/detect/yunet_detector.cpp` 里的：

```cpp
  s.provider = provider;
  s.params = params;
```

换成：

```cpp
  s.provider = provider;
  s.settings = settings;
  s.params = params;
```

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
  ExecutionProvider provider = ExecutionProvider::CoreML;
```

换成：

```cpp
  ExecutionProvider provider = ExecutionProvider::Auto;  // providerOrder: CoreML then CPU on macOS
```

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
  // "student/CoreML", "student/CPU" or "fake"; with a face model both, e.g. "student/CoreML + yunet/CoreML"
```

换成：

```cpp
  // "student/<backend>" (providerLabel: "student/CoreML", "student/CPU", on Linux + NVIDIA "student/TensorRT") or
  // "fake"; with a face model both, e.g. "student/CoreML + yunet/CoreML"
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
std::unique_ptr<Detector> makeDetector(const RedactOptions& o, int width, int height) {
  std::unique_ptr<Detector> detector;
  if (o.fakeDetections) {
    detector = std::make_unique<FakeDetector>(*o.fakeDetections);
  } else {
    detector = std::make_unique<OnnxYoloDetector>(*o.modelDir, o.provider, o.config.decode, width, height);
  }
  if (!o.faceModelDir) return detector;
  auto face = std::make_unique<YuNetDetector>(*o.faceModelDir, o.provider, o.config.face, width, height);
```

换成：

```cpp
std::unique_ptr<Detector> makeDetector(const RedactOptions& o, int width, int height) {
  const SessionSettings settings{o.config.ortIntraThreads, o.config.trtFp16};
  std::unique_ptr<Detector> detector;
  if (o.fakeDetections) {
    detector = std::make_unique<FakeDetector>(*o.fakeDetections);
  } else {
    detector = std::make_unique<OnnxYoloDetector>(*o.modelDir, o.provider, o.config.decode, width, height, settings);
  }
  if (!o.faceModelDir) return detector;
  auto face = std::make_unique<YuNetDetector>(*o.faceModelDir, o.provider, o.config.face, width, height, settings);
```

把 `cpp/apps/vc_redact/redact_cli.cpp` 里的：

```cpp
  --ep <provider>           coreml | cpu (default coreml, falls back to cpu)
```

换成：

```cpp
  --ep <provider>           auto | tensorrt | cuda | coreml | cpu (default auto: TensorRT, CUDA, CPU on
                            Linux; CoreML, CPU on macOS); a provider that cannot be used falls back
                            (one line on stderr), an explicit one to cpu
```

把 `cpp/apps/vc_detect/detect_cli.cpp` 里的：

```cpp
  return R"(Usage: vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep cpu|coreml] [--score <x>]
```

换成：

```cpp
  return R"(Usage: vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep <provider>] [--score <x>]
```

把 `cpp/apps/vc_detect/detect_cli.cpp` 里的：

```cpp
  --ep <provider> cpu | coreml (default cpu)
```

换成：

```cpp
  --ep <provider> cpu | coreml | auto | tensorrt | cuda (default cpu; see vc_redact --help)
```

- [ ] **Step 5: 跑测试**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*:YuNetDecode.*:YuNetDetector.*:DetectionMerge.*:MergedDetector.*:Detector.*:ExecutionProvider.*'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```

预期：`0`；第四组 `[  PASSED  ] 93 tests.`（`ExecutionProvider` 11 个、`OnnxYoloDetector` 15 个；CoreML 相关用例照旧，`AnUnavailableGpuProviderFallsBackToTheCpu` 在本机看到 `vc: TensorRT execution provider unavailable (TensorrtExecutionProvider is not in this onnxruntime build); using CPU`）；第六组 `[  PASSED  ] 82 tests.`。

- [ ] **Step 6: 文档**

把 `cpp/README.md` 里的：

```markdown
| `--ep` | coreml | `coreml` / `cpu`，检测使用的 onnxruntime 执行提供者 |
```

换成：

```markdown
| `--ep` | auto | `auto` / `tensorrt` / `cuda` / `coreml` / `cpu`，检测使用的 onnxruntime 执行提供者。`auto` 在 Linux 上依次试 TensorRT、CUDA、CPU，在 macOS 上试 CoreML、CPU（与原来的默认 `coreml` 相同）；显式指定的用不了时退到 CPU。跳过的每一个在 stderr 写一行原因，例如 `vc: TensorRT execution provider unavailable (…); trying CUDA`。TensorRT / CUDA 的会话在 GPU 计划里接上，在那之前它们总是报不可用 |
```

把 `cpp/README.md` 里的：

```markdown
`student/CPU` 表示用的是 CPU（显式指定了 `--ep cpu`，或 CoreML 无法使用、已回退，回退时 stderr 会有提示）；
```

换成：

```markdown
`student/CPU` 表示用的是 CPU（显式指定了 `--ep cpu`，或前面的执行提供者无法使用、已回退，回退时 stderr 会有提示）；Linux + NVIDIA 上会是 `student/TensorRT`、`student/CUDA`（GPU 计划）；
```

把 `cpp/README.md` 里的：

```markdown
- **模型必须是 float32**：
```

换成：

```markdown
- **TensorRT 引擎缓存**（GPU 计划里用上）：`$VC_CACHE/trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`，与 CoreML 缓存同一做法，按模型内容和输入尺寸分目录（学生模型是它的输入尺寸，如 `1920x1088`；YuNet 是补边后的帧尺寸，如 `3840x2176`）。
- **模型必须是 float32**：
```

把 `cpp/README.md` 里的：

```markdown
| `--ep` | cpu | `cpu` / `coreml`；做一致性比较时建议用 `cpu`，避免 CoreML 带来额外的数值差异 |
```

换成：

```markdown
| `--ep` | cpu | `cpu` / `coreml`（也接受 `auto` / `tensorrt` / `cuda`，回退规则同 `vc_redact`）；做一致性比较时建议用 `cpu`，避免 CoreML 带来额外的数值差异 |
```

把 `cpp/README.md` 里的：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*:YuNetDecode.*:YuNetDetector.*:DetectionMerge.*:MergedDetector.*:Detector.*'
```

换成：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*:YuNetDecode.*:YuNetDetector.*:DetectionMerge.*:MergedDetector.*:Detector.*:ExecutionProvider.*'
```

- [ ] **Step 7: 提交**

```bash
git add cpp/src/detect/execution_provider.h cpp/src/detect/ort_session.h cpp/src/detect/ort_session.cpp \
    cpp/src/detect/onnx_yolo_detector.h cpp/src/detect/onnx_yolo_detector.cpp cpp/src/detect/yunet_detector.h \
    cpp/src/detect/yunet_detector.cpp cpp/apps/vc_redact/redact_job.h cpp/apps/vc_redact/redact_job.cpp \
    cpp/apps/vc_redact/redact_cli.cpp cpp/apps/vc_detect/detect_cli.cpp cpp/tests/execution_provider_test.cpp \
    cpp/tests/CMakeLists.txt cpp/tests/onnx_yolo_detector_test.cpp cpp/tests/detect_cli_test.cpp \
    cpp/tests/redact_cli_test.cpp cpp/README.md
git commit -m "feat: add auto, tensorrt and cuda execution providers with ordered fallback"
```

### Task 5: `--hwaccel cuda` 与按平台的 `auto`

**Files:**
- Modify: `cpp/src/io/video_reader.h`, `cpp/src/io/video_reader.cpp`
- Modify: `cpp/apps/vc_redact/redact_job.h`（`defaultRedactHwAccel` 声明、`hwAccel` 默认值）、`cpp/apps/vc_redact/redact_job.cpp`（定义）
- Modify: `cpp/apps/vc_redact/redact_cli.cpp`, `cpp/apps/vc_extract/extract_cli.cpp`（帮助文本）
- Test: `cpp/tests/video_reader_test.cpp`, `cpp/tests/redact_cli_test.cpp`, `cpp/tests/extract_cli_test.cpp`
- Modify: `cpp/README.md`（`vc_redact` 与 `vc_extract` 的 `--hwaccel` 两行）

**Interfaces:**
- Consumes: `Platform`、`buildPlatform()`（Task 1）。
- Produces（GPU 计划 1c / 2b 使用）:
  - `enum class HwAccel { None, VideoToolbox, Cuda, Auto };`，`parseHwAccel("none"|"videotoolbox"|"cuda"|"auto")`。
  - `enum class HwDevice { None, VideoToolbox, Cuda };`
  - `struct HwDecodePlan { HwDevice device = HwDevice::None; bool required = false; bool noteOnFallback = false; };`
  - `HwDecodePlan hwDecodePlan(HwAccel requested, Platform platform);`
  - `HwAccel defaultRedactHwAccel(Platform platform);`（`redact_job.h`）—— Linux `Auto`，其余 `None`。
  - 打不开 CUDA 时 stderr：`vc: CUDA (NVDEC) decoding unavailable (<原因>); decoding in software`。

- [ ] **Step 1: 写失败的测试**

把 `cpp/tests/video_reader_test.cpp` 里的：

```cpp
  EXPECT_EQ(parseHwAccel("auto"), HwAccel::Auto);
  EXPECT_THROW(parseHwAccel("cuda"), std::invalid_argument);
}
```

换成：

```cpp
  EXPECT_EQ(parseHwAccel("auto"), HwAccel::Auto);
  EXPECT_EQ(parseHwAccel("cuda"), HwAccel::Cuda);
  EXPECT_THROW(parseHwAccel("vulkan"), std::invalid_argument);
  EXPECT_THROW(parseHwAccel("nvdec"), std::invalid_argument);
}

// spec 2026-10-20 §3: auto is NVDEC on Linux (software with a note when it cannot be opened) and VideoToolbox on
// macOS (silently software otherwise, as before); an explicit cuda falls back with a note, videotoolbox is required.
TEST(VideoReader, PlansTheHardwareDecoderPerPlatform) {
  const auto same = [](const HwDecodePlan& a, const HwDecodePlan& b) {
    return a.device == b.device && a.required == b.required && a.noteOnFallback == b.noteOnFallback;
  };
  for (const Platform p : {Platform::Linux, Platform::MacOS, Platform::Other}) {
    EXPECT_TRUE(same(hwDecodePlan(HwAccel::None, p), HwDecodePlan{}));
    EXPECT_TRUE(same(hwDecodePlan(HwAccel::VideoToolbox, p), HwDecodePlan{HwDevice::VideoToolbox, true, false}));
    EXPECT_TRUE(same(hwDecodePlan(HwAccel::Cuda, p), HwDecodePlan{HwDevice::Cuda, false, true}));
  }
  EXPECT_TRUE(same(hwDecodePlan(HwAccel::Auto, Platform::Linux), HwDecodePlan{HwDevice::Cuda, false, true}));
  EXPECT_TRUE(same(hwDecodePlan(HwAccel::Auto, Platform::MacOS), HwDecodePlan{HwDevice::VideoToolbox, false, false}));
  EXPECT_TRUE(same(hwDecodePlan(HwAccel::Auto, Platform::Other), HwDecodePlan{}));
}

// Without an NVIDIA GPU (always on macOS) cuda decodes in software and says so on stderr, once.
TEST(VideoReader, CudaFallsBackToSoftwareWithANote) {
  testing::internal::CaptureStderr();
  VideoReader reader(test::basicVideo().string(), HwAccel::Cuda);
  int frames = 0;
  while (reader.next() != nullptr) ++frames;
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(frames, 60);
  if (reader.usingHardware()) GTEST_SKIP() << "NVDEC decoded this video";
  EXPECT_NE(log.find("vc: CUDA (NVDEC) decoding unavailable ("), std::string::npos) << log;
  EXPECT_NE(log.find("); decoding in software"), std::string::npos) << log;
}
```

把 `cpp/tests/redact_cli_test.cpp` 里的：

```cpp
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--hwaccel", "cuda"},
```

换成：

```cpp
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--hwaccel", "vulkan"},
```

把 `cpp/tests/redact_cli_test.cpp` 里的：

```cpp
TEST(RedactCli, ParsesEveryExecutionProvider) {
```

换成：

```cpp
// spec 2026-10-20 §3: --hwaccel cuda parses on every platform (on macOS the reader then decodes in software);
// the default is auto on Linux and none elsewhere.
TEST(RedactCli, ParsesCudaDecodingAndThePlatformDefault) {
  EXPECT_EQ(parseRedactArgs({"--input", "a.mp4", "--out", "o", "--model", "m", "--hwaccel", "cuda"}).hwAccel,
            HwAccel::Cuda);
  EXPECT_EQ(parseRedactArgs({"--input", "a.mp4", "--out", "o", "--model", "m"}).hwAccel,
            defaultRedactHwAccel(buildPlatform()));
  EXPECT_EQ(defaultRedactHwAccel(Platform::Linux), HwAccel::Auto);
  EXPECT_EQ(defaultRedactHwAccel(Platform::MacOS), HwAccel::None);
  EXPECT_EQ(defaultRedactHwAccel(Platform::Other), HwAccel::None);
}

TEST(RedactCli, ParsesEveryExecutionProvider) {
```

把 `cpp/tests/extract_cli_test.cpp` 里的：

```cpp
      {"--input", "a.mp4", "--out", "o", "--hwaccel", "cuda"},  // unsupported here
```

换成：

```cpp
      {"--input", "a.mp4", "--out", "o", "--hwaccel", "vulkan"},  // unknown
```

（`RedactCli.AppliesDefaults` 里的 `EXPECT_EQ(o.hwAccel, HwAccel::None);` 保持不动：本机是 macOS，默认仍是 `none`。）

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`video_reader_test.cpp` 报 `no member named 'Cuda' in 'vc::HwAccel'`、`unknown type name 'HwDecodePlan'`，`redact_cli_test.cpp` 报 `use of undeclared identifier 'defaultRedactHwAccel'`。

- [ ] **Step 3: 实现**

把 `cpp/src/io/video_reader.h` 里的：

```cpp
#include "common/luma_plane.h"
```

换成：

```cpp
#include "common/luma_plane.h"
#include "common/platform.h"
```

把 `cpp/src/io/video_reader.h` 里的：

```cpp
enum class HwAccel { None, VideoToolbox, Auto };

// Parses "none" | "videotoolbox" | "auto"; throws std::invalid_argument.
HwAccel parseHwAccel(const std::string& text);
```

换成：

```cpp
enum class HwAccel { None, VideoToolbox, Cuda, Auto };

// Parses "none" | "videotoolbox" | "cuda" | "auto"; throws std::invalid_argument.
HwAccel parseHwAccel(const std::string& text);

// The hardware decoder a requested mode means on a platform (spec 2026-10-20 §3).
enum class HwDevice { None, VideoToolbox, Cuda };
struct HwDecodePlan {
  HwDevice device = HwDevice::None;  // the device to try; None decodes in software
  bool required = false;             // a device that cannot be opened is an error instead of software decoding
  bool noteOnFallback = false;       // software decoding instead of the device gets a line on stderr
};
// none: software. videotoolbox: VideoToolbox, required. cuda: CUDA (NVDEC), software with a note when it cannot be
// opened (on macOS it never can). auto: CUDA with that note on Linux, VideoToolbox silently falling back on macOS,
// software elsewhere.
HwDecodePlan hwDecodePlan(HwAccel requested, Platform platform);
```

把 `cpp/src/io/video_reader.h` 里的：

```cpp
// A decoded frame in CPU memory, always 8-bit YUV (planar or NV12), so
// plane 0 is the luma plane.
```

换成：

```cpp
// A decoded frame in CPU memory, always 8-bit YUV (planar or NV12), so
// plane 0 is the luma plane. Hardware frames (VideoToolbox, CUDA) are copied back to memory.
```

把 `cpp/src/io/video_reader.h` 里的：

```cpp
  // Throws VcError when the file cannot be opened, has no decodable video
  // stream, or VideoToolbox was requested explicitly and is unavailable.
  // HwAccel::Auto falls back to software decoding.
```

换成：

```cpp
  // Throws VcError when the file cannot be opened, has no decodable video
  // stream, or VideoToolbox was requested explicitly and is unavailable.
  // HwAccel::Auto and HwAccel::Cuda fall back to software decoding (hwDecodePlan).
```

把 `cpp/src/io/video_reader.cpp` 里的：

```cpp
#include <algorithm>
#include <functional>
#include <stdexcept>
```

换成：

```cpp
#include <algorithm>
#include <cstdio>
#include <functional>
#include <stdexcept>
```

把 `cpp/src/io/video_reader.cpp` 里的：

```cpp
AVPixelFormat preferVideoToolbox(AVCodecContext* ctx, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    if (*f == AV_PIX_FMT_VIDEOTOOLBOX) return *f;
  }
  return avcodec_default_get_format(ctx, formats);
}

}  // namespace

HwAccel parseHwAccel(const std::string& text) {
  if (text == "none") return HwAccel::None;
  if (text == "videotoolbox") return HwAccel::VideoToolbox;
  if (text == "auto") return HwAccel::Auto;
  throw std::invalid_argument("unknown hwaccel '" + text + "' (none|videotoolbox|auto)");
}
```

换成：

```cpp
AVPixelFormat preferVideoToolbox(AVCodecContext* ctx, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    if (*f == AV_PIX_FMT_VIDEOTOOLBOX) return *f;
  }
  return avcodec_default_get_format(ctx, formats);
}

AVPixelFormat preferCuda(AVCodecContext* ctx, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    if (*f == AV_PIX_FMT_CUDA) return *f;
  }
  return avcodec_default_get_format(ctx, formats);
}

bool isHardwareFrame(int format) { return format == AV_PIX_FMT_VIDEOTOOLBOX || format == AV_PIX_FMT_CUDA; }

}  // namespace

HwAccel parseHwAccel(const std::string& text) {
  if (text == "none") return HwAccel::None;
  if (text == "videotoolbox") return HwAccel::VideoToolbox;
  if (text == "cuda") return HwAccel::Cuda;
  if (text == "auto") return HwAccel::Auto;
  throw std::invalid_argument("unknown hwaccel '" + text + "' (none|videotoolbox|cuda|auto)");
}

HwDecodePlan hwDecodePlan(HwAccel requested, Platform platform) {
  switch (requested) {
    case HwAccel::None:
      return {};
    case HwAccel::VideoToolbox:
      return {HwDevice::VideoToolbox, true, false};
    case HwAccel::Cuda:
      return {HwDevice::Cuda, false, true};
    case HwAccel::Auto:
      if (platform == Platform::Linux) return {HwDevice::Cuda, false, true};
      if (platform == Platform::MacOS) return {HwDevice::VideoToolbox, false, false};
      return {};
  }
  return {};
}
```

把 `cpp/src/io/video_reader.cpp` 里的：

```cpp
void VideoReader::Impl::attachHardware(HwAccel hwAccel) {
  if (hwAccel == HwAccel::None) return;
  AVBufferRef* device = nullptr;
  const int ret =
      av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_VIDEOTOOLBOX, nullptr, nullptr, 0);
  if (ret < 0) {
    if (hwAccel == HwAccel::VideoToolbox) {
      throw VcError("VideoToolbox unavailable: " + avErrorText(ret));
    }
    return;  // Auto: software decoding
  }
  hwDevice.reset(device);
  decoder->hw_device_ctx = av_buffer_ref(device);
  decoder->get_format = preferVideoToolbox;
}
```

换成：

```cpp
void VideoReader::Impl::attachHardware(HwAccel hwAccel) {
  const HwDecodePlan plan = hwDecodePlan(hwAccel, buildPlatform());
  if (plan.device == HwDevice::None) return;
  const bool cuda = plan.device == HwDevice::Cuda;
  const char* name = cuda ? "CUDA (NVDEC)" : "VideoToolbox";
  AVBufferRef* device = nullptr;
  const int ret = av_hwdevice_ctx_create(&device, cuda ? AV_HWDEVICE_TYPE_CUDA : AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                                         nullptr, nullptr, 0);
  if (ret < 0) {
    if (plan.required) throw VcError(std::string(name) + " unavailable: " + avErrorText(ret));
    if (plan.noteOnFallback) {
      std::fprintf(stderr, "vc: %s decoding unavailable (%s); decoding in software\n", name, avErrorText(ret).c_str());
    }
    return;
  }
  hwDevice.reset(device);
  decoder->hw_device_ctx = av_buffer_ref(device);
  decoder->get_format = cuda ? preferCuda : preferVideoToolbox;
}
```

把 `cpp/src/io/video_reader.cpp` 里的：

```cpp
  const AVFrame* frame = decoded.get();
  if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
```

换成：

```cpp
  const AVFrame* frame = decoded.get();
  if (isHardwareFrame(frame->format)) {
```

（CUDA 帧和 VideoToolbox 帧一样用 `av_hwframe_transfer_data` 拷回内存，NVDEC 给的是 NV12（10 位是 P010，随后照旧转成 NV12），之后的流程不变；这条路径在本机走不到，GPU 计划 1c 在目标机上验证。）

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
struct RedactOptions {
```

换成：

```cpp
// vc_redact's --hwaccel default (spec 2026-10-20 §3): auto on Linux (CUDA, else software), none elsewhere (on macOS
// software decoding is faster than VideoToolbox for 1080p H.264).
HwAccel defaultRedactHwAccel(Platform platform);

struct RedactOptions {
```

把 `cpp/apps/vc_redact/redact_job.h` 里的：

```cpp
  HwAccel hwAccel = HwAccel::None;
```

换成：

```cpp
  HwAccel hwAccel = defaultRedactHwAccel(buildPlatform());
```

把 `cpp/apps/vc_redact/redact_job.cpp` 里的：

```cpp
size_t pipelineQueueCapacity(int width, int height) {
```

换成：

```cpp
HwAccel defaultRedactHwAccel(Platform platform) { return platform == Platform::Linux ? HwAccel::Auto : HwAccel::None; }

size_t pipelineQueueCapacity(int width, int height) {
```

把 `cpp/apps/vc_redact/redact_cli.cpp` 里的：

```cpp
  --hwaccel <mode>          decoder: none | auto | videotoolbox (default none)
```

换成：

```cpp
  --hwaccel <mode>          decoder: none | auto | videotoolbox | cuda (default none on macOS, auto on
                            Linux). auto: CUDA (NVDEC) on Linux, VideoToolbox on macOS, software
                            when it cannot be opened; cuda falls back to software with a line on
                            stderr (always on macOS); an explicit videotoolbox that fails is an error
```

把 `cpp/apps/vc_extract/extract_cli.cpp` 里的：

```cpp
  --hwaccel <mode>        auto | none | videotoolbox (default auto)
```

换成：

```cpp
  --hwaccel <mode>        auto | none | videotoolbox | cuda (default auto)
```

- [ ] **Step 4: 跑测试**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*:VideoWriter.*:FrameFileName.*'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```

预期：`0`；第二组 `[  PASSED  ] 47 tests.`（`CudaFallsBackToSoftwareWithANote` 在本机看到 `vc: CUDA (NVDEC) decoding unavailable (...); decoding in software`）；第三组 `[  PASSED  ] 16 tests.`；第六组 `[  PASSED  ] 83 tests.`。

- [ ] **Step 5: 文档**

把 `cpp/README.md` 里的：

```markdown
| `--hwaccel` | none | 解码方式：`none` / `auto` / `videotoolbox`（与 `vc_extract` 不同，默认不用硬解） |
```

换成：

```markdown
| `--hwaccel` | macOS `none`，Linux `auto` | 解码方式：`none` / `auto` / `videotoolbox` / `cuda`。`auto` 在 Linux 上试 CUDA（NVDEC，解完拷回内存），打不开退回软件解码并在 stderr 写一行；在 macOS 上试 VideoToolbox，打不开悄悄退回软件解码（与 `vc_extract` 不同，macOS 上默认不用硬解，1080p H.264 软解更快）。`cuda` 打不开时退回软件解码并写一行 `vc: CUDA (NVDEC) decoding unavailable (原因); decoding in software`（macOS 上总是这样）；显式的 `videotoolbox` 打不开则报错 |
```

把 `cpp/README.md` 里的：

```markdown
| `--hwaccel` | auto | `auto` / `none` / `videotoolbox`；
```

换成：

```markdown
| `--hwaccel` | auto | `auto` / `none` / `videotoolbox` / `cuda`（`auto` 与 `cuda` 的含义同 `vc_redact`）；
```

- [ ] **Step 6: 提交**

```bash
git add cpp/src/io/video_reader.h cpp/src/io/video_reader.cpp cpp/apps/vc_redact/redact_job.h \
    cpp/apps/vc_redact/redact_job.cpp cpp/apps/vc_redact/redact_cli.cpp cpp/apps/vc_extract/extract_cli.cpp \
    cpp/tests/video_reader_test.cpp cpp/tests/redact_cli_test.cpp cpp/tests/extract_cli_test.cpp cpp/README.md
git commit -m "feat: accept --hwaccel cuda and default to auto on Linux"
```

### Task 6: `vc_eval integrity` 检查 H.264 与 Level

**Files:**
- Modify: `py/vc_eval/integrity.py`（`StreamInfo.level`、`_stream`、Level 表与 `required_h264_level()`、`_CHECKS`、`_check_h264()`、`check_integrity()` 拆出 `_compare()`）
- Modify: `py/vc_eval/report.py`（`INTEGRITY_CHECKS`、完整性表头）、`py/vc_eval/cli.py`（子命令帮助）
- Test: `py/tests/test_eval_integrity.py`, `py/tests/test_eval_run.py`
- Modify: `py/README.md`（单项子命令一节）、`docs/USAGE.md`（"检查结果"第 3 条）

**Interfaces:**
- Consumes: 无（与 Task 3 的 `h264LevelIdc()` 用同一张表，两边的测试用同一组例子：4K60 → 52、1080p 59.94 → 42、1080p30 → 40）。
- Produces（Task 10、GPU 计划 3b 的 `accept_gpu.sh` 使用）:
  - `StreamInfo.level: int | None`（ffprobe 的 `level`，-99 或缺失时 None）
  - `H264_LEVELS`（`(level_idc, MaxMBPS, MaxFS)` 元组）
  - `required_h264_level(width: int, height: int, fps: float) -> int | None`
  - `IntegrityReport.checks` 多两项 `"h264"`、`"h264_level"`（排在原来六项之后），`details` 同名；`report.INTEGRITY_CHECKS` 同样多这两项，完整性表多 `H.264`、`Level` 两列。

- [ ] **Step 1: 写失败的测试**

把 `py/tests/test_eval_integrity.py` 里的：

```python
from vc_eval.integrity import FFMPEG, check_integrity, probe
```

换成：

```python
from vc_eval.integrity import FFMPEG, check_integrity, probe, required_h264_level
```

把 `py/tests/test_eval_integrity.py` 里的：

```python
def _video(path, seconds=1, audio=True, rate=30, faststart=False, size="160x120"):
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
```

换成：

```python
def _video(path, seconds=1, audio=True, rate=30, faststart=False, size="160x120", codec="libx264", level=None):
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:a", "aac", "-shortest"]
    args += ["-c:v", codec, "-pix_fmt", "yuv420p"]
    if codec == "libx265":
        args += ["-x265-params", "log-level=error"]
    if level is not None:
        args += ["-level", level]
```

把 `py/tests/test_eval_integrity.py` 里的：

```python
    assert r.passed and set(r.checks) == {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync"}
```

换成：

```python
    assert r.passed and set(r.checks) == {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync",
                                          "h264", "h264_level"}
```

把 `py/tests/test_eval_integrity.py` 里的：

```python
CHECKS = {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync"}
```

换成：

```python
CHECKS = {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync", "h264", "h264_level"}
```

在 `py/tests/test_eval_integrity.py` 末尾追加：

```python


@pytest.mark.parametrize("width, height, fps, level", [
    (3840, 2160, 60, 52),           # 4K60 needs 5.2 (spec 2026-10-20 §3)
    (3840, 2160, 60000 / 1001, 52),
    (3840, 2160, 30, 51),
    (1920, 1080, 60000 / 1001, 42),
    (1920, 1080, 30, 40),
    (1280, 720, 60, 32),
    (1280, 720, 30, 31),
    (160, 120, 30, 11),
    (7680, 4320, 60, 61),
    (7680, 4320, 120, 62),
    (7680, 4320, 240, None),        # beyond the last level
])
def test_required_h264_level_follows_table_a1(width, height, fps, level):
    assert required_h264_level(width, height, fps) == level


def test_an_h264_output_at_a_sufficient_level_passes(tmp_path):
    src = _video(tmp_path / "src.mp4", seconds=0.2, audio=False, rate=60, size="1280x720")
    r = check_integrity(src, src)
    assert r.checks["h264"] is True and r.checks["h264_level"] is True
    assert r.details["h264_level"].endswith("1280x720 at 60.00 fps needs 3.2")


def test_an_hevc_output_fails_the_h264_checks(tmp_path):
    src = _video(tmp_path / "src.mp4", audio=False)
    out = _video(tmp_path / "out.mp4", audio=False, codec="libx265")
    r = check_integrity(src, out)
    assert r.checks["h264"] is False and r.checks["h264_level"] is False and not r.passed
    assert r.details["h264"] == "redacted video codec hevc"
    assert "not H.264" in r.details["h264_level"]
    assert r.checks["frame_count"] is True and r.checks["resolution"] is True  # only the codec checks fail


def test_a_level_below_what_the_size_and_rate_need_fails(tmp_path):
    src = _video(tmp_path / "src.mp4", seconds=0.2, audio=False, rate=60, size="1280x720")
    out = _video(tmp_path / "out.mp4", seconds=0.2, audio=False, rate=60, size="1280x720", level="3.1")
    assert probe(out).video.level == 31
    r = check_integrity(src, out)
    assert r.checks["h264"] is True and r.checks["h264_level"] is False and not r.passed
    assert r.details["h264_level"] == "level 3.1; 1280x720 at 60.00 fps needs 3.2"


def test_an_unknown_level_fails_the_level_check(tmp_path, monkeypatch):
    src, out = _video(tmp_path / "src.mp4"), _video(tmp_path / "out.mp4")

    def no_level(data):
        for stream in data["streams"]:
            stream.pop("level", None)

    _drop_from_probe(monkeypatch, out, no_level)
    assert probe(out).video.level is None
    r = check_integrity(src, out)
    assert r.checks["h264_level"] is False and "level unknown" in r.details["h264_level"]
```

把 `py/tests/test_eval_run.py` 里的：

```python
    assert "| 分辨率 |" in integrity and _row(integrity, "| seg | 10 |").count("✓") == 6
```

换成：

```python
    assert "| 分辨率 |" in integrity and "| H.264 | Level |" in integrity and _row(integrity, "| seg | 10 |").count("✓") == 8
```

- [ ] **Step 2: 确认测试失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_integrity.py -p no:cacheprovider -q
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_run.py -p no:cacheprovider -q
cd ..
```

预期：第一个 FAIL，收集阶段 `ImportError: cannot import name 'required_h264_level' from 'vc_eval.integrity'`；第二个 FAIL，`test_choice_needs_every_stem_and_the_integrity_checks` 断言失败（表里还没有 `H.264` / `Level` 两列）。

- [ ] **Step 3: 实现**

把 `py/vc_eval/integrity.py` 里的：

```python
    width: int | None = None
    height: int | None = None
```

换成：

```python
    width: int | None = None
    height: int | None = None
    level: int | None = None  # H.264 / HEVC level as ffprobe reports it (level_idc: 52 for 5.2); None when unknown
```

把 `py/vc_eval/integrity.py` 里的：

```python
        width=_number(s.get("width"), int),
        height=_number(s.get("height"), int),
    )
```

换成：

```python
        width=_number(s.get("width"), int),
        height=_number(s.get("height"), int),
        level=_level(s.get("level")),
    )


def _level(value) -> int | None:
    """ffprobe's level, None for a missing or unknown one (it prints -99 when the stream has none)."""
    level = _number(value, int)
    return level if level is not None and level > 0 else None


# H.264 Table A-1 (ITU-T H.264 Annex A): level_idc, MaxMBPS (macroblocks per second), MaxFS (macroblocks per frame).
# Level 1b has the limits of level 1 and is left out.
H264_LEVELS = (
    (10, 1485, 99), (11, 3000, 396), (12, 6000, 396), (13, 11880, 396), (20, 11880, 396), (21, 19800, 792),
    (22, 20250, 1620), (30, 40500, 1620), (31, 108000, 3600), (32, 216000, 5120), (40, 245760, 8192),
    (41, 245760, 8192), (42, 522240, 8704), (50, 589824, 22080), (51, 983040, 36864), (52, 2073600, 36864),
    (60, 4177920, 139264), (61, 8355840, 139264), (62, 16711680, 139264),
)


def required_h264_level(width: int, height: int, fps: float) -> int | None:
    """level_idc of the lowest H.264 level whose frame size and macroblock rate hold width x height at fps
    (3840x2160 at 60 fps: 52, i.e. 5.2); None beyond level 6.2. Mirrors h264LevelIdc in cpp/src/io/video_writer.cpp."""
    frame_mbs = math.ceil(width / 16) * math.ceil(height / 16)
    for level, max_mbps, max_fs in H264_LEVELS:
        if frame_mbs <= max_fs and frame_mbs * fps <= max_mbps:
            return level
    return None


def _level_text(level: int) -> str:
    return f"{level // 10}.{level % 10}"
```

把 `py/vc_eval/integrity.py` 里的：

```python
_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync")
```

换成：

```python
_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync", "h264", "h264_level")
```

把 `py/vc_eval/integrity.py` 里的：

```python
def check_integrity(source: Path, redacted: Path) -> IntegrityReport:
    """Source and redacted must agree; a source that cannot be probed raises ValueError, a broken output fails."""
    s = probe(source)
    try:
        r = probe(redacted)
    except ValueError as exc:
        return _unreadable(str(exc))
    frame = 1.0 / s.fps if s.fps > 0 else 1.0 / 30
```

换成：

```python
def _check_h264(report: IntegrityReport, r: MediaInfo) -> None:
    """The output is H.264 (spec 2026-10-20 §3) and its level is at least the one its size and frame rate need."""
    codec = r.video.codec or "unknown"
    report.checks["h264"] = r.video.codec == "h264"
    report.details["h264"] = f"redacted video codec {codec}"
    if r.video.codec != "h264":
        report.checks["h264_level"] = False
        report.details["h264_level"] = f"not checked: the redacted video is {codec}, not H.264"
        return
    if not (r.video.width and r.video.height and r.fps > 0):
        report.checks["h264_level"] = False
        report.details["h264_level"] = f"unknown size or frame rate ({_size(r.video)}, {r.fps:g} fps)"
        return
    need = required_h264_level(r.video.width, r.video.height, r.fps)
    have = "unknown" if r.video.level is None else _level_text(r.video.level)
    wanted = "beyond 6.2" if need is None else _level_text(need)
    report.checks["h264_level"] = need is not None and r.video.level is not None and r.video.level >= need
    report.details["h264_level"] = f"level {have}; {_size(r.video)} at {r.fps:.2f} fps needs {wanted}"


def check_integrity(source: Path, redacted: Path) -> IntegrityReport:
    """Source and redacted must agree, and the output must be H.264 at a sufficient level; a source that cannot be
    probed raises ValueError, a broken output fails."""
    s = probe(source)
    try:
        r = probe(redacted)
    except ValueError as exc:
        return _unreadable(str(exc))
    report = _compare(s, r)
    _check_h264(report, r)
    return report


def _compare(s: MediaInfo, r: MediaInfo) -> IntegrityReport:
    """The checks of the redacted video `r` against its source `s`: decoding, frames, size, duration and audio."""
    frame = 1.0 / s.fps if s.fps > 0 else 1.0 / 30
```

（`_compare` 的其余部分就是原来 `check_integrity` 从 `report = IntegrityReport(...)` 起的函数体，一字不改，包括其中几处提前 `return report`。）

把 `py/vc_eval/report.py` 里的：

```python
INTEGRITY_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync")
```

换成：

```python
INTEGRITY_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync", "h264", "h264_level")
```

把 `py/vc_eval/report.py` 里的：

```python
        "| 段 | fps | 可解码 | 帧数 | 分辨率 | 时长 | 音频 | 音画同步 | 未通过项 |", "|---|---|---|---|---|---|---|---|---|",
```

换成：

```python
        "| 段 | fps | 可解码 | 帧数 | 分辨率 | 时长 | 音频 | 音画同步 | H.264 | Level | 未通过项 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
```

把 `py/vc_eval/cli.py` 里的：

```python
    i = sub.add_parser("integrity", help="frame count, duration, audio and A/V sync of the output")
```

换成：

```python
    i = sub.add_parser("integrity", help="frame count, duration, audio, A/V sync, H.264 and its level of the output")
```

- [ ] **Step 4: 跑测试**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_integrity.py -p no:cacheprovider -q
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_run.py -p no:cacheprovider -q
cd ..
```

预期：`42 passed`；`113 passed`（假 `vc_redact` 拷贝的是 libx264 编的 160×120 30 fps 原片，Level 1.1 正好够）。

- [ ] **Step 5: 文档**

把 `py/README.md` 里的：

```markdown
### vc_eval 的测试
```

换成：

```markdown
`integrity` 的检查项依次是 `decodable`、`frame_count`、`resolution`、`duration`、`audio_kept`、`av_sync`，以及（设计 2026-10-20 §3）`h264`（输出是 H.264）和 `h264_level`（ffprobe 报的 Level 不低于输出的分辨率与帧率所需，按 H.264 Table A-1：4K60 要 5.2，1080p 59.94 fps 要 4.2，1080p 30 fps 要 4.0；Level 读不到也算未通过）。`run` 的完整性表相应多 `H.264`、`Level` 两列。

### vc_eval 的测试
```

把 `docs/USAGE.md` 里的：

```markdown
3. **完整性**（需要 Python 环境）：帧数、时长、音频、音画同步。
```

换成：

```markdown
3. **完整性**（需要 Python 环境）：帧数、时长、音频、音画同步，以及输出是 H.264、Level 够它的分辨率和帧率（4K60 要 5.2）。
```

- [ ] **Step 6: 提交**

```bash
git add py/vc_eval/integrity.py py/vc_eval/report.py py/vc_eval/cli.py py/tests/test_eval_integrity.py \
    py/tests/test_eval_run.py py/README.md docs/USAGE.md
git commit -m "feat: check that the redacted video is H.264 at a sufficient level"
```

### Task 7: 马赛克网格纯函数（`mosaic_grid.h`），CPU 马赛克改用它

**Files:**
- Create: `cpp/src/mask/mosaic_grid.h`
- Modify: `cpp/src/mask/mosaic.h`（`PixelRect` 搬进 `mosaic_grid.h`、`applyMosaic` 注释）
- Modify: `cpp/src/mask/mosaic.cpp`（删 `ceilShift` / `clipped` / `snapped`，`pixelateGrid` 改按 `grid::CellGrid` 走，`applyMosaic` 用 `grid::maskCells`）
- Create: `cpp/tests/mosaic_grid_test.cpp`
- Modify: `cpp/tests/CMakeLists.txt`
- Modify: `cpp/README.md`（"马赛克"一条、第五组过滤命令）

当前的马赛克（提交 0f2aa85 起）有两种框：`snap` 为 true（人脸）时框边向外扩到块大小的整数倍；`snap` 为 false（车牌）时框边不动，只打框内，但仍按画面网格分块，被框边切开的块取框内那部分的均值。纯函数要同时描述这两种：块从画面网格上数（`col0 = area.x1 / cellW`），每块的像素范围是网格块与区域的交集。

**Interfaces:**
- Consumes: `VC_HOST_DEVICE`（Task 1 的 `common/host_device.h`）；`MosaicRect{rect, block, snap}`（现有）。
- Produces（GPU 计划 2b 的 CUDA 马赛克核函数使用，全部 `VC_HOST_DEVICE`、不用标准库）:
  - `struct PixelRect { int x1 = 0; int y1 = 0; int x2 = 0; int y2 = 0; bool empty() const; };`（命名空间 `vc`，仍可经 `mask/mosaic.h` 拿到）
  - `grid::clampInt(int, int, int)`、`grid::minInt(int, int)`、`grid::maxInt(int, int)`、`grid::ceilShift(int value, int shift)`
  - `PixelRect grid::clipRect(const PixelRect&, int width, int height);`
  - `PixelRect grid::snapRect(const PixelRect&, int block, int width, int height);`
  - `struct grid::CellGrid { PixelRect area{}; int cellW = 1; int cellH = 1; int col0 = 0; int row0 = 0; int cols = 0; int rows = 0; int cells() const; };`
  - `CellGrid grid::cellGrid(const PixelRect& area, int cellW, int cellH);`、`PixelRect grid::cellRect(const CellGrid&, int col, int row);`
  - `uint8_t grid::roundedMean(uint64_t sum, uint64_t count);`
  - `struct grid::MaskCells { CellGrid luma{}; CellGrid chroma{}; };`
  - `MaskCells grid::maskCells(const PixelRect& rect, int block, bool snap, int width, int height, int log2ChromaW, int log2ChromaH);`
  - GPU 版的用法：一个 CUDA 线程块处理一个 `cellRect`，求和后用 `roundedMean`，与 `pixelateGrid` 的结果逐字节相同。

- [ ] **Step 1: 写失败的测试**

新建 `cpp/tests/mosaic_grid_test.cpp`：

```cpp
#include "mask/mosaic_grid.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mask/mosaic.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

bool sameRect(const PixelRect& a, const PixelRect& b) { return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2; }

std::string text(const PixelRect& r) {
  return "[" + std::to_string(r.x1) + ", " + std::to_string(r.y1) + ")-(" + std::to_string(r.x2) + ", " +
         std::to_string(r.y2) + ")";
}

TEST(MosaicGrid, ClipsToThePlane) {
  EXPECT_TRUE(sameRect(grid::clipRect({-5, -1, 700, 400}, 640, 360), PixelRect{0, 0, 640, 360}));
  EXPECT_TRUE(sameRect(grid::clipRect({10, 20, 30, 40}, 640, 360), PixelRect{10, 20, 30, 40}));
  EXPECT_TRUE(grid::clipRect({700, 10, 800, 20}, 640, 360).empty());
}

// The grid is anchored at the frame's top-left corner: edges move out to multiples of the block, then the frame
// clips them.
TEST(MosaicGrid, SnapsToTheFramesGrid) {
  EXPECT_TRUE(sameRect(grid::snapRect({85, 88, 215, 192}, 24, 640, 360), PixelRect{72, 72, 216, 192}));
  EXPECT_TRUE(sameRect(grid::snapRect({96, 96, 120, 120}, 24, 640, 360), PixelRect{96, 96, 120, 120}));
  EXPECT_TRUE(sameRect(grid::snapRect({630, 350, 640, 360}, 16, 640, 360), PixelRect{624, 336, 640, 360}));
  EXPECT_TRUE(sameRect(grid::snapRect({0, 0, 1, 1}, 256, 640, 360), PixelRect{0, 0, 256, 256}));
}

TEST(MosaicGrid, CeilShiftRoundsUp) {
  EXPECT_EQ(grid::ceilShift(0, 1), 0);
  EXPECT_EQ(grid::ceilShift(5, 1), 3);
  EXPECT_EQ(grid::ceilShift(6, 1), 3);
  EXPECT_EQ(grid::ceilShift(5, 2), 2);
  EXPECT_EQ(grid::ceilShift(7, 0), 7);
}

// A snapped 101 x 50 area in 16 px cells: 7 columns (the last 5 px wide, cut by the plane's edge) and 4 rows.
TEST(MosaicGrid, CellsOfASnappedAreaStartOnItsEdges) {
  const grid::CellGrid g = grid::cellGrid({32, 16, 133, 66}, 16, 16);
  EXPECT_EQ(g.col0, 2);
  EXPECT_EQ(g.row0, 1);
  EXPECT_EQ(g.cols, 7);
  EXPECT_EQ(g.rows, 4);
  EXPECT_EQ(g.cells(), 28);
  EXPECT_TRUE(sameRect(grid::cellRect(g, 0, 0), PixelRect{32, 16, 48, 32}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 6, 3), PixelRect{128, 64, 133, 66}));
  EXPECT_EQ(grid::cellGrid({10, 10, 10, 20}, 8, 8).cells(), 0);
}

// An area off the grid (an unsnapped plate) touches the frame grid's cells and keeps only their parts inside it:
// [10, 50) x [10, 40) in 16 px cells is 4 columns ([10, 16) ... [48, 50)) and 3 rows ([10, 16) ... [32, 40)).
TEST(MosaicGrid, CellsOfAnUnsnappedAreaAreCutByItsEdges) {
  const grid::CellGrid g = grid::cellGrid({10, 10, 50, 40}, 16, 16);
  EXPECT_EQ(g.col0, 0);
  EXPECT_EQ(g.row0, 0);
  EXPECT_EQ(g.cols, 4);
  EXPECT_EQ(g.rows, 3);
  EXPECT_TRUE(sameRect(grid::cellRect(g, 0, 0), PixelRect{10, 10, 16, 16}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 1, 1), PixelRect{16, 16, 32, 32}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 3, 2), PixelRect{48, 32, 50, 40}));
}

TEST(MosaicGrid, MeanRoundsHalfUp) {
  EXPECT_EQ(grid::roundedMean(5, 2), 3);  // 2.5
  EXPECT_EQ(grid::roundedMean(4, 3), 1);  // 1.33
  EXPECT_EQ(grid::roundedMean(5, 3), 2);  // 1.67
  EXPECT_EQ(grid::roundedMean(255 * 256, 256), 255);
  EXPECT_EQ(grid::roundedMean(0, 7), 0);
}

// 4:2:0 chroma: half the area (rounded out) in half-size cells; 4:2:2 halves only the width; 4:4:4 keeps both; an
// unsnapped rect keeps its own edges.
TEST(MosaicGrid, MaskCellsPerChromaLayoutAndSnap) {
  const PixelRect rect{85, 88, 215, 192};
  const grid::MaskCells c420 = grid::maskCells(rect, 24, true, 640, 360, 1, 1);
  EXPECT_TRUE(sameRect(c420.luma.area, PixelRect{72, 72, 216, 192}));
  EXPECT_EQ(c420.luma.cellW, 24);
  EXPECT_TRUE(sameRect(c420.chroma.area, PixelRect{36, 36, 108, 96}));
  EXPECT_EQ(c420.chroma.cellW, 12);
  EXPECT_EQ(c420.chroma.cellH, 12);
  EXPECT_EQ(c420.chroma.cells(), c420.luma.cells());
  const grid::MaskCells c422 = grid::maskCells(rect, 24, true, 640, 360, 1, 0);
  EXPECT_TRUE(sameRect(c422.chroma.area, PixelRect{36, 72, 108, 192}));
  EXPECT_EQ(c422.chroma.cellW, 12);
  EXPECT_EQ(c422.chroma.cellH, 24);
  const grid::MaskCells c444 = grid::maskCells(rect, 24, true, 640, 360, 0, 0);
  EXPECT_TRUE(sameRect(c444.chroma.area, c444.luma.area));
  // The unsnapped NV12 case of Mosaic.AnUnsnappedNv12RectIsPixelatedOnlyInsideItself: chroma [5, 26) x [4, 21).
  const grid::MaskCells plate = grid::maskCells({11, 9, 51, 41}, 16, false, 96, 64, 1, 1);
  EXPECT_TRUE(sameRect(plate.luma.area, PixelRect{11, 9, 51, 41}));
  EXPECT_TRUE(sameRect(plate.chroma.area, PixelRect{5, 4, 26, 21}));
  // An odd-sized 4:2:0 frame: the chroma plane is ceil(width / 2) wide, and so is the area at the right edge.
  const grid::MaskCells odd = grid::maskCells({620, 0, 641, 16}, 16, true, 641, 361, 1, 1);
  EXPECT_TRUE(sameRect(odd.luma.area, PixelRect{608, 0, 641, 16}));
  EXPECT_TRUE(sameRect(odd.chroma.area, PixelRect{304, 0, 321, 8}));
  for (const bool snap : {true, false}) {  // outside the frame: nothing
    EXPECT_EQ(grid::maskCells({700, 10, 800, 20}, 16, snap, 640, 360, 1, 1).luma.cells(), 0);
    EXPECT_EQ(grid::maskCells({700, 10, 800, 20}, 16, snap, 640, 360, 1, 1).chroma.cells(), 0);
  }
}

// Every pixel of the area lies in exactly one cell and no cell is empty, for random masks, blocks and snap flags.
TEST(MosaicGrid, CellsPartitionTheArea) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> coord(-40, 700);
  for (int trial = 0; trial < 200; ++trial) {
    const int block = kMosaicBlockLadder[static_cast<size_t>(trial) % kMosaicBlockLadder.size()];
    const bool snap = trial % 2 == 0;
    const PixelRect rect{coord(rng), coord(rng), coord(rng), coord(rng)};
    const grid::CellGrid g = grid::maskCells(rect, block, snap, 641, 361, 1, 1).luma;
    std::vector<int> hits(641 * 361, 0);
    for (int row = 0; row < g.rows; ++row) {
      for (int col = 0; col < g.cols; ++col) {
        const PixelRect cell = grid::cellRect(g, col, row);
        ASSERT_FALSE(cell.empty()) << text(rect) << " block " << block << " snap " << snap;
        for (int y = cell.y1; y < cell.y2; ++y) {
          for (int x = cell.x1; x < cell.x2; ++x) ++hits[static_cast<size_t>(y) * 641 + static_cast<size_t>(x)];
        }
      }
    }
    for (int y = 0; y < 361; ++y) {
      for (int x = 0; x < 641; ++x) {
        const bool inside = x >= g.area.x1 && x < g.area.x2 && y >= g.area.y1 && y < g.area.y2;
        ASSERT_EQ(hits[static_cast<size_t>(y) * 641 + static_cast<size_t>(x)], inside ? 1 : 0)
            << text(rect) << " block " << block << " snap " << snap << " at " << x << "," << y;
      }
    }
  }
}

// The mosaic as it was before it used mosaic_grid.h (commit 0f2aa85), kept here as the reference: applyMosaic must
// produce the same bytes.
void legacyPixelate(uint8_t* data, int stride, const PixelRect& area, int cellW, int cellH, int step, int offset) {
  for (int gy = area.y1 / cellH * cellH; gy < area.y2; gy += cellH) {
    const int by = std::max(gy, area.y1);
    const int ey = std::min(gy + cellH, area.y2);
    for (int gx = area.x1 / cellW * cellW; gx < area.x2; gx += cellW) {
      const int bx = std::max(gx, area.x1);
      const int ex = std::min(gx + cellW, area.x2);
      uint64_t sum = 0;
      for (int y = by; y < ey; ++y) {
        const uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) sum += row[x * step];
      }
      const uint64_t count = static_cast<uint64_t>(ey - by) * static_cast<uint64_t>(ex - bx);
      const auto mean = static_cast<uint8_t>((sum + count / 2) / count);
      for (int y = by; y < ey; ++y) {
        uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) row[x * step] = mean;
      }
    }
  }
}

void legacyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects) {
  const auto format = static_cast<AVPixelFormat>(frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const bool nv12 = format == AV_PIX_FMT_NV12;
  const int sw = desc->log2_chroma_w;
  const int sh = desc->log2_chroma_h;
  const auto ceilShift = [](int value, int shift) { return -((-value) >> shift); };
  const auto clipped = [&](const PixelRect& r) {
    return PixelRect{std::clamp(r.x1, 0, frame.width), std::clamp(r.y1, 0, frame.height), std::clamp(r.x2, 0, frame.width),
                     std::clamp(r.y2, 0, frame.height)};
  };
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const PixelRect r = clipped(m.rect);
    if (r.empty()) continue;
    const int b = m.block;
    const auto up = [b](int v) { return static_cast<int>((static_cast<int64_t>(v) + b - 1) / b * b); };
    const PixelRect area = m.snap ? clipped({r.x1 / b * b, r.y1 / b * b, up(r.x2), up(r.y2)}) : r;
    legacyPixelate(frame.data[0], frame.linesize[0], area, b, b, 1, 0);
    const PixelRect c{area.x1 >> sw, area.y1 >> sh, ceilShift(area.x2, sw), ceilShift(area.y2, sh)};
    if (nv12) {
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 2, 0);
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 2, 1);
    } else {
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 1, 0);
      legacyPixelate(frame.data[2], frame.linesize[2], c, b >> sw, b >> sh, 1, 0);
    }
  }
}

// Every byte of every plane of two frames of the same format and size.
bool sameBytes(const AVFrame& a, const AVFrame& b) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(a.format));
  const bool nv12 = a.format == AV_PIX_FMT_NV12;
  for (int plane = 0; plane < (nv12 ? 2 : 3); ++plane) {
    const int cw = (a.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
    const int w = plane == 0 ? a.width : (nv12 ? 2 * cw : cw);
    const int h = plane == 0 ? a.height : (a.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
    for (int y = 0; y < h; ++y) {
      if (!std::equal(a.data[plane] + static_cast<ptrdiff_t>(y) * a.linesize[plane],
                      a.data[plane] + static_cast<ptrdiff_t>(y) * a.linesize[plane] + w,
                      b.data[plane] + static_cast<ptrdiff_t>(y) * b.linesize[plane])) {
        return false;
      }
    }
  }
  return true;
}

// spec 2026-10-20 §4: the CPU mosaic on the shared grid is byte for byte what it was, for every layout the reader
// emits, odd frame sizes, snapped and unsnapped rects, overlapping rects of mixed blocks included.
TEST(MosaicGrid, ApplyMosaicMatchesThePreviousImplementationByteForByte) {
  std::mt19937 rng(2026);
  for (const AVPixelFormat format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
                                     AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUVJ420P}) {
    for (const auto& [width, height] : {std::pair{640, 360}, std::pair{641, 361}, std::pair{1920, 1080}}) {
      std::uniform_int_distribution<int> x(-50, width + 50);
      std::uniform_int_distribution<int> y(-50, height + 50);
      std::uniform_int_distribution<size_t> step(0, kMosaicBlockLadder.size() - 1);
      for (int trial = 0; trial < 6; ++trial) {
        std::vector<MosaicRect> rects;
        for (int k = 0; k < 4; ++k) {
          rects.push_back(MosaicRect{{x(rng), y(rng), x(rng), y(rng)}, kMosaicBlockLadder[step(rng)], k % 2 == 0});
        }
        rects.push_back(MosaicRect{{x(rng), y(rng), x(rng) + 300, y(rng) + 200}, 16, trial % 2 == 0});
        av::FramePtr shared = test::makeNoiseFrame(width, height, format, static_cast<unsigned>(trial + width));
        av::FramePtr legacy = test::deepCopy(*shared);
        applyMosaic(*shared, rects);
        legacyMosaic(*legacy, rects);
        ASSERT_TRUE(sameBytes(*shared, *legacy))
            << av_get_pix_fmt_name(format) << " " << width << "x" << height << " trial " << trial;
      }
    }
  }
}

}  // namespace
}  // namespace vc
```

把 `cpp/tests/CMakeLists.txt` 里的：

```cmake
  mosaic_test.cpp
```

换成：

```cmake
  mosaic_grid_test.cpp
  mosaic_test.cpp
```

- [ ] **Step 2: 确认测试编译失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：FAIL，`mosaic_grid_test.cpp:1:10: fatal error: 'mask/mosaic_grid.h' file not found`。

- [ ] **Step 3: 纯函数，CPU 马赛克改用它**

新建 `cpp/src/mask/mosaic_grid.h`：

```cpp
#pragma once

// The mosaic grid as pure functions (spec 2026-10-20 §4): which cells a mask covers and the pixels of each, on the
// luma and on a subsampled chroma plane. applyMosaic (mosaic.cpp) uses them on the CPU; the CUDA mosaic (stage 2b)
// calls the same functions in its kernels, so both pixelate exactly the same cells. Plain C++ without the standard
// library (device code cannot call it), every function VC_HOST_DEVICE.

#include <cstdint>

#include "common/host_device.h"

namespace vc {

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  VC_HOST_DEVICE bool empty() const { return x2 <= x1 || y2 <= y1; }
};

namespace grid {

VC_HOST_DEVICE inline int clampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

VC_HOST_DEVICE inline int minInt(int a, int b) { return a < b ? a : b; }

VC_HOST_DEVICE inline int maxInt(int a, int b) { return a > b ? a : b; }

// ceil(value / 2^shift) for any int (an arithmetic right shift rounds down).
VC_HOST_DEVICE inline int ceilShift(int value, int shift) { return -((-value) >> shift); }

// `r` clipped to a width x height plane.
VC_HOST_DEVICE inline PixelRect clipRect(const PixelRect& r, int width, int height) {
  return {clampInt(r.x1, 0, width), clampInt(r.y1, 0, height), clampInt(r.x2, 0, width), clampInt(r.y2, 0, height)};
}

// `r` (inside the frame) with its edges moved out to multiples of `block`, clipped to the frame: the grid is
// anchored at the frame's top-left corner, not at the mask.
VC_HOST_DEVICE inline PixelRect snapRect(const PixelRect& r, int block, int width, int height) {
  const auto down = [block](int v) { return v / block * block; };
  const auto up = [block](int v) { return static_cast<int>((static_cast<int64_t>(v) + block - 1) / block * block); };
  return clipRect(PixelRect{down(r.x1), down(r.y1), up(r.x2), up(r.y2)}, width, height);
}

// The cells of one plane that `area` (plane pixels, inside the plane) touches, on the cellW x cellH grid anchored
// at the plane's origin: cols x rows cells from grid cell (col0, row0). A cell the area's edges cut keeps only its
// part inside the area (a snapped area has no such cells except at the plane's right and bottom edges).
struct CellGrid {
  PixelRect area{};
  int cellW = 1;
  int cellH = 1;
  int col0 = 0;
  int row0 = 0;
  int cols = 0;
  int rows = 0;

  VC_HOST_DEVICE int cells() const { return cols * rows; }
};

VC_HOST_DEVICE inline CellGrid cellGrid(const PixelRect& area, int cellW, int cellH) {
  CellGrid g;
  g.area = area;
  g.cellW = cellW;
  g.cellH = cellH;
  if (!area.empty()) {
    g.col0 = area.x1 / cellW;
    g.row0 = area.y1 / cellH;
    g.cols = (area.x2 - 1) / cellW - g.col0 + 1;
    g.rows = (area.y2 - 1) / cellH - g.row0 + 1;
  }
  return g;
}

// The pixels of cell (col, row) of `g` inside its area, 0 <= col < cols, 0 <= row < rows.
VC_HOST_DEVICE inline PixelRect cellRect(const CellGrid& g, int col, int row) {
  const int gx = (g.col0 + col) * g.cellW;
  const int gy = (g.row0 + row) * g.cellH;
  return {maxInt(gx, g.area.x1), maxInt(gy, g.area.y1), minInt(gx + g.cellW, g.area.x2),
          minInt(gy + g.cellH, g.area.y2)};
}

// The mean of `count` samples summing to `sum`, rounded half up.
VC_HOST_DEVICE inline uint8_t roundedMean(uint64_t sum, uint64_t count) {
  return static_cast<uint8_t>((sum + count / 2) / count);
}

// The luma and chroma cells one mask covers: `rect` (frame pixels) clipped to the frame, snapped out to the grid
// of `block` when `snap` (vc_redact's faces; plates keep their edges), its chroma area the luma area shifted by the
// format's chroma subsampling (log2, rounded outwards) with (block >> log2) cells. Both grids are empty when the
// clipped rect is.
struct MaskCells {
  CellGrid luma{};
  CellGrid chroma{};
};

VC_HOST_DEVICE inline MaskCells maskCells(const PixelRect& rect, int block, bool snap, int width, int height,
                                          int log2ChromaW, int log2ChromaH) {
  MaskCells m;
  const PixelRect r = clipRect(rect, width, height);
  if (r.empty()) return m;
  const PixelRect area = snap ? snapRect(r, block, width, height) : r;
  m.luma = cellGrid(area, block, block);
  const PixelRect chroma{area.x1 >> log2ChromaW, area.y1 >> log2ChromaH, ceilShift(area.x2, log2ChromaW),
                         ceilShift(area.y2, log2ChromaH)};
  m.chroma = cellGrid(chroma, block >> log2ChromaW, block >> log2ChromaH);
  return m;
}

}  // namespace grid
}  // namespace vc
```

把 `cpp/src/mask/mosaic.h` 里的：

```cpp
#include <unordered_map>
#include <vector>

struct AVFrame;
```

换成：

```cpp
#include <unordered_map>
#include <vector>

#include "mask/mosaic_grid.h"  // PixelRect and the grid shared with the CUDA mosaic

struct AVFrame;
```

把 `cpp/src/mask/mosaic.h` 里的：

```cpp
// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

// Smallest pixel rectangle covering the box,
```

换成：

```cpp
// Smallest pixel rectangle covering the box,
```

把 `cpp/src/mask/mosaic.h` 里的：

```cpp
// Pixelates each rect on the grid of its block size b anchored at the frame's top-left corner (spec §4):
```

换成：

```cpp
// Pixelates each rect on the grid of its block size b anchored at the frame's top-left corner (spec §4; the cells
// come from grid::maskCells, mosaic_grid.h):
```

把 `cpp/src/mask/mosaic.cpp` 里的：

```cpp
int ceilShift(int value, int shift) { return -((-value) >> shift); }

PixelRect clipped(const PixelRect& r, int width, int height) {
  return {std::clamp(r.x1, 0, width), std::clamp(r.y1, 0, height), std::clamp(r.x2, 0, width),
          std::clamp(r.y2, 0, height)};
}

// `r` (inside the frame) with its edges moved out to multiples of `block`, clipped to the frame.
PixelRect snapped(const PixelRect& r, int block, int width, int height) {
  const auto down = [block](int v) { return v / block * block; };
  const auto up = [block](int v) {
    return static_cast<int>((static_cast<int64_t>(v) + block - 1) / block * block);
  };
  return clipped({down(r.x1), down(r.y1), up(r.x2), up(r.y2)}, width, height);
}

// Sets the part inside `area` (plane pixels) of every cell of the cellW x cellH grid anchored at the plane's
// origin to that part's rounded mean. `step` is the byte distance between neighbouring samples, `offset` the
// first sample's byte (NV12 chroma: 2, 0 / 1).
void pixelateGrid(uint8_t* data, int stride, const PixelRect& area, int cellW, int cellH, int step, int offset) {
  for (int gy = area.y1 / cellH * cellH; gy < area.y2; gy += cellH) {
    const int by = std::max(gy, area.y1);
    const int ey = std::min(gy + cellH, area.y2);
    for (int gx = area.x1 / cellW * cellW; gx < area.x2; gx += cellW) {
      const int bx = std::max(gx, area.x1);
      const int ex = std::min(gx + cellW, area.x2);
      uint64_t sum = 0;
      for (int y = by; y < ey; ++y) {
        const uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) sum += row[x * step];
      }
      const uint64_t count = static_cast<uint64_t>(ey - by) * static_cast<uint64_t>(ex - bx);
      const auto mean = static_cast<uint8_t>((sum + count / 2) / count);
      for (int y = by; y < ey; ++y) {
        uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) row[x * step] = mean;
      }
    }
  }
}
```

换成：

```cpp
// Sets every cell of `cells` (plane pixels, each cell's part inside the area) to its rounded mean. `step` is the
// byte distance between neighbouring samples, `offset` the first sample's byte (NV12 chroma: 2, 0 / 1).
void pixelateGrid(uint8_t* data, int stride, const grid::CellGrid& cells, int step, int offset) {
  for (int row = 0; row < cells.rows; ++row) {
    for (int col = 0; col < cells.cols; ++col) {
      const PixelRect cell = grid::cellRect(cells, col, row);
      uint64_t sum = 0;
      for (int y = cell.y1; y < cell.y2; ++y) {
        const uint8_t* line = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = cell.x1; x < cell.x2; ++x) sum += line[x * step];
      }
      const uint64_t count = static_cast<uint64_t>(cell.y2 - cell.y1) * static_cast<uint64_t>(cell.x2 - cell.x1);
      const uint8_t mean = grid::roundedMean(sum, count);
      for (int y = cell.y1; y < cell.y2; ++y) {
        uint8_t* line = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = cell.x1; x < cell.x2; ++x) line[x * step] = mean;
      }
    }
  }
}
```

把 `cpp/src/mask/mosaic.cpp` 里的：

```cpp
  const int sw = desc->log2_chroma_w;
  const int sh = desc->log2_chroma_h;
  // Largest blocks first: where masks overlap, the finer mosaic drawn last is what shows.
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const PixelRect r = clipped(m.rect, frame.width, frame.height);
    if (r.empty()) continue;
    const PixelRect area = m.snap ? snapped(r, m.block, frame.width, frame.height) : r;
    pixelateGrid(frame.data[0], frame.linesize[0], area, m.block, m.block, 1, 0);
    const PixelRect c{area.x1 >> sw, area.y1 >> sh, ceilShift(area.x2, sw), ceilShift(area.y2, sh)};
    const int cw = m.block >> sw;
    const int ch = m.block >> sh;
    if (nv12) {
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 2, 0);
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 2, 1);
    } else {
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 1, 0);
      pixelateGrid(frame.data[2], frame.linesize[2], c, cw, ch, 1, 0);
    }
  }
```

换成：

```cpp
  // Largest blocks first: where masks overlap, the finer mosaic drawn last is what shows.
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const grid::MaskCells cells = grid::maskCells(m.rect, m.block, m.snap, frame.width, frame.height,
                                                  desc->log2_chroma_w, desc->log2_chroma_h);
    if (cells.luma.cells() == 0) continue;
    pixelateGrid(frame.data[0], frame.linesize[0], cells.luma, 1, 0);
    if (nv12) {
      pixelateGrid(frame.data[1], frame.linesize[1], cells.chroma, 2, 0);
      pixelateGrid(frame.data[1], frame.linesize[1], cells.chroma, 2, 1);
    } else {
      pixelateGrid(frame.data[1], frame.linesize[1], cells.chroma, 1, 0);
      pixelateGrid(frame.data[2], frame.linesize[2], cells.chroma, 1, 0);
    }
  }
```

（取值、取整、溢出行为都与原来相同：`snapRect` 的 `up` 照旧先在 64 位里算再转 `int`，`clipRect` 照旧在转换之后裁剪；`cellGrid` 的 `col0` 就是原来 `gx` 的起点 `area.x1 / cellW`，`cols` 就是原来 `gx` 循环的次数，`cellRect` 就是原来的 `[max(gx, x1), min(gx + cellW, x2))`。`MosaicGrid.ApplyMosaicMatchesThePreviousImplementationByteForByte` 把 0f2aa85 的实现原样留在测试里，在随机的 `snap` 取值上逐字节比对。）

- [ ] **Step 4: 跑测试（原有 `Mosaic.*` 一个不改，全部照过）**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 2>&1 | grep -c 'warning:'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:MosaicGrid.*:BoundedQueue.*:RedactConfig.*:Records.*'
"${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
git diff --stat -- cpp/tests/mosaic_test.cpp
```

预期：`0`；第五组 `[  PASSED  ] 140 tests.`（`Mosaic` 21 个、`MosaicGrid` 9 个）；第六组 `[  PASSED  ] 83 tests.`（含 `RedactJob.PixelatesAPlateOnlyInsideItsMask`）；最后一条没有输出（`mosaic_test.cpp` 没改）。

- [ ] **Step 5: 文档**

把 `cpp/README.md` 里的：

```markdown
向外对齐 16 px 网格会让屏上面积翻到约 2.3 倍）。
```

换成：

```markdown
向外对齐 16 px 网格会让屏上面积翻到约 2.3 倍）。“框 → 要打哪些块、每块的像素范围”（含车牌那种被框边切开的块）写成 `cpp/src/mask/mosaic_grid.h` 里的纯函数（`grid::maskCells`、`grid::cellRect`、`grid::roundedMean`，标了 `VC_HOST_DEVICE`），GPU 计划里的 CUDA 马赛克用同一套，所以两边逐像素相同。
```

把 `cpp/README.md` 里的：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:BoundedQueue.*:RedactConfig.*:Records.*'
```

换成：

```markdown
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:MosaicGrid.*:BoundedQueue.*:RedactConfig.*:Records.*'
```

- [ ] **Step 6: 提交**

```bash
git add cpp/src/mask/mosaic_grid.h cpp/src/mask/mosaic.h cpp/src/mask/mosaic.cpp cpp/tests/mosaic_grid_test.cpp \
    cpp/tests/CMakeLists.txt cpp/README.md
git commit -m "refactor: compute the mosaic grid with pure functions the GPU can share"
```

### Task 8: `scripts/redact.sh` 的 `VC_JOBS` / `VC_GPUS` 与 `batch.json`

**Files:**
- Modify（整个换掉）: `scripts/redact.sh`
- Create: `scripts/batch_json.py`
- Create: `py/tests/test_redact_script.py`
- Modify: `docs/USAGE.md`（"批量"一节）

**Interfaces:**
- Consumes: `stats.json` 的 `frames`、`elapsed_s`、`detector`、`encoder`、`pipeline`（Task 3 加了 `pipeline`）。
- Produces（Task 9 的 Dockerfile 拷 `scripts/batch_json.py`；GPU 计划 3b 的 `bench_gpu.sh` 读 `batch.json`）:
  - 环境变量 `VC_JOBS`（1–64；默认 `VC_GPUS` 非空时 4，否则 1）、`VC_GPUS`（逗号分隔的 GPU 序号或 UUID）、`VC_PYTHON`（默认 `python3`）。
  - `<out>/batch.json`：`{"jobs", "gpus", "processes_at_once", "wall_s", "succeeded", "failed", "frames", "throughput_fps", "videos": [{"video", "output_dir", "gpu", "exit_status", "frames", "elapsed_s", "fps", "detector", "encoder", "pipeline"}]}`，`videos` 按输入顺序，失败的统计字段为 null。
  - `scripts/batch_json.py --out <dir> --results <tsv> --started <epoch 秒> --jobs <n> [--gpus <text>]`；`<tsv>` 每行 `<输入序号>\t<退出码>\t<gpu>\t<视频路径>`。

脚本要能在 macOS 的 `/bin/bash` 3.2 上跑（没有 `wait -n`、没有关联数组）：每个视频在后台子 shell 里跑，结束时把退出码写进 `<临时目录>/<序号>.status`（先写 `.partial` 再改名），主循环每 0.2 秒看一次哪个结束了；GPU 槽位按"从上次分到的下一块卡开始，找第一块还有空位的"轮流分配，所有视频一样长时就是按顺序 0、1、0、1……

- [ ] **Step 1: 写失败的测试**

新建 `py/tests/test_redact_script.py`：

```python
"""scripts/redact.sh with a fake vc_redact: parallel jobs, GPUs in turn, failures and batch.json (spec 2026-10-20 §5)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "redact.sh"

# Stands in for vc_redact: notes how many runs are going on and which GPU it was given, can wait until
# FAKE_BARRIER runs go on at once (exit 9 after 5 s), fails for stems starting with "bad" and otherwise writes
# a stats.json with 120 frames in 2 s.
FAKE = r"""#!/usr/bin/env bash
set -euo pipefail
while [ $# -gt 0 ]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
stem="$(basename "$input")"; stem="${stem%.*}"
state="$FAKE_STATE"
mkdir -p "$state/running"
: > "$state/running/$stem"
ls "$state/running" | wc -l | tr -d ' ' >> "$state/concurrency"
echo "$stem ${CUDA_VISIBLE_DEVICES-unset}" >> "$state/gpus"
echo "fake vc_redact on $stem"
if [ -n "${FAKE_BARRIER:-}" ]; then
  for _ in $(seq 50); do
    if [ "$(ls "$state/running" | wc -l | tr -d ' ')" -ge "$FAKE_BARRIER" ]; then break; fi
    sleep 0.1
  done
  if [ "$(ls "$state/running" | wc -l | tr -d ' ')" -lt "$FAKE_BARRIER" ]; then echo "barrier timeout" >&2; exit 9; fi
fi
sleep "${FAKE_SLEEP:-0}"
rm -f "$state/running/$stem"
case "$stem" in bad*) echo "fake failure on $stem" >&2; exit 1 ;; esac
mkdir -p "$out/$stem"
printf '{"frames": 120, "elapsed_s": 2.0, "detector": "student/CPU", "encoder": "libx264", "pipeline": "cpu-frames"}\n' \
  > "$out/$stem/stats.json"
"""


@pytest.fixture
def batch(tmp_path):
    """Runs redact.sh on named videos (created as empty files) with the fake; returns (completed process, out dir, state dir)."""
    fake = tmp_path / "vc_redact"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_text("x")
    (model / "model.json").write_text("{}")
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "out"

    def run(names, **env_extra):
        videos = []
        for name in names:
            path = tmp_path / "in" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            videos.append(str(path))
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "VC_GPUS", "VC_JOBS")}
        env.update(VC_REDACT=str(fake), VC_MODEL=str(model), VC_FACE_MODEL="", VC_SANDBOX="0",
                   VC_PYTHON=sys.executable, FAKE_STATE=str(state), **env_extra)
        done = subprocess.run([str(SCRIPT), str(out), *videos], env=env, capture_output=True, text=True, timeout=60)
        return done, out, state

    return run


def _batch_json(out):
    return json.loads((out / "batch.json").read_text(encoding="utf-8"))


def _gpu_of(state):
    return dict(line.split() for line in (state / "gpus").read_text().splitlines())


def _concurrency(state):
    return [int(n) for n in (state / "concurrency").read_text().split()]


def test_runs_every_video_in_turn_and_writes_batch_json(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4"])
    assert done.returncode == 0, done.stderr
    assert max(_concurrency(state)) == 1  # VC_JOBS defaults to 1 without VC_GPUS
    assert _gpu_of(state) == {"a": "unset", "b": "unset", "c": "unset"}
    b = _batch_json(out)
    assert [Path(v["video"]).name for v in b["videos"]] == ["a.mp4", "b.mp4", "c.mp4"]
    for v in b["videos"]:
        assert v["exit_status"] == 0 and v["gpu"] is None
        assert v["frames"] == 120 and v["elapsed_s"] == 2.0 and v["fps"] == 60.0
        assert (v["detector"], v["encoder"], v["pipeline"]) == ("student/CPU", "libx264", "cpu-frames")
        assert v["output_dir"] == str(out / Path(v["video"]).stem)
    assert (b["jobs"], b["gpus"], b["processes_at_once"]) == (1, [], 1)
    assert (b["succeeded"], b["failed"], b["frames"]) == (3, 0, 360)
    assert b["wall_s"] > 0 and b["throughput_fps"] == pytest.approx(360 / b["wall_s"])
    assert "done: 3 of 3 redacted" in done.stdout and "=== throughput:" in done.stdout


def test_a_failed_video_is_reported_and_the_rest_still_run(batch):
    done, out, _ = batch(["a.mp4", "bad.mp4", "c.mp4"])
    assert done.returncode == 1
    assert "=== FAILED (exit 1): " in done.stderr and "bad.mp4" in done.stderr
    assert "failed: " in done.stderr
    b = _batch_json(out)
    by_name = {Path(v["video"]).name: v for v in b["videos"]}
    assert by_name["bad.mp4"]["exit_status"] == 1
    assert by_name["bad.mp4"]["frames"] is None and by_name["bad.mp4"]["fps"] is None
    assert by_name["c.mp4"]["exit_status"] == 0
    assert (b["succeeded"], b["failed"], b["frames"]) == (2, 1, 240)


def test_vc_jobs_runs_that_many_at_once(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4"], VC_JOBS="3", FAKE_BARRIER="3")
    assert done.returncode == 0, done.stdout + done.stderr  # the barrier needs all three running together
    assert max(_concurrency(state)) == 3
    assert _batch_json(out)["processes_at_once"] == 3
    for stem in ("a", "b", "c"):  # each run's output is printed once, in one piece
        assert done.stdout.count(f"fake vc_redact on {stem}\n") == 1


def test_vc_gpus_hands_the_videos_out_in_turn(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4", "d.mp4"], VC_GPUS="0,1", VC_JOBS="2", FAKE_BARRIER="4")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _gpu_of(state) == {"a": "0", "b": "1", "c": "0", "d": "1"}
    b = _batch_json(out)
    assert [v["gpu"] for v in b["videos"]] == ["0", "1", "0", "1"]
    assert (b["jobs"], b["gpus"], b["processes_at_once"]) == (2, ["0", "1"], 4)
    assert "[gpu 1] " in done.stdout


def test_no_gpu_runs_more_than_vc_jobs_at_once(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4", "d.mp4"], VC_GPUS="0,1", VC_JOBS="1", FAKE_SLEEP="0.3")
    assert done.returncode == 0, done.stderr
    assert max(_concurrency(state)) <= 2
    gpus = list(_gpu_of(state).values())
    assert sorted(gpus) == ["0", "0", "1", "1"]


def test_vc_gpus_defaults_to_four_jobs_per_gpu(batch):
    done, out, _ = batch(["a.mp4"], VC_GPUS="0,1")
    assert done.returncode == 0, done.stderr
    assert (_batch_json(out)["jobs"], _batch_json(out)["processes_at_once"]) == (4, 8)


def test_a_second_video_with_the_same_stem_is_refused_without_running(batch, tmp_path):
    done, out, state = batch(["x/a.mp4", "y/a.mp4"])
    assert done.returncode == 1
    assert "=== FAILED (exit 3): " in done.stderr and "y/a.mp4" in done.stderr
    assert len((state / "gpus").read_text().splitlines()) == 1  # vc_redact ran once
    assert [v["exit_status"] for v in _batch_json(out)["videos"]] == [0, 3]


def test_cuda_visible_devices_is_left_alone_without_vc_gpus(batch):
    done, _, state = batch(["a.mp4"], CUDA_VISIBLE_DEVICES="7")
    assert done.returncode == 0, done.stderr
    assert _gpu_of(state) == {"a": "7"}


@pytest.mark.parametrize("env, message", [
    ({"VC_JOBS": "0"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_JOBS": "65"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_JOBS": "two"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_GPUS": "0,,1"}, "VC_GPUS must list GPU indices or UUIDs"),
    ({"VC_GPUS": "0;1"}, "VC_GPUS must list GPU indices or UUIDs"),
])
def test_bad_vc_jobs_and_vc_gpus_are_refused_before_running(batch, env, message):
    done, out, state = batch(["a.mp4"], **env)
    assert done.returncode == 2 and message in done.stderr
    assert not (state / "gpus").exists() and not (out / "batch.json").exists()
```

- [ ] **Step 2: 确认测试失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_redact_script.py -p no:cacheprovider -q
cd ..
```

预期：FAIL，`12 failed, 1 passed`（现在的脚本不写 `batch.json`、不认 `VC_JOBS` / `VC_GPUS`；`VC_JOBS=3` 那条要等假 `vc_redact` 的 5 秒同步超时，整个文件约半分钟）。只有 `test_cuda_visible_devices_is_left_alone_without_vc_gpus` 通过。

- [ ] **Step 3: 实现**

新建 `scripts/batch_json.py`：

```python
"""Writes <out>/batch.json for scripts/redact.sh (spec 2026-10-20 §5): every video's result and the batch throughput.

Standard library only: it runs wherever redact.sh does (the Docker image, a bare Linux machine, macOS).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Copied from each successful run's stats.json as they are (None when the field is missing).
STATS_FIELDS = ("detector", "encoder", "pipeline")


def read_results(path: Path) -> list[tuple[int, int, str | None, str]]:
    """redact.sh's lines "<position>\\t<exit status>\\t<gpu>\\t<video>", in input order."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        position, status, gpu, video = line.split("\t", 3)
        rows.append((int(position), int(status), gpu or None, video))
    return sorted(rows)


def video_entry(out: Path, status: int, gpu: str | None, video: str) -> dict:
    """One video's entry; the stats of a failed run stay None (exit 3 leaves an earlier run's stats.json there)."""
    output_dir = out / Path(video).stem
    entry = {
        "video": video, "output_dir": str(output_dir), "gpu": gpu, "exit_status": status,
        "frames": None, "elapsed_s": None, "fps": None, **{name: None for name in STATS_FIELDS},
    }
    if status != 0:
        return entry
    try:
        stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        entry["error"] = f"cannot read stats.json: {exc}"
        return entry
    frames, elapsed = stats.get("frames"), stats.get("elapsed_s")
    entry["frames"], entry["elapsed_s"] = frames, elapsed
    entry["fps"] = frames / elapsed if frames and elapsed else None
    entry.update({name: stats.get(name) for name in STATS_FIELDS})
    return entry


def batch_summary(entries: list[dict], wall_s: float, jobs: int, gpus: list[str]) -> dict:
    """The whole batch: frames of the successful runs over the batch's wall time is its throughput."""
    ok = [e for e in entries if e["exit_status"] == 0]
    frames = sum(e["frames"] or 0 for e in ok)
    return {
        "jobs": jobs,
        "gpus": gpus,
        "processes_at_once": jobs * max(1, len(gpus)),
        "wall_s": wall_s,
        "succeeded": len(ok),
        "failed": len(entries) - len(ok),
        "frames": frames,
        "throughput_fps": frames / wall_s if wall_s > 0 else None,
        "videos": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--started", type=float, required=True, help="epoch seconds when the batch started")
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--gpus", default="")
    args = parser.parse_args(argv)
    gpus = [g for g in args.gpus.split(",") if g]
    entries = [video_entry(args.out, status, gpu, video) for _, status, gpu, video in read_results(args.results)]
    summary = batch_summary(entries, time.time() - args.started, args.jobs, gpus)
    partial = args.out / "batch.json.partial"
    partial.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, args.out / "batch.json")
    fps = summary["throughput_fps"]
    print(f"=== throughput: {fps or 0:.1f} fps ({summary['frames']} frames from {summary['succeeded']} video(s) "
          f"in {summary['wall_s']:.1f} s, {summary['processes_at_once']} at once); {args.out / 'batch.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

把 `scripts/redact.sh` 整个换成：

```bash
#!/usr/bin/env bash
# Batch redaction with the validated models: scripts/redact.sh <output dir> <video or directory>...
#
# Every video goes to <output dir>/<video stem>/ (see `vc_redact --help`). A directory argument takes the
# .mp4 / .mov / .mkv files directly inside it. A failed video is reported and the rest still run; the exit
# code is 1 when any video failed. A video whose stem an earlier one already has would share its output
# directory: it is refused (exit 3) without running. <output dir>/batch.json lists every video's result
# and the batch throughput (spec 2026-10-20 §5).
#
# Environment overrides:
#   VC_REDACT       vc_redact binary              (default build/cpp-release/vc_redact)
#   VC_MODEL        student model directory       (default data/models/r3)
#   VC_FACE_MODEL   YuNet face model directory    (default data/models/yunet; empty = student only)
#   VC_DETECT_FPS   detections per second, 5 to 60 (default: detect_fps of configs/redact.json, 10)
#   VC_REDACT_ARGS  extra vc_redact options, e.g. "--hwaccel videotoolbox" or "--overwrite"
#   VC_GPUS         GPUs to use, e.g. "0,1": videos go to them in turn, each vc_redact runs with
#                   CUDA_VISIBLE_DEVICES set to its GPU (default: none; CUDA_VISIBLE_DEVICES is left alone)
#   VC_JOBS         vc_redact processes at once per GPU of VC_GPUS, or in all without VC_GPUS
#                   (default 4 per GPU, 1 without VC_GPUS); 1 to 64
#   VC_PYTHON       python3 that writes batch.json (default python3)
#   VC_SANDBOX      1 (default on macOS): run inside sandbox-exec with outbound network denied
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VC_REDACT:-$ROOT/build/cpp-release/vc_redact}"
MODEL="${VC_MODEL:-$ROOT/data/models/r3}"
FACE="${VC_FACE_MODEL-$ROOT/data/models/yunet}"
DETECT_FPS="${VC_DETECT_FPS:-}"
GPUS_TEXT="${VC_GPUS:-}"
PYTHON="${VC_PYTHON:-python3}"
read -r -a EXTRA <<< "${VC_REDACT_ARGS:-}"
SANDBOX_PROFILE='(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'
MAX_JOBS=64
POLL_SECONDS=0.2

usage() { echo "usage: $0 <output dir> <video or directory>..." >&2; exit 2; }
fail() { echo "redact.sh: $*" >&2; exit 2; }

[ $# -ge 2 ] || usage
out="$1"; shift
[ -x "$BIN" ] || fail "no vc_redact at $BIN (build it: cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && cmake --build build/cpp-release -j 4)"
for dir in "$MODEL" ${FACE:+"$FACE"}; do
  [ -s "$dir/model.onnx" ] && [ -s "$dir/model.json" ] || fail "$dir has no model.onnx + model.json"
done

gpus=()
if [ -n "$GPUS_TEXT" ]; then
  IFS=',' read -r -a gpus <<< "$GPUS_TEXT"
  for gpu in ${gpus[@]+"${gpus[@]}"}; do
    [[ "$gpu" =~ ^[A-Za-z0-9-]+$ ]] || fail "VC_GPUS must list GPU indices or UUIDs separated by commas, got '$GPUS_TEXT'"
  done
fi
n_gpus=${#gpus[@]}
if [ "$n_gpus" -gt 0 ]; then default_jobs=4; else default_jobs=1; fi
JOBS="${VC_JOBS:-$default_jobs}"
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || [ "$JOBS" -gt "$MAX_JOBS" ]; then
  fail "VC_JOBS must be an integer in [1, $MAX_JOBS], got '$JOBS'"
fi
if [ "$n_gpus" -gt 0 ]; then total_slots=$((JOBS * n_gpus)); else total_slots=$JOBS; fi
command -v "$PYTHON" > /dev/null || fail "$PYTHON is needed to write batch.json (set VC_PYTHON)"

videos=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r -d '' f; do videos+=("$f"); done < <(
      find "$arg" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) -print0 | sort -z)
  elif [ -f "$arg" ]; then
    videos+=("$arg")
  else
    fail "$arg is neither a file nor a directory"
  fi
done
[ ${#videos[@]} -gt 0 ] || fail "no .mp4 / .mov / .mkv files found"
for video in "${videos[@]}"; do
  case "$video" in
    *$'\t'* | *$'\n'*) fail "a video path holds a tab or a newline: $video" ;;
  esac
done

runner=()
if [ "${VC_SANDBOX:-1}" = 1 ] && command -v sandbox-exec > /dev/null; then
  runner=(sandbox-exec -p "$SANDBOX_PROFILE")
fi

mkdir -p "$out"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/redact.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
started="$("$PYTHON" -c 'import time; print(f"{time.time():.3f}")')"
: > "$tmp/results.tsv"

failed=()
pids=()      # by job number, while the job runs
job_video=()
job_gpu=()   # the GPU (as listed in VC_GPUS), empty without VC_GPUS
job_slot=()  # index into gpus, -1 without VC_GPUS
on_gpu=()    # jobs running per index into gpus
for ((g = 0; g < n_gpus; g++)); do on_gpu[g]=0; done
running=0
next_gpu=0

# One line per video for batch.json: input position, exit status, GPU, path.
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$tmp/results.tsv"; }

# Starts vc_redact on `video` in the background; with more than one at once its output goes to a log
# that is printed when it ends, so that the runs do not interleave.
launch() {
  local job=$1 video=$2 gpu=$3
  local args=(--input "$video" --out "$out" --model "$MODEL")
  if [ -n "$FACE" ]; then args+=(--face-model "$FACE"); fi
  if [ -n "$DETECT_FPS" ]; then args+=(--detect-fps "$DETECT_FPS"); fi
  echo "=== $(date +%T) ${gpu:+[gpu $gpu] }$video"
  (
    if [ -n "$gpu" ]; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
    if [ "$total_slots" -gt 1 ]; then exec > "$tmp/$job.log" 2>&1; fi
    if ${runner[@]+"${runner[@]}"} "$BIN" "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"}; then status=0; else status=$?; fi
    echo "$status" > "$tmp/$job.status.partial"
    mv "$tmp/$job.status.partial" "$tmp/$job.status"
  ) &
  pids[job]=$!
}

# Collects one finished job: prints its log, records its status and frees its slot.
finish() {
  local job=$1 status
  status="$(cat "$tmp/$job.status")"
  wait "${pids[job]}" || true
  unset 'pids[job]'
  if [ "$total_slots" -gt 1 ]; then cat "$tmp/$job.log"; fi
  if [ "$status" != 0 ]; then
    echo "=== FAILED (exit $status): ${job_video[job]}" >&2
    failed+=("${job_video[job]}")
  fi
  record "$job" "$status" "${job_gpu[job]}" "${job_video[job]}"
  running=$((running - 1))
  if [ "${job_slot[job]}" -ge 0 ]; then on_gpu[job_slot[job]]=$((on_gpu[job_slot[job]] - 1)); fi
}

# Waits until a running job ends and collects it.
reap_one() {
  local job
  while :; do
    for job in "${!pids[@]}"; do
      if [ -f "$tmp/$job.status" ]; then
        finish "$job"
        return
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

# The index of the next GPU in turn with a free slot (there is one whenever fewer than total_slots run).
free_gpu() {
  local k g
  for ((k = 0; k < n_gpus; k++)); do
    g=$(((next_gpu + k) % n_gpus))
    if [ "${on_gpu[g]}" -lt "$JOBS" ]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

seen=()
for ((job = 0; job < ${#videos[@]}; job++)); do
  video="${videos[job]}"
  stem="$(basename "$video")"
  stem="${stem%.*}"
  duplicate=0
  for s in ${seen[@]+"${seen[@]}"}; do
    if [ "$s" = "$stem" ]; then duplicate=1; fi
  done
  if [ "$duplicate" = 1 ]; then
    echo "=== FAILED (exit 3): $video (an earlier video already writes $out/$stem)" >&2
    failed+=("$video")
    record "$job" 3 "" "$video"
    continue
  fi
  seen+=("$stem")
  while [ "$running" -ge "$total_slots" ]; do reap_one; done
  gpu=""
  slot=-1
  if [ "$n_gpus" -gt 0 ]; then
    slot="$(free_gpu)"
    gpu="${gpus[slot]}"
    on_gpu[slot]=$((on_gpu[slot] + 1))
    next_gpu=$(((slot + 1) % n_gpus))
  fi
  job_video[job]="$video"
  job_gpu[job]="$gpu"
  job_slot[job]=$slot
  launch "$job" "$video" "$gpu"
  running=$((running + 1))
done
while [ "$running" -gt 0 ]; do reap_one; done

echo "=== $(date +%T) done: $(( ${#videos[@]} - ${#failed[@]} )) of ${#videos[@]} redacted into $out"
if ! "$PYTHON" -I "$ROOT/scripts/batch_json.py" --out "$out" --results "$tmp/results.tsv" --started "$started" \
    --jobs "$JOBS" --gpus "$GPUS_TEXT"; then
  echo "redact.sh: could not write $out/batch.json" >&2
  exit 1
fi
if [ ${#failed[@]} -gt 0 ]; then
  printf '    failed: %s\n' "${failed[@]}" >&2
  exit 1
fi
```

- [ ] **Step 4: 跑测试与 shellcheck**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_redact_script.py -p no:cacheprovider -q
cd ..
command -v shellcheck > /dev/null && shellcheck scripts/redact.sh && echo "shellcheck clean"
```

预期：`13 passed`（约 5 秒）；`shellcheck clean`。

- [ ] **Step 5: 文档**

把 `docs/USAGE.md` 里的：

```markdown
- 同名视频（不同目录下的 `A.mp4`）会写到同一个输出目录。第二个会被拒绝（退出码 3），请分开输出目录。
- 环境变量：
  - `VC_MODEL`、`VC_FACE_MODEL`：换模型；`VC_FACE_MODEL=` 设为空表示只用学生模型；
  - `VC_DETECT_FPS=15`：检测频率（5–60，默认用 `configs/redact.json` 的 10）；
  - `VC_SANDBOX=0`：不套沙箱。
```

换成：

```markdown
- 同名视频（不同目录下的 `A.mp4`）会写到同一个输出目录。第二个不运行、直接记为失败（退出码 3），请分开输出目录。
- 环境变量：
  - `VC_MODEL`、`VC_FACE_MODEL`：换模型；`VC_FACE_MODEL=` 设为空表示只用学生模型；
  - `VC_DETECT_FPS=15`：检测频率（5–60，默认用 `configs/redact.json` 的 10）；
  - `VC_JOBS=2`：同时跑几个 `vc_redact`（1–64，默认 1；设了 `VC_GPUS` 时是每块卡几个，默认 4）。同时不止一个时，每个视频的输出在它跑完后整段打印，不会交错；
  - `VC_GPUS=0,1`：用哪几块 NVIDIA 卡（Linux），视频按顺序轮流分到各卡，每个 `vc_redact` 只看到自己那块（`CUDA_VISIBLE_DEVICES`）；不设时 `CUDA_VISIBLE_DEVICES` 保持原样；
  - `VC_PYTHON`：写 `batch.json` 用的 python3（默认 `python3`，只用标准库）；
  - `VC_SANDBOX=0`：不套沙箱。
- 跑完后输出目录下有 `batch.json`（每次覆盖上一次的）：每个视频的 `exit_status`、`gpu`，成功的还有 `frames`、`elapsed_s`、`fps`、`detector`、`encoder`、`pipeline`（取自它的 `stats.json`，失败的为 null）；整批的 `wall_s`（墙钟秒数）、`frames`、`throughput_fps`（成功视频的总帧数 ÷ 整批墙钟时间）、`processes_at_once`。最后一行 `=== throughput: …` 打印同样的合计。
```

- [ ] **Step 6: 提交**

```bash
git add scripts/redact.sh scripts/batch_json.py py/tests/test_redact_script.py docs/USAGE.md
git commit -m "feat: run redact.sh videos in parallel, hand them to GPUs in turn and write batch.json"
```

### Task 9: Linux 交付的静态部分（`setup_linux.sh`、Dockerfile）

**Files:**
- Create: `scripts/setup_linux.sh`
- Create: `docker/Dockerfile`, `docker/build.sh`, `docker/entrypoint.sh`, `.dockerignore`
- Create: `py/tests/test_linux_deploy.py`
- Modify: `docs/USAGE.md`（§1 一句、§6 一条、新增 §7 Linux 一节）、`cpp/README.md`（"依赖"后加 Linux 一段）

**Interfaces:**
- Consumes: `scripts/redact.sh`、`scripts/batch_json.py`（Task 8，镜像里拷它们）；`cpp/cmake/ThirdParty.cmake` 的 Linux 包名 `onnxruntime-linux-x64-gpu-1.30.0`（Task 1）；`VC_WITH_CUDA`（Task 1）。
- Produces（GPU 计划 1c 在目标机上填写并使用）:
  - `scripts/setup_linux.sh [--docker-build | --check-pins]`；环境变量 `VC_ALLOW_DOWNLOAD=1`（必需）、`VC_CACHE`、`VC_FFMPEG_PREFIX`、`VC_THIRD_PARTY_DIR`、`VC_BUILD_DIR`、`TENSORRT_VERSION`（开发机必需）。
  - `PINS` 表：`名字|URL|sha256`；`--check-pins` 每行打印 `ok <名字> <URL> <sha256>` 或 `missing <名字> <URL> (…)`，有空项时退出码 1。目前空着待填的：`onnxruntime-linux-x64-gpu-1.30.0.tgz`、`nv-codec-headers-12.2.72.0.tar.gz`、`cmake-3.31.6-linux-x86_64.tar.gz`、`cuda-keyring-ubuntu2204.deb`、`cuda-keyring-ubuntu2404.deb`。
  - `docker/build.sh [tag]`；必需 `VC_ALLOW_DOWNLOAD=1`、`CUDA_DEVEL_DIGEST`、`CUDA_RUNTIME_DIGEST`（`sha256:` 加 64 位十六进制）、`TENSORRT_VERSION`，且 `--check-pins` 必须通过。
  - 镜像入口 `docker run … vc-redact <输入> <输出>`（`docker/entrypoint.sh` 调 `redact.sh <输出> <输入>`；`VC_REDACT_SH` 可换成别的脚本，供测试用）。

**不编造校验和。** 本机能确认的只有两项：`json.hpp` 的 sha256 与 `cpp/README.md` 里 macOS 安装用的同一个；FFmpeg 8.1.2 源码包的 sha256 取自本机 Homebrew 的 ffmpeg 8.1.2 公式（`/opt/homebrew/Cellar/ffmpeg/8.1.2/.brew/ffmpeg.rb` 里的 `sha256`，即 macOS 版所用的同一个包）。其余各项留空，脚本在空值时拒绝下载；基础镜像的 digest 不写进仓库，`docker/build.sh` 要求构建时传入。

- [ ] **Step 1: 写失败的测试**

新建 `py/tests/test_linux_deploy.py`：

```python
"""Static checks of the Linux deployment files (spec 2026-10-20 §2): docker/, scripts/setup_linux.sh.

Nothing here builds an image or downloads anything: the scripts are only run up to their refusals, and the
Dockerfile is read as text. shellcheck and hadolint run when they are installed and are skipped otherwise.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "scripts" / "setup_linux.sh"
BUILD = REPO / "docker" / "build.sh"
ENTRYPOINT = REPO / "docker" / "entrypoint.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile"
SHELL_SCRIPTS = [SETUP, BUILD, ENTRYPOINT, REPO / "scripts" / "redact.sh"]


def _run(cmd, **env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("VC_ALLOW_DOWNLOAD", "CUDA_DEVEL_DIGEST", "CUDA_RUNTIME_DIGEST",
                                                          "TENSORRT_VERSION")}
    env.update(env_extra)
    return subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=30)


def _pins():
    """(status, name, url, sha256 or None) of every pin, as setup_linux.sh --check-pins lists them."""
    done = _run([SETUP, "--check-pins"])
    assert done.returncode in (0, 1), done.stderr
    pins = []
    for line in done.stdout.splitlines():
        status, name, url, rest = line.split(" ", 3)
        pins.append((status, name, url, rest if status == "ok" else None))
    return pins


def test_setup_refuses_to_download_without_permission(tmp_path):
    done = _run([SETUP], VC_CACHE=str(tmp_path))
    assert done.returncode == 2
    assert "set VC_ALLOW_DOWNLOAD=1" in done.stderr
    assert not any(tmp_path.iterdir())  # nothing was created, let alone downloaded


@pytest.mark.skipif(os.uname().sysname == "Linux", reason="checks the refusal off Linux")
def test_setup_refuses_anything_but_ubuntu_on_linux(tmp_path):
    done = _run([SETUP], VC_ALLOW_DOWNLOAD="1", VC_CACHE=str(tmp_path))
    assert done.returncode == 2 and "Linux only" in done.stderr
    assert not any(tmp_path.iterdir())


def test_setup_rejects_unknown_options():
    done = _run([SETUP, "--bogus"])
    assert done.returncode == 2 and "usage:" in done.stderr


def test_check_pins_fails_while_a_checksum_is_missing():
    pins = _pins()
    missing = [name for status, name, _, _ in pins if status == "missing"]
    assert _run([SETUP, "--check-pins"]).returncode == (1 if missing else 0)
    assert {status for status, _, _, _ in pins} <= {"ok", "missing"}


def test_pins_cover_the_downloads_over_https_with_real_checksums():
    pins = _pins()
    names = {name for _, name, _, _ in pins}
    assert {"onnxruntime-linux-x64-gpu-1.30.0.tgz", "json.hpp", "ffmpeg-8.1.2.tar.xz",
            "nv-codec-headers-12.2.72.0.tar.gz", "cmake-3.31.6-linux-x86_64.tar.gz"} <= names
    for status, name, url, sha in pins:
        assert url.startswith("https://"), name
        if status == "ok":
            assert re.fullmatch(r"[0-9a-f]{64}", sha), f"{name}: {sha!r} is not a sha256"
    known = {name: sha for status, name, _, sha in pins if status == "ok"}
    # Published checksums: cpp/README.md's for json.hpp, Homebrew's ffmpeg 8.1.2 formula for the FFmpeg tarball.
    assert known["json.hpp"] == "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
    assert known["ffmpeg-8.1.2.tar.xz"] == "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"


@pytest.mark.parametrize("env, message", [
    ({}, "set VC_ALLOW_DOWNLOAD=1"),
    ({"VC_ALLOW_DOWNLOAD": "1"}, "CUDA_DEVEL_DIGEST must be the sha256:<64 hex digits> digest"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:abc"}, "CUDA_DEVEL_DIGEST must be the sha256"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:" + "a" * 64}, "CUDA_RUNTIME_DIGEST must be the sha256"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:" + "a" * 64, "CUDA_RUNTIME_DIGEST": "sha256:" + "b" * 64},
     "TENSORRT_VERSION must name"),
])
def test_image_build_refuses_without_pinned_digests(env, message):
    done = _run([BUILD], **env)
    assert done.returncode == 2 and message in done.stderr


def test_dockerfile_pins_both_base_images_by_digest():
    text = DOCKERFILE.read_text(encoding="utf-8")
    froms = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2
    assert froms[0].startswith("FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04@${CUDA_DEVEL_DIGEST} AS build")
    assert froms[1] == "FROM nvidia/cuda:${CUDA_TAG}-cudnn-runtime-ubuntu22.04@${CUDA_RUNTIME_DIGEST}"
    assert "ARG CUDA_DEVEL_DIGEST\n" in text and "ARG CUDA_RUNTIME_DIGEST\n" in text  # no default digest
    assert "latest" not in text and "sha256:" not in text  # nothing made up in the file itself


def test_dockerfile_copies_only_files_that_exist():
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            for source in line.split()[1:-1]:
                assert (REPO / source).exists(), source


def test_dockerignore_keeps_personal_data_and_builds_out_of_the_context():
    ignored = (REPO / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in ("data/", "build/", ".venv/", ".git/"):
        assert entry in ignored


def test_entrypoint_puts_the_input_first(tmp_path):
    fake = tmp_path / "redact.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    done = _run([ENTRYPOINT, "/in", "/out"], VC_REDACT_SH=str(fake))
    assert done.returncode == 0 and done.stdout.splitlines() == ["/out", "/in"]
    assert _run([ENTRYPOINT, "/in"], VC_REDACT_SH=str(fake)).returncode == 2


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_shell_scripts_pass_shellcheck():
    done = _run(["shellcheck", *SHELL_SCRIPTS])
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("hadolint") is None, reason="hadolint is not installed")
def test_dockerfile_passes_hadolint():
    done = _run(["hadolint", DOCKERFILE])
    assert done.returncode == 0, done.stdout + done.stderr
```

- [ ] **Step 2: 确认测试失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_linux_deploy.py -p no:cacheprovider -q -rs
cd ..
```

预期：FAIL，`15 failed, 1 skipped`（脚本和 Dockerfile 都还不存在；`hadolint is not installed` 那条跳过）。

- [ ] **Step 3: 安装脚本**

新建 `scripts/setup_linux.sh`（建好后 `chmod +x scripts/setup_linux.sh`）：

```bash
#!/usr/bin/env bash
# Sets up Ubuntu 22.04 / 24.04 with an NVIDIA GPU for vc_redact and builds it with VC_WITH_CUDA=ON
# (spec 2026-10-20 §2):
#
#   scripts/setup_linux.sh                 development machine: checks the driver (>= 550), installs the apt
#                                          packages (sudo), builds FFmpeg 8 and vc_redact with its tests
#   scripts/setup_linux.sh --docker-build  docker/Dockerfile's build stage: root, no GPU, apt done by the Dockerfile,
#                                          vc_redact without tests
#   scripts/setup_linux.sh --check-pins    lists the pinned downloads; exit 1 while a checksum is empty (no network)
#
# Everything but --check-pins downloads, so it refuses to run unless VC_ALLOW_DOWNLOAD=1. Each download is checked
# against its sha256 in PINS and kept in the cache all checkouts share ($VC_CACHE/downloads); a file there whose
# checksum matches is not downloaded again.
#
# Environment:
#   VC_ALLOW_DOWNLOAD=1  required for anything but --check-pins
#   VC_CACHE             cache root (default ~/.cache/video-compliance)
#   VC_FFMPEG_PREFIX     FFmpeg 8 install prefix (default $VC_CACHE/linux/ffmpeg-8.1.2)
#   VC_THIRD_PARTY_DIR   onnxruntime and nlohmann/json (default $VC_CACHE/third_party, as cpp/cmake/ThirdParty.cmake)
#   VC_BUILD_DIR         vc_redact's build directory (default build/cpp-linux)
#   TENSORRT_VERSION     apt version of the TensorRT 10 runtime libraries onnxruntime 1.30.0 needs, e.g.
#                        "10.x.y.z-1+cuda12.8" (development machine only; required there)
set -euo pipefail
# A failure inside $(...) must stop the script too (bash >= 4.4; macOS's bash 3.2 only ever reaches the refusals).
shopt -s inherit_errexit 2> /dev/null || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${VC_CACHE:-$HOME/.cache/video-compliance}"
DOWNLOADS="$CACHE/downloads"
FFMPEG_VERSION=8.1.2
NV_CODEC_VERSION=12.2.72.0   # Video Codec SDK 12.2: driver >= 550 (13.x would need 570)
CMAKE_VERSION=3.31.6         # Ubuntu 22.04's cmake 3.22 is older than the 3.24 cpp/CMakeLists.txt needs
CUDA_APT_VERSION=12-8        # the CUDA of docker/Dockerfile's nvidia/cuda:12.8.1 images
ORT_PACKAGE=onnxruntime-linux-x64-gpu-1.30.0
FFMPEG_PREFIX="${VC_FFMPEG_PREFIX:-$CACHE/linux/ffmpeg-$FFMPEG_VERSION}"
THIRD_PARTY="${VC_THIRD_PARTY_DIR:-$CACHE/third_party}"
BUILD_DIR="${VC_BUILD_DIR:-$ROOT/build/cpp-linux}"
MIN_DRIVER=550

# name|url|sha256 of every download. An empty sha256 has not been looked up yet: the target-machine plan fills it
# from the publisher's checksum and commits it, and until then the script refuses to download anything. Never put
# a checksum here that was not computed from (or published for) the real file.
PINS=(
  "$ORT_PACKAGE.tgz|https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/$ORT_PACKAGE.tgz|"
  "json.hpp|https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp|aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
  # The sha256 of Homebrew's ffmpeg 8.1.2 formula (the same tarball the macOS build comes from).
  "ffmpeg-$FFMPEG_VERSION.tar.xz|https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz|464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
  "nv-codec-headers-$NV_CODEC_VERSION.tar.gz|https://github.com/FFmpeg/nv-codec-headers/releases/download/n$NV_CODEC_VERSION/nv-codec-headers-$NV_CODEC_VERSION.tar.gz|"
  "cmake-$CMAKE_VERSION-linux-x86_64.tar.gz|https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.tar.gz|"
  "cuda-keyring-ubuntu2204.deb|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb|"
  "cuda-keyring-ubuntu2404.deb|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb|"
)

die() { echo "setup_linux.sh: $*" >&2; exit 2; }
usage() { echo "usage: $0 [--docker-build | --check-pins]" >&2; exit 2; }

# pin <name> <field>: field 2 is the url, 3 the sha256.
pin() {
  local entry name url sha
  for entry in "${PINS[@]}"; do
    IFS='|' read -r name url sha <<< "$entry"
    if [ "$name" = "$1" ]; then
      if [ "$2" = 2 ]; then echo "$url"; else echo "$sha"; fi
      return 0
    fi
  done
  die "no pin named $1"
}

# Lists every pin as "ok <name> <url> <sha256>" or "missing <name> <url> ..."; returns 1 when a checksum is missing
# or not 64 hex digits.
check_pins() {
  local entry name url sha bad=0
  for entry in "${PINS[@]}"; do
    IFS='|' read -r name url sha <<< "$entry"
    if [[ "$sha" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ok $name $url $sha"
    else
      echo "missing $name $url (no sha256 yet: fill it in PINS on the target machine)"
      bad=1
    fi
  done
  return "$bad"
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# fetch <name>: the path of the verified download in $DOWNLOADS, downloading it only when it is missing or wrong.
fetch() {
  local name=$1 url sha file partial
  url="$(pin "$name" 2)"
  sha="$(pin "$name" 3)"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || die "no sha256 pinned for $name (scripts/setup_linux.sh --check-pins)"
  file="$DOWNLOADS/$name"
  if [ -f "$file" ] && [ "$(sha256_of "$file")" = "$sha" ]; then
    echo "$file"
    return 0
  fi
  mkdir -p "$DOWNLOADS"
  partial="$(mktemp "$DOWNLOADS/.$name.XXXXXX")"
  echo "setup_linux.sh: downloading $url" >&2
  if ! curl -fL --retry 3 -o "$partial" "$url"; then
    rm -f "$partial"
    die "download failed: $url"
  fi
  if [ "$(sha256_of "$partial")" != "$sha" ]; then
    rm -f "$partial"
    die "sha256 of $name does not match the pin; the file was not kept"
  fi
  mv "$partial" "$file"
  echo "$file"
}

check_os() {
  [ "$(uname -s)" = Linux ] || die "Linux only (this is $(uname -s)); macOS builds as cpp/README.md describes"
  [ "$(uname -m)" = x86_64 ] || die "x86_64 only (the onnxruntime GPU package is built for it), this is $(uname -m)"
  [ -r /etc/os-release ] || die "cannot read /etc/os-release"
  # shellcheck source=/dev/null
  . /etc/os-release
  if [ "${ID:-}" != ubuntu ] || { [ "${VERSION_ID:-}" != 22.04 ] && [ "${VERSION_ID:-}" != 24.04 ]; }; then
    die "Ubuntu 22.04 or 24.04 only, this is ${PRETTY_NAME:-unknown}"
  fi
}

check_driver() {
  command -v nvidia-smi > /dev/null || die "nvidia-smi not found: install the NVIDIA driver >= $MIN_DRIVER first"
  local version major
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  major="${version%%.*}"
  if ! [[ "$major" =~ ^[0-9]+$ ]] || [ "$major" -lt "$MIN_DRIVER" ]; then
    die "NVIDIA driver '$version' is older than $MIN_DRIVER (NVENC / NVDEC of nv-codec-headers $NV_CODEC_VERSION need it)"
  fi
}

# The apt packages of the development machine: build tools, libx264, GoogleTest, and from NVIDIA's repository the
# CUDA toolkit, cuDNN 9 and the TensorRT runtime pinned to $TENSORRT_VERSION.
install_apt_packages() {
  [ -n "${TENSORRT_VERSION:-}" ] || die "TENSORRT_VERSION must name the TensorRT apt version onnxruntime 1.30.0 needs"
  local sudo=() repo keyring
  if [ "$(id -u)" != 0 ]; then sudo=(sudo); fi
  repo="ubuntu${VERSION_ID//./}"
  keyring="$(fetch "cuda-keyring-$repo.deb")"
  "${sudo[@]}" dpkg -i "$keyring"
  "${sudo[@]}" apt-get update
  local packages=(build-essential pkg-config nasm curl ca-certificates xz-utils python3 python3-venv
                  libx264-dev libgtest-dev "cuda-toolkit-$CUDA_APT_VERSION" libcudnn9-cuda-12
                  "libnvinfer10=$TENSORRT_VERSION" "libnvinfer-plugin10=$TENSORRT_VERSION"
                  "libnvonnxparsers10=$TENSORRT_VERSION")
  if [ "$VERSION_ID" = 24.04 ]; then packages+=(cmake); fi
  "${sudo[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
}

# The cmake to use: Ubuntu 24.04's, or on 22.04 Kitware's pinned binary unpacked into the cache.
cmake_binary() {
  if [ "${VERSION_ID:-}" = 24.04 ] && command -v cmake > /dev/null; then
    command -v cmake
    return 0
  fi
  local dir="$CACHE/linux/cmake-$CMAKE_VERSION-linux-x86_64" tarball
  if [ ! -x "$dir/bin/cmake" ]; then
    tarball="$(fetch "cmake-$CMAKE_VERSION-linux-x86_64.tar.gz")"
    mkdir -p "$CACHE/linux"
    tar -xzf "$tarball" -C "$CACHE/linux"
  fi
  echo "$dir/bin/cmake"
}

# onnxruntime's Linux GPU package and nlohmann/json, where cpp/cmake/ThirdParty.cmake looks for them.
unpack_third_party() {
  local tarball header
  mkdir -p "$THIRD_PARTY"
  if [ ! -f "$THIRD_PARTY/$ORT_PACKAGE/include/onnxruntime_cxx_api.h" ]; then
    tarball="$(fetch "$ORT_PACKAGE.tgz")"
    tar -xzf "$tarball" -C "$THIRD_PARTY"
  fi
  if [ ! -f "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann/json.hpp" ]; then
    header="$(fetch json.hpp)"
    mkdir -p "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann"
    cp "$header" "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann/json.hpp"
  fi
}

# FFmpeg 8 with NVENC / NVDEC (nv-codec-headers, MIT) and libx264 (GPL: the build is GPL), shared libraries
# that find each other through their RPATH.
build_ffmpeg() {
  if [ -x "$FFMPEG_PREFIX/bin/ffmpeg" ] && "$FFMPEG_PREFIX/bin/ffmpeg" -hide_banner -encoders 2> /dev/null | grep -q h264_nvenc; then
    echo "setup_linux.sh: FFmpeg $FFMPEG_VERSION is already in $FFMPEG_PREFIX"
    return 0
  fi
  local src headers ffmpeg
  src="$(mktemp -d)"
  headers="$(fetch "nv-codec-headers-$NV_CODEC_VERSION.tar.gz")"
  ffmpeg="$(fetch "ffmpeg-$FFMPEG_VERSION.tar.xz")"
  tar -xzf "$headers" -C "$src"
  make -C "$src/nv-codec-headers-$NV_CODEC_VERSION" install PREFIX="$FFMPEG_PREFIX"
  tar -xJf "$ffmpeg" -C "$src"
  (
    cd "$src/ffmpeg-$FFMPEG_VERSION"
    PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig" ./configure --prefix="$FFMPEG_PREFIX" \
      --enable-shared --disable-static --disable-doc --enable-gpl --enable-libx264 \
      --enable-ffnvcodec --enable-nvenc --enable-nvdec --enable-cuvid \
      --extra-ldflags="-Wl,-rpath,$FFMPEG_PREFIX/lib"
    make -j "$(nproc)"
    make install
  )
  rm -rf "$src"
}

# vc_redact with the CUDA code paths; tests ON on the development machine (filtered gtest groups, cpp/README.md).
build_vc_redact() {
  local tests=$1 cmake
  cmake="$(cmake_binary)"
  "$cmake" -S "$ROOT/cpp" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DVC_WITH_CUDA=ON -DVC_BUILD_TESTS="$tests" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH="$FFMPEG_PREFIX" \
    -DVC_THIRD_PARTY_DIR="$THIRD_PARTY"
  "$cmake" --build "$BUILD_DIR" -j "$(nproc)"
  echo "setup_linux.sh: built $BUILD_DIR/vc_redact"
}

mode=dev
case "${1:-}" in
  "") ;;
  --docker-build) mode=docker ;;
  --check-pins)
    check_pins
    exit $?
    ;;
  *) usage ;;
esac
[ $# -le 1 ] || usage

[ "${VC_ALLOW_DOWNLOAD:-}" = 1 ] || die "this downloads onnxruntime, FFmpeg and more (see PINS); set VC_ALLOW_DOWNLOAD=1 to allow it"
check_os
check_pins > /dev/null || die "some pinned downloads have no sha256 yet: run $0 --check-pins"
if [ "$mode" = dev ]; then
  check_driver
  install_apt_packages
fi
unpack_third_party
build_ffmpeg
if [ "$mode" = dev ]; then build_vc_redact ON; else build_vc_redact OFF; fi
```

- [ ] **Step 4: Docker 镜像**

新建 `docker/Dockerfile`：

```dockerfile
# syntax=docker/dockerfile:1
# vc_redact for Linux + NVIDIA (spec 2026-10-20 §2): a build stage on the CUDA devel image, a runtime stage on the
# CUDA runtime image with cuDNN and the TensorRT runtime. Build it with docker/build.sh, which passes the base image
# digests (looked up on the target machine) and refuses without them. Run:
#   docker run --gpus all -v <videos>:/in -v <output>:/out -v <models>:/models -v <cache>:/cache vc-redact /in /out
# Models (/models/r3, /models/yunet) and caches (/cache: TensorRT engines) stay outside the image.
ARG CUDA_TAG=12.8.1
ARG CUDA_DEVEL_DIGEST
ARG CUDA_RUNTIME_DIGEST

FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04@${CUDA_DEVEL_DIGEST} AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config nasm curl ca-certificates xz-utils \
      libx264-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cpp/ cpp/
COPY scripts/setup_linux.sh scripts/
ENV VC_CACHE=/opt/vc-cache VC_FFMPEG_PREFIX=/opt/ffmpeg VC_THIRD_PARTY_DIR=/opt/third_party VC_BUILD_DIR=/opt/vc/build
# Building the image is the consent to its downloads (all of them pinned by sha256 in setup_linux.sh).
RUN VC_ALLOW_DOWNLOAD=1 scripts/setup_linux.sh --docker-build

FROM nvidia/cuda:${CUDA_TAG}-cudnn-runtime-ubuntu22.04@${CUDA_RUNTIME_DIGEST}
ARG TENSORRT_VERSION
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ -z "${TENSORRT_VERSION}" ]; then echo "TENSORRT_VERSION is required (docker/build.sh)" >&2; exit 1; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends libx264-163 python3 python3-numpy \
      "libnvinfer10=${TENSORRT_VERSION}" "libnvinfer-plugin10=${TENSORRT_VERSION}" \
      "libnvonnxparsers10=${TENSORRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*
# The same paths as in the build stage: vc_redact and FFmpeg find their libraries through their RPATH.
COPY --from=build /opt/ffmpeg/ /opt/ffmpeg/
COPY --from=build /opt/third_party/onnxruntime-linux-x64-gpu-1.30.0/lib/ /opt/third_party/onnxruntime-linux-x64-gpu-1.30.0/lib/
COPY --from=build /opt/vc/build/vc_redact /opt/vc/bin/vc_redact
COPY scripts/redact.sh scripts/batch_json.py /opt/vc/scripts/
COPY configs/ /opt/vc/configs/
COPY py/vc_eval/ /opt/vc/py/vc_eval/
COPY py/vc_label/__init__.py py/vc_label/contracts.py /opt/vc/py/vc_label/
COPY docker/entrypoint.sh /opt/vc/entrypoint.sh
ENV VC_REDACT=/opt/vc/bin/vc_redact VC_MODEL=/models/r3 VC_FACE_MODEL=/models/yunet VC_CACHE=/cache VC_SANDBOX=0 \
    PATH=/opt/ffmpeg/bin:$PATH PYTHONPATH=/opt/vc/py
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/opt/vc/entrypoint.sh"]
```

新建 `docker/build.sh`（建好后 `chmod +x docker/build.sh`）：

```bash
#!/usr/bin/env bash
# Builds the vc-redact image from docker/Dockerfile (spec 2026-10-20 §2) with its base images pinned by digest:
#
#   VC_ALLOW_DOWNLOAD=1 CUDA_DEVEL_DIGEST=sha256:<64 hex> CUDA_RUNTIME_DIGEST=sha256:<64 hex> \
#     TENSORRT_VERSION=<apt version> docker/build.sh [tag]
#
# The digests are those of nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 and nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
# (`docker buildx imagetools inspect <image>` prints them) and TENSORRT_VERSION the apt version of the TensorRT 10
# runtime onnxruntime 1.30.0 needs; the target-machine plan looks them up and records them. Building downloads the
# base images and the pinned packages, so it also needs VC_ALLOW_DOWNLOAD=1. The default tag is vc-redact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tag="${1:-vc-redact}"
fail() { echo "docker/build.sh: $*" >&2; exit 2; }

[ $# -le 1 ] || fail "usage: $0 [tag]"
[ "${VC_ALLOW_DOWNLOAD:-}" = 1 ] || fail "building the image downloads base images and packages; set VC_ALLOW_DOWNLOAD=1 to allow it"
for name in CUDA_DEVEL_DIGEST CUDA_RUNTIME_DIGEST; do
  value="${!name:-}"
  [[ "$value" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "$name must be the sha256:<64 hex digits> digest of its base image, got '$value'"
done
[ -n "${TENSORRT_VERSION:-}" ] || fail "TENSORRT_VERSION must name the TensorRT apt version onnxruntime 1.30.0 needs"
"$ROOT/scripts/setup_linux.sh" --check-pins > /dev/null || fail "some pinned downloads have no sha256 yet: run scripts/setup_linux.sh --check-pins"
command -v docker > /dev/null || fail "docker not found"
exec docker build -f "$ROOT/docker/Dockerfile" \
  --build-arg "CUDA_DEVEL_DIGEST=$CUDA_DEVEL_DIGEST" --build-arg "CUDA_RUNTIME_DIGEST=$CUDA_RUNTIME_DIGEST" \
  --build-arg "TENSORRT_VERSION=$TENSORRT_VERSION" -t "$tag" "$ROOT"
```

新建 `docker/entrypoint.sh`（建好后 `chmod +x docker/entrypoint.sh`）：

```bash
#!/usr/bin/env bash
# The image's entry point: `docker run ... vc-redact <input> <output>` runs scripts/redact.sh <output> <input>
# (spec 2026-10-20 §2 puts the input first). VC_JOBS, VC_GPUS and the other redact.sh variables pass through -e.
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: docker run --gpus all -v <videos>:/in -v <output>:/out -v <models>:/models -v <cache>:/cache vc-redact /in /out" >&2
  exit 2
fi
exec "${VC_REDACT_SH:-/opt/vc/scripts/redact.sh}" "$2" "$1"
```

新建 `.dockerignore`：

```text
# The build context is the repository: keep out the videos, models and labels (personal data), builds and caches.
data/
build/
.venv/
.git/
.claude/
**/__pycache__/
```

- [ ] **Step 5: 跑测试与静态检查**

```bash
chmod +x scripts/setup_linux.sh docker/build.sh docker/entrypoint.sh
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_linux_deploy.py -p no:cacheprovider -q -rs
cd ..
"${SB[@]}" scripts/setup_linux.sh --check-pins; echo "check-pins exit $?"
command -v hadolint > /dev/null || echo "hadolint not installed: Dockerfile lint skipped (target-machine plan runs it)"
```

预期：`15 passed, 1 skipped`（跳过的是 hadolint）；`--check-pins` 打印 7 行，`json.hpp` 和 `ffmpeg-8.1.2.tar.xz` 两行是 `ok`，其余 5 行 `missing`，`check-pins exit 1`；最后一行提示 hadolint 没装。shellcheck 由 `test_shell_scripts_pass_shellcheck` 跑过（四个脚本零告警）。

- [ ] **Step 6: 文档**

把 `docs/USAGE.md` 里的：

```markdown
目前只支持 **macOS + Apple Silicon**：检测走 CoreML，编码走 VideoToolbox。Linux 或 NVIDIA 服务器的部署还没做，见文末"还没做的"。
```

换成：

```markdown
日常使用的是 **macOS + Apple Silicon**：检测走 CoreML，编码走 VideoToolbox。Linux + NVIDIA 的部署做了一部分，见 §7。
```

把 `docs/USAGE.md` 里的：

```markdown
- **Linux / NVIDIA 服务器部署：** 生产目标是 1 小时视频 ≤ 10 分钟。onnxruntime 换成 Linux + CUDA 版，编码换成 NVENC，都还没做。
```

换成：

```markdown
- **Linux / NVIDIA 服务器部署的 GPU 部分：** 生产目标是 1 小时素材 ≤ 10 分钟，见 §7 的"还要 GPU 机器的部分"。
```

把 `docs/USAGE.md` 里的：

````markdown
## 测完 4K 后请反馈
````

换成：

````markdown
## 7. Linux + NVIDIA（进行中）

设计见 `docs/superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md`。目标：一台 Linux + NVIDIA 机器上，合计 1 小时 4K60 素材 10 分钟内处理完（约 360 帧/秒，按批量吞吐算）。

**现在已有的（本机验证过的部分）：**

- 输出一律 H.264（两个平台一致），HEVC 原片码率乘 `output_bitrate_factor_hevc`，Level 按分辨率和帧率指定；`vc_eval integrity` 检查 H.264 与 Level。
- `--ep auto|tensorrt|cuda|coreml|cpu` 与 `--hwaccel cuda`；Linux 上默认 `--ep auto`（TensorRT → CUDA → CPU）和 `--hwaccel auto`（NVDEC，打不开退回软解）。**GPU 部分完成之前，TensorRT / CUDA 总是报不可用并退到 CPU**；NVDEC 与 NVENC 的代码已在，但还没在 NVIDIA 机器上跑过。
- CMake 开关 `-DVC_WITH_CUDA=ON`（需要 CUDA 12 的 nvcc）。
- `scripts/redact.sh` 的 `VC_JOBS`、`VC_GPUS` 与 `batch.json`（见 §2"批量"）。
- 裸机安装脚本 `scripts/setup_linux.sh`（Ubuntu 22.04 / 24.04，驱动 ≥ 550）和 Docker 镜像 `docker/Dockerfile`（用 `docker/build.sh` 构建）。**两者都还没在目标机上跑过**：有几项下载的 sha256 还空着（`scripts/setup_linux.sh --check-pins` 列出），基础镜像的 digest 和 TensorRT 的版本要在目标机上查实后填写；空着时脚本拒绝运行。

**裸机（开发机）：**

```bash
scripts/setup_linux.sh --check-pins     # 不联网：列出固定的下载和校验和
VC_ALLOW_DOWNLOAD=1 TENSORRT_VERSION=<apt 版本> scripts/setup_linux.sh
```

它会联网下载（onnxruntime GPU 包、FFmpeg 8.1.2 源码、nv-codec-headers 12.2.72.0，Ubuntu 22.04 上还有 CMake 3.31.6，以及 apt 包），不设 `VC_ALLOW_DOWNLOAD=1` 直接拒绝；下载放在所有 checkout 共享的 `~/.cache/video-compliance/downloads/`，已有且校验通过就不再下载。装好后可执行文件在 `build/cpp-linux/vc_redact`，FFmpeg 在 `~/.cache/video-compliance/linux/ffmpeg-8.1.2/`（带 NVENC / NVDEC 和 libx264；libx264 是 GPL，整份 FFmpeg 随之是 GPL，与 YOLO 的 AGPL 一起记入"生产许可待定"）。

**Docker（生产）：**

```bash
VC_ALLOW_DOWNLOAD=1 CUDA_DEVEL_DIGEST=sha256:<…> CUDA_RUNTIME_DIGEST=sha256:<…> TENSORRT_VERSION=<…> docker/build.sh
docker run --gpus all -v <素材>:/in -v <输出>:/out -v <模型>:/models -v <缓存>:/cache vc-redact /in /out
```

模型目录挂到 `/models`（里面是 `r3/` 和 `yunet/`），`/cache` 放 TensorRT 引擎缓存（`trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`）。`-e VC_JOBS=4 -e VC_GPUS=0,1` 照常传给 `redact.sh`。构建上下文排除 `data/`、`build/`（`.dockerignore`），视频和模型不会进镜像。

**调参（Linux）：**

| 项 | 默认 | 说明 |
|---|---|---|
| `VC_JOBS` | 每块卡 4 | 每块卡同时几个 `vc_redact`。消费级卡（RTX 4090）的驱动限制同时编码的会话数（约 8 路），每卡的 `VC_JOBS` 不要超过它 |
| `detect_fps` | 10 | 同 §4 |
| `nvenc_preset` | `p4` | NVENC 预设，`p1` 最快 |
| `trt_fp16` | true | TensorRT 用 FP16 |
| `ort_intra_threads` | 0 | 多个进程时调小（例如 CPU 核数 ÷ 同时运行的进程数），避免会话互相抢 CPU |

实测数字和推荐的 `VC_JOBS` 等 GPU 部分的压测跑完再填。

**还要 GPU 机器的部分（另一份计划）：** TensorRT / CUDA 会话与引擎缓存、NVDEC / NVENC 实测；帧全程留在显存（CUDA 前处理、CUDA 马赛克、硬件帧贯通、显存预算与退路）；压测脚本 `scripts/bench_gpu.sh` 与验收脚本 `scripts/accept_gpu.sh`；查实并填好的 sha256、digest 与 TensorRT 版本。

## 测完 4K 后请反馈
````

把 `cpp/README.md` 里的：

```markdown
## 构建
```

换成：

```markdown
**Linux（NVIDIA）**：`scripts/setup_linux.sh` 在 Ubuntu 22.04 / 24.04 上装齐依赖并以 `-DVC_WITH_CUDA=ON` 构建到 `build/cpp-linux`（会联网，要先设 `VC_ALLOW_DOWNLOAD=1`；下载按脚本里的 sha256 表校验，见 `docs/USAGE.md` §7）。它从源码编 FFmpeg 8.1.2（NVENC / NVDEC + libx264）装到 `~/.cache/video-compliance/linux/ffmpeg-8.1.2/`，CMake 用 `-DCMAKE_PREFIX_PATH=<那个目录>` 找到它；onnxruntime 用 Linux GPU 包 `onnxruntime-linux-x64-gpu-1.30.0`，解压到上面同一个 `third_party` 目录。生产用的 Docker 镜像见 `docker/Dockerfile`。

## 构建
```

- [ ] **Step 7: 提交**

```bash
git add scripts/setup_linux.sh docker/Dockerfile docker/build.sh docker/entrypoint.sh .dockerignore \
    py/tests/test_linux_deploy.py docs/USAGE.md cpp/README.md
git commit -m "build: add the Linux setup script and a two-stage Docker image"
```

### Task 10: 本机验收（控制者执行）

前 9 个任务各自跑过它们改动到的测试组和 pytest 文件；这里只做三件需要真实视频的事，外加 Release 构建。全量测试不在本机跑（交给 CI 或目标机）。全部在 worktree 里执行，`data/` 用主 checkout 的绝对路径只读。开始前确认本机没有别的重任务：`ps -Ao pcpu,comm -r | head -5` 里除系统进程外没有持续占用 CPU 的进程。

**Files:** 不改仓库里的文件；产物都在 `build/accept/`、`build/baseline-src/`、`build/baseline-release/`、`build/cpp-release/`（`build/` 被 `.gitignore` 忽略）。

**Interfaces:**
- Consumes: Task 1–9 的全部产物；主 checkout 的 `data/videos/sg_drive_2400s.mp4`（1080p 59.94 fps H.264，7250 帧，2 分钟）、`data/models/r3`、`data/models/yunet`。
- Produces: 交给用户的验收结论（贴进最后的回复和 PR 描述）。

- [ ] **Step 1: Release 构建（`VC_WITH_CUDA=OFF`）零警告**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
mkdir -p build/accept
rm -rf build/cpp-release
"${SB[@]}" cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF
"${SB[@]}" cmake --build build/cpp-release -j 4 > build/accept/release-build.log 2>&1; echo "build exit $?"
grep -c 'warning:' build/accept/release-build.log
grep -c 'VC_WITH_CUDA' build/cpp-release/compile_commands.json
```

预期：`build exit 0`；警告计数 `0`；`VC_WITH_CUDA` 计数 `0`（`OFF` 时没有这个宏）。

- [ ] **Step 2: 基线构建（本分支出发时的 `main`）**

`git archive` 把本分支与 `main` 的分叉点（写计划时是 0f2aa85）的 `cpp/` 原样导出到 `build/baseline-src/`，不动 git 的状态（不建分支、不建 worktree）：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
rm -rf build/baseline-src build/baseline-release
mkdir -p build/baseline-src
BASE=$(git merge-base HEAD main)
echo "baseline $BASE"
git archive "$BASE" cpp | tar -x -C build/baseline-src
"${SB[@]}" cmake -S build/baseline-src/cpp -B build/baseline-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF
"${SB[@]}" cmake --build build/baseline-release -j 4 > build/accept/baseline-build.log 2>&1; echo "build exit $?"
```

预期：打印 `baseline <提交号>`（与写计划时一致则是 `0f2aa85…`）、`build exit 0`，得到 `build/baseline-release/vc_redact`。

- [ ] **Step 3: `sg_drive_2400s` 上 `masks.jsonl` 与基线逐字节相同**

两边都用各自的默认检测后端（新版 `--ep auto`，在 macOS 上就是 CoreML → CPU；基线默认 `coreml`）和 `--encoder software`（两边都只用 libx264：H.264 原片的码率不变、Level 由 libx264 自己选，编码参数与改动前完全一样，所以解码出来的每一帧也应相同，这顺带验证了马赛克改用 `mosaic_grid.h` 之后逐像素不变）。一次只跑一个：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
MAIN=/Users/warsen/Projects/video-compliance-process
for side in baseline new; do
  bin=build/cpp-release/vc_redact
  [ "$side" = baseline ] && bin=build/baseline-release/vc_redact
  "${SB[@]}" "$bin" --input "$MAIN/data/videos/sg_drive_2400s.mp4" --out "build/accept/$side" \
      --model "$MAIN/data/models/r3" --face-model "$MAIN/data/models/yunet" --encoder software --overwrite \
      > "build/accept/$side.log" 2>&1
  echo "$side exit $?"
done
cmp build/accept/baseline/sg_drive_2400s/masks.jsonl build/accept/new/sg_drive_2400s/masks.jsonl && echo "masks.jsonl identical"
cmp build/accept/baseline/sg_drive_2400s/detections.jsonl build/accept/new/sg_drive_2400s/detections.jsonl && echo "detections.jsonl identical"
for side in baseline new; do
  "${SB[@]}" ffmpeg -v error -i "build/accept/$side/sg_drive_2400s/sg_drive_2400s.redacted.mp4" -map 0:v:0 -f framemd5 - \
      | grep -v '^#' > "build/accept/$side.framemd5"
done
cmp build/accept/baseline.framemd5 build/accept/new.framemd5 && echo "decoded frames identical"
"${SB[@]}" env PYTHONPATH=py "$MAIN/.venv/bin/python" -m vc_eval integrity --source "$MAIN/data/videos/sg_drive_2400s.mp4" \
    --redacted build/accept/new/sg_drive_2400s/sg_drive_2400s.redacted.mp4 --json build/accept/new-integrity.json
"${SB[@]}" "$MAIN/.venv/bin/python" - <<'EOF'
import json
b = json.load(open("build/accept/baseline/sg_drive_2400s/stats.json", encoding="utf-8"))
n = json.load(open("build/accept/new/sg_drive_2400s/stats.json", encoding="utf-8"))
print("baseline:", b["frames"], b["detector"], b["encoder"])
print("new:     ", n["frames"], n["detector"], n["encoder"], n["output_bit_rate"], n["pipeline"])
print("config keys added:", sorted(set(n["config"]) - set(b["config"])))
EOF
```

预期：两次都 `exit 0`；`masks.jsonl identical`、`detections.jsonl identical`、`decoded frames identical`；`integrity:` 一行八项都是 `ok`（`h264_level` 为 4.2，libx264 对 1080p 59.94 fps 自己选的就是 4.2）；两边都是 `7250 student/CoreML + yunet/CoreML libx264`，新版 `output_bit_rate` 等于原片码率 `6200390`、`pipeline` 为 `cpu-frames`；多出的配置键正好是 `['nvenc_preset', 'ort_intra_threads', 'output_bitrate_factor_hevc', 'trt_fp16']`。

如果 `masks.jsonl` 不同：先把基线再跑一次（输出到 `build/accept/baseline2`）并与 `build/accept/baseline` 比较。基线自己两次就不同，说明是 CoreML 的数值不确定，改用 `--ep cpu` 把两边各跑一次再比（CPU 上 7250 帧约 5–10 分钟）；基线自己相同而新版不同，就是回归，停下来按 superpowers:systematic-debugging 查，不要放过。

- [ ] **Step 4: HEVC 原片编成 H.264（含 10 位与 4K60）**

用本机 ffmpeg 从 `sg_drive_2400s` 现场生成两段 HEVC：5 秒 1080p 59.94 fps 10 位（带音频），2 秒放大到 4K60 的 8 位（不带音频）。1080p 那段用真模型，4K 那段用空的回放检测（只验编码，避免 YuNet 在 3840×2176 上首次编译 CoreML）；两段都用默认编码器（本机是 `h264_videotoolbox`）：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
MAIN=/Users/warsen/Projects/video-compliance-process
A=build/accept
"${SB[@]}" ffmpeg -v error -y -ss 30 -t 5 -i "$MAIN/data/videos/sg_drive_2400s.mp4" -map 0:v:0 -map '0:a?' \
    -c:v libx265 -preset ultrafast -pix_fmt yuv420p10le -x265-params log-level=error -c:a copy "$A/hevc10_1080p.mp4"
"${SB[@]}" ffmpeg -v error -y -ss 30 -t 2 -i "$MAIN/data/videos/sg_drive_2400s.mp4" -map 0:v:0 \
    -vf scale=3840:2160,fps=60 -c:v libx265 -preset ultrafast -pix_fmt yuv420p -x265-params log-level=error "$A/hevc_4k60.mp4"
: > "$A/none.jsonl"
"${SB[@]}" build/cpp-release/vc_redact --input "$A/hevc10_1080p.mp4" --out "$A/hevc" \
    --model "$MAIN/data/models/r3" --face-model "$MAIN/data/models/yunet" --overwrite > "$A/hevc10_1080p.log" 2>&1
echo "1080p exit $?"
"${SB[@]}" build/cpp-release/vc_redact --input "$A/hevc_4k60.mp4" --out "$A/hevc" --fake-detections "$A/none.jsonl" \
    --overwrite > "$A/hevc_4k60.log" 2>&1
echo "4k exit $?"
grep 'note:' "$A/hevc10_1080p.log"
for f in hevc10_1080p hevc_4k60; do
  "${SB[@]}" env PYTHONPATH=py "$MAIN/.venv/bin/python" -m vc_eval integrity --source "$A/$f.mp4" \
      --redacted "$A/hevc/$f/$f.redacted.mp4" --json "$A/$f-integrity.json"
done
"${SB[@]}" "$MAIN/.venv/bin/python" - <<'EOF'
import json, subprocess
for f in ("hevc10_1080p", "hevc_4k60"):
    src = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=codec_name,bit_rate", "-of", "json", f"build/accept/{f}.mp4"],
                                    capture_output=True, text=True, check=True).stdout)["streams"][0]
    st = json.load(open(f"build/accept/hevc/{f}/stats.json", encoding="utf-8"))
    i = json.load(open(f"build/accept/{f}-integrity.json", encoding="utf-8"))
    want = round(int(src["bit_rate"]) * 1.5)
    print(f, src["codec_name"], "->", st["encoder"], "output_bit_rate", st["output_bit_rate"],
          "== round(1.5 x source)" if abs(st["output_bit_rate"] - want) <= 1 else f"!= {want}",
          "|", i["details"]["h264_level"], "| integrity", "PASS" if i["passed"] else "FAIL")
EOF
```

预期：两次都 `exit 0`；`note: the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0`；两条 `integrity:` 八项全 `ok`；最后两行形如
`hevc10_1080p hevc -> h264_videotoolbox output_bit_rate 11582201 == round(1.5 x source) | level 4.2; 1920x1080 at 59.94 fps needs 4.2 | integrity PASS` 和
`hevc_4k60 hevc -> h264_videotoolbox output_bit_rate 23971206 == round(1.5 x source) | level 5.2; 3840x2160 at 60.00 fps needs 5.2 | integrity PASS`（码率的具体数字随 x265 的输出而定，判据是"等于原码率 × 1.5"和两个 PASS）。

- [ ] **Step 5: 交给用户**

把 Step 1–4 的结果写进最后的回复（以及之后的 PR 描述，注明"全量测试待 CI / 目标机"），然后按 superpowers:finishing-a-development-branch 收尾；合并进 `main` 之后，按 CLAUDE.md 立即把主 checkout `/Users/warsen/Projects/video-compliance-process` 切到最新的 `main`（本地有未提交改动时先问用户，不用 stash），并告诉用户可以在本地验证了。

---

## 已定的做法

设计没写到、或代码里有两种合理读法的地方，按下面定（实现时不再另找办法；GPU 计划要改的会写明）：

1. **编码器候选按平台裁剪。** `h264_nvenc` 只在非 macOS 上尝试，`h264_videotoolbox` 只在 macOS 上尝试：macOS 上根本没有 NVENC，每次都试会在 stderr 多出一行"could not be opened"。`--encoder software` 仍只用 libx264。
2. **Level 只给 VideoToolbox 和 NVENC 指定。** 本机实测 `h264_videotoolbox` 自己选的 Level 不够（1080p 59.94 fps 给 4.0、4K60 给 5.1；只设 Level 不设 profile 又会退成 Baseline），所以设 High profile 和 clamp(所需 Level, 3.0, 5.2)；NVENC 设 Level 和 `nvenc_preset`（本机走不到，1c 在目标机上验证）；libx264 实测自己选的够（1080p 59.94 → 4.2，4K60 → 5.2），不动它，这样 libx264 的输出与改动前逐字节相同，Task 10 才能比对解码后的每一帧。副作用：H.264 原片走 VideoToolbox 时的输出（Level、显式 High）与以前不同，属于"输出改成 H.264"这一处允许的变化。
3. **编码器的输入一律 8 位 4:2:0。** 设计点名 10 位和 4:2:2，4:4:4 也按同样处理（硬件编解码器普遍不支持 H.264 4:4:4），note 对任何"不是 8 位 4:2:0"的原片都打。10 位在 `VideoReader` 里本来就转成 NV12；4:2:2 / 4:4:4 在 `VideoWriter::writeVideo` 现有的转换分支里转，马赛克仍打在原格式的帧上。
4. **转换的 note 在运行结束时打印**（`redactWarnings`，`vc_redact: note: …`），与已有的 `detect_fps` note 同一个出口，便于测试；不是开始时打印。
5. **`stats.json` 加 `output_bit_rate` 和 `pipeline`**（`"cpu-frames"`）。`pipeline` 是 `batch.json` 要的（设计 §4 定了它的两个取值），`output_bit_rate` 让码率系数看得见、测得了。运行结束的摘要文字不变。
6. **"是不是 H.264 原片"看 `codecpar->codec_id`**，不看解码器名字（`libdav1d` 之类的名字不可靠）。
7. **TensorRT / CUDA 在本计划里总是"不可用"**，原因三选一（`gpuProviderUnavailableReason`）：onnxruntime 里没有这个提供者（macOS 包就是这样）、构建没开 `VC_WITH_CUDA`、或"stage 1c 才有"。1c 把第三支换成真正的会话设置，接口不变。检查用 `Ort::GetAvailableProviders()`。
8. **回退提示的措辞**：下一个不是最后一个时写 `trying <next>`，是最后一个（CPU）时写 `using <next>`。macOS 上 CoreML → CPU 的两种提示与改动前一字不差。
9. **显式的 `tensorrt` 直接退到 CPU，不经过 CUDA**（设计：显式指定的用不了时"沿用 coreml 的做法：stderr 提示后退到 CPU"）。
10. **`vc_detect` 也接受新的 `--ep` 取值**（共用解析函数），默认仍是 `cpu`。
11. **缓存目录按需计算**（`SessionCaches` 存的是函数）：只有真的尝试 CoreML / TensorRT 时才去读模型文件算 sha256，`--ep cpu` 不读，读不了文件的报错时机也与以前一样。
12. **`--hwaccel auto` 的读法**：任务里的"Linux 上试 cuda，否则 none"读作"Linux 上先试 CUDA、打不开退回软件（stderr 一行）"；macOS 上 `auto` 保持原样（先试 VideoToolbox，打不开悄悄软解），所以 `vc_extract` 在 macOS 上的默认行为不变。显式 `cuda` 永远不报错（退回软件并写一行），显式 `videotoolbox` 打不开仍然报错。`vc_extract` 在 Linux 上的 `auto` 也会先试 CUDA。
13. **NVDEC 拷回内存的代码写成通用的**（`AV_PIX_FMT_CUDA` + `av_hwframe_transfer_data`，与 VideoToolbox 同一分支），本机只能走到"打不开"那一支，1c 在目标机上验证。
14. **`redact.sh` 遇到同名视频预先拒绝**（第二个不运行，记退出码 3）：并行时两个进程会同时通过 `vc_redact` 自己的输出目录检查、写进同一个目录。这与文档写的"第二个会被拒绝"一致；比较的是文件名去掉扩展名后的精确字符串，macOS 上只差大小写的两个名字（`A.mp4` 与 `a.mp4`）若被并行处理仍可能撞上同一个目录，这种情况文档里提示分开输出目录。
15. **`redact.sh` 的输出**：同时只跑一个时，`vc_redact` 的输出照旧实时打印；同时跑多个时，每个视频的输出先进临时日志、结束时整段打印，不交错。开始一行仍是 `=== HH:MM:SS <视频>`，有 GPU 时多 `[gpu N] `。
16. **分卡**：下一个视频分给"从上次分到的那块的下一块开始、第一块还有空位的卡"；所有视频一样长时就是严格的 0、1、0、1……
17. **`batch.json` 由 `scripts/batch_json.py`（只用标准库）写**：bash 3.2 解析不了 JSON，Ubuntu 默认不带 jq；找不到 python3 时 `redact.sh` 开始前就拒绝（退出码 2）。每次运行覆盖上一次的 `batch.json`；墙钟时间用 Python 的 `time.time()`。视频路径里有制表符或换行时拒绝（结果表是 TSV）。`VC_JOBS` 上限 64。
18. **`setup_linux.sh` 的依赖来源**：libx264 用 Ubuntu 仓库的 `libx264-dev`（apt 签名校验），FFmpeg 仍从源码编并 `--enable-libx264`——本机找不到一个能固定 sha256 的 x264 源码包，不编造；CUDA 工具包、cuDNN 9、TensorRT 运行库走 NVIDIA 的 apt 仓库（仓库的 keyring 包按 sha256 固定，TensorRT 按 `TENSORRT_VERSION` 固定版本），代替设计里"下载并按 sha256 校验 TensorRT"的压缩包，校验由 apt 的签名承担。目标机计划若坚持用 TensorRT 压缩包，就在 `PINS` 里加一项。
19. **版本**：nv-codec-headers 用 12.2.72.0（Video Codec SDK 12.2，驱动 ≥ 550；13.x 要 570）；FFmpeg 用 8.1.2（与本机 Homebrew 相同，sha256 取自其公式）；Ubuntu 22.04 自带 CMake 3.22 低于项目要求的 3.24，用 Kitware 的 3.31.6 二进制包；CUDA 取 12.8.1（`nvidia/cuda:12.8.1-cudnn-devel/runtime-ubuntu22.04`、`cuda-toolkit-12-8`），**要在目标机上对照 onnxruntime 1.30.0 的发布说明确认**，不对就改 `CUDA_TAG` / `CUDA_APT_VERSION`。
20. **镜像里的 Python** 用 Ubuntu 22.04 的 `python3` 和 `python3-numpy`（apt），只拷 `vc_eval` 和 `vc_label` 的 `__init__.py`、`contracts.py`；`scripts/bench_gpu.sh`、`scripts/accept_gpu.sh` 在 3b 写出来后再加进运行阶段。
21. **加了三个设计没点名的小文件**：`.dockerignore`（构建上下文是整个仓库，必须把含个人信息的 `data/` 挡在外面）、`docker/build.sh`（强制传入 digest 与 TensorRT 版本，否则 `FROM …@` 只会报一句看不懂的 "invalid reference format"）、`docker/entrypoint.sh`（设计的用法是 `vc-redact /in /out`，输入在前，而 `redact.sh` 输出在前）。
22. **`PixelRect` 搬进 `mosaic_grid.h`**（仍在命名空间 `vc`，`mask/mosaic.h` 照旧提供它），网格函数放在 `vc::grid`，不用标准库；`maskCells` 带 `snap` 参数，块从画面网格上数（`col0 = area.x1 / cellW`），所以车牌那种不对齐网格的框也由同一组函数描述；`snapRect` 先在 64 位里算再转 `int`、`clipRect` 在转换后裁剪，与原来逐位相同（包括超大坐标时的行为）。
23. **警告选项只给 C++ 源文件**（生成器表达式）：C++ 的编译命令与以前相同（Task 1 Step 4 核对），以后的 `.cu` 文件不会把 `-Wpedantic` 直接交给 nvcc。
24. **Linux 上 `VC_WITH_CUDA=OFF` 也找 onnxruntime 的 GPU 包**：它也带 CPU 执行提供者，只管一个包。
25. **基线**用 `git archive $(git merge-base HEAD main) cpp` 导出到 `build/baseline-src/` 再构建（写计划时是 0f2aa85，已含车牌不对齐网格的改动，所以 `masks.jsonl` 的 `snap` 字段两边都有），不建分支、不加 worktree。

## Self-Review

**1. 设计覆盖（本计划负责的 §6 本机阶段）**

| 设计条目 | 任务 |
|---|---|
| §2 `VC_WITH_CUDA`（默认 OFF，`CMAKE_CUDA_ARCHITECTURES` 默认 `86;89`，宏隔开，OFF 不链接 CUDA） | Task 1（`ON` 无 nvcc 时报错已核对；`OFF` 编译命令不变） |
| §2 onnxruntime 按平台选包、Docker 里在 `/opt` | Task 1（ThirdParty）、Task 9（`VC_THIRD_PARTY_DIR=/opt/third_party`） |
| §2 FFmpeg 8 源码编、NVENC / NVDEC + libx264、GPL 记入待定 | Task 9（`build_ffmpeg`，USAGE §7 写明 GPL） |
| §2 Docker 两阶段、digest 锁定、模型与缓存挂载、用法 `/in /out` | Task 9（Dockerfile、`build.sh`、`entrypoint.sh`；`bench_gpu.sh` / `accept_gpu.sh` 留给 3b，见"已定的做法"20） |
| §2 `setup_linux.sh`：22.04 / 24.04、驱动 ≥ 550、apt、sha256 校验、共享缓存、`VC_ALLOW_DOWNLOAD=1`、编 `vc_redact`（ON） | Task 9 |
| §2 本机静态检查（shellcheck / hadolint，没有就跳过） | Task 9（shellcheck 跑，hadolint 跳过并注明） |
| §3 `--ep` 五个取值、`auto` 顺序、回退 stderr、显式退 CPU、`detector` 写实际后端 | Task 4 |
| §3 TensorRT 缓存目录 `$VC_CACHE/trt-cache/<sha256>/<w>x<h>/` | Task 4（`tensorRtCacheDirectory`） |
| §3 `--hwaccel cuda`、Linux 默认 `auto`、打不开退回软解一行 | Task 5 |
| §3 输出一律 H.264、编码器顺序、`encoder` 写进 stats | Task 3 |
| §3 码率：H.264 不变、非 H.264 × `output_bitrate_factor_hevc`（1.5，[1, 4]）、读不到按 8 Mb/s × 系数 | Task 2（键）、Task 3（`h264OutputBitRate`） |
| §3 4K60 Level 5.2 由编码器按分辨率与帧率定；`integrity` 检查 H.264 与 Level | Task 3（`h264LevelIdc`，VT / NVENC 指定）、Task 6 |
| §3 10 位 / 4:2:2 转 8 位 4:2:0，stderr 一行 | Task 3 |
| §3 本机能验证的各项（HEVC → H.264 与码率、解析、`auto` 回退的假后端测试、新键、integrity） | Task 2–6，Task 10 Step 4 |
| §4 马赛克网格纯函数（`__host__ __device__`），CPU 版改用并逐字节不变 | Task 1（宏）、Task 7 |
| §4 `pipeline` 字段 | Task 3（只有 `cpu-frames`；`gpu` 在 2b） |
| §5 `VC_JOBS`（每卡默认 4，无 GPU 默认 1）、`VC_GPUS` 轮流与 `CUDA_VISIBLE_DEVICES`、失败语义不变、同名拒绝、`batch.json`（耗时、fps、后端、编码器、路径、合计吞吐） | Task 8 |
| §5 调参项写进 USAGE 的 Linux 一节（`VC_JOBS`、`detect_fps`、`nvenc_preset`、`trt_fp16`、`ort_intra_threads`、消费级卡会话上限） | Task 2（§4 表）、Task 9（§7） |
| §5 三个脚本的 shellcheck | Task 9（本计划只有 `redact.sh` 与两个新脚本；`bench_gpu.sh`、`accept_gpu.sh` 属于 3b） |
| §6 `OFF` 的行为与现在逐字节相同（H.264 输出除外） | Task 10 Step 3（`masks.jsonl`、`detections.jsonl`、解码后的帧） |
| §7 本机只跑过滤后的 gtest 与指定 pytest 文件 | 每个任务的运行步骤；六组过滤命令扩成含新套件 |
| 1c、2b、3b | 不在本计划，开头列了它们将覆盖的内容 |

没有找到本计划范围内没落到任务的条目。

**2. 占位扫描**：全文没有 TBD / TODO / "以后再做" / "类似 Task N"。`setup_linux.sh` 的 `PINS` 里五个空 sha256、`docker/build.sh` 必填的两个 digest 与 `TENSORRT_VERSION` 是按要求故意留空、由脚本拒绝的值，不是计划占位；`gpuProviderUnavailableReason` 的 "stage 1c" 分支是有意的阶段边界（"已定的做法"7）。每个改代码的步骤都给了完整代码，每个"把 … 换成"的原文都取自当前文件。

**3. 类型与命名一致性**：`Platform` / `buildPlatform()` / `builtWithCuda()`（Task 1）在 Task 3、4、5 中同名使用；`RedactConfig::outputBitrateFactorHevc`、`nvencPreset`、`trtFp16`、`ortIntraThreads`（Task 2）在 Task 3（`writerConfig`）与 Task 4（`makeDetector` 里的 `SessionSettings{o.config.ortIntraThreads, o.config.trtFp16}`）中一致；`SessionSettings` 的成员顺序是 `intraOpThreads, tensorRtFp16`，测试里的 `SessionSettings{1, true}` 与之相符；`WriterConfig::nvencPreset` 与 `setEncoderOptions` 一致；`RedactStats::outputBitRate` / `formatNote` / `pipeline` 与 `stats.json` 的 `output_bit_rate` / `pipeline`、`batch_json.py` 读的 `pipeline` 一致；`h264LevelIdc` 与 Python 的 `required_h264_level` 用同一张表、同一组测试值；`HwDecodePlan{device, required, noteOnFallback}` 与测试里的聚合初始化顺序一致；`grid::MaskCells{luma, chroma}`、`grid::CellGrid{area, cellW, cellH, cols, rows}` 与 `mosaic.cpp` 的用法一致；`check_pins` 的输出格式（`ok <名字> <URL> <sha256>`）与 `test_linux_deploy.py` 的解析一致。

**4. 回放核对**：计划写完后，把全文的"新建 / 把 … 换成 / 整个换成 / 末尾追加"按顺序回放到 `main`（0f2aa85）的一份干净副本上：159 处操作，每处原文都恰好匹配一次。在回放出的副本上：Debug 构建零警告，六组过滤测试共 397 个全部通过（18 / 47 / 16 / 93 / 140 / 83，与 Task 7 之后各组的预期相同；各任务中间的计数按套件增减算出），`test_eval_integrity.py` 42、`test_eval_run.py` 113、`test_redact_script.py` 13 个通过，`test_linux_deploy.py` 15 个通过 1 个跳过（hadolint），Release 构建零警告且没有 `VC_WITH_CUDA` 宏。各步"确认失败"的预期是按改动前的代码推出的，其中 Task 8 的 `12 failed, 1 passed` 在现有脚本上实际跑过。
