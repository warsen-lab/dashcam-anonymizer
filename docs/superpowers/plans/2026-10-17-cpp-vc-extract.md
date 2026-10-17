# C++ 基础工程与抽帧模块 vc_extract 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建好 C++ 工程骨架（CMake + FFmpeg + GoogleTest），实现规格 §4 的抽帧模块 `vc_extract`：按“基础频率 + 运动加密 + 近重复去除 / 片段模式”从视频抽 JPEG 帧并写 `frames.csv`。

**Architecture:** `vc_core` 静态库放可复用组件（`FrameSampler` 纯逻辑、`LumaThumb` 运动打分、`VideoReader` 解码、`JpegWriter` 编码），后续 `vc_redact` 直接复用；`vc_extract_lib` 放命令行解析和抽帧流程，`vc_extract` 可执行文件只做退出码映射。测试用 GoogleTest，测试视频由本机 ffmpeg 命令现场生成。

**Tech Stack:** C++20、CMake ≥ 3.24、FFmpeg 8（libavformat / libavcodec / libswscale / libavutil，VideoToolbox 硬解）、GoogleTest。

**规格：** `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md`（§4 抽帧模块、§3 数据契约、§9 测试策略）

**计划路线（共 5 份，本份为第 1 份）：**

| # | 计划 | 规格章节 |
|---|---|---|
| 1 | C++ 基础工程与 vc_extract（本份） | §3、§4 |
| 2 | Python 环境与老师集成预标注、规则标记 | §5 |
| 3 | 训练与 ONNX 导出 | §6 |
| 4 | vc_redact 脱敏流水线 | §7 |
| 5 | 评测、参数扫描与验收报告 | §8 |

## Global Constraints

- C++20，`-Wall -Wextra -Wpedantic` 下**零警告**；CMake ≥ 3.24；构建并行度 `-j 4`（本机资源有限）。
- FFmpeg 通过 `cpp/cmake/FindFFmpeg.cmake` 查找（本机没有 pkg-config）；本机为 FFmpeg 8.1，位于 `/opt/homebrew/opt/ffmpeg`。
- **不联网**：除下面“开工前准备”里经用户同意的 `brew install googletest` 外，不下载任何东西。
- **只跑相关测试**：每个任务只用 `--gtest_filter` 跑本任务的用例，不跑全量；测试视频全部由本机 `ffmpeg` 生成到 `build/cpp/tests/tmp/`，都是 1–2 秒的 640×360 小视频。
- 抽帧参数默认值：`--fps 5`（**低于 5 报错**）、`--max-fps 15`、`--motion-threshold 0.05`、缩略图 160×90、JPEG `qscale 2`（≈ 质量 95）。
- 输出：`<out>/<视频文件名去扩展名>/<帧号:06d>.jpg` 与同目录 `frames.csv`，表头固定为 `video,frame,pts_ms,reason,motion_score`；`reason ∈ {base, motion, clip}`；`pts_ms` 保留 1 位小数、`motion_score` 保留 4 位小数。
- 退出码：0 成功 / 1 处理失败 / 2 参数错误 / 3 输出目录已存在且未加 `--overwrite`。
- 提交信息用 conventional commits（`feat:` / `test:` / `docs:` / `chore:`），不加署名行。
- 在特性分支 `feat/cpp-vc-extract` 上工作（如用 worktree，合并后要把用户本地 checkout 更新到最新）。

### 与规格的偏差（已在计划阶段决定，需要在交付时告知用户）

1. **C++ 端不引入 OpenCV**：缩略图自己做块平均，JPEG 用 FFmpeg 自带的 MJPEG 编码器。省掉约 100MB 的 OpenCV 源码下载和编译；Python 端仍用 opencv-python。计划 4 如确有需要再评估。
2. 规格里的 `--clip 起点,时长 --all-frames` 简化为 `--clip 起点,时长`：片段模式本身就是导出窗口内每一帧。
3. “JPEG 质量 95”落实为 MJPEG `qscale 2`（FFmpeg 的最高画质档）。

## 文件结构

```
cpp/
  CMakeLists.txt                     根工程：vc_core、vc_extract_lib、vc_extract、tests
  README.md                          构建、测试、vc_extract 用法
  cmake/FindFFmpeg.cmake             无 pkg-config 的 FFmpeg 查找，导出 FFmpeg::FFmpeg
  src/common/luma_plane.h            8-bit 亮度平面只读视图（io 与 sampler 共用）
  src/common/vc_error.h/.cpp         VcError、FFmpeg 错误码转文字
  src/sampler/frame_sampler.h/.cpp   抽帧决策（纯逻辑）
  src/sampler/luma_thumb.h/.cpp      160×90 缩略图与平均差
  src/io/av_ptr.h                    FFmpeg 对象 RAII（仅 src/io 内部使用）
  src/io/video_reader.h/.cpp         解封装 + 解码 + 硬解下载 + 统一为 8-bit YUV
  src/io/jpeg_writer.h/.cpp          帧 → JPEG 文件
  apps/vc_extract/extract_options.h  ExtractOptions
  apps/vc_extract/extract_cli.h/.cpp 命令行解析、UsageError、帮助文本
  apps/vc_extract/extract_job.h/.cpp runExtract 主流程、OutputExistsError、ExtractStats
  apps/vc_extract/main.cpp           退出码映射、统计打印
  tests/CMakeLists.txt
  tests/test_media.h/.cpp            用 ffmpeg 现场生成测试视频
  tests/frame_sampler_test.cpp
  tests/luma_thumb_test.cpp
  tests/video_reader_test.cpp
  tests/jpeg_writer_test.cpp
  tests/extract_cli_test.cpp
  tests/extract_job_test.cpp
```

构建目录统一为仓库根下的 `build/cpp`（已在 `.gitignore` 中）。

---

## 开工前准备（由主会话执行，需要用户同意）

GoogleTest 本机没有，需要联网安装一次（Homebrew bottle，约 1–2MB）。**先向用户说明并征得同意**，再执行：

- [ ] **Step 1: 征得用户同意后安装 GoogleTest**

```bash
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install googletest
```

- [ ] **Step 2: 确认 CMake 能找到它**

Run: `ls /opt/homebrew/lib/cmake/GTest/GTestConfig.cmake`
Expected: 打印出该路径（文件存在）

- [ ] **Step 3: 建分支**

```bash
git checkout -b feat/cpp-vc-extract
```

---

### Task 1: CMake 骨架与 FrameSampler

**Files:**
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/tests/CMakeLists.txt`
- Create: `cpp/src/sampler/frame_sampler.h`
- Create: `cpp/src/sampler/frame_sampler.cpp`
- Test: `cpp/tests/frame_sampler_test.cpp`

**Interfaces:**
- Consumes: 无
- Produces（`namespace vc`，头文件 `sampler/frame_sampler.h`）：
  - `enum class SampleReason { None, Base, Motion, Clip };`
  - `const char* toString(SampleReason)` → `"none" | "base" | "motion" | "clip"`
  - `struct ClipWindow { double startMs; double durationMs; };`
  - `inline constexpr double kMinBaseFps = 5.0;`
  - `struct SamplerConfig { double baseFps = 5.0; double maxFps = 15.0; double motionThreshold = 0.05; std::optional<double> dedupThreshold; std::optional<ClipWindow> clip; };`
  - `class FrameSampler { explicit FrameSampler(const SamplerConfig&); SampleReason offer(double ptsMs, double motionScore, double diffToLastKept); };` 构造时配置非法抛 `std::invalid_argument`。
  - CMake 目标 `vc_core`（静态库，`PUBLIC` 包含目录 `cpp/src`）、`vc_tests`（测试可执行文件）。

决策规则（实现与测试都以此为准）：
1. 设置了 `clip`：`startMs ≤ pts < startMs + durationMs` 返回 `Clip`，否则 `None`，忽略其他规则。
2. 第一帧总是保留（`Base`）。
3. 距上次保留 ≥ `1000/baseFps − 1ms` 为“基础到期”；`motionScore ≥ motionThreshold` 且距上次保留 ≥ `1000/maxFps − 1ms` 为“运动到期”；两者都不满足 → `None`。
4. 到期但设置了 `dedupThreshold` 且 `diffToLastKept < dedupThreshold` → `None`（不更新上次保留时间）。
5. 保留：基础到期返回 `Base`，否则 `Motion`。

- [ ] **Step 1: 写 CMake 文件和失败的测试**

`cpp/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(video_compliance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(vc_core
  src/sampler/frame_sampler.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

option(VC_BUILD_TESTS "Build the GoogleTest suite" ON)
if(VC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

`cpp/tests/CMakeLists.txt`：

```cmake
find_package(GTest REQUIRED)

add_executable(vc_tests
  frame_sampler_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vc_tests)
```

`cpp/tests/frame_sampler_test.cpp`：

```cpp
#include "sampler/frame_sampler.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vc {
namespace {

double ptsAt60Fps(int index) { return index * 1000.0 / 60.0; }

struct Kept {
  int index;
  SampleReason reason;
};

// Offers frames [0, count) at 60 fps with the given per-frame scores.
template <typename Motion, typename DiffToKept>
std::vector<Kept> offerFrames(FrameSampler& sampler, int count, Motion motion,
                              DiffToKept diffToKept) {
  std::vector<Kept> kept;
  for (int i = 0; i < count; ++i) {
    const SampleReason reason = sampler.offer(ptsAt60Fps(i), motion(i), diffToKept(i));
    if (reason != SampleReason::None) kept.push_back({i, reason});
  }
  return kept;
}

std::vector<int> indicesOf(const std::vector<Kept>& kept) {
  std::vector<int> indices;
  for (const Kept& k : kept) indices.push_back(k.index);
  return indices;
}

double still(int) { return 0.0; }
double different(int) { return 1.0; }

TEST(FrameSampler, RejectsBaseFpsBelowFive) {
  SamplerConfig config;
  config.baseFps = 4.9;
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, RejectsMaxFpsBelowBaseFps) {
  SamplerConfig config;
  config.baseFps = 10.0;
  config.maxFps = 8.0;
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, RejectsEmptyClip) {
  SamplerConfig config;
  config.clip = ClipWindow{0.0, 0.0};
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, KeepsEveryTwelfthFrameAt60FpsWhenStill) {
  FrameSampler sampler(SamplerConfig{});

  const std::vector<Kept> kept = offerFrames(sampler, 120, still, different);

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{0, 12, 24, 36, 48, 60, 72, 84, 96, 108}));
  for (const Kept& k : kept) EXPECT_EQ(k.reason, SampleReason::Base);
}

TEST(FrameSampler, SwitchesToMaxFpsWhileMotionIsHigh) {
  FrameSampler sampler(SamplerConfig{});  // base 5 fps, max 15 fps, threshold 0.05

  const std::vector<Kept> kept =
      offerFrames(sampler, 120, [](int i) { return i < 60 ? 0.0 : 0.5; }, different);

  std::vector<int> expected = {0, 12, 24, 36, 48, 60};
  for (int i = 64; i < 120; i += 4) expected.push_back(i);
  EXPECT_EQ(indicesOf(kept), expected);
  EXPECT_EQ(kept[5].reason, SampleReason::Base);    // frame 60 was due at the base rate
  EXPECT_EQ(kept[6].reason, SampleReason::Motion);  // frame 64 only due because of motion
}

TEST(FrameSampler, DedupSkipsFramesSimilarToLastKept) {
  SamplerConfig config;
  config.dedupThreshold = 0.01;
  FrameSampler sampler(config);

  const std::vector<Kept> kept =
      offerFrames(sampler, 30, still, [](int i) { return i < 24 ? 0.001 : 0.5; });

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{0, 24}));
}

TEST(FrameSampler, ClipKeepsEveryFrameInsideWindowOnly) {
  SamplerConfig config;
  config.clip = ClipWindow{500.0, 100.0};
  FrameSampler sampler(config);

  const std::vector<Kept> kept = offerFrames(sampler, 60, still, different);

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{30, 31, 32, 33, 34, 35}));
  for (const Kept& k : kept) EXPECT_EQ(k.reason, SampleReason::Clip);
}

TEST(FrameSampler, ReasonNamesMatchCsvValues) {
  EXPECT_STREQ(toString(SampleReason::None), "none");
  EXPECT_STREQ(toString(SampleReason::Base), "base");
  EXPECT_STREQ(toString(SampleReason::Motion), "motion");
  EXPECT_STREQ(toString(SampleReason::Clip), "clip");
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 配置并构建，确认失败**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`
Expected: FAIL，CMake 报 `Cannot find source file: src/sampler/frame_sampler.cpp`

- [ ] **Step 3: 写实现**

`cpp/src/sampler/frame_sampler.h`：

```cpp
#pragma once

#include <optional>

namespace vc {

enum class SampleReason { None, Base, Motion, Clip };

// "none" | "base" | "motion" | "clip" — the values written to frames.csv.
const char* toString(SampleReason reason);

struct ClipWindow {
  double startMs = 0;
  double durationMs = 0;
};

inline constexpr double kMinBaseFps = 5.0;

struct SamplerConfig {
  double baseFps = 5.0;           // must be >= kMinBaseFps
  double maxFps = 15.0;           // rate used while motionScore >= motionThreshold
  double motionThreshold = 0.05;  // mean abs luma diff to the previous frame, [0, 1]
  // A due frame whose diff to the last kept frame is below this is skipped.
  std::optional<double> dedupThreshold;
  // When set, every frame inside the window is kept and nothing else.
  std::optional<ClipWindow> clip;
};

// Decides which decoded frames to keep. Feed every decoded frame in display
// order; the caller computes the scores, so this class is pure logic.
class FrameSampler {
 public:
  // Throws std::invalid_argument for an invalid config.
  explicit FrameSampler(const SamplerConfig& config);

  // motionScore: diff to the previous decoded frame (0 for the first frame).
  // diffToLastKept: diff to the last kept frame (ignored before the first keep).
  SampleReason offer(double ptsMs, double motionScore, double diffToLastKept);

 private:
  SamplerConfig config_;
  std::optional<double> lastKeptMs_;
};

}  // namespace vc
```

`cpp/src/sampler/frame_sampler.cpp`：

```cpp
#include "sampler/frame_sampler.h"

#include <cstdio>
#include <stdexcept>

namespace vc {
namespace {

// Container timestamps are rounded; accept a frame up to 1 ms early.
constexpr double kTimingToleranceMs = 1.0;

}  // namespace

const char* toString(SampleReason reason) {
  switch (reason) {
    case SampleReason::Base:
      return "base";
    case SampleReason::Motion:
      return "motion";
    case SampleReason::Clip:
      return "clip";
    case SampleReason::None:
      break;
  }
  return "none";
}

FrameSampler::FrameSampler(const SamplerConfig& config) : config_(config) {
  if (config_.baseFps < kMinBaseFps) {
    char message[80];
    std::snprintf(message, sizeof(message), "fps must be >= %g, got %g", kMinBaseFps,
                  config_.baseFps);
    throw std::invalid_argument(message);
  }
  if (config_.maxFps < config_.baseFps) {
    throw std::invalid_argument("max fps must be >= base fps");
  }
  if (config_.motionThreshold < 0) {
    throw std::invalid_argument("motion threshold must be >= 0");
  }
  if (config_.dedupThreshold && *config_.dedupThreshold < 0) {
    throw std::invalid_argument("dedup threshold must be >= 0");
  }
  if (config_.clip && (config_.clip->startMs < 0 || config_.clip->durationMs <= 0)) {
    throw std::invalid_argument("clip needs start >= 0 and duration > 0");
  }
}

SampleReason FrameSampler::offer(double ptsMs, double motionScore, double diffToLastKept) {
  if (config_.clip) {
    const double start = config_.clip->startMs;
    const bool inside = ptsMs >= start && ptsMs < start + config_.clip->durationMs;
    return inside ? SampleReason::Clip : SampleReason::None;
  }
  if (!lastKeptMs_) {
    lastKeptMs_ = ptsMs;
    return SampleReason::Base;
  }
  const double elapsed = ptsMs - *lastKeptMs_;
  const bool dueAtBase = elapsed >= 1000.0 / config_.baseFps - kTimingToleranceMs;
  const bool dueAtMax = elapsed >= 1000.0 / config_.maxFps - kTimingToleranceMs;
  const bool moving = motionScore >= config_.motionThreshold;
  if (!dueAtBase && !(moving && dueAtMax)) {
    return SampleReason::None;
  }
  if (config_.dedupThreshold && diffToLastKept < *config_.dedupThreshold) {
    return SampleReason::None;
  }
  lastKeptMs_ = ptsMs;
  return dueAtBase ? SampleReason::Base : SampleReason::Motion;
}

}  // namespace vc
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*'`
Expected: 构建无警告；`[  PASSED  ] 8 tests.`

- [ ] **Step 5: 提交**

```bash
git add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/src/sampler/frame_sampler.h cpp/src/sampler/frame_sampler.cpp cpp/tests/frame_sampler_test.cpp
git commit -m "feat: add C++ project skeleton and FrameSampler"
```

---

### Task 2: 亮度缩略图与运动打分

**Files:**
- Create: `cpp/src/common/luma_plane.h`
- Create: `cpp/src/sampler/luma_thumb.h`
- Create: `cpp/src/sampler/luma_thumb.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 增加 `src/sampler/luma_thumb.cpp`）
- Modify: `cpp/tests/CMakeLists.txt`（增加 `luma_thumb_test.cpp`）
- Test: `cpp/tests/luma_thumb_test.cpp`

**Interfaces:**
- Consumes: 无
- Produces（`namespace vc`）：
  - `common/luma_plane.h`：`struct LumaPlane { const uint8_t* data; int width; int height; int stride; };`
  - `sampler/luma_thumb.h`：`struct LumaThumb { int width; int height; std::vector<uint8_t> pixels; };`、`inline constexpr int kThumbWidth = 160; inline constexpr int kThumbHeight = 90;`、`LumaThumb makeLumaThumb(const LumaPlane&, int outWidth = kThumbWidth, int outHeight = kThumbHeight);`、`double meanAbsDiff(const LumaThumb&, const LumaThumb&);`（[0,1]，尺寸不一致抛 `std::invalid_argument`）

- [ ] **Step 1: 写失败的测试，并把新文件登记到 CMake**

`cpp/CMakeLists.txt` 中 `vc_core` 改为：

```cmake
add_library(vc_core
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
)
```

`cpp/tests/CMakeLists.txt` 中 `vc_tests` 改为：

```cmake
add_executable(vc_tests
  frame_sampler_test.cpp
  luma_thumb_test.cpp
)
```

`cpp/tests/luma_thumb_test.cpp`：

```cpp
#include "sampler/luma_thumb.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vc {
namespace {

LumaPlane planeOf(const std::vector<uint8_t>& pixels, int width, int height, int stride) {
  return LumaPlane{pixels.data(), width, height, stride};
}

TEST(LumaThumb, UniformPlaneGivesUniformThumb) {
  const std::vector<uint8_t> pixels(320 * 180, 100);

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 320, 180, 320));

  EXPECT_EQ(thumb.width, 160);
  EXPECT_EQ(thumb.height, 90);
  ASSERT_EQ(thumb.pixels.size(), 160u * 90u);
  for (uint8_t p : thumb.pixels) EXPECT_EQ(p, 100);
}

TEST(LumaThumb, AveragesEachBlock) {
  const std::vector<uint8_t> pixels = {0, 0, 255, 255,  //
                                       0, 0, 255, 255};

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 4, 2, 4), 2, 1);

  EXPECT_EQ(thumb.pixels, (std::vector<uint8_t>{0, 255}));
}

TEST(LumaThumb, IgnoresRowPadding) {
  // width 2, stride 4: the last two bytes of each row are padding.
  const std::vector<uint8_t> pixels = {10, 20, 255, 255,  //
                                       30, 40, 255, 255};

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 2, 2, 4), 1, 1);

  EXPECT_EQ(thumb.pixels, (std::vector<uint8_t>{25}));
}

TEST(LumaThumb, RejectsThumbLargerThanPlane) {
  const std::vector<uint8_t> pixels(4, 0);
  EXPECT_THROW(makeLumaThumb(planeOf(pixels, 2, 2, 2), 3, 1), std::invalid_argument);
}

TEST(LumaThumb, MeanAbsDiffIsNormalised) {
  const LumaThumb black{2, 1, {0, 0}};
  const LumaThumb white{2, 1, {255, 255}};
  const LumaThumb half{2, 1, {0, 255}};

  EXPECT_DOUBLE_EQ(meanAbsDiff(black, black), 0.0);
  EXPECT_DOUBLE_EQ(meanAbsDiff(black, white), 1.0);
  EXPECT_DOUBLE_EQ(meanAbsDiff(black, half), 0.5);
}

TEST(LumaThumb, MeanAbsDiffRejectsSizeMismatch) {
  const LumaThumb one{1, 1, {0}};
  const LumaThumb two{2, 1, {0, 0}};
  EXPECT_THROW(meanAbsDiff(one, two), std::invalid_argument);
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 构建，确认失败**

Run: `cmake --build build/cpp -j 4`
Expected: FAIL，`Cannot find source file: src/sampler/luma_thumb.cpp`

- [ ] **Step 3: 写实现**

`cpp/src/common/luma_plane.h`：

```cpp
#pragma once

#include <cstdint>

namespace vc {

// Read-only view of an 8-bit luma plane.
struct LumaPlane {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;  // bytes per row, >= width
};

}  // namespace vc
```

`cpp/src/sampler/luma_thumb.h`：

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include "common/luma_plane.h"

namespace vc {

// Small grayscale thumbnail used for motion and near-duplicate scoring.
struct LumaThumb {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // row-major, width * height
};

inline constexpr int kThumbWidth = 160;
inline constexpr int kThumbHeight = 90;

// Box-average downscale. Throws std::invalid_argument for an invalid plane or
// an output size that is non-positive or larger than the input.
LumaThumb makeLumaThumb(const LumaPlane& plane, int outWidth = kThumbWidth,
                        int outHeight = kThumbHeight);

// Mean absolute pixel difference normalised to [0, 1]. Throws
// std::invalid_argument when the sizes differ or the thumbnails are empty.
double meanAbsDiff(const LumaThumb& a, const LumaThumb& b);

}  // namespace vc
```

`cpp/src/sampler/luma_thumb.cpp`：

```cpp
#include "sampler/luma_thumb.h"

#include <cstdlib>
#include <stdexcept>

namespace vc {

LumaThumb makeLumaThumb(const LumaPlane& plane, int outWidth, int outHeight) {
  if (plane.data == nullptr || plane.width <= 0 || plane.height <= 0 ||
      plane.stride < plane.width) {
    throw std::invalid_argument("makeLumaThumb: invalid luma plane");
  }
  if (outWidth <= 0 || outHeight <= 0 || outWidth > plane.width ||
      outHeight > plane.height) {
    throw std::invalid_argument("makeLumaThumb: invalid thumbnail size");
  }
  LumaThumb thumb{outWidth, outHeight,
                  std::vector<uint8_t>(static_cast<size_t>(outWidth) * outHeight)};
  for (int oy = 0; oy < outHeight; ++oy) {
    const int y0 = oy * plane.height / outHeight;
    const int y1 = (oy + 1) * plane.height / outHeight;
    for (int ox = 0; ox < outWidth; ++ox) {
      const int x0 = ox * plane.width / outWidth;
      const int x1 = (ox + 1) * plane.width / outWidth;
      uint64_t sum = 0;
      for (int y = y0; y < y1; ++y) {
        const uint8_t* row = plane.data + static_cast<size_t>(y) * plane.stride;
        for (int x = x0; x < x1; ++x) {
          sum += row[x];
        }
      }
      const uint64_t count = static_cast<uint64_t>(y1 - y0) * static_cast<uint64_t>(x1 - x0);
      thumb.pixels[static_cast<size_t>(oy) * outWidth + ox] =
          static_cast<uint8_t>((sum + count / 2) / count);
    }
  }
  return thumb;
}

double meanAbsDiff(const LumaThumb& a, const LumaThumb& b) {
  if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size() ||
      a.pixels.empty()) {
    throw std::invalid_argument("meanAbsDiff: thumbnail size mismatch");
  }
  uint64_t sum = 0;
  for (size_t i = 0; i < a.pixels.size(); ++i) {
    sum += static_cast<uint64_t>(std::abs(int{a.pixels[i]} - int{b.pixels[i]}));
  }
  return static_cast<double>(sum) / (255.0 * static_cast<double>(a.pixels.size()));
}

}  // namespace vc
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='LumaThumb.*'`
Expected: 构建无警告；`[  PASSED  ] 6 tests.`

- [ ] **Step 5: 提交**

```bash
git add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/src/common/luma_plane.h cpp/src/sampler/luma_thumb.h cpp/src/sampler/luma_thumb.cpp cpp/tests/luma_thumb_test.cpp
git commit -m "feat: add luma thumbnail motion scoring"
```

---

### Task 3: FFmpeg 接入与 VideoReader

**Files:**
- Create: `cpp/cmake/FindFFmpeg.cmake`
- Create: `cpp/src/common/vc_error.h`
- Create: `cpp/src/common/vc_error.cpp`
- Create: `cpp/src/io/av_ptr.h`
- Create: `cpp/src/io/video_reader.h`
- Create: `cpp/src/io/video_reader.cpp`
- Create: `cpp/tests/test_media.h`
- Create: `cpp/tests/test_media.cpp`
- Modify: `cpp/CMakeLists.txt`（加 FFmpeg 查找与新源文件，见 Step 1 全文）
- Modify: `cpp/tests/CMakeLists.txt`（见 Step 1 全文）
- Test: `cpp/tests/video_reader_test.cpp`

**Interfaces:**
- Consumes: `LumaPlane`（Task 2）；测试里用 `makeLumaThumb` / `meanAbsDiff`（Task 2）
- Produces（`namespace vc`）：
  - `common/vc_error.h`：`class VcError : public std::runtime_error`；`std::string avErrorText(int errnum)`；`int checkAv(int ret, const std::string& what)`（`ret < 0` 时抛 `VcError("<what>: <ffmpeg 文字>")`）
  - `io/av_ptr.h`（仅 `src/io` 内部）：`namespace vc::av` 下的 `FramePtr`、`PacketPtr`、`CodecContextPtr`、`InputContextPtr`、`SwsPtr`、`BufferPtr` 及 `allocFrame()`、`allocPacket()`、`allocSws()`
  - `io/video_reader.h`：`enum class HwAccel { None, VideoToolbox, Auto };`、`HwAccel parseHwAccel(const std::string&)`（非法抛 `std::invalid_argument`）、`struct VideoInfo { int width; int height; double fps; double durationMs; std::string codecName; };`、`struct VideoFrame { int64_t index; double ptsMs; const AVFrame* av; LumaPlane luma() const; };`、`class VideoReader { VideoReader(const std::string& path, HwAccel); const VideoInfo& info() const; bool usingHardware() const; int64_t decodeErrors() const; const VideoFrame* next(); };`
  - 测试辅助 `tests/test_media.h`（`namespace vc::test`）：`tmpDir()`、`makeMedia(name, args)`、`basicVideo()`（2 秒 30fps 共 60 帧，文件名 `basic.mp4`）、`staticThenNoiseVideo()`（前 1 秒静止灰、后 1 秒随机噪声，共 60 帧，文件名 `static_then_noise.mp4`）、`tenBitVideo()`（1 秒 10-bit HEVC 共 30 帧）、`audioOnlyFile()`（只有音频）

`VideoReader` 行为约定：
- 输出帧一律是 CPU 内存里的 8-bit YUV：8-bit 平面格式或 NV12 直接给出；VideoToolbox 帧先下载；其他格式（如 10-bit）用 swscale 转成 NV12。所以 `luma()` 永远是 8-bit。
- `ptsMs = (best_effort_timestamp − 流 start_time) × time_base × 1000`；没有时间戳时按 `index × 1000 / fps`。
- 单个包或帧被解码器判为损坏（`AVERROR_INVALIDDATA`）时计数并继续；其他错误抛 `VcError`。
- `HwAccel::VideoToolbox` 不可用时构造函数抛 `VcError`；`HwAccel::Auto` 静默回退到软解。`usingHardware()` 只在真正收到硬件帧后才为 true。

- [ ] **Step 1: 写测试辅助、失败的测试，并更新 CMake**

`cpp/CMakeLists.txt` 全文替换为：

```cmake
cmake_minimum_required(VERSION 3.24)
project(video_compliance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(FFmpeg REQUIRED)

add_library(vc_core
  src/common/vc_error.cpp
  src/io/video_reader.cpp
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vc_core PUBLIC FFmpeg::FFmpeg)

option(VC_BUILD_TESTS "Build the GoogleTest suite" ON)
if(VC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

`cpp/cmake/FindFFmpeg.cmake`：

```cmake
# Finds libavformat, libavcodec, libavutil and libswscale without pkg-config.
# Defines the imported target FFmpeg::FFmpeg.

set(_ffmpeg_hints "")
if(APPLE)
  execute_process(
    COMMAND brew --prefix ffmpeg
    OUTPUT_VARIABLE _ffmpeg_brew_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_ffmpeg_brew_prefix)
    list(APPEND _ffmpeg_hints "${_ffmpeg_brew_prefix}")
  endif()
endif()

find_path(FFmpeg_INCLUDE_DIR libavcodec/avcodec.h HINTS ${_ffmpeg_hints} PATH_SUFFIXES include)

set(_ffmpeg_library_vars "")
foreach(_lib avformat avcodec swscale avutil)
  find_library(FFmpeg_${_lib}_LIBRARY NAMES ${_lib} HINTS ${_ffmpeg_hints} PATH_SUFFIXES lib)
  list(APPEND _ffmpeg_library_vars FFmpeg_${_lib}_LIBRARY)
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg REQUIRED_VARS FFmpeg_INCLUDE_DIR ${_ffmpeg_library_vars})

if(FFmpeg_FOUND AND NOT TARGET FFmpeg::FFmpeg)
  add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
  target_include_directories(FFmpeg::FFmpeg INTERFACE "${FFmpeg_INCLUDE_DIR}")
  target_link_libraries(FFmpeg::FFmpeg INTERFACE
    "${FFmpeg_avformat_LIBRARY}"
    "${FFmpeg_avcodec_LIBRARY}"
    "${FFmpeg_swscale_LIBRARY}"
    "${FFmpeg_avutil_LIBRARY}")
endif()
```

`cpp/tests/CMakeLists.txt` 全文替换为：

```cmake
find_package(GTest REQUIRED)
find_program(VC_FFMPEG_EXECUTABLE ffmpeg REQUIRED)

add_executable(vc_tests
  test_media.cpp
  frame_sampler_test.cpp
  luma_thumb_test.cpp
  video_reader_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_core GTest::gtest_main)
target_compile_definitions(vc_tests PRIVATE
  VC_FFMPEG="${VC_FFMPEG_EXECUTABLE}"
  VC_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")

include(GoogleTest)
gtest_discover_tests(vc_tests)
```

`cpp/tests/test_media.h`：

```cpp
#pragma once

#include <filesystem>
#include <string>

// Fixture media generated on demand with the ffmpeg CLI (no downloads).
namespace vc::test {

// <build>/tests/tmp, created on first use.
std::filesystem::path tmpDir();

// Runs `ffmpeg -v error -y <args> <tmpDir>/<name>` unless the file exists.
std::filesystem::path makeMedia(const std::string& name, const std::string& args);

// 2 s, 640x360 @ 30 fps H.264 moving test pattern (60 frames).
std::filesystem::path basicVideo();
// 1 s static grey then 1 s random noise, 640x360 @ 30 fps H.264 (60 frames).
std::filesystem::path staticThenNoiseVideo();
// 1 s, 640x360 @ 30 fps 10-bit HEVC (30 frames).
std::filesystem::path tenBitVideo();
// 1 s AAC audio, no video stream.
std::filesystem::path audioOnlyFile();

}  // namespace vc::test
```

`cpp/tests/test_media.cpp`：

```cpp
#include "test_media.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace vc::test {
namespace fs = std::filesystem;

fs::path tmpDir() {
  const fs::path dir = VC_TEST_TMP_DIR;
  fs::create_directories(dir);
  return dir;
}

fs::path makeMedia(const std::string& name, const std::string& args) {
  const fs::path output = tmpDir() / name;
  if (fs::exists(output)) return output;
  // ctest may run test processes in parallel: write a private file, then rename.
  const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-" + name);
  const std::string command = std::string("\"") + VC_FFMPEG + "\" -v error -y " + args + " \"" +
                              partial.string() + "\"";
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("ffmpeg failed: " + command);
  }
  fs::rename(partial, output);
  return output;
}

fs::path basicVideo() {
  return makeMedia("basic.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=2 -c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path staticThenNoiseVideo() {
  return makeMedia(
      "static_then_noise.mp4",
      "-f lavfi -i color=c=gray:s=640x360:r=30:d=1,format=yuv420p "
      "-f lavfi -i \"nullsrc=s=640x360:r=30:d=1,geq=lum='random(1)*255':cb=128:cr=128,"
      "format=yuv420p\" "
      "-filter_complex \"[0:v][1:v]concat=n=2:v=1:a=0[v]\" -map \"[v]\" "
      "-c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path tenBitVideo() {
  return makeMedia("ten_bit.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=1 -c:v libx265 -pix_fmt yuv420p10le "
                   "-x265-params log-level=error");
}

fs::path audioOnlyFile() {
  return makeMedia("audio_only.m4a", "-f lavfi -i sine=f=440:d=1 -c:a aac");
}

}  // namespace vc::test
```

`cpp/tests/video_reader_test.cpp`：

```cpp
#include "io/video_reader.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "common/vc_error.h"
#include "sampler/luma_thumb.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
}

namespace vc {
namespace {

TEST(VideoReader, ReadsEveryFrameWithTimestamps) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  EXPECT_EQ(reader.info().width, 640);
  EXPECT_EQ(reader.info().height, 360);
  EXPECT_NEAR(reader.info().fps, 30.0, 0.01);
  EXPECT_NEAR(reader.info().durationMs, 2000.0, 1.0);
  EXPECT_EQ(reader.info().codecName, "h264");

  int64_t count = 0;
  while (const VideoFrame* frame = reader.next()) {
    EXPECT_EQ(frame->index, count);
    EXPECT_NEAR(frame->ptsMs, static_cast<double>(count) * 1000.0 / 30.0, 1.0);
    const LumaPlane luma = frame->luma();
    EXPECT_EQ(luma.width, 640);
    EXPECT_EQ(luma.height, 360);
    EXPECT_GE(luma.stride, 640);
    ++count;
  }

  EXPECT_EQ(count, 60);
  EXPECT_EQ(reader.decodeErrors(), 0);
  EXPECT_FALSE(reader.usingHardware());
  EXPECT_EQ(reader.next(), nullptr);  // stays at end of stream
}

TEST(VideoReader, ConvertsTenBitVideoToEightBitNv12) {
  VideoReader reader(test::tenBitVideo().string(), HwAccel::None);

  int64_t count = 0;
  while (const VideoFrame* frame = reader.next()) {
    EXPECT_EQ(frame->av->format, AV_PIX_FMT_NV12);
    ++count;
  }

  EXPECT_EQ(count, 30);
}

TEST(VideoReader, ThrowsForMissingFile) {
  EXPECT_THROW(VideoReader((test::tmpDir() / "missing.mp4").string(), HwAccel::None), VcError);
}

TEST(VideoReader, ThrowsForFileWithoutVideo) {
  EXPECT_THROW(VideoReader(test::audioOnlyFile().string(), HwAccel::None), VcError);
}

TEST(VideoReader, VideoToolboxMatchesSoftwareDecode) {
  std::unique_ptr<VideoReader> hardware;
  try {
    hardware = std::make_unique<VideoReader>(test::basicVideo().string(), HwAccel::VideoToolbox);
  } catch (const VcError& e) {
    GTEST_SKIP() << e.what();
  }
  VideoReader software(test::basicVideo().string(), HwAccel::None);

  int64_t count = 0;
  while (const VideoFrame* sw = software.next()) {
    const LumaThumb expected = makeLumaThumb(sw->luma());
    const VideoFrame* hw = hardware->next();
    ASSERT_NE(hw, nullptr);
    EXPECT_EQ(hw->av->format, AV_PIX_FMT_NV12);
    EXPECT_LT(meanAbsDiff(makeLumaThumb(hw->luma()), expected), 0.01);
    ++count;
  }

  EXPECT_EQ(hardware->next(), nullptr);
  EXPECT_EQ(count, 60);
  EXPECT_TRUE(hardware->usingHardware());
}

TEST(VideoReader, ParsesHwAccelNames) {
  EXPECT_EQ(parseHwAccel("none"), HwAccel::None);
  EXPECT_EQ(parseHwAccel("videotoolbox"), HwAccel::VideoToolbox);
  EXPECT_EQ(parseHwAccel("auto"), HwAccel::Auto);
  EXPECT_THROW(parseHwAccel("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 构建，确认失败**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`
Expected: CMake 输出 `-- Found FFmpeg: /opt/homebrew/opt/ffmpeg/include`，随后 FAIL：`Cannot find source file: src/common/vc_error.cpp`

- [ ] **Step 3: 写实现**

`cpp/src/common/vc_error.h`：

```cpp
#pragma once

#include <stdexcept>
#include <string>

namespace vc {

// Failure while reading, decoding or writing media.
class VcError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Human-readable text for an FFmpeg AVERROR code.
std::string avErrorText(int errnum);

// Throws VcError("<what>: <ffmpeg message>") when ret < 0, otherwise returns ret.
int checkAv(int ret, const std::string& what);

}  // namespace vc
```

`cpp/src/common/vc_error.cpp`：

```cpp
#include "common/vc_error.h"

extern "C" {
#include <libavutil/error.h>
}

namespace vc {

std::string avErrorText(int errnum) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errnum, buffer, sizeof(buffer));
  return buffer;
}

int checkAv(int ret, const std::string& what) {
  if (ret < 0) {
    throw VcError(what + ": " + avErrorText(ret));
  }
  return ret;
}

}  // namespace vc
```

`cpp/src/io/av_ptr.h`：

```cpp
#pragma once

#include <memory>

#include "common/vc_error.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

// RAII owners for FFmpeg objects. Internal to src/io.
namespace vc::av {

struct FrameDeleter {
  void operator()(AVFrame* p) const { av_frame_free(&p); }
};
struct PacketDeleter {
  void operator()(AVPacket* p) const { av_packet_free(&p); }
};
struct CodecContextDeleter {
  void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
};
struct InputContextDeleter {
  void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
};
struct SwsDeleter {
  void operator()(SwsContext* p) const { sws_freeContext(p); }
};
struct BufferDeleter {
  void operator()(AVBufferRef* p) const { av_buffer_unref(&p); }
};

using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using InputContextPtr = std::unique_ptr<AVFormatContext, InputContextDeleter>;
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferDeleter>;

inline FramePtr allocFrame() {
  FramePtr frame(av_frame_alloc());
  if (!frame) throw VcError("av_frame_alloc failed");
  return frame;
}

inline PacketPtr allocPacket() {
  PacketPtr packet(av_packet_alloc());
  if (!packet) throw VcError("av_packet_alloc failed");
  return packet;
}

inline SwsPtr allocSws() {
  SwsPtr sws(sws_alloc_context());
  if (!sws) throw VcError("sws_alloc_context failed");
  return sws;
}

}  // namespace vc::av
```

`cpp/src/io/video_reader.h`：

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "common/luma_plane.h"

struct AVFrame;

namespace vc {

enum class HwAccel { None, VideoToolbox, Auto };

// Parses "none" | "videotoolbox" | "auto"; throws std::invalid_argument.
HwAccel parseHwAccel(const std::string& text);

struct VideoInfo {
  int width = 0;
  int height = 0;
  double fps = 0;         // average frame rate, 0 when unknown
  double durationMs = 0;  // container duration, 0 when unknown
  std::string codecName;
};

// A decoded frame in CPU memory, always 8-bit YUV (planar or NV12), so
// plane 0 is the luma plane.
struct VideoFrame {
  int64_t index = 0;  // 0-based, display order
  double ptsMs = 0;   // relative to the stream start
  const AVFrame* av = nullptr;

  LumaPlane luma() const;
};

// Sequential demux + decode of the best video stream in a file.
class VideoReader {
 public:
  // Throws VcError when the file cannot be opened, has no decodable video
  // stream, or VideoToolbox was requested explicitly and is unavailable.
  // HwAccel::Auto falls back to software decoding.
  VideoReader(const std::string& path, HwAccel hwAccel);
  ~VideoReader();
  VideoReader(const VideoReader&) = delete;
  VideoReader& operator=(const VideoReader&) = delete;

  const VideoInfo& info() const;
  // True once at least one frame came out of the hardware decoder.
  bool usingHardware() const;
  // Packets/frames the decoder rejected as corrupt; decoding continues past them.
  int64_t decodeErrors() const;

  // Next frame, or nullptr at end of stream. The pointer and its AVFrame stay
  // valid until the next call. Throws VcError on unrecoverable errors.
  const VideoFrame* next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/io/video_reader.cpp`：

```cpp
#include "io/video_reader.h"

#include <stdexcept>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/hwcontext.h>
}

namespace vc {
namespace {

// 8-bit YUV layouts whose plane 0 can be read directly as luma.
bool isDirect8BitYuv(int format) {
  switch (format) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_NV12:
      return true;
    default:
      return false;
  }
}

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

LumaPlane VideoFrame::luma() const {
  return LumaPlane{av->data[0], av->width, av->height, av->linesize[0]};
}

struct VideoReader::Impl {
  av::InputContextPtr input;
  av::CodecContextPtr decoder;
  av::BufferPtr hwDevice;
  av::SwsPtr sws;
  av::PacketPtr packet = av::allocPacket();
  av::FramePtr decoded = av::allocFrame();
  av::FramePtr downloaded = av::allocFrame();
  av::FramePtr converted = av::allocFrame();
  int streamIndex = -1;
  AVRational timeBase{0, 1};
  int64_t startPts = 0;
  VideoInfo info;
  VideoFrame current;
  int64_t nextIndex = 0;
  int64_t decodeErrors = 0;
  bool sawHardwareFrame = false;
  bool inputDrained = false;
  bool decoderDrained = false;

  void open(const std::string& path, HwAccel hwAccel);
  void attachHardware(HwAccel hwAccel);
  void feedDecoder();
  const VideoFrame* publish();
  const AVFrame* toCpu8Bit();
};

void VideoReader::Impl::open(const std::string& path, HwAccel hwAccel) {
  AVFormatContext* rawInput = nullptr;
  checkAv(avformat_open_input(&rawInput, path.c_str(), nullptr, nullptr), "open " + path);
  input.reset(rawInput);
  checkAv(avformat_find_stream_info(input.get(), nullptr), "read stream info of " + path);

  const AVCodec* codec = nullptr;
  streamIndex = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (streamIndex < 0 || codec == nullptr) {
    throw VcError("no decodable video stream in " + path);
  }
  const AVStream* stream = input->streams[streamIndex];
  timeBase = stream->time_base;
  startPts = stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time;

  decoder.reset(avcodec_alloc_context3(codec));
  if (!decoder) throw VcError("avcodec_alloc_context3 failed");
  checkAv(avcodec_parameters_to_context(decoder.get(), stream->codecpar),
          "copy codec parameters");
  decoder->pkt_timebase = timeBase;
  decoder->thread_count = 0;  // FFmpeg picks the thread count
  attachHardware(hwAccel);
  checkAv(avcodec_open2(decoder.get(), codec, nullptr), "open decoder " + std::string(codec->name));

  info.width = decoder->width;
  info.height = decoder->height;
  const AVRational rate = stream->avg_frame_rate;
  info.fps = rate.num > 0 && rate.den > 0 ? av_q2d(rate) : 0.0;
  info.durationMs = input->duration > 0 ? static_cast<double>(input->duration) / 1000.0 : 0.0;
  info.codecName = codec->name;
}

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

void VideoReader::Impl::feedDecoder() {
  while (true) {
    const int ret = av_read_frame(input.get(), packet.get());
    if (ret == AVERROR_EOF) {
      checkAv(avcodec_send_packet(decoder.get(), nullptr), "flush decoder");
      inputDrained = true;
      return;
    }
    checkAv(ret, "read packet");
    if (packet->stream_index != streamIndex) {
      av_packet_unref(packet.get());
      continue;
    }
    const int sent = avcodec_send_packet(decoder.get(), packet.get());
    av_packet_unref(packet.get());
    if (sent == AVERROR_INVALIDDATA) {
      ++decodeErrors;
      return;
    }
    checkAv(sent, "send packet");
    return;
  }
}

const VideoFrame* VideoReader::Impl::publish() {
  const int64_t ts = decoded->best_effort_timestamp;
  current.index = nextIndex++;
  if (ts == AV_NOPTS_VALUE) {
    current.ptsMs = info.fps > 0 ? static_cast<double>(current.index) * 1000.0 / info.fps : 0.0;
  } else {
    current.ptsMs = static_cast<double>(ts - startPts) * av_q2d(timeBase) * 1000.0;
  }
  current.av = toCpu8Bit();
  return &current;
}

const AVFrame* VideoReader::Impl::toCpu8Bit() {
  const AVFrame* frame = decoded.get();
  if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
    sawHardwareFrame = true;
    av_frame_unref(downloaded.get());
    checkAv(av_hwframe_transfer_data(downloaded.get(), frame, 0), "download hardware frame");
    checkAv(av_frame_copy_props(downloaded.get(), frame), "copy frame properties");
    frame = downloaded.get();
  }
  if (isDirect8BitYuv(frame->format)) return frame;

  if (!sws) sws = av::allocSws();
  av_frame_unref(converted.get());
  converted->format = AV_PIX_FMT_NV12;
  converted->width = frame->width;
  converted->height = frame->height;
  checkAv(sws_scale_frame(sws.get(), converted.get(), frame), "convert frame to NV12");
  checkAv(av_frame_copy_props(converted.get(), frame), "copy frame properties");
  return converted.get();
}

VideoReader::VideoReader(const std::string& path, HwAccel hwAccel)
    : impl_(std::make_unique<Impl>()) {
  impl_->open(path, hwAccel);
}

VideoReader::~VideoReader() = default;

const VideoInfo& VideoReader::info() const { return impl_->info; }

bool VideoReader::usingHardware() const { return impl_->sawHardwareFrame; }

int64_t VideoReader::decodeErrors() const { return impl_->decodeErrors; }

const VideoFrame* VideoReader::next() {
  Impl& s = *impl_;
  while (!s.decoderDrained) {
    const int ret = avcodec_receive_frame(s.decoder.get(), s.decoded.get());
    if (ret == AVERROR(EAGAIN)) {
      if (s.inputDrained) {
        s.decoderDrained = true;
        break;
      }
      s.feedDecoder();
      continue;
    }
    if (ret == AVERROR_EOF) {
      s.decoderDrained = true;
      break;
    }
    if (ret == AVERROR_INVALIDDATA) {
      ++s.decodeErrors;
      continue;
    }
    checkAv(ret, "decode frame");
    return s.publish();
  }
  return nullptr;
}

}  // namespace vc
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*'`
Expected: 构建无警告；`[  PASSED  ] 6 tests.`（在没有 VideoToolbox 的机器上 `VideoToolboxMatchesSoftwareDecode` 显示 SKIPPED，其余 5 个通过）

- [ ] **Step 5: 确认前两个任务的测试仍然通过**

Run: `./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'`
Expected: `[  PASSED  ] 14 tests.`

- [ ] **Step 6: 提交**

```bash
git add cpp/CMakeLists.txt cpp/cmake/FindFFmpeg.cmake cpp/src/common/vc_error.h cpp/src/common/vc_error.cpp cpp/src/io/av_ptr.h cpp/src/io/video_reader.h cpp/src/io/video_reader.cpp cpp/tests/CMakeLists.txt cpp/tests/test_media.h cpp/tests/test_media.cpp cpp/tests/video_reader_test.cpp
git commit -m "feat: add FFmpeg VideoReader with VideoToolbox and 10-bit support"
```

---

### Task 4: JpegWriter

**Files:**
- Create: `cpp/src/io/jpeg_writer.h`
- Create: `cpp/src/io/jpeg_writer.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 增加 `src/io/jpeg_writer.cpp`）
- Modify: `cpp/tests/CMakeLists.txt`（增加 `jpeg_writer_test.cpp`）
- Test: `cpp/tests/jpeg_writer_test.cpp`

**Interfaces:**
- Consumes: `VcError`、`checkAv`、`vc::av::*`（Task 3）；测试用 `VideoReader`、`test::basicVideo()`、`test::tenBitVideo()`、`test::tmpDir()`（Task 3）
- Produces（`io/jpeg_writer.h`）：`inline constexpr int kBestJpegQscale = 2; inline constexpr int kWorstJpegQscale = 31;`、`class JpegWriter { explicit JpegWriter(int qscale = kBestJpegQscale); void write(const AVFrame& frame, const std::filesystem::path& path); };`，qscale 越界抛 `std::invalid_argument`，写失败抛 `VcError`。

实现要点：用 `sws_scale_frame` 把任意软件格式转成全范围 `YUV420P`（`color_range = AVCOL_RANGE_JPEG`），交给 MJPEG 编码器；`AV_CODEC_FLAG_QSCALE` 模式下量化值**取自帧的 `quality` 字段**，所以每帧都要设 `quality = global_quality`，否则画质参数不生效。

- [ ] **Step 1: 写失败的测试，并登记到 CMake**

`cpp/CMakeLists.txt` 中 `vc_core` 改为：

```cmake
add_library(vc_core
  src/common/vc_error.cpp
  src/io/jpeg_writer.cpp
  src/io/video_reader.cpp
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
)
```

`cpp/tests/CMakeLists.txt` 中 `vc_tests` 改为：

```cmake
add_executable(vc_tests
  test_media.cpp
  frame_sampler_test.cpp
  jpeg_writer_test.cpp
  luma_thumb_test.cpp
  video_reader_test.cpp
)
```

`cpp/tests/jpeg_writer_test.cpp`：

```cpp
#include "io/jpeg_writer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "io/video_reader.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

std::vector<unsigned char> readBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

TEST(JpegWriter, WritesJpegThatDecodesToSameSize) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path path = test::tmpDir() / "jpeg_writer_basic.jpg";

  JpegWriter().write(*frame->av, path);

  const std::vector<unsigned char> bytes = readBytes(path);
  ASSERT_GT(bytes.size(), 4u);
  EXPECT_EQ(bytes[0], 0xFF);
  EXPECT_EQ(bytes[1], 0xD8);  // SOI marker
  EXPECT_EQ(bytes[bytes.size() - 2], 0xFF);
  EXPECT_EQ(bytes[bytes.size() - 1], 0xD9);  // EOI marker
  VideoReader decoded(path.string(), HwAccel::None);
  EXPECT_EQ(decoded.info().width, 640);
  EXPECT_EQ(decoded.info().height, 360);
  EXPECT_NE(decoded.next(), nullptr);
}

TEST(JpegWriter, LowerQscaleGivesLargerFile) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path best = test::tmpDir() / "jpeg_writer_q2.jpg";
  const fs::path worse = test::tmpDir() / "jpeg_writer_q20.jpg";

  JpegWriter(2).write(*frame->av, best);
  JpegWriter(20).write(*frame->av, worse);

  EXPECT_GT(fs::file_size(best), 2 * fs::file_size(worse));
}

TEST(JpegWriter, WritesNv12FramesFromTenBitVideo) {
  VideoReader reader(test::tenBitVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path path = test::tmpDir() / "jpeg_writer_ten_bit.jpg";

  JpegWriter().write(*frame->av, path);

  VideoReader decoded(path.string(), HwAccel::None);
  EXPECT_EQ(decoded.info().width, 640);
  EXPECT_EQ(decoded.info().height, 360);
}

TEST(JpegWriter, RejectsQscaleOutsideRange) {
  EXPECT_THROW(JpegWriter(1), std::invalid_argument);
  EXPECT_THROW(JpegWriter(32), std::invalid_argument);
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 构建，确认失败**

Run: `cmake --build build/cpp -j 4`
Expected: FAIL，`Cannot find source file: src/io/jpeg_writer.cpp`

- [ ] **Step 3: 写实现**

`cpp/src/io/jpeg_writer.h`：

```cpp
#pragma once

#include <filesystem>
#include <memory>

struct AVFrame;

namespace vc {

inline constexpr int kBestJpegQscale = 2;
inline constexpr int kWorstJpegQscale = 31;

// Encodes decoded frames as baseline JPEG files with FFmpeg's MJPEG encoder.
class JpegWriter {
 public:
  // qscale is FFmpeg's MJPEG quantiser: 2 (best, ~quality 95) .. 31 (worst).
  // Throws std::invalid_argument outside that range.
  explicit JpegWriter(int qscale = kBestJpegQscale);
  ~JpegWriter();
  JpegWriter(const JpegWriter&) = delete;
  JpegWriter& operator=(const JpegWriter&) = delete;

  // Writes `frame` (any software pixel format) to `path`. Throws VcError.
  void write(const AVFrame& frame, const std::filesystem::path& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/io/jpeg_writer.cpp`：

```cpp
#include "io/jpeg_writer.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {

struct JpegWriter::Impl {
  int qscale = kBestJpegQscale;
  av::CodecContextPtr encoder;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr yuv = av::allocFrame();
  av::PacketPtr packet = av::allocPacket();

  void ensureEncoder(int width, int height);
};

void JpegWriter::Impl::ensureEncoder(int width, int height) {
  if (encoder && encoder->width == width && encoder->height == height) return;
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  if (codec == nullptr) throw VcError("MJPEG encoder not available");
  encoder.reset(avcodec_alloc_context3(codec));
  if (!encoder) throw VcError("avcodec_alloc_context3 failed");
  encoder->width = width;
  encoder->height = height;
  encoder->pix_fmt = AV_PIX_FMT_YUV420P;
  encoder->color_range = AVCOL_RANGE_JPEG;
  encoder->time_base = AVRational{1, 25};
  encoder->flags |= AV_CODEC_FLAG_QSCALE;
  encoder->global_quality = qscale * FF_QP2LAMBDA;
  checkAv(avcodec_open2(encoder.get(), codec, nullptr), "open MJPEG encoder");
}

JpegWriter::JpegWriter(int qscale) : impl_(std::make_unique<Impl>()) {
  if (qscale < kBestJpegQscale || qscale > kWorstJpegQscale) {
    throw std::invalid_argument("JPEG qscale must be in [2, 31], got " + std::to_string(qscale));
  }
  impl_->qscale = qscale;
}

JpegWriter::~JpegWriter() = default;

void JpegWriter::write(const AVFrame& frame, const std::filesystem::path& path) {
  Impl& s = *impl_;
  s.ensureEncoder(frame.width, frame.height);

  av_frame_unref(s.yuv.get());
  s.yuv->format = AV_PIX_FMT_YUV420P;
  s.yuv->width = frame.width;
  s.yuv->height = frame.height;
  s.yuv->color_range = AVCOL_RANGE_JPEG;
  checkAv(sws_scale_frame(s.sws.get(), s.yuv.get(), &frame), "convert frame for JPEG");
  // MJPEG with AV_CODEC_FLAG_QSCALE takes its quantiser from the frame.
  s.yuv->quality = s.encoder->global_quality;

  checkAv(avcodec_send_frame(s.encoder.get(), s.yuv.get()), "encode JPEG");
  checkAv(avcodec_receive_packet(s.encoder.get(), s.packet.get()), "receive JPEG packet");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(s.packet->data), s.packet->size);
  av_packet_unref(s.packet.get());
  if (!out) throw VcError("write " + path.string() + " failed");
}

}  // namespace vc
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='JpegWriter.*'`
Expected: 构建无警告；`[  PASSED  ] 4 tests.`（参考值：q2 约 22KB，q20 约 7KB）

- [ ] **Step 5: 提交**

```bash
git add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/src/io/jpeg_writer.h cpp/src/io/jpeg_writer.cpp cpp/tests/jpeg_writer_test.cpp
git commit -m "feat: add MJPEG-based JpegWriter"
```

---

### Task 5: vc_extract 命令行解析

**Files:**
- Create: `cpp/apps/vc_extract/extract_options.h`
- Create: `cpp/apps/vc_extract/extract_cli.h`
- Create: `cpp/apps/vc_extract/extract_cli.cpp`
- Modify: `cpp/CMakeLists.txt`（新增 `vc_extract_lib`）
- Modify: `cpp/tests/CMakeLists.txt`（增加 `extract_cli_test.cpp`，链接 `vc_extract_lib`）
- Test: `cpp/tests/extract_cli_test.cpp`

**Interfaces:**
- Consumes: `SamplerConfig`、`ClipWindow`、`FrameSampler`（Task 1）；`HwAccel`、`parseHwAccel`（Task 3）；`kBestJpegQscale`、`kWorstJpegQscale`（Task 4）
- Produces：
  - `vc_extract/extract_options.h`：`struct ExtractOptions { std::filesystem::path input; std::filesystem::path outRoot; SamplerConfig sampler; HwAccel hwAccel = HwAccel::Auto; int jpegQscale = kBestJpegQscale; bool overwrite = false; };`
  - `vc_extract/extract_cli.h`：`class UsageError : public std::runtime_error`；`ExtractOptions parseExtractArgs(const std::vector<std::string>& args);`（不含程序名，非法抛 `UsageError`）；`std::string extractUsage();`
  - CMake 目标 `vc_extract_lib`（`PUBLIC` 包含目录 `cpp/apps`，所以头文件写作 `"vc_extract/…"`）

命令行（`--clip` 的单位是秒，内部转毫秒）：

```
vc_extract --input <video> --out <dir> [--fps 5] [--max-fps 15] [--motion-threshold 0.05]
           [--dedup <x>] [--clip <startSec>,<durSec>] [--hwaccel auto|none|videotoolbox]
           [--jpeg-q 2] [--overwrite]
```

- [ ] **Step 1: 写失败的测试，并登记到 CMake**

`cpp/CMakeLists.txt` 在 `target_link_libraries(vc_core …)` 之后、`option(VC_BUILD_TESTS …)` 之前加入：

```cmake
add_library(vc_extract_lib
  apps/vc_extract/extract_cli.cpp
)
target_include_directories(vc_extract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_extract_lib PUBLIC vc_core)
```

`cpp/tests/CMakeLists.txt` 全文替换为：

```cmake
find_package(GTest REQUIRED)
find_program(VC_FFMPEG_EXECUTABLE ffmpeg REQUIRED)

add_executable(vc_tests
  test_media.cpp
  extract_cli_test.cpp
  frame_sampler_test.cpp
  jpeg_writer_test.cpp
  luma_thumb_test.cpp
  video_reader_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_extract_lib vc_core GTest::gtest_main)
target_compile_definitions(vc_tests PRIVATE
  VC_FFMPEG="${VC_FFMPEG_EXECUTABLE}"
  VC_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")

include(GoogleTest)
gtest_discover_tests(vc_tests)
```

`cpp/tests/extract_cli_test.cpp`：

```cpp
#include "vc_extract/extract_cli.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace vc {
namespace {

using Args = std::vector<std::string>;

TEST(ExtractCli, AppliesDefaults) {
  const Args args = {"--input", "trip.mp4", "--out", "frames"};

  const ExtractOptions options = parseExtractArgs(args);

  EXPECT_EQ(options.input, "trip.mp4");
  EXPECT_EQ(options.outRoot, "frames");
  EXPECT_DOUBLE_EQ(options.sampler.baseFps, 5.0);
  EXPECT_DOUBLE_EQ(options.sampler.maxFps, 15.0);
  EXPECT_DOUBLE_EQ(options.sampler.motionThreshold, 0.05);
  EXPECT_FALSE(options.sampler.dedupThreshold.has_value());
  EXPECT_FALSE(options.sampler.clip.has_value());
  EXPECT_EQ(options.hwAccel, HwAccel::Auto);
  EXPECT_EQ(options.jpegQscale, 2);
  EXPECT_FALSE(options.overwrite);
}

TEST(ExtractCli, ParsesEveryOption) {
  const Args args = {"--input", "a.mp4",  "--out",     "o",    "--fps",
                     "10",      "--max-fps", "30",     "--motion-threshold",
                     "0.1",     "--dedup", "0.01",     "--clip", "1.5,2",
                     "--hwaccel", "none",  "--jpeg-q", "5",    "--overwrite"};

  const ExtractOptions options = parseExtractArgs(args);

  EXPECT_DOUBLE_EQ(options.sampler.baseFps, 10.0);
  EXPECT_DOUBLE_EQ(options.sampler.maxFps, 30.0);
  EXPECT_DOUBLE_EQ(options.sampler.motionThreshold, 0.1);
  ASSERT_TRUE(options.sampler.dedupThreshold.has_value());
  EXPECT_DOUBLE_EQ(*options.sampler.dedupThreshold, 0.01);
  ASSERT_TRUE(options.sampler.clip.has_value());
  EXPECT_DOUBLE_EQ(options.sampler.clip->startMs, 1500.0);
  EXPECT_DOUBLE_EQ(options.sampler.clip->durationMs, 2000.0);
  EXPECT_EQ(options.hwAccel, HwAccel::None);
  EXPECT_EQ(options.jpegQscale, 5);
  EXPECT_TRUE(options.overwrite);
}

TEST(ExtractCli, RejectsInvalidArguments) {
  const std::vector<Args> invalid = {
      {"--out", "o"},                                           // missing --input
      {"--input", "a.mp4"},                                     // missing --out
      {"--input", "a.mp4", "--out", "o", "--fps", "4"},         // below 5 fps
      {"--input", "a.mp4", "--out", "o", "--fps", "5x"},        // not a number
      {"--input", "a.mp4", "--out", "o", "--max-fps", "4"},     // below base fps
      {"--input", "a.mp4", "--out", "o", "--clip", "1.5"},      // no duration
      {"--input", "a.mp4", "--out", "o", "--clip", "1,0"},      // empty window
      {"--input", "a.mp4", "--out", "o", "--hwaccel", "cuda"},  // unsupported here
      {"--input", "a.mp4", "--out", "o", "--jpeg-q", "1"},      // out of range
      {"--input", "a.mp4", "--out", "o", "--bogus", "1"},       // unknown option
      {"--input"},                                              // missing value
  };
  for (const Args& args : invalid) {
    EXPECT_THROW(parseExtractArgs(args), UsageError) << ::testing::PrintToString(args);
  }
}

TEST(ExtractCli, UsageMentionsEveryOption) {
  const std::string usage = extractUsage();
  for (const char* option : {"--input", "--out", "--fps", "--max-fps", "--motion-threshold",
                             "--dedup", "--clip", "--hwaccel", "--jpeg-q", "--overwrite"}) {
    EXPECT_NE(usage.find(option), std::string::npos) << option;
  }
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 构建，确认失败**

Run: `cmake --build build/cpp -j 4`
Expected: FAIL，`Cannot find source file: apps/vc_extract/extract_cli.cpp`

- [ ] **Step 3: 写实现**

`cpp/apps/vc_extract/extract_options.h`：

```cpp
#pragma once

#include <filesystem>

#include "io/jpeg_writer.h"
#include "io/video_reader.h"
#include "sampler/frame_sampler.h"

namespace vc {

struct ExtractOptions {
  std::filesystem::path input;    // video file
  std::filesystem::path outRoot;  // frames go to outRoot / <input stem>
  SamplerConfig sampler;
  HwAccel hwAccel = HwAccel::Auto;
  int jpegQscale = kBestJpegQscale;
  bool overwrite = false;
};

}  // namespace vc
```

`cpp/apps/vc_extract/extract_cli.h`：

```cpp
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "vc_extract/extract_options.h"

namespace vc {

// Invalid command line.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Parses the arguments after the program name. Throws UsageError.
ExtractOptions parseExtractArgs(const std::vector<std::string>& args);

std::string extractUsage();

}  // namespace vc
```

`cpp/apps/vc_extract/extract_cli.cpp`：

```cpp
#include "vc_extract/extract_cli.h"

#include <cstddef>

namespace vc {
namespace {

double parseDouble(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects a number, got '" + text + "'");
}

int parseInt(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects an integer, got '" + text + "'");
}

// "<startSec>,<durationSec>"
ClipWindow parseClip(const std::string& text) {
  const std::size_t comma = text.find(',');
  if (comma == std::string::npos) {
    throw UsageError("--clip expects <startSec>,<durationSec>, got '" + text + "'");
  }
  const double start = parseDouble("--clip", text.substr(0, comma));
  const double duration = parseDouble("--clip", text.substr(comma + 1));
  return ClipWindow{start * 1000.0, duration * 1000.0};
}

}  // namespace

ExtractOptions parseExtractArgs(const std::vector<std::string>& args) {
  ExtractOptions options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& option = args[i];
    if (option == "--overwrite") {
      options.overwrite = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      throw UsageError(option.rfind("--", 0) == 0 ? option + " needs a value"
                                                  : "unexpected argument '" + option + "'");
    }
    const std::string& value = args[++i];
    if (option == "--input") {
      options.input = value;
    } else if (option == "--out") {
      options.outRoot = value;
    } else if (option == "--fps") {
      options.sampler.baseFps = parseDouble(option, value);
    } else if (option == "--max-fps") {
      options.sampler.maxFps = parseDouble(option, value);
    } else if (option == "--motion-threshold") {
      options.sampler.motionThreshold = parseDouble(option, value);
    } else if (option == "--dedup") {
      options.sampler.dedupThreshold = parseDouble(option, value);
    } else if (option == "--clip") {
      options.sampler.clip = parseClip(value);
    } else if (option == "--hwaccel") {
      try {
        options.hwAccel = parseHwAccel(value);
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (option == "--jpeg-q") {
      options.jpegQscale = parseInt(option, value);
    } else {
      throw UsageError("unknown option '" + option + "'");
    }
  }

  if (options.input.empty()) throw UsageError("--input is required");
  if (options.outRoot.empty()) throw UsageError("--out is required");
  if (options.jpegQscale < kBestJpegQscale || options.jpegQscale > kWorstJpegQscale) {
    throw UsageError("--jpeg-q must be in [2, 31]");
  }
  try {
    FrameSampler validate(options.sampler);
  } catch (const std::invalid_argument& e) {
    throw UsageError(e.what());
  }
  return options;
}

std::string extractUsage() {
  return R"(Usage: vc_extract --input <video> --out <dir> [options]

Extracts JPEG frames into <dir>/<video stem>/ and writes frames.csv there.

Options:
  --fps <n>               base sampling rate, >= 5 (default 5)
  --max-fps <n>           rate while motion is high (default 15)
  --motion-threshold <x>  mean abs luma diff in [0,1] that counts as motion (default 0.05)
  --dedup <x>             skip frames whose diff to the last kept frame is below x
  --clip <start>,<dur>    keep every frame in [start, start+dur) seconds, nothing else
  --hwaccel <mode>        auto | none | videotoolbox (default auto)
  --jpeg-q <n>            MJPEG qscale 2 (best) .. 31 (default 2)
  --overwrite             replace an existing output directory
  -h, --help              show this help

Exit codes: 0 ok, 1 processing error, 2 invalid arguments, 3 output exists.
)";
}

}  // namespace vc
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*'`
Expected: 构建无警告；`[  PASSED  ] 4 tests.`

- [ ] **Step 5: 提交**

```bash
git add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/apps/vc_extract/extract_options.h cpp/apps/vc_extract/extract_cli.h cpp/apps/vc_extract/extract_cli.cpp cpp/tests/extract_cli_test.cpp
git commit -m "feat: add vc_extract command-line parsing"
```

---

### Task 6: 抽帧主流程、vc_extract 可执行文件与说明文档

**Files:**
- Create: `cpp/apps/vc_extract/extract_job.h`
- Create: `cpp/apps/vc_extract/extract_job.cpp`
- Create: `cpp/apps/vc_extract/main.cpp`
- Create: `cpp/README.md`
- Modify: `cpp/CMakeLists.txt`（`vc_extract_lib` 加 `extract_job.cpp`，新增 `vc_extract` 可执行文件）
- Modify: `cpp/tests/CMakeLists.txt`（增加 `extract_job_test.cpp`）
- Test: `cpp/tests/extract_job_test.cpp`

**Interfaces:**
- Consumes: `ExtractOptions`、`UsageError`、`parseExtractArgs`、`extractUsage`（Task 5）；`VideoReader`、`VideoFrame`（Task 3）；`JpegWriter`（Task 4）；`FrameSampler`、`toString(SampleReason)`（Task 1）；`makeLumaThumb`、`meanAbsDiff`、`kThumbWidth`、`kThumbHeight`（Task 2）
- Produces（`vc_extract/extract_job.h`）：
  - `class OutputExistsError : public VcError`
  - `struct ExtractStats { std::filesystem::path outputDir; int64_t decodedFrames, keptFrames, keptBase, keptMotion, keptClip, decodeErrors; bool usedHardware; double elapsedSec; };`
  - `ExtractStats runExtract(const ExtractOptions& options);`
  - 可执行文件 `build/cpp/vc_extract`

`runExtract` 流程约定：
1. **先打开输入**（`VideoReader`）再动输出目录，这样输入有问题时不会删掉已有输出。
2. 输出目录 `outRoot / input.stem()`：已存在且未加 `--overwrite` → 抛 `OutputExistsError`；加了就整个删除后重建。
3. 对每个解码帧：算 160×90 缩略图（视频比这还小时取视频尺寸）；`motion` = 与上一解码帧的差（首帧为 0）；`diffToLastKept` = 与上一保留帧的差（还没有保留帧时为 1.0）；交给 `FrameSampler`。
4. 保留的帧写 `<帧号:06d>.jpg`，并往 `frames.csv` 追加一行。
5. 片段模式下，第一帧超出窗口时立刻停止解码。

- [ ] **Step 1: 写失败的测试，并登记到 CMake**

`cpp/CMakeLists.txt` 中 `vc_extract_lib` 一段改为（并在其后新增可执行文件）：

```cmake
add_library(vc_extract_lib
  apps/vc_extract/extract_cli.cpp
  apps/vc_extract/extract_job.cpp
)
target_include_directories(vc_extract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_extract_lib PUBLIC vc_core)

add_executable(vc_extract apps/vc_extract/main.cpp)
target_link_libraries(vc_extract PRIVATE vc_extract_lib)
```

`cpp/tests/CMakeLists.txt` 中 `vc_tests` 改为：

```cmake
add_executable(vc_tests
  test_media.cpp
  extract_cli_test.cpp
  extract_job_test.cpp
  frame_sampler_test.cpp
  jpeg_writer_test.cpp
  luma_thumb_test.cpp
  video_reader_test.cpp
)
```

`cpp/tests/extract_job_test.cpp`：

```cpp
#include "vc_extract/extract_job.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

// A clean output root per test.
fs::path freshOutRoot(const std::string& name) {
  const fs::path root = test::tmpDir() / "extract" / name;
  fs::remove_all(root);
  return root;
}

ExtractOptions optionsFor(const fs::path& input, const fs::path& outRoot) {
  ExtractOptions options;
  options.input = input;
  options.outRoot = outRoot;
  options.hwAccel = HwAccel::None;
  return options;
}

std::vector<std::string> readLines(const fs::path& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

// Frame numbers of the frames.csv rows (header skipped) with the given reason.
std::vector<int> framesWithReason(const std::vector<std::string>& lines,
                                  const std::string& reason) {
  std::vector<int> frames;
  for (size_t i = 1; i < lines.size(); ++i) {
    std::stringstream row(lines[i]);
    std::string video, frame, pts, rowReason;
    std::getline(row, video, ',');
    std::getline(row, frame, ',');
    std::getline(row, pts, ',');
    std::getline(row, rowReason, ',');
    if (rowReason == reason) frames.push_back(std::stoi(frame));
  }
  return frames;
}

TEST(ExtractJob, SamplesAtBaseRateAndWritesCsv) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("base_rate"));
  options.sampler.motionThreshold = 1.0;  // nothing counts as motion

  const ExtractStats stats = runExtract(options);

  EXPECT_EQ(stats.outputDir, options.outRoot / "basic");
  EXPECT_EQ(stats.decodedFrames, 60);
  EXPECT_EQ(stats.keptFrames, 10);
  EXPECT_EQ(stats.keptBase, 10);
  EXPECT_EQ(stats.decodeErrors, 0);
  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  ASSERT_EQ(lines.size(), 11u);
  EXPECT_EQ(lines[0], "video,frame,pts_ms,reason,motion_score");
  EXPECT_EQ(lines[1], "basic.mp4,0,0.0,base,0.0000");
  EXPECT_EQ(framesWithReason(lines, "base"),
            (std::vector<int>{0, 6, 12, 18, 24, 30, 36, 42, 48, 54}));
  EXPECT_TRUE(fs::exists(stats.outputDir / "000000.jpg"));
  EXPECT_TRUE(fs::exists(stats.outputDir / "000054.jpg"));
  EXPECT_FALSE(fs::exists(stats.outputDir / "000001.jpg"));
}

TEST(ExtractJob, SamplesFasterWhileMotionIsHigh) {
  ExtractOptions options = optionsFor(test::staticThenNoiseVideo(), freshOutRoot("motion"));
  options.sampler.motionThreshold = 0.03;  // static half scores 0, noise half ~0.08

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  EXPECT_EQ(framesWithReason(lines, "base"), (std::vector<int>{0, 6, 12, 18, 24, 30}));
  std::vector<int> motion;
  for (int i = 32; i < 60; i += 2) motion.push_back(i);
  EXPECT_EQ(framesWithReason(lines, "motion"), motion);
  EXPECT_EQ(stats.keptMotion, 14);
}

TEST(ExtractJob, DedupDropsStaticFrames) {
  ExtractOptions options = optionsFor(test::staticThenNoiseVideo(), freshOutRoot("dedup"));
  options.sampler.motionThreshold = 1.0;
  options.sampler.dedupThreshold = 0.01;

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  EXPECT_EQ(framesWithReason(lines, "base"), (std::vector<int>{0, 30, 36, 42, 48, 54}));
  EXPECT_EQ(stats.keptFrames, 6);
}

TEST(ExtractJob, ClipExportsEveryFrameInWindow) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("clip"));
  options.sampler.clip = ClipWindow{500.0, 500.0};

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  std::vector<int> expected;
  for (int i = 15; i < 30; ++i) expected.push_back(i);
  EXPECT_EQ(framesWithReason(lines, "clip"), expected);
  EXPECT_EQ(stats.keptClip, 15);
  EXPECT_EQ(stats.decodedFrames, 31);  // stops at the first frame past the window
}

TEST(ExtractJob, RefusesExistingOutputUnlessOverwrite) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("overwrite"));
  const ExtractStats first = runExtract(options);
  std::ofstream(first.outputDir / "stale.txt") << "old";

  EXPECT_THROW(runExtract(options), OutputExistsError);
  EXPECT_TRUE(fs::exists(first.outputDir / "stale.txt"));

  options.overwrite = true;
  runExtract(options);
  EXPECT_FALSE(fs::exists(first.outputDir / "stale.txt"));
}

TEST(ExtractJob, BadInputLeavesExistingOutputAlone) {
  const fs::path outRoot = freshOutRoot("bad_input");
  fs::create_directories(outRoot / "missing");
  std::ofstream(outRoot / "missing" / "keep.txt") << "keep";
  ExtractOptions options = optionsFor(test::tmpDir() / "missing.mp4", outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), VcError);

  EXPECT_TRUE(fs::exists(outRoot / "missing" / "keep.txt"));
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 构建，确认失败**

Run: `cmake --build build/cpp -j 4`
Expected: FAIL，`Cannot find source file: apps/vc_extract/extract_job.cpp`

- [ ] **Step 3: 写实现**

`cpp/apps/vc_extract/extract_job.h`：

```cpp
#pragma once

#include <cstdint>
#include <filesystem>

#include "common/vc_error.h"
#include "vc_extract/extract_options.h"

namespace vc {

// The output directory already exists and overwrite was not requested.
class OutputExistsError : public VcError {
 public:
  using VcError::VcError;
};

struct ExtractStats {
  std::filesystem::path outputDir;
  int64_t decodedFrames = 0;
  int64_t keptFrames = 0;
  int64_t keptBase = 0;
  int64_t keptMotion = 0;
  int64_t keptClip = 0;
  int64_t decodeErrors = 0;
  bool usedHardware = false;
  double elapsedSec = 0;
};

// Decodes options.input and writes <outRoot>/<stem>/<frame:06d>.jpg plus
// frames.csv. The input is opened before the output directory is touched, so a
// bad input never deletes existing output. Throws OutputExistsError or VcError.
ExtractStats runExtract(const ExtractOptions& options);

}  // namespace vc
```

`cpp/apps/vc_extract/extract_job.cpp`：

```cpp
#include "vc_extract/extract_job.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "io/jpeg_writer.h"
#include "io/video_reader.h"
#include "sampler/frame_sampler.h"
#include "sampler/luma_thumb.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

void prepareOutputDir(const fs::path& dir, bool overwrite) {
  if (fs::exists(dir)) {
    if (!overwrite) {
      throw OutputExistsError("output directory exists: " + dir.string() +
                              " (pass --overwrite to replace it)");
    }
    fs::remove_all(dir);
  }
  fs::create_directories(dir);
}

std::string frameFileName(int64_t index) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06lld.jpg", static_cast<long long>(index));
  return name;
}

std::string csvRow(const std::string& video, const VideoFrame& frame, SampleReason reason,
                   double motion) {
  char numbers[96];
  std::snprintf(numbers, sizeof(numbers), "%lld,%.1f,%s,%.4f",
                static_cast<long long>(frame.index), frame.ptsMs, toString(reason), motion);
  return video + "," + numbers + "\n";
}

void countReason(ExtractStats& stats, SampleReason reason) {
  ++stats.keptFrames;
  if (reason == SampleReason::Base) ++stats.keptBase;
  if (reason == SampleReason::Motion) ++stats.keptMotion;
  if (reason == SampleReason::Clip) ++stats.keptClip;
}

}  // namespace

ExtractStats runExtract(const ExtractOptions& options) {
  const auto started = std::chrono::steady_clock::now();
  VideoReader reader(options.input.string(), options.hwAccel);
  FrameSampler sampler(options.sampler);
  JpegWriter jpeg(options.jpegQscale);

  ExtractStats stats;
  stats.outputDir = options.outRoot / options.input.stem();
  prepareOutputDir(stats.outputDir, options.overwrite);
  std::ofstream csv(stats.outputDir / "frames.csv");
  csv << "video,frame,pts_ms,reason,motion_score\n";

  const std::string video = options.input.filename().string();
  const int thumbWidth = std::min(kThumbWidth, reader.info().width);
  const int thumbHeight = std::min(kThumbHeight, reader.info().height);
  const double clipEndMs = options.sampler.clip
                               ? options.sampler.clip->startMs + options.sampler.clip->durationMs
                               : std::numeric_limits<double>::infinity();
  std::optional<LumaThumb> previous;
  std::optional<LumaThumb> lastKept;

  while (const VideoFrame* frame = reader.next()) {
    ++stats.decodedFrames;
    if (frame->ptsMs >= clipEndMs) break;
    LumaThumb thumb = makeLumaThumb(frame->luma(), thumbWidth, thumbHeight);
    const double motion = previous ? meanAbsDiff(thumb, *previous) : 0.0;
    const double diffToKept = lastKept ? meanAbsDiff(thumb, *lastKept) : 1.0;
    const SampleReason reason = sampler.offer(frame->ptsMs, motion, diffToKept);
    if (reason != SampleReason::None) {
      jpeg.write(*frame->av, stats.outputDir / frameFileName(frame->index));
      csv << csvRow(video, *frame, reason, motion);
      countReason(stats, reason);
      lastKept = thumb;
    }
    previous = std::move(thumb);
  }

  csv.close();
  if (!csv) throw VcError("write " + (stats.outputDir / "frames.csv").string() + " failed");
  stats.decodeErrors = reader.decodeErrors();
  stats.usedHardware = reader.usingHardware();
  stats.elapsedSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return stats;
}

}  // namespace vc
```

`cpp/apps/vc_extract/main.cpp`：

```cpp
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "vc_extract/extract_cli.h"
#include "vc_extract/extract_job.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;
constexpr int kExitOutputExists = 3;

void printStats(const vc::ExtractStats& s) {
  const double fps = s.elapsedSec > 0 ? static_cast<double>(s.decodedFrames) / s.elapsedSec : 0;
  std::printf("output:        %s\n", s.outputDir.string().c_str());
  std::printf("decoded:       %lld frames (%s decode)\n", static_cast<long long>(s.decodedFrames),
              s.usedHardware ? "hardware" : "software");
  std::printf("kept:          %lld (base %lld, motion %lld, clip %lld)\n",
              static_cast<long long>(s.keptFrames), static_cast<long long>(s.keptBase),
              static_cast<long long>(s.keptMotion), static_cast<long long>(s.keptClip));
  std::printf("decode errors: %lld\n", static_cast<long long>(s.decodeErrors));
  std::printf("elapsed:       %.2f s (%.1f fps)\n", s.elapsedSec, fps);
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << vc::extractUsage();
      return kExitOk;
    }
  }

  vc::ExtractOptions options;
  try {
    options = vc::parseExtractArgs(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_extract: " << e.what() << "\n\n" << vc::extractUsage();
    return kExitUsage;
  }

  av_log_set_level(AV_LOG_ERROR);
  try {
    printStats(vc::runExtract(options));
    return kExitOk;
  } catch (const vc::OutputExistsError& e) {
    std::cerr << "vc_extract: " << e.what() << '\n';
    return kExitOutputExists;
  } catch (const std::exception& e) {
    std::cerr << "vc_extract: " << e.what() << '\n';
    return kExitFailure;
  }
}
```

- [ ] **Step 4: 构建并跑本任务测试**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='ExtractJob.*:ExtractCli.*'`
Expected: 构建无警告；`[  PASSED  ] 10 tests.`

- [ ] **Step 5: 手工冒烟：可执行文件与退出码**

依次运行（测试视频在上一步已生成在 `build/cpp/tests/tmp/`）：

```bash
rm -rf build/cpp/smoke
./build/cpp/vc_extract --input build/cpp/tests/tmp/basic.mp4 --out build/cpp/smoke; echo "exit=$?"
./build/cpp/vc_extract --input build/cpp/tests/tmp/basic.mp4 --out build/cpp/smoke; echo "exit=$?"
./build/cpp/vc_extract --input build/cpp/tests/tmp/basic.mp4 --out build/cpp/smoke --fps 4; echo "exit=$?"
./build/cpp/vc_extract --input build/cpp/tests/tmp/nope.mp4 --out build/cpp/smoke --overwrite; echo "exit=$?"
```

Expected：
1. 打印统计，`kept: 10 (base 10, motion 0, clip 0)`，在 Apple Silicon 上为 `hardware decode`，`exit=0`
2. `vc_extract: output directory exists: … (pass --overwrite to replace it)`，`exit=3`
3. `vc_extract: fps must be >= 5, got 4` 加帮助文本，`exit=2`
4. `vc_extract: open …/nope.mp4: No such file or directory`，`exit=1`；`build/cpp/smoke/basic/` 仍然保留第 1 次的输出

- [ ] **Step 6: 写说明文档**

`cpp/README.md`：

````markdown
# C++ 模块

行车记录仪视频脱敏的 C++ 部分。当前包含抽帧工具 `vc_extract`（设计见
`docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §4）。

## 依赖

- CMake ≥ 3.24，支持 C++20 的编译器
- FFmpeg 开发库：macOS `brew install ffmpeg`；Linux `libavformat-dev libavcodec-dev libswscale-dev libavutil-dev`
- GoogleTest：macOS `brew install googletest`；Linux `libgtest-dev`

## 构建

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j 4
```

## 测试

测试视频由本机 `ffmpeg` 现场生成到 `build/cpp/tests/tmp/`，不需要联网。本机请按模块过滤运行：

```bash
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*'
./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
```

全量（`ctest --test-dir build/cpp`）交给 CI 运行。

## vc_extract

```bash
./build/cpp/vc_extract --input data/videos/trip01_0003.mp4 --out data/frames
```

输出写到 `data/frames/trip01_0003/`：

- `<帧号:06d>.jpg`：保留下来的帧
- `frames.csv`：`video,frame,pts_ms,reason,motion_score`，其中 `reason` 为 `base`（基础频率）、`motion`（运动加密）或 `clip`（片段模式）

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 5 | 基础抽帧频率，不能低于 5 |
| `--max-fps` | 15 | 运动分超过阈值时使用的频率 |
| `--motion-threshold` | 0.05 | 与上一帧 160×90 缩略图的平均亮度差（0–1） |
| `--dedup` | 关闭 | 与上一张保留帧差异低于该值时跳过，训练采集建议 0.01 |
| `--clip S,D` | 关闭 | 导出第 S 秒起 D 秒内的每一帧，用于时间覆盖率评测 |
| `--hwaccel` | auto | `auto` / `none` / `videotoolbox` |
| `--jpeg-q` | 2 | MJPEG 量化值，2 最好（约等于质量 95） |
| `--overwrite` | 关闭 | 覆盖已有输出目录 |

`motion_score` 会写进 CSV，阈值可以在拿到真实视频后按分布调整。

退出码：0 成功，1 处理失败，2 参数错误，3 输出目录已存在。
````

- [ ] **Step 7: 提交**

```bash
git add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/apps/vc_extract/extract_job.h cpp/apps/vc_extract/extract_job.cpp cpp/apps/vc_extract/main.cpp cpp/tests/extract_job_test.cpp cpp/README.md
git commit -m "feat: add vc_extract frame extraction tool"
```

---

## 收尾

- [ ] **按模块复跑一遍本计划的全部用例**（分三次过滤运行，不跑 `ctest` 全量）：

```bash
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*'
./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
```

Expected：分别 `14`、`10`、`10` 个用例通过。

- [ ] **用一段真实的行车记录仪视频试跑**（用户提供路径；只跑一段 2 分钟视频，不批量）：

```bash
./build/cpp/vc_extract --input <用户提供的视频> --out data/frames --dedup 0.01
```

记录：解码 fps、是否硬解、各 `reason` 的数量、`motion_score` 的大致分布（看 `frames.csv`），作为计划 2 调参的依据。

- [ ] **交付时告知用户**本计划“与规格的偏差”三条，以及真实视频试跑的结果。
