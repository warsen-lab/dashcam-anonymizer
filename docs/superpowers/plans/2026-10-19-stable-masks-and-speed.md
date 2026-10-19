# 打码观感与速度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按已批准的设计 `docs/superpowers/specs/2026-10-19-stable-masks-and-speed-design.md`（下称"本设计"）改造 `vc_redact` 和 `vc_eval`：检测频率按视频帧率取固定步长（5–60，默认 10），学生模型和 YuNet 并行检测；`BoxTimeline` 改成"一个目标一个平滑框"（插值、丢失后原地保持、幽灵再匹配、大小迟滞、面积上限、低分确认）；马赛克网格锚定整幅画面、块大小按轨迹固定；`vc_eval` 增加观感指标、参照误检排除和可配置召回目标；最后用真实数据校准并验收（人脸 ≥ 95%、车牌 ≥ 98%，打码面积 p99 ≤ 10%、最大 ≤ 30%，帧间跳变 ≤ 5%，同轨多框帧 0，10 档 ≥ 200 fps）。

**Architecture:** 改动集中在现有模块里，不加新的进程或线程模型。`cpp/apps/vc_redact/detect_rate.*`（新）负责"视频帧率 → 关键帧步长"；`Processor` 不再用 `FrameSampler`，按帧序号取模选关键帧；`MergedDetector` 用 `std::async` 让第二个检测器在另一个线程上与第一个同时跑；`BoxTimeline` 重写为轨迹模型（每条轨迹有编号、每帧至多一个框），输出 `TimelineOutput{masks, confirmed}`，`Processor` 据此延后一个关键帧写 `detections.jsonl`；`mask/mosaic.*` 改成画面坐标网格加 `TrackBlocks`（每条轨迹的块大小）。`vc_eval` 新增 `viewing.py`，`coverage.py` 增加排除表和召回目标，`run.py` / `report.py` / `cli.py` 接上。

**Tech Stack:** C++20、CMake ≥ 3.24、FFmpeg 8、onnxruntime 1.30.0（CoreML EP）、nlohmann/json 3.12.0、GoogleTest；Python 3.14、numpy、pytest（仓库根目录 `.venv`）。

## Global Constraints

- **观看体验优先**（本设计 §1、§6）：接受召回小幅下降，验收下限是人脸 ≥ 95%、车牌 ≥ 98%（排除人工核查过的参照误检后，三段合计）。`vc_eval` 的默认召回目标仍是 0.99（主设计的生产目标），本轮验收显式传 `--recall-target face=0.95,plate=0.98`。
- C++20，`-Wall -Wextra -Wpedantic` 零警告（本机 Apple clang 对"位置初始化漏掉一个没有默认初始化器的成员"会报 `-Wmissing-field-initializers`，所以新结构体的每个成员都写默认初始化器，例如 `std::vector<float> confirmScore{};`）。Debug 测试构建在 `build/cpp`：`cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug`；Release 在 `build/cpp-release`：`cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF`。
- **完全不联网。** 每条构建、测试、运行命令都包在 `sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))' …` 里；下文的命令块先定义 `SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')`，再用 `"${SB[@]}"`。onnxruntime 遥测保持关闭（`cpp/src/detect/ort_session.cpp` 已关，不要动）。不安装、不升级任何包。
- **只跑过滤后的 gtest。** `cpp/README.md` 列了六组过滤命令，每个任务写明它要跑的过滤器；不得运行不带过滤器的 `vc_tests` 或 `ctest`。新测试套件 `DetectRate` 并入第六组（Task 1 改 README）。
- **只跑本任务改动的 pytest 文件**，在 `py/` 目录里：`/Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/<file>.py -p no:cacheprovider -q`（worktree 里没有 `.venv`，所以用主 checkout 的绝对路径）。不得运行不带路径的 pytest。
- **提交：** conventional commits，不加 Co-Authored-By 或任何署名行；不 push；绝不 `git stash`。
- **本机一次只跑一个重任务**：构建、`vc_redact`、`vc_eval run` 不并行，也不和别的会话的重任务重叠。
- **工作位置：** Task 1–10 在 worktree 分支 `feat/stable-masks` 上做（superpowers:using-git-worktrees），只用仓库里的测试夹具。Task 11 要用 `data/`（不在 git 里，只在主 checkout），在 `feat/stable-masks` 合并进 `main`、并把主 checkout 切到最新 `main`（合并在本地完成：`git checkout main`，不 pull、不联网；本地有未提交改动时先问用户，不用 stash）之后，在主 checkout `/Users/warsen/Projects/video-compliance-process` 里做。
- **没提到的行为一律不变：** `vc_extract` 与 `FrameSampler`；`--detect-every-frame`；"缓冲满 120 帧强制出一个关键帧"；两轮匹配规则（`match_max_distance`、`match_max_distance_loose`、宽松轮的宽高 3 倍以内）；`expand_fraction`、`motion_expand_fraction`、`min_box_px` 的含义；检测框合并规则（同类 IoU ≥ 0.5 取并集、分数取较大值）；输出先写 `.partial`、失败删光本次文件的规则。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `cpp/apps/vc_redact/detect_rate.h`, `.cpp` | 新建（Task 1） | 视频帧率（`avg_frame_rate`，退回 `r_frame_rate`）、`detect_fps` 范围 [5, 60]、`stride`、实际检测频率 |
| `cpp/tests/detect_rate_test.cpp` | 新建（Task 1） | `DetectRate.*` |
| `cpp/apps/vc_redact/redact_config.h`, `.cpp` | 改（Task 2、5、6、11） | 删 `max_detect_fps` / `motion_threshold`（出现即报错并说明已删除）；`detect_fps` ∈ [5, 60]；新键 `shrink_per_frame`、`max_mask_area_ratio`、`face_confirm_score`、`plate_confirm_score`；`timelineFor()`；Task 11 写入校准值 |
| `configs/redact.json` | 改（Task 2、5、6、11） | 与 `RedactConfig` 默认值逐键相等（测试保证） |
| `cpp/apps/vc_redact/redact_cli.h`, `.cpp` | 改（Task 2） | `--detect-fps` 只覆盖 `detect_fps`；帮助文本 |
| `cpp/apps/vc_redact/redact_job.h`, `.cpp` | 改（Task 2、3、4、6、7） | 固定步长关键帧；`stats.json` 的 `video_fps`、`keyframe_stride`、`detect_fps_actual`、`stages.detect_student` / `detect_face`；摘要两行新内容；`note:` 提示；延后一个关键帧写 `detections.jsonl`；按轨迹块大小打码 |
| `cpp/apps/vc_redact/records.h`, `.cpp` | 改（Task 4、6、7） | `masks.jsonl` 每个框加 `track`、`block`；`detections.jsonl` 加 `confirmed` |
| `cpp/src/detect/merged_detector.h`, `.cpp` | 改（Task 3） | 两个检测器并行；各自忙碌时间 |
| `cpp/src/track/box_timeline.h`, `.cpp` | 重写（Task 4），改（Task 5、6） | 轨迹编号、每轨每帧一框、`held`、幽灵从保持位置插值；大小迟滞与面积上限；低分确认 |
| `cpp/src/mask/mosaic.h`, `.cpp` | 重写（Task 7） | 画面坐标网格、块大小档位、`TrackBlocks`、大块先打 |
| `cpp/CMakeLists.txt`, `cpp/tests/CMakeLists.txt` | 改（Task 1、3） | `detect_rate.cpp`、`detect_rate_test.cpp`；`vc_detect_lib` 链接 `Threads::Threads` |
| `cpp/tests/box_timeline_test.cpp` | 重写（Task 4），追加（Task 5、6） | 见 Task 4 的保留 / 改写 / 删除清单 |
| `cpp/tests/mosaic_test.cpp` | 重写（Task 7） | 网格、档位、色度、边缘、重叠顺序 |
| `cpp/tests/merged_detector_test.cpp`, `redact_config_test.cpp`, `redact_cli_test.cpp`, `redact_job_test.cpp`, `records_test.cpp` | 改 | 各任务的测试 |
| `scripts/redact.sh` | 改（Task 2） | `VC_DETECT_FPS` → `--detect-fps` |
| `py/vc_eval/viewing.py` | 新建（Task 8） | 观感指标（1/4 栅格，全部帧） |
| `py/vc_eval/records.py` | 改（Task 6、9） | `detections_by_frame(confirmed_only=…)`；`read_excluded()` |
| `py/vc_eval/temporal.py` | 改（Task 6） | 逐帧参照只用 `confirmed` 不为 false 的框 |
| `py/vc_eval/coverage.py` | 改（Task 9、10） | 排除参照误检；召回目标；`parse_recall_targets()` |
| `py/vc_eval/run.py` | 改（Task 2、8、9、10） | fps 上限 60；观感；排除计数与 `miss_list`；召回目标 |
| `py/vc_eval/report.py` | 改（Task 8、9、10） | "观感"一节、"排除的参照误检"一节、召回目标 |
| `py/vc_eval/cli.py` | 改（Task 8、10） | `viewing` 子命令；`--recall-target`（`run` 和 `coverage`） |
| `py/tests/test_eval_viewing.py` | 新建（Task 8） | 观感指标 |
| `py/tests/test_eval_coverage.py`, `py/tests/test_eval_run.py` | 改 | 各任务的测试 |
| `cpp/README.md`, `docs/USAGE.md` | 改（各任务顺带） | 新参数、新字段、实测速度（Task 11） |
| `docs/reports/stable-masks-2026-10-19.md` | 新建（Task 11） | 校准与验收结果 |
| `data/labels/sg_drive_1200s/excluded_refs.csv`, `data/labels/sg_drive_60s/excluded_refs.csv` | 新建（Task 11，不进 git） | 人工核查过的老师误检 |

任务顺序：Task 1 → 2 → 3 → 4 → 5 → 6 → 7（C++，后一个依赖前一个），Task 8 → 9 → 10（Python，只依赖 Task 4 的 `track` 字段和 Task 6 的 `confirmed` 字段的格式，可以在 Task 7 之后顺序做），Task 11 最后。

---

### Task 1: 检测档位的纯函数（`detect_rate`）

**Files:**
- Create: `cpp/apps/vc_redact/detect_rate.h`, `cpp/apps/vc_redact/detect_rate.cpp`
- Create: `cpp/tests/detect_rate_test.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_redact_lib` 的源文件列表，第 57–64 行）
- Modify: `cpp/tests/CMakeLists.txt`（`add_executable(vc_tests …)` 列表）
- Modify: `cpp/README.md`（"测试"一节第六组过滤命令）

**Interfaces:**
- Consumes: `VcError`（`common/vc_error.h`）；`AVRational`、`av_q2d`（`libavutil/rational.h`）。
- Produces（Task 2 使用）:
  - `inline constexpr double kMinDetectFps = 5.0; inline constexpr double kMaxDetectFps = 60.0;`
  - `double streamFrameRate(AVRational averageRate, AVRational baseRate, const std::string& video);` —— 平均帧率可用就用它，否则 `r_frame_rate`，都不可用抛 `VcError`（消息含视频名和 "frame rate"）。
  - `struct DetectSchedule { int64_t stride = 1; double videoFps = 0; double actualFps = 0; bool aboveVideoRate = false; };`
  - `DetectSchedule detectSchedule(double videoFps, double detectFps);` —— `stride = max(1, ⌊videoFps / detectFps + 0.5⌋)`；`detectFps` 不在 [5, 60] 或 `videoFps` 不在 (0, 1e6] 时抛 `std::invalid_argument`。
  - `DetectSchedule everyFrameSchedule(double videoFps);` —— `--detect-every-frame` 用，stride 1。
  - `bool onSchedule(const DetectSchedule& s, int64_t frameIndex);` —— `frameIndex % stride == 0`。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/detect_rate_test.cpp`：

```cpp
#include "vc_redact/detect_rate.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

struct StrideCase {
  double videoFps;
  double detectFps;
  int64_t stride;
};

// stride = max(1, floor(video fps / detect_fps + 0.5)) for the recommended rates (spec 2026-10-19 §2) on
// common frame rates, including the 59.94 fps of the dashcam segments.
TEST(DetectRate, StrideFollowsTheVideoFrameRate) {
  const double ntsc60 = 60000.0 / 1001.0;
  const double film = 24000.0 / 1001.0;
  const std::vector<StrideCase> cases = {
      {60, 5, 12},     {60, 10, 6},     {60, 15, 4},     {60, 30, 2},     {60, 60, 1},
      {ntsc60, 5, 12}, {ntsc60, 10, 6}, {ntsc60, 15, 4}, {ntsc60, 30, 2}, {ntsc60, 60, 1},
      {30, 5, 6},      {30, 10, 3},     {30, 15, 2},     {30, 30, 1},
      {25, 5, 5},      {25, 10, 3},     {25, 15, 2},     {25, 25, 1},     {25, 30, 1},
      {film, 5, 5},    {film, 10, 2},   {film, 15, 2},   {film, 30, 1},
  };
  for (const StrideCase& c : cases) {
    const DetectSchedule s = detectSchedule(c.videoFps, c.detectFps);
    EXPECT_EQ(s.stride, c.stride) << c.videoFps << " fps video, detect_fps " << c.detectFps;
    EXPECT_DOUBLE_EQ(s.videoFps, c.videoFps);
    EXPECT_DOUBLE_EQ(s.actualFps, c.videoFps / static_cast<double>(c.stride));
  }
}

// The spec's example: 10 on a 25 fps video is every third frame, 8.3 detections a second.
TEST(DetectRate, TenOnATwentyFiveFpsVideoIsEveryThirdFrame) {
  const DetectSchedule s = detectSchedule(25, 10);
  EXPECT_EQ(s.stride, 3);
  EXPECT_NEAR(s.actualFps, 8.333, 1e-3);
  EXPECT_FALSE(s.aboveVideoRate);
}

// A request above the video frame rate detects on every frame and says so; exactly the video rate is not above it.
TEST(DetectRate, ARequestAboveTheVideoRateDetectsEveryFrame) {
  const DetectSchedule above = detectSchedule(25, 30);
  EXPECT_EQ(above.stride, 1);
  EXPECT_DOUBLE_EQ(above.actualFps, 25);
  EXPECT_TRUE(above.aboveVideoRate);
  const DetectSchedule equal = detectSchedule(30, 30);
  EXPECT_EQ(equal.stride, 1);
  EXPECT_FALSE(equal.aboveVideoRate);
  EXPECT_TRUE(detectSchedule(60000.0 / 1001.0, 60).aboveVideoRate);
}

TEST(DetectRate, AcceptsFiveToSixtyAndNothingElse) {
  EXPECT_EQ(detectSchedule(60, 5).stride, 12);
  EXPECT_EQ(detectSchedule(60, 60).stride, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {4.999, 60.001, 0.0, -10.0, nan, inf}) {
    EXPECT_THROW(detectSchedule(60, bad), std::invalid_argument) << bad;
  }
  for (const double bad : {0.0, -30.0, nan, inf, 2e6}) {
    EXPECT_THROW(detectSchedule(bad, 10), std::invalid_argument) << bad;
    EXPECT_THROW(everyFrameSchedule(bad), std::invalid_argument) << bad;
  }
}

TEST(DetectRate, EveryFrameScheduleHasStrideOne) {
  const DetectSchedule s = everyFrameSchedule(29.97);
  EXPECT_EQ(s.stride, 1);
  EXPECT_DOUBLE_EQ(s.actualFps, 29.97);
  EXPECT_FALSE(s.aboveVideoRate);
}

TEST(DetectRate, KeyframesAreEveryStrideFramesByIndex) {
  const DetectSchedule s = detectSchedule(60, 10);
  std::vector<int64_t> keys;
  for (int64_t i = 0; i < 20; ++i) {
    if (onSchedule(s, i)) keys.push_back(i);
  }
  EXPECT_EQ(keys, (std::vector<int64_t>{0, 6, 12, 18}));
}

TEST(DetectRate, TheFrameRateFallsBackFromTheAverageToTheBaseRate) {
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{60000, 1001}, AVRational{30, 1}, "a.mp4"), 60000.0 / 1001.0);
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{0, 1}, AVRational{25, 1}, "a.mp4"), 25.0);
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{25, 0}, AVRational{30000, 1001}, "a.mp4"), 30000.0 / 1001.0);
  try {
    streamFrameRate(AVRational{0, 1}, AVRational{0, 0}, "clip.mp4");
    ADD_FAILURE() << "accepted a stream without a frame rate";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("clip.mp4"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("frame rate"), std::string::npos) << e.what();
  }
}

}  // namespace
}  // namespace vc
```

在 `cpp/tests/CMakeLists.txt` 的 `add_executable(vc_tests` 列表里，`box_timeline_test.cpp` 之后加一行 `detect_rate_test.cpp`。

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && "${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`fatal error: 'vc_redact/detect_rate.h' file not found`。

- [ ] **Step 3: 实现**

`cpp/apps/vc_redact/detect_rate.h`：

```cpp
#pragma once

#include <cstdint>
#include <string>

extern "C" {
#include <libavutil/rational.h>
}

namespace vc {

// detect_fps bounds (spec 2026-10-19 §2).
inline constexpr double kMinDetectFps = 5.0;
inline constexpr double kMaxDetectFps = 60.0;

// The frame rate keyframes are spaced by: the video stream's average frame rate (avg_frame_rate),
// else its base rate (r_frame_rate). Throws VcError naming `video` when neither is a positive rate.
double streamFrameRate(AVRational averageRate, AVRational baseRate, const std::string& video);

// Where vc_redact detects: on every `stride`-th frame by frame index (index % stride == 0), the same
// over the whole video.
struct DetectSchedule {
  int64_t stride = 1;
  double videoFps = 0;
  double actualFps = 0;         // videoFps / stride: detections per second of video
  bool aboveVideoRate = false;  // the requested rate exceeded the video's, so every frame is a keyframe
};

// stride = max(1, floor(videoFps / detectFps + 0.5)). Throws std::invalid_argument unless videoFps is in
// (0, 1e6] and detectFps in [kMinDetectFps, kMaxDetectFps].
DetectSchedule detectSchedule(double videoFps, double detectFps);

// --detect-every-frame: stride 1. Throws std::invalid_argument unless videoFps is in (0, 1e6].
DetectSchedule everyFrameSchedule(double videoFps);

// True when frame `frameIndex` is a keyframe of the schedule.
bool onSchedule(const DetectSchedule& s, int64_t frameIndex);

}  // namespace vc
```

`cpp/apps/vc_redact/detect_rate.cpp`：

```cpp
#include "vc_redact/detect_rate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/vc_error.h"

namespace vc {
namespace {

// Above this a "frame rate" is a broken header, and the stride would overflow int64_t.
constexpr double kMaxVideoFps = 1e6;

bool usable(AVRational rate) { return rate.num > 0 && rate.den > 0; }

void requireVideoFps(double videoFps) {
  if (!(videoFps > 0 && videoFps <= kMaxVideoFps)) {
    throw std::invalid_argument("the video frame rate must be in (0, 1e6] fps");
  }
}

}  // namespace

double streamFrameRate(AVRational averageRate, AVRational baseRate, const std::string& video) {
  if (usable(averageRate)) return av_q2d(averageRate);
  if (usable(baseRate)) return av_q2d(baseRate);
  throw VcError("cannot space the keyframes of " + video +
                ": its video stream has neither an average frame rate nor a base frame rate");
}

DetectSchedule detectSchedule(double videoFps, double detectFps) {
  requireVideoFps(videoFps);
  if (!(detectFps >= kMinDetectFps && detectFps <= kMaxDetectFps)) {
    throw std::invalid_argument("detect_fps must be in [5, 60]");
  }
  const auto stride = std::max<int64_t>(1, static_cast<int64_t>(std::floor(videoFps / detectFps + 0.5)));
  return DetectSchedule{stride, videoFps, videoFps / static_cast<double>(stride), detectFps > videoFps};
}

DetectSchedule everyFrameSchedule(double videoFps) {
  requireVideoFps(videoFps);
  return DetectSchedule{1, videoFps, videoFps, false};
}

bool onSchedule(const DetectSchedule& s, int64_t frameIndex) { return frameIndex % s.stride == 0; }

}  // namespace vc
```

`cpp/CMakeLists.txt` 的 `vc_redact_lib` 源文件列表加一行（放在 `apps/vc_detect/detect_images.cpp` 之后、`apps/vc_redact/records.cpp` 之前）：

```cmake
  apps/vc_redact/detect_rate.cpp
```

`cpp/README.md` "测试"一节第六组过滤命令改成：

```bash
./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && "${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='DetectRate.*'
```

预期：`[  PASSED  ] 7 tests.`，构建无警告。

- [ ] **Step 5: 提交**

```bash
git add cpp/apps/vc_redact/detect_rate.h cpp/apps/vc_redact/detect_rate.cpp cpp/tests/detect_rate_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/README.md
git commit -m "feat: keyframe stride from the video frame rate and detect_fps"
```

---

### Task 2: `vc_redact` 按固定步长选关键帧

**Files:**
- Modify: `cpp/apps/vc_redact/redact_config.h`（第 15–40 行）、`cpp/apps/vc_redact/redact_config.cpp`（全文件的键表、校验、解析、序列化）
- Modify: `configs/redact.json`
- Modify: `cpp/apps/vc_redact/redact_cli.h`（第 10–12 行注释）、`cpp/apps/vc_redact/redact_cli.cpp`（第 89–98 行、第 102–126 行）
- Modify: `cpp/apps/vc_redact/redact_job.h`（`RedactStats`，第 40–68 行）、`cpp/apps/vc_redact/redact_job.cpp`（includes、`Processor` 第 171–290 行、`runPipeline`、`writeStats`、`redactWarnings`、`redactSummary`、`runRedact`）
- Modify: `scripts/redact.sh`
- Modify: `py/vc_eval/run.py`（`MIN_DETECT_FPS` 附近、`_validate`）、`py/vc_eval/cli.py`（`--fps` 帮助）、`py/README.md`（`--fps` 一条）
- Modify: `cpp/README.md`、`docs/USAGE.md`
- Test: `cpp/tests/redact_config_test.cpp`、`cpp/tests/redact_cli_test.cpp`、`cpp/tests/redact_job_test.cpp`、`py/tests/test_eval_run.py`

**Interfaces:**
- Consumes: Task 1 的 `kMinDetectFps`、`kMaxDetectFps`、`streamFrameRate`、`detectSchedule`、`everyFrameSchedule`、`onSchedule`、`DetectSchedule`。
- Produces:
  - `RedactConfig` 去掉 `maxDetectFps`、`motionThreshold`；`detectFps` 校验为 [5, 60]，消息 `"<origin>: detect_fps must be in [5, 60]"`；配置里出现 `max_detect_fps` / `motion_threshold` 时 `VcError("<origin>: <key> was removed: keyframes now come every round(video fps / detect_fps) frames, whatever the motion (delete the key)")`。
  - `RedactStats` 新字段：`double videoFps = 0; int64_t keyframeStride = 0; double detectFpsActual = 0; double detectFpsRequested = 0; bool detectFpsAboveVideoRate = false;`
  - `stats.json` 新键 `video_fps`、`keyframe_stride`、`detect_fps_actual`（紧跟 `detect_every_frame`）。
  - `redactSummary` 在 `frames:` 行之后多一行 `detect rate:   %.2f /s (every %lld frames of a %.2f fps video)`。
  - `redactWarnings` 在请求超过视频帧率时第一条为 `note: detect_fps <g> is above the video's <.2f> fps, so every frame is a keyframe`。
  - `scripts/redact.sh` 的 `VC_DETECT_FPS`。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/redact_config_test.cpp`：

把第 20–25 行的 `kAllKeys` 换成：

```cpp
// The keys of configs/redact.json (and of redactConfigToJson).
const std::set<std::string> kAllKeys = {
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "mosaic_min_block_px",
    "mosaic_blocks_per_long_side", "face_score_threshold", "face_nms_iou"};
```

`expectSameConfig` 删掉 `maxDetectFps`、`motionThreshold` 两行；`nonDefaultConfig` 删掉 `c.maxDetectFps = 30.0;`、`c.motionThreshold = 0.125;` 两行。

`EmptyObjectGivesDefaultsAndKeysOverride` 换成：

```cpp
TEST(RedactConfig, EmptyObjectGivesDefaultsAndKeysOverride) {
  expectDefaults(parseRedactConfig("{}", "t"));
  const RedactConfig c = parseRedactConfig(R"({"detect_fps": 15, "expand_fraction": 0.2})", "t");
  EXPECT_DOUBLE_EQ(c.detectFps, 15);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.2);
}
```

`RejectsBadInput` 的列表里，把 `R"({"detect_fps": 20, "max_detect_fps": 10})"` 换成 `R"({"detect_fps": 61})"`，并在同一行后面加 `R"({"detect_fps": 4.99})"`。

`RejectsOutOfRangeValuesAndNulls` 的列表换成（删掉 `motion_threshold`、`max_detect_fps` 的四项）：

```cpp
  const std::vector<std::string> bad = {
      R"({"nms_iou": 1.5})",           R"({"min_box_px": -1})",
      R"({"detect_fps": null})",       R"({"score_threshold": null})",
      R"({"nms_iou": null})",          R"({"match_max_distance": null})",
      R"({"match_max_distance_loose": null})",
      R"({"expand_fraction": null})",  R"({"motion_expand_fraction": null})",
      R"({"min_box_px": null})",       R"({"mosaic_blocks_per_long_side": null})",
      R"({"face_score_threshold": null})", R"({"face_nms_iou": null})",
  };
```

`ValidateRefusesEveryNonFiniteDoubleField` 的 `fields` 删掉 `max_detect_fps`、`motion_threshold` 两项。

`DetectFpsMessageFormatsTheMinimum` 换成：

```cpp
TEST(RedactConfig, DetectFpsMessageGivesTheRange) {
  for (const char* text : {R"({"detect_fps": 4})", R"({"detect_fps": 61})"}) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), "cfg: detect_fps must be in [5, 60]");
    }
  }
}

TEST(RedactConfig, AcceptsTheDetectFpsBounds) {
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"detect_fps": 5})", "t").detectFps, 5);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"detect_fps": 60})", "t").detectFps, 60);
}

// The motion-triggered rate is gone (spec 2026-10-19 §2): an old config that still sets it is refused with a message
// that says the key was removed, not that it is an unknown typo.
TEST(RedactConfig, RejectsTheRemovedKeysByName) {
  for (const std::string key : {"max_detect_fps", "motion_threshold"}) {
    try {
      parseRedactConfig("{\"" + key + "\": 15}", "cfg");
      ADD_FAILURE() << "accepted " << key;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("cfg: " + key + " was removed", 0), 0u) << e.what();
    }
  }
}
```

`SerialisesEveryKey` 里 `EXPECT_EQ(j.size(), 14u);` 改成 `EXPECT_EQ(j.size(), 12u);`。

`cpp/tests/redact_cli_test.cpp`：

`AppliesDefaults` 删掉 `EXPECT_DOUBLE_EQ(o.config.maxDetectFps, RedactConfig{}.maxDetectFps);` 一行。

`DetectFpsOverridesAndRaisesTheMaximum`、`DetectFpsKeepsAHigherMaximum`、`DetectFpsOverridesTheConfigFile` 三个测试整体换成：

```cpp
TEST(RedactCli, DetectFpsOverridesTheDefault) {
  const RedactOptions o =
      parseRedactArgs({"--detect-fps", "30", "--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 30);
}

TEST(RedactCli, DetectFpsOverridesTheConfigFile) {
  const std::filesystem::path cfg = test::tmpDir() / "redact_cli_fps.json";
  std::ofstream(cfg) << R"({"detect_fps": 10})";
  const RedactOptions o = parseRedactArgs(
      {"--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl", "--config", cfg.string(), "--detect-fps", "20"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 20);
}
```

`RejectsBadCommandLines`：在函数开头 `bad` 文件之后加

```cpp
  const std::filesystem::path removed = test::tmpDir() / "redact_cli_removed.json";
  std::ofstream(removed) << R"({"detect_fps": 10, "max_detect_fps": 15})";
```

并在 `cases` 里 `{"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "2"},` 之后加：

```cpp
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "4.9"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "61"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--config", removed.string()},
```

`cpp/tests/redact_job_test.cpp`：

在 `baseOptions` 之后加一个夹具：

```cpp
// 2 s, 320x240 @ 25 fps (50 frames): a frame rate the 60 fps detection rates do not divide.
fs::path video25fps() {
  return test::makeMedia("basic_25fps.mp4", "-f lavfi -i testsrc2=s=320x240:r=25:d=2 -c:v libx264 -pix_fmt yuv420p -g 25");
}
```

`MasksTargetsThatMoveSeveralBoxWidthsBetweenKeyframes` 删掉 `o.config.maxDetectFps = o.config.detectFps;  // keyframes exactly every 200 ms, motion or not` 这一行（5 fps 在 30 fps 视频上就是每 6 帧一个关键帧）。

`ForcesAKeyframeWhenTheBufferGrowsTooLong` 上方的注释换成：

```cpp
// 500 frames at 1000 fps: 5 fps is a 200-frame stride, which would leave 200 frames buffered, so the
// 120-frame limit has to force keyframes (at 120 and 320).
```

`DetectEveryFrameMakesEveryFrameAKeyframe` 末尾加：

```cpp
  EXPECT_EQ(s.keyframeStride, 1);
  EXPECT_NEAR(s.detectFpsActual, 30.0, 1e-6);
```

在 `DetectEveryFrameMakesEveryFrameAKeyframe` 之后加三个测试：

```cpp
// Keyframes come every round(video fps / detect_fps) frames by frame index: 10 on a 30 fps video is every third
// frame, and stats.json says how often that really is (spec 2026-10-19 §2).
TEST(RedactJob, KeyframesComeEveryStrideFramesOfTheVideoFrameRate) {
  RedactOptions o = baseOptions("redact_stride");
  o.config.detectFps = 10.0;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 20);
  EXPECT_EQ(s.keyframeStride, 3);
  EXPECT_NEAR(s.videoFps, 30.0, 1e-6);
  EXPECT_NEAR(s.detectFpsActual, 10.0, 1e-6);
  std::ifstream detections(s.outputDir / "detections.jsonl");
  for (std::string line; std::getline(detections, line);) {
    EXPECT_EQ(nlohmann::json::parse(line).at("frame").get<int>() % 3, 0) << line;
  }
  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("keyframe_stride").get<int>(), 3);
  EXPECT_NEAR(stats.at("detect_fps_actual").get<double>(), 10.0, 1e-6);
  EXPECT_NEAR(stats.at("video_fps").get<double>(), 30.0, 1e-6);
}

// The spec's example: 10 on a 25 fps video is every third frame, 8.3 detections a second.
TEST(RedactJob, TenOnATwentyFiveFpsVideoDetectsEveryThirdFrame) {
  RedactOptions o = baseOptions("redact_25fps");
  o.input = video25fps();
  o.config.detectFps = 10.0;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 50);
  EXPECT_EQ(s.keyframes, 17);  // 0, 3, ..., 48
  EXPECT_EQ(s.keyframeStride, 3);
  EXPECT_NEAR(s.detectFpsActual, 25.0 / 3.0, 1e-6);
  EXPECT_TRUE(redactWarnings(s).empty());
}

TEST(RedactJob, ARequestAboveTheVideoRateDetectsEveryFrameAndSaysSo) {
  RedactOptions o = baseOptions("redact_above_rate");
  o.input = video25fps();
  o.config.detectFps = 30.0;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 50);
  EXPECT_EQ(s.keyframeStride, 1);
  const std::vector<std::string> warnings = redactWarnings(s);
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0], "note: detect_fps 30 is above the video's 25.00 fps, so every frame is a keyframe");
}
```

`TheSummaryShowsTheSetupAndTheParallelStages`：在 `s.encode = StageStat{2.4, 600};` 之后加

```cpp
  s.videoFps = 60000.0 / 1001.0;
  s.keyframeStride = 6;
  s.detectFpsActual = s.videoFps / 6;
```

并在最后一个断言之前加

```cpp
  EXPECT_NE(summary.find("detect rate:   9.99 /s (every 6 frames of a 59.94 fps video)\n"), std::string::npos) << summary;
```

`py/tests/test_eval_run.py`：`test_bad_configuration_is_refused_before_any_run` 的参数表里把五处 `"at least 5"` 换成 `"from 5 to 60"`，并在 `({"fps_values": [5, float("inf")]}, …)` 之后加一项 `({"fps_values": [5, 61]}, "from 5 to 60"),`。

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`no member named 'keyframeStride' in 'vc::RedactStats'`（以及 `detectFpsActual`、`videoFps`）。

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_run.py -p no:cacheprovider -q -k bad_configuration
```

预期：新加的 `[5, 61]` 一项 FAIL（`DID NOT RAISE`），改了消息的五项 FAIL（`Regex pattern did not match`）。

- [ ] **Step 3: 实现**

`cpp/apps/vc_redact/redact_config.h` 第 15–40 行换成：

```cpp
// configs/redact.json (proxy-validation spec §5.2, updated by spec 2026-10-19); the defaults equal the
// repository file. DecodeParams and TimelineConfig keep their own defaults, which vc_detect and the unit
// tests use.
//
// `face` is the YuNet face model's (vc_redact --face-model; unused without it). Its score threshold
// 0.3 is the lower bound of the teachers' low-confidence face band (configs/labeling.json
// face_low_score [0.3, 0.5]) and its NMS IoU the YuNet teacher's 0.3, both chosen before looking at
// any held-out video, not tuned on it.
struct RedactConfig {
  double detectFps = 10.0;  // keyframes per second of video, in [5, 60]; spaced by frame index (detect_rate.h)
  DecodeParams decode{.scoreThreshold = 0.05f};
  TimelineConfig timeline{.expandFraction = 0.3};
  MosaicConfig mosaic;
  YuNetParams face{.scoreThreshold = 0.3f, .nmsIou = 0.3f};
};

// Keys may be omitted; unknown keys, the removed max_detect_fps and motion_threshold, wrong types and
// out-of-range values throw VcError("<origin>: …"). detect_fps is in [5, 60]; mosaic_min_block_px and
// mosaic_blocks_per_long_side must be JSON integers in [1, 4096]; every other value must be a finite
// number. face_score_threshold is in [0, 1], face_nms_iou in (0, 1].
RedactConfig parseRedactConfig(const std::string& text, const std::string& origin);
RedactConfig readRedactConfig(const std::filesystem::path& path);
void validateRedactConfig(const RedactConfig& config, const std::string& origin);
nlohmann::ordered_json redactConfigToJson(const RedactConfig& config);
```

`cpp/apps/vc_redact/redact_config.cpp`：

- `#include "sampler/frame_sampler.h"` 换成 `#include "vc_redact/detect_rate.h"`。
- 第 16–20 行的 `kKeys` 换成：

```cpp
constexpr std::array<std::string_view, 12> kKeys = {
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "mosaic_min_block_px",
    "mosaic_blocks_per_long_side", "face_score_threshold", "face_nms_iou"};

// Keys of older configs that no longer exist (spec 2026-10-19 §2): refused with a message that says so.
constexpr std::array<std::string_view, 2> kRemovedKeys = {"max_detect_fps", "motion_threshold"};
```

- `validateRedactConfig` 里删掉 `max_detect_fps`、`motion_threshold` 的两条 `isfinite` 检查和 `max_detect_fps must be >= detect_fps`、`motion_threshold must be in [0, 1]` 两条，把 `detect_fps must be >= …` 一条换成：

```cpp
  require(c.detectFps >= kMinDetectFps && c.detectFps <= kMaxDetectFps,
          "detect_fps must be in [" + plainNumber(kMinDetectFps) + ", " + plainNumber(kMaxDetectFps) + "]");
```

- `parseRedactConfig` 的键检查循环换成：

```cpp
  for (const auto& item : j.items()) {
    if (std::find(kRemovedKeys.begin(), kRemovedKeys.end(), item.key()) != kRemovedKeys.end()) {
      throw VcError(origin + ": " + item.key() +
                    " was removed: keyframes now come every round(video fps / detect_fps) frames, whatever the motion "
                    "(delete the key)");
    }
    if (std::find(kKeys.begin(), kKeys.end(), item.key()) == kKeys.end()) {
      throw VcError(origin + ": unknown key '" + item.key() + "'");
    }
  }
```

  并删掉 `c.maxDetectFps = …`、`c.motionThreshold = …` 两行。
- `redactConfigToJson` 删掉 `j["max_detect_fps"]`、`j["motion_threshold"]` 两行。

`configs/redact.json` 删掉 `"max_detect_fps": 15.0,` 和 `"motion_threshold": 0.05,` 两行。

`cpp/apps/vc_redact/redact_cli.h` 第 10–11 行注释改成：

```cpp
// Parses the arguments after the program name. Throws UsageError, also for an unreadable or
// invalid --config and for a --detect-fps outside [5, 60].
```

`cpp/apps/vc_redact/redact_cli.cpp`：删掉 `#include <algorithm>`；第 91–94 行换成

```cpp
    if (detectFps) o.config.detectFps = *detectFps;
```

`redactUsage()` 里 `--detect-fps` 的两行换成：

```
  --detect-fps <n>          detections per second of video, 5 to 60 (overrides the config): a
                            keyframe every round(video fps / n) frames, every frame once n reaches
                            the video frame rate
```

`cpp/apps/vc_redact/redact_job.h` 的 `RedactStats` 里，在 `int64_t audioPackets = 0;` 之后加：

```cpp
  // Keyframe spacing (spec 2026-10-19 §2): a keyframe every keyframeStride frames of a videoFps video, i.e.
  // detectFpsActual detections per second of video. detectFpsRequested is the configured detect_fps (0 with
  // --detect-every-frame); detectFpsAboveVideoRate when it exceeded the video frame rate.
  double videoFps = 0;
  int64_t keyframeStride = 0;
  double detectFpsActual = 0;
  double detectFpsRequested = 0;
  bool detectFpsAboveVideoRate = false;
```

并把 `StageStat track;` 的注释改成 `// keyframe choice + BoxTimeline; items = frames`。

`cpp/apps/vc_redact/redact_job.cpp`：

- 删掉 `#include "sampler/frame_sampler.h"` 和 `#include "sampler/luma_thumb.h"`，加 `#include "vc_redact/detect_rate.h"`。
- 第 41–42 行注释改成 `// A keyframe is forced after this many buffered frames, whatever the stride (for very high frame rates).`
- 第 171–290 行的 `Processor` 整个换成：

```cpp
// Keyframe choice, detection, BoxTimeline and mosaic: everything between decode and encode.
class Processor {
 public:
  Processor(const RedactOptions& options, Detector& detector, const VideoInfo& info, const DetectSchedule& schedule,
            std::string video, std::ostream& detections, std::ostream& masks, BoundedQueue<Item>& out,
            RedactStats& stats)
      : options_(options),
        detector_(detector),
        schedule_(schedule),
        video_(std::move(video)),
        detections_(detections),
        masks_(masks),
        out_(out),
        stats_(stats),
        timeline_(options.config.timeline, info.width, info.height) {}

  void onVideo(VideoItem item) {
    const Clock::time_point started = Clock::now();
    // A keyframe every schedule_.stride frames by frame index (spec 2026-10-19 §2), the first frame always
    // (BoxTimeline starts on a keyframe), and one whenever kMaxBufferedFrames frames are waiting.
    const bool key = options_.detectEveryFrame || !sawKeyframe_ || onSchedule(schedule_, item.index) ||
                     buffer_.size() >= kMaxBufferedFrames;
    stats_.track.busySec += secondsSince(started);
    if (!key) {
      buffer_.push_back(std::move(item));
      return;
    }
    sawKeyframe_ = true;

    const VideoFrame view = viewOf(item);
    const Clock::time_point detectStarted = Clock::now();
    const std::vector<SourcedBox> found = detector_.detectSourced(view);
    stats_.detect.busySec += secondsSince(detectStarted);
    ++stats_.detect.items;
    ++stats_.keyframes;
    stats_.detections += static_cast<int64_t>(found.size());
    std::vector<Box> boxes;
    boxes.reserve(found.size());
    for (const SourcedBox& b : found) {
      detections_ << detectionRecord(video_, item.index, item.ptsMs, b.box, detector_.classes(), b.src);
      boxes.push_back(b.box);
    }

    const Clock::time_point trackStarted = Clock::now();
    const std::vector<std::vector<MaskBox>> masks =
        timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    emit(masks);
    buffer_.push_back(std::move(item));
  }

  void onEnd() {
    const Clock::time_point started = Clock::now();
    const std::vector<std::vector<MaskBox>> masks = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    emit(masks);
    if (!out_.push(Item{EndItem{}})) throw PipelineAborted{};
  }

 private:
  static VideoFrame viewOf(const VideoItem& item) {
    VideoFrame v;
    v.index = item.index;
    v.ptsMs = item.ptsMs;
    v.av = item.frame.get();
    return v;
  }

  std::vector<FrameTime> times() const {
    std::vector<FrameTime> t;
    t.reserve(buffer_.size());
    for (const VideoItem& f : buffer_) t.push_back(FrameTime{f.index, f.ptsMs});
    return t;
  }

  // Writes masks.jsonl, pixelates and sends every buffered frame downstream, in order.
  void emit(const std::vector<std::vector<MaskBox>>& masks) {
    if (masks.size() != buffer_.size()) throw VcError("internal error: masks do not match the buffered frames");
    for (size_t k = 0; k < buffer_.size(); ++k) {
      VideoItem& frame = buffer_[k];
      masks_ << maskRecord(video_, frame.index, masks[k], detector_.classes());
      if (!masks[k].empty()) {
        const Clock::time_point started = Clock::now();
        checkAv(av_frame_make_writable(frame.frame.get()), "make frame writable");
        std::vector<PixelRect> rects;
        rects.reserve(masks[k].size());
        for (const MaskBox& m : masks[k]) {
          rects.push_back(coveringRect(m.x1, m.y1, m.x2, m.y2, frame.frame->width, frame.frame->height));
        }
        applyMosaic(*frame.frame, rects, options_.config.mosaic);
        stats_.mosaic.busySec += secondsSince(started);
        ++stats_.maskedFrames;
        stats_.maskBoxes += static_cast<int64_t>(masks[k].size());
      }
      ++stats_.mosaic.items;
      if (!out_.push(Item{std::move(frame)})) throw PipelineAborted{};
    }
    buffer_.clear();
  }

  const RedactOptions& options_;
  Detector& detector_;
  DetectSchedule schedule_;
  std::string video_;
  std::ostream& detections_;
  std::ostream& masks_;
  BoundedQueue<Item>& out_;
  RedactStats& stats_;
  BoxTimeline timeline_;
  bool sawKeyframe_ = false;
  std::vector<VideoItem> buffer_;
};
```

- `runPipeline` 的签名在 `Detector& detector,` 之后加一个参数 `const DetectSchedule& schedule,`，构造 `Processor` 时把 `info` 之后的实参改成 `info, schedule, options.input.filename().string(), …`。
- `writeStats` 里 `j["detect_every_frame"] = o.detectEveryFrame;` 之后加：

```cpp
  j["video_fps"] = s.videoFps;
  j["keyframe_stride"] = s.keyframeStride;
  j["detect_fps_actual"] = s.detectFpsActual;
```

- `redactWarnings` 的函数体开头（`std::vector<std::string> warnings;` 之后）加：

```cpp
  if (stats.detectFpsAboveVideoRate) {
    warnings.push_back(formatted("note: detect_fps %g is above the video's %.2f fps, so every frame is a keyframe",
                                 stats.detectFpsRequested, stats.videoFps));
  }
```

- `redactSummary` 在 `frames:` 那一行之后加：

```cpp
  out += formatted("detect rate:   %.2f /s (every %lld frames of a %.2f fps video)\n", s.detectFpsActual,
                   static_cast<long long>(s.keyframeStride), s.videoFps);
```

- `runRedact` 里 `requireMp4Audio(audio);` 之后、`makeDetector` 之前加：

```cpp
  // The keyframe spacing comes from the video's own frame rate (spec 2026-10-19 §2); a stream without one
  // is refused here, before the model is loaded.
  const AVStream* videoStream = reader.videoStream();
  const double videoFps =
      streamFrameRate(videoStream->avg_frame_rate, videoStream->r_frame_rate, options.input.string());
  const DetectSchedule schedule =
      options.detectEveryFrame ? everyFrameSchedule(videoFps) : detectSchedule(videoFps, options.config.detectFps);
  stats.videoFps = schedule.videoFps;
  stats.keyframeStride = schedule.stride;
  stats.detectFpsActual = schedule.actualFps;
  stats.detectFpsRequested = options.detectEveryFrame ? 0.0 : options.config.detectFps;
  stats.detectFpsAboveVideoRate = schedule.aboveVideoRate;
```

  并把 `runPipeline(options, reader, *detector, audio, partial, started, stats);` 改成 `runPipeline(options, reader, *detector, schedule, audio, partial, started, stats);`。

（测试视频 `rawH264Stream()` 是裸 H.264，libavformat 的 raw 解复用器总会给 `avg_frame_rate` 填上 `-framerate` 的默认值 25，所以 `ADecodeThreadFailureFailsTheRunAndLeavesNothing` 照旧在解码线程报 "has no timestamp"，不需要改。）

`scripts/redact.sh`：注释的环境变量表在 `VC_FACE_MODEL` 一行之后加

```bash
#   VC_DETECT_FPS   detections per second, 5 to 60 (default: detect_fps of configs/redact.json, 10)
```

`FACE=…` 一行之后加 `DETECT_FPS="${VC_DETECT_FPS:-}"`；循环里 `[ -n "$FACE" ] && args+=(--face-model "$FACE")` 之后加

```bash
  [ -n "$DETECT_FPS" ] && args+=(--detect-fps "$DETECT_FPS")
```

`py/vc_eval/run.py`：`MIN_DETECT_FPS = 5.0` 之后加 `MAX_DETECT_FPS = 60.0`；`_validate` 里的检查换成

```python
    if not all(math.isfinite(fps) and MIN_DETECT_FPS <= fps <= MAX_DETECT_FPS for fps in cfg.fps_values):
        raise ValueError(f"detection rates must be finite numbers from {MIN_DETECT_FPS:g} to {MAX_DETECT_FPS:g} fps")
```

`py/vc_eval/cli.py` 的 `--fps` 帮助改成 `help="comma-separated detection rates, each from 5 to 60"`；`py/README.md` vc_eval 一节 `--fps 5,10,15,30` 那条里的“每个值必须是有限数且 ≥ 5”改成“每个值必须是 5 到 60 之间的有限数”。

`cpp/README.md` 的 vc_redact 一节：

- 参数表 `--detect-fps` 一行换成：`| --detect-fps <n> | 10 | 每秒检测几次（关键帧），5 到 60，覆盖配置里的 detect_fps。关键帧按帧序号每隔 stride = max(1, ⌊视频帧率 / n + 0.5⌋) 帧一个，整段不变（60 fps 视频上 5/10/15/30/60 是每 12/6/4/2/1 帧）；n 达到视频帧率就是逐帧检测，超过时 stderr 多一行 note:。视频帧率取视频流的 avg_frame_rate，读不到退回 r_frame_rate，都没有就报错。缓冲满 120 帧时仍强制出一个关键帧 |`
- `stats.json` 一条末尾加：`；关键帧间隔 video_fps、keyframe_stride、detect_fps_actual（实际每秒检测次数 = video_fps / keyframe_stride）`。
- "掉头"那条局限里的"缓解办法：运动触发的关键帧（画面运动大时检测频率升到 `max_detect_fps`，阈值 `motion_threshold`），或者调高检测频率（`--detect-fps`）"换成"缓解办法：调高检测频率（`--detect-fps`）"。
- 运行结束统计行的说明加一项："检测频率（`detect rate`：实际每秒检测次数、每隔几帧、视频帧率）"。

`docs/USAGE.md`：

- §2 "批量"的环境变量列表加一项：`` - `VC_DETECT_FPS=15`：检测频率（5–60，默认用 `configs/redact.json` 的 10）；`` 
- §4 表格 `detect_fps` 一行换成：`| detect_fps | 10 | 每秒检测几次，5–60。关键帧按帧序号每隔 ⌊视频帧率 / detect_fps + 0.5⌋ 帧一个；推荐档位是 5、10、15、30、60 中不超过视频帧率的（60 fps 视频 10 档 = 每 6 帧）。调大更稳，但更慢 |`；删掉 `max_detect_fps / motion_threshold` 那一行。

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactConfig.*:RedactCli.*:RedactJob.*:DetectRate.*'
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_run.py -p no:cacheprovider -q
```

预期：gtest 全部 PASSED、零警告；pytest 全部通过。

再检查 `scripts/redact.sh` 把 `VC_DETECT_FPS` 传给了 `vc_redact`（用一个只打印参数的假 `vc_redact`，不碰真模型）：

```bash
T=/private/tmp/claude-501/redact_sh_check; rm -rf "$T"; mkdir -p "$T/m" "$T/f"
printf '#!/bin/sh\necho "$@"\n' > "$T/vc_redact"; chmod +x "$T/vc_redact"
for d in m f; do echo x > "$T/$d/model.onnx"; echo x > "$T/$d/model.json"; done
echo x > "$T/a.mp4"
VC_REDACT="$T/vc_redact" VC_MODEL="$T/m" VC_FACE_MODEL="$T/f" VC_DETECT_FPS=15 VC_SANDBOX=0 bash scripts/redact.sh "$T/out" "$T/a.mp4" | grep -- '--detect-fps 15'
rm -rf "$T"
```

预期：打印出一行含 `--detect-fps 15` 的参数，退出码 0。

- [ ] **Step 5: 提交**

```bash
git add cpp/apps/vc_redact configs/redact.json cpp/tests/redact_config_test.cpp cpp/tests/redact_cli_test.cpp cpp/tests/redact_job_test.cpp scripts/redact.sh py/vc_eval/run.py py/vc_eval/cli.py py/tests/test_eval_run.py py/README.md cpp/README.md docs/USAGE.md
git commit -m "feat: fixed keyframe stride from the video frame rate, drop motion-triggered rates"
```

---
### Task 3: 两个检测器并行

**Files:**
- Modify: `cpp/src/detect/merged_detector.h`（第 23–43 行）、`cpp/src/detect/merged_detector.cpp`（includes、第 80–90 行）
- Modify: `cpp/CMakeLists.txt`（`find_package(Threads REQUIRED)` 挪到 `vc_detect_lib` 之前，`vc_detect_lib` 链接 `Threads::Threads`）
- Modify: `cpp/apps/vc_redact/redact_job.h`（`RedactStats`）、`cpp/apps/vc_redact/redact_job.cpp`（`writeStats`、`redactSummary`、`runRedact`）
- Modify: `cpp/README.md`（"人脸模型"一节的"合并"和"速度代价"两条）
- Test: `cpp/tests/merged_detector_test.cpp`、`cpp/tests/redact_job_test.cpp`

**Interfaces:**
- Consumes: `Detector`、`SourcedBox`、`mergeDetections`、`kMergeIou`（不变）。
- Produces:
  - `MergedDetector::detectSourced` 在调用线程上跑第一个检测器、在 `std::async(std::launch::async, …)` 线程上同时跑第二个，两者都返回后按原规则合并；任一抛错时先等另一个结束，再重新抛出（第一个的错误优先）。
  - `double MergedDetector::firstBusySec() const; double MergedDetector::secondBusySec() const;` —— 各自 `detectSourced` 的累计耗时（两者重叠）。
  - `RedactStats::detectStudent`、`RedactStats::detectFace`（`StageStat`；没有人脸模型时 `detectStudent == detect`、`detectFace` 为 `{0, 0}`）；`stats.json` 的 `stages.detect_student`、`stages.detect_face`；`stages.detect` 是每个关键帧检测的墙钟时间之和。
  - 有人脸模型时摘要多一行 `detectors:     student %.1f ms + face %.1f ms per keyframe, run in parallel: %.1f ms`。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/merged_detector_test.cpp`：includes 加 `<atomic>`、`<chrono>`、`<thread>`。在 `FixedDetector` 类之后加：

```cpp
// Sleeps, then returns fixed boxes or throws VcError("<source> failed"); `finished` is set when the sleep is over.
class DelayedDetector : public Detector {
 public:
  DelayedDetector(std::vector<std::string> classes, std::vector<Box> boxes, std::string source, int delayMs,
                  bool fails = false)
      : classes_(std::move(classes)), boxes_(std::move(boxes)), source_(std::move(source)), delayMs_(delayMs),
        fails_(fails) {}
  std::vector<Box> detect(const VideoFrame&) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
    finished = true;
    if (fails_) throw VcError(source_ + " failed");
    return boxes_;
  }
  const std::vector<std::string>& classes() const override { return classes_; }
  std::string source() const override { return source_; }
  std::atomic<bool> finished{false};

 private:
  std::vector<std::string> classes_;
  std::vector<Box> boxes_;
  std::string source_;
  int delayMs_;
  bool fails_;
};
```

在文件末尾的 `Detector.DescribesItselfAsSourceAndBackend` 之前加：

```cpp
// Both detectors run at once (spec 2026-10-19 §2): a call takes about as long as the slower one (600 ms), not
// the sum (900 ms), and each detector's own busy time is kept.
TEST(MergedDetector, RunsBothDetectorsAtTheSameTime) {
  auto student = std::make_unique<DelayedDetector>(std::vector<std::string>{"face", "plate"},
                                                   std::vector<Box>{Box{1, 0, 0, 10, 10, 0.5f}}, "student", 300);
  auto yunet = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"},
                                                 std::vector<Box>{Box{0, 50, 50, 60, 60, 0.6f}}, "yunet", 600);
  MergedDetector merged(std::move(student), std::move(yunet));
  const auto started = std::chrono::steady_clock::now();
  const std::vector<SourcedBox> boxes = merged.detectSourced(VideoFrame{});
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  EXPECT_GE(wall, 0.6);
  EXPECT_LT(wall, 0.85) << "one after the other it would take 0.9 s";
  EXPECT_EQ(boxes.size(), 2u);
  EXPECT_GE(merged.firstBusySec(), 0.3);
  EXPECT_LT(merged.firstBusySec(), 0.55);
  EXPECT_GE(merged.secondBusySec(), 0.6);
}

// Either detector failing fails the call with its own error, and the other one has finished by then: nothing
// keeps running on the frame after the call returns.
TEST(MergedDetector, EitherDetectorThrowingFailsTheCall) {
  for (const bool firstFails : {true, false}) {
    auto student = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "student",
                                                     firstFails ? 0 : 200, firstFails);
    auto yunet = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "yunet",
                                                   firstFails ? 200 : 0, !firstFails);
    DelayedDetector* s = student.get();
    DelayedDetector* y = yunet.get();
    MergedDetector merged(std::move(student), std::move(yunet));
    try {
      merged.detectSourced(VideoFrame{});
      ADD_FAILURE() << "no exception";
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), firstFails ? "student failed" : "yunet failed");
    }
    EXPECT_TRUE(s->finished);
    EXPECT_TRUE(y->finished);
  }
}

// The parallel merge is the serial one: both detectors' boxes, the second's classes mapped by name, through
// mergeDetections, in the same order.
TEST(MergedDetector, MergesExactlyAsTheSerialMerge) {
  std::vector<Box> studentBoxes;
  std::vector<Box> yunetBoxes;
  for (int i = 0; i < 12; ++i) {
    const auto f = static_cast<float>(i);
    studentBoxes.push_back(Box{i % 2, 13 * f, 7 * f, 13 * f + 20 + f, 7 * f + 15, 0.1f * f});
    yunetBoxes.push_back(Box{0, 13 * f + static_cast<float>(i % 3), 7 * f - 1, 13 * f + 22, 7 * f + 14 + f, 0.05f * f});
  }
  // The student's classes are {"plate", "face"}: YuNet's "face" (its class 0) is the student's class 1.
  MergedDetector merged(
      std::make_unique<FixedDetector>(std::vector<std::string>{"plate", "face"}, studentBoxes, "student", ""),
      std::make_unique<FixedDetector>(std::vector<std::string>{"face"}, yunetBoxes, "yunet", ""));
  std::vector<SourcedBox> first;
  std::vector<SourcedBox> second;
  for (const Box& b : studentBoxes) first.push_back(SourcedBox{b, "student"});
  for (Box b : yunetBoxes) {
    b.cls = 1;
    second.push_back(SourcedBox{b, "yunet"});
  }
  const std::vector<SourcedBox> expected = mergeDetections(first, second, kMergeIou);
  const std::vector<SourcedBox> actual = merged.detectSourced(VideoFrame{});
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    const Box& e = expected[i].box;
    expectSourced(actual[i], e.cls, e.x1, e.y1, e.x2, e.y2, e.score, expected[i].src);
  }
}
```

`cpp/tests/redact_job_test.cpp`：

`AFaceModelAddsItsBoxesAndMergesTheOnesBothDetectorsFind` 末尾的

```cpp
  std::ifstream statsFile(s.outputDir / "stats.json");
  EXPECT_EQ(nlohmann::json::parse(statsFile).at("detector").get<std::string>(), "fake + yunet/CPU");
```

换成

```cpp
  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("detector").get<std::string>(), "fake + yunet/CPU");
  // Both detectors' own busy times, one item per keyframe each; detect is the wall time of the two together.
  EXPECT_EQ(s.detectStudent.items, s.keyframes);
  EXPECT_EQ(s.detectFace.items, s.keyframes);
  EXPECT_EQ(stats.at("stages").at("detect_student").at("items").get<int64_t>(), s.keyframes);
  EXPECT_EQ(stats.at("stages").at("detect_face").at("items").get<int64_t>(), s.keyframes);
  EXPECT_GT(s.detectFace.busySec, 0.0);
```

`ReportsTheMainStageAndTheSetupTime` 末尾加：

```cpp
  // Without a face model the student is the only detector: its busy time is detect's, the face model has none.
  EXPECT_DOUBLE_EQ(s.detectStudent.busySec, s.detect.busySec);
  EXPECT_EQ(s.detectStudent.items, s.detect.items);
  EXPECT_EQ(s.detectFace.items, 0);
  EXPECT_TRUE(stats.at("stages").contains("detect_student"));
  EXPECT_TRUE(stats.at("stages").contains("detect_face"));
```

`TheSummaryShowsTheSetupAndTheParallelStages`：在 `s.detect = StageStat{1.5, 50};` 之后加 `s.detectStudent = StageStat{1.2, 50};` 和 `s.detectFace = StageStat{0.6, 50};`，并在最后加：

```cpp
  EXPECT_NE(summary.find("detectors:     student 24.0 ms + face 12.0 ms per keyframe, run in parallel: 30.0 ms\n"),
            std::string::npos)
      << summary;
  s.detectFace = StageStat{};
  EXPECT_EQ(redactSummary(s).find("detectors:"), std::string::npos);  // one detector: no such line
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`no member named 'firstBusySec' in 'vc::MergedDetector'`、`no member named 'detectStudent' in 'vc::RedactStats'`。

- [ ] **Step 3: 实现**

`cpp/src/detect/merged_detector.h` 第 23–43 行换成：

```cpp
// Runs two detectors on every frame and merges their boxes (mergeDetections). The classes are the
// first detector's; the second's boxes are mapped onto them by class name. detectSourced() runs the
// two at the same time (spec 2026-10-19 §2): the first on the calling thread, the second on a thread of
// its own; each holds its own onnxruntime session and buffers, and both only read the frame.
class MergedDetector : public Detector {
 public:
  // Throws VcError when a class of `second` is not one of `first`'s classes.
  MergedDetector(std::unique_ptr<Detector> first, std::unique_ptr<Detector> second, float iouThreshold = kMergeIou);

  std::vector<Box> detect(const VideoFrame& frame) override;
  // Either detector throwing fails the call, once the other one has returned (the first's error wins).
  std::vector<SourcedBox> detectSourced(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override { return first_->classes(); }
  // "student+yunet"
  std::string source() const override;
  // "student/CoreML + yunet/CPU"
  std::string description() const override;

  // Seconds each detector has spent in detectSourced() so far; the two overlap.
  double firstBusySec() const { return firstBusySec_; }
  double secondBusySec() const { return secondBusySec_; }

 private:
  std::unique_ptr<Detector> first_;
  std::unique_ptr<Detector> second_;
  float iouThreshold_;
  std::vector<int> secondClassMap_;  // second's class index -> first's
  double firstBusySec_ = 0;
  double secondBusySec_ = 0;
};
```

`cpp/src/detect/merged_detector.cpp`：includes 加 `<chrono>`、`<exception>`、`<future>`；匿名命名空间里（`unionOf` 之后）加：

```cpp
// A detector's boxes and how long it took to find them.
struct Timed {
  std::vector<SourcedBox> boxes{};
  double sec = 0;
};

Timed timedDetect(Detector& detector, const VideoFrame& frame) {
  const auto started = std::chrono::steady_clock::now();
  Timed out;
  out.boxes = detector.detectSourced(frame);
  out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return out;
}
```

`MergedDetector::detectSourced`（第 80–90 行）换成：

```cpp
std::vector<SourcedBox> MergedDetector::detectSourced(const VideoFrame& frame) {
  // The second detector starts on a thread of its own, then the first runs here. The future is always
  // waited for, also when the first detector throws, so nothing is still reading the frame afterwards.
  std::future<Timed> pending =
      std::async(std::launch::async, [this, &frame] { return timedDetect(*second_, frame); });
  Timed a;
  std::exception_ptr firstError;
  try {
    a = timedDetect(*first_, frame);
  } catch (...) {
    firstError = std::current_exception();
  }
  Timed b;
  std::exception_ptr secondError;
  try {
    b = pending.get();
  } catch (...) {
    secondError = std::current_exception();
  }
  if (firstError) std::rethrow_exception(firstError);
  if (secondError) std::rethrow_exception(secondError);
  firstBusySec_ += a.sec;
  secondBusySec_ += b.sec;
  for (SourcedBox& s : b.boxes) {
    if (s.box.cls < 0 || static_cast<size_t>(s.box.cls) >= secondClassMap_.size()) {
      throw VcError("the " + second_->source() + " detector returned class index " + std::to_string(s.box.cls));
    }
    s.box.cls = secondClassMap_[static_cast<size_t>(s.box.cls)];
  }
  return mergeDetections(a.boxes, b.boxes, iouThreshold_);
}
```

`cpp/CMakeLists.txt`：把第 55 行的 `find_package(Threads REQUIRED)` 挪到 `add_library(vc_detect_lib STATIC` 之前，并把 `vc_detect_lib` 的链接行改成

```cmake
target_link_libraries(vc_detect_lib PUBLIC vc_core vc_onnxruntime Threads::Threads PRIVATE vc_nlohmann_json)
```

（`vc_redact_lib` 那一行保留 `Threads::Threads`，不用改。）

`cpp/apps/vc_redact/redact_job.h` 的 `RedactStats`：`StageStat detect;` 一行和它的注释换成

```cpp
  StageStat detect;         // items = keyframes; wall time of each keyframe's detection (with a face model both
                            // detectors at once, plus the merge)
  StageStat detectStudent;  // the student (or fake) detector's own busy time; equals detect without a face model
  StageStat detectFace;     // the face model's own busy time (it overlaps detectStudent); {0, 0} without one
```

`cpp/apps/vc_redact/redact_job.cpp`：

- `writeStats` 里 `j["stages"]["detect"] = stageJson(s.detect);` 之后加

```cpp
  j["stages"]["detect_student"] = stageJson(s.detectStudent);
  j["stages"]["detect_face"] = stageJson(s.detectFace);
```

- `redactSummary` 里 `stages (fps):` 那一行之后加

```cpp
  if (s.detectFace.items > 0) {
    const auto perKeyframeMs = [](const StageStat& stage) {
      return stage.items > 0 ? 1000.0 * stage.busySec / static_cast<double>(stage.items) : 0.0;
    };
    out += formatted("detectors:     student %.1f ms + face %.1f ms per keyframe, run in parallel: %.1f ms\n",
                     perKeyframeMs(s.detectStudent), perKeyframeMs(s.detectFace), perKeyframeMs(s.detect));
  }
```

- `runRedact` 里 `stats.detector = detector->description();` 之后加

```cpp
    if (const auto* merged = dynamic_cast<const MergedDetector*>(detector.get())) {
      stats.detectStudent = StageStat{merged->firstBusySec(), stats.detect.items};
      stats.detectFace = StageStat{merged->secondBusySec(), stats.detect.items};
    } else {
      stats.detectStudent = stats.detect;
    }
```

`cpp/README.md`：

- "合并"一条的"每个关键帧先跑学生、再跑 YuNet（`stats.json` 的 `detect` 计时包含两者和合并）"换成"每个关键帧学生和 YuNet 同时跑（两个线程，各用自己的 onnxruntime 会话和缓冲区），两者都返回后再合并；任一出错整次运行失败。`stats.json` 的 `detect` 是每个关键帧检测的墙钟时间之和，`detect_student`、`detect_face` 是两个检测器各自的忙碌时间（两者重叠），统计行多一行 `detectors:` 给出每个关键帧的毫秒数"。
- `stages` 那段说明里 `detect` 的解释后面加"（有人脸模型时另有 `detect_student`、`detect_face`）"。

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && "${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='DetectionMerge.*:MergedDetector.*:Detector.*:RedactJob.*'
```

预期：全部 PASSED，零警告。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/detect/merged_detector.h cpp/src/detect/merged_detector.cpp cpp/CMakeLists.txt cpp/apps/vc_redact/redact_job.h cpp/apps/vc_redact/redact_job.cpp cpp/tests/merged_detector_test.cpp cpp/tests/redact_job_test.cpp cpp/README.md
git commit -m "perf: run the student and face detectors in parallel on each keyframe"
```

---

### Task 4: `BoxTimeline` 改成"一个目标一条轨迹、每帧一个框"

本任务实现本设计 §3.1 的插值（不含迟滞，迟滞在 Task 5）、§3.2 的删除、§3.3 的丢失与幽灵，以及 `masks.jsonl` 的 `track` 和新的 `origin` 取值。

**Files:**
- Rewrite: `cpp/src/track/box_timeline.h`, `cpp/src/track/box_timeline.cpp`
- Rewrite: `cpp/tests/box_timeline_test.cpp`
- Modify: `cpp/apps/vc_redact/records.cpp`（`maskRecord`）、`cpp/apps/vc_redact/records.h`（注释）
- Modify: `cpp/apps/vc_redact/redact_job.cpp`（`Processor::onVideo`、`Processor::onEnd` 里对 `advance` / `finish` 的调用）
- Modify: `cpp/README.md`、`docs/USAGE.md`
- Test: `cpp/tests/box_timeline_test.cpp`、`cpp/tests/records_test.cpp`、`cpp/tests/redact_job_test.cpp`

**Interfaces:**
- Consumes: `Box`（`detect/box.h`）；`TimelineConfig` 的五个现有字段与含义不变。
- Produces（Task 5、6、7 和 `vc_eval` 依赖）:
  - `enum class MaskOrigin { Detected, Interpolated, Held };`，`toString` 给出 `"detected" | "interpolated" | "held"`（`Extended` 删除）。
  - `struct MaskBox { int cls; float x1, y1, x2, y2; MaskOrigin origin; int64_t track = 0; };` —— 轨迹编号从 1 起，同一视频内不重复，`finish()` 后重置。
  - `struct TimelineOutput { std::vector<std::vector<MaskBox>> masks{}; std::vector<bool> confirmed{}; };` —— `confirmed` 对应"这批帧起始的那个关键帧"的每个框（`Keyframe::boxes` 顺序），本任务里全是 `true`（Task 6 才有 `false`）；第一次 `advance` 返回空。
  - `TimelineOutput BoxTimeline::advance(const std::vector<FrameTime>& frames, const Keyframe& next);`
  - `TimelineOutput BoxTimeline::finish(const std::vector<FrameTime>& frames);`
  - `masks.jsonl` 每个框的字段顺序 `cls, x1, y1, x2, y2, track, origin`。
- 行为（本设计 §3.1–§3.3）：
  - 两端都检出（匹配上）：每帧一个框 = 两端检测框按 (t − k0)/(k1 − k0) 线性插值（中心和宽高），按 `expand_fraction` 外扩、再加 `motion_expand_fraction` × 本间隔每个轴的位移、撑到 `min_box_px`、裁到画面内。k0 那一帧 `origin` 为 `detected`（幽灵为 `held`），其余为 `interpolated`。
  - 在 k1 没配上：[k0, k1) 原地保持，有速度就按速度外推，位移每个轴各自截到半个框宽（x）/ 框高（y）（`std::clamp`，整段不超过）；外扩的运动项用截断后的整段位移。k0 为 `detected`，其余为 `held`。轨迹变成幽灵，位置是保持到 k1 的位置。
  - 幽灵在 k2 配上：[k1, k2) 从保持位置插值到 k2 的框（k1 为 `held`，其余 `interpolated`）；没配上：轨迹在 k1 结束，[k1, k2) 不打码。
  - 新出现的框只从它自己的关键帧起打码；宽松配对和第一次配对不再保留两端原地框；不再有扫过区域；没配上的轨迹不再留两份插值。宽松配对配上的轨迹不再另留幽灵。
  - 速度：仍从轨迹最后一次真实检测（锚点）量到新检测，再除以帧数。
  - `finish()`：最后一个关键帧上检出的轨迹按上面的规则保持（运动项用"尾段帧数"对应的截断位移）；幽灵不打码。

**测试改动清单（`cpp/tests/box_timeline_test.cpp` 整个重写，下面是与旧文件的对应）：**

| 旧测试 | 处理 | 原因 |
|---|---|---|
| `InterpolatesMatchedBoxes`、`FallsBackToFrameIndexWhenTimestampsStall`、`PredictsWithVelocityToMatchFastMovers`、`ExpandsByFractionAndMotionAndEnforcesTheMinimumSize`、`ClipsToTheFrame`、`ExpandsForMotionInAnyDirection`、`MatchesGreedilyByCost`、`InterpolatesByPtsForUnevenTimestamps`、`ClipsAtTheRightAndBottomOfANonSquareFrame`、`OneTrackMatchesAtMostOneBox`、`RejectsNonFiniteKeyframeBoxes`、`FallsBackToTheIndexForAFrameWithoutAUsableTimestamp`、`FallsBackToTheIndexWhenAFrameLiesOutsideTheKeyframeTimestamps`、`NeverEmitsAMaskWithNonFiniteCoordinates` | 改写期望 | 每帧框数从 3（插值 + 两端原地框）或 4 变成 1，宽松 / 首次配对不再保留两端框（§3.2）；`OneTrackMatchesAtMostOneBox` 并入 `TracksKeepTheirIdsAndNewTargetsGetNewOnes` |
| `ExtendsDisappearingAndAppearingBoxesOverTheInterval` | 拆成 `HoldsATrackMissedAtTheNextKeyframe` 和 `NeverMasksABoxBeforeItsFirstKeyframe` | 消失的框改为 `held`；新出现的框不再向前延伸（§3.2、§3.5） |
| `NeverMatchesAcrossClasses` | 改写 | 新出现的车牌不再向前延伸，人脸改为 `held`（§3.2、§3.3） |
| `FinishExtendsForwardWithVelocity`、`FinishCapsExtrapolationAtOneInterval` | 合并为 `FinishHoldsAlongTheVelocityUpToHalfABoxSide` | 外推改为截到半个框边（§3.3），不再"扫过两侧" |
| `FinishMarginUsesAbsoluteVelocityAndCap` | 改写为 `FinishMarginIsTheCappedShift` | 运动外扩取截断后的位移 |
| `ExtrapolatesADisappearingBoxAlongItsVelocity`、`ExtensionMarginUsesAbsoluteVelocity` | 合并为 `HeldMasksFollowTheVelocityAndGrowByTheCappedShift` | 外推截到半个框边，运动外扩取截断后的位移（§3.3） |
| `BridgesOneMissedKeyframeWithAGhostTrack` | 改写为 `ReMatchesAGhostFromItsHeldPosition` | 幽灵从保持位置插值，不再叠"下一个框原地"和"从最后真实检测插值"两份（§3.2、§3.3） |
| `DropsAGhostAfterTwoMissedKeyframes`、`UnmatchedGhostIsExtendedAtItsKeyFrame` | 合并为 `AGhostMissedAgainEndsUnmasked` | 没再配上的幽灵那一段不打码（§3.3） |
| `FinishExtendsAGhostTrack` | 改写为 `FinishDoesNotMaskAGhost` | 视频结尾没再配上的幽灵不打码（§3.3） |
| `MatchedGhostVelocityIsRecomputed`、`MatchedGhostVelocityIsMeasuredFromItsLastDetection`、`AGhostWithoutVelocityStaysInPlace` | 改写为 `AReMatchedGhostTakesItsVelocityFromItsLastDetection`、`AGhostWithoutAVelocityIsHeldInPlaceAndReMatched` | 速度规则不变，期望值按 `held` 重算 |
| `RejectsCallsOutOfOrder`、`RejectsNonIncreasingFrames`、`RejectsNonFiniteConfig`、`RejectsNonFiniteKeyframeTimestamps`、`MatchesATargetThatStops`、`InterpolatesBoxesAtTheFloatLimitsWithoutOverflowing`、`ALooseMatchNeedsCloseEnoughBoxesOfASimilarSize` | 保留（只改 `.masks` 和 `Held`） | 行为不变 |
| `CoversFastMoversOnEveryFrame`、`CoversTwoTargetsThatCrossEachOther`、`CoversPlatesMovingVertically` | 保留，覆盖率只查到最后一个关键帧为止，并加"每轨每帧一框" | 最后一个关键帧之后只外推半个框边（§3.3），快目标会离开打码框；插值部分从第 1 帧起就有（宽松配对直接插值） |
| `ALooseMatchInterpolatesAndHoldsBothEndpoints` | 改写为 `ALooseMatchInterpolatesAlone` | §3.2 删除两端原地框 |
| `ALooseMatchSeedsTheVelocity` | 改写 | 不再有旧轨迹的幽灵；尾段按截断外推 |
| `ALooseMatchKeepsTheExtrapolationOfATrackWithAVelocity` | 删除 | 宽松配对不再保留没配上时的扫过区域（§3.2） |
| `AnUnmatchedTrackMasksTheRegionItSweeps` | 删除 | 扫过区域整个删除（§3.2）；保持行为由 `HeldMasksFollowTheVelocity…`、`FinishHolds…` 覆盖 |
| `CoversATargetThatStopsAfterAMissedKeyframe` | 删除 | 它依赖"下一个框在整个区间原地保持"（§3.2 删除） |
| `CoversTwoNearbyTargetsPairedTheWrongWayRound`、`AFirstPairingHoldsBothEndBoxes` | 删除 | 依赖首次配对保留两端框（§3.2 删除）；首次配对现在只插值，由 `InterpolatesMatchedBoxes` 覆盖 |
| `AMatchedGhostIsAlsoInterpolatedFromItsLastDetection`、`CoversATargetThatStopsOrTurnsRoundWhileMissed` | 删除 | "从最后一次真实检测插值"的补充框和扫过区域都删除（§3.2）；漏检期间掉头 / 停下盖不全是 §3.5 接受的代价 |
| `ALooseMatchLeavesAGhostOfTheOldTrack`、`FinishCoversAPlateMissedAtTheLastKeyframeNextToALookAlike` | 删除，换成 `EveryTrackHasAtMostOneMaskPerFrame`（同一场景） | 宽松配对的旧轨迹不再留幽灵（一个目标一个框）；新测试在同一场景上证明每轨每帧至多一框、只有三种 origin |
| （新） | `TracksKeepTheirIdsAndNewTargetsGetNewOnes`、`EveryTrackHasAtMostOneMaskPerFrame` | §3.3 的 `track`；§7"每条轨迹每帧至多一个框"、"不再产生 extended" |

- [ ] **Step 1: 写失败的测试**

把 `cpp/tests/box_timeline_test.cpp` 整个换成：

```cpp
#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;
constexpr int kPlate = 1;

// Margins and minimum size as given, everything else at its default.
TimelineConfig config(double expand, double motion, double minBox) {
  TimelineConfig c;
  c.expandFraction = expand;
  c.motionExpandFraction = motion;
  c.minBoxPx = minBox;
  return c;
}

// No margins and no minimum size: each mask is exactly its (interpolated or held) box.
TimelineConfig exact() { return config(0.0, 0.0, 0.0); }

Box face(float x1, float y1, float x2, float y2) { return Box{kFace, x1, y1, x2, y2, 0.9f}; }

// Frames first..last-1 at 60 fps pts, or all at `frozenPts` when it is >= 0.
std::vector<FrameTime> frames(int64_t first, int64_t last, double frozenPts = -1) {
  std::vector<FrameTime> out;
  for (int64_t i = first; i < last; ++i) out.push_back({i, frozenPts >= 0 ? frozenPts : i * 1000.0 / 60.0});
  return out;
}

Keyframe key(int64_t index, std::vector<Box> boxes, double frozenPts = -1) {
  return Keyframe{{index, frozenPts >= 0 ? frozenPts : index * 1000.0 / 60.0}, std::move(boxes)};
}

void expectBox(const MaskBox& m, float x1, float y1, float x2, float y2, MaskOrigin origin) {
  EXPECT_NEAR(m.x1, x1, 1e-3);
  EXPECT_NEAR(m.y1, y1, 1e-3);
  EXPECT_NEAR(m.x2, x2, 1e-3);
  EXPECT_NEAR(m.y2, y2, 1e-3);
  EXPECT_EQ(m.origin, origin);
}

// Fraction of the box [x1, x2) x [y1, y2) inside the union of the masks, sampled on a 0.5 px grid.
double coverage(const std::vector<MaskBox>& masks, double x1, double y1, double x2, double y2) {
  int samples = 0;
  int inside = 0;
  for (double y = y1 + 0.25; y < y2; y += 0.5) {
    for (double x = x1 + 0.25; x < x2; x += 0.5) {
      ++samples;
      if (std::any_of(masks.begin(), masks.end(),
                      [=](const MaskBox& m) { return x >= m.x1 && x < m.x2 && y >= m.y1 && y < m.y2; })) {
        ++inside;
      }
    }
  }
  return samples == 0 ? 1.0 : static_cast<double>(inside) / samples;
}

double coverage(const std::vector<MaskBox>& masks, const Box& b) { return coverage(masks, b.x1, b.y1, b.x2, b.y2); }

size_t countOrigin(const std::vector<MaskBox>& masks, MaskOrigin origin) {
  return static_cast<size_t>(
      std::count_if(masks.begin(), masks.end(), [origin](const MaskBox& m) { return m.origin == origin; }));
}

// No frame carries two masks of one track (spec 2026-10-19 §3).
void expectOneMaskPerTrack(const std::vector<std::vector<MaskBox>>& masks) {
  for (size_t f = 0; f < masks.size(); ++f) {
    std::set<int64_t> seen;
    for (const MaskBox& m : masks[f]) {
      EXPECT_TRUE(seen.insert(m.track).second) << "track " << m.track << " twice on frame " << f;
    }
  }
}

// A target moving uniformly: its box on frame f.
struct Mover {
  int cls = kFace;
  double x0 = 0;
  double y0 = 0;
  double w = 0;
  double h = 0;
  double vx = 0;  // px per frame
  double vy = 0;

  Box at(int64_t f) const {
    const double x = x0 + vx * static_cast<double>(f);
    const double y = y0 + vy * static_cast<double>(f);
    return Box{cls, static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + w), static_cast<float>(y + h),
               0.9f};
  }
};

constexpr int64_t kStep = 12;  // 60 fps video, 5 fps keyframes

// A perfect detector (every mover found on every keyframe) over `intervals` keyframe intervals, then
// finish() for one more interval. Returns the masks of every frame, by frame index.
std::vector<std::vector<MaskBox>> runPerfectDetector(BoxTimeline& timeline, const std::vector<Mover>& movers,
                                                     int64_t intervals) {
  const auto boxesAt = [&](int64_t f) {
    std::vector<Box> boxes;
    for (const Mover& m : movers) boxes.push_back(m.at(f));
    return boxes;
  };
  std::vector<std::vector<MaskBox>> all;
  timeline.advance({}, key(0, boxesAt(0)));
  for (int64_t k = 1; k <= intervals; ++k) {
    auto masks = timeline.advance(frames((k - 1) * kStep, k * kStep), key(k * kStep, boxesAt(k * kStep))).masks;
    for (auto& m : masks) all.push_back(std::move(m));
  }
  auto tail = timeline.finish(frames(intervals * kStep, (intervals + 1) * kStep)).masks;
  for (auto& m : tail) all.push_back(std::move(m));
  return all;
}

void expectFinite(const std::vector<std::vector<MaskBox>>& masks) {
  for (size_t k = 0; k < masks.size(); ++k) {
    for (const MaskBox& m : masks[k]) {
      EXPECT_TRUE(std::isfinite(m.x1) && std::isfinite(m.y1) && std::isfinite(m.x2) && std::isfinite(m.y2))
          << "frame offset " << k;
    }
  }
}

TEST(BoxTimeline, InterpolatesMatchedBoxes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_TRUE(timeline.advance({}, key(0, {face(100, 100, 120, 120)})).masks.empty());
  const TimelineOutput out = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  EXPECT_EQ(out.confirmed, std::vector<bool>{true});  // the box of keyframe 0 is masked
  ASSERT_EQ(out.masks.size(), 12u);
  for (const auto& m : out.masks) {
    ASSERT_EQ(m.size(), 1u);  // one target, one mask: no end boxes held next to the interpolation
    EXPECT_EQ(m[0].track, 1);
  }
  expectBox(out.masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  expectBox(out.masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToFrameIndexWhenTimestampsStall) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}, 0.0));
  const auto masks = timeline.advance(frames(0, 12, 0.0), key(12, {face(112, 100, 132, 120)}, 0.0)).masks;
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

// A track missed at the next keyframe is held at its box over the interval (no velocity yet: in place).
TEST(BoxTimeline, HoldsATrackMissedAtTheNextKeyframe) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {})).masks;
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    ASSERT_EQ(masks[k].size(), 1u);
    expectBox(masks[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

// The frames before a target's first detection are not masked (spec §3.5).
TEST(BoxTimeline, NeverMasksABoxBeforeItsFirstKeyframe) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {}));
  const auto before = timeline.advance(frames(0, 12), key(12, {face(300, 300, 320, 320)})).masks;
  ASSERT_EQ(before.size(), 12u);
  for (const auto& m : before) EXPECT_TRUE(m.empty());
  const auto after = timeline.advance(frames(12, 24), key(24, {face(300, 300, 320, 320)})).masks;
  ASSERT_EQ(after[0].size(), 1u);
  expectBox(after[0][0], 300, 300, 320, 320, MaskOrigin::Detected);
}

TEST(BoxTimeline, NeverMatchesAcrossClasses) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {Box{kPlate, 100, 100, 120, 120, 0.9f}})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // the face, held; the plate starts at 12
  EXPECT_EQ(masks[6][0].cls, kFace);
  EXPECT_EQ(masks[6][0].origin, MaskOrigin::Held);
  const auto next = timeline.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(next[0].size(), 1u);  // the plate, held; the face's ghost is not matched at 24 and masks nothing
  EXPECT_EQ(next[0][0].cls, kPlate);
  EXPECT_EQ(next[0][0].track, 2);
}

TEST(BoxTimeline, PredictsWithVelocityToMatchFastMovers) {
  // 18 px per interval, then 22 px: the first pass needs the prediction (22 / 20 > 1).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
  EXPECT_EQ(masks[6][0].track, 1);

  // Without a velocity only the second pass pairs them, and the pair is interpolated alone all the same.
  BoxTimeline fresh(exact(), 1000, 1000);
  fresh.advance({}, key(12, {face(118, 100, 138, 120)}));
  const auto loose = fresh.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)})).masks;
  ASSERT_EQ(loose[6].size(), 1u);
  expectBox(loose[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ExpandsByFractionAndMotionAndEnforcesTheMinimumSize) {
  BoxTimeline timeline(config(0.15, 0.5, 16.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 110)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 110)})).masks;
  // x margin 0.15 * 20 + 0.5 * 12 = 9; y margin 1.5 gives 13 px, grown to 16 around y = 105
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 91, 97, 129, 113, MaskOrigin::Detected);
}

TEST(BoxTimeline, ClipsToTheFrame) {
  BoxTimeline timeline(config(0.0, 0.0, 16.0), 1000, 1000);
  timeline.advance({}, key(0, {face(0, 0, 4, 4)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(0, 0, 4, 4)})).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 0, 0, 10, 10, MaskOrigin::Detected);
}

// After the last keyframe a track is held, moving along its velocity by at most half its width (spec §3.3).
TEST(BoxTimeline, FinishHoldsAlongTheVelocityUpToHalfABoxSide) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // 1 px per frame
  const auto masks = timeline.finish(frames(12, 42)).masks;
  ASSERT_EQ(masks.size(), 30u);
  for (const auto& m : masks) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }
  expectBox(masks[0][0], 112, 100, 132, 120, MaskOrigin::Detected);
  expectBox(masks[3][0], 115, 100, 135, 120, MaskOrigin::Held);   // frame 15: 3 px along
  expectBox(masks[10][0], 122, 100, 142, 120, MaskOrigin::Held);  // frame 22: 10 px, half the 20 px width
  expectBox(masks[29][0], 122, 100, 142, 120, MaskOrigin::Held);  // frame 41: still 10 px
}

TEST(BoxTimeline, RejectsCallsOutOfOrder) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_THROW(timeline.advance(frames(0, 3), key(3, {})), std::invalid_argument);
  timeline.advance({}, key(0, {}));
  EXPECT_THROW(timeline.advance(frames(1, 6), key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.advance(frames(0, 6), key(5, {})), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{0.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_STREQ(toString(MaskOrigin::Interpolated), "interpolated");
  EXPECT_STREQ(toString(MaskOrigin::Held), "held");
  EXPECT_STREQ(toString(MaskOrigin::Detected), "detected");
}

// Missed at 24, found again at 36: the ghost is matched and interpolated from where it was held (spec §3.3).
TEST(BoxTimeline, ReMatchesAGhostFromItsHeldPosition) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // 1 px per frame
  const auto gap = timeline.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(gap[6].size(), 1u);
  expectBox(gap[6][0], 118, 100, 138, 120, MaskOrigin::Held);  // frame 18: 6 px along

  const auto bridged = timeline.advance(frames(24, 36), key(36, {face(136, 100, 156, 120)})).masks;
  for (const auto& m : bridged) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }
  // held to 112 + min(12, 10) = 122 at frame 24, then interpolated to 136 at 36
  expectBox(bridged[0][0], 122, 100, 142, 120, MaskOrigin::Held);
  expectBox(bridged[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
}

// Missed at 24 and again at 36: the track ended at 24, and frames 24 to 35 are not masked.
TEST(BoxTimeline, AGhostMissedAgainEndsUnmasked) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto held = timeline.advance(frames(12, 24), key(24, {})).masks;
  for (const auto& m : held) EXPECT_EQ(m.size(), 1u);
  const auto ghost = timeline.advance(frames(24, 36), key(36, {})).masks;
  ASSERT_EQ(ghost.size(), 12u);
  for (const auto& m : ghost) EXPECT_TRUE(m.empty());
  const auto after = timeline.advance(frames(36, 48), key(48, {})).masks;
  for (const auto& m : after) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, FinishDoesNotMaskAGhost) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto tail = timeline.finish(frames(24, 30)).masks;
  ASSERT_EQ(tail.size(), 6u);
  for (const auto& m : tail) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, ExpandsForMotionInAnyDirection) {
  BoxTimeline timeline(config(0.0, 0.5, 0.0), 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)})).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  // displacement 12 px in x and in y, moving left / up: 0.5 * 12 = 6 px on every side
  expectBox(masks[0][0], 106, 106, 138, 138, MaskOrigin::Detected);
}

TEST(BoxTimeline, HeldMasksFollowTheVelocityAndGrowByTheCappedShift) {
  BoxTimeline plain(exact(), 1000, 1000);
  plain.advance({}, key(0, {face(100, 100, 120, 120)}));
  plain.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = plain.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18: 1 px per frame, 6 px along
  expectBox(masks[6][0], 118, 100, 138, 120, MaskOrigin::Held);

  BoxTimeline moving(config(0.0, 0.5, 0.0), 1000, 1000);
  moving.advance({}, key(0, {face(112, 112, 132, 132)}));
  moving.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto gone = moving.advance(frames(12, 24), key(24, {})).masks;
  // The shift over the interval is 12 px, capped at 10 (half the side): 0.5 * 10 = 5 px of margin per side.
  ASSERT_EQ(gone[0].size(), 1u);
  expectBox(gone[0][0], 95, 95, 125, 125, MaskOrigin::Detected);
  ASSERT_EQ(gone[6].size(), 1u);  // frame 18: 6 px up and left
  expectBox(gone[6][0], 89, 89, 119, 119, MaskOrigin::Held);
}

TEST(BoxTimeline, FinishMarginIsTheCappedShift) {
  BoxTimeline timeline(config(0.0, 0.5, 0.0), 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto masks = timeline.finish(frames(12, 40)).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 95, 95, 125, 125, MaskOrigin::Detected);  // 28 tail frames: shift capped at 10, margin 5
  ASSERT_EQ(masks[20].size(), 1u);                                  // frame 32: 20 px along, capped at 10
  expectBox(masks[20][0], 85, 85, 115, 115, MaskOrigin::Held);
}

TEST(BoxTimeline, MatchesGreedilyByCost) {
  // Both previous faces are candidates for the next one (costs 1.25 and 0.25, limit 2.0): the cheaper pair
  // wins and the other face is held.
  TimelineConfig c = exact();
  c.matchMaxDistance = 2.0;
  BoxTimeline timeline(c, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(130, 100, 150, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(125, 100, 145, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 2u);
  expectBox(masks[6][0], 100, 100, 120, 120, MaskOrigin::Held);
  EXPECT_EQ(masks[6][0].track, 1);
  expectBox(masks[6][1], 127.5f, 100, 147.5f, 120, MaskOrigin::Interpolated);
  EXPECT_EQ(masks[6][1].track, 2);
}

TEST(BoxTimeline, InterpolatesByPtsForUnevenTimestamps) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> uneven = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, 200.0}, {4, 210.0}};
  const auto masks = timeline.advance(uneven, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  ASSERT_EQ(masks.size(), 5u);
  ASSERT_EQ(masks[3].size(), 1u);  // pts 200 of 270: 100 + 27 * 200 / 270
  expectBox(masks[3][0], 120, 100, 150, 130, MaskOrigin::Interpolated);

  // A repeated pts is not usable: positions fall back to the frame index.
  BoxTimeline stalled(exact(), 1000, 1000);
  stalled.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> repeated = {{0, 0.0}, {1, 10.0}, {2, 10.0}, {3, 30.0}};
  const auto byIndex = stalled.advance(repeated, Keyframe{{4, 100.0}, {face(124, 100, 154, 130)}}).masks;
  ASSERT_EQ(byIndex.size(), 4u);
  ASSERT_EQ(byIndex[1].size(), 1u);
  expectBox(byIndex[1][0], 106, 100, 136, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[2].size(), 1u);
  expectBox(byIndex[2][0], 112, 100, 142, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[3].size(), 1u);
  expectBox(byIndex[3][0], 118, 100, 148, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ClipsAtTheRightAndBottomOfANonSquareFrame) {
  BoxTimeline timeline(config(0.0, 0.0, 16.0), 640, 360);
  timeline.advance({}, key(0, {face(630, 350, 640, 360)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(630, 350, 640, 360)})).masks;
  // 10 px grown to 16 around the centre (635, 355) is 627..643 x 347..363, clipped to 640 x 360
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 627, 347, 640, 360, MaskOrigin::Detected);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 627, 347, 640, 360, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, MatchesATargetThatStops) {
  // The face speeds up to 3 px per frame, then stops: the velocity predicts a centre 36 px away
  // (cost 1.8), but the box is exactly where it was (zero-velocity cost 0).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  timeline.advance(frames(12, 24), key(24, {face(154, 100, 174, 120)}));
  const auto masks = timeline.advance(frames(24, 36), key(36, {face(154, 100, 174, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 30
  expectBox(masks[6][0], 154, 100, 174, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, RejectsNonIncreasingFrames) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  std::vector<FrameTime> repeated = frames(0, 6);
  repeated[3] = repeated[2];
  EXPECT_THROW(timeline.advance(repeated, key(6, {})), std::invalid_argument);
  std::vector<FrameTime> swapped = frames(0, 6);
  std::swap(swapped[3], swapped[4]);
  EXPECT_THROW(timeline.advance(swapped, key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.finish(repeated), std::invalid_argument);
  EXPECT_THROW(timeline.finish(swapped), std::invalid_argument);
  // The rejected calls left the timeline untouched.
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(100, 100, 120, 120)})).masks;
  EXPECT_EQ(masks.size(), 6u);
}

TEST(BoxTimeline, RejectsNonFiniteConfig) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    EXPECT_THROW(BoxTimeline(TimelineConfig{bad, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, bad, 0.1, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, bad, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, bad}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, bad}, 10, 10), std::invalid_argument);
  }
  EXPECT_THROW(BoxTimeline(TimelineConfig{-1.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, -0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, -0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, -1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, 0.0}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, -1.0}, 10, 10), std::invalid_argument);
}

// Each target keeps its track id from keyframe to keyframe whatever the order of the boxes; a new target gets
// the next id. Masks come in the order of the tracks on the keyframe they start at.
TEST(BoxTimeline, TracksKeepTheirIdsAndNewTargetsGetNewOnes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(500, 100, 520, 120)}));
  const auto first =
      timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120), face(500, 112, 520, 132)})).masks;
  ASSERT_EQ(first[6].size(), 2u);
  EXPECT_EQ(first[6][0].track, 1);
  EXPECT_EQ(first[6][1].track, 2);
  const auto second = timeline
                          .advance(frames(12, 24), key(24, {face(500, 124, 520, 144), face(800, 800, 820, 820),
                                                            face(124, 100, 144, 120)}))
                          .masks;
  ASSERT_EQ(second[6].size(), 2u);  // the face at 800 starts at 24
  EXPECT_EQ(second[6][0].track, 1);
  EXPECT_EQ(second[6][1].track, 2);
  const auto third = timeline.advance(frames(24, 36), key(36, {})).masks;
  ASSERT_EQ(third[0].size(), 3u);  // all three held, in the order of keyframe 24's boxes
  EXPECT_EQ(third[0][0].track, 2);
  EXPECT_EQ(third[0][1].track, 3);
  EXPECT_EQ(third[0][2].track, 1);
}

// Detected at 12 (centre 122), missed at 24, found at 36 (centre (150, 118)): the velocity runs from the last
// real detection, (28, 8) / 24 frames, not from the ghost's held position.
TEST(BoxTimeline, AReMatchedGhostTakesItsVelocityFromItsLastDetection) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  timeline.advance(frames(24, 36), key(36, {face(140, 108, 160, 128)}));
  const auto masks = timeline.finish(frames(36, 42)).masks;
  ASSERT_EQ(masks[3].size(), 1u);  // frame 39: (7/6, 1/3) px per frame for 3 frames
  expectBox(masks[3][0], 143.5f, 109, 163.5f, 129, MaskOrigin::Held);
}

TEST(BoxTimeline, AGhostWithoutAVelocityIsHeldInPlaceAndReMatched) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto held = timeline.advance(frames(0, 12), key(12, {})).masks;  // no velocity yet: held where it was
  for (size_t k = 0; k < held.size(); ++k) {
    ASSERT_EQ(held[k].size(), 1u);
    expectBox(held[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
  const auto bridged = timeline.advance(frames(12, 24), key(24, {face(118, 100, 138, 120)})).masks;
  ASSERT_EQ(bridged[0].size(), 1u);
  expectBox(bridged[0][0], 100, 100, 120, 120, MaskOrigin::Held);
  ASSERT_EQ(bridged[6].size(), 1u);
  expectBox(bridged[6][0], 109, 100, 129, 120, MaskOrigin::Interpolated);
  // From the detection at 0 to the one at 24: 18 px in 24 frames (from the ghost it would be 18 in 12).
  const auto tail = timeline.finish(frames(24, 31)).masks;
  ASSERT_EQ(tail.size(), 7u);
  expectBox(tail[0][0], 118, 100, 138, 120, MaskOrigin::Detected);
  expectBox(tail[6][0], 122.5f, 100, 142.5f, 120, MaskOrigin::Held);
}

TEST(BoxTimeline, RejectsNonFiniteKeyframeBoxes) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -inf}) {
    const std::vector<Box> broken = {face(bad, 100, 120, 120), face(100, bad, 120, 120), face(100, 100, bad, 120),
                                     face(100, 100, 120, bad)};
    for (const Box& box : broken) {
      BoxTimeline first(exact(), 1000, 1000);  // as the very first keyframe
      EXPECT_THROW(first.advance({}, key(0, {box})), std::invalid_argument);
      // The rejected call left the timeline untouched.
      EXPECT_TRUE(first.advance({}, key(0, {face(100, 100, 120, 120)})).masks.empty());

      BoxTimeline later(exact(), 1000, 1000);  // and as a later one
      later.advance({}, key(0, {face(100, 100, 120, 120)}));
      EXPECT_THROW(later.advance(frames(0, 12), key(12, {face(100, 100, 120, 120), box})), std::invalid_argument);
      const auto masks = later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)})).masks;
      ASSERT_EQ(masks.size(), 12u);
      ASSERT_EQ(masks[6].size(), 1u);
      expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
    }
  }
}

TEST(BoxTimeline, RejectsNonFiniteKeyframeTimestamps) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    BoxTimeline first(exact(), 1000, 1000);
    EXPECT_THROW(first.advance({}, Keyframe{{0, bad}, {face(100, 100, 120, 120)}}), std::invalid_argument);
    BoxTimeline later(exact(), 1000, 1000);
    later.advance({}, key(0, {face(100, 100, 120, 120)}));
    EXPECT_THROW(later.advance(frames(0, 12), Keyframe{{12, bad}, {face(112, 100, 132, 120)}}),
                 std::invalid_argument);
    EXPECT_EQ(later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)})).masks.size(), 12u);
  }
}

TEST(BoxTimeline, FallsBackToTheIndexForAFrameWithoutAUsableTimestamp) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> broken = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, nan}, {4, 210.0}};
  const auto masks = timeline.advance(broken, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  ASSERT_EQ(masks.size(), 5u);
  expectFinite(masks);
  // By index (pts alone would put frame 1 at 101.0): 100 + 27 * k / 5.
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 116.2f, 100, 146.2f, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToTheIndexWhenAFrameLiesOutsideTheKeyframeTimestamps) {
  // Keyframes at index 0 (pts 100) and 5 (pts 270); x1 100 -> 127.
  const auto run = [](const std::vector<double>& pts) {
    BoxTimeline timeline(exact(), 1000, 1000);
    timeline.advance({}, Keyframe{{0, 100.0}, {face(100, 100, 130, 130)}});
    std::vector<FrameTime> list;
    for (size_t i = 0; i < pts.size(); ++i) list.push_back({static_cast<int64_t>(i), pts[i]});
    return timeline.advance(list, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  };
  // All inside: by pts, frame 1 sits at (110 - 100) / 170 of the interval.
  auto masks = run({100, 110, 120, 130, 140});
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 100.0f + 27.0f * 10.0f / 170.0f, 100, 130.0f + 27.0f * 10.0f / 170.0f, 130,
            MaskOrigin::Interpolated);
  // A frame before the previous keyframe's pts, at the next keyframe's pts, or after it: by index
  // (100 + 27 * 1 / 5 = 105.4 for frame 1).
  for (const std::vector<double>& pts : {std::vector<double>{90, 110, 120, 130, 140},
                                         std::vector<double>{100, 110, 120, 130, 270},
                                         std::vector<double>{100, 110, 120, 130, 300}}) {
    masks = run(pts);
    ASSERT_EQ(masks.size(), 5u);
    ASSERT_EQ(masks[1].size(), 1u);
    expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  }
}

TEST(BoxTimeline, NeverEmitsAMaskWithNonFiniteCoordinates) {
  // Finite coordinates whose width overflows a float: with no expansion fraction the margin would be
  // 0 * inf = NaN if computed in float. The target must stay masked, so every frame carries a finite mask.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, 3e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(-3e38f, 100, 3e38f, 120)})).masks;
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  for (size_t k = 0; k < masks.size(); ++k) {
    ASSERT_EQ(masks[k].size(), 1u) << "frame offset " << k;
    expectBox(masks[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
  const auto tail = timeline.finish(frames(6, 12)).masks;
  ASSERT_EQ(tail.size(), 6u);
  expectFinite(tail);
  for (size_t k = 0; k < tail.size(); ++k) {
    ASSERT_EQ(tail[k].size(), 1u) << "frame offset " << k;
    expectBox(tail[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

// Interpolating between two float-extreme boxes must not overflow either: (q - p) alone is 5.9e38.
TEST(BoxTimeline, InterpolatesBoxesAtTheFloatLimitsWithoutOverflowing) {
  // The centre moves 5.9e38 (118 box sizes), so the boxes only match with a huge matching distance.
  BoxTimeline timeline(TimelineConfig{1000.0, 0.0, 0.0, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, -2.9e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(2.9e38f, 100, 3e38f, 120)})).masks;
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  // Halfway the box spans [-5e36, 5e36], which covers the whole frame width.
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 0, 100, 1000, 120, MaskOrigin::Interpolated);
}

// Plates and faces moving faster than one mean box side per keyframe interval (the first pass cannot match
// them) are matched by the second pass and covered on every frame up to the last keyframe, interpolated from
// the first interval on. After the last keyframe a track is only held half a box side further (spec §3.3), so
// the tail is not checked.
TEST(BoxTimeline, CoversFastMoversOnEveryFrame) {
  struct Shape {
    const char* name;
    int cls;
    double w;
    double h;
  };
  for (const Shape& shape : {Shape{"plate 48x16", kPlate, 48, 16}, Shape{"plate 24x8", kPlate, 24, 8},
                             Shape{"face 20x24", kFace, 20, 24}}) {
    for (const double widths : {1.6, 2.0, 3.0, 5.0}) {
      SCOPED_TRACE(std::string(shape.name) + ", " + std::to_string(widths) + " widths per interval");
      const Mover mover{shape.cls, 200, 500, shape.w, shape.h, widths * shape.w / kStep, 0};
      BoxTimeline timeline(TimelineConfig{}, 4096, 2160);
      const auto masks = runPerfectDetector(timeline, {mover}, 8);
      ASSERT_EQ(masks.size(), static_cast<size_t>(9 * kStep));
      expectOneMaskPerTrack(masks);
      int uncovered = 0;
      for (int64_t f = 0; f <= 8 * kStep; ++f) {
        if (coverage(masks[static_cast<size_t>(f)], mover.at(f)) < 0.8) ++uncovered;
      }
      EXPECT_EQ(uncovered, 0);
      for (int64_t f = 1; f < 8 * kStep; ++f) {
        if (f % kStep == 0) continue;  // keyframes carry the detection itself
        EXPECT_EQ(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
      }
    }
  }
}

// Two plates on neighbouring rows driving towards each other at two box widths per interval: they pass each
// other between keyframes 4 and 5, and both stay covered up to the last keyframe, one mask each.
TEST(BoxTimeline, CoversTwoTargetsThatCrossEachOther) {
  const Mover right{kPlate, 100, 500, 48, 16, 8, 0};
  const Mover left{kPlate, 900, 508, 48, 16, -8, 0};
  BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
  const auto masks = runPerfectDetector(timeline, {right, left}, 8);
  expectOneMaskPerTrack(masks);
  for (int64_t f = 0; f <= 8 * kStep; ++f) {
    const auto& m = masks[static_cast<size_t>(f)];
    EXPECT_GE(coverage(m, right.at(f)), 0.8) << "right-moving plate, frame " << f;
    EXPECT_GE(coverage(m, left.at(f)), 0.8) << "left-moving plate, frame " << f;
  }
}

// A plate moving vertically several of its (small) heights per interval: the second pass measures both axes in
// the larger side of the boxes, so 9 heights of a 24 x 8 plate are 3 sides.
TEST(BoxTimeline, CoversPlatesMovingVertically) {
  for (const double heights : {3.0, 6.0, 9.0}) {
    SCOPED_TRACE(std::to_string(heights) + " heights per interval");
    const Mover mover{kPlate, 500, 200, 24, 8, 0, heights * 8 / kStep};
    BoxTimeline timeline(TimelineConfig{}, 4096, 2160);
    const auto masks = runPerfectDetector(timeline, {mover}, 8);
    expectOneMaskPerTrack(masks);
    int uncovered = 0;
    for (int64_t f = 0; f <= 8 * kStep; ++f) {
      if (coverage(masks[static_cast<size_t>(f)], mover.at(f)) < 0.8) ++uncovered;
    }
    EXPECT_EQ(uncovered, 0);
    for (int64_t f = 1; f < 8 * kStep; ++f) {
      if (f % kStep == 0) continue;
      EXPECT_EQ(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
    }
  }
}

// A box three widths away (beyond the first pass) is matched by the second and interpolated; nothing else is
// masked (spec §3.2: no end boxes held).
TEST(BoxTimeline, ALooseMatchInterpolatesAlone) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)})).masks;
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float x = 100 + 5.0f * static_cast<float>(k);  // 60 px over 12 frames
    expectBox(masks[k][0], x, 100, x + 20, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
}

// The loose match seeds the velocity like a normal one: the next interval is matched by the first pass along the
// prediction, and the old track leaves no ghost behind.
TEST(BoxTimeline, ALooseMatchSeedsTheVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(220, 100, 240, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 190, 100, 210, 120, MaskOrigin::Interpolated);
  const auto tail = timeline.finish(frames(24, 30)).masks;
  ASSERT_EQ(tail[3].size(), 1u);  // frame 27: 5 px per frame for 3 frames is 15, capped at 10
  expectBox(tail[3][0], 230, 100, 250, 120, MaskOrigin::Held);
}

// The second pass gates each axis by the larger box size along it and pairs only similar sizes.
TEST(BoxTimeline, ALooseMatchNeedsCloseEnoughBoxesOfASimilarSize) {
  const auto interpolated = [](const TimelineConfig& c, const Box& from, const Box& to) {
    BoxTimeline timeline(c, 4000, 4000);
    timeline.advance({}, key(0, {from}));
    const auto masks = timeline.advance(frames(0, 12), key(12, {to})).masks;
    return countOrigin(masks[6], MaskOrigin::Interpolated);
  };
  // Both axes count in the larger side of the two boxes (20 here): 6 sides right and 3 down of a
  // 20 x 10 box is matched, and so is 6.5 heights down; 6.5 sides either way is not.
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(220, 160, 240, 170)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(100, 165, 120, 175)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(230, 100, 250, 110)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(100, 230, 120, 240)), 0u);
  // The gate is the configured value.
  TimelineConfig tight = exact();
  tight.matchMaxDistanceLoose = 2.0;
  EXPECT_EQ(interpolated(tight, face(100, 100, 120, 110), face(160, 100, 180, 110)), 0u);
  EXPECT_EQ(interpolated(tight, face(100, 100, 120, 110), face(140, 100, 160, 110)), 1u);
  // Sizes within a factor of 3 on each axis pair, beyond it they do not; classes never mix.
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 220, 160)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 221, 120)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 180, 161)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), Box{kPlate, 160, 100, 180, 120, 0.9f}), 0u);
}

// The scene that used to stack masks: a plate missed at keyframe 36 while a look-alike turns up 100 px away (and
// stays), two plates crossing, and a face appearing late. Every track masks a frame at most once, and every mask
// is detected, interpolated or held (no `extended` any more).
TEST(BoxTimeline, EveryTrackHasAtMostOneMaskPerFrame) {
  const Mover plate{kPlate, 400, 500, 48, 16, 3, 0};
  const Mover right{kPlate, 100, 700, 48, 16, 8, 0};
  const Mover left{kPlate, 900, 708, 48, 16, -8, 0};
  const float lookAlikeX = plate.at(3 * kStep).x1 + 100;
  const Box lookAlike{kPlate, lookAlikeX, 520, lookAlikeX + 48, 536, 0.9f};
  BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
  std::vector<std::vector<MaskBox>> masks;
  timeline.advance({}, key(0, {plate.at(0), right.at(0), left.at(0)}));
  for (int64_t k = 1; k <= 6; ++k) {
    std::vector<Box> boxes{right.at(k * kStep), left.at(k * kStep)};
    if (k != 3) boxes.push_back(plate.at(k * kStep));
    if (k >= 3) boxes.push_back(lookAlike);
    if (k >= 4) boxes.push_back(face(1000, 200, 1020, 224));
    for (auto& m : timeline.advance(frames((k - 1) * kStep, k * kStep), key(k * kStep, boxes)).masks) {
      masks.push_back(std::move(m));
    }
  }
  for (auto& m : timeline.finish(frames(6 * kStep, 7 * kStep)).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), static_cast<size_t>(7 * kStep));
  expectOneMaskPerTrack(masks);
  for (const auto& frame : masks) {
    for (const MaskBox& m : frame) {
      const std::string origin = toString(m.origin);
      EXPECT_TRUE(origin == "detected" || origin == "interpolated" || origin == "held") << origin;
      EXPECT_GE(m.track, 1);
    }
  }
}

}  // namespace
}  // namespace vc
```

`cpp/tests/records_test.cpp`：`MaskLineListsBoxesWithTheirOrigin` 换成

```cpp
TEST(Records, MaskLineListsBoxesWithTheirTrackAndOrigin) {
  const std::vector<MaskBox> boxes{MaskBox{1, 1.0f, 2.0f, 3.5f, 4.0f, MaskOrigin::Interpolated, 7}};
  EXPECT_EQ(maskRecord("a.mp4", 3, boxes, kClasses),
            R"({"video":"a.mp4","frame":3,"boxes":[{"cls":"plate","x1":1.0,"y1":2.0,"x2":3.5,"y2":4.0,"track":7,"origin":"interpolated"}]})"
            "\n");
  EXPECT_EQ(maskRecord("a.mp4", 4, {}, kClasses), R"({"video":"a.mp4","frame":4,"boxes":[]})" "\n");
}
```

`WritesNegativeZeroAsZero` 里的 `MaskBox{1, -0.0f, -0.04f, 1.0f, 1.0f, MaskOrigin::Extended}` 换成 `MaskBox{1, -0.0f, -0.04f, 1.0f, 1.0f, MaskOrigin::Held, 1}`，期望字符串换成

```cpp
            R"({"video":"a.mp4","frame":0,"boxes":[{"cls":"plate","x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,"track":1,"origin":"held"}]})"
```

`cpp/tests/redact_job_test.cpp`：`MasksTargetsThatMoveSeveralBoxWidthsBetweenKeyframes` 从 `const auto in = decodeAll(o.input);` 起到函数结束换成：

```cpp
  const auto in = decodeAll(o.input);
  const auto out = decodeAll(s.video);
  ASSERT_EQ(out.size(), 60u);
  ASSERT_EQ(in.size(), 60u);
  for (int f = 0; f < 60; ++f) {
    for (const Target& t : targets) {
      const Rect r = rectAt(t, f);
      EXPECT_GT(roughness(*in[f], r.x1, r.y1, r.x2, r.y2), 0.45) << t.cls << " frame " << f;
      // After the last keyframe (54) a target is only held half a box side further (spec 2026-10-19 §3.3), so
      // these targets, moving 2 to 2.5 widths per interval, leave their masks there.
      if (f <= 54) EXPECT_LT(roughness(*out[f], r.x1, r.y1, r.x2, r.y2), 0.1) << t.cls << " frame " << f;
    }
    // Away from both paths the checkerboard survives the encoder, so a leak would show.
    EXPECT_GT(roughness(*out[f], 300, 150, 340, 200), 0.4) << "frame " << f;
  }
  // Each target is one track: exactly one box of its class on every frame, detected on the keyframes (0, 6, ...,
  // 54), interpolated between them and held after the last one.
  std::ifstream masks(s.outputDir / "masks.jsonl");
  for (std::string line; std::getline(masks, line);) {
    const nlohmann::json record = nlohmann::json::parse(line);
    const int frame = record.at("frame").get<int>();
    for (const Target& t : targets) {
      int boxes = 0;
      std::string origin;
      for (const auto& box : record.at("boxes")) {
        if (box.at("cls") != t.cls) continue;
        ++boxes;
        origin = box.at("origin").get<std::string>();
      }
      EXPECT_EQ(boxes, 1) << t.cls << " frame " << frame;
      const std::string expected = frame > 54 ? "held" : frame % 6 == 0 ? "detected" : "interpolated";
      EXPECT_EQ(origin, expected) << t.cls << " frame " << frame;
    }
  }
}
```

并把它上方注释的最后一句改成 "…so they are also the ground truth: the box is pixelated on every output frame up to the last keyframe."

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`no type named 'TimelineOutput' in namespace 'vc'`、`no member named 'Held' in 'vc::MaskOrigin'`、`no member named 'track' in 'vc::MaskBox'`。

- [ ] **Step 3: 实现 `BoxTimeline`**

`cpp/src/track/box_timeline.h` 整个换成：

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "detect/box.h"

namespace vc {

enum class MaskOrigin { Detected, Interpolated, Held };

// "detected" | "interpolated" | "held" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

struct MaskBox {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  MaskOrigin origin = MaskOrigin::Detected;
  int64_t track = 0;  // the track the mask is for (ids from 1); a track has at most one mask per frame
};

struct FrameTime {
  int64_t index = 0;
  double ptsMs = 0;
};

struct Keyframe {
  FrameTime time;
  std::vector<Box> boxes;
};

struct TimelineConfig {
  double matchMaxDistance = 1.0;      // first pass: max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
  double matchMaxDistanceLoose = 6.0; // second pass: max |dx| and |dy| / the largest side of the two boxes
};

// What advance() and finish() return.
struct TimelineOutput {
  // The masks of each frame passed in, in the same order.
  std::vector<std::vector<MaskBox>> masks{};
  // One flag per box of the keyframe those frames start at, in Keyframe::boxes order: whether that box is
  // masked. Empty for the first advance() call.
  std::vector<bool> confirmed{};
};

// Turns detections on keyframes into one mask per target and frame (spec 2026-10-19 §3). Boxes on
// consecutive keyframes are matched into tracks; a track has at most one mask per frame.
//
// Matching runs in two greedy, one-to-one passes over same-class pairs, cheapest first; both try the
// target staying put and keeping its velocity. The first pass takes pairs whose centres are at most
// matchMaxDistance mean box sides apart. The second pass, over what is left, takes pairs of similar size
// (each side within a factor of 3 of the other's) whose centres are at most matchMaxDistanceLoose times
// the largest side of the two boxes apart along each axis.
//
// A matched track is interpolated between its two keyframe boxes (centre and size, linearly in time). A
// track missed at the next keyframe is held at its box, moved along its velocity by at most half the
// box's width along x and half its height along y; it lives on for one keyframe as a ghost at the end of
// that hold. A ghost matched at the following keyframe is interpolated from where it was held to the new
// box; an unmatched ghost ends, and the interval after its last hold is not masked. A box that starts a
// track is masked from its own keyframe on, never before. Every mask is grown by expandFraction of its box
// plus motionExpandFraction of the interval's displacement along each axis (for a hold, of its capped
// shift), to at least minBoxPx, and clipped to the frame. A track's velocity runs from its last real
// detection. Feed keyframes in order; frames are identified by decode index.
class BoxTimeline {
 public:
  // Throws std::invalid_argument for a non-positive frame size or invalid config.
  BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight);

  // `frames`: every frame from the previous keyframe (inclusive) to `next` (exclusive), in order.
  // Returns their masks in the same order. The first call records `next` only: `frames` must be
  // empty and the result is empty. Throws std::invalid_argument when the call order is violated,
  // when a box coordinate or the timestamp of `next` is not finite (the timeline is left
  // untouched). A non-finite or out-of-range pts of one of `frames` is not an error: the
  // positions in that interval then come from the decode indices.
  TimelineOutput advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // Masks for the frames from the last keyframe (inclusive) to the end of the video: every track
  // detected on that keyframe is held there as above; ghosts end. Resets the timeline.
  TimelineOutput finish(const std::vector<FrameTime>& frames);

 private:
  // A velocity (px per frame) or a shift (px).
  struct Offset {
    double x = 0;
    double y = 0;
  };

  // A target followed from keyframe to keyframe.
  struct Track {
    int64_t id = 0;
    Box box{};                         // on the last keyframe: the detection, or where a ghost was held to
    std::optional<Offset> velocity{};  // px per frame index
    bool ghost = false;                // missed on the last keyframe (lives for one keyframe)
    Box anchor{};                      // the target's last real detection (the box itself unless a ghost)
    double anchorSteps = 0;            // frames from `anchor` to the last keyframe (0 unless a ghost)
    int keyBox = -1;                   // index of `box` in the last keyframe's boxes; -1 for a ghost
  };

  struct Match {
    size_t track = 0;
    size_t next = 0;
  };

  std::vector<Match> match(const Keyframe& next, double steps) const;
  // How far a held track has moved `since` frames after its keyframe: along its velocity, at most half
  // its box's width along x and half its height along y.
  Offset heldShift(const Track& track, double since) const;
  // Appends the mask of `track` for `region`, grown by expandFraction of its size plus
  // motionExpandFraction of `motion` per axis, to at least minBoxPx, clipped to the frame; nothing when
  // it lies outside the frame.
  void emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion,
            MaskOrigin origin) const;
  // The masks of `track` held from its keyframe `from` over `frames` (which start there); the motion
  // margin is the shift reached `span` frames after the keyframe.
  void hold(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames, const FrameTime& from,
            const Track& track, double span) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<FrameTime> keyTime_;  // the last keyframe
  std::vector<Track> tracks_;         // its boxes in order, then its ghosts
  size_t keyBoxes_ = 0;               // number of boxes on the last keyframe
  int64_t nextId_ = 1;
};

}  // namespace vc
```

`cpp/src/track/box_timeline.cpp` 整个换成：

```cpp
#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vc {
namespace {

// The second matching pass pairs boxes only when each side is within this factor of the other's.
constexpr double kMaxLooseSizeRatio = 3.0;

// A held box moves along its velocity by at most this fraction of its width (x) and height (y) (spec
// 2026-10-19 §3.3). A ghost lives for exactly one keyframe (the old kMaxMissedKeyframes = 1).
constexpr double kMaxHeldShift = 0.5;

struct Point {
  double x = 0;
  double y = 0;
};

Point centerOf(const Box& b) { return {(static_cast<double>(b.x1) + b.x2) / 2.0, (static_cast<double>(b.y1) + b.y2) / 2.0}; }

// Box sizes are computed in double: the float difference of two finite coordinates can overflow to
// inf, and inf later turns into a NaN margin (0 * inf) that would drop the mask silently.
double widthOf(const Box& b) { return b.x2 > b.x1 ? static_cast<double>(b.x2) - b.x1 : 0.0; }
double heightOf(const Box& b) { return b.y2 > b.y1 ? static_cast<double>(b.y2) - b.y1 : 0.0; }

// The nearest float, saturating instead of overflowing to inf.
float toFloat(double v) {
  constexpr double kMax = std::numeric_limits<float>::max();
  return static_cast<float>(std::clamp(v, -kMax, kMax));
}

Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = toFloat(b.x1 + dx);
  s.x2 = toFloat(b.x2 + dx);
  s.y1 = toFloat(b.y1 + dy);
  s.y2 = toFloat(b.y2 + dy);
  return s;
}

Box lerp(const Box& a, const Box& b, double t) {
  const auto mix = [t](float p, float q) {
    return toFloat(static_cast<double>(p) + (static_cast<double>(q) - static_cast<double>(p)) * t);
  };
  Box out = a;
  out.x1 = mix(a.x1, b.x1);
  out.y1 = mix(a.y1, b.y1);
  out.x2 = mix(a.x2, b.x2);
  out.y2 = mix(a.y2, b.y2);
  out.score = std::max(a.score, b.score);
  return out;
}

bool similarSizes(const Box& a, const Box& b) {
  const auto within = [](double p, double q) { return p <= kMaxLooseSizeRatio * q && q <= kMaxLooseSizeRatio * p; };
  return within(widthOf(a), widthOf(b)) && within(heightOf(a), heightOf(b));
}

void growTo(double& lo, double& hi, double minimum) {
  if (hi - lo >= minimum) return;
  const double c = (lo + hi) / 2.0;
  lo = c - minimum / 2.0;
  hi = c + minimum / 2.0;
}

// Position of each frame between two keyframes in [0, 1): by PTS when they increase strictly
// inside the interval, otherwise by decode index. Every test is written so that a NaN fails it.
std::vector<double> fractions(const std::vector<FrameTime>& frames, const FrameTime& from, const FrameTime& to) {
  bool usePts = to.ptsMs > from.ptsMs;
  for (size_t k = 0; usePts && k < frames.size(); ++k) {
    const double pts = frames[k].ptsMs;
    const bool inRange = pts >= from.ptsMs && pts < to.ptsMs;
    const bool increasing = k == 0 || pts > frames[k - 1].ptsMs;
    if (!(inRange && increasing)) usePts = false;
  }
  std::vector<double> t;
  t.reserve(frames.size());
  for (const FrameTime& f : frames) {
    t.push_back(usePts ? (f.ptsMs - from.ptsMs) / (to.ptsMs - from.ptsMs)
                       : static_cast<double>(f.index - from.index) / static_cast<double>(to.index - from.index));
  }
  return t;
}

// Frame indices must be strictly increasing.
void requireIncreasing(const std::vector<FrameTime>& frames) {
  for (size_t k = 1; k < frames.size(); ++k) {
    if (frames[k].index <= frames[k - 1].index) {
      throw std::invalid_argument("BoxTimeline: frame indices must be strictly increasing");
    }
  }
}

// A broken box or timestamp must stop the run: comparisons with NaN are false, so it would otherwise
// slip through matching and clamping into a mask that covers nothing.
void requireFinite(const Keyframe& key) {
  if (!std::isfinite(key.time.ptsMs)) throw std::invalid_argument("BoxTimeline: keyframe timestamp is not finite");
  for (const Box& b : key.boxes) {
    if (!std::isfinite(b.x1) || !std::isfinite(b.y1) || !std::isfinite(b.x2) || !std::isfinite(b.y2)) {
      throw std::invalid_argument("BoxTimeline: keyframe box has non-finite coordinates");
    }
  }
}

bool validConfig(const TimelineConfig& c) {
  return std::isfinite(c.matchMaxDistance) && c.matchMaxDistance > 0 && std::isfinite(c.expandFraction) &&
         c.expandFraction >= 0 && std::isfinite(c.motionExpandFraction) && c.motionExpandFraction >= 0 &&
         std::isfinite(c.minBoxPx) && c.minBoxPx >= 0 && std::isfinite(c.matchMaxDistanceLoose) &&
         c.matchMaxDistanceLoose > 0;
}

}  // namespace

const char* toString(MaskOrigin origin) {
  switch (origin) {
    case MaskOrigin::Detected:
      return "detected";
    case MaskOrigin::Interpolated:
      return "interpolated";
    case MaskOrigin::Held:
      return "held";
  }
  return "detected";
}

BoxTimeline::BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight)
    : config_(config), width_(frameWidth), height_(frameHeight) {
  if (frameWidth <= 0 || frameHeight <= 0) throw std::invalid_argument("BoxTimeline: frame size must be positive");
  if (!validConfig(config)) throw std::invalid_argument("BoxTimeline: invalid config");
}

std::vector<BoxTimeline::Match> BoxTimeline::match(const Keyframe& next, double steps) const {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  // The cheaper of "stayed put" and "kept its velocity" (a target that stops still matches).
  const auto cheapest = [&](const Track& track, const auto& costFrom) {
    const Point from = centerOf(track.box);
    double cost = costFrom(from);
    if (const auto& v = track.velocity) {
      cost = std::min(cost, costFrom(Point{from.x + v->x * steps, from.y + v->y * steps}));
    }
    return cost;
  };
  std::vector<bool> usedTrack(tracks_.size(), false);
  std::vector<bool> usedNext(next.boxes.size(), false);
  std::vector<Match> pairs;
  const auto takeGreedily = [&](std::vector<Candidate> candidates) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
    for (const Candidate& c : candidates) {
      if (usedTrack[c.i] || usedNext[c.j]) continue;
      usedTrack[c.i] = true;
      usedNext[c.j] = true;
      pairs.push_back(Match{c.i, c.j});
    }
  };

  // First pass: centre distance in mean box sides.
  std::vector<Candidate> strict;
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Box& a = tracks_[i].box;
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& b = next.boxes[j];
      if (a.cls != b.cls) continue;
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (widthOf(a) + heightOf(a) + widthOf(b) + heightOf(b)) / 4.0);
      const double cost =
          cheapest(tracks_[i], [&](Point p) { return std::hypot(actual.x - p.x, actual.y - p.y) / size; });
      if (cost <= config_.matchMaxDistance) strict.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(strict));

  // Second pass, over what is left: both axes in the largest side of the two boxes, similar sizes only.
  std::vector<Candidate> loose;
  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (usedTrack[i]) continue;
    const Box& a = tracks_[i].box;
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& b = next.boxes[j];
      if (usedNext[j] || a.cls != b.cls || !similarSizes(a, b)) continue;
      const Point actual = centerOf(b);
      const double side = std::max({1.0, widthOf(a), heightOf(a), widthOf(b), heightOf(b)});
      const double cost = cheapest(tracks_[i], [&](Point p) {
        return std::max(std::abs(actual.x - p.x), std::abs(actual.y - p.y)) / side;
      });
      if (cost <= config_.matchMaxDistanceLoose) loose.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(loose));
  return pairs;
}

BoxTimeline::Offset BoxTimeline::heldShift(const Track& track, double since) const {
  if (!track.velocity) return {};
  const double capX = kMaxHeldShift * widthOf(track.box);
  const double capY = kMaxHeldShift * heightOf(track.box);
  return {std::clamp(track.velocity->x * since, -capX, capX), std::clamp(track.velocity->y * since, -capY, capY)};
}

void BoxTimeline::emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion,
                       MaskOrigin origin) const {
  const double mx = config_.expandFraction * widthOf(region) + config_.motionExpandFraction * motion.x;
  const double my = config_.expandFraction * heightOf(region) + config_.motionExpandFraction * motion.y;
  double x1 = region.x1 - mx;
  double x2 = region.x2 + mx;
  double y1 = region.y1 - my;
  double y2 = region.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  x1 = std::clamp(x1, 0.0, static_cast<double>(width_));
  x2 = std::clamp(x2, 0.0, static_cast<double>(width_));
  y1 = std::clamp(y1, 0.0, static_cast<double>(height_));
  y2 = std::clamp(y2, 0.0, static_cast<double>(height_));
  if (!(x2 > x1) || !(y2 > y1)) return;
  out.push_back(MaskBox{region.cls, static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                        static_cast<float>(y2), origin, track.id});
}

void BoxTimeline::hold(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames,
                       const FrameTime& from, const Track& track, double span) const {
  const Offset reach = heldShift(track, span);
  const Offset motion{std::abs(reach.x), std::abs(reach.y)};
  for (size_t k = 0; k < frames.size(); ++k) {
    const Offset s = heldShift(track, static_cast<double>(frames[k].index - from.index));
    emit(masks[k], track, shifted(track.box, s.x, s.y), motion,
         frames[k].index == from.index ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

TimelineOutput BoxTimeline::advance(const std::vector<FrameTime>& frames, const Keyframe& next) {
  requireFinite(next);
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    tracks_.clear();
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      tracks_.push_back(Track{nextId_++, next.boxes[j], std::nullopt, false, next.boxes[j], 0.0, static_cast<int>(j)});
    }
    keyTime_ = next.time;
    keyBoxes_ = next.boxes.size();
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (frames.empty() || frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prevTime.index);
  std::vector<std::optional<size_t>> matchedTo(tracks_.size());
  std::vector<std::optional<size_t>> matchedFrom(next.boxes.size());
  for (const Match& m : match(next, steps)) {
    matchedTo[m.track] = m.next;
    matchedFrom[m.next] = m.track;
  }

  TimelineOutput out;
  out.masks.resize(frames.size());
  const std::vector<double> t = fractions(frames, prevTime, next.time);
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    if (matchedTo[i]) {
      const Box& to = next.boxes[*matchedTo[i]];
      const Point a = centerOf(track.box);
      const Point b = centerOf(to);
      const Offset motion{std::abs(b.x - a.x), std::abs(b.y - a.y)};
      for (size_t k = 0; k < frames.size(); ++k) {
        const MaskOrigin origin =
            k > 0 ? MaskOrigin::Interpolated : track.ghost ? MaskOrigin::Held : MaskOrigin::Detected;
        emit(out.masks[k], track, lerp(track.box, to, t[k]), motion, origin);
      }
    } else if (!track.ghost) {
      hold(out.masks, frames, prevTime, track, steps);
    }
    // An unmatched ghost ends here: the interval after its hold is not masked.
  }
  out.confirmed.assign(keyBoxes_, true);

  // The boxes of `next` in order (a matched one continues its track), then the ghosts of the real
  // detections that were missed, at the end of their hold.
  std::vector<Track> tracks;
  tracks.reserve(next.boxes.size() + tracks_.size());
  for (size_t j = 0; j < next.boxes.size(); ++j) {
    const Box& box = next.boxes[j];
    Track fresh{0, box, std::nullopt, false, box, 0.0, static_cast<int>(j)};
    if (const auto from = matchedFrom[j]) {
      const Track& old = tracks_[*from];
      const Point a = centerOf(old.anchor);
      const Point b = centerOf(box);
      const double span = old.anchorSteps + steps;
      fresh.id = old.id;
      fresh.velocity = Offset{(b.x - a.x) / span, (b.y - a.y) / span};
    } else {
      fresh.id = nextId_++;
    }
    tracks.push_back(fresh);
  }
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    if (matchedTo[i] || track.ghost) continue;
    const Offset s = heldShift(track, steps);
    tracks.push_back(Track{track.id, shifted(track.box, s.x, s.y), track.velocity, true, track.anchor,
                           track.anchorSteps + steps, -1});
  }
  keyTime_ = next.time;
  tracks_ = std::move(tracks);
  keyBoxes_ = next.boxes.size();
  return out;
}

TimelineOutput BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (!frames.empty() && frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  TimelineOutput out;
  out.masks.resize(frames.size());
  // The motion margin of the tail: the shift one frame past its end, as for an interval's next keyframe.
  const double span = frames.empty() ? 0.0 : static_cast<double>(frames.back().index - prevTime.index + 1);
  for (const Track& track : tracks_) {
    if (!track.ghost) hold(out.masks, frames, prevTime, track, span);
  }
  out.confirmed.assign(keyBoxes_, true);
  keyTime_.reset();
  tracks_.clear();
  keyBoxes_ = 0;
  nextId_ = 1;
  return out;
}

}  // namespace vc
```

- [ ] **Step 4: 接上 `records` 和 `Processor`**

`cpp/apps/vc_redact/records.cpp` 的 `maskRecord` 里，`b["y2"] = round1(m.y2, "y2");` 之后加 `b["track"] = m.track;`。`records.h` 里 `maskRecord` 的注释改成：

```cpp
// One masks.jsonl line for a frame (boxes may be empty), newline included; each box is
// {cls, x1, y1, x2, y2, track, origin}. Throws VcError as detectionRecord does for class indices and
// for non-finite coordinates.
```

`cpp/apps/vc_redact/redact_job.cpp` 的 `Processor::onVideo` 里

```cpp
    const std::vector<std::vector<MaskBox>> masks =
        timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    emit(masks);
```

换成

```cpp
    const TimelineOutput step = timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    emit(step.masks);
```

`Processor::onEnd` 里

```cpp
    const std::vector<std::vector<MaskBox>> masks = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    emit(masks);
```

换成

```cpp
    const TimelineOutput step = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    emit(step.masks);
```

- [ ] **Step 5: 文档**

`cpp/README.md` vc_redact 一节：

- `masks.jsonl` 一条换成：`` - `masks.jsonl`：每一帧一行，列出这一帧实际打码的框：`cls`、坐标、`track`（轨迹编号，从 1 起；同一条轨迹每帧至多一个框）和来源 `origin`（`detected` 为轨迹在这个关键帧上检测到的框，`interpolated` 为前后两个关键帧之间插值出来的，`held` 为目标在下一个关键帧没被检出时原地保持（按速度外推，至多半个框边）的框） ``
- "快速移动的目标"一整条换成：

```
- **补框：一个目标一个框**（设计见 `docs/superpowers/specs/2026-10-19-stable-masks-and-speed-design.md` §3）：前后两个关键帧上的框分两轮配对（同类、一对一、代价小的先配）。第一轮按中心距离除以平均边长，不超过 `match_max_distance`；剩下的再配第二轮：x、y 方向的位移都除以两框所有边里最长的那条，都不超过 `match_max_distance_loose`（默认 6），且两框的宽、高各自相差不超过 3 倍，所以一个间隔里移动好几个框宽的车牌、人脸也能配上。配上的框连成一条轨迹，**每条轨迹每帧至多一个框**：两个关键帧之间按中心和宽高线性插值；下一个关键帧上没配上时，在原地保持，已有速度就按速度外推，但每个轴的位移不超过半个框宽 / 框高；这条轨迹再留一个关键帧（幽灵），在下一个关键帧配上了就从保持到的位置插值过去，没配上就结束，中间那一段不再打码。速度从轨迹最后一次真实检测量起。目标第一次被检出之前的帧不打码。
```

- "已知的局限"整段换成：

```
- **已知的局限**（设计 §3.5 接受的代价）：
  - 目标在两个关键帧之间掉头（反向运动）时，那个间隔里是线性插值，掉头那一段可能盖不全。
  - 目标在某个关键帧漏检、而在漏检的间隔里停下或掉头时，原地保持（至多外推半个框边）的框可能盖不全。
  - 第二轮配错（例如两个相邻同类目标在还没有速度的第一个间隔里交错），配错的那一段可能盖不全；第二轮配上一个相距较远的同类误检时，插值框会在两者之间移动。
  - 视频最后一个关键帧之后的几帧只按速度外推半个框边，快速移动的目标可能离开打码框。
```

`docs/USAGE.md` §5 的"补框的残余漏洞"一条换成：`` - **补框的残余漏洞：** 每个目标每帧只有一个打码框。目标在两次检测之间掉头、在漏检的那一段停下或掉头，或在第一次检测前就出现，中间几帧可能盖不全（`cpp/README.md` 已知局限）。 ``

- [ ] **Step 6: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Records.*:RedactJob.*'
```

预期：全部 PASSED（`BoxTimeline` 38 个），零警告。

- [ ] **Step 7: 提交**

```bash
git add cpp/src/track/box_timeline.h cpp/src/track/box_timeline.cpp cpp/tests/box_timeline_test.cpp cpp/apps/vc_redact/records.h cpp/apps/vc_redact/records.cpp cpp/apps/vc_redact/redact_job.cpp cpp/tests/records_test.cpp cpp/tests/redact_job_test.cpp cpp/README.md docs/USAGE.md
git commit -m "feat: one mask per track per frame in BoxTimeline, held and ghost masks without sweeps"
```

---

### Task 5: 大小迟滞与面积上限

本设计 §3.1 的"变大及时、变小慢"和面积上限，加两个配置键。平滑的是**大小**（宽、高），不是各条边：输出框始终以当帧原始框的中心为中心，所以跟着目标走，快速移动的目标不会沿运动方向拉长。

**Files:**
- Modify: `cpp/src/track/box_timeline.h`（`TimelineConfig`、类注释、私有成员）、`cpp/src/track/box_timeline.cpp`（`validConfig`、`emit`、`hold`、`advance` 末尾、`finish` 末尾，新增 `smoothed`、`cappedArea`）
- Modify: `cpp/apps/vc_redact/redact_config.h`（注释）、`cpp/apps/vc_redact/redact_config.cpp`、`configs/redact.json`
- Modify: `cpp/README.md`、`docs/USAGE.md`
- Test: `cpp/tests/box_timeline_test.cpp`（改 `config()`，追加测试）、`cpp/tests/redact_config_test.cpp`

**Interfaces:**
- Consumes: Task 4 的 `BoxTimeline`、`Track::id`、`MaskOrigin::Held`。
- Produces:
  - `TimelineConfig::shrinkPerFrame`（默认 0.02，范围 (0, 1]）、`TimelineConfig::maxMaskAreaRatio`（默认 4.0，≥ 1）。
  - 每条轨迹保存平滑后的宽 W、高 H（`shown_`）。每一帧，原始框（插值或保持后的框，按 `expand_fraction` / `motion_expand_fraction` 外扩、撑到 `min_box_px`，**裁剪之前**）宽 w、高 h、中心 c：
    - w ≥ W 时 W′ = w，否则 W′ = max(w, W·(1 − shrinkPerFrame))；H′ 同理；`origin` 为 `held` 的帧不收缩（W′ = max(W, w)，H′ = max(H, h)），即丢失后保留最后的大小、放在保持的位置上；
    - 若 W′·H′ > maxMaskAreaRatio·w·h，W′、H′ 按同一比例 f 向 w、h 收缩：W″ = w + f·(W′ − w)、H″ = h + f·(H′ − h)，f 取恰好满足上限的值（闭式解 f = 2q / (p + √(p² + 4abq))，a = W′ − w，b = H′ − h，p = a·h + b·w，q = ratio·w·h − w·h）；
    - 输出框以 c 为中心、大小 W″ × H″，再裁到画面内；所以它总包含原始框，宽、高不超过 max(原始框的、上一帧平滑后的)。
    - 新轨迹从 W = w、H = h 开始；原始框整个落在画面外时，或轨迹结束时，忘掉它的大小。`shrinkPerFrame = 1` 时平滑不起作用（每个框就是它自己）。
  - 配置键 `shrink_per_frame`（`"shrink_per_frame must be in (0, 1]"`）、`max_mask_area_ratio`（`"max_mask_area_ratio must be >= 1"`），放在 `min_box_px` 之后。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/box_timeline_test.cpp`：`config()` 换成下面这样（之前的测试都按"每个框就是它自己"写的期望，`shrinkPerFrame = 1` 时平滑不起作用），并在 `exact()` 之后加 `smooth()`：

```cpp
// Margins and minimum size as given, everything else at its default except the mask smoothing, which is off:
// with shrinkPerFrame 1 a mask may shrink to its box at once, so each mask is its own box.
TimelineConfig config(double expand, double motion, double minBox) {
  TimelineConfig c;
  c.expandFraction = expand;
  c.motionExpandFraction = motion;
  c.minBoxPx = minBox;
  c.shrinkPerFrame = 1.0;
  return c;
}

// No margins and no minimum size: each mask is exactly its (interpolated or held) box.
TimelineConfig exact() { return config(0.0, 0.0, 0.0); }

// exact(), with the mask smoothing and the area cap as given.
TimelineConfig smooth(double shrink, double ratio) {
  TimelineConfig c = exact();
  c.shrinkPerFrame = shrink;
  c.maxMaskAreaRatio = ratio;
  return c;
}
```

在文件末尾（`}  // namespace` 之前）追加：

```cpp
// A mask grows at once (spec 2026-10-19 §3.1): the face doubles in 2 frames and frame 1's mask is its box.
TEST(BoxTimeline, AMaskGrowsAtOnce) {
  BoxTimeline timeline(smooth(0.02, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(125, 125, 175, 175)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(100, 100, 200, 200)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 112.5f, 112.5f, 187.5f, 187.5f, MaskOrigin::Interpolated);
}

// ... and shrinks slowly: its width and height by at most shrinkPerFrame of themselves per frame, never below the
// box's, around the box's centre. The face halves in 2 frames (100 -> 75 -> 50 px) and then stays; with 10 % per
// frame the mask's side goes 100, 90, 81, 72.9, 65.61, 59.049, 53.1441, then 50 (0.9 x 53.1441 is less).
TEST(BoxTimeline, AMaskShrinksByAtMostItsShrinkFractionPerFrame) {
  BoxTimeline timeline(smooth(0.1, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  std::vector<std::vector<MaskBox>> masks = timeline.advance(frames(0, 2), key(2, {face(125, 125, 175, 175)})).masks;
  for (auto& m : timeline.advance(frames(2, 8), key(8, {face(125, 125, 175, 175)})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 8u);
  const double sides[] = {100, 90, 81, 72.9, 65.61, 59.049, 53.1441, 50};
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const auto half = static_cast<float>(sides[k] / 2);
    expectBox(masks[k][0], 150 - half, 150 - half, 150 + half, 150 + half,
              k == 0 || k == 2 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
}

// A lost target's mask keeps its smoothed size while it is held, and moves with the held box. The face shrinks to
// 50 px while moving 2 px a frame (masks of 100, 90 and 81 px), then is missed at keyframe 6: frames 3 to 5 keep
// 81 px around the held centre.
TEST(BoxTimeline, AHeldMaskKeepsItsSmoothedSizeAtTheHeldPosition) {
  BoxTimeline timeline(smooth(0.1, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  std::vector<std::vector<MaskBox>> masks = timeline.advance(frames(0, 2), key(2, {face(129, 125, 179, 175)})).masks;
  for (auto& m : timeline.advance(frames(2, 6), key(6, {})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 6u);
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 107, 105, 197, 195, MaskOrigin::Interpolated);  // 90 px around (152, 150)
  ASSERT_EQ(masks[2].size(), 1u);
  expectBox(masks[2][0], 113.5f, 109.5f, 194.5f, 190.5f, MaskOrigin::Detected);  // 81 px around (154, 150)
  for (size_t k = 3; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float cx = 154 + 2.0f * static_cast<float>(k - 2);  // held along the velocity
    expectBox(masks[k][0], cx - 40.5f, 109.5f, cx + 40.5f, 190.5f, MaskOrigin::Held);
  }
}

// Some single mask contains the whole box.
bool covers(const std::vector<MaskBox>& masks, float x1, float y1, float x2, float y2) {
  return std::any_of(masks.begin(), masks.end(), [=](const MaskBox& m) {
    return m.x1 <= x1 + 1e-3f && m.y1 <= y1 + 1e-3f && m.x2 >= x2 - 1e-3f && m.y2 >= y2 - 1e-3f;
  });
}

// The smoothing is of the size, around the box's own centre (spec §3.1): a fast target's mask follows it instead
// of trailing behind. Its width and height never exceed the larger of the box's and the previous mask's, and its
// centre is the box's.
TEST(BoxTimeline, AFastTargetIsNotSmearedAlongItsMotion) {
  // A 20 px face at 5 px a frame: every mask is the box itself, nothing is left behind.
  BoxTimeline steady(smooth(0.02, 4.0), 1000, 1000);
  steady.advance({}, key(0, {face(100, 100, 120, 120)}));
  std::vector<std::vector<MaskBox>> masks = steady.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)})).masks;
  for (auto& m : steady.advance(frames(12, 24), key(24, {face(220, 100, 240, 120)})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 24u);
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float x = 100 + 5.0f * static_cast<float>(k);
    expectBox(masks[k][0], x, 100, x + 20, 120, k % 12 == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }

  // A face shrinking from 100 to 50 px while it moves 175 px in 12 frames: the mask's side shrinks 2 % a frame,
  // centred on the box wherever the box is.
  BoxTimeline shrinking(smooth(0.02, 4.0), 1000, 1000);
  shrinking.advance({}, key(0, {face(100, 100, 200, 200)}));
  const auto moving = shrinking.advance(frames(0, 12), key(12, {face(300, 125, 350, 175)})).masks;
  ASSERT_EQ(moving.size(), 12u);
  double side = 0;
  for (size_t k = 0; k < moving.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(moving[k].size(), 1u);
    const MaskBox& m = moving[k][0];
    const double t = static_cast<double>(k) / 12;
    const double cx = 150 + 175 * t;  // the interpolated box's centre (its y stays 150)
    const double box = 100 - 50 * t;  // and its width and height
    const double previous = side;
    side = k == 0 ? box : std::max(box, 0.98 * side);
    EXPECT_NEAR((m.x1 + m.x2) / 2, cx, 1e-3);
    EXPECT_NEAR((m.y1 + m.y2) / 2, 150, 1e-3);
    EXPECT_NEAR(m.x2 - m.x1, side, 1e-3);
    EXPECT_NEAR(m.y2 - m.y1, side, 1e-3);
    EXPECT_LE(m.x2 - m.x1, std::max(box, previous) + 1e-3);
    EXPECT_LE(m.y2 - m.y1, std::max(box, previous) + 1e-3);
    EXPECT_TRUE(covers(moving[k], static_cast<float>(cx - box / 2), static_cast<float>(150 - box / 2),
                       static_cast<float>(cx + box / 2), static_cast<float>(150 + box / 2)));
  }
}

// A mask covers at most max_mask_area_ratio times its box: beyond that its width and height are pulled the same
// fraction of their way towards the box's. Frame 1's box is 120..180 (60 px, 3600 px^2); the side may only shrink
// from 100 to 98 (9604 px^2, 2.67 times), so it is pulled towards 60 until the mask is 2 x 3600 = 7200 px^2:
// sqrt(7200) px square around the box's centre.
TEST(BoxTimeline, CapsTheMaskAreaAtTheConfiguredMultipleOfItsBox) {
  BoxTimeline timeline(smooth(0.02, 2.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(140, 140, 160, 160)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  const double half = std::sqrt(7200.0) / 2;
  const MaskBox& m = masks[1][0];
  EXPECT_NEAR(m.x1, 150 - half, 1e-3);
  EXPECT_NEAR(m.x2, 150 + half, 1e-3);
  EXPECT_NEAR(m.y1, 150 - half, 1e-3);
  EXPECT_NEAR(m.y2, 150 + half, 1e-3);
}

// Frame 1's box is 145..255 x 100..120 (110 x 20 = 2200 px^2); the width may only shrink from 200 to 196 (3920 px^2),
// the height is the box's. The width is pulled towards 110 until the mask is 1.5 x 2200 = 3300 px^2: 165 x 20 around
// the box's centre.
TEST(BoxTimeline, TheAreaCapPullsBothSidesTowardsTheBox) {
  BoxTimeline timeline(smooth(0.02, 1.5), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 300, 120)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(190, 100, 210, 120)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 117.5f, 100, 282.5f, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, RejectsAnInvalidShrinkOrAreaRatio) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {0.0, -0.1, 1.5, nan, inf}) {
    EXPECT_THROW(BoxTimeline(smooth(bad, 4.0), 10, 10), std::invalid_argument) << bad;
  }
  for (const double bad : {0.99, 0.0, -1.0, nan, inf}) {
    EXPECT_THROW(BoxTimeline(smooth(0.02, bad), 10, 10), std::invalid_argument) << bad;
  }
  EXPECT_NO_THROW(BoxTimeline(smooth(1.0, 1.0), 10, 10));
  EXPECT_DOUBLE_EQ(TimelineConfig{}.shrinkPerFrame, 0.02);
  EXPECT_DOUBLE_EQ(TimelineConfig{}.maxMaskAreaRatio, 4.0);
}
```

`cpp/tests/redact_config_test.cpp`：

- `kAllKeys` 在 `"min_box_px",` 之后加 `"shrink_per_frame", "max_mask_area_ratio",`。
- `expectSameConfig` 在 `minBoxPx` 一行之后加

```cpp
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, d.timeline.shrinkPerFrame);
  EXPECT_DOUBLE_EQ(c.timeline.maxMaskAreaRatio, d.timeline.maxMaskAreaRatio);
```

- `nonDefaultConfig` 在 `c.timeline.minBoxPx = 24.0;` 之后加 `c.timeline.shrinkPerFrame = 0.125;` 和 `c.timeline.maxMaskAreaRatio = 2.5;`。
- `RejectsBadInput` 列表末尾加 `R"({"shrink_per_frame": 0})", R"({"shrink_per_frame": 1.5})", R"({"shrink_per_frame": "0.02"})", R"({"max_mask_area_ratio": 0.5})", R"({"max_mask_area_ratio": 1e400})",`。
- `RejectsOutOfRangeValuesAndNulls` 列表末尾加 `R"({"shrink_per_frame": null})", R"({"max_mask_area_ratio": null})",`。
- `ValidateRefusesEveryNonFiniteDoubleField` 的 `fields` 在 `min_box_px` 之后加

```cpp
      {"shrink_per_frame", [](RedactConfig& c, double v) { c.timeline.shrinkPerFrame = v; }},
      {"max_mask_area_ratio", [](RedactConfig& c, double v) { c.timeline.maxMaskAreaRatio = v; }},
```

- `SerialisesEveryKey` 的 `12u` 改成 `14u`。
- 加一个测试：

```cpp
// Mask smoothing (spec 2026-10-19 §3.1): a mask's width and height shrink by at most 2 % per frame, and a mask
// covers at most 4 times its box. The values are calibrated in the last task of the plan.
TEST(RedactConfig, ReadsTheMaskSmoothingSettings) {
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.shrinkPerFrame, 0.02);
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.maxMaskAreaRatio, 4.0);
  const RedactConfig c = parseRedactConfig(R"({"shrink_per_frame": 0.05, "max_mask_area_ratio": 3})", "t");
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, 0.05);
  EXPECT_DOUBLE_EQ(c.timeline.maxMaskAreaRatio, 3.0);
  EXPECT_NO_THROW(parseRedactConfig(R"({"shrink_per_frame": 1, "max_mask_area_ratio": 1})", "t"));
}
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`no member named 'shrinkPerFrame' in 'vc::TimelineConfig'`。

- [ ] **Step 3: 实现**

`cpp/src/track/box_timeline.h`：

- includes 加 `#include <unordered_map>`。
- `TimelineConfig` 在 `matchMaxDistanceLoose` 之后加：

```cpp
  double shrinkPerFrame = 0.02;   // per frame a mask's width and height shrink by at most this fraction, (0, 1]
  double maxMaskAreaRatio = 4.0;  // a mask covers at most this many times the area of its box, >= 1
```

- 类注释在 "…clipped to the frame." 之后、"A track's velocity runs…" 之前插入：

```
// The mask's size is smoothed per track (spec §3.1), around the centre of the box it is for: its width and
// height grow to the box's at once but shrink by at most shrinkPerFrame of themselves per frame, never
// below the box's (not at all while the track is held), so the mask always contains the box and follows a
// moving target without trailing behind it. A mask larger than maxMaskAreaRatio times the box has both
// sides pulled the same fraction of their way towards the box's until it is exactly that large.
```

- `private:` 里 `Match` 之后加：

```cpp
  // A track's smoothed mask size (spec §3.1).
  struct Size {
    double w = 0;
    double h = 0;
  };

  // The smoothed size after `shown` for a box of size `raw`: each side grows to raw's at once and otherwise
  // shrinks by at most `shrink` of itself, never below raw's; while `held` it does not shrink at all.
  static Size smoothed(Size shown, Size raw, double shrink, bool held);
  // `size` (at least `raw` on each side) with both sides pulled the same fraction of their way towards raw's,
  // just enough to bring its area down to `ratio` times raw's; unchanged when it is no larger than that.
  static Size cappedArea(Size size, Size raw, double ratio);
```

- `emit` 和 `hold` 的声明去掉末尾的 `const`；`emit` 的注释换成：

```cpp
  // Appends the mask of `track` for `region`: the region grown by expandFraction of its size plus
  // motionExpandFraction of `motion` per axis and to at least minBoxPx (the raw box), its size smoothed
  // against the track's previous mask (shown_) around the raw box's centre, clipped to the frame; nothing
  // when the raw box lies outside the frame.
```

- 成员里 `int64_t nextId_ = 1;` 之后加 `std::unordered_map<int64_t, Size> shown_;  // each live track's smoothed mask size`。

`cpp/src/track/box_timeline.cpp`：

- `validConfig` 的返回表达式末尾（`c.matchMaxDistanceLoose > 0` 之后）加
  `&& std::isfinite(c.shrinkPerFrame) && c.shrinkPerFrame > 0 && c.shrinkPerFrame <= 1 && std::isfinite(c.maxMaskAreaRatio) && c.maxMaskAreaRatio >= 1`。
- 在 `heldShift` 之前加两个静态成员函数：

```cpp
BoxTimeline::Size BoxTimeline::smoothed(Size shown, Size raw, double shrink, bool held) {
  const auto side = [shrink, held](double previous, double wanted) {
    if (wanted >= previous) return wanted;
    return held ? previous : std::max(wanted, previous * (1 - shrink));
  };
  return Size{side(shown.w, raw.w), side(shown.h, raw.h)};
}

BoxTimeline::Size BoxTimeline::cappedArea(Size size, Size raw, double ratio) {
  const double limit = ratio * raw.w * raw.h;
  if (size.w * size.h <= limit) return size;
  // Both sides a fraction f of their way out from raw's, (w + f a) x (h + f b) with a = W - w, b = H - h, is an
  // area that grows with f from w h <= limit (ratio >= 1) to W H > limit. It equals the limit at the positive root
  // of a b f^2 + p f - q = 0, p = a h + b w > 0, q = limit - w h >= 0, written in a form that also holds for a b = 0.
  const double a = size.w - raw.w;
  const double b = size.h - raw.h;
  const double p = a * raw.h + b * raw.w;
  const double q = limit - raw.w * raw.h;
  const double f = std::clamp(2 * q / (p + std::sqrt(p * p + 4 * a * b * q)), 0.0, 1.0);
  return Size{raw.w + f * a, raw.h + f * b};
}
```

- `emit` 换成（去掉 `const`）：

```cpp
void BoxTimeline::emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion,
                       MaskOrigin origin) {
  const double mx = config_.expandFraction * widthOf(region) + config_.motionExpandFraction * motion.x;
  const double my = config_.expandFraction * heightOf(region) + config_.motionExpandFraction * motion.y;
  double x1 = region.x1 - mx;
  double x2 = region.x2 + mx;
  double y1 = region.y1 - my;
  double y2 = region.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  const auto frameW = static_cast<double>(width_);
  const auto frameH = static_cast<double>(height_);
  if (!(std::min(x2, frameW) > std::max(x1, 0.0)) || !(std::min(y2, frameH) > std::max(y1, 0.0))) {
    shown_.erase(track.id);  // off the frame: a mask that comes back starts afresh
    return;
  }
  const Size raw{x2 - x1, y2 - y1};
  const auto it = shown_.find(track.id);
  const Size size = it == shown_.end() ? raw
                                       : cappedArea(smoothed(it->second, raw, config_.shrinkPerFrame,
                                                             origin == MaskOrigin::Held),
                                                    raw, config_.maxMaskAreaRatio);
  shown_[track.id] = size;
  const double cx = (x1 + x2) / 2.0;
  const double cy = (y1 + y2) / 2.0;
  out.push_back(MaskBox{region.cls, static_cast<float>(std::clamp(cx - size.w / 2.0, 0.0, frameW)),
                        static_cast<float>(std::clamp(cy - size.h / 2.0, 0.0, frameH)),
                        static_cast<float>(std::clamp(cx + size.w / 2.0, 0.0, frameW)),
                        static_cast<float>(std::clamp(cy + size.h / 2.0, 0.0, frameH)), origin, track.id});
}
```

- `hold` 的定义去掉 `const`（函数体不变）。
- `advance` 末尾，`keyTime_ = next.time;` 与 `tracks_ = std::move(tracks);` 这两行之前（不是开头第一次调用分支里的那个 `keyTime_ = next.time;`）加：

```cpp
  // Tracks that ended forget their masks.
  std::erase_if(shown_, [&tracks](const auto& entry) {
    return std::none_of(tracks.begin(), tracks.end(), [&entry](const Track& t) { return t.id == entry.first; });
  });
```

- `finish` 里 `tracks_.clear();` 之后加 `shown_.clear();`。

`cpp/apps/vc_redact/redact_config.cpp`：

- `kKeys` 改成 14 项（`std::array<std::string_view, 14>`），在 `"min_box_px",` 之后加 `"shrink_per_frame", "max_mask_area_ratio",`。
- `validateRedactConfig`：在 `min_box_px must be finite` 之后加

```cpp
  require(std::isfinite(c.timeline.shrinkPerFrame), "shrink_per_frame must be finite");
  require(std::isfinite(c.timeline.maxMaskAreaRatio), "max_mask_area_ratio must be finite");
```

  在 `min_box_px must be >= 0` 之后加

```cpp
  require(c.timeline.shrinkPerFrame > 0 && c.timeline.shrinkPerFrame <= 1, "shrink_per_frame must be in (0, 1]");
  require(c.timeline.maxMaskAreaRatio >= 1, "max_mask_area_ratio must be >= 1");
```

- `parseRedactConfig`：`c.timeline.minBoxPx = …` 之后加

```cpp
    c.timeline.shrinkPerFrame = j.value("shrink_per_frame", c.timeline.shrinkPerFrame);
    c.timeline.maxMaskAreaRatio = j.value("max_mask_area_ratio", c.timeline.maxMaskAreaRatio);
```

- `redactConfigToJson`：`j["min_box_px"] = …` 之后加

```cpp
  j["shrink_per_frame"] = c.timeline.shrinkPerFrame;
  j["max_mask_area_ratio"] = c.timeline.maxMaskAreaRatio;
```

`cpp/apps/vc_redact/redact_config.h` 的 `parseRedactConfig` 注释里 "detect_fps is in [5, 60];" 之后加 "shrink_per_frame is in (0, 1], max_mask_area_ratio >= 1;"。

`configs/redact.json` 在 `"min_box_px": 16.0,` 之后加两行：

```json
  "shrink_per_frame": 0.02,
  "max_mask_area_ratio": 4.0,
```

`cpp/README.md` 的"补框：一个目标一个框"一条末尾加：`每条轨迹的打码框大小还会平滑（平滑的是宽、高，不是各条边）：需要变大时立刻变大，需要变小时宽、高每帧最多缩小 shrink_per_frame（默认 0.02），且不小于当帧要盖的框；框始终以当帧要盖的框的中心为中心，所以快速移动的目标不会被拖出一条长尾；目标丢失、原地保持的那几帧不缩小。面积超过 max_mask_area_ratio（默认 4）倍时，宽、高按同一比例向要盖的框收缩到恰好 4 倍。`

`docs/USAGE.md` §4 表格在 `expand_fraction` 一行之后加两行：

```
| `shrink_per_frame` | 0.02 | 打码框的宽、高每帧最多缩小的比例（变大不受限）。调大框收得快、面积小，但更容易跳；调小更稳、面积大 |
| `max_mask_area_ratio` | 4 | 打码框面积最多是当帧要盖的框的几倍。调小面积小 |
```

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:RedactConfig.*:RedactJob.*'
```

预期：全部 PASSED（`BoxTimeline` 45 个），零警告。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/track/box_timeline.h cpp/src/track/box_timeline.cpp cpp/tests/box_timeline_test.cpp cpp/apps/vc_redact/redact_config.h cpp/apps/vc_redact/redact_config.cpp configs/redact.json cpp/tests/redact_config_test.cpp cpp/README.md docs/USAGE.md
git commit -m "feat: per-track mask size hysteresis and an area cap"
```

---

### Task 6: 低分检测的确认

本设计 §3.4：低于本类确认门槛的框，只有在相邻两个关键帧被匹配进同一条轨迹时才打码，确认后从第一次出现起打码；没确认的写进 `detections.jsonl` 并标 `confirmed: false`。

**Files:**
- Modify: `cpp/src/track/box_timeline.h`（`TimelineConfig::confirmScore`、`Track::confirmed`、`needsConfirmation`、类注释）、`cpp/src/track/box_timeline.cpp`（`validConfig`、`advance`、`finish`）
- Modify: `cpp/apps/vc_redact/redact_config.h`, `.cpp`（两个键、`timelineFor`）、`configs/redact.json`
- Modify: `cpp/apps/vc_redact/records.h`, `.cpp`（`detectionRecord` 的 `confirmed`）
- Modify: `cpp/apps/vc_redact/redact_job.cpp`（`Processor`：`timelineFor`、延后一个关键帧写检测）
- Modify: `py/vc_eval/records.py`（`detections_by_frame`）、`py/vc_eval/temporal.py`
- Modify: `cpp/README.md`、`docs/USAGE.md`
- Test: `cpp/tests/box_timeline_test.cpp`、`cpp/tests/redact_config_test.cpp`、`cpp/tests/records_test.cpp`、`cpp/tests/redact_job_test.cpp`、`py/tests/test_eval_coverage.py`

**Interfaces:**
- Consumes: Task 4 的 `TimelineOutput::confirmed`、Task 5 的 `shown_`。
- Produces:
  - `TimelineConfig::confirmScore`（`std::vector<float>`，按类别下标；没有对应项的类别不需要确认；每项 ∈ [0, 1]）。
  - `TimelineOutput::confirmed[i] == false` 当且仅当起始关键帧的第 i 个框分数低于门槛、它的轨迹此前没确认、且在下一个关键帧没配上（或视频在此结束）。
  - 没确认的轨迹没配上就结束，不留幽灵；配上即确认（前一段从它的第一个关键帧起插值打码）；高分框出现即打码；确认过的轨迹之后的低分框不再需要确认。
  - `RedactConfig::faceConfirmScore = 0.5f`、`RedactConfig::plateConfirmScore = 0.15f`；配置键 `face_confirm_score`、`plate_confirm_score`（各 ∈ [0, 1]，放在最后）。
  - `TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes);` —— `"face"` → `faceConfirmScore`，`"plate"` → `plateConfirmScore`，其他类别 0。
  - `std::string detectionRecord(…, const std::string& src, std::optional<bool> confirmed = std::nullopt);` —— 给了就在 `src` 之后写 `"confirmed": true|false`；`vc_redact` 每一行都写，`vc_detect` 不写（不变）。
  - `detections.jsonl` 的行在下一个关键帧（或视频结尾）才写出，顺序仍按帧号。
  - Python：`detections_by_frame(path, confirmed_only=False)`；`evaluate_temporal` 的逐帧参照只用没有标 `"confirmed": false` 的框。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/box_timeline_test.cpp`：在 `smooth()` 之后加

```cpp
// exact() with the spec's thresholds (§3.4): faces scoring below 0.5 and plates below 0.15 need confirming.
TimelineConfig confirming() {
  TimelineConfig c = exact();
  c.confirmScore = {0.5f, 0.15f};
  return c;
}

Box weakFace(float x1, float y1, float x2, float y2) { return Box{kFace, x1, y1, x2, y2, 0.3f}; }
```

文件末尾追加：

```cpp
// A box below its class's threshold seen on one keyframe only is not masked, and reported unconfirmed.
TEST(BoxTimeline, ALowScoreBoxSeenOnceIsNotMasked) {
  BoxTimeline timeline(confirming(), 1000, 1000);
  timeline.advance({}, key(0, {weakFace(100, 100, 120, 120)}));
  const TimelineOutput out = timeline.advance(frames(0, 12), key(12, {}));
  for (const auto& m : out.masks) EXPECT_TRUE(m.empty());
  EXPECT_EQ(out.confirmed, std::vector<bool>{false});

  // Nor at the end of the video.
  BoxTimeline ending(confirming(), 1000, 1000);
  ending.advance({}, key(0, {weakFace(100, 100, 120, 120)}));
  const TimelineOutput tail = ending.finish(frames(0, 6));
  for (const auto& m : tail.masks) EXPECT_TRUE(m.empty());
  EXPECT_EQ(tail.confirmed, std::vector<bool>{false});
}

// Matched into the same track at the next keyframe, it is confirmed there and masked from its first keyframe on
// (the interval waits for that keyframe anyway, so this adds no delay). Once confirmed, the track needs no
// further confirmation, also when it is lost.
TEST(BoxTimeline, ALowScoreBoxConfirmedAtTheNextKeyframeIsMaskedFromItsFirstKeyframe) {
  BoxTimeline timeline(confirming(), 1000, 1000);
  timeline.advance({}, key(0, {weakFace(100, 100, 120, 120)}));
  const TimelineOutput out = timeline.advance(frames(0, 12), key(12, {weakFace(112, 100, 132, 120)}));
  EXPECT_EQ(out.confirmed, std::vector<bool>{true});
  for (const auto& m : out.masks) ASSERT_EQ(m.size(), 1u);
  expectBox(out.masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  expectBox(out.masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
  const TimelineOutput lost = timeline.advance(frames(12, 24), key(24, {}));
  EXPECT_EQ(lost.confirmed, std::vector<bool>{true});
  ASSERT_EQ(lost.masks[6].size(), 1u);
  EXPECT_EQ(lost.masks[6][0].origin, MaskOrigin::Held);
}

// At or above its class's threshold a box is masked at once; thresholds go by class, whatever the source.
TEST(BoxTimeline, AHighScoreBoxIsMaskedAtOnceAndThresholdsGoByClass) {
  BoxTimeline timeline(confirming(), 1000, 1000);
  timeline.advance({}, key(0, {Box{kFace, 100, 100, 120, 120, 0.5f}, Box{kPlate, 300, 300, 348, 316, 0.15f}}));
  const TimelineOutput atThreshold = timeline.advance(frames(0, 12), key(12, {}));
  EXPECT_EQ(atThreshold.confirmed, (std::vector<bool>{true, true}));
  EXPECT_EQ(atThreshold.masks[6].size(), 2u);

  // 0.3 is low for a face (0.5) and high for a plate (0.15).
  BoxTimeline mixed(confirming(), 1000, 1000);
  mixed.advance({}, key(0, {Box{kFace, 100, 100, 120, 120, 0.3f}, Box{kPlate, 300, 300, 348, 316, 0.3f}}));
  const TimelineOutput out = mixed.advance(frames(0, 12), key(12, {}));
  EXPECT_EQ(out.confirmed, (std::vector<bool>{false, true}));
  ASSERT_EQ(out.masks[6].size(), 1u);
  EXPECT_EQ(out.masks[6][0].cls, kPlate);
}

// "Two consecutive keyframes": seen at 0, missed at 12, seen again at 24 is not a confirmation. The unconfirmed
// track leaves no ghost, so the box at 24 starts a new, again unconfirmed track.
TEST(BoxTimeline, AnUnconfirmedTrackLeavesNoGhost) {
  BoxTimeline timeline(confirming(), 1000, 1000);
  timeline.advance({}, key(0, {weakFace(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {}));
  const TimelineOutput again = timeline.advance(frames(12, 24), key(24, {weakFace(100, 100, 120, 120)}));
  for (const auto& m : again.masks) EXPECT_TRUE(m.empty());
  EXPECT_TRUE(again.confirmed.empty());  // keyframe 12 had no boxes
  const TimelineOutput tail = timeline.finish(frames(24, 30));
  for (const auto& m : tail.masks) EXPECT_TRUE(m.empty());
  EXPECT_EQ(tail.confirmed, std::vector<bool>{false});
}

TEST(BoxTimeline, RejectsAnInvalidConfirmScore) {
  for (const float bad : {-0.1f, 1.5f, std::numeric_limits<float>::quiet_NaN()}) {
    TimelineConfig c = confirming();
    c.confirmScore[1] = bad;
    EXPECT_THROW(BoxTimeline(c, 10, 10), std::invalid_argument) << bad;
  }
  EXPECT_TRUE(TimelineConfig{}.confirmScore.empty());  // by default nothing needs confirming
}
```

`cpp/tests/redact_config_test.cpp`：

- `kAllKeys` 末尾加 `"face_confirm_score", "plate_confirm_score"`。
- `expectSameConfig` 末尾加

```cpp
  EXPECT_FLOAT_EQ(c.faceConfirmScore, d.faceConfirmScore);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, d.plateConfirmScore);
```

- `nonDefaultConfig` 在 `return c;` 之前加 `c.faceConfirmScore = 0.375f;` 和 `c.plateConfirmScore = 0.0625f;`。
- `RejectsBadInput` 末尾加 `R"({"face_confirm_score": 1.5})", R"({"plate_confirm_score": -0.1})", R"({"face_confirm_score": "0.5"})",`。
- `RejectsOutOfRangeValuesAndNulls` 末尾加 `R"({"face_confirm_score": null})", R"({"plate_confirm_score": null})",`。
- `ValidateRefusesEveryNonFiniteDoubleField` 的 `fields` 末尾加

```cpp
      {"face_confirm_score", [](RedactConfig& c, double v) { c.faceConfirmScore = static_cast<float>(v); }},
      {"plate_confirm_score", [](RedactConfig& c, double v) { c.plateConfirmScore = static_cast<float>(v); }},
```

- `SerialisesEveryKey` 的 `14u` 改成 `16u`。
- `FloatSettingsAreWrittenWithSixDecimals` 末尾加

```cpp
  EXPECT_EQ(j.at("face_confirm_score").dump(), "0.5");
  EXPECT_EQ(j.at("plate_confirm_score").dump(), "0.15");
```

- 加两个测试：

```cpp
// Low-score confirmation (spec 2026-10-19 §3.4): faces below 0.5 and plates below 0.15 are masked only once seen on
// two consecutive keyframes. The values are calibrated in the last task of the plan.
TEST(RedactConfig, ReadsTheConfirmScores) {
  EXPECT_FLOAT_EQ(RedactConfig{}.faceConfirmScore, 0.5f);
  EXPECT_FLOAT_EQ(RedactConfig{}.plateConfirmScore, 0.15f);
  const RedactConfig c = parseRedactConfig(R"({"face_confirm_score": 0.4, "plate_confirm_score": 0})", "t");
  EXPECT_FLOAT_EQ(c.faceConfirmScore, 0.4f);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, 0.0f);
}

// The thresholds go to the timeline by the detector's class names, in its class order; other classes need none.
TEST(RedactConfig, ConfirmScoresFollowTheDetectorsClassNames) {
  RedactConfig c;
  c.faceConfirmScore = 0.4f;
  c.plateConfirmScore = 0.2f;
  c.timeline.expandFraction = 0.25;
  EXPECT_EQ(timelineFor(c, {"plate", "face"}).confirmScore, (std::vector<float>{0.2f, 0.4f}));
  EXPECT_EQ(timelineFor(c, {"face", "plate", "sign"}).confirmScore, (std::vector<float>{0.4f, 0.2f, 0.0f}));
  EXPECT_DOUBLE_EQ(timelineFor(c, {"face"}).expandFraction, 0.25);
}
```

`cpp/tests/records_test.cpp` 加：

```cpp
TEST(Records, DetectionLineCarriesTheConfirmationWhenGiven) {
  const Box b{0, 812.06f, 433.5f, 846.2f, 475.0f, 0.27f};
  EXPECT_EQ(detectionRecord("trip.mp4", 1234, 20566.74, b, kClasses, "student", false),
            R"({"video":"trip.mp4","frame":1234,"pts_ms":20566.7,"cls":"face","x1":812.1,"y1":433.5,)"
            R"("x2":846.2,"y2":475.0,"score":0.27,"src":"student","confirmed":false})"
            "\n");
  EXPECT_NE(detectionRecord("trip.mp4", 1, 0.0, b, kClasses, "yunet", true).find(R"("confirmed":true})"),
            std::string::npos);
}
```

`cpp/tests/redact_job_test.cpp` 加：

```cpp
// Low-score boxes (below face_confirm_score, 0.5 by default) are masked only once their track is matched at two
// consecutive keyframes, and then from the first of them on; one seen once is not masked and is written to
// detections.jsonl with "confirmed": false (spec 2026-10-19 §3.4). Keyframes every 6 frames (5 fps on 30 fps).
TEST(RedactJob, MasksALowScoreFaceOnlyOnceItIsConfirmed) {
  RedactOptions o = baseOptions("redact_confirm");
  o.config.faceConfirmScore = 0.5f;  // the threshold this test is written for, whatever the calibrated default
  const fs::path fake = test::tmpDir() / "redact_confirm.jsonl";
  {
    std::ofstream out(fake);
    const auto face = [&](int frame, int x1) {
      out << R"({"video":"noisy_audio.mp4","frame":)" << frame << R"(,"pts_ms":0,"cls":"face","x1":)" << x1
          << R"(,"y1":100,"x2":)" << x1 + 80 << R"(,"y2":180,"score":0.3,"src":"fake"})" << '\n';
    };
    face(12, 100);  // seen once
    face(24, 300);  // seen on two consecutive keyframes
    face(30, 300);
  }
  o.fakeDetections = fake;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 10);
  std::map<int, bool> confirmed;
  std::ifstream detections(s.outputDir / "detections.jsonl");
  for (std::string line; std::getline(detections, line);) {
    const nlohmann::json d = nlohmann::json::parse(line);
    confirmed[d.at("frame").get<int>()] = d.at("confirmed").get<bool>();
  }
  EXPECT_EQ(confirmed, (std::map<int, bool>{{12, false}, {24, true}, {30, true}}));
  std::ifstream masks(s.outputDir / "masks.jsonl");
  for (std::string line; std::getline(masks, line);) {
    const nlohmann::json record = nlohmann::json::parse(line);
    const int frame = record.at("frame").get<int>();
    // from the first of the two keyframes (24) to the end of the hold after the second (35)
    EXPECT_EQ(record.at("boxes").size(), frame >= 24 && frame < 36 ? 1u : 0u) << "frame " << frame;
  }
  EXPECT_EQ(s.maskedFrames, 12);
}
```

`py/tests/test_eval_coverage.py` 在 `test_temporal_accepts_masks_covering_exactly_the_frames_of_the_dense_run` 之后加：

```python
def test_temporal_leaves_out_dense_boxes_vc_redact_never_masked(tmp_path):
    """A box the dense run marked "confirmed": false (a low-score box seen once) is no reference; the others are."""
    dense = tmp_path / "dense" / "detections.jsonl"
    dense.parent.mkdir()
    box = Box("face", 100, 100, 140, 150, score=0.3, src="student")
    records = [
        {**Detection("a.mp4", 0, 0.0, box).to_record(), "confirmed": False},
        {**Detection("a.mp4", 1, 16.7, box).to_record(), "confirmed": True},
    ]
    dense.write_text("".join(json.dumps(r) + "\n" for r in records))
    _dense_stats(dense, 2)
    sparse = _masks(tmp_path / "s.jsonl", "a.mp4", {}, [0, 1])
    report = evaluate_temporal(dense, sparse, H)
    assert (report.classes["face"].hits, report.classes["face"].total) == (0, 1)
    assert [m.frame for m in report.misses] == [1]
    assert set(detections_by_frame(dense)[1]) == {0, 1}  # without confirmed_only every box is read
    assert set(detections_by_frame(dense, confirmed_only=True)[1]) == {1}
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py -p no:cacheprovider -q -k never_masked
```

预期：编译失败，`no member named 'confirmScore' in 'vc::TimelineConfig'`；pytest FAIL：`assert (0, 2) == (0, 1)`（逐帧参照还算着那个 `confirmed: false` 的框）。

- [ ] **Step 3: 实现 `BoxTimeline` 的确认**

`cpp/src/track/box_timeline.h`：

- `TimelineConfig` 末尾加：

```cpp
  // Per class index: a box scoring below its class's entry is masked only once its track is matched at
  // two consecutive keyframes (spec 2026-10-19 §3.4); a class without an entry needs no confirmation.
  std::vector<float> confirmScore{};
```

- 类注释末尾（"Feed keyframes in order; …"之前）插入：

```
// A box scoring below confirmScore[cls] is masked only once its track is matched at the next keyframe,
// and then from its own keyframe on; a track that is not confirmed that way ends unmasked and leaves no
// ghost. Higher-scoring boxes are masked at once, and a confirmed track needs no further confirmation.
// TimelineOutput::confirmed says which boxes of the starting keyframe were masked.
```

- `Track` 末尾加 `bool confirmed = true;  // masked: a high score, or matched at two consecutive keyframes`。
- 私有成员函数里加 `bool needsConfirmation(const Box& box) const;  // scores below its class's confirmScore`。

`cpp/src/track/box_timeline.cpp`：

- `validConfig` 换成：

```cpp
bool validConfig(const TimelineConfig& c) {
  const bool numbers =
      std::isfinite(c.matchMaxDistance) && c.matchMaxDistance > 0 && std::isfinite(c.expandFraction) &&
      c.expandFraction >= 0 && std::isfinite(c.motionExpandFraction) && c.motionExpandFraction >= 0 &&
      std::isfinite(c.minBoxPx) && c.minBoxPx >= 0 && std::isfinite(c.matchMaxDistanceLoose) &&
      c.matchMaxDistanceLoose > 0 && std::isfinite(c.shrinkPerFrame) && c.shrinkPerFrame > 0 &&
      c.shrinkPerFrame <= 1 && std::isfinite(c.maxMaskAreaRatio) && c.maxMaskAreaRatio >= 1;
  // NaN fails both comparisons.
  return numbers && std::all_of(c.confirmScore.begin(), c.confirmScore.end(),
                                [](float s) { return s >= 0.0f && s <= 1.0f; });
}
```

- 在 `heldShift` 之前加：

```cpp
bool BoxTimeline::needsConfirmation(const Box& box) const {
  if (box.cls < 0 || static_cast<size_t>(box.cls) >= config_.confirmScore.size()) return false;
  return box.score < config_.confirmScore[static_cast<size_t>(box.cls)];
}
```

- `advance` 整个换成：

```cpp
TimelineOutput BoxTimeline::advance(const std::vector<FrameTime>& frames, const Keyframe& next) {
  requireFinite(next);
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    tracks_.clear();
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& box = next.boxes[j];
      tracks_.push_back(
          Track{nextId_++, box, std::nullopt, false, box, 0.0, static_cast<int>(j), !needsConfirmation(box)});
    }
    keyTime_ = next.time;
    keyBoxes_ = next.boxes.size();
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (frames.empty() || frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prevTime.index);
  std::vector<std::optional<size_t>> matchedTo(tracks_.size());
  std::vector<std::optional<size_t>> matchedFrom(next.boxes.size());
  for (const Match& m : match(next, steps)) {
    matchedTo[m.track] = m.next;
    matchedFrom[m.next] = m.track;
  }

  TimelineOutput out;
  out.masks.resize(frames.size());
  out.confirmed.assign(keyBoxes_, false);
  const std::vector<double> t = fractions(frames, prevTime, next.time);
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    // A match confirms a track: two consecutive keyframes, or a ghost (always confirmed) found again.
    const bool masked = track.confirmed || matchedTo[i].has_value();
    if (track.keyBox >= 0) out.confirmed[static_cast<size_t>(track.keyBox)] = masked;
    if (matchedTo[i]) {
      const Box& to = next.boxes[*matchedTo[i]];
      const Point a = centerOf(track.box);
      const Point b = centerOf(to);
      const Offset motion{std::abs(b.x - a.x), std::abs(b.y - a.y)};
      for (size_t k = 0; k < frames.size(); ++k) {
        const MaskOrigin origin =
            k > 0 ? MaskOrigin::Interpolated : track.ghost ? MaskOrigin::Held : MaskOrigin::Detected;
        emit(out.masks[k], track, lerp(track.box, to, t[k]), motion, origin);
      }
    } else if (!track.ghost && masked) {
      hold(out.masks, frames, prevTime, track, steps);
    }
    // An unmatched ghost, or an unconfirmed track seen once, ends here unmasked.
  }

  // The boxes of `next` in order (a matched one continues its track, confirmed), then the ghosts of the
  // confirmed real detections that were missed, at the end of their hold.
  std::vector<Track> tracks;
  tracks.reserve(next.boxes.size() + tracks_.size());
  for (size_t j = 0; j < next.boxes.size(); ++j) {
    const Box& box = next.boxes[j];
    Track fresh{0, box, std::nullopt, false, box, 0.0, static_cast<int>(j), true};
    if (const auto from = matchedFrom[j]) {
      const Track& old = tracks_[*from];
      const Point a = centerOf(old.anchor);
      const Point b = centerOf(box);
      const double span = old.anchorSteps + steps;
      fresh.id = old.id;
      fresh.velocity = Offset{(b.x - a.x) / span, (b.y - a.y) / span};
    } else {
      fresh.id = nextId_++;
      fresh.confirmed = !needsConfirmation(box);
    }
    tracks.push_back(fresh);
  }
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    if (matchedTo[i] || track.ghost || !track.confirmed) continue;
    const Offset s = heldShift(track, steps);
    tracks.push_back(Track{track.id, shifted(track.box, s.x, s.y), track.velocity, true, track.anchor,
                           track.anchorSteps + steps, -1, true});
  }
  // Tracks that ended forget their masks.
  std::erase_if(shown_, [&tracks](const auto& entry) {
    return std::none_of(tracks.begin(), tracks.end(), [&entry](const Track& t) { return t.id == entry.first; });
  });
  keyTime_ = next.time;
  tracks_ = std::move(tracks);
  keyBoxes_ = next.boxes.size();
  return out;
}
```

- `finish` 整个换成：

```cpp
TimelineOutput BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (!frames.empty() && frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  TimelineOutput out;
  out.masks.resize(frames.size());
  out.confirmed.assign(keyBoxes_, false);
  // The motion margin of the tail: the shift one frame past its end, as for an interval's next keyframe.
  const double span = frames.empty() ? 0.0 : static_cast<double>(frames.back().index - prevTime.index + 1);
  for (const Track& track : tracks_) {
    if (track.keyBox >= 0) out.confirmed[static_cast<size_t>(track.keyBox)] = track.confirmed;
    if (!track.ghost && track.confirmed) hold(out.masks, frames, prevTime, track, span);
  }
  keyTime_.reset();
  tracks_.clear();
  shown_.clear();
  keyBoxes_ = 0;
  nextId_ = 1;
  return out;
}
```

- [ ] **Step 4: 实现配置、记录和 `Processor`**

`cpp/apps/vc_redact/redact_config.h`：includes 加 `<vector>`；`RedactConfig` 末尾加

```cpp
  // Boxes scoring below these are masked only once their track is matched at two consecutive keyframes
  // (spec 2026-10-19 §3.4), by class whatever the detector; a merged box goes by its (higher) merged score.
  float faceConfirmScore = 0.5f;
  float plateConfirmScore = 0.15f;
```

`parseRedactConfig` 注释里 "face_score_threshold is in [0, 1]" 改成 "face_score_threshold, face_confirm_score and plate_confirm_score are in [0, 1]"；文件末尾（`redactConfigToJson` 声明之后）加

```cpp
// config.timeline with confirmScore set for a detector with these classes: "face" -> faceConfirmScore,
// "plate" -> plateConfirmScore, any other class 0 (needs no confirmation).
TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes);
```

`cpp/apps/vc_redact/redact_config.cpp`：

- `kKeys` 改成 16 项，末尾加 `"face_confirm_score", "plate_confirm_score"`。
- `validateRedactConfig`：`face_nms_iou must be finite` 之后加

```cpp
  require(std::isfinite(c.faceConfirmScore), "face_confirm_score must be finite");
  require(std::isfinite(c.plateConfirmScore), "plate_confirm_score must be finite");
```

  `face_nms_iou must be in (0, 1]` 之后加

```cpp
  require(c.faceConfirmScore >= 0 && c.faceConfirmScore <= 1, "face_confirm_score must be in [0, 1]");
  require(c.plateConfirmScore >= 0 && c.plateConfirmScore <= 1, "plate_confirm_score must be in [0, 1]");
```

- `parseRedactConfig`：`c.face.nmsIou = …` 之后加

```cpp
    c.faceConfirmScore = j.value("face_confirm_score", c.faceConfirmScore);
    c.plateConfirmScore = j.value("plate_confirm_score", c.plateConfirmScore);
```

- `redactConfigToJson`：末尾 `return j;` 之前加

```cpp
  j["face_confirm_score"] = jsonFloat(c.faceConfirmScore);
  j["plate_confirm_score"] = jsonFloat(c.plateConfirmScore);
```

- 文件末尾（命名空间结束前）加

```cpp
TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes) {
  TimelineConfig t = config.timeline;
  t.confirmScore.clear();
  for (const std::string& name : classes) {
    t.confirmScore.push_back(name == "face"    ? config.faceConfirmScore
                             : name == "plate" ? config.plateConfirmScore
                                               : 0.0f);
  }
  return t;
}
```

`configs/redact.json` 在 `"face_nms_iou": 0.3` 之后加（注意前一行补逗号）：

```json
  "face_confirm_score": 0.5,
  "plate_confirm_score": 0.15
```

`cpp/apps/vc_redact/records.h`：includes 加 `<optional>`；`detectionRecord` 的声明和注释换成

```cpp
// One detections.jsonl line (spec §3.2), newline included; with `confirmed` a last field saying whether
// the box was masked (false: a low-score box seen on one keyframe only, spec 2026-10-19 §3.4). Throws
// VcError when box.cls is not an index into `classes`, or when ptsMs, a box coordinate or the score is
// not finite (or overflows when rounded). Invalid UTF-8 in `video` or `src` is written as U+FFFD.
std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src,
                            std::optional<bool> confirmed = std::nullopt);
```

`cpp/apps/vc_redact/records.cpp`：定义的签名同样加 `std::optional<bool> confirmed`（不写默认值），`j["src"] = src;` 之后加 `if (confirmed) j["confirmed"] = *confirmed;`。

`cpp/apps/vc_redact/redact_job.cpp` 的 `Processor`：

- 构造函数初始化列表里 `timeline_(options.config.timeline, info.width, info.height)` 换成 `timeline_(timelineFor(options.config, detector.classes()), info.width, info.height)`。
- `onVideo` 从 `std::vector<Box> boxes;` 到函数结尾换成：

```cpp
    std::vector<Box> boxes;
    boxes.reserve(found.size());
    for (const SourcedBox& b : found) boxes.push_back(b.box);

    const Clock::time_point trackStarted = Clock::now();
    const TimelineOutput step = timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    // The previous keyframe's boxes are written now that the timeline has said which of them it masked.
    writeDetections(step.confirmed);
    pending_ = KeyframeDetections{item.index, item.ptsMs, found};
    emit(step.masks);
    buffer_.push_back(std::move(item));
  }
```

- `onEnd` 换成：

```cpp
  void onEnd() {
    const Clock::time_point started = Clock::now();
    const TimelineOutput step = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    writeDetections(step.confirmed);
    emit(step.masks);
    if (!out_.push(Item{EndItem{}})) throw PipelineAborted{};
  }
```

- `private:` 开头加：

```cpp
  // A keyframe's detections, written once the next keyframe (or the end) says which were masked.
  struct KeyframeDetections {
    int64_t frame = 0;
    double ptsMs = 0;
    std::vector<SourcedBox> boxes{};
  };

  // Writes the pending keyframe's detections.jsonl lines with their confirmation flags.
  void writeDetections(const std::vector<bool>& confirmed) {
    if (!pending_) return;
    if (confirmed.size() != pending_->boxes.size()) {
      throw VcError("internal error: confirmation flags do not match the keyframe's detections");
    }
    for (size_t i = 0; i < confirmed.size(); ++i) {
      const SourcedBox& b = pending_->boxes[i];
      detections_ << detectionRecord(video_, pending_->frame, pending_->ptsMs, b.box, detector_.classes(), b.src,
                                     static_cast<bool>(confirmed[i]));
    }
    pending_.reset();
  }
```

- 成员里 `std::vector<VideoItem> buffer_;` 之后加 `std::optional<KeyframeDetections> pending_;`；文件 includes 加 `<optional>`。

`py/vc_eval/records.py` 的 `detections_by_frame` 换成：

```python
def detections_by_frame(path: Path, confirmed_only: bool = False) -> tuple[set[str], dict[int, list[Box]]]:
    """(video names, frame -> boxes) of a detections JSONL; an empty file gives no detections.

    With `confirmed_only`, boxes vc_redact wrote with "confirmed": false (low-score boxes seen on one keyframe only,
    which it never masked) are left out.
    """
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    videos: set[str] = set()
    by_frame: dict[int, list[Box]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                d = Detection.from_record(record)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path} line {n}: bad detection record ({e})") from None
            if confirmed_only and record.get("confirmed") is False:
                continue
            videos.add(d.video)
            by_frame[d.frame].append(d.box)
    return videos, dict(by_frame)
```

`py/vc_eval/temporal.py` 的 `evaluate_temporal` 第一行换成 `dense_videos, dense = detections_by_frame(dense_detections, confirmed_only=True)`，docstring 末尾加一句 "Boxes the dense run never masked (written "confirmed": false) are no reference."。

`cpp/README.md`：

- `detections.jsonl` 一条末尾加：`；每行最后有 confirmed（true / false），false 表示分数低于本类确认门槛、只在一个关键帧出现、没有打码的框。一个关键帧的检测要等到下一个关键帧（或视频结尾）才写出`。
- "补框：一个目标一个框"一条末尾加：`分数低于确认门槛（face_confirm_score 默认 0.5、plate_confirm_score 默认 0.15，按类别判断，与来源无关，合并框按合并后的分数）的框，只有在相邻两个关键帧被配进同一条轨迹时才打码，确认后从它第一次出现的关键帧起打码（那一段本来就要等下一个关键帧才输出，不增加延迟）；高分框出现即打码，确认过的轨迹之后的低分框不再需要确认。`
- "已知的局限"加一条：`  - 只被低分检到一次的真目标会漏，例如快速横穿、只露一次的小脸。`

`docs/USAGE.md` §4 表格末尾加两行：

```
| `face_confirm_score` | 0.5 | 人脸分数低于它的框要在相邻两个关键帧都出现才打码。调低多打、调高少打（误检少，但只露一次的小脸会漏） |
| `plate_confirm_score` | 0.15 | 车牌的同一门槛 |
```

- [ ] **Step 5: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:RedactConfig.*:Records.*:RedactJob.*'
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py -p no:cacheprovider -q
```

预期：gtest 全部 PASSED（`BoxTimeline` 50 个），零警告；pytest 全部通过。

- [ ] **Step 6: 提交**

```bash
git add cpp/src/track cpp/apps/vc_redact configs/redact.json cpp/tests/box_timeline_test.cpp cpp/tests/redact_config_test.cpp cpp/tests/records_test.cpp cpp/tests/redact_job_test.cpp py/vc_eval/records.py py/vc_eval/temporal.py py/tests/test_eval_coverage.py cpp/README.md docs/USAGE.md
git commit -m "feat: mask low-score boxes only once confirmed on two consecutive keyframes"
```

---

### Task 7: 马赛克网格锚定画面、块大小按轨迹固定

本设计 §4。

**Files:**
- Rewrite: `cpp/src/mask/mosaic.h`, `cpp/src/mask/mosaic.cpp`
- Rewrite: `cpp/tests/mosaic_test.cpp`
- Modify: `cpp/src/track/box_timeline.h`（`MaskBox::block`）
- Modify: `cpp/apps/vc_redact/records.cpp`, `.h`（`block` 字段）
- Modify: `cpp/apps/vc_redact/redact_job.cpp`（`Processor::emit`、成员 `blocks_`）
- Modify: `cpp/README.md`、`docs/USAGE.md`
- Test: `cpp/tests/mosaic_test.cpp`、`cpp/tests/records_test.cpp`、`cpp/tests/redact_job_test.cpp`

**Interfaces:**
- Consumes: `MaskBox::track`（Task 4）；`MosaicConfig{minBlockPx = 16, blocksPerLongSide = 8}`（不变）。
- Produces:
  - `inline constexpr std::array<int, 9> kMosaicBlockLadder = {16, 24, 32, 48, 64, 96, 128, 192, 256};`
  - `int mosaicBlockSize(const PixelRect&, const MosaicConfig&);` —— 不变：`max(minBlockPx, ⌈长边 / blocksPerLongSide⌉)`，"想要的块大小"。
  - `int ladderBlock(int wanted);` —— 档位表里第一个 ≥ `wanted` 的值；超过 256 时取 256。
  - `int nextTrackBlock(int current, int wanted);` —— `current == 0`（新轨迹）取 `ladderBlock(wanted)`；之后只有 `wanted > 1.5 × current` 时才升到 `ladderBlock(wanted)`，从不降档。
  - `class TrackBlocks { explicit TrackBlocks(const MosaicConfig&); int blockFor(int64_t track, const PixelRect& rect); void endFrame(); };` —— 一帧里没有框的轨迹在 `endFrame()` 时被忘掉。
  - `struct MosaicRect { PixelRect rect; int block = 16; };`
  - `void applyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects);` —— 每个框的四边向外扩到块大小的整数倍坐标（`x1 = ⌊x1/b⌋·b`、`x2 = ⌈x2/b⌉·b`），再裁到画面内；网格从画面原点起算，每个网格块取块内均值；画面右、下边缘的不完整块取实际像素的均值；色度平面用 `b >> log2_chroma_w` × `b >> log2_chroma_h`（4:2:0 为 b/2）、坐标同样换算；按块大小从大到小打（相同块大小保持原顺序）。块大小必须是 ≥ 2 的偶数，否则在动任何像素之前抛 `std::invalid_argument`。不再有"窄于半块的尾块并入"规则；旧签名 `applyMosaic(AVFrame&, const std::vector<PixelRect>&, const MosaicConfig&)` 删除。
  - `MaskBox::block`（`int`，BoxTimeline 填 0，`vc_redact` 填入）；`masks.jsonl` 每个框字段顺序 `cls, x1, y1, x2, y2, track, origin, block`。

**测试改动清单（`cpp/tests/mosaic_test.cpp` 整个重写）：** `BlockSizeFollowsTheLongSide`、`CoveringRectRoundsOutwardAndClips`、`CoveringRectRoundsAFractionalBottomEdgeUp`、`CoveringRectClampsHugeValuesBeforeConverting`、`CoveringRectRefusesNonFiniteCoordinates` 原样保留；`RefusesSharedBuffersAndOtherFormats` 改用新签名；`PixelatesYuv420pAndLeavesTheRestAlone`、`PixelatesNv12InterleavedChroma`、`ClipsRectsToTheFrame`、`PixelatesAnOddOriginAndExtent`、`PixelatesAWideRectWithOddChromaBlocks`、`PixelatesARectWhoseBlockSizeIsOdd`、`MergesARemainderOfExactlyHalfTheOddBlockSizeMinusOne`、`PixelatesAnOddFrameSize`、`PixelatesYuv422pAndYuv444p`、`AppliesOverlappingRectsInOrder`、`MergesAOnePixelRemainderIntoThePreviousBlock`、`KeepsAHalfBlockRemainderAsItsOwnBlock`、`RejectsAnInvalidConfigBeforeTouchingAnyPixel` 删除——它们验证的是"网格从框的左上角起算"、"块大小随框变"、"尾块并入"和"按列表顺序打码"，这些都被 §4 取代；对应的新测试是 `PixelatesOnTheGridOfTheWholeFrame`、`PixelatesNv12OnTheSameGrid`、`AMovingMaskKeepsItsPattern`、`PixelatesYuv422pAndYuv444pOnTheGrid`、`CellsCutByTheFrameEdgeAverageWhatIsThere`、`AppliesLargerBlocksFirst`、`RejectsAnInvalidBlockBeforeTouchingAnyPixel`，加上档位的 `LadderBlockIsTheFirstStepNotBelowTheWantedSize`、`ATrackMovesUpTheLadderOnlyPastOneAndAHalfTimesItsBlock`、`TrackBlocksFollowEachTrackAndForgetEndedOnes`。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/mosaic_test.cpp` 整个换成：

```cpp
#include "mask/mosaic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "common/vc_error.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

// Every cellW x cellH cell of the grid anchored at the plane's origin, inside `area`, is uniform and equals the
// rounded mean of `before` over the cell's part inside `area`; everything outside `area` is unchanged. `area`'s
// left and top edges lie on the grid.
void expectGrid(const AVFrame& after, const AVFrame& before, int plane, int step, int offset, PixelRect area,
                int cellW, int cellH, int planeWidth, int planeHeight) {
  const auto px = [&](const AVFrame& f, int x, int y) {
    return f.data[plane][y * f.linesize[plane] + x * step + offset];
  };
  for (int y = 0; y < planeHeight; ++y) {
    for (int x = 0; x < planeWidth; ++x) {
      const bool inside = x >= area.x1 && x < area.x2 && y >= area.y1 && y < area.y2;
      if (!inside) {
        ASSERT_EQ(px(after, x, y), px(before, x, y)) << "plane " << plane << " outside at " << x << "," << y;
        continue;
      }
      const int cx1 = x / cellW * cellW;
      const int cy1 = y / cellH * cellH;
      const int cx2 = std::min(cx1 + cellW, area.x2);
      const int cy2 = std::min(cy1 + cellH, area.y2);
      uint64_t sum = 0;
      for (int yy = cy1; yy < cy2; ++yy) {
        for (int xx = cx1; xx < cx2; ++xx) sum += px(before, xx, yy);
      }
      const uint64_t n = static_cast<uint64_t>(cx2 - cx1) * static_cast<uint64_t>(cy2 - cy1);
      ASSERT_EQ(px(after, x, y), (sum + n / 2) / n) << "plane " << plane << " inside at " << x << "," << y;
    }
  }
}

// All three planes of a planar frame: luma over `luma` in block x block cells, both chroma planes over `chroma`
// in chromaW x chromaH cells.
void expectPlanarGrid(const AVFrame& after, const AVFrame& before, PixelRect luma, int block, PixelRect chroma,
                      int chromaW, int chromaH) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(after.format));
  const int cw = (after.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
  const int ch = (after.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
  expectGrid(after, before, 0, 1, 0, luma, block, block, after.width, after.height);
  expectGrid(after, before, 1, 1, 0, chroma, chromaW, chromaH, cw, ch);
  expectGrid(after, before, 2, 1, 0, chroma, chromaW, chromaH, cw, ch);
}

// One rect with one block on a noise frame, checked with expectPlanarGrid.
void expectPlanarGridMosaic(AVPixelFormat format, int width, int height, PixelRect rect, int block, PixelRect luma,
                            PixelRect chroma, int chromaW, int chromaH, unsigned seed) {
  auto frame = test::makeNoiseFrame(width, height, format, seed);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{rect, block}});
  expectPlanarGrid(*frame, *before, luma, block, chroma, chromaW, chromaH);
}

// Every pixel of every plane of two YUV 4:2:0 frames of the same size is the same.
bool samePixels420(const AVFrame& a, const AVFrame& b) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? a.width : (a.width + 1) / 2;
    const int h = plane == 0 ? a.height : (a.height + 1) / 2;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (a.data[plane][y * a.linesize[plane] + x] != b.data[plane][y * b.linesize[plane] + x]) return false;
      }
    }
  }
  return true;
}

TEST(Mosaic, BlockSizeFollowsTheLongSide) {
  EXPECT_EQ(mosaicBlockSize({0, 0, 40, 30}, MosaicConfig{}), 16);
  EXPECT_EQ(mosaicBlockSize({0, 0, 200, 100}, MosaicConfig{}), 25);
  EXPECT_THROW(mosaicBlockSize({0, 0, 10, 10}, MosaicConfig{0, 8}), std::invalid_argument);
}

TEST(Mosaic, LadderBlockIsTheFirstStepNotBelowTheWantedSize) {
  EXPECT_EQ(ladderBlock(1), 16);
  EXPECT_EQ(ladderBlock(16), 16);
  EXPECT_EQ(ladderBlock(17), 24);
  EXPECT_EQ(ladderBlock(25), 32);
  EXPECT_EQ(ladderBlock(100), 128);
  EXPECT_EQ(ladderBlock(256), 256);
  EXPECT_EQ(ladderBlock(1000), 256);  // the top of the ladder
}

TEST(Mosaic, ATrackMovesUpTheLadderOnlyPastOneAndAHalfTimesItsBlock) {
  EXPECT_EQ(nextTrackBlock(0, 20), 24);   // a new track: the first step >= 20
  EXPECT_EQ(nextTrackBlock(24, 36), 24);  // 36 is 1.5 x 24, not past it
  EXPECT_EQ(nextTrackBlock(24, 37), 48);  // past it: the first step >= 37
  EXPECT_EQ(nextTrackBlock(48, 10), 48);  // never down
  EXPECT_EQ(nextTrackBlock(256, 1000), 256);
}

TEST(Mosaic, TrackBlocksFollowEachTrackAndForgetEndedOnes) {
  TrackBlocks blocks{MosaicConfig{}};
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 100, 50}), 16);  // ceil(100 / 8) = 13, at least 16
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 300, 50}), 48);  // ceil(300 / 8) = 38
  blocks.endFrame();
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 180, 50}), 16);  // 23 is not past 1.5 x 16 = 24
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 100, 50}), 48);  // never down
  blocks.endFrame();
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 200, 50}), 32);  // 25 is past 24
  blocks.endFrame();                                   // track 2 had no mask on this frame: forgotten
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 100, 50}), 16);  // so it starts afresh
  EXPECT_THROW(TrackBlocks(MosaicConfig{0, 8}), std::invalid_argument);
}

TEST(Mosaic, CoveringRectRoundsOutwardAndClips) {
  const PixelRect r = coveringRect(10.2f, 5.7f, 20.1f, 9.0f, 100, 100);
  EXPECT_EQ(r.x1, 10);
  EXPECT_EQ(r.y1, 5);
  EXPECT_EQ(r.x2, 21);
  EXPECT_EQ(r.y2, 9);
  const PixelRect c = coveringRect(-5, -5, 10, 10, 8, 8);
  EXPECT_EQ(c.x1, 0);
  EXPECT_EQ(c.x2, 8);
  EXPECT_TRUE(coveringRect(200, 200, 300, 300, 100, 100).empty());
}

TEST(Mosaic, CoveringRectRoundsAFractionalBottomEdgeUp) {
  const PixelRect r = coveringRect(10.2f, 5.7f, 20.1f, 9.3f, 100, 100);
  EXPECT_EQ(r.x1, 10);
  EXPECT_EQ(r.y1, 5);
  EXPECT_EQ(r.x2, 21);
  EXPECT_EQ(r.y2, 10);
}

TEST(Mosaic, CoveringRectClampsHugeValuesBeforeConverting) {
  const float huge = 1e30f;
  const PixelRect all = coveringRect(-huge, -huge, huge, huge, 100, 80);
  EXPECT_EQ(all.x1, 0);
  EXPECT_EQ(all.y1, 0);
  EXPECT_EQ(all.x2, 100);
  EXPECT_EQ(all.y2, 80);
  EXPECT_TRUE(coveringRect(huge, huge, huge, huge, 100, 80).empty());
  EXPECT_TRUE(coveringRect(-huge, -huge, -huge, -huge, 100, 80).empty());
}

TEST(Mosaic, CoveringRectRefusesNonFiniteCoordinates) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -inf}) {
    EXPECT_THROW(coveringRect(bad, 10, 20, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, bad, 20, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, 10, bad, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, 10, 20, bad, 100, 100), VcError);
  }
}

// {10, 10, 50, 40} with 16 px blocks covers the frame's cells [0, 64) x [0, 48); chroma, 8 px cells, [0, 32) x [0, 24).
TEST(Mosaic, PixelatesOnTheGridOfTheWholeFrame) {
  expectPlanarGridMosaic(AV_PIX_FMT_YUV420P, 96, 64, {10, 10, 50, 40}, 16, {0, 0, 64, 48}, {0, 0, 32, 24}, 8, 8, 1);
}

TEST(Mosaic, PixelatesNv12OnTheSameGrid) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 2);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16}});
  expectGrid(*frame, *before, 0, 1, 0, {0, 0, 64, 48}, 16, 16, 96, 64);
  expectGrid(*frame, *before, 1, 2, 0, {0, 0, 32, 24}, 8, 8, 48, 32);
  expectGrid(*frame, *before, 1, 2, 1, {0, 0, 32, 24}, 8, 8, 48, 32);
}

// The pattern is anchored to the picture, not to the mask (spec 2026-10-19 §4): a mask that moves within the
// same cells changes nothing, and the cells two positions share keep their values.
TEST(Mosaic, AMovingMaskKeepsItsPattern) {
  auto source = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 3);
  auto a = test::deepCopy(*source);
  auto b = test::deepCopy(*source);
  auto c = test::deepCopy(*source);
  applyMosaic(*a, {MosaicRect{{10, 10, 50, 40}, 16}});
  applyMosaic(*b, {MosaicRect{{14, 12, 54, 42}, 16}});  // 4 px right, 2 px down: the same cells
  EXPECT_TRUE(samePixels420(*a, *b));
  applyMosaic(*c, {MosaicRect{{26, 10, 66, 40}, 16}});  // the cells [16, 80) x [0, 48)
  for (int y = 0; y < 48; ++y) {
    for (int x = 16; x < 64; ++x) {
      ASSERT_EQ(a->data[0][y * a->linesize[0] + x], c->data[0][y * c->linesize[0] + x]) << x << "," << y;
    }
  }
}

TEST(Mosaic, PixelatesYuv422pAndYuv444pOnTheGrid) {
  // 4:2:2: chroma half as wide only, cells 8 x 16 over [0, 32) x [0, 48); 4:4:4: chroma as luma.
  expectPlanarGridMosaic(AV_PIX_FMT_YUV422P, 96, 64, {11, 9, 50, 42}, 16, {0, 0, 64, 48}, {0, 0, 32, 48}, 8, 16, 10);
  expectPlanarGridMosaic(AV_PIX_FMT_YUV444P, 96, 64, {11, 9, 50, 42}, 16, {0, 0, 64, 48}, {0, 0, 64, 48}, 16, 16, 11);
}

// The cells the right and bottom edges of a 97 x 63 frame cut average the pixels that are there: {80, 40, 97, 63}
// with 16 px blocks is [80, 97) x [32, 63), cells 16 and 1 px wide, 16 and 15 px high; chroma (49 x 32) is
// [40, 49) x [16, 32), cells 8 and 1 wide.
TEST(Mosaic, CellsCutByTheFrameEdgeAverageWhatIsThere) {
  expectPlanarGridMosaic(AV_PIX_FMT_YUV420P, 97, 63, {80, 40, 97, 63}, 16, {80, 32, 97, 63}, {40, 16, 49, 32}, 8, 8, 9);
}

// Overlapping masks are pixelated largest block first, so the finer mosaic drawn last is what shows where they
// overlap, whatever their order in the list. 16 px cells straddle the 24 px ones, so the order is visible.
TEST(Mosaic, AppliesLargerBlocksFirst) {
  const MosaicRect coarse{{0, 0, 72, 48}, 24};
  const MosaicRect fine{{16, 16, 48, 48}, 16};
  auto together = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 12);
  auto coarseFirst = test::deepCopy(*together);
  auto fineFirst = test::deepCopy(*together);
  applyMosaic(*together, {fine, coarse});
  applyMosaic(*coarseFirst, {coarse});
  const auto afterCoarse = test::deepCopy(*coarseFirst);
  applyMosaic(*coarseFirst, {fine});
  expectPlanarGrid(*coarseFirst, *afterCoarse, {16, 16, 48, 48}, 16, {8, 8, 24, 24}, 8, 8);
  EXPECT_TRUE(samePixels420(*together, *coarseFirst));
  applyMosaic(*fineFirst, {fine});
  applyMosaic(*fineFirst, {coarse});
  EXPECT_FALSE(samePixels420(*together, *fineFirst));
}

TEST(Mosaic, RejectsAnInvalidBlockBeforeTouchingAnyPixel) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 15);
  const auto before = test::deepCopy(*frame);
  for (const int bad : {0, -16, 15, 1}) {
    EXPECT_THROW(applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16}, MosaicRect{{60, 10, 90, 40}, bad}}),
                 std::invalid_argument)
        << bad;
  }
  EXPECT_TRUE(samePixels420(*frame, *before));
}

TEST(Mosaic, RefusesSharedBuffersAndOtherFormats) {
  auto frame = test::makeNoiseFrame(32, 32, AV_PIX_FMT_YUV420P, 4);
  av::FramePtr other = av::allocFrame();
  ASSERT_GE(av_frame_ref(other.get(), frame.get()), 0);
  EXPECT_THROW(applyMosaic(*frame, {MosaicRect{{0, 0, 16, 16}, 16}}), VcError);

  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = 16;
  rgb->height = 16;
  ASSERT_GE(av_frame_get_buffer(rgb.get(), 0), 0);
  EXPECT_THROW(applyMosaic(*rgb, {MosaicRect{{0, 0, 8, 8}, 16}}), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/records_test.cpp`：`MaskLineListsBoxesWithTheirTrackAndOrigin` 里的 `MaskBox{1, 1.0f, 2.0f, 3.5f, 4.0f, MaskOrigin::Interpolated, 7}` 改成 `MaskBox{1, 1.0f, 2.0f, 3.5f, 4.0f, MaskOrigin::Interpolated, 7, 24}`，期望里 `"origin":"interpolated"}` 改成 `"origin":"interpolated","block":24}`；`WritesNegativeZeroAsZero` 的期望里 `"origin":"held"}` 改成 `"origin":"held","block":0}`。

`cpp/tests/redact_job_test.cpp` 的 `MasksEveryFrameAndKeepsTheAudio`：把 "The mask is the fake box grown by 15 % per side…" 那两行注释换成

```cpp
  // The mask is the fake box grown by 15 % per side, (85, 88)-(215, 192); its long side asks for ceil(130 / 8) = 17 px
  // blocks, the ladder's 24, on the frame's grid: the cell [96, 120) x [96, 120) is flat in the output and noisy in
  // the input, on every frame.
```

并在函数末尾加：

```cpp
  // Every mask records the block it was pixelated with; one static target keeps one block.
  std::ifstream masksFile(dir / "masks.jsonl");
  for (std::string line; std::getline(masksFile, line);) {
    for (const auto& box : nlohmann::json::parse(line).at("boxes")) EXPECT_EQ(box.at("block").get<int>(), 24) << line;
  }
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4
```

预期：编译失败，`no type named 'MosaicRect' in namespace 'vc'`、`use of undeclared identifier 'ladderBlock'`、`no member named 'block' in 'vc::MaskBox'`（`MaskBox{…, 7, 24}` 的初始化元素过多）。

- [ ] **Step 3: 实现**

`cpp/src/mask/mosaic.h` 整个换成：

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct AVFrame;

namespace vc {

struct MosaicConfig {
  int minBlockPx = 16;        // no block is asked for below this
  int blocksPerLongSide = 8;  // a mask asks for about this many blocks along its long side
};

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

// Smallest pixel rectangle covering the box, clipped to the frame. Throws VcError when a
// coordinate is not finite (a broken box must stop the run, not become a mask).
PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight);

// The block a mask asks for: max(minBlockPx, ceil(long side / blocksPerLongSide)). Throws
// std::invalid_argument when a setting is below 1.
int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config);

// The block sizes masks are pixelated with (spec 2026-10-19 §4).
inline constexpr std::array<int, 9> kMosaicBlockLadder = {16, 24, 32, 48, 64, 96, 128, 192, 256};

// The smallest ladder step >= `wanted`; the top step for anything larger.
int ladderBlock(int wanted);

// The block of a track's next mask: ladderBlock(wanted) for a new track (`current` 0); afterwards
// `current`, moved up to ladderBlock(wanted) only when `wanted` exceeds 1.5 times `current`. It never
// moves down.
int nextTrackBlock(int current, int wanted);

// The blocks of every track over consecutive frames (spec §4).
class TrackBlocks {
 public:
  // Throws std::invalid_argument when a setting is below 1.
  explicit TrackBlocks(const MosaicConfig& config);
  // The block for `track`'s mask `rect` on the current frame.
  int blockFor(int64_t track, const PixelRect& rect);
  // Ends the current frame: a track without a mask on it is forgotten (BoxTimeline masks a track on
  // consecutive frames until it ends).
  void endFrame();

 private:
  MosaicConfig config_;
  std::unordered_map<int64_t, int> previous_;
  std::unordered_map<int64_t, int> current_;
};

// A rect and the block it is pixelated with.
struct MosaicRect {
  PixelRect rect;
  int block = kMosaicBlockLadder[0];
};

// Pixelates each rect on the grid of its block size b anchored at the frame's top-left corner (spec §4):
// the rect's edges move out to multiples of b (x1 = floor(x1 / b) * b, x2 = ceil(x2 / b) * b), are
// clipped to the frame, and every b x b cell inside becomes the rounded mean of its pixels; a cell the
// frame's right or bottom edge cuts averages the pixels that are there. Chroma uses (b >> log2_chroma_w)
// x (b >> log2_chroma_h) cells on the same grid (b / 2 for 4:2:0). Rects are applied largest block first
// (equal blocks in list order), so where masks overlap the finer mosaic drawn last shows. Supports 8-bit
// planar YUV and NV12. Throws std::invalid_argument for a block that is not an even number >= 2
// (checked first, before any pixel is touched) and VcError for other pixel formats or when the frame's
// buffers are shared (call av_frame_make_writable first).
void applyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects);

}  // namespace vc
```

`cpp/src/mask/mosaic.cpp` 整个换成：

```cpp
#include "mask/mosaic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/vc_error.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

// A track's block moves up the ladder only when its mask asks for more than this many times the block.
constexpr double kBlockUpgradeFactor = 1.5;

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

// Sets every cell of the cellW x cellH grid anchored at the plane's origin inside `area` (plane pixels,
// its edges on the grid or at the plane's edge) to the cell's rounded mean. `step` is the byte distance
// between neighbouring samples, `offset` the first sample's byte (NV12 chroma: 2, 0 / 1).
void pixelateGrid(uint8_t* data, int stride, const PixelRect& area, int cellW, int cellH, int step, int offset) {
  for (int by = area.y1; by < area.y2; by += cellH) {
    const int ey = std::min(by + cellH, area.y2);
    for (int bx = area.x1; bx < area.x2; bx += cellW) {
      const int ex = std::min(bx + cellW, area.x2);
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

void requireValidConfig(const MosaicConfig& config) {
  if (config.minBlockPx < 1 || config.blocksPerLongSide < 1) {
    throw std::invalid_argument("mosaic block settings must be >= 1");
  }
}

// Even, so that the chroma grid of 4:2:0 and 4:2:2 (half the block) lines up with the luma grid.
void requireValidBlock(int block) {
  if (block < 2 || block % 2 != 0) {
    throw std::invalid_argument("mosaic block sizes must be even and >= 2, got " + std::to_string(block));
  }
}

}  // namespace

PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight) {
  if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
    throw VcError("mask box has non-finite coordinates");
  }
  // Clamp while still a floating-point number: converting an out-of-range value to int is undefined.
  const auto edge = [](double v, int size) {
    return static_cast<int>(std::clamp(v, 0.0, static_cast<double>(size)));
  };
  return {edge(std::floor(x1), frameWidth), edge(std::floor(y1), frameHeight), edge(std::ceil(x2), frameWidth),
          edge(std::ceil(y2), frameHeight)};
}

int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config) {
  requireValidConfig(config);
  const int64_t longSide = std::max(rect.x2 - rect.x1, rect.y2 - rect.y1);
  const int64_t perBlock = (longSide + config.blocksPerLongSide - 1) / config.blocksPerLongSide;
  return static_cast<int>(std::max<int64_t>(config.minBlockPx, perBlock));
}

int ladderBlock(int wanted) {
  for (const int step : kMosaicBlockLadder) {
    if (step >= wanted) return step;
  }
  return kMosaicBlockLadder.back();
}

int nextTrackBlock(int current, int wanted) {
  if (current <= 0) return ladderBlock(wanted);
  if (static_cast<double>(wanted) > kBlockUpgradeFactor * current) return std::max(current, ladderBlock(wanted));
  return current;
}

TrackBlocks::TrackBlocks(const MosaicConfig& config) : config_(config) { requireValidConfig(config); }

int TrackBlocks::blockFor(int64_t track, const PixelRect& rect) {
  const auto it = previous_.find(track);
  const int block = nextTrackBlock(it == previous_.end() ? 0 : it->second, mosaicBlockSize(rect, config_));
  current_[track] = block;
  return block;
}

void TrackBlocks::endFrame() {
  previous_ = std::move(current_);
  current_.clear();
}

void applyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects) {
  for (const MosaicRect& r : rects) requireValidBlock(r.block);  // before any pixel is touched
  const auto format = static_cast<AVPixelFormat>(frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const bool nv12 = format == AV_PIX_FMT_NV12;
  const bool planar =
      desc != nullptr && (desc->flags & AV_PIX_FMT_FLAG_PLANAR) != 0 &&
      (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_PAL)) == 0 &&
      desc->nb_components == 3 && desc->comp[0].depth == 8 && desc->comp[1].plane == 1 &&
      desc->comp[2].plane == 2;
  if (!nv12 && !planar) {
    throw VcError(std::string("mosaic: unsupported pixel format ") + (desc ? desc->name : "unknown"));
  }
  if (!av_frame_is_writable(&frame)) {
    throw VcError("mosaic: frame buffers are shared; make the frame writable first");
  }
  const int sw = desc->log2_chroma_w;
  const int sh = desc->log2_chroma_h;
  // Largest blocks first: where masks overlap, the finer mosaic drawn last is what shows.
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const PixelRect r = clipped(m.rect, frame.width, frame.height);
    if (r.empty()) continue;
    const PixelRect area = snapped(r, m.block, frame.width, frame.height);
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
}

}  // namespace vc
```

`cpp/src/track/box_timeline.h` 的 `MaskBox` 在 `track` 之后加：

```cpp
  int block = 0;      // mosaic block size in px; BoxTimeline leaves 0, vc_redact fills it in (spec §4)
```

`cpp/apps/vc_redact/records.cpp` 的 `maskRecord` 里 `b["origin"] = toString(m.origin);` 之后加 `b["block"] = m.block;`；`records.h` 的注释改成 `{cls, x1, y1, x2, y2, track, origin, block}`。

`cpp/apps/vc_redact/redact_job.cpp` 的 `Processor`：

- 构造函数初始化列表在 `timeline_(…)` 之后加 `blocks_(options.config.mosaic)`；成员在 `BoxTimeline timeline_;` 之后加 `TrackBlocks blocks_;`。
- `emit` 换成：

```cpp
  // Writes masks.jsonl, pixelates and sends every buffered frame downstream, in order. A mask's block comes
  // from its track (TrackBlocks), so a target keeps its mosaic from frame to frame (spec 2026-10-19 §4).
  void emit(const std::vector<std::vector<MaskBox>>& masks) {
    if (masks.size() != buffer_.size()) throw VcError("internal error: masks do not match the buffered frames");
    for (size_t k = 0; k < buffer_.size(); ++k) {
      VideoItem& frame = buffer_[k];
      const Clock::time_point started = Clock::now();
      std::vector<MaskBox> boxes = masks[k];
      std::vector<MosaicRect> rects;
      rects.reserve(boxes.size());
      for (MaskBox& m : boxes) {
        const PixelRect rect = coveringRect(m.x1, m.y1, m.x2, m.y2, frame.frame->width, frame.frame->height);
        m.block = blocks_.blockFor(m.track, rect);
        rects.push_back(MosaicRect{rect, m.block});
      }
      blocks_.endFrame();
      if (!boxes.empty()) {
        checkAv(av_frame_make_writable(frame.frame.get()), "make frame writable");
        applyMosaic(*frame.frame, rects);
        ++stats_.maskedFrames;
        stats_.maskBoxes += static_cast<int64_t>(boxes.size());
      }
      stats_.mosaic.busySec += secondsSince(started);
      ++stats_.mosaic.items;
      masks_ << maskRecord(video_, frame.index, boxes, detector_.classes());
      if (!out_.push(Item{std::move(frame)})) throw PipelineAborted{};
    }
    buffer_.clear();
  }
```

`cpp/README.md`：`masks.jsonl` 一条末尾加 `，以及打码用的马赛克块大小 block`；在"补框：一个目标一个框"一条之后加一条：

```
- **马赛克**（设计 §4）：网格固定在画面坐标上，不随打码框移动：打码区域的四边向外扩到块大小的整数倍，每块取块内均值，画面右、下边缘的不完整块取实际像素的均值；色度平面按子采样换算（4:2:0 为块大小的一半）。块大小按轨迹固定：轨迹开始时取档位 {16, 24, 32, 48, 64, 96, 128, 192, 256} 中第一个不小于 max(mosaic_min_block_px, ⌈长边 / mosaic_blocks_per_long_side⌉) 的值，之后只有这个计算值超过当前块大小的 1.5 倍才升档，不降档（超过 256 的一律用 256）。打码框互相重叠时按块大小从大到小打，细的盖在粗的上面。
```

`docs/USAGE.md` §2 "输出"表格里 `masks.jsonl` 一行换成 `| masks.jsonl | 每帧打了哪些框（一帧一行）：坐标、轨迹编号 track、来源 origin、马赛克块大小 block |`；`detections.jsonl` 一行末尾加 `；confirmed 为 false 的是只出现一次的低分框，没有打码`。

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='Mosaic.*:Records.*:BoxTimeline.*:RedactJob.*'
```

预期：全部 PASSED（`Mosaic` 16 个），零警告。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/mask/mosaic.h cpp/src/mask/mosaic.cpp cpp/src/track/box_timeline.h cpp/apps/vc_redact/records.h cpp/apps/vc_redact/records.cpp cpp/apps/vc_redact/redact_job.cpp cpp/tests/mosaic_test.cpp cpp/tests/records_test.cpp cpp/tests/redact_job_test.cpp cpp/README.md docs/USAGE.md
git commit -m "feat: mosaic grid anchored to the frame with a fixed block per track"
```

---

### Task 8: `vc_eval` 的观感指标

本设计 §5 第一条：每次运行都在全部帧上算打码面积、帧间跳变、框数、同轨多框帧，写进 `summary.json`，`summary.md` 加"观感"一节。另加 `vc_eval viewing` 子命令，给没有参照标注的视频（Task 11 的 64 分钟全片）单独算。

**Files:**
- Create: `py/vc_eval/viewing.py`
- Create: `py/tests/test_eval_viewing.py`
- Modify: `py/vc_eval/run.py`（imports、`_evaluate`、`_pooled_entry`、`_sweep_and_report`）
- Modify: `py/vc_eval/report.py`（imports、新增 `_viewing_cells`、`_viewing_lines`，`summary_md`）
- Modify: `py/vc_eval/cli.py`（`viewing` 子命令）
- Modify: `docs/USAGE.md`（§2 "检查结果"）、`py/README.md`（单项子命令、产物、测试命令）
- Test: `py/tests/test_eval_viewing.py`、`py/tests/test_eval_run.py`

**Interfaces:**
- Consumes: `masks.jsonl` 的行格式 `{"video", "frame", "boxes": [{"cls", "x1", "y1", "x2", "y2", "track"?, "origin", "block"?}]}`（Task 4、7）；旧输出没有 `track` 也能读。
- Produces:
  - `RASTER_PX = 4`、`FLICKER_FRACTION = 0.01`。
  - `@dataclass(frozen=True, eq=False) class ViewingReport: video: str; area: np.ndarray; boxes: np.ndarray; flicker_pairs: int; pairs: int; multi_box_frames: int | None`，属性 `frames`。
  - `evaluate_viewing(masks_path: Path, width: int, height: int) -> ViewingReport` —— 要求一行一帧、帧号 0, 1, 2… 连续、只有一个视频；`multi_box_frames` 在没有任何框带 `track` 时为 `None`。
  - `pool_viewing(reports: list[ViewingReport]) -> ViewingReport`（帧拼接，帧对不跨段）。
  - `viewing_json(r) -> dict`：`{"frames", "area": {"mean", "p95", "p99", "max"}, "flicker": {"pairs", "over", "share"}, "boxes": {"median", "p95"}, "multi_box_frames"}`。
  - `summary_line(r) -> str`（CLI 打印）。
  - `summary.json`：每个 `runs[i]["viewing"]` 和每个 `pooled[i]["viewing"]`。
  - `python -m vc_eval viewing --masks <jsonl> --width <w> --height <h> [--json <file>]`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_eval_viewing.py`：

```python
import json

import pytest

from vc_eval.cli import main
from vc_eval.viewing import FLICKER_FRACTION, RASTER_PX, evaluate_viewing, pool_viewing, viewing_json

W, H = 160, 120  # a 40 x 30 raster of 4 px cells: 1200 cells


def _box(x1, y1, x2, y2, track=None, cls="face"):
    b = {"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "origin": "detected"}
    if track is not None:
        b["track"] = track
    return b


def _masks(path, frames, video="a.mp4"):
    with path.open("w") as f:
        for i, boxes in enumerate(frames):
            f.write(json.dumps({"video": video, "frame": i, "boxes": boxes}) + "\n")
    return path


def test_the_raster_is_four_pixels_and_a_jump_is_one_percent():
    assert RASTER_PX == 4 and FLICKER_FRACTION == 0.01


def test_area_is_the_union_of_the_masks_on_the_raster(tmp_path):
    frames = [
        [_box(0, 0, 40, 40)],  # 10 x 10 cells
        [_box(0, 0, 40, 40), _box(20, 0, 60, 40)],  # overlapping: 15 x 10 cells, not 200
        [],
        [_box(1, 1, 3, 3)],  # rounded out to one whole cell
        [_box(-50, -50, 8, 400)],  # clipped to the picture: 2 columns x 30 rows
    ]
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert r.frames == 5 and r.video == "a.mp4"
    assert r.area.tolist() == pytest.approx([100 / 1200, 150 / 1200, 0.0, 1 / 1200, 60 / 1200])
    assert r.boxes.tolist() == [1, 2, 0, 1, 1]


def test_flicker_counts_frame_pairs_whose_masked_regions_differ_on_more_than_one_percent(tmp_path):
    twelve = _box(0, 0, 48, 4)  # 12 cells: exactly 1 % of 1200
    thirteen = _box(0, 0, 52, 4)  # 13 cells
    frames = [[], [twelve], [twelve], [thirteen], [], [thirteen]]
    # pairs: 12 cells differ (not more than 1 %), 0, 1, 13 (over), 13 (over)
    r = evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H)
    assert (r.flicker_pairs, r.pairs) == (2, 5)
    assert viewing_json(r)["flicker"] == {"pairs": 5, "over": 2, "share": 0.4}


def test_area_and_box_statistics(tmp_path):
    frames = [[_box(0, 0, 4, 4)] * k for k in range(100)]  # k boxes on frame k, all on the same cell
    r = viewing_json(evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H))
    assert r["frames"] == 100
    assert r["boxes"] == {"median": 49.5, "p95": pytest.approx(94.05)}
    assert r["area"] == {
        "mean": pytest.approx(0.99 / 1200),
        "p95": pytest.approx(1 / 1200),
        "p99": pytest.approx(1 / 1200),
        "max": pytest.approx(1 / 1200),
    }


def test_frames_where_a_track_has_two_boxes_are_counted(tmp_path):
    frames = [
        [_box(0, 0, 8, 8, 1), _box(20, 0, 28, 8, 2)],
        [_box(0, 0, 8, 8, 1), _box(30, 0, 38, 8, 1)],
        [_box(0, 0, 8, 8, 1)],
    ]
    assert evaluate_viewing(_masks(tmp_path / "m.jsonl", frames), W, H).multi_box_frames == 1
    untracked = [[_box(0, 0, 8, 8)], [_box(0, 0, 8, 8), _box(30, 0, 38, 8)]]  # an older vc_redact: no track ids
    assert evaluate_viewing(_masks(tmp_path / "old.jsonl", untracked), W, H).multi_box_frames is None


@pytest.mark.parametrize(
    "lines, message",
    [
        ([{"video": "a.mp4", "frame": 1, "boxes": []}], "frame 1 where frame 0 was expected"),
        ([{"video": "a.mp4", "frame": 0, "boxes": []}, {"video": "a.mp4", "frame": 2, "boxes": []}], "frame 2 where frame 1"),
        ([{"video": "a.mp4", "frame": 0, "boxes": []}, {"video": "b.mp4", "frame": 1, "boxes": []}], "'b.mp4'"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [{"x1": 0}]}], "line 1: bad mask record"),
        ([{"video": "a.mp4", "frame": 0, "boxes": [{**_box(0, 0, 4, 4), "track": "x"}]}], "line 1: bad mask record"),
        ([], "has no frames"),
    ],
)
def test_refuses_masks_that_are_not_one_ordered_line_per_frame_of_one_video(tmp_path, lines, message):
    path = tmp_path / "m.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match=message):
        evaluate_viewing(path, W, H)


def test_refuses_a_picture_without_pixels(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_viewing(_masks(tmp_path / "m.jsonl", [[]]), 0, H)


def test_pooling_concatenates_the_frames_and_adds_up_the_pairs(tmp_path):
    a = evaluate_viewing(_masks(tmp_path / "a.jsonl", [[_box(0, 0, 40, 40, 1)], []]), W, H)
    b = evaluate_viewing(_masks(tmp_path / "b.jsonl", [[], [], [_box(0, 0, 4, 4)]], video="b.mp4"), W, H)
    pooled = viewing_json(pool_viewing([a, b]))
    assert pooled["frames"] == 5
    assert pooled["flicker"] == {"pairs": 3, "over": 1, "share": pytest.approx(1 / 3)}  # no pair across a and b
    assert pooled["area"]["max"] == pytest.approx(100 / 1200)
    assert pooled["multi_box_frames"] == 0  # a has track ids and never repeats one; b has none


def test_cli_viewing_prints_the_metrics_and_writes_json(tmp_path, capsys):
    masks = _masks(tmp_path / "m.jsonl", [[_box(0, 0, 40, 40, 1)], [_box(0, 0, 40, 40, 1)]])
    args = ["viewing", "--masks", str(masks), "--width", str(W), "--height", str(H), "--json", str(tmp_path / "v.json")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("viewing a.mp4: area mean 8.33%, p95 8.33%, p99 8.33%, max 8.33%;")
    assert "flicker 0.00% (0/1 pairs)" in out and "boxes median 1, p95 1" in out and "same-track multi-box frames 0" in out
    assert json.loads((tmp_path / "v.json").read_text())["area"]["max"] == pytest.approx(100 / 1200)
    assert main(["viewing", "--masks", str(tmp_path / "missing.jsonl"), "--width", "160", "--height", "120"]) == 2
    assert "vc_eval:" in capsys.readouterr().err
```

`py/tests/test_eval_run.py` 在 `test_summary_md_details_the_smallest_rate_when_none_passes` 之后加：

```python
def test_the_viewing_metrics_of_every_run_go_into_the_summary(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    viewing = summary["runs"][0]["viewing"]
    # MASK and PLATE_MASK on every frame of the 160 x 120 picture: 99 + 144 of the 1200 raster cells
    assert viewing["area"] == {
        "mean": pytest.approx(0.2025), "p95": pytest.approx(0.2025), "p99": pytest.approx(0.2025), "max": pytest.approx(0.2025),
    }
    assert viewing["flicker"] == {"pairs": N - 1, "over": 0, "share": 0.0}
    assert viewing["boxes"] == {"median": 2.0, "p95": 2.0}
    assert viewing["multi_box_frames"] is None  # the fake masks carry no track ids
    assert summary["pooled"][0]["viewing"]["frames"] == N
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    section = md.split("## 观感")[1].split("\n## ")[0]
    assert "| seg | 5 | 20.25% | 20.25% | 20.25% | 20.25% | 0.0%（0/29） | 2 | 2 | — |" in section
    assert "| 合计 | 10 | 20.25% |" in section
    assert "1/4" in section and "1%" in section
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_viewing.py -p no:cacheprovider -q
"${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_run.py -p no:cacheprovider -q -k viewing_metrics
```

预期：第一条收集失败（`ModuleNotFoundError: No module named 'vc_eval.viewing'`）；第二条 `test_the_viewing_metrics_of_every_run_go_into_the_summary` FAIL（`KeyError: 'viewing'`）。

- [ ] **Step 3: 实现**

`py/vc_eval/viewing.py`：

```python
"""Viewing metrics of a redaction (spec 2026-10-19 §5): how much of the picture the masks cover, how often the masked
region jumps between frames, how many boxes a frame carries, and whether a track ever has two boxes on one frame.

Every frame is measured, on a raster of RASTER_PX x RASTER_PX pixel cells: a box covers every cell it touches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RASTER_PX = 4  # cells of 4 x 4 px: a quarter of the resolution along each axis
FLICKER_FRACTION = 0.01  # a frame pair jumps when the masked regions differ on more than 1 % of the picture


@dataclass(frozen=True, eq=False)
class ViewingReport:
    video: str
    area: np.ndarray  # per frame: share of the picture under the union of its masks
    boxes: np.ndarray  # per frame: number of mask boxes
    flicker_pairs: int  # consecutive frame pairs whose masked regions differ on more than FLICKER_FRACTION
    pairs: int  # consecutive frame pairs
    multi_box_frames: int | None  # frames on which one track has more than one box; None when no box names a track

    @property
    def frames(self) -> int:
        return len(self.area)


def _cells(x1: float, y1: float, x2: float, y2: float, cols: int, rows: int) -> tuple[slice, slice]:
    """Raster rows and columns a box touches, clipped to the picture (empty for a box outside it)."""
    c0 = min(cols, max(0, math.floor(x1 / RASTER_PX)))
    c1 = max(c0, min(cols, math.ceil(x2 / RASTER_PX)))
    r0 = min(rows, max(0, math.floor(y1 / RASTER_PX)))
    r1 = max(r0, min(rows, math.ceil(y2 / RASTER_PX)))
    return slice(r0, r1), slice(c0, c1)


def evaluate_viewing(masks_path: Path, width: int, height: int) -> ViewingReport:
    """The viewing metrics of a masks.jsonl of a width x height video.

    Raises ValueError for a bad line, frames that are not 0, 1, 2, ... one line each in order, lines of several videos,
    a file without frames or a picture without pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"the picture size must be positive, got {width}x{height}")
    cols, rows = math.ceil(width / RASTER_PX), math.ceil(height / RASTER_PX)
    cells = cols * rows
    areas: list[float] = []
    counts: list[int] = []
    flicker = 0
    multi = 0
    tracked = False
    video: str | None = None
    previous: np.ndarray | None = None
    with masks_path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                name, frame = rec["video"], int(rec["frame"])
                coords = [(float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])) for b in rec["boxes"]]
                tracks = [int(b["track"]) for b in rec["boxes"] if "track" in b]
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{masks_path} line {n}: bad mask record ({e})") from None
            if frame != len(areas):
                raise ValueError(
                    f"{masks_path} line {n}: frame {frame} where frame {len(areas)} was expected (one line per frame, in order)"
                )
            if video is None:
                video = name
            elif name != video:
                raise ValueError(f"{masks_path} line {n}: video {name!r}, but the lines before are for {video!r}")
            grid = np.zeros((rows, cols), dtype=bool)
            for x1, y1, x2, y2 in coords:
                grid[_cells(x1, y1, x2, y2, cols, rows)] = True
            areas.append(np.count_nonzero(grid) / cells)
            counts.append(len(coords))
            tracked = tracked or bool(tracks)
            multi += len(tracks) != len(set(tracks))
            if previous is not None and np.count_nonzero(grid ^ previous) > FLICKER_FRACTION * cells:
                flicker += 1
            previous = grid
    if video is None:
        raise ValueError(f"{masks_path} has no frames")
    return ViewingReport(video, np.array(areas), np.array(counts), flicker, len(areas) - 1, multi if tracked else None)


def pool_viewing(reports: list[ViewingReport]) -> ViewingReport:
    """All frames of all reports together; a frame pair never spans two reports."""
    if not reports:
        raise ValueError("no viewing reports to pool")
    multis = [r.multi_box_frames for r in reports if r.multi_box_frames is not None]
    return ViewingReport(
        ",".join(r.video for r in reports),
        np.concatenate([r.area for r in reports]),
        np.concatenate([r.boxes for r in reports]),
        sum(r.flicker_pairs for r in reports),
        sum(r.pairs for r in reports),
        sum(multis) if multis else None,
    )


def viewing_json(r: ViewingReport) -> dict:
    return {
        "frames": r.frames,
        "area": {
            "mean": float(np.mean(r.area)),
            "p95": float(np.percentile(r.area, 95)),
            "p99": float(np.percentile(r.area, 99)),
            "max": float(np.max(r.area)),
        },
        "flicker": {"pairs": r.pairs, "over": r.flicker_pairs, "share": None if r.pairs == 0 else r.flicker_pairs / r.pairs},
        "boxes": {"median": float(np.median(r.boxes)), "p95": float(np.percentile(r.boxes, 95))},
        "multi_box_frames": r.multi_box_frames,
    }


def summary_line(r: ViewingReport) -> str:
    j = viewing_json(r)
    a, fl, b = j["area"], j["flicker"], j["boxes"]
    flicker = "n/a (one frame)" if fl["share"] is None else f"{fl['share']:.2%} ({fl['over']}/{fl['pairs']} pairs)"
    multi = "n/a (no track ids)" if j["multi_box_frames"] is None else str(j["multi_box_frames"])
    return (
        f"viewing {r.video}: area mean {a['mean']:.2%}, p95 {a['p95']:.2%}, p99 {a['p99']:.2%}, max {a['max']:.2%}; "
        f"flicker {flicker}; boxes median {b['median']:g}, p95 {b['p95']:g}; same-track multi-box frames {multi}"
    )
```

`py/vc_eval/cli.py`：

- `_parser()` 里 `integrity` 子命令之后加：

```python
    v = sub.add_parser("viewing", help="masked area, flicker and boxes per frame of a masks.jsonl (every frame)")
    v.add_argument("--masks", type=Path, required=True)
    v.add_argument("--width", type=int, required=True, help="frame width of the video in px")
    v.add_argument("--height", type=int, required=True, help="frame height of the video in px")
    v.add_argument("--json", type=Path)
```

- `_integrity` 之后加：

```python
def _viewing(args) -> int:
    from vc_eval.viewing import evaluate_viewing, summary_line, viewing_json

    r = evaluate_viewing(args.masks, args.width, args.height)
    print(summary_line(r))
    _write_json(args.json, viewing_json(r))
    return 0
```

- `main` 里的分派表加 `"viewing": _viewing`；模块 docstring 改成 `"""python -m vc_eval coverage | temporal | integrity | viewing | run (proxy-validation spec §6, spec 2026-10-19 §5)."""`。

`py/vc_eval/run.py`：

- imports 加 `from vc_eval.viewing import ViewingReport, evaluate_viewing, pool_viewing, viewing_json`。
- `_evaluate` 的返回类型改成 `tuple[dict, CoverageReport, TemporalReport, ViewingReport]`；`integrity = …` 之后加 `viewing = evaluate_viewing(run_dir / "masks.jsonl", coverage.width, coverage.height)`；`entry` 里 `"stray": …` 之后加 `"viewing": viewing_json(viewing),`；`return entry, coverage, temporal, viewing`。
- `_pooled_entry` 的签名加一个参数 `viewings: dict[tuple[str, float], ViewingReport]`（放在 `reports` 之后），返回的字典里 `"stray": …` 之后加 `"viewing": viewing_json(pool_viewing([viewings[(stem, fps)] for stem in stems])),`。
- `_sweep_and_report`：`reports: … = {}` 之后加 `viewings: dict[tuple[str, float], ViewingReport] = {}`；循环里 `entry, coverage, temporal = _evaluate(cfg, stem, fps)` 改成 `entry, coverage, temporal, viewing = _evaluate(cfg, stem, fps)`，并在 `reports[(stem, fps)] = (coverage, temporal)` 之后加 `viewings[(stem, fps)] = viewing`；`pooled = [_pooled_entry(cfg.stems, fps, runs, reports, viewings) for fps in sorted(cfg.fps_values)]`。

`py/vc_eval/report.py`：

- imports 加 `from vc_eval.viewing import FLICKER_FRACTION, RASTER_PX`。
- `_integrity_lines` 之后加：

```python
def _viewing_cells(v: dict) -> str:
    a, fl, b = v["area"], v["flicker"], v["boxes"]
    flicker = DASH if fl["share"] is None else f"{fl['share']:.1%}（{fl['over']}/{fl['pairs']}）"
    multi = DASH if v["multi_box_frames"] is None else str(v["multi_box_frames"])
    return f"{a['mean']:.2%} | {a['p95']:.2%} | {a['p99']:.2%} | {a['max']:.2%} | {flicker} | {b['median']:g} | {b['p95']:g} | {multi}"


def _viewing_lines(summary: dict) -> list[str]:
    runs = summary["runs"]
    lines = [
        "",
        "## 观感（全部帧，只报告）",
        "",
        f"按 1/{RASTER_PX} 分辨率栅格逐帧计算：打码面积是这一帧所有打码框的并集占画面的比例；"
        f"帧间跳变是相邻两帧打码区域的异或超过画面 {FLICKER_FRACTION:.0%} 的帧对占比；"
        "同轨多框帧是同一条轨迹在同一帧有不止一个框的帧数，应为 0（没有轨迹编号的旧输出记为 —）。合计行把各段的帧放在一起算。",
        "",
        "| 段 | fps | 面积均值 | 面积 95 分位 | 面积 99 分位 | 面积最大 | 帧间跳变 | 每帧框数中位数 | 每帧框数 95 分位 | 同轨多框帧 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fps in sorted({r["fps"] for r in runs}):
        lines += [f"| {r['stem']} | {fps:g} | {_viewing_cells(r['viewing'])} |" for r in runs if r["fps"] == fps]
        lines.append(f"| 合计 | {fps:g} | {_viewing_cells(_pooled_at(summary, fps)['viewing'])} |")
    return lines
```

- `summary_md` 的列表里 `*_integrity_lines(runs),` 之后加 `*_viewing_lines(summary),`。

`docs/USAGE.md` §2 "检查结果"在第 3 条之后加：

```
4. **观感**（需要 Python 环境）：每帧打码面积、帧间跳变、每帧框数、同一轨迹在同一帧有没有多个框。
   ```bash
   PYTHONPATH=py .venv/bin/python -m vc_eval viewing --masks <输出>/masks.jsonl --width 1920 --height 1080
   ```
```

`py/README.md` 的 vc_eval 一节：

- "产物"一条里的“误打面积、速度”改成“误打面积、观感（全部帧、1/4 分辨率栅格：每帧打码面积的均值 / 95 分位 / 99 分位 / 最大、帧间跳变占比、每帧框数中位数 / 95 分位、同轨多框帧数；各段一行，合计行把各段的帧放在一起算；`summary.json` 的 `runs[].viewing` 和 `pooled[].viewing`）、速度”。
- "单项子命令"里 `integrity` 的命令块之后加一个命令块：

  ```bash
  PYTHONPATH=py .venv/bin/python -m vc_eval viewing --masks data/redacted/fps10/<stem>/masks.jsonl --width 1920 --height 1080 [--json out.json]
  ```

  并在其后的说明段末尾加：“`viewing` 在全部帧上算观感指标，不需要参照标注（没有标注的整段视频也能算），`masks.jsonl` 必须一帧一行、帧号从 0 连续；退出码 0，输入有问题时 2。”
- "vc_eval 的测试"的命令里加上 `py/tests/test_eval_viewing.py`。

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_viewing.py tests/test_eval_run.py -p no:cacheprovider -q
```

预期：全部通过。

- [ ] **Step 5: 提交**

```bash
git add py/vc_eval/viewing.py py/vc_eval/run.py py/vc_eval/report.py py/vc_eval/cli.py py/tests/test_eval_viewing.py py/tests/test_eval_run.py docs/USAGE.md py/README.md
git commit -m "feat: viewing metrics in vc_eval: masked area, flicker, boxes per frame"
```

---

### Task 9: 参照误检排除表（`excluded_refs.csv`）

本设计 §5 第二条：若 `data/labels/<段>/excluded_refs.csv` 存在，与其中某一行同帧、同类、IoU ≥ 0.5 的参照框不计入召回；`summary.md` 写明每段排除了多少个、原因是什么。另外在 `summary.json` 的每次运行里列出所有没遮住的高置信参照框（`miss_list`），Task 11 用它核对要写的排除行。

**Files:**
- Modify: `py/vc_eval/records.py`（`EXCLUDED_REFS`、`ExcludedRef`、`read_excluded`）
- Modify: `py/vc_eval/coverage.py`（`EXCLUDE_IOU`、`Exclusion`、`LabelInputs.excluded`、`CoverageReport.excluded` / `unmatched_exclusions`、`evaluate_coverage`、`pool`）
- Modify: `py/vc_eval/run.py`（`coverage_json`、`_evaluate`）
- Modify: `py/vc_eval/report.py`（`_exclusion_lines`、`summary_md`）
- Modify: `py/README.md`（vc_eval 一节）
- Test: `py/tests/test_eval_coverage.py`、`py/tests/test_eval_run.py`

**Interfaces:**
- Consumes: `vc_label.boxes.iou`；`load_label_inputs` 的检查顺序（排除表在 `flags.csv` 之后读，坏的排除表在第一次 `vc_redact` 之前就报错）。
- Produces:
  - `EXCLUDED_REFS = "excluded_refs.csv"`、`EXCLUDED_COLUMNS = ("frame", "cls", "x1", "y1", "x2", "y2", "reason")`。
  - `@dataclass(frozen=True) class ExcludedRef: frame: int; box: Box; reason: str; line: int`。
  - `read_excluded(labels_dir: Path) -> list[ExcludedRef]` —— 没有文件返回 `[]`；缺列、坏数、类别不是 face / plate、空框、空原因都抛 `ValueError`（消息带文件和行号）。
  - `@dataclass(frozen=True) class Exclusion: frame: int; box: Box; reason: str`（被排除的参照框）。
  - `CoverageReport.excluded: list[Exclusion]`、`CoverageReport.unmatched_exclusions: list[ExcludedRef]`（一个参照框都没配上的行）。被排除的框不进任何比率（高置信、低置信、尽力、分档、热力图、时间补洞），也不再当作参照参与误打面积的计算。
  - `coverage_json(c)["excluded"] == {"count": int, "reasons": {原因: 个数}, "unmatched_rows": int}`。
  - `summary.json` 的 `runs[i]["miss_list"]`：`[{"frame", "cls", "x1", "y1", "x2", "y2", "covered"}]`（高置信漏遮，`covered` 保留 3 位小数）。
  - `summary.md` 的"排除的参照误检"一节，每段一行：`- <段>：排除 N 个（原因 n，…）`，有没配上的行时追加 `；另有 M 行没有配上任何参照框，请检查帧号、类别和坐标`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_eval_coverage.py`：imports 改成

```python
from vc_eval.coverage import CoverageReport, _empty_class, evaluate_coverage, pool
from vc_eval.metrics import Rate
from vc_eval.records import detections_by_frame, read_excluded, read_masks, read_uncertain
```

在 `_masks` 之后加：

```python
# A parked bus's tail light the teachers took for a plate: 60 x 20 px, must-mask at 1080 p.
LIGHT = Box("plate", 1800, 1000, 1860, 1020, src="teacher_owlv2")


def _excluded(labels, rows):
    with (labels / "excluded_refs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cls", "x1", "y1", "x2", "y2", "reason"])
        w.writerows(rows)
```

文件末尾加：

```python
def test_excluded_refs_leave_matching_targets_out_of_every_rate(tmp_path):
    frames = _frames(tmp_path, "a", [0, 10])
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    labels = _labels(tmp_path, "a", {0: [face, LIGHT], 10: [face, LIGHT]})
    _excluded(labels, [[0, "plate", 1802, 1001, 1862, 1021, "公交尾灯"]])  # IoU 0.85 with the light, frame 0 only
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {0: [face], 10: [face]}, [0, 10])
    r = evaluate_coverage(frames, labels, masks)
    assert (r.classes["plate"].rate.hits, r.classes["plate"].rate.total) == (0, 1)  # frame 10's light still counts
    assert sum(rate.total for rate in r.classes["plate"].bins.values()) == 1
    assert [(e.frame, e.box.cls, e.box.x1, e.reason) for e in r.excluded] == [(0, "plate", 1800, "公交尾灯")]
    assert r.unmatched_exclusions == []
    assert [m.frame for m in r.misses] == [10]


def test_an_iou_of_exactly_one_half_excludes(tmp_path):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [LIGHT]})
    _excluded(labels, [[0, "plate", 1800, 1000, 1860, 1040, "x"]])  # 1200 / 2400 px^2
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert r.classes["plate"].rate.total == 0 and len(r.excluded) == 1


@pytest.mark.parametrize(
    "row",
    [
        [0, "plate", 1830, 1000, 1890, 1020, "x"],  # IoU 1/3
        [0, "face", 1800, 1000, 1860, 1020, "x"],  # another class
        [5, "plate", 1800, 1000, 1860, 1020, "x"],  # a frame without labels
    ],
)
def test_a_row_that_matches_no_reference_box_excludes_nothing_and_is_reported(tmp_path, row):
    frames = _frames(tmp_path, "a", [0])
    labels = _labels(tmp_path, "a", {0: [LIGHT]})
    _excluded(labels, [row])
    r = evaluate_coverage(frames, labels, _masks(tmp_path / "m.jsonl", "a.mp4", {}, [0]))
    assert r.classes["plate"].rate.total == 1 and r.excluded == []
    assert [(e.frame, e.box.cls, e.line) for e in r.unmatched_exclusions] == [(row[0], row[1], 2)]


def test_pooled_exclusions_are_concatenated(tmp_path):
    for stem in ("a", "b"):
        frames = _frames(tmp_path, stem, [0])
        labels = _labels(tmp_path, stem, {0: [LIGHT]})
        _excluded(labels, [[0, "plate", 1800, 1000, 1860, 1020, "公交尾灯"], [3, "plate", 1, 1, 2, 2, "x"]])
    reports = [
        evaluate_coverage(tmp_path / "frames" / s, tmp_path / "labels" / s, _masks(tmp_path / f"{s}.jsonl", f"{s}.mp4", {}, [0]))
        for s in ("a", "b")
    ]
    pooled = pool(reports)
    assert len(pooled.excluded) == 2 and len(pooled.unmatched_exclusions) == 2


def test_read_excluded_without_a_file_excludes_nothing(tmp_path):
    assert read_excluded(tmp_path) == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("frame,cls,x1,y1,x2,y2\n0,plate,1,1,2,2\n", "no reason column"),
        ("frame,cls,x1,y1,x2,y2,reason\nzero,plate,1,1,2,2,x\n", "line 2: bad row"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,sign,1,1,2,2,x\n", "line 2: class 'sign'"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,x\n0,plate,5,5,5,9,y\n", "line 3: the box is empty"),
        ("frame,cls,x1,y1,x2,y2,reason\n0,plate,1,1,2,2,  \n", "line 2: no reason"),
    ],
)
def test_read_excluded_names_the_line_of_a_bad_row(tmp_path, text, message):
    (tmp_path / "excluded_refs.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_excluded(tmp_path)
```

`py/tests/test_eval_run.py`：

- `_break_labels` 的 `elif kind == "labels for other frames":` 分支之后加：

```python
    elif kind == "bad excluded refs":
        (labels / "excluded_refs.csv").write_text("frame,cls,x1,y1,x2,y2,reason\n0,sign,1,1,2,2,x\n", encoding="utf-8")
```

  并在 `test_bad_label_inputs_of_any_segment_are_refused_before_any_run` 的参数表末尾加 `("bad excluded refs", r"excluded_refs\.csv line 2: class 'sign'"),`。
- 文件末尾加：

```python
def test_excluded_refs_are_left_out_of_p1_and_listed_per_segment_in_the_summary(tmp_path, monkeypatch):
    light = Box("plate", 10, 90, 40, 110, src="teacher_owlv2")  # away from MASK and PLATE_MASK: never masked
    _setup(tmp_path, teachers=(*TEACHERS, light))
    with (tmp_path / "labels" / STEM / "excluded_refs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cls", "x1", "y1", "x2", "y2", "reason"])
        w.writerows([[fr, "plate", 10, 90, 40, 110, "停着的公交尾灯"] for fr in range(0, N, 6)])
        w.writerow([3, "plate", 10, 90, 40, 110, "停着的公交尾灯"])  # frame 3 is not labelled: matches nothing
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    coverage = summary["runs"][0]["coverage"]
    assert (coverage["plate"]["hits"], coverage["plate"]["total"]) == (5, 5)  # with the light it would be 5 / 10
    assert coverage["excluded"] == {"count": 5, "reasons": {"停着的公交尾灯": 5}, "unmatched_rows": 1}
    assert summary["chosen_fps"] == 5
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    section = md.split("## 排除的参照误检")[1].split("\n## ")[0]
    assert "- seg：排除 5 个（停着的公交尾灯 5）；另有 1 行没有配上任何参照框" in section


def test_summary_json_lists_every_confident_miss_with_its_box(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([], masks=(MASK,)))  # the plate is never masked
    misses = summary["runs"][0]["miss_list"]
    assert [m["frame"] for m in misses] == [0, 6, 12, 18, 24]
    assert misses[0] == {"frame": 0, "cls": "plate", "x1": 100.0, "y1": 80.0, "x2": 150.0, "y2": 100.0, "covered": 0.0}
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py -p no:cacheprovider -q
```

预期：收集失败，`ImportError: cannot import name 'read_excluded' from 'vc_eval.records'`。

- [ ] **Step 3: 实现**

`py/vc_eval/records.py`：

- imports 加 `from dataclasses import dataclass`，`from vc_label.contracts import Box, Detection` 改成 `from vc_label.contracts import LABEL_CLASSES, Box, Detection`。
- `FLAG_COLUMNS = …` 之后加：

```python
# Teacher false positives a person checked (spec 2026-10-19 §5): reference boxes left out of the coverage.
EXCLUDED_REFS = "excluded_refs.csv"
EXCLUDED_COLUMNS = ("frame", "cls", "x1", "y1", "x2", "y2", "reason")


@dataclass(frozen=True)
class ExcludedRef:
    """A row of labels_dir/excluded_refs.csv: a reference box found to be a teacher false positive, and why."""

    frame: int
    box: Box
    reason: str
    line: int  # the row's line in the file, for the report
```

- 文件末尾加：

```python
def read_excluded(labels_dir: Path) -> list[ExcludedRef]:
    """The rows of labels_dir/excluded_refs.csv; none when there is no such file.

    Raises ValueError for a header without the columns, a bad number, a class that is not a label class, an empty box
    or an empty reason; the message names the file and the line.
    """
    path = labels_dir / EXCLUDED_REFS
    if not path.exists():
        return []
    rows: list[ExcludedRef] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in EXCLUDED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no {', '.join(missing)} column")
        for row in reader:
            where = f"{path} line {reader.line_num}"
            try:
                frame = int(row["frame"])
                box = Box(row["cls"], float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{where}: bad row ({e})") from None
            if box.cls not in LABEL_CLASSES:
                raise ValueError(f"{where}: class {box.cls!r} is not one of {', '.join(LABEL_CLASSES)}")
            if box.area <= 0:
                raise ValueError(f"{where}: the box is empty")
            reason = (row["reason"] or "").strip()
            if not reason:
                raise ValueError(f"{where}: no reason given")
            rows.append(ExcludedRef(frame, box, reason, reader.line_num))
    return rows
```

`py/vc_eval/coverage.py`：

- 模块 docstring 末尾加一段：

```
Reference boxes listed in data/labels/<stem>/excluded_refs.csv (teacher false positives a person checked; same frame,
same class, IoU >= 0.5 with a row) are left out altogether: they count in no rate and are no reference for the stray
area (spec 2026-10-19 §5).
```

- imports 加 `from vc_label.boxes import iou`；`from vc_eval.records import …` 改成 `from vc_eval.records import ExcludedRef, detections_by_frame, fused_file, read_excluded, read_masks, read_uncertain`。
- `UNCERTAIN_TOL_PX = …` 之后加 `EXCLUDE_IOU = 0.5  # a reference box is excluded by a row of its frame and class it overlaps this much`。
- `Miss` 之后加：

```python
@dataclass(frozen=True)
class Exclusion:
    """A reference box an excluded_refs.csv row left out of the coverage."""

    frame: int
    box: Box
    reason: str
```

- `CoverageReport` 在 `stray_p95: float = 0.0` 之后加：

```python
    excluded: list[Exclusion] = field(default_factory=list)  # reference boxes excluded_refs.csv left out
    unmatched_exclusions: list[ExcludedRef] = field(default_factory=list)  # rows that matched no reference box
```

- `LabelInputs` 末尾加字段 `excluded: list[ExcludedRef]`，docstring 末尾加 "…, and the rows of excluded_refs.csv"；`load_label_inputs` 在 `uncertain = read_uncertain(labels_dir)` 之后加 `excluded = read_excluded(labels_dir)`，返回 `LabelInputs(used, video, width, height, targets, uncertain, excluded)`，docstring 的 Raises 列表里加 "a bad excluded_refs.csv"。
- `_is_uncertain` 之后加：

```python
def _excluding_row(target: Box, rows: list[ExcludedRef]) -> ExcludedRef | None:
    """The first row of the target's class whose box overlaps it with IoU >= EXCLUDE_IOU."""
    return next((r for r in rows if r.box.cls == target.cls and iou(r.box, target) >= EXCLUDE_IOU), None)
```

- `evaluate_coverage`：`stray = []` 之后加

```python
    excluded_by_frame: dict[int, list[ExcludedRef]] = {}
    for row in labels.excluded:
        excluded_by_frame.setdefault(row.frame, []).append(row)
    matched_rows: set[int] = set()
```

  把 `frame_targets = [b for b in targets.get(ref.frame, []) if b.cls in LABEL_CLASSES]` 换成

```python
        frame_targets = []
        for t in (b for b in targets.get(ref.frame, []) if b.cls in LABEL_CLASSES):
            row = _excluding_row(t, excluded_by_frame.get(ref.frame, []))
            if row is None:
                frame_targets.append(t)
            else:
                matched_rows.add(row.line)
                report.excluded.append(Exclusion(ref.frame, t, row.reason))
```

  并在 `report.stray_max = …` 之前加 `report.unmatched_exclusions = [row for row in labels.excluded if row.line not in matched_rows]`。
- `pool` 的 docstring 里 "Misses, confident and uncertain, are concatenated" 改成 "Misses (confident and uncertain) and exclusions (matched and unmatched rows) are concatenated"；返回的 `CoverageReport(…)` 里加

```python
        excluded=[e for r in reports for e in r.excluded],
        unmatched_exclusions=[e for r in reports for e in r.unmatched_exclusions],
```

`py/vc_eval/run.py`：

- imports 加 `from collections import Counter`。
- `coverage_json` 在 `out["passed"] = c.passed()` 之前加

```python
    # reference boxes excluded_refs.csv left out (teacher false positives a person checked), and rows that matched nothing
    out["excluded"] = {
        "count": len(c.excluded),
        "reasons": dict(Counter(e.reason for e in c.excluded)),
        "unmatched_rows": len(c.unmatched_exclusions),
    }
```

- `_evaluate` 的 `entry` 在 `"misses": …` 之后加

```python
        # every confident miss with its reference box: what excluded_refs.csv rows are written from
        "miss_list": [
            {"frame": m.frame, "cls": m.box.cls, "x1": m.box.x1, "y1": m.box.y1, "x2": m.box.x2, "y2": m.box.y2,
             "covered": round(m.covered, 3)}
            for m in coverage.misses
        ],
```

`py/vc_eval/report.py`：`_tier_lines` 之后加

```python
def _exclusion_lines(summary: dict) -> list[str]:
    """Per segment, how many reference boxes excluded_refs.csv left out and why (the same at every rate)."""
    runs = summary["runs"]
    fps = runs[0]["fps"] if runs else None
    lines = [
        "",
        "## 排除的参照误检",
        "",
        "`data/labels/<段>/excluded_refs.csv` 列出人工核查过的老师误检：与其中一行同帧、同类、IoU ≥ 0.5 的参照框不计入任何覆盖率，"
        "也不再当作参照，打在它们上面的码计入误打面积。",
        "",
    ]
    for r in (r for r in runs if r["fps"] == fps):
        e = r["coverage"]["excluded"]
        reasons = "，".join(f"{reason} {n}" for reason, n in e["reasons"].items()) or "无"
        line = f"- {r['stem']}：排除 {e['count']} 个（{reasons}）"
        if e["unmatched_rows"]:
            line += f"；另有 {e['unmatched_rows']} 行没有配上任何参照框，请检查帧号、类别和坐标"
        lines.append(line)
    return lines
```

  `summary_md` 的列表里 `*_tier_lines(summary),` 之后加 `*_exclusion_lines(summary),`。

`py/README.md` vc_eval 一节在 `--fps` 那条之后加一条：

```
- 参照误检排除：`data/labels/<stem>/excluded_refs.csv`（列 `frame,cls,x1,y1,x2,y2,reason`，人工核查后填写，不进 git）里的每一行，把同帧、同类、IoU ≥ 0.5 的参照框从所有覆盖率里去掉，这些框也不再当作参照（打在上面的码计入误打面积）。`summary.md` 的“排除的参照误检”一节按段写排除了多少个、各原因几个，没配上任何参照框的行单独提醒；文件格式不对（缺列、坏数、类别不是 face / plate、空框、空原因）开跑前就报错。`summary.json` 的 `runs[].miss_list` 列出每个没遮住的高置信参照框（帧号、类别、坐标、覆盖比例），核查后要排除的照抄坐标进这个文件。
```

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py tests/test_eval_run.py -p no:cacheprovider -q
```

预期：全部通过。

- [ ] **Step 5: 提交**

```bash
git add py/vc_eval/records.py py/vc_eval/coverage.py py/vc_eval/run.py py/vc_eval/report.py py/tests/test_eval_coverage.py py/tests/test_eval_run.py py/README.md
git commit -m "feat: leave audited teacher false positives out of the coverage via excluded_refs.csv"
```

---

### Task 10: 可配置的召回目标（`--recall-target`）

本设计 §5 第三条：`vc_eval run --recall-target face=0.95,plate=0.98`，默认仍是 0.99。`vc_eval coverage` 也接受同一参数（Task 11 校准时直接用它）。

**Files:**
- Modify: `py/vc_eval/coverage.py`（`default_recall_targets`、`parse_recall_targets`、`CoverageReport.recall_targets`、`class_passed`、`evaluate_coverage`、`pool`）
- Modify: `py/vc_eval/run.py`（`RunConfig.recall_targets`、`_validate`、`_evaluate`、`coverage_json`）
- Modify: `py/vc_eval/report.py`（召回目标行、样本量提醒用各类目标）
- Modify: `py/vc_eval/cli.py`（`run`、`coverage` 的 `--recall-target`）
- Modify: `py/README.md`（vc_eval 一节）
- Test: `py/tests/test_eval_coverage.py`、`py/tests/test_eval_run.py`

**Interfaces:**
- Consumes: `CLASS_PASS = 0.99`（`metrics.py`，不变）。
- Produces:
  - `default_recall_targets() -> dict[str, float]`（每类 `CLASS_PASS`）。
  - `parse_recall_targets(text: str) -> dict[str, float]`：`"face=0.95,plate=0.98"` → `{"face": 0.95, "plate": 0.98}`；没给的类别取 `CLASS_PASS`；未知类别、缺 `=`、重复、非数字、不在 (0, 1] 都抛 `ValueError`（消息以 `--recall-target` 开头）。
  - `CoverageReport.recall_targets: dict[str, float]`；`class_passed(cls)` 用 `recall_targets[cls]`。
  - `evaluate_coverage(frames_dir, labels_dir, masks_path, recall_targets: dict[str, float] | None = None)`。
  - `pool()` 遇到不同的召回目标抛 `ValueError("cannot pool coverage judged against different recall targets")`。
  - `coverage_json(c)["recall_targets"]`；`RunConfig.recall_targets: dict[str, float] | None = None`。
  - `summary.md` 开头多一行 `- P1 召回目标：人脸 ≥ 95%，车牌 ≥ 98%（vc_eval run --recall-target 设定，默认各 99%）`；样本量提醒写各类自己的目标。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_eval_coverage.py`：imports 的第一行改成 `from vc_eval.coverage import CoverageReport, _empty_class, evaluate_coverage, parse_recall_targets, pool`，文件末尾加：

```python
@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {"face": 0.99, "plate": 0.99}),
        ("face=0.95,plate=0.98", {"face": 0.95, "plate": 0.98}),
        (" plate = 0.98 ", {"face": 0.99, "plate": 0.98}),
        ("face=1", {"face": 1.0, "plate": 0.99}),
    ],
)
def test_parse_recall_targets(text, expected):
    assert parse_recall_targets(text) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("car=0.9", "car"),
        ("face", "<class>=<value>"),
        ("face=0.9,face=0.95", "twice"),
        ("face=high", "not a number"),
        ("face=0", r"\(0, 1\]"),
        ("plate=1.5", r"\(0, 1\]"),
        ("face=nan", r"\(0, 1\]"),
    ],
)
def test_parse_recall_targets_refuses_bad_text(text, message):
    with pytest.raises(ValueError, match=message):
        parse_recall_targets(text)


def test_a_class_passes_at_its_own_recall_target(tmp_path):
    frames = _frames(tmp_path, "a", list(range(20)))
    face = Box("face", 100, 100, 130, 140, src="teacher_retinaface")
    labels = _labels(tmp_path, "a", {fr: [face] for fr in range(20)})
    masks = _masks(tmp_path / "m.jsonl", "a.mp4", {fr: [face] for fr in range(1, 20)}, list(range(20)))  # 19 of 20
    default = evaluate_coverage(frames, labels, masks)
    assert default.recall_targets == {"face": 0.99, "plate": 0.99} and not default.class_passed("face")
    lowered = evaluate_coverage(frames, labels, masks, {"face": 0.95})
    assert lowered.recall_targets == {"face": 0.95, "plate": 0.99}
    assert lowered.class_passed("face")  # 19 / 20 is exactly 95 %


def test_pooling_needs_the_same_recall_targets():
    def classes():
        return {c: _empty_class(H) for c in ("face", "plate")}

    a = CoverageReport("a.mp4", 1, W, H, classes(), recall_targets={"face": 0.95, "plate": 0.98})
    b = CoverageReport("b.mp4", 1, W, H, classes())
    with pytest.raises(ValueError, match="different recall targets"):
        pool([a, b])
    assert pool([a, a]).recall_targets == {"face": 0.95, "plate": 0.98}
```

`py/tests/test_eval_run.py`：

- `test_bad_configuration_is_refused_before_any_run` 的参数表末尾加 `({"recall_targets": {"car": 0.9}}, "recall targets"),` 和 `({"recall_targets": {"face": 1.5}}, "recall targets"),`。
- 在 `test_no_sample_size_warning_for_a_class_that_fails_or_has_no_targets` 之后加：

```python
def test_recall_targets_set_the_p1_bar_per_class(tmp_path, monkeypatch):
    summary = _grid_run(tmp_path, monkeypatch, missed_faces=5, recall_targets={"face": 0.95, "plate": 0.98})
    pooled = [p for p in summary["pooled"] if p["fps"] == 5][0]
    assert (pooled["coverage"]["face"]["hits"], pooled["coverage"]["face"]["total"]) == (195, 200)
    assert pooled["coverage"]["recall_targets"] == {"face": 0.95, "plate": 0.98}
    assert pooled["passed"] is True and summary["chosen_fps"] == 5  # 97.5 % clears 95 %; the default 99 % would not
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "- P1 召回目标：人脸 ≥ 95%，车牌 ≥ 98%" in md
    lo = f"{wilson_interval(195, 200)[0]:.1%}"
    assert f"- 人脸 n=200：实测 195/200，95% 区间下限只有 {lo}，样本不足以证明 ≥ 95%" in md


def test_the_default_recall_target_is_named_in_the_summary(tmp_path, monkeypatch):
    _setup(tmp_path)
    _python_dets(monkeypatch)
    summary = run_all(_cfg(tmp_path), runner=_fake_runner([]))
    assert summary["runs"][0]["coverage"]["recall_targets"] == {"face": 0.99, "plate": 0.99}
    md = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "- P1 召回目标：人脸 ≥ 99%，车牌 ≥ 99%" in md
```

- `test_cli_run_parses_arguments_and_maps_exit_codes` 里，`assert cfg.redacted_root == Path("data/redacted")` 之后加 `assert cfg.recall_targets == {"face": 0.99, "plate": 0.99}`；`assert main([*argv, "--redacted-root", …]) == 0 and …` 那一行之后加

```python
    assert main([*argv, "--recall-target", "face=0.95,plate=0.98"]) == 0
    assert seen[-1].recall_targets == {"face": 0.95, "plate": 0.98}
    capsys.readouterr()
    assert main([*argv, "--recall-target", "face=2"]) == 2
    assert "--recall-target" in capsys.readouterr().err
```

- `test_cli_coverage_and_errors` 末尾加

```python
    args = ["coverage", "--frames", str(tmp_path / "frames" / STEM), "--labels", str(tmp_path / "labels" / STEM), "--masks", str(masks)]
    assert main([*args, "--recall-target", "plate=x"]) == 2
    assert "--recall-target" in capsys.readouterr().err
```

- [ ] **Step 2: 运行，确认失败**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py -p no:cacheprovider -q -k recall
```

预期：收集失败，`ImportError: cannot import name 'parse_recall_targets' from 'vc_eval.coverage'`。

- [ ] **Step 3: 实现**

`py/vc_eval/coverage.py`：

- `from vc_eval.metrics import (…)` 不变（已有 `CLASS_PASS`）。
- `EXCLUDE_IOU = …` 之后加：

```python
def default_recall_targets() -> dict[str, float]:
    """P1's bar per class: CLASS_PASS (the main design's production target) unless the run sets another."""
    return dict.fromkeys(LABEL_CLASSES, CLASS_PASS)


def parse_recall_targets(text: str) -> dict[str, float]:
    """"face=0.95,plate=0.98" -> {"face": 0.95, "plate": 0.98}; a class left out keeps CLASS_PASS.

    Raises ValueError for an unknown class, a part without "=", a class given twice, or a value that is not a number in
    (0, 1].
    """
    targets = default_recall_targets()
    seen: set[str] = set()
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        cls, sep, value = part.partition("=")
        cls = cls.strip()
        if not sep or cls not in LABEL_CLASSES:
            raise ValueError(f"--recall-target expects <class>=<value> with class {' or '.join(LABEL_CLASSES)}, got {part!r}")
        if cls in seen:
            raise ValueError(f"--recall-target gives {cls} twice")
        try:
            target = float(value)
        except ValueError:
            raise ValueError(f"--recall-target {cls}: {value.strip()!r} is not a number") from None
        if not 0 < target <= 1:  # NaN fails too
            raise ValueError(f"--recall-target {cls}: {value.strip()} must be in (0, 1]")
        seen.add(cls)
        targets[cls] = target
    return targets
```

- `CoverageReport` 末尾加字段 `recall_targets: dict[str, float] = field(default_factory=default_recall_targets)  # P1's bar per class`，`class_passed` 换成：

```python
    def class_passed(self, cls: str) -> bool:
        rate = self.classes[cls].rate
        return rate.total > 0 and rate.value >= self.recall_targets[cls]
```

- `evaluate_coverage` 的签名改成 `def evaluate_coverage(frames_dir: Path, labels_dir: Path, masks_path: Path, recall_targets: dict[str, float] | None = None) -> CoverageReport:`，加 docstring `"""recall_targets: P1's bar per class (default_recall_targets() for the classes it leaves out)."""`，`report = CoverageReport(…)` 一行换成

```python
    report = CoverageReport(
        video, len(labels.used), width, height, {c: _empty_class(height) for c in LABEL_CLASSES},
        recall_targets={**default_recall_targets(), **(recall_targets or {})},
    )
```

- `pool`：`first = reports[0]` 之后加

```python
    if any(r.recall_targets != first.recall_targets for r in reports):
        raise ValueError("cannot pool coverage judged against different recall targets")
```

  返回的 `CoverageReport(…)` 里加 `recall_targets=dict(first.recall_targets),`。

`py/vc_eval/run.py`：

- `RunConfig` 末尾加 `recall_targets: dict[str, float] | None = None  # P1's bar per class; None: CLASS_PASS for every class`。
- `_validate` 里 `if cfg.consistency_count < 1 …` 之前加

```python
    if cfg.recall_targets is not None:
        bad = {cls: v for cls, v in cfg.recall_targets.items() if cls not in LABEL_CLASSES or not 0 < v <= 1}
        if bad:
            raise ValueError(f"recall targets must be {' / '.join(LABEL_CLASSES)} in (0, 1], got {bad}")
```

- `_evaluate` 的 `evaluate_coverage(…)` 调用加第四个实参 `cfg.recall_targets`。
- `coverage_json` 在 `out["passed"] = c.passed()` 之前加 `out["recall_targets"] = dict(c.recall_targets)`。

`py/vc_eval/cli.py`：

- `coverage` 子命令和 `run` 子命令各加

```python
    c.add_argument("--recall-target", default="", help="P1 recall per class, e.g. face=0.95,plate=0.98 (default 0.99 each)")
```

  （`run` 的那个写成 `r.add_argument(…)`）。
- `_coverage`：`from vc_eval.coverage import evaluate_coverage` 改成 `from vc_eval.coverage import evaluate_coverage, parse_recall_targets`，调用改成 `evaluate_coverage(args.frames, args.labels, args.masks, parse_recall_targets(args.recall_target))`。
- `_run`：`from vc_eval.coverage import parse_recall_targets`，`RunConfig(…)` 里加 `recall_targets=parse_recall_targets(args.recall_target),`。

`py/vc_eval/report.py`：

- `_thin_samples`、`_thin_sample_line`、`_candidate_line` 换成：

```python
def _target_text(target: float) -> str:
    return f"{target * 100:g}%"


def _targets(coverage: dict) -> dict[str, float]:
    """P1's bar per class; summaries written before recall targets existed used CLASS_PASS for every class."""
    return coverage.get("recall_targets") or dict.fromkeys(LABEL_CLASSES, CLASS_PASS)


def _thin_samples(coverage: dict) -> list[tuple[str, dict, float]]:
    """Classes that pass P1 although their sample cannot prove it: the lower end of the 95 % interval is under the bar."""
    targets = _targets(coverage)
    return [
        (cls, coverage[cls], targets[cls])
        for cls in LABEL_CLASSES
        if coverage[cls]["total"] and coverage[cls]["value"] >= targets[cls] and coverage[cls]["ci"][0] < targets[cls]
    ]


def _thin_sample_line(cls: str, rate: dict, target: float) -> str:
    lead = "即使全部遮住" if rate["hits"] == rate["total"] else f"实测 {rate['hits']}/{rate['total']}"
    return f"- {CLS_NAME[cls]} n={rate['total']}：{lead}，95% 区间下限只有 {rate['ci'][0]:.1%}，样本不足以证明 ≥ {_target_text(target)}"


def _candidate_line(summary: dict) -> str:
    chosen = summary["chosen_fps"]
    if chosen is None:
        return "- 候选检测频率：无满足 P1–P3 的频率"
    coverage = _pooled_at(summary, chosen)["coverage"]
    line = f"- 候选检测频率：{chosen:g} fps（P1–P3 通过；P4 待你看完 index.html 确认）"
    if coverage["p2"] == "n/a":
        line += f"（没有热力格达到 {HEATMAP_MIN_TARGETS} 个目标，P2 空洞满足；空间上的薄弱点只能靠 P4 抽查把关）"
    thin = _thin_samples(coverage)
    if thin:
        names = "、".join(CLS_NAME[cls] for cls, _, _ in thin)
        line += f"；注意：{names}的目标数太少，样本量不足以证明达到召回目标（见覆盖率表下的提醒）"
    return line


def _target_line(summary: dict) -> str:
    targets = _targets(summary["pooled"][0]["coverage"]) if summary["pooled"] else dict.fromkeys(LABEL_CLASSES, CLASS_PASS)
    shown = "，".join(f"{CLS_NAME[cls]} ≥ {_target_text(targets[cls])}" for cls in LABEL_CLASSES)
    return f"- P1 召回目标：{shown}（`vc_eval run --recall-target` 设定，默认各 {_target_text(CLASS_PASS)}）"
```

- `_head_lines` 返回的列表里 `_candidate_line(summary),` 之后加 `_target_line(summary),`。
- `_coverage_lines` 里 `thin = [_thin_sample_line(cls, rate) for cls, rate in _thin_samples(c)]` 换成 `thin = [_thin_sample_line(cls, rate, target) for cls, rate, target in _thin_samples(c)]`。

`py/README.md` vc_eval 一节：P1 那一行的“人脸、车牌各自 ≥ 99%”改成“人脸、车牌各自 ≥ 99%（`--recall-target` 可按类改，本轮验收用人脸 95%、车牌 98%）”；在 `--face-model` 那条之后加一条：

```
- `--recall-target face=0.95,plate=0.98`（可选）：P1 的每类召回目标，没写的类别仍是 0.99；`summary.md` 开头一行写明，样本量提醒按各类自己的目标写；`coverage` 子命令也接受同一参数。格式不对（未知类别、缺 `=`、重复、不在 (0, 1]）退出码 2。
```

- [ ] **Step 4: 运行，确认通过**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
cd py && "${SB[@]}" /Users/warsen/Projects/video-compliance-process/.venv/bin/python -m pytest tests/test_eval_coverage.py tests/test_eval_run.py -p no:cacheprovider -q
```

预期：全部通过（原有的样本量提醒测试仍按 "≥ 99%" 通过：默认目标的文字仍是 `99%`）。

- [ ] **Step 5: 提交**

```bash
git add py/vc_eval/coverage.py py/vc_eval/run.py py/vc_eval/report.py py/vc_eval/cli.py py/tests/test_eval_coverage.py py/tests/test_eval_run.py py/README.md
git commit -m "feat: per-class recall targets for vc_eval run and coverage"
```

---

### Task 11: 校准与验收（控制者用真实数据做）

本设计 §6。只在留出段 `sg_drive_1200s`、`sg_drive_2400s` 上校准 `shrink_per_frame`、两个确认门槛、`expand_fraction`、`motion_expand_fraction`；在从没用过的 `sg_drive_60s` 和 64 分钟全片 `data/videos/sg_full.mp4` 上只确认、不调。任何一项验收不过就停下，把数字报给用户，不要扩大网格，也不要拿 60s 或全片来调。

**Files:**
- Create（不进 git）：`data/labels/sg_drive_1200s/excluded_refs.csv`、`data/labels/sg_drive_60s/excluded_refs.csv`、`data/calib/calib.py`、`data/calib/*.json`、`data/reports/calib/results.jsonl`、`data/reports/stable-3seg/`、`data/reports/stable-full/`、`data/redacted/calib/`、`data/redacted/stable-3seg/`、`data/redacted/stable-full/`、`data/redacted/speed/`
- Modify: `cpp/apps/vc_redact/redact_config.h`（默认值）、`configs/redact.json`、`cpp/tests/redact_config_test.cpp`
- Modify: `docs/USAGE.md`、`cpp/README.md`
- Create: `docs/reports/stable-masks-2026-10-19.md`

**Interfaces:**
- Consumes: Task 1–10 的全部产物；`data/redacted/r3-yunet-3seg/fps10/<stem>/masks.jsonl`（上一轮 r3 + YuNet 的 10 fps 结果，用来核对排除表）；`data/reports/r3-yunet-3seg/img/<stem>/miss_*.jpg`（上一轮人工核查过的漏遮小图）。
- Produces: 校准后的默认值 S（`shrink_per_frame`）、F（`face_confirm_score`）、P（`plate_confirm_score`）、E（`expand_fraction`）、M（`motion_expand_fraction`），写进 `RedactConfig` 和 `configs/redact.json`；验收报告；`docs/USAGE.md` 的实测速度。

- [ ] **Step 1: 准备**

`feat/stable-masks` 已合并进 `main`，主 checkout 在 `main` 上（`git -C /Users/warsen/Projects/video-compliance-process status` 干净、`git log -1` 是合并提交；合并是本地的，不需要 pull）。从 `main` 切出校准分支并构建 Release：

```bash
cd /Users/warsen/Projects/video-compliance-process
git checkout -b feat/stable-masks-calibration
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && "${SB[@]}" cmake --build build/cpp-release -j 4
```

预期：构建成功，零警告。以下命令都在主 checkout 根目录运行，一次一个。

- [ ] **Step 2: 写排除表并人工核对**

上一轮（`docs/reports/proxy-validation-2026-10-19-yunet.md`）人工核查过的老师误检，坐标来自用上一轮的 `masks.jsonl` 重算 `evaluate_coverage` 得到的高置信漏遮列表（1200s 32 个、60s 7 个；`sg_drive_2400s` 没有）以及 1200s 第 4656 帧 `merged.jsonl` 里唯一的人脸框。**不排除**的两个：1200s 第 2832 帧（公交车侧面的小标识，难判断）和 60s 第 384 帧（真车牌，迎面公交的前牌）。

`data/labels/sg_drive_1200s/excluded_refs.csv`：

```csv
frame,cls,x1,y1,x2,y2,reason
192,plate,1884.7,1001.5,1920.0,1068.5,等红灯时停在画面右下角的公交尾灯（老师误检）
384,plate,1886.4,1002.1,1919.9,1067.5,等红灯时停在画面右下角的公交尾灯（老师误检）
432,plate,1886.5,1002.0,1920.0,1067.6,等红灯时停在画面右下角的公交尾灯（老师误检）
480,plate,1886.2,1002.0,1919.9,1067.2,等红灯时停在画面右下角的公交尾灯（老师误检）
528,plate,1886.4,1002.4,1920.0,1067.9,等红灯时停在画面右下角的公交尾灯（老师误检）
576,plate,1886.2,1002.1,1920.0,1067.4,等红灯时停在画面右下角的公交尾灯（老师误检）
624,plate,1886.8,1002.5,1920.0,1067.2,等红灯时停在画面右下角的公交尾灯（老师误检）
672,plate,1886.7,1002.8,1920.0,1066.8,等红灯时停在画面右下角的公交尾灯（老师误检）
720,plate,1886.9,1002.7,1919.9,1067.3,等红灯时停在画面右下角的公交尾灯（老师误检）
768,plate,1887.1,1002.3,1920.0,1067.1,等红灯时停在画面右下角的公交尾灯（老师误检）
816,plate,1887.1,1001.8,1920.0,1066.8,等红灯时停在画面右下角的公交尾灯（老师误检）
864,plate,1887.2,1003.1,1920.0,1067.2,等红灯时停在画面右下角的公交尾灯（老师误检）
912,plate,1887.1,1002.4,1920.0,1066.7,等红灯时停在画面右下角的公交尾灯（老师误检）
960,plate,1887.4,1002.5,1920.0,1067.9,等红灯时停在画面右下角的公交尾灯（老师误检）
1008,plate,1887.1,1002.4,1920.0,1067.1,等红灯时停在画面右下角的公交尾灯（老师误检）
1056,plate,1887.1,1002.0,1920.0,1066.9,等红灯时停在画面右下角的公交尾灯（老师误检）
1104,plate,1887.4,1002.8,1920.0,1066.7,等红灯时停在画面右下角的公交尾灯（老师误检）
1152,plate,1886.9,1002.7,1920.0,1066.6,等红灯时停在画面右下角的公交尾灯（老师误检）
1200,plate,1886.8,1002.7,1920.0,1066.3,等红灯时停在画面右下角的公交尾灯（老师误检）
1248,plate,1886.3,1001.8,1920.0,1066.1,等红灯时停在画面右下角的公交尾灯（老师误检）
1296,plate,1886.8,1002.2,1920.0,1066.5,等红灯时停在画面右下角的公交尾灯（老师误检）
1344,plate,1886.6,1002.4,1920.0,1066.4,等红灯时停在画面右下角的公交尾灯（老师误检）
1392,plate,1886.1,1002.0,1920.0,1066.3,等红灯时停在画面右下角的公交尾灯（老师误检）
1440,plate,1886.4,1002.0,1920.0,1066.5,等红灯时停在画面右下角的公交尾灯（老师误检）
1488,plate,1886.8,1002.3,1920.0,1066.8,等红灯时停在画面右下角的公交尾灯（老师误检）
1536,plate,1886.9,1002.8,1920.0,1066.6,等红灯时停在画面右下角的公交尾灯（老师误检）
1584,plate,1886.9,1002.3,1920.0,1066.3,等红灯时停在画面右下角的公交尾灯（老师误检）
1680,plate,1886.8,1002.1,1920.0,1066.2,等红灯时停在画面右下角的公交尾灯（老师误检）
1728,plate,1886.7,1002.9,1920.0,1066.6,等红灯时停在画面右下角的公交尾灯（老师误检）
3072,plate,1561.5,739.1,1605.7,760.2,水马上的字（老师误检）
4128,plate,157.1,613.8,194.2,632.7,公交线路号 LED 屏（老师误检）
4656,face,1655.1,24.1,1721.9,113.7,花丛被认成人脸（老师误检）
```

`data/labels/sg_drive_60s/excluded_refs.csv`：

```csv
frame,cls,x1,y1,x2,y2,reason
1632,plate,1748.1,699.5,1812.2,713.8,车顶文字条（老师误检）
3936,plate,1101.9,658.2,1135.1,668.9,货车的 SLOW VEHICLE 警示牌（老师误检）
4080,plate,1288.5,647.8,1346.7,664.3,货车的 SLOW VEHICLE 警示牌（老师误检）
4128,plate,1482.9,632.9,1566.6,655.9,货车的 SLOW VEHICLE 警示牌（老师误检）
4320,plate,1634.0,764.2,1696.0,800.1,TOYOTA 字样旁的 L 牌（老师误检）
7152,plate,1766.1,668.8,1799.2,703.3,公交车身彩绘（老师误检）
```

人工核对（用 Read 看图）：`data/reports/r3-yunet-3seg/img/sg_drive_1200s/` 里的 `miss_000192_0.jpg`、`miss_001200_18.jpg`、`miss_001728_28.jpg`（尾灯）、`miss_003072_30.jpg`（水马）、`miss_004128_31.jpg`（LED 屏）、`miss_002832_29.jpg`（**保留**的小标识），`data/reports/r3-yunet-3seg/img/sg_drive_60s/` 里的全部 7 张 `miss_*.jpg`（`miss_000384_0.jpg` 是**保留**的真车牌）。第 4656 帧的人脸框用下面的命令裁出来再看：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
mkdir -p data/reports/calib
"${SB[@]}" .venv/bin/python -c "import cv2; im = cv2.imread('data/frames/sg_drive_1200s/004656.jpg'); cv2.imwrite('data/reports/calib/face_004656.jpg', cv2.rectangle(im[0:300, 1400:1920].copy(), (255, 24), (322, 114), (0, 0, 255), 2))"
```

（矩形在裁剪图里的坐标 = 原坐标减去 (1400, 0)。）看到的和表里的原因对不上（例如第 4656 帧框里是真人脸），就删掉那一行，并在验收报告里写明。

- [ ] **Step 3: 用上一轮的结果核对排除表**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" env PYTHONPATH=py .venv/bin/python - <<'EOF'
from pathlib import Path
from vc_eval.coverage import evaluate_coverage, pool

reports = []
for stem in ("sg_drive_1200s", "sg_drive_2400s", "sg_drive_60s"):
    r = evaluate_coverage(Path("data/frames") / stem, Path("data/labels") / stem,
                          Path("data/redacted/r3-yunet-3seg/fps10") / stem / "masks.jsonl")
    print(stem, "excluded", len(r.excluded), "unmatched rows", len(r.unmatched_exclusions),
          "misses", [(m.frame, m.box.cls) for m in r.misses])
    reports.append(r)
p = pool(reports)
face, plate = p.classes["face"].rate, p.classes["plate"].rate
print("pooled face", face.hits, face.total, "plate", plate.hits, plate.total)
assert len(p.unmatched_exclusions) == 0
assert (plate.hits, plate.total) == (766, 768), (plate.hits, plate.total)
assert face.hits == face.total
EOF
```

预期：1200s 排除 32 个（Step 2 删过行就相应少），60s 排除 6 个，没配上的行 0；剩下的漏遮只有 1200s 的 (2832, plate) 和 60s 的 (384, plate)；合计车牌 766/768，与上一轮报告"校正后 766/768"一致，人脸全遮住。断言不过就停下检查排除表。

- [ ] **Step 4: 写校准脚本**

`data/calib/calib.py`：

```python
"""One calibration point (plan 2026-10-19, Task 11): redacts the two calibration stems with configs/redact.json plus
the given overrides, then prints and appends to data/reports/calib/results.jsonl the pooled recall (after
excluded_refs.csv, targets face 0.95 / plate 0.98), the pooled viewing metrics and the throughput.

Run from the repository root: PYTHONPATH=py .venv/bin/python data/calib/calib.py <name> [key=value ...]
"""

import json
import subprocess
import sys
from pathlib import Path

from vc_eval.coverage import evaluate_coverage, parse_recall_targets, pool
from vc_eval.viewing import evaluate_viewing, pool_viewing, viewing_json

STEMS = ("sg_drive_1200s", "sg_drive_2400s")  # calibration only: never sg_drive_60s or sg_full
TARGETS = parse_recall_targets("face=0.95,plate=0.98")
KEYS = {"shrink_per_frame", "face_confirm_score", "plate_confirm_score", "expand_fraction", "motion_expand_fraction"}


def main() -> None:
    name, overrides = sys.argv[1], dict(arg.split("=", 1) for arg in sys.argv[2:])
    unknown = set(overrides) - KEYS
    if unknown:
        sys.exit(f"calib.py calibrates only {sorted(KEYS)}, not {sorted(unknown)}")
    config = json.loads(Path("configs/redact.json").read_text(encoding="utf-8"))
    config.update({k: float(v) for k, v in overrides.items()})
    config_path = Path("data/calib") / f"{name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    out = Path("data/redacted/calib") / name
    coverages, viewings, elapsed, frames = [], [], 0.0, 0
    for stem in STEMS:
        subprocess.run(
            ["build/cpp-release/vc_redact", "--input", f"data/videos/{stem}.mp4", "--out", str(out),
             "--model", "data/models/r3", "--face-model", "data/models/yunet", "--config", str(config_path), "--overwrite"],
            check=True, stdout=subprocess.DEVNULL,
        )
        masks = out / stem / "masks.jsonl"
        coverage = evaluate_coverage(Path("data/frames") / stem, Path("data/labels") / stem, masks, TARGETS)
        coverages.append(coverage)
        viewings.append(evaluate_viewing(masks, coverage.width, coverage.height))
        stats = json.loads((out / stem / "stats.json").read_text(encoding="utf-8"))
        elapsed += stats["elapsed_s"]
        frames += stats["frames"]
    c = pool(coverages)
    v = viewing_json(pool_viewing(viewings))
    row = {
        "name": name,
        **{k: float(x) for k, x in overrides.items()},
        "face": f"{c.classes['face'].rate.hits}/{c.classes['face'].rate.total}",
        "plate": f"{c.classes['plate'].rate.hits}/{c.classes['plate'].rate.total}",
        "recall_ok": c.class_passed("face") and c.class_passed("plate"),
        "area_mean": v["area"]["mean"],
        "area_p99": v["area"]["p99"],
        "area_max": v["area"]["max"],
        "flicker": v["flicker"]["share"],
        "boxes_p95": v["boxes"]["p95"],
        "multi_box_frames": v["multi_box_frames"],
        "fps": frames / elapsed,
    }
    print(json.dumps(row, ensure_ascii=False))
    results = Path("data/reports/calib/results.jsonl")
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 校准（三轮，逐个参数，每次两段约 2 分钟）**

每条命令前都先定义 `SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')`；命令形如 `"${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py <名字> <键=值>…`，一条跑完再跑下一条。每一行结果里 `recall_ok` 为真表示两段合计人脸 ≥ 95%、车牌 ≥ 98%（已排除参照误检）。

**A. 收缩速度 `shrink_per_frame`**（其余用 Task 6 的默认值：确认门槛 0.5 / 0.15，外扩 0.3 / 0.5）：

```bash
for s in 0.02 0.05 0.1 0.2; do "${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py A-shrink$s shrink_per_frame=$s; done
```

选法：在 `recall_ok` 为真、`area_p99 ≤ 0.10`、`area_max ≤ 0.30` 的行里，取 `flicker` 最小的（相同取 `area_mean` 小的），它的值记为 **S**。没有一行同时满足三条时，取 `recall_ok` 为真的行里 `area_p99` 最小的（都不满足 `recall_ok` 就取 0.02），记下"A 阶段没有满足面积上限的值"，继续——最终验收会把它判为不通过。

**B. 确认门槛**（`shrink_per_frame=S`）。人脸先变（车牌 0.15）：

```bash
for f in 0.3 0.4; do "${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py B-face$f shrink_per_frame=$S face_confirm_score=$f; done
```

人脸 0.5 那一点就是 A 阶段 S 那一行。选法：`recall_ok` 为真的值里取最大的（门槛越高误打越少），记为 **F**；都不满足取 0.3。然后车牌：

```bash
for p in 0.1 0.2; do "${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py B-plate$p shrink_per_frame=$S face_confirm_score=$F plate_confirm_score=$p; done
```

车牌 0.15 那一点是"S、F、0.15"那一行（F = 0.5 时就是 A 阶段 S 那一行，否则是 B-face$F 那一行）。选法：`recall_ok` 为真的值里取最大的，记为 **P**；都不满足取 0.1。

**C. 外扩 `expand_fraction` / `motion_expand_fraction`**（S、F、P 固定）。(0.3, 0.5) 那一点就是"S、F、P"那一行，再跑三点：

```bash
"${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py C-e0.2-m0.5 shrink_per_frame=$S face_confirm_score=$F plate_confirm_score=$P expand_fraction=0.2 motion_expand_fraction=0.5
"${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py C-e0.3-m0.25 shrink_per_frame=$S face_confirm_score=$F plate_confirm_score=$P expand_fraction=0.3 motion_expand_fraction=0.25
"${SB[@]}" env PYTHONPATH=py .venv/bin/python data/calib/calib.py C-e0.2-m0.25 shrink_per_frame=$S face_confirm_score=$F plate_confirm_score=$P expand_fraction=0.2 motion_expand_fraction=0.25
```

选法：`recall_ok` 为真且 `flicker ≤ 0.05` 的行里取 `area_p99` 最小的（相同取 `area_mean` 小的），记为 **E** / **M**；都不满足就保留 0.3 / 0.5。

把五个值和 `data/reports/calib/results.jsonl` 的全部行记下来，Step 11 写进报告。所有行的 `multi_box_frames` 都必须是 0，不是就停下（这是 Task 4 的缺陷）。

- [ ] **Step 6: 把校准值写成默认值（先改测试）**

下面的 S、F、P、E、M 指 Step 5 选出的数，写成十进制小数字面量（例如 `0.05`、`0.4f`）。

`cpp/tests/redact_config_test.cpp`：

- `DefaultsAreTheValidatedSettings` 里 `EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.3);` 换成下面五行，并把测试上方注释的第一句改成 "vc_redact's defaults: 10 fps and score 0.05 from the 2026-10-18 proxy validation, the mask and confirmation settings from the 2026-10-19 calibration (docs/reports/stable-masks-2026-10-19.md); the lower-level structs keep their own, which vc_detect and the unit tests use."：

```cpp
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, E);
  EXPECT_DOUBLE_EQ(c.timeline.motionExpandFraction, M);
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, S);
  EXPECT_FLOAT_EQ(c.faceConfirmScore, F);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, P);
```

- `ReadsTheMaskSmoothingSettings` 的 `EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.shrinkPerFrame, 0.02);` 改成 `…, S);`。
- `ReadsTheConfirmScores` 的 `0.5f`、`0.15f` 两个默认值断言改成 `F`、`P`。
- `FloatSettingsAreWrittenWithSixDecimals` 的 `"0.5"`、`"0.15"` 改成 F、P 的十进制文本（例如 F = 0.4 时为 `"0.4"`）。

运行确认失败（在主 checkout 里先配置 Debug 构建）：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && "${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactConfig.*'
```

预期：五个值里与旧默认值不同的那几项 FAIL（全都和旧值相同时，这一步不会失败，直接进入下一步）。

实现：`cpp/apps/vc_redact/redact_config.h` 的 `RedactConfig` 里

```cpp
  TimelineConfig timeline{.expandFraction = E, .motionExpandFraction = M, .shrinkPerFrame = S};
```

`float faceConfirmScore = F;`、`float plateConfirmScore = P;`；结构体上方注释加一句 "The mask and confirmation settings are the 2026-10-19 calibration's (docs/reports/stable-masks-2026-10-19.md)."。`configs/redact.json` 的 `expand_fraction`、`motion_expand_fraction`、`shrink_per_frame`、`face_confirm_score`、`plate_confirm_score` 改成同样的值。`docs/USAGE.md` §4 表格这五行的"默认"列改成同样的值。

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" cmake --build build/cpp -j 4 && "${SB[@]}" ./build/cpp/tests/vc_tests --gtest_filter='RedactConfig.*:RedactJob.*'
"${SB[@]}" cmake --build build/cpp-release -j 4
git add cpp/apps/vc_redact/redact_config.h configs/redact.json cpp/tests/redact_config_test.cpp docs/USAGE.md
git commit -m "feat: calibrated mask smoothing, confirmation and margin defaults"
```

预期：全部 PASSED；Release 重新构建好，后面的确认用的就是新默认值。

- [ ] **Step 7: 三段确认（含从没用过的 60s）**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
"${SB[@]}" env PYTHONPATH=py .venv/bin/python -m vc_eval run --stems sg_drive_1200s,sg_drive_2400s,sg_drive_60s \
    --model data/models/r3 --face-model data/models/yunet --fps 10 --recall-target face=0.95,plate=0.98 \
    --out data/reports/stable-3seg --redacted-root data/redacted/stable-3seg --overwrite
"${SB[@]}" env PYTHONPATH=py .venv/bin/python - <<'EOF'
import json
s = json.load(open("data/reports/stable-3seg/summary.json", encoding="utf-8"))
p = next(x for x in s["pooled"] if x["fps"] == 10)
c, v = p["coverage"], p["viewing"]
checks = {
    "faces >= 95% (three segments, after exclusions)": c["face"]["value"] >= 0.95,
    "plates >= 98% (three segments, after exclusions)": c["plate"]["value"] >= 0.98,
    "same-track multi-box frames == 0": all(r["viewing"]["multi_box_frames"] == 0 for r in s["runs"]),
    "integrity passes on every segment": p["integrity_passed"],
}
for name, ok in checks.items():
    print("PASS" if ok else "FAIL", name)
print("face", c["face"]["hits"], c["face"]["total"], "plate", c["plate"]["hits"], c["plate"]["total"])
print("excluded", {r["stem"]: r["coverage"]["excluded"] for r in s["runs"]})
print("viewing (pooled, report only here)", v)
for r in s["runs"]:
    print(r["stem"], "misses", r["miss_list"])
EOF
```

预期：四项都是 PASS；`excluded` 里 1200s 32 个、60s 6 个、`unmatched_rows` 都是 0。把每个 `miss_list` 里的漏遮在 `data/reports/stable-3seg/index.html` 里看一遍（真目标还是老师误检），新出现的老师误检**不**补进排除表（那是本轮之后的人工核查），只在报告里列出。

- [ ] **Step 8: 64 分钟全片（观感与速度）**

先确认本机没有别的重任务（`ps -Ao pcpu,comm -r | head -5` 里除系统进程外没有持续占用 CPU 的进程），再跑：

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
mkdir -p data/reports/stable-full
"${SB[@]}" build/cpp-release/vc_redact --input data/videos/sg_full.mp4 --out data/redacted/stable-full \
    --model data/models/r3 --face-model data/models/yunet --overwrite
"${SB[@]}" env PYTHONPATH=py .venv/bin/python -m vc_eval viewing --masks data/redacted/stable-full/sg_full/masks.jsonl \
    --width 1920 --height 1080 --json data/reports/stable-full/viewing.json
"${SB[@]}" env PYTHONPATH=py .venv/bin/python -m vc_eval integrity --source data/videos/sg_full.mp4 \
    --redacted data/redacted/stable-full/sg_full/sg_full.redacted.mp4 --json data/reports/stable-full/integrity.json
"${SB[@]}" env PYTHONPATH=py .venv/bin/python - <<'EOF'
import json
v = json.load(open("data/reports/stable-full/viewing.json", encoding="utf-8"))
i = json.load(open("data/reports/stable-full/integrity.json", encoding="utf-8"))
st = json.load(open("data/redacted/stable-full/sg_full/stats.json", encoding="utf-8"))
fps = st["frames"] / st["elapsed_s"]
checks = {
    "masked area p99 <= 10%": v["area"]["p99"] <= 0.10,
    "masked area max <= 30%": v["area"]["max"] <= 0.30,
    "flicker <= 5%": v["flicker"]["share"] <= 0.05,
    "same-track multi-box frames == 0": v["multi_box_frames"] == 0,
    "speed >= 200 fps at 10": fps >= 200,
    "integrity": i["passed"],
}
for name, ok in checks.items():
    print("PASS" if ok else "FAIL", name)
print(f"{fps:.1f} fps, detect_fps_actual {st['detect_fps_actual']:.2f}, stride {st['keyframe_stride']}, "
      f"detect {st['stages']['detect']}, student {st['stages']['detect_student']}, face {st['stages']['detect_face']}")
print("viewing", v)
EOF
```

预期：六项都是 PASS；`detect_fps_actual` 约 9.99（59.94 fps 视频每 6 帧一次）。全片约 230,000 帧，按 200 fps 约 20 分钟。

- [ ] **Step 9: 各档位实测速度（`sg_drive_2400s`，1080p 59.94 fps，约 7,200 帧）**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
for f in 5 10 15 30 60; do
  "${SB[@]}" build/cpp-release/vc_redact --input data/videos/sg_drive_2400s.mp4 --out data/redacted/speed/fps$f \
      --model data/models/r3 --face-model data/models/yunet --detect-fps $f --overwrite
done
"${SB[@]}" env PYTHONPATH=py .venv/bin/python - <<'EOF'
import json
print("| 档位 | 实际每秒检测 | 每隔几帧 | 吞吐（帧 / 总耗时） | 每个关键帧检测（墙钟） | 学生 / YuNet 各自 |")
print("|---|---|---|---|---|---|")
for f in (5, 10, 15, 30, 60):
    s = json.load(open(f"data/redacted/speed/fps{f}/sg_drive_2400s/stats.json", encoding="utf-8"))
    st = s["stages"]
    ms = lambda stage: 1000 * stage["busy_s"] / max(1, stage["items"])  # noqa: E731
    print(f"| {f} | {s['detect_fps_actual']:.2f} | {s['keyframe_stride']} | {s['frames'] / s['elapsed_s']:.0f} fps | "
          f"{ms(st['detect']):.1f} ms | {ms(st['detect_student']):.1f} / {ms(st['detect_face']):.1f} ms |")
EOF
```

预期：打印出五行的表；60 档一行的 stderr 有一行 `vc_redact: note: detect_fps 60 is above the video's 59.94 fps, so every frame is a keyframe`。把表原样留给 Step 11。

- [ ] **Step 10: 打码前后对比图（交给用户人眼确认，P4）**

```bash
SB=(sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))')
mkdir -p data/reports/stable-full/compare
for t in 300 900 1500 2100 2700 3300; do
  "${SB[@]}" ffmpeg -v error -y -ss $t -i data/videos/sg_full.mp4 -ss $t -i data/redacted/stable-full/sg_full/sg_full.redacted.mp4 \
      -filter_complex "[0:v][1:v]hstack=inputs=2,scale=1920:-2" -frames:v 1 -q:v 3 data/reports/stable-full/compare/compare_${t}s.jpg
done
```

预期：6 张左原片、右打码的对比图。自己先用 Read 看一遍（马赛克贴着目标、相邻帧不该整片重排已由单元测试保证，这里看有没有明显漏打和大片误打），再在最后的回复里把这 6 个路径和 `data/reports/stable-3seg/index.html` 交给用户确认。

- [ ] **Step 11: 文档与报告**

`docs/reports/stable-masks-2026-10-19.md`（中文，数字全部来自上面各步的输出，不估算）：

```markdown
# 打码观感与速度：校准与验收（2026-10-19）

设计：`docs/superpowers/specs/2026-10-19-stable-masks-and-speed-design.md`；计划：`docs/superpowers/plans/2026-10-19-stable-masks-and-speed.md`。原始结果：`data/reports/calib/results.jsonl`、`data/reports/stable-3seg/`、`data/reports/stable-full/`。

## 排除的参照误检
（Step 2 两张表的行数和原因汇总；Step 3 的核对结果：上一轮结果重算为车牌 766/768；保留没排除的两个及原因）

## 校准（只用 sg_drive_1200s、sg_drive_2400s）
（results.jsonl 每行一行的表：名字、改的参数、人脸、车牌、recall_ok、面积均值 / p99 / 最大、跳变、框数 p95、吞吐；每阶段按 Step 5 的规则选了哪个值、为什么）
选定：shrink_per_frame = S，face_confirm_score = F，plate_confirm_score = P，expand_fraction = E，motion_expand_fraction = M。

## 验收（本设计 §6）
| 方面 | 之前 | 目标 | 实测 | 通过 |
|---|---|---|---|---|
| 人脸召回（排除参照误检后，三段合计） | 48/48 | ≥ 95% | Step 7 | |
| 车牌召回（排除参照误检后，三段合计） | 766/768 | ≥ 98% | Step 7 | |
| 每帧打码面积 p99（全片） | 25% | ≤ 10% | Step 8 | |
| 每帧打码面积最大值（全片） | 79% | ≤ 30% | Step 8 | |
| 帧间跳变（全片） | 20% | ≤ 5% | Step 8 | |
| 同一轨迹在同一帧有多个框的帧数 | — | 0 | Step 7、8 | |
| 速度（10 档，M5，1080p60，全片） | 152 fps | ≥ 200 fps | Step 8 | |
| 输出完整性 | 通过 | 通过 | Step 7、8 | |

三段各自的观感指标（Step 7 的 summary.json）也列一张表，只报告。

## 各档位速度（sg_drive_2400s）
（Step 9 的表）

## 漏遮核查
（Step 7 每段 miss_list 的逐个核查：真目标 / 老师误检）

## 对比图（P4，待用户确认）
（Step 10 的 6 个路径）

## 已知问题
（在对比图或 index.html 里看到的大片误打、打码框跳动或漏打的具体例子，写明帧号和原因；没看到就写没看到）
```

`docs/USAGE.md`：

- §3 "速度"一条换成 `- **速度：** 1080p 59.94 fps 实测（sg_drive_2400s，Apple M5）：` 后接 Step 9 的表（五行），再加一句 `64 分钟全片 10 档：<Step 8 的吞吐> fps。`
- §5 "结果数字"一条换成：三段留出视频（排除 38 个人工核查过的老师误检后）人脸 <Step 7>、车牌 <Step 7>；全片每帧打码面积均值 / p99 / 最大 <Step 8>，帧间跳变 <Step 8>；详见 `docs/reports/stable-masks-2026-10-19.md`。
- §5 "误打"一条换成全片面积均值和最大值的实测数。

`cpp/README.md` "人脸模型"一节的"速度代价"一条末尾加一句：`两个检测器并行后（2026-10-19），10 档每个关键帧检测的墙钟时间约 <Step 9 10 档一行的检测毫秒数> ms，整体约 <Step 9 10 档一行的吞吐> fps，见 docs/USAGE.md 的各档位速度表。`

```bash
git add docs/reports/stable-masks-2026-10-19.md docs/USAGE.md cpp/README.md
git commit -m "docs: stable-mask calibration, acceptance results and measured speeds"
```

- [ ] **Step 12: 收尾**

Step 7、8 的十项全部 PASS 时：按 superpowers:finishing-a-development-branch 把 `feat/stable-masks-calibration` 交给用户决定合并方式，合并后把主 checkout 切回最新的 `main`；回复里给出验收表、对比图路径和 `index.html`，请用户做 P4 人眼确认。任何一项 FAIL：不合并、不再调参，把 Step 5 的全部结果行和失败项报给用户，由用户决定下一步（例如改用另一组网格、或接受某项不达标）。

---

## 规格覆盖与已定的取舍

**本设计各节 → 任务：** §2 检测频率（档位、步长、实际频率、超过视频帧率逐帧并提示、范围报错、帧率退回、删键、120 帧保护、默认 10、`VC_DETECT_FPS`）→ Task 1、2；§2 并行检测与计时 → Task 3；§2 实测速度写进 `docs/USAGE.md` → Task 11 Step 9、11；§3.1 插值与外扩 → Task 4，迟滞与面积上限 → Task 5；§3.2 删除的行为 → Task 4；§3.3 丢失、幽灵、`track` / `origin` → Task 4；§3.4 低分确认与 `confirmed: false` → Task 6；§3.5 已知代价 → Task 4、6 的文档；§4 马赛克 → Task 7；§5 观感指标 → Task 8，排除表 → Task 9，`--recall-target` → Task 10；§6 校准与验收、对比图 → Task 11；§7 测试 → 各任务 Step 1；§8 不做。

**规格没写死、本计划定下的做法：**

1. 请求值**严格大于**视频帧率才打印提示（等于时就是逐帧，没什么可提示）；提示在运行结束后与解码错误警告一起打印到 stderr（`vc_redact: note: …`），不改 `runRedact` 的接口。
2. 第一帧总是关键帧（`BoxTimeline` 需要从关键帧开始）；关键帧判定为 `帧序号 % stride == 0`，与"缓冲满 120 帧强制关键帧"并存。
3. 视频帧率高于 1e6 视为坏文件头，报错（防止步长换算溢出）。
4. 丢失后的外推上限按轴分别截断：x 方向不超过半个框宽、y 方向不超过半个框高，用 `clamp`（按速度走到上限后停住），不是把速度整体缩小。保持框的运动外扩用截断后的整段位移；视频结尾那一段用"尾段帧数"对应的位移。
5. 幽灵在 k1 那一帧的 `origin` 记为 `held`，之后的帧记为 `interpolated`。
6. 宽松配对配上的轨迹不再另留幽灵（旧代码为防配错留的），因为那会让同一目标出现两个框；配错的代价按 §3.5 接受。
7. 没确认的低分轨迹没配上就结束，不留幽灵："相邻两个关键帧"不包括隔一个关键帧再出现。
8. `detections.jsonl` 的每一行都写 `confirmed`（`true` / `false`），写出时间推迟到下一个关键帧；`vc_detect` 的输出不变（没有这个字段）。`vc_eval temporal` 的逐帧参照去掉 `confirmed: false` 的框。
9. 大小迟滞平滑的是**大小**，不是各条边（规格 §3.1 已按协调者确认的规则改写）：每条轨迹保存平滑后的宽、高，输出框以当帧原始框的中心为中心，所以快速移动的目标不会沿运动方向拉长；原始框按裁剪前的大小算，输出框最后再裁到画面内；`origin` 为 `held` 的帧不收缩（丢失后保留最后的大小、放在保持位置上），检出帧和插值帧照常收缩。
10. 面积上限让宽、高按同一比例 f 向原始框的宽、高收缩，用闭式解 f = 2q / (p + √(p² + 4abq))（一条轴没超出时 a·b = 0，同一式子仍成立）。`shrink_per_frame` 的范围定为 (0, 1]（取 1 时平滑不起作用），`max_mask_area_ratio` ≥ 1。
11. 块大小档位最高 256：计算值超过 256 时用 256（所以 `mosaic_min_block_px` 大于 256 不再起作用）；块大小必须是 ≥ 2 的偶数，保证 4:2:0 / 4:2:2 色度网格与亮度网格对齐。
12. 观感指标要求 `masks.jsonl` 一帧一行、帧号从 0 连续；没有 `track` 字段的旧输出把"同轨多框帧"记为 `null` / "—"。另加了 `vc_eval viewing` 子命令，因为 64 分钟全片没有参照标注，`vc_eval run` 用不了。
13. 被排除的参照框同时退出误打面积的"参照"（打在误检上的码算误打）；一个参照框都没配上的排除行在报告里单独提醒；`summary.json` 增加 `miss_list` 便于写排除表。
14. `--recall-target` 也加到 `vc_eval coverage`，校准脚本直接用。
15. `stats.json` 没有人脸模型时 `detect_face` 为 `{busy 0, items 0}`，`detect_student` 等于 `detect`。
16. 观感验收（面积 p99 / 最大、跳变）按 64 分钟全片判定（规格表里"现在"的数字就是全片的）；三段的观感只报告；召回按三段合计判定；同轨多框帧在三段和全片上都必须是 0。
17. 校准不用 `vc_eval run`（它每次都会重跑逐帧参照），而用 `data/calib/calib.py` 直接跑两段 `vc_redact` 并合并算召回和观感；按 A（收缩）→ B（确认门槛）→ C（外扩）逐个参数、按写死的选法挑值。
