# C++ 脱敏流水线（vc_redact）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `vc_redact`：解码视频 → 关键帧上用 ONNX 学生模型（或测试用的 `FakeDetector`）检测人脸 / 车牌 → `BoxTimeline` 给每一帧算打码框 → 在 YUV 平面上像素化 → 按原编码格式重新编码，音频直接拷贝；同时写出 `detections.jsonl`、`masks.jsonl`、`stats.json`。另加调试工具 `vc_detect`（对一批 JPEG 跑同一个检测器，用于 C++ / Python 一致性检查）。

**Architecture:** 在现有 `cpp/` 工程上扩展。纯逻辑组件放 `src/`（`detect/` 检测相关、`track/` BoxTimeline、`mask/` 马赛克、`io/` 新增 VideoWriter 并扩展 VideoReader 的音频透传），全部可单测；onnxruntime 与 nlohmann/json 只链接进 `vc_detect_lib`。`apps/vc_redact/` 是流水线编排：解码线程 → 有界队列 → 主线程（选关键帧、检测、补框、打码）→ 有界队列 → 编码封装线程。测试全部用 ffmpeg 现场生成的小视频、`FakeDetector` 和一个随仓库提交的 1KB 常量 ONNX 模型。

**Tech Stack:** C++20、CMake ≥ 3.24、FFmpeg 8（libavformat / libavcodec / libswscale / libavutil）、onnxruntime 1.30.0（官方 macOS arm64 预编译包，含 CoreML EP）、nlohmann/json 3.12.0（单头文件）、GoogleTest 1.18。

**规格：** `docs/superpowers/specs/2026-10-18-redaction-proxy-validation-design.md` §4.1、§5（主设计 `2026-10-17-dashcam-redaction-mvp-design.md` §3.2、§3.3、§7）。

**计划路线：** 1 vc_extract（已完成）→ 2 vc_label（已完成）→ 3 vc_train（并行）→ **4 本计划** → 5 代理评测。

## Global Constraints

- C++20，`-Wall -Wextra -Wpedantic` 零警告；不引入 OpenCV；FFmpeg ≥ 8（已有 `cmake/FindFFmpeg.cmake`）。
- 第三方包**已解压在共享缓存**，CMake 只从这里找，配置阶段不下载任何东西：
  - `~/.cache/video-compliance/third_party/onnxruntime-osx-arm64-1.30.0/`（`include/`、`lib/libonnxruntime.dylib`；sha256 of the tgz `6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012`）。注意：包里自带的 `lib/cmake/onnxruntime/onnxruntimeTargets.cmake` 把头文件目录写成了不存在的 `include/onnxruntime`，**不要用 `find_package(onnxruntime)`**，用本计划的 `cmake/ThirdParty.cmake` 自建导入目标。
  - `~/.cache/video-compliance/third_party/nlohmann_json-3.12.0/include/nlohmann/json.hpp`（sha256 `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`）。
- **不联网，不安装任何东西。**
- 构建与测试（从仓库根目录，worktree 内）：`cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`；测试**只按过滤器运行**本任务相关的用例，如 `./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*'`，不得运行不带过滤器的 `vc_tests` 或 `ctest`。
- 推理几何（规格 §4.1）：`r = min(W_in / W, H_in / H)`；缩放后尺寸 `round(W·r) × round(H·r)`；图像放左上角，其余填 114；RGB；像素 / 255；模型输出坐标除以 `r`。
- YOLOv8 输出 `[1, 4 + nc, N]`：前 4 行 `cx, cy, w, h`（输入像素），其后每类一行分数；每个锚点取最高分类别；分数 `>=` 阈值保留；按类 NMS，IoU `>=` 阈值即抑制（与 `vc_label.boxes.nms` 一致）。
- 检测记录 JSON 字段及顺序固定为 `video, frame, pts_ms, cls, x1, y1, x2, y2, score, src`；打码记录为 `video, frame, boxes[{cls, x1, y1, x2, y2, origin}]`，`origin ∈ {detected, interpolated, extended}`；坐标与 `pts_ms` 保留 1 位小数，`score` 4 位。`masks.jsonl` **每帧一行**。
- 帧号 = `VideoReader` 的解码序号（与 `vc_extract` 的 `frames.csv` 一致）。
- 宁可失败也不输出没打码的内容：检测器任何错误中止整段；视频先写 `<stem>.redacted.mp4.partial`，成功后重命名；失败时删除本次写出的所有文件。
- 提交信息用 conventional commits，不加署名行。在分支 `feat/vc-redact` 上工作（worktree `.claude/worktrees/vc-redact`）。

## 文件结构

```
configs/redact.json                  vc_redact 默认参数（与 RedactConfig 默认值一致，测试保证）
cpp/
  CMakeLists.txt                     新增 vc_detect_lib、vc_redact_lib、vc_redact、vc_detect
  cmake/ThirdParty.cmake             nlohmann/json 与 onnxruntime 导入目标（只读缓存）
  README.md                          第三方包、vc_redact、vc_detect 用法
  src/common/vc_error.h              新增 UsageError、OutputExistsError（从 vc_extract 头文件移来）
  src/common/bounded_queue.h         线程间有界队列
  src/detect/box.h, box.cpp          Box、iou、nmsPerClass
  src/detect/letterbox.h, .cpp       Letterbox、InputTensorFiller（swscale → RGB float）
  src/detect/detector.h              Detector 接口
  src/detect/model_info.h, .cpp      model.json 解析
  src/detect/yolo_decode.h, .cpp     decodeYolov8、DecodeParams
  src/detect/fake_detector.h, .cpp   FakeDetector（重放 detections.jsonl）
  src/detect/onnx_yolo_detector.h, .cpp  OnnxYoloDetector（CoreML / CPU）
  src/track/box_timeline.h, .cpp     BoxTimeline
  src/mask/mosaic.h, .cpp            像素化
  src/io/video_reader.h, .cpp        扩展：音频透传、被丢弃流列表
  src/io/video_writer.h, .cpp        VideoWriter（VideoToolbox / libx264 / libx265 + 音频拷贝）
  apps/vc_redact/redact_config.h, .cpp   configs/redact.json 解析与校验
  apps/vc_redact/records.h, .cpp     detections / masks 记录格式
  apps/vc_redact/redact_job.h, .cpp  runRedact（三线程流水线）
  apps/vc_redact/redact_cli.h, .cpp  命令行解析
  apps/vc_redact/main.cpp
  apps/vc_detect/detect_images.h, .cpp   runDetectImages
  apps/vc_detect/main.cpp
  tests/data/make_const_yolo.py      生成常量 ONNX 测试模型的脚本
  tests/data/const_yolo/model.onnx, model.json
  tests/test_media.h, .cpp           新增帧构造、带音频 / 字幕的测试视频、流统计
  tests/*_test.cpp                   各组件测试
```

---

### Task 1: 检测基础：Box、NMS、letterbox 与输入张量

**Files:**
- Create: `cpp/src/detect/box.h`, `cpp/src/detect/box.cpp`
- Create: `cpp/src/detect/letterbox.h`, `cpp/src/detect/letterbox.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 加两个源文件）
- Modify: `cpp/tests/test_media.h`, `cpp/tests/test_media.cpp`（帧构造工具）
- Modify: `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/box_test.cpp`, `cpp/tests/letterbox_test.cpp`

**Interfaces:**
- Consumes: `vc::av::FramePtr`、`av::allocFrame()`、`av::allocSws()`、`av::SwsPtr`（`src/io/av_ptr.h`）；`VcError`、`checkAv`（`src/common/vc_error.h`）。
- Produces:
  - `struct Box { int cls; float x1, y1, x2, y2, score; float width() const; float height() const; float area() const; }`
  - `float iou(const Box&, const Box&)`；`std::vector<Box> nmsPerClass(std::vector<Box> boxes, float iouThreshold)`
  - `inline constexpr int kPadValue = 114;`
  - `struct Letterbox { int frameWidth, frameHeight, inputWidth, inputHeight, scaledWidth, scaledHeight; double r; }`
  - `Letterbox makeLetterbox(int frameWidth, int frameHeight, int inputWidth, int inputHeight)`（尺寸非正抛 `std::invalid_argument`）
  - `class InputTensorFiller { explicit InputTensorFiller(const Letterbox&); void fill(const AVFrame& frame, float* out); }`（`out` 为 `3 * inputWidth * inputHeight` 个 float，平面顺序 R、G、B；帧尺寸不符抛 `VcError`）
  - 测试工具（`namespace vc::test`）：`av::FramePtr makeFrame(int w, int h, AVPixelFormat fmt, uint8_t y, uint8_t u, uint8_t v)`、`av::FramePtr makeNoiseFrame(int w, int h, AVPixelFormat fmt, unsigned seed)`、`av::FramePtr deepCopy(const AVFrame&)`

- [ ] **Step 1: Write the failing tests**

`cpp/tests/box_test.cpp`：

```cpp
#include "detect/box.h"

#include <gtest/gtest.h>

#include <vector>

namespace vc {
namespace {

Box box(int cls, float x1, float y1, float x2, float y2, float score) {
  return Box{cls, x1, y1, x2, y2, score};
}

TEST(Box, IouOfOverlappingAndDisjointBoxes) {
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 10, 10, 1), box(0, 5, 0, 15, 10, 1)), 50.0f / 150.0f);
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 10, 10, 1), box(0, 20, 20, 30, 30, 1)), 0.0f);
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 0, 0, 1), box(0, 0, 0, 0, 0, 1)), 0.0f);
}

TEST(Box, NmsKeepsTheBestBoxPerClass) {
  const std::vector<Box> kept = nmsPerClass(
      {box(0, 0, 0, 10, 10, 0.6f), box(0, 1, 0, 11, 10, 0.9f), box(1, 0, 0, 10, 10, 0.5f),
       box(0, 50, 50, 60, 60, 0.3f)},
      0.5f);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
  EXPECT_EQ(kept[1].cls, 1);
  EXPECT_FLOAT_EQ(kept[2].score, 0.3f);
}

TEST(Box, NmsSuppressesAtExactlyTheThreshold) {
  // IoU 1/3 with threshold 1/3 is suppressed, like vc_label.boxes.nms (keeps only IoU < threshold).
  EXPECT_EQ(nmsPerClass({box(0, 0, 0, 10, 10, 0.9f), box(0, 5, 0, 15, 10, 0.8f)}, 50.0f / 150.0f).size(), 1u);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/letterbox_test.cpp`：

```cpp
#include "detect/letterbox.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {

float at(const std::vector<float>& t, const Letterbox& lb, int c, int y, int x) {
  return t[(static_cast<size_t>(c) * lb.inputHeight + y) * lb.inputWidth + x];
}

TEST(Letterbox, ScaleFitsTheLimitingSide) {
  const Letterbox a = makeLetterbox(1920, 1080, 1920, 1088);
  EXPECT_DOUBLE_EQ(a.r, 1.0);
  EXPECT_EQ(a.scaledWidth, 1920);
  EXPECT_EQ(a.scaledHeight, 1080);
  const Letterbox b = makeLetterbox(3840, 2160, 1920, 1088);
  EXPECT_DOUBLE_EQ(b.r, 0.5);
  EXPECT_EQ(b.scaledWidth, 1920);
  EXPECT_EQ(b.scaledHeight, 1080);
  const Letterbox c = makeLetterbox(100, 50, 64, 64);
  EXPECT_DOUBLE_EQ(c.r, 0.64);
  EXPECT_EQ(c.scaledWidth, 64);
  EXPECT_EQ(c.scaledHeight, 32);
  EXPECT_THROW(makeLetterbox(0, 10, 64, 64), std::invalid_argument);
}

TEST(Letterbox, FillsRgbAtTheTopLeftAndPadsWith114) {
  // BT.601 limited-range red
  const auto red = test::makeFrame(128, 64, AV_PIX_FMT_YUV420P, 81, 90, 240);
  const Letterbox lb = makeLetterbox(128, 64, 64, 64);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller filler(lb);
  filler.fill(*red, t.data());
  for (int y : {0, 15, 31}) {
    for (int x : {0, 32, 63}) {
      EXPECT_GT(at(t, lb, 0, y, x), 0.85f);
      EXPECT_LT(at(t, lb, 1, y, x), 0.15f);
      EXPECT_LT(at(t, lb, 2, y, x), 0.15f);
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int y : {32, 50, 63}) EXPECT_FLOAT_EQ(at(t, lb, c, y, 10), 114.0f / 255.0f);
  }
}

TEST(Letterbox, KeepsChannelOrderForBlueNv12) {
  // BT.601 limited-range blue
  const auto blue = test::makeFrame(64, 64, AV_PIX_FMT_NV12, 41, 240, 110);
  const Letterbox lb = makeLetterbox(64, 64, 64, 64);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller(lb).fill(*blue, t.data());
  EXPECT_LT(at(t, lb, 0, 10, 10), 0.15f);
  EXPECT_LT(at(t, lb, 1, 10, 10), 0.15f);
  EXPECT_GT(at(t, lb, 2, 10, 10), 0.85f);
}

TEST(Letterbox, RejectsFramesOfAnotherSize) {
  const auto frame = test::makeFrame(32, 32, AV_PIX_FMT_YUV420P, 128, 128, 128);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller filler(makeLetterbox(64, 64, 64, 64));
  EXPECT_THROW(filler.fill(*frame, t.data()), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/test_media.h` 增加（文件顶部加 `#include <cstdint>` 与 `#include "io/av_ptr.h"`）：

```cpp
// Frame whose luma bytes are all y and chroma bytes u / v (8-bit planar YUV or NV12), limited range.
av::FramePtr makeFrame(int width, int height, AVPixelFormat format, uint8_t y, uint8_t u, uint8_t v);
// Same layout filled with deterministic pseudo-random bytes.
av::FramePtr makeNoiseFrame(int width, int height, AVPixelFormat format, unsigned seed);
// Independent copy with its own writable buffers.
av::FramePtr deepCopy(const AVFrame& frame);
```

`cpp/tests/test_media.cpp` 增加（`#include <random>`、`#include <stdexcept>`，以及 `extern "C" { #include <libavutil/pixdesc.h> }`）：

```cpp
namespace {

av::FramePtr allocated(int width, int height, AVPixelFormat format) {
  av::FramePtr frame = av::allocFrame();
  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->color_range = AVCOL_RANGE_MPEG;
  if (av_frame_get_buffer(frame.get(), 0) < 0) throw std::runtime_error("av_frame_get_buffer failed");
  return frame;
}

// fill(plane, x, y) gives the byte for luma (plane 0) or chroma U (1) / V (2) sample (x, y).
template <typename Fill>
void fillPlanes(AVFrame& frame, Fill fill) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
  const int cw = AV_CEIL_RSHIFT(frame.width, desc->log2_chroma_w);
  const int ch = AV_CEIL_RSHIFT(frame.height, desc->log2_chroma_h);
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) frame.data[0][y * frame.linesize[0] + x] = fill(0, x, y);
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      if (frame.format == AV_PIX_FMT_NV12) {
        frame.data[1][y * frame.linesize[1] + 2 * x] = fill(1, x, y);
        frame.data[1][y * frame.linesize[1] + 2 * x + 1] = fill(2, x, y);
      } else {
        frame.data[1][y * frame.linesize[1] + x] = fill(1, x, y);
        frame.data[2][y * frame.linesize[2] + x] = fill(2, x, y);
      }
    }
  }
}

}  // namespace

av::FramePtr makeFrame(int width, int height, AVPixelFormat format, uint8_t y, uint8_t u, uint8_t v) {
  av::FramePtr frame = allocated(width, height, format);
  const uint8_t values[3] = {y, u, v};
  fillPlanes(*frame, [&](int plane, int, int) { return values[plane]; });
  return frame;
}

av::FramePtr makeNoiseFrame(int width, int height, AVPixelFormat format, unsigned seed) {
  av::FramePtr frame = allocated(width, height, format);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  fillPlanes(*frame, [&](int, int, int) { return static_cast<uint8_t>(byte(rng)); });
  return frame;
}

av::FramePtr deepCopy(const AVFrame& source) {
  av::FramePtr frame = allocated(source.width, source.height, static_cast<AVPixelFormat>(source.format));
  if (av_frame_copy(frame.get(), &source) < 0 || av_frame_copy_props(frame.get(), &source) < 0) {
    throw std::runtime_error("av_frame_copy failed");
  }
  return frame;
}
```

`cpp/tests/CMakeLists.txt`：`add_executable(vc_tests …)` 中加入 `box_test.cpp` 与 `letterbox_test.cpp`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`
Expected: 编译失败 — `detect/box.h` / `detect/letterbox.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/src/detect/box.h`：

```cpp
#pragma once

#include <vector>

namespace vc {

// Axis-aligned box in frame pixels; cls indexes the detector's class list.
struct Box {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  float score = 0;

  float width() const { return x2 > x1 ? x2 - x1 : 0.0f; }
  float height() const { return y2 > y1 ? y2 - y1 : 0.0f; }
  float area() const { return width() * height(); }
};

float iou(const Box& a, const Box& b);

// Greedy non-maximum suppression within each class, highest score first (ties keep input
// order). A box is suppressed when its IoU with a kept box of its class is >= iouThreshold.
std::vector<Box> nmsPerClass(std::vector<Box> boxes, float iouThreshold);

}  // namespace vc
```

`cpp/src/detect/box.cpp`：

```cpp
#include "detect/box.h"

#include <algorithm>

namespace vc {

float iou(const Box& a, const Box& b) {
  const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0.0f || h <= 0.0f) return 0.0f;
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Box> nmsPerClass(std::vector<Box> boxes, float iouThreshold) {
  std::stable_sort(boxes.begin(), boxes.end(),
                   [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> kept;
  for (const Box& b : boxes) {
    const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const Box& k) {
      return k.cls == b.cls && iou(k, b) >= iouThreshold;
    });
    if (!suppressed) kept.push_back(b);
  }
  return kept;
}

}  // namespace vc
```

`cpp/src/detect/letterbox.h`：

```cpp
#pragma once

#include <memory>

struct AVFrame;

namespace vc {

inline constexpr int kPadValue = 114;

// Frame -> model input mapping (proxy-validation spec §4.1): the frame is scaled by r, placed at
// the top-left corner and the rest of the input is filled with kPadValue.
struct Letterbox {
  int frameWidth = 0;
  int frameHeight = 0;
  int inputWidth = 0;
  int inputHeight = 0;
  int scaledWidth = 0;   // round(frameWidth * r)
  int scaledHeight = 0;  // round(frameHeight * r)
  double r = 1.0;        // min(inputWidth / frameWidth, inputHeight / frameHeight)
};

// Throws std::invalid_argument for non-positive sizes.
Letterbox makeLetterbox(int frameWidth, int frameHeight, int inputWidth, int inputHeight);

// Converts 8-bit YUV frames of one size into the model's [1, 3, H, W] float input (RGB / 255).
class InputTensorFiller {
 public:
  explicit InputTensorFiller(const Letterbox& letterbox);
  ~InputTensorFiller();
  InputTensorFiller(const InputTensorFiller&) = delete;
  InputTensorFiller& operator=(const InputTensorFiller&) = delete;

  // out must hold 3 * inputWidth * inputHeight floats (planes R, G, B). Throws VcError when the
  // frame size differs from the letterbox's or the conversion fails.
  void fill(const AVFrame& frame, float* out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/detect/letterbox.cpp`：

```cpp
#include "detect/letterbox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {

Letterbox makeLetterbox(int frameWidth, int frameHeight, int inputWidth, int inputHeight) {
  if (frameWidth <= 0 || frameHeight <= 0 || inputWidth <= 0 || inputHeight <= 0) {
    throw std::invalid_argument("letterbox sizes must be positive");
  }
  Letterbox lb;
  lb.frameWidth = frameWidth;
  lb.frameHeight = frameHeight;
  lb.inputWidth = inputWidth;
  lb.inputHeight = inputHeight;
  lb.r = std::min(static_cast<double>(inputWidth) / frameWidth,
                  static_cast<double>(inputHeight) / frameHeight);
  lb.scaledWidth = std::min(inputWidth, static_cast<int>(std::lround(frameWidth * lb.r)));
  lb.scaledHeight = std::min(inputHeight, static_cast<int>(std::lround(frameHeight * lb.r)));
  return lb;
}

struct InputTensorFiller::Impl {
  Letterbox lb;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr rgb = av::allocFrame();
};

InputTensorFiller::InputTensorFiller(const Letterbox& letterbox)
    : impl_(std::make_unique<Impl>()) {
  impl_->lb = letterbox;
  impl_->sws->flags = SWS_BILINEAR;
}

InputTensorFiller::~InputTensorFiller() = default;

void InputTensorFiller::fill(const AVFrame& frame, float* out) {
  Impl& s = *impl_;
  const Letterbox& lb = s.lb;
  if (frame.width != lb.frameWidth || frame.height != lb.frameHeight) {
    throw VcError("frame is " + std::to_string(frame.width) + "x" + std::to_string(frame.height) +
                  ", the detector was set up for " + std::to_string(lb.frameWidth) + "x" +
                  std::to_string(lb.frameHeight));
  }
  av_frame_unref(s.rgb.get());
  s.rgb->format = AV_PIX_FMT_RGB24;
  s.rgb->width = lb.scaledWidth;
  s.rgb->height = lb.scaledHeight;
  checkAv(sws_scale_frame(s.sws.get(), s.rgb.get(), &frame), "convert frame to RGB");

  const size_t plane = static_cast<size_t>(lb.inputWidth) * lb.inputHeight;
  std::fill(out, out + 3 * plane, static_cast<float>(kPadValue) / 255.0f);
  for (int y = 0; y < lb.scaledHeight; ++y) {
    const uint8_t* row = s.rgb->data[0] + static_cast<ptrdiff_t>(y) * s.rgb->linesize[0];
    for (int x = 0; x < lb.scaledWidth; ++x) {
      const size_t at = static_cast<size_t>(y) * lb.inputWidth + x;
      out[at] = row[3 * x] / 255.0f;
      out[plane + at] = row[3 * x + 1] / 255.0f;
      out[2 * plane + at] = row[3 * x + 2] / 255.0f;
    }
  }
}

}  // namespace vc
```

`cpp/CMakeLists.txt`：`add_library(vc_core STATIC …)` 中加入 `src/detect/box.cpp` 与 `src/detect/letterbox.cpp`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*'`
Expected: `[  PASSED  ] 7 tests.`，编译无警告。

- [ ] **Step 5: Commit**

```bash
git add cpp/src/detect cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/test_media.h cpp/tests/test_media.cpp cpp/tests/box_test.cpp cpp/tests/letterbox_test.cpp
git commit -m "feat: add box NMS, letterbox geometry and RGB input tensor filler"
```

---

### Task 2: 第三方接入、model.json、YOLOv8 解码与 FakeDetector

**Files:**
- Create: `cpp/cmake/ThirdParty.cmake`
- Create: `cpp/src/detect/detector.h`
- Create: `cpp/src/detect/model_info.h`, `cpp/src/detect/model_info.cpp`
- Create: `cpp/src/detect/yolo_decode.h`, `cpp/src/detect/yolo_decode.cpp`
- Create: `cpp/src/detect/fake_detector.h`, `cpp/src/detect/fake_detector.cpp`
- Modify: `cpp/CMakeLists.txt`, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/model_info_test.cpp`, `cpp/tests/yolo_decode_test.cpp`, `cpp/tests/fake_detector_test.cpp`

**Interfaces:**
- Consumes: Task 1 `Box`、`nmsPerClass`、`Letterbox`、`makeLetterbox`、`kPadValue`；`VideoFrame`（`src/io/video_reader.h`：`index`、`ptsMs`、`av`）；`VcError`。
- Produces:
  - CMake 目标 `vc_nlohmann_json`（INTERFACE）、`vc_onnxruntime`（IMPORTED SHARED）、`vc_detect_lib`（STATIC，链接 `vc_core`、`vc_nlohmann_json`）
  - `class Detector { virtual std::vector<Box> detect(const VideoFrame&) = 0; virtual const std::vector<std::string>& classes() const = 0; virtual std::string source() const = 0; virtual std::string backend() const; }`
  - `struct ModelInfo { std::string name; std::vector<std::string> classes; int inputWidth; int inputHeight; }`；`ModelInfo parseModelInfo(const std::string& text, const std::string& origin)`；`ModelInfo readModelInfo(const std::filesystem::path&)`
  - `struct DecodeParams { float scoreThreshold = 0.15f; float nmsIou = 0.5f; }`；`std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors, const Letterbox& lb, const DecodeParams& params)`
  - `class FakeDetector : public Detector { explicit FakeDetector(const std::filesystem::path& jsonl, std::vector<std::string> classes = {"face", "plate"}); const std::vector<int64_t>& calls() const; }`；`source()` 为 `"fake"`。一行 `{"frame": N, "fail": true}` 使 `detect()` 在第 N 帧抛 `VcError`（供测试流水线失败路径）。

- [ ] **Step 1: Write the failing tests**

`cpp/tests/model_info_test.cpp`：

```cpp
#include "detect/model_info.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

const std::string kValid =
    R"({"name":"m","classes":["face","plate"],"input":{"width":1920,"height":1088,"color":"RGB",)"
    R"("scale":0.00392156862745098,"letterbox":"top_left","pad_value":114},"output_format":"yolov8",)"
    R"("train_manifest_sha256":"","metrics":{"val_map50":0.5}})";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  text.replace(text.find(from), from.size(), to);
  return text;
}

TEST(ModelInfo, ParsesTheVcTrainContract) {
  const ModelInfo m = parseModelInfo(kValid, "model.json");
  EXPECT_EQ(m.name, "m");
  EXPECT_EQ(m.classes, (std::vector<std::string>{"face", "plate"}));
  EXPECT_EQ(m.inputWidth, 1920);
  EXPECT_EQ(m.inputHeight, 1088);
}

TEST(ModelInfo, RejectsUnsupportedValues) {
  const std::vector<std::string> bad = {
      replaced(kValid, R"("RGB")", R"("BGR")"),
      replaced(kValid, R"("top_left")", R"("center")"),
      replaced(kValid, R"("yolov8")", R"("yolov5")"),
      replaced(kValid, "1088", "1080"),
      replaced(kValid, R"("pad_value":114)", R"("pad_value":0)"),
      replaced(kValid, "0.00392156862745098", "1.0"),
      replaced(kValid, R"(["face","plate"])", "[]"),
      R"({"classes":["face"],"output_format":"yolov8"})",
      "{",
      "[]",
  };
  for (const std::string& text : bad) {
    try {
      parseModelInfo(text, "model.json");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("model.json: ", 0), 0u) << e.what();
    }
  }
}

}  // namespace
}  // namespace vc
```

`cpp/tests/yolo_decode_test.cpp`：

```cpp
#include "detect/yolo_decode.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace vc {
namespace {

// columns: (cx, cy, w, h, face, plate) -> row-major [6][N]
std::vector<float> pred(const std::vector<std::array<float, 6>>& columns) {
  const size_t n = columns.size();
  std::vector<float> out(6 * n);
  for (size_t a = 0; a < n; ++a) {
    for (size_t row = 0; row < 6; ++row) out[row * n + a] = columns[a][row];
  }
  return out;
}

// Same input as py/tests/test_train_geometry.py::test_decode_maps_back_by_r_thresholds_and_suppresses_per_class
TEST(YoloDecode, MapsBackByRThresholdsAndSuppressesPerClass) {
  const std::vector<float> p = pred({{32, 16, 8, 4, 0.9f, 0.0f},
                                     {100, 100, 10, 10, 0.1f, 0.05f},
                                     {33, 16, 8, 4, 0.6f, 0.0f},
                                     {32, 16, 8, 4, 0.0f, 0.5f}});
  const std::vector<Box> boxes = decodeYolov8(p.data(), 2, 4, makeLetterbox(1000, 1000, 500, 500), {});
  ASSERT_EQ(boxes.size(), 2u);
  EXPECT_EQ(boxes[0].cls, 0);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9f);
  EXPECT_NEAR(boxes[0].x1, 56, 1e-3);
  EXPECT_NEAR(boxes[0].y1, 24, 1e-3);
  EXPECT_NEAR(boxes[0].x2, 72, 1e-3);
  EXPECT_NEAR(boxes[0].y2, 40, 1e-3);
  EXPECT_EQ(boxes[1].cls, 1);
  EXPECT_FLOAT_EQ(boxes[1].score, 0.5f);
}

TEST(YoloDecode, ClipsToTheFrame) {
  const std::vector<float> p = pred({{2, 2, 10, 10, 0.8f, 0.0f}});
  const std::vector<Box> boxes = decodeYolov8(p.data(), 2, 1, makeLetterbox(20, 20, 20, 20), {});
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_FLOAT_EQ(boxes[0].x1, 0);
  EXPECT_FLOAT_EQ(boxes[0].y1, 0);
  EXPECT_FLOAT_EQ(boxes[0].x2, 7);
  EXPECT_FLOAT_EQ(boxes[0].y2, 7);
}

TEST(YoloDecode, KeepsScoresEqualToTheThreshold) {
  const std::vector<float> p = pred({{10, 10, 4, 4, 0.15f, 0.0f}});
  EXPECT_EQ(decodeYolov8(p.data(), 2, 1, makeLetterbox(20, 20, 20, 20), {}).size(), 1u);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/fake_detector_test.cpp`：

```cpp
#include "detect/fake_detector.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path writeFile(const std::string& name, const std::string& text) {
  const fs::path path = test::tmpDir() / name;
  std::ofstream(path) << text;
  return path;
}

VideoFrame frameNumber(int64_t index) {
  VideoFrame frame;
  frame.index = index;
  return frame;
}

TEST(FakeDetector, ReplaysBoxesByFrame) {
  const fs::path path = writeFile(
      "fake_ok.jsonl",
      R"({"video":"v.mp4","frame":0,"pts_ms":0,"cls":"face","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9,"src":"t"})"
      "\n\n"
      R"({"video":"v.mp4","frame":2,"pts_ms":66.7,"cls":"plate","x1":10,"y1":20,"x2":30,"y2":40,"score":0.5,"src":"t"})"
      "\n"
      R"({"video":"v.mp4","frame":2,"pts_ms":66.7,"cls":"face","x1":5,"y1":5,"x2":9,"y2":9,"score":0.7,"src":"t"})"
      "\n");
  FakeDetector detector(path);
  EXPECT_EQ(detector.source(), "fake");
  EXPECT_EQ(detector.classes(), (std::vector<std::string>{"face", "plate"}));
  const std::vector<Box> first = detector.detect(frameNumber(0));
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first[0].cls, 0);
  EXPECT_FLOAT_EQ(first[0].x2, 3);
  EXPECT_FLOAT_EQ(first[0].score, 0.9f);
  EXPECT_TRUE(detector.detect(frameNumber(1)).empty());
  const std::vector<Box> third = detector.detect(frameNumber(2));
  ASSERT_EQ(third.size(), 2u);
  EXPECT_EQ(third[0].cls, 1);
  EXPECT_EQ(detector.calls(), (std::vector<int64_t>{0, 1, 2}));
}

TEST(FakeDetector, RejectsUnknownClassesAndBadLines) {
  const fs::path unknown = writeFile(
      "fake_unknown.jsonl",
      R"({"frame":0,"cls":"car","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9})" "\n");
  EXPECT_THROW(FakeDetector{unknown}, VcError);
  const fs::path broken = writeFile(
      "fake_broken.jsonl",
      R"({"frame":0,"cls":"face","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9})" "\n{oops\n");
  try {
    FakeDetector detector(broken);
    FAIL() << "expected VcError";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
  }
  EXPECT_THROW(FakeDetector{test::tmpDir() / "missing.jsonl"}, VcError);
}

TEST(FakeDetector, FailsOnRequestedFrames) {
  const fs::path path = writeFile("fake_fail.jsonl", R"({"frame":3,"fail":true})" "\n");
  FakeDetector detector(path);
  EXPECT_TRUE(detector.detect(frameNumber(2)).empty());
  EXPECT_THROW(detector.detect(frameNumber(3)), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入三个测试文件；`target_link_libraries(vc_tests PRIVATE vc_extract_lib vc_detect_lib vc_core GTest::gtest_main)`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`
Expected: 配置失败（`vc_detect_lib` 未定义）或编译失败（`detect/model_info.h` not found）

- [ ] **Step 3: Write minimal implementation**

`cpp/cmake/ThirdParty.cmake`：

```cmake
# nlohmann/json (header only) and the ONNX Runtime prebuilt package, unpacked once into the cache
# shared by every checkout (see cpp/README.md). Nothing is downloaded at configure time.
#
# Defines:
#   vc_nlohmann_json  INTERFACE target with the json.hpp include directory
#   vc_onnxruntime    IMPORTED SHARED target for libonnxruntime
#
# The package's own lib/cmake/onnxruntime config points at a non-existent include/onnxruntime
# directory, so the imported target is defined here instead.

set(VC_THIRD_PARTY_DIR "$ENV{HOME}/.cache/video-compliance/third_party"
    CACHE PATH "Directory holding the unpacked third-party packages")
set(VC_NLOHMANN_JSON_INCLUDE "${VC_THIRD_PARTY_DIR}/nlohmann_json-3.12.0/include"
    CACHE PATH "Directory containing nlohmann/json.hpp")
set(VC_ONNXRUNTIME_ROOT "${VC_THIRD_PARTY_DIR}/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "Unpacked ONNX Runtime package (include/, lib/)")

if(NOT EXISTS "${VC_NLOHMANN_JSON_INCLUDE}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json.hpp not found under VC_NLOHMANN_JSON_INCLUDE=${VC_NLOHMANN_JSON_INCLUDE} (see cpp/README.md)")
endif()
add_library(vc_nlohmann_json INTERFACE)
target_include_directories(vc_nlohmann_json SYSTEM INTERFACE "${VC_NLOHMANN_JSON_INCLUDE}")

find_library(VC_ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${VC_ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT VC_ONNXRUNTIME_LIBRARY OR NOT EXISTS "${VC_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found under VC_ONNXRUNTIME_ROOT=${VC_ONNXRUNTIME_ROOT} (see cpp/README.md)")
endif()
add_library(vc_onnxruntime SHARED IMPORTED)
set_target_properties(vc_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${VC_ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include")
# libonnxruntime's install name is @rpath/libonnxruntime.1.dylib.
list(APPEND CMAKE_BUILD_RPATH "${VC_ONNXRUNTIME_ROOT}/lib")
```

`cpp/CMakeLists.txt`：在 `find_package(FFmpeg REQUIRED)` 之后加 `include(ThirdParty)`；在 `vc_extract_lib` 之前加：

```cmake
add_library(vc_detect_lib STATIC
  src/detect/fake_detector.cpp
  src/detect/model_info.cpp
  src/detect/yolo_decode.cpp
)
target_link_libraries(vc_detect_lib PUBLIC vc_core vc_nlohmann_json)
```

`cpp/src/detect/detector.h`：

```cpp
#pragma once

#include <string>
#include <vector>

#include "detect/box.h"
#include "io/video_reader.h"

namespace vc {

// Finds faces and plates in one decoded frame (spec §3.3).
class Detector {
 public:
  virtual ~Detector() = default;

  // Boxes in frame pixels; Box::cls indexes classes(). Throws VcError.
  virtual std::vector<Box> detect(const VideoFrame& frame) = 0;
  virtual const std::vector<std::string>& classes() const = 0;
  // The `src` written to detections.jsonl ("student", "fake").
  virtual std::string source() const = 0;
  // Execution backend for reports ("CoreML", "CPU"); empty when not applicable.
  virtual std::string backend() const { return {}; }
};

}  // namespace vc
```

`cpp/src/detect/model_info.h`：

```cpp
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc {

// The model.json written by `vc_train export` (proxy-validation spec §4.4).
struct ModelInfo {
  std::string name;
  std::vector<std::string> classes;
  int inputWidth = 0;
  int inputHeight = 0;
};

// Throws VcError("<origin>: …") for invalid JSON, missing fields or values this pipeline does not
// implement: color must be "RGB", letterbox "top_left", pad_value 114, scale 1/255,
// output_format "yolov8", and the input size positive multiples of 32.
ModelInfo parseModelInfo(const std::string& text, const std::string& origin);
ModelInfo readModelInfo(const std::filesystem::path& path);

}  // namespace vc
```

`cpp/src/detect/model_info.cpp`：

```cpp
#include "detect/model_info.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/letterbox.h"

namespace vc {

ModelInfo parseModelInfo(const std::string& text, const std::string& origin) {
  const auto fail = [&](const std::string& why) { return VcError(origin + ": " + why); };
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw fail(e.what());
  }
  if (!j.is_object()) throw fail("the root must be a JSON object");
  try {
    ModelInfo m;
    m.name = j.value("name", std::string());
    if (j.at("output_format").get<std::string>() != "yolov8") throw fail("output_format must be \"yolov8\"");
    m.classes = j.at("classes").get<std::vector<std::string>>();
    if (m.classes.empty()) throw fail("classes is empty");
    const nlohmann::json& input = j.at("input");
    m.inputWidth = input.at("width").get<int>();
    m.inputHeight = input.at("height").get<int>();
    if (m.inputWidth <= 0 || m.inputHeight <= 0 || m.inputWidth % 32 != 0 || m.inputHeight % 32 != 0) {
      throw fail("input width and height must be positive multiples of 32");
    }
    if (input.at("color").get<std::string>() != "RGB") throw fail("input.color must be \"RGB\"");
    if (input.at("letterbox").get<std::string>() != "top_left") throw fail("input.letterbox must be \"top_left\"");
    if (input.at("pad_value").get<int>() != kPadValue) throw fail("input.pad_value must be 114");
    if (std::abs(input.at("scale").get<double>() - 1.0 / 255.0) > 1e-9) throw fail("input.scale must be 1/255");
    return m;
  } catch (const nlohmann::json::exception& e) {
    throw fail(e.what());
  }
}

ModelInfo readModelInfo(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parseModelInfo(text.str(), path.string());
}

}  // namespace vc
```

`cpp/src/detect/yolo_decode.h`：

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include "detect/box.h"
#include "detect/letterbox.h"

namespace vc {

struct DecodeParams {
  float scoreThreshold = 0.15f;
  float nmsIou = 0.5f;
};

// Decodes a raw YOLOv8 output laid out as [4 + numClasses][numAnchors] (the [1, 4 + nc, N] ONNX
// output): rows cx, cy, w, h in model-input pixels, then one score row per class. Each anchor
// takes its best class; anchors scoring below the threshold are dropped; boxes are divided by
// lb.r, clipped to the frame and suppressed per class. Same rules as vc_train.geometry.decode_yolov8.
std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors,
                              const Letterbox& lb, const DecodeParams& params);

}  // namespace vc
```

`cpp/src/detect/yolo_decode.cpp`：

```cpp
#include "detect/yolo_decode.h"

#include <algorithm>

namespace vc {

std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors,
                              const Letterbox& lb, const DecodeParams& params) {
  const auto value = [&](int row, int64_t anchor) {
    return pred[static_cast<int64_t>(row) * numAnchors + anchor];
  };
  const float inv = static_cast<float>(1.0 / lb.r);
  const float fw = static_cast<float>(lb.frameWidth);
  const float fh = static_cast<float>(lb.frameHeight);
  std::vector<Box> boxes;
  for (int64_t a = 0; a < numAnchors; ++a) {
    int best = 0;
    float score = value(4, a);
    for (int c = 1; c < numClasses; ++c) {
      const float s = value(4 + c, a);
      if (s > score) {
        score = s;
        best = c;
      }
    }
    if (score < params.scoreThreshold) continue;
    const float cx = value(0, a);
    const float cy = value(1, a);
    const float w = value(2, a);
    const float h = value(3, a);
    Box b;
    b.cls = best;
    b.score = score;
    b.x1 = std::clamp((cx - w / 2) * inv, 0.0f, fw);
    b.y1 = std::clamp((cy - h / 2) * inv, 0.0f, fh);
    b.x2 = std::clamp((cx + w / 2) * inv, 0.0f, fw);
    b.y2 = std::clamp((cy + h / 2) * inv, 0.0f, fh);
    boxes.push_back(b);
  }
  return nmsPerClass(std::move(boxes), params.nmsIou);
}

}  // namespace vc
```

`cpp/src/detect/fake_detector.h`：

```cpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "detect/detector.h"

namespace vc {

// Replays detections.jsonl records (spec §3.2) by frame number, for tests and for checking the
// pipeline without a model. A record {"frame": N, "fail": true} makes detect() throw for frame N.
class FakeDetector : public Detector {
 public:
  // Throws VcError naming the file and line for unreadable files, bad JSON or unknown classes.
  explicit FakeDetector(const std::filesystem::path& jsonl,
                        std::vector<std::string> classes = {"face", "plate"});

  std::vector<Box> detect(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override { return classes_; }
  std::string source() const override { return "fake"; }

  // Frame numbers detect() was called for, in call order.
  const std::vector<int64_t>& calls() const { return calls_; }

 private:
  std::vector<std::string> classes_;
  std::map<int64_t, std::vector<Box>> byFrame_;
  std::set<int64_t> failing_;
  std::vector<int64_t> calls_;
};

}  // namespace vc
```

`cpp/src/detect/fake_detector.cpp`：

```cpp
#include "detect/fake_detector.h"

#include <algorithm>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {

FakeDetector::FakeDetector(const std::filesystem::path& jsonl, std::vector<std::string> classes)
    : classes_(std::move(classes)) {
  std::ifstream in(jsonl);
  if (!in) throw VcError("cannot read " + jsonl.string());
  std::string line;
  int lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string where = jsonl.string() + " line " + std::to_string(lineNo);
    try {
      const nlohmann::json j = nlohmann::json::parse(line);
      const int64_t frame = j.at("frame").get<int64_t>();
      if (j.value("fail", false)) {
        failing_.insert(frame);
        continue;
      }
      const std::string cls = j.at("cls").get<std::string>();
      const auto it = std::find(classes_.begin(), classes_.end(), cls);
      if (it == classes_.end()) throw VcError(where + ": unknown class '" + cls + "'");
      Box b;
      b.cls = static_cast<int>(it - classes_.begin());
      b.x1 = j.at("x1").get<float>();
      b.y1 = j.at("y1").get<float>();
      b.x2 = j.at("x2").get<float>();
      b.y2 = j.at("y2").get<float>();
      b.score = j.value("score", 1.0f);
      byFrame_[frame].push_back(b);
    } catch (const nlohmann::json::exception& e) {
      throw VcError(where + ": " + e.what());
    }
  }
}

std::vector<Box> FakeDetector::detect(const VideoFrame& frame) {
  calls_.push_back(frame.index);
  if (failing_.count(frame.index) != 0) {
    throw VcError("fake detector failure at frame " + std::to_string(frame.index));
  }
  const auto it = byFrame_.find(frame.index);
  return it == byFrame_.end() ? std::vector<Box>{} : it->second;
}

}  // namespace vc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='ModelInfo.*:YoloDecode.*:FakeDetector.*'`
Expected: `[  PASSED  ] 8 tests.`，无编译警告。

Also run: `./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*'`
Expected: `[  PASSED  ] 7 tests.`

- [ ] **Step 5: Commit**

```bash
git add cpp/cmake/ThirdParty.cmake cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/src/detect cpp/tests/model_info_test.cpp cpp/tests/yolo_decode_test.cpp cpp/tests/fake_detector_test.cpp
git commit -m "feat: parse model.json, decode YOLOv8 output and add FakeDetector"
```

---

### Task 3: OnnxYoloDetector（onnxruntime，CoreML 优先、CPU 兜底）

**Files:**
- Create: `cpp/src/detect/onnx_yolo_detector.h`, `cpp/src/detect/onnx_yolo_detector.cpp`
- Create: `cpp/tests/data/make_const_yolo.py`
- Create: `cpp/tests/data/const_yolo/model.json`
- Create（由脚本生成后提交）: `cpp/tests/data/const_yolo/model.onnx`
- Modify: `cpp/CMakeLists.txt`, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/onnx_yolo_detector_test.cpp`

**Interfaces:**
- Consumes: Task 1 `Letterbox`、`makeLetterbox`、`InputTensorFiller`；Task 2 `Detector`、`ModelInfo`、`readModelInfo`、`DecodeParams`、`decodeYolov8`；测试工具 `test::makeFrame`。
- Produces:
  - `enum class ExecutionProvider { CoreML, Cpu };`；`ExecutionProvider parseExecutionProvider(const std::string&)`（`"coreml"` | `"cpu"`，否则 `std::invalid_argument`）
  - `class OnnxYoloDetector : public Detector { OnnxYoloDetector(const std::filesystem::path& modelDir, ExecutionProvider provider, const DecodeParams& params, int frameWidth, int frameHeight); }`；`source()` 为 `"student"`，`backend()` 为 `"CoreML"` 或 `"CPU"`；`modelDir` 下需有 `model.onnx` 与 `model.json`，两者输入形状或类别数不一致时构造抛 `VcError`。

- [ ] **Step 1: Generate the constant test model and write the failing test**

`cpp/tests/data/make_const_yolo.py`：

```python
"""Writes <out>/model.onnx: images [1,3,64,64] -> output0 [1,6,4] = fixed YOLOv8 columns + 0 * mean(images).

Run once with the repo venv (onnx is installed there); the output is committed:
    .venv/bin/python cpp/tests/data/make_const_yolo.py cpp/tests/data/const_yolo
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COLUMNS = [  # cx, cy, w, h, face, plate in 64x64 input pixels
    (16, 16, 8, 8, 0.9, 0.0),
    (17, 16, 8, 8, 0.6, 0.0),  # overlaps the first face, suppressed by NMS
    (40, 40, 16, 8, 0.0, 0.7),
    (50, 10, 4, 4, 0.05, 0.05),  # below the 0.15 threshold
]


def main(out_dir: Path) -> None:
    boxes = np.array(COLUMNS, np.float32).T[None]  # [1, 6, 4]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["boxes", "zeroed"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "const_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])],
        initializer=[numpy_helper.from_array(boxes, "boxes"), numpy_helper.from_array(np.zeros(1, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

Run: `.venv/bin/python -I cpp/tests/data/make_const_yolo.py cpp/tests/data/const_yolo`
Expected: 生成 `cpp/tests/data/const_yolo/model.onnx`（约 1KB）。

`cpp/tests/data/const_yolo/model.json`：

```json
{
  "name": "const_yolo",
  "classes": ["face", "plate"],
  "input": {"width": 64, "height": 64, "color": "RGB", "scale": 0.00392156862745098, "letterbox": "top_left", "pad_value": 114},
  "output_format": "yolov8",
  "train_manifest_sha256": "",
  "metrics": {"val_map50": 0.0}
}
```

`cpp/tests/onnx_yolo_detector_test.cpp`：

```cpp
#include "detect/onnx_yolo_detector.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path constModel() { return fs::path(VC_TEST_DATA_DIR) / "const_yolo"; }

VideoFrame view(const AVFrame* frame) {
  VideoFrame v;
  v.av = frame;
  return v;
}

std::string readText(const fs::path& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// A copy of the constant model whose model.json has `from` replaced by `to`.
fs::path variant(const std::string& name, const std::string& from, const std::string& to) {
  const fs::path dir = test::tmpDir() / name;
  fs::create_directories(dir);
  fs::copy_file(constModel() / "model.onnx", dir / "model.onnx", fs::copy_options::overwrite_existing);
  std::string json = readText(constModel() / "model.json");
  json.replace(json.find(from), from.size(), to);
  std::ofstream(dir / "model.json") << json;
  return dir;
}

void expectConstBoxes(const std::vector<Box>& boxes, float tolerance) {
  ASSERT_EQ(boxes.size(), 2u);
  // input 64x64, frame 128x128 -> r = 0.5
  EXPECT_EQ(boxes[0].cls, 0);
  EXPECT_NEAR(boxes[0].score, 0.9, tolerance);
  EXPECT_NEAR(boxes[0].x1, 24, tolerance * 100);
  EXPECT_NEAR(boxes[0].y1, 24, tolerance * 100);
  EXPECT_NEAR(boxes[0].x2, 40, tolerance * 100);
  EXPECT_NEAR(boxes[0].y2, 40, tolerance * 100);
  EXPECT_EQ(boxes[1].cls, 1);
  EXPECT_NEAR(boxes[1].x1, 64, tolerance * 100);
  EXPECT_NEAR(boxes[1].y1, 72, tolerance * 100);
  EXPECT_NEAR(boxes[1].x2, 96, tolerance * 100);
  EXPECT_NEAR(boxes[1].y2, 88, tolerance * 100);
}

TEST(OnnxYoloDetector, DecodesTheConstantModelOnCpu) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  EXPECT_EQ(detector.backend(), "CPU");
  EXPECT_EQ(detector.source(), "student");
  EXPECT_EQ(detector.classes(), (std::vector<std::string>{"face", "plate"}));
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
}

TEST(OnnxYoloDetector, CoreMlGivesTheSameBoxes) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
}

TEST(OnnxYoloDetector, RejectsAFrameOfAnotherSize) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  const auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 128, 128, 128);
  EXPECT_THROW(detector.detect(view(frame.get())), VcError);
}

TEST(OnnxYoloDetector, RejectsModelJsonThatDisagreesWithTheOnnx) {
  const fs::path wrongSize = variant("const_yolo_size", R"("width": 64)", R"("width": 128)");
  EXPECT_THROW(OnnxYoloDetector(wrongSize, ExecutionProvider::Cpu, DecodeParams{}, 128, 128), VcError);
  const fs::path wrongClasses =
      variant("const_yolo_classes", R"(["face", "plate"])", R"(["face", "plate", "car"])");
  EXPECT_THROW(OnnxYoloDetector(wrongClasses, ExecutionProvider::Cpu, DecodeParams{}, 128, 128), VcError);
}

TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_THROW(parseExecutionProvider("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `onnx_yolo_detector_test.cpp`；`target_compile_definitions` 增加 `VC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"`。

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `detect/onnx_yolo_detector.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/CMakeLists.txt`：`vc_detect_lib` 加入 `src/detect/onnx_yolo_detector.cpp`，并改为 `target_link_libraries(vc_detect_lib PUBLIC vc_core vc_nlohmann_json vc_onnxruntime)`。

`cpp/src/detect/onnx_yolo_detector.h`：

```cpp
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "detect/yolo_decode.h"

namespace vc {

enum class ExecutionProvider { CoreML, Cpu };

// "coreml" | "cpu"; throws std::invalid_argument.
ExecutionProvider parseExecutionProvider(const std::string& text);

// YOLOv8 student model (model.onnx + model.json from `vc_train export`) on onnxruntime.
class OnnxYoloDetector : public Detector {
 public:
  // Frames passed to detect() must be frameWidth x frameHeight. CoreML falls back to the CPU
  // provider (with a message on stderr) when it cannot be attached. Throws VcError when the files
  // are missing or invalid, when model.json disagrees with the ONNX input shape or class count, or
  // when onnxruntime fails.
  OnnxYoloDetector(const std::filesystem::path& modelDir, ExecutionProvider provider,
                   const DecodeParams& params, int frameWidth, int frameHeight);
  ~OnnxYoloDetector() override;
  OnnxYoloDetector(const OnnxYoloDetector&) = delete;
  OnnxYoloDetector& operator=(const OnnxYoloDetector&) = delete;

  std::vector<Box> detect(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override;
  std::string source() const override { return "student"; }
  std::string backend() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/detect/onnx_yolo_detector.cpp`：

```cpp
#include "detect/onnx_yolo_detector.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "detect/letterbox.h"
#include "detect/model_info.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

Ort::Env& ortEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vc");
  return env;
}

std::string shapeText(const std::vector<int64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
  return text + "]";
}

}  // namespace

ExecutionProvider parseExecutionProvider(const std::string& text) {
  if (text == "coreml") return ExecutionProvider::CoreML;
  if (text == "cpu") return ExecutionProvider::Cpu;
  throw std::invalid_argument("unknown execution provider '" + text + "' (coreml|cpu)");
}

struct OnnxYoloDetector::Impl {
  ModelInfo info;
  DecodeParams params;
  Letterbox lb;
  std::unique_ptr<InputTensorFiller> filler;
  Ort::Session session{nullptr};
  std::string inputName;
  std::string outputName;
  std::string backend = "CPU";
  int64_t anchors = 0;
  std::vector<float> input;

  void checkShapes(const fs::path& onnx) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
      throw VcError(onnx.string() + ": expected one input and at least one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
    const std::vector<int64_t> inShape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> wanted{1, 3, info.inputHeight, info.inputWidth};
    if (inShape != wanted) {
      throw VcError(onnx.string() + ": input shape " + shapeText(inShape) +
                    " does not match model.json input " + shapeText(wanted));
    }
    const std::vector<int64_t> outShape =
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const int64_t rows = 4 + static_cast<int64_t>(info.classes.size());
    if (outShape.size() != 3 || outShape[0] != 1 || outShape[1] != rows || outShape[2] <= 0) {
      throw VcError(onnx.string() + ": output shape " + shapeText(outShape) + " does not match the " +
                    std::to_string(info.classes.size()) + " classes in model.json (expected [1, " +
                    std::to_string(rows) + ", N])");
    }
    anchors = outShape[2];
  }
};

OnnxYoloDetector::OnnxYoloDetector(const fs::path& modelDir, ExecutionProvider provider,
                                   const DecodeParams& params, int frameWidth, int frameHeight)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.info = readModelInfo(modelDir / "model.json");
  s.params = params;
  s.lb = makeLetterbox(frameWidth, frameHeight, s.info.inputWidth, s.info.inputHeight);
  s.filler = std::make_unique<InputTensorFiller>(s.lb);
  s.input.resize(3 * static_cast<size_t>(s.info.inputWidth) * s.info.inputHeight);
  const fs::path onnx = modelDir / "model.onnx";
  if (!fs::is_regular_file(onnx)) throw VcError("missing " + onnx.string());
  try {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == ExecutionProvider::CoreML) {
      try {
        options.AppendExecutionProvider(
            "CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"},
                       {"RequireStaticInputShapes", "1"}});
        s.backend = "CoreML";
      } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "vc: CoreML execution provider unavailable (%s); using CPU\n", e.what());
      }
    }
    s.session = Ort::Session(ortEnv(), onnx.c_str(), options);
    s.checkShapes(onnx);
  } catch (const Ort::Exception& e) {
    throw VcError("onnxruntime: " + onnx.string() + ": " + e.what());
  }
}

OnnxYoloDetector::~OnnxYoloDetector() = default;

std::vector<Box> OnnxYoloDetector::detect(const VideoFrame& frame) {
  Impl& s = *impl_;
  if (frame.av == nullptr) throw VcError("detector: frame has no pixels");
  s.filler->fill(*frame.av, s.input.data());
  try {
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> shape{1, 3, s.info.inputHeight, s.info.inputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, s.input.data(), s.input.size(),
                                                        shape.data(), shape.size());
    const char* inputs[] = {s.inputName.c_str()};
    const char* outputs[] = {s.outputName.c_str()};
    std::vector<Ort::Value> result =
        s.session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    return decodeYolov8(result[0].GetTensorData<float>(), static_cast<int>(s.info.classes.size()),
                        s.anchors, s.lb, s.params);
  } catch (const Ort::Exception& e) {
    throw VcError(std::string("onnxruntime: inference failed: ") + e.what());
  }
}

const std::vector<std::string>& OnnxYoloDetector::classes() const { return impl_->info.classes; }

std::string OnnxYoloDetector::backend() const { return impl_->backend; }

}  // namespace vc
```

若 `AppendExecutionProvider("CoreML", …)` 的选项名在 1.30.0 中不同，以 `~/.cache/video-compliance/third_party/onnxruntime-osx-arm64-1.30.0/include/coreml_provider_factory.h` 中的 `kCoremlProviderOption_*` 常量为准（`ModelFormat`、`MLComputeUnits`、`RequireStaticInputShapes`）。

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='OnnxYoloDetector.*'`
Expected: `[  PASSED  ] 5 tests.`；报告中记下 `CoreMlGivesTheSameBoxes` 实际用到的后端（CoreML 或 CPU）。

- [ ] **Step 5: Commit**

```bash
git add cpp/src/detect/onnx_yolo_detector.h cpp/src/detect/onnx_yolo_detector.cpp cpp/tests/data cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/onnx_yolo_detector_test.cpp
git commit -m "feat: run the YOLOv8 student model with onnxruntime (CoreML, CPU fallback)"
```

---

### Task 4: BoxTimeline（关键帧检测框 → 逐帧打码框）

**Files:**
- Create: `cpp/src/track/box_timeline.h`, `cpp/src/track/box_timeline.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 加 `src/track/box_timeline.cpp`）, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/box_timeline_test.cpp`

**Interfaces:**
- Consumes: Task 1 `Box`。
- Produces:
  - `enum class MaskOrigin { Detected, Interpolated, Extended };`；`const char* toString(MaskOrigin)`（`"detected"` / `"interpolated"` / `"extended"`）
  - `struct MaskBox { int cls; float x1, y1, x2, y2; MaskOrigin origin; }`
  - `struct FrameTime { int64_t index; double ptsMs; }`；`struct Keyframe { FrameTime time; std::vector<Box> boxes; }`
  - `struct TimelineConfig { double matchMaxDistance = 1.0; double expandFraction = 0.15; double motionExpandFraction = 0.5; double minBoxPx = 16.0; }`
  - `class BoxTimeline { BoxTimeline(const TimelineConfig&, int frameWidth, int frameHeight); std::vector<std::vector<MaskBox>> advance(const std::vector<FrameTime>& frames, const Keyframe& next); std::vector<std::vector<MaskBox>> finish(const std::vector<FrameTime>& frames); }`

**行为（规格主设计 §7.2）：**
- `advance` 第一次调用只记录关键帧，`frames` 必须为空，返回空。之后每次调用传入“上一个关键帧（含）到 `next`（不含）”的全部帧，返回这些帧的打码框（与 `frames` 同序）。
- 关联：同类框之间，代价 = |预测中心 − 下一帧中心| / max(1, 两框宽高平均)；预测中心 = 上一框中心 + 速度 × 帧数差（速度来自上一个区间的配对，单位：像素 / 帧序号；没有则为 0）；代价 ≤ `matchMaxDistance` 的候选按代价升序贪心配对。
- 配对的框：线性插值，比例按 PTS（区间内 PTS 严格递增且落在两关键帧之间时），否则按帧序号；上一关键帧那一帧标 `Detected`，其余 `Interpolated`；运动外扩量 = 两框中心在 x / y 上的位移。
- 上一关键帧没配上的框：沿速度外推覆盖整个区间（无速度则原地），关键帧那一帧 `Detected`，其余 `Extended`；运动外扩量 = |速度| × 区间帧数。
- 下一关键帧没配上的框：原样覆盖整个区间（向前延长一个间隔），全部 `Extended`。
- 外扩：每边加 `expandFraction × 宽（高）+ motionExpandFraction × 位移`；宽或高小于 `minBoxPx` 时绕中心补足；裁到画面内，裁后无面积的丢弃。
- `finish`：最后一个关键帧（含）之后的帧，所有框沿速度外推，关键帧那一帧 `Detected`，其余 `Extended`；之后状态清空。
- 违反调用约定（首次调用带帧、`frames` 不从上一关键帧开始、`next` 不在 `frames` 之后）抛 `std::invalid_argument`；构造参数非法同样抛 `std::invalid_argument`。

- [ ] **Step 1: Write the failing test**

`cpp/tests/box_timeline_test.cpp`：

```cpp
#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;
constexpr int kPlate = 1;

TimelineConfig exact() { return TimelineConfig{1.0, 0.0, 0.0, 0.0}; }

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

TEST(BoxTimeline, InterpolatesMatchedBoxes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_TRUE(timeline.advance({}, key(0, {face(100, 100, 120, 120)})).empty());
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToFrameIndexWhenTimestampsStall) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}, 0.0));
  const auto masks = timeline.advance(frames(0, 12, 0.0), key(12, {face(112, 100, 132, 120)}, 0.0));
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ExtendsDisappearingAndAppearingBoxesOverTheInterval) {
  BoxTimeline gone(exact(), 1000, 1000);
  gone.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto a = gone.advance(frames(0, 12), key(12, {}));
  for (size_t k = 0; k < a.size(); ++k) {
    ASSERT_EQ(a[k].size(), 1u);
    expectBox(a[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Extended);
  }
  BoxTimeline appears(exact(), 1000, 1000);
  appears.advance({}, key(0, {}));
  const auto b = appears.advance(frames(0, 12), key(12, {face(300, 300, 320, 320)}));
  for (const auto& m : b) {
    ASSERT_EQ(m.size(), 1u);
    expectBox(m[0], 300, 300, 320, 320, MaskOrigin::Extended);
  }
}

TEST(BoxTimeline, NeverMatchesAcrossClasses) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {Box{kPlate, 100, 100, 120, 120, 0.9f}}));
  ASSERT_EQ(masks[6].size(), 2u);
  EXPECT_EQ(masks[6][0].origin, MaskOrigin::Extended);
  EXPECT_EQ(masks[6][1].origin, MaskOrigin::Extended);
}

TEST(BoxTimeline, PredictsWithVelocityToMatchFastMovers) {
  // 18 px per interval, then 22 px: unmatched without the prediction (22 / 20 > 1).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)}));
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);

  BoxTimeline fresh(exact(), 1000, 1000);
  fresh.advance({}, key(12, {face(118, 100, 138, 120)}));
  const auto unpredicted = fresh.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)}));
  EXPECT_EQ(unpredicted[6].size(), 2u);
}

TEST(BoxTimeline, ExpandsByFractionAndMotionAndEnforcesTheMinimumSize) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.15, 0.5, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 110)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 110)}));
  // x margin 0.15 * 20 + 0.5 * 12 = 9; y margin 1.5 gives 13 px, grown to 16 around y = 105
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 91, 97, 129, 113, MaskOrigin::Detected);
}

TEST(BoxTimeline, ClipsToTheFrame) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.0, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(0, 0, 4, 4)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(0, 0, 4, 4)}));
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 0, 0, 10, 10, MaskOrigin::Detected);
}

TEST(BoxTimeline, FinishExtendsForwardWithVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = timeline.finish(frames(12, 18));
  ASSERT_EQ(masks.size(), 6u);
  ASSERT_EQ(masks[0].size(), 1u);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[0][0], 112, 100, 132, 120, MaskOrigin::Detected);
  expectBox(masks[3][0], 115, 100, 135, 120, MaskOrigin::Extended);  // frame 15, 1 px per frame
}

TEST(BoxTimeline, RejectsCallsOutOfOrder) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_THROW(timeline.advance(frames(0, 3), key(3, {})), std::invalid_argument);
  timeline.advance({}, key(0, {}));
  EXPECT_THROW(timeline.advance(frames(1, 6), key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.advance(frames(0, 6), key(5, {})), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{0.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_STREQ(toString(MaskOrigin::Interpolated), "interpolated");
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `box_timeline_test.cpp`。

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `track/box_timeline.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/src/track/box_timeline.h`：

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "detect/box.h"

namespace vc {

enum class MaskOrigin { Detected, Interpolated, Extended };

// "detected" | "interpolated" | "extended" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

struct MaskBox {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  MaskOrigin origin = MaskOrigin::Detected;
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
  double matchMaxDistance = 1.0;      // max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
};

// Turns detections on keyframes into masks for every frame (main spec §7.2): boxes matched across
// keyframes are interpolated, unmatched ones are extended over one keyframe interval, and every
// mask is expanded. Feed keyframes in order; frames are identified by decode index.
class BoxTimeline {
 public:
  // Throws std::invalid_argument for a non-positive frame size or invalid config.
  BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight);

  // `frames`: every frame from the previous keyframe (inclusive) to `next` (exclusive), in order.
  // Returns their masks in the same order. The first call records `next` only: `frames` must be
  // empty and the result is empty. Throws std::invalid_argument when the call order is violated.
  std::vector<std::vector<MaskBox>> advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // Masks for the frames from the last keyframe (inclusive) to the end, extending every box
  // forward; resets the timeline.
  std::vector<std::vector<MaskBox>> finish(const std::vector<FrameTime>& frames);

 private:
  struct Velocity {
    double x = 0;
    double y = 0;
  };

  std::vector<std::pair<size_t, size_t>> match(const Keyframe& next, double steps) const;
  void add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
           MaskOrigin origin) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<Keyframe> prev_;
  std::vector<std::optional<Velocity>> velocity_;  // per box of prev_, px per frame index
  double lastSteps_ = 0;                           // frames in the last interval
};

}  // namespace vc
```

`cpp/src/track/box_timeline.cpp`：

```cpp
#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc {
namespace {

struct Point {
  double x = 0;
  double y = 0;
};

Point centerOf(const Box& b) { return {(b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0}; }

Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = static_cast<float>(b.x1 + dx);
  s.x2 = static_cast<float>(b.x2 + dx);
  s.y1 = static_cast<float>(b.y1 + dy);
  s.y2 = static_cast<float>(b.y2 + dy);
  return s;
}

Box lerp(const Box& a, const Box& b, double t) {
  const auto mix = [t](float p, float q) { return static_cast<float>(p + (q - p) * t); };
  Box out = a;
  out.x1 = mix(a.x1, b.x1);
  out.y1 = mix(a.y1, b.y1);
  out.x2 = mix(a.x2, b.x2);
  out.y2 = mix(a.y2, b.y2);
  out.score = std::max(a.score, b.score);
  return out;
}

void growTo(double& lo, double& hi, double minimum) {
  if (hi - lo >= minimum) return;
  const double c = (lo + hi) / 2.0;
  lo = c - minimum / 2.0;
  hi = c + minimum / 2.0;
}

// Position of each frame between two keyframes in [0, 1): by PTS when they increase strictly
// inside the interval, otherwise by decode index.
std::vector<double> fractions(const std::vector<FrameTime>& frames, const FrameTime& from, const FrameTime& to) {
  bool usePts = to.ptsMs > from.ptsMs;
  for (size_t k = 0; usePts && k < frames.size(); ++k) {
    const double pts = frames[k].ptsMs;
    if (pts < from.ptsMs || pts >= to.ptsMs || (k > 0 && pts <= frames[k - 1].ptsMs)) usePts = false;
  }
  std::vector<double> t;
  t.reserve(frames.size());
  for (const FrameTime& f : frames) {
    t.push_back(usePts ? (f.ptsMs - from.ptsMs) / (to.ptsMs - from.ptsMs)
                       : static_cast<double>(f.index - from.index) / static_cast<double>(to.index - from.index));
  }
  return t;
}

}  // namespace

const char* toString(MaskOrigin origin) {
  switch (origin) {
    case MaskOrigin::Detected:
      return "detected";
    case MaskOrigin::Interpolated:
      return "interpolated";
    case MaskOrigin::Extended:
      return "extended";
  }
  return "detected";
}

BoxTimeline::BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight)
    : config_(config), width_(frameWidth), height_(frameHeight) {
  if (frameWidth <= 0 || frameHeight <= 0) throw std::invalid_argument("BoxTimeline: frame size must be positive");
  if (!(config.matchMaxDistance > 0) || config.expandFraction < 0 || config.motionExpandFraction < 0 ||
      config.minBoxPx < 0) {
    throw std::invalid_argument("BoxTimeline: invalid config");
  }
}

std::vector<std::pair<size_t, size_t>> BoxTimeline::match(const Keyframe& next, double steps) const {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  const Keyframe& prev = *prev_;
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < prev.boxes.size(); ++i) {
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& a = prev.boxes[i];
      const Box& b = next.boxes[j];
      if (a.cls != b.cls) continue;
      Point predicted = centerOf(a);
      if (velocity_[i]) {
        predicted.x += velocity_[i]->x * steps;
        predicted.y += velocity_[i]->y * steps;
      }
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (a.width() + a.height() + b.width() + b.height()) / 4.0);
      const double cost = std::hypot(actual.x - predicted.x, actual.y - predicted.y) / size;
      if (cost <= config_.matchMaxDistance) candidates.push_back({cost, i, j});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
  std::vector<bool> usedPrev(prev.boxes.size(), false);
  std::vector<bool> usedNext(next.boxes.size(), false);
  std::vector<std::pair<size_t, size_t>> pairs;
  for (const Candidate& c : candidates) {
    if (usedPrev[c.i] || usedNext[c.j]) continue;
    usedPrev[c.i] = true;
    usedNext[c.j] = true;
    pairs.emplace_back(c.i, c.j);
  }
  return pairs;
}

void BoxTimeline::add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
                      MaskOrigin origin) const {
  const double mx = config_.expandFraction * box.width() + config_.motionExpandFraction * motionX;
  const double my = config_.expandFraction * box.height() + config_.motionExpandFraction * motionY;
  double x1 = box.x1 - mx;
  double x2 = box.x2 + mx;
  double y1 = box.y1 - my;
  double y2 = box.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  x1 = std::clamp(x1, 0.0, static_cast<double>(width_));
  x2 = std::clamp(x2, 0.0, static_cast<double>(width_));
  y1 = std::clamp(y1, 0.0, static_cast<double>(height_));
  y2 = std::clamp(y2, 0.0, static_cast<double>(height_));
  if (x2 <= x1 || y2 <= y1) return;
  out.push_back(MaskBox{box.cls, static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                        static_cast<float>(y2), origin});
}

std::vector<std::vector<MaskBox>> BoxTimeline::advance(const std::vector<FrameTime>& frames,
                                                       const Keyframe& next) {
  if (!prev_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    prev_ = next;
    velocity_.assign(next.boxes.size(), std::nullopt);
    return {};
  }
  const Keyframe& prev = *prev_;
  if (frames.empty() || frames.front().index != prev.time.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prev.time.index);
  const std::vector<std::pair<size_t, size_t>> pairs = match(next, steps);
  std::vector<bool> prevUsed(prev.boxes.size(), false);
  std::vector<bool> nextUsed(next.boxes.size(), false);
  for (const auto& [i, j] : pairs) {
    prevUsed[i] = true;
    nextUsed[j] = true;
  }
  const std::vector<double> t = fractions(frames, prev.time, next.time);
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prev.time.index;
    const double since = static_cast<double>(frames[k].index - prev.time.index);
    for (const auto& [i, j] : pairs) {
      const Point a = centerOf(prev.boxes[i]);
      const Point b = centerOf(next.boxes[j]);
      add(masks[k], lerp(prev.boxes[i], next.boxes[j], t[k]), std::abs(b.x - a.x), std::abs(b.y - a.y),
          atKey ? MaskOrigin::Detected : MaskOrigin::Interpolated);
    }
    for (size_t i = 0; i < prev.boxes.size(); ++i) {
      if (prevUsed[i]) continue;
      const Velocity v = velocity_[i].value_or(Velocity{});
      add(masks[k], shifted(prev.boxes[i], v.x * since, v.y * since), std::abs(v.x) * steps,
          std::abs(v.y) * steps, atKey ? MaskOrigin::Detected : MaskOrigin::Extended);
    }
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      if (!nextUsed[j]) add(masks[k], next.boxes[j], 0.0, 0.0, MaskOrigin::Extended);
    }
  }
  std::vector<std::optional<Velocity>> velocity(next.boxes.size());
  for (const auto& [i, j] : pairs) {
    const Point a = centerOf(prev.boxes[i]);
    const Point b = centerOf(next.boxes[j]);
    velocity[j] = Velocity{(b.x - a.x) / steps, (b.y - a.y) / steps};
  }
  velocity_ = std::move(velocity);
  lastSteps_ = steps;
  prev_ = next;
  return masks;
}

std::vector<std::vector<MaskBox>> BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!prev_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const Keyframe& prev = *prev_;
  if (!frames.empty() && frames.front().index != prev.time.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  const double steps = lastSteps_ > 0 ? lastSteps_ : 1.0;
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prev.time.index;
    const double since = static_cast<double>(frames[k].index - prev.time.index);
    for (size_t i = 0; i < prev.boxes.size(); ++i) {
      const Velocity v = velocity_[i].value_or(Velocity{});
      add(masks[k], shifted(prev.boxes[i], v.x * since, v.y * since), std::abs(v.x) * steps,
          std::abs(v.y) * steps, atKey ? MaskOrigin::Detected : MaskOrigin::Extended);
    }
  }
  prev_.reset();
  velocity_.clear();
  lastSteps_ = 0;
  return masks;
}

}  // namespace vc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*'`
Expected: `[  PASSED  ] 9 tests.`

- [ ] **Step 5: Commit**

```bash
git add cpp/src/track cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/box_timeline_test.cpp
git commit -m "feat: add BoxTimeline keyframe matching, interpolation and extension"
```

---

### Task 5: Mosaic（在 YUV 平面上像素化）

**Files:**
- Create: `cpp/src/mask/mosaic.h`, `cpp/src/mask/mosaic.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 加 `src/mask/mosaic.cpp`）, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/mosaic_test.cpp`

**Interfaces:**
- Consumes: `VcError`；测试工具 `test::makeNoiseFrame`、`test::deepCopy`。
- Produces:
  - `struct MosaicConfig { int minBlockPx = 16; int blocksPerLongSide = 8; }`
  - `struct PixelRect { int x1, y1, x2, y2; bool empty() const; }`（半开区间）
  - `PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight)`
  - `int mosaicBlockSize(const PixelRect&, const MosaicConfig&)`（`max(minBlockPx, ceil(长边 / blocksPerLongSide))`；配置 < 1 抛 `std::invalid_argument`）
  - `void applyMosaic(AVFrame& frame, const std::vector<PixelRect>& rects, const MosaicConfig& config)`：块网格从矩形左上角起；亮度按块取均值（四舍五入），色度在色度分辨率上按 `ceil(块 / 2^subsampling)` 的块取均值；支持 8-bit 平面 YUV 与 NV12；其他格式或缓冲不可写抛 `VcError`。

- [ ] **Step 1: Write the failing test**

`cpp/tests/mosaic_test.cpp`：

```cpp
#include "mask/mosaic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "common/vc_error.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
}

namespace vc {
namespace {

// Every block of `rect` (bw x bh, grid from its top-left) in the plane is uniform and equals the
// rounded mean of `before`; everything outside `rect` is unchanged.
void expectPixelated(const AVFrame& after, const AVFrame& before, int plane, int step, int offset,
                     PixelRect rect, int bw, int bh, int planeWidth, int planeHeight) {
  const auto px = [&](const AVFrame& f, int x, int y) {
    return f.data[plane][y * f.linesize[plane] + x * step + offset];
  };
  for (int y = 0; y < planeHeight; ++y) {
    for (int x = 0; x < planeWidth; ++x) {
      const bool inside = x >= rect.x1 && x < rect.x2 && y >= rect.y1 && y < rect.y2;
      if (!inside) {
        ASSERT_EQ(px(after, x, y), px(before, x, y)) << "plane " << plane << " outside at " << x << "," << y;
        continue;
      }
      const int bx = rect.x1 + (x - rect.x1) / bw * bw;
      const int by = rect.y1 + (y - rect.y1) / bh * bh;
      const int ex = std::min(bx + bw, rect.x2);
      const int ey = std::min(by + bh, rect.y2);
      uint64_t sum = 0;
      for (int yy = by; yy < ey; ++yy) {
        for (int xx = bx; xx < ex; ++xx) sum += px(before, xx, yy);
      }
      const uint64_t n = static_cast<uint64_t>(ex - bx) * static_cast<uint64_t>(ey - by);
      ASSERT_EQ(px(after, x, y), (sum + n / 2) / n) << "plane " << plane << " inside at " << x << "," << y;
    }
  }
}

TEST(Mosaic, BlockSizeFollowsTheLongSide) {
  EXPECT_EQ(mosaicBlockSize({0, 0, 40, 30}, MosaicConfig{}), 16);
  EXPECT_EQ(mosaicBlockSize({0, 0, 200, 100}, MosaicConfig{}), 25);
  EXPECT_THROW(mosaicBlockSize({0, 0, 10, 10}, MosaicConfig{0, 8}), std::invalid_argument);
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

TEST(Mosaic, PixelatesYuv420pAndLeavesTheRestAlone) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 1);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{10, 10, 50, 40}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {10, 10, 50, 40}, 16, 16, 96, 64);
  expectPixelated(*frame, *before, 1, 1, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
  expectPixelated(*frame, *before, 2, 1, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
}

TEST(Mosaic, PixelatesNv12InterleavedChroma) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 2);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{10, 10, 50, 40}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {10, 10, 50, 40}, 16, 16, 96, 64);
  expectPixelated(*frame, *before, 1, 2, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
  expectPixelated(*frame, *before, 1, 2, 1, {5, 5, 25, 20}, 8, 8, 48, 32);
}

TEST(Mosaic, ClipsRectsToTheFrame) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 3);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{90, 60, 120, 80}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {90, 60, 96, 64}, 16, 16, 96, 64);
}

TEST(Mosaic, RefusesSharedBuffersAndOtherFormats) {
  auto frame = test::makeNoiseFrame(32, 32, AV_PIX_FMT_YUV420P, 4);
  av::FramePtr other = av::allocFrame();
  ASSERT_GE(av_frame_ref(other.get(), frame.get()), 0);
  EXPECT_THROW(applyMosaic(*frame, {PixelRect{0, 0, 16, 16}}, MosaicConfig{}), VcError);

  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = 16;
  rgb->height = 16;
  ASSERT_GE(av_frame_get_buffer(rgb.get(), 0), 0);
  EXPECT_THROW(applyMosaic(*rgb, {PixelRect{0, 0, 8, 8}}, MosaicConfig{}), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `mosaic_test.cpp`。

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `mask/mosaic.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/src/mask/mosaic.h`：

```cpp
#pragma once

#include <vector>

struct AVFrame;

namespace vc {

struct MosaicConfig {
  int minBlockPx = 16;         // no block is smaller than this
  int blocksPerLongSide = 8;   // at most this many blocks along a mask's long side
};

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

// Smallest pixel rectangle covering the box, clipped to the frame.
PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight);

// max(minBlockPx, ceil(long side / blocksPerLongSide)); throws std::invalid_argument when a
// setting is below 1.
int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config);

// Replaces every block of every rect (grid from the rect's top-left) by its mean: luma at full
// resolution, chroma at chroma resolution with correspondingly smaller blocks. Supports 8-bit
// planar YUV and NV12. Throws VcError for other pixel formats or when the frame's buffers are
// shared (call av_frame_make_writable first).
void applyMosaic(AVFrame& frame, const std::vector<PixelRect>& rects, const MosaicConfig& config);

}  // namespace vc
```

`cpp/src/mask/mosaic.cpp`：

```cpp
#include "mask/mosaic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

int ceilShift(int value, int shift) { return -((-value) >> shift); }

PixelRect clipped(const PixelRect& r, int width, int height) {
  return {std::clamp(r.x1, 0, width), std::clamp(r.y1, 0, height), std::clamp(r.x2, 0, width),
          std::clamp(r.y2, 0, height)};
}

// Sets every blockW x blockH block of `rect` (plane pixels) to its rounded mean. `step` is the
// byte distance between neighbouring samples, `offset` the first sample's byte (NV12 chroma: 2, 0/1).
void pixelate(uint8_t* data, int stride, const PixelRect& rect, int blockW, int blockH, int step, int offset) {
  for (int by = rect.y1; by < rect.y2; by += blockH) {
    const int ey = std::min(by + blockH, rect.y2);
    for (int bx = rect.x1; bx < rect.x2; bx += blockW) {
      const int ex = std::min(bx + blockW, rect.x2);
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

}  // namespace

PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight) {
  const PixelRect r{static_cast<int>(std::floor(x1)), static_cast<int>(std::floor(y1)),
                    static_cast<int>(std::ceil(x2)), static_cast<int>(std::ceil(y2))};
  return clipped(r, frameWidth, frameHeight);
}

int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config) {
  if (config.minBlockPx < 1 || config.blocksPerLongSide < 1) {
    throw std::invalid_argument("mosaic block settings must be >= 1");
  }
  const int longSide = std::max(rect.x2 - rect.x1, rect.y2 - rect.y1);
  return std::max(config.minBlockPx, (longSide + config.blocksPerLongSide - 1) / config.blocksPerLongSide);
}

void applyMosaic(AVFrame& frame, const std::vector<PixelRect>& rects, const MosaicConfig& config) {
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
  for (const PixelRect& raw : rects) {
    const PixelRect r = clipped(raw, frame.width, frame.height);
    if (r.empty()) continue;
    const int block = mosaicBlockSize(r, config);
    pixelate(frame.data[0], frame.linesize[0], r, block, block, 1, 0);
    const PixelRect c{r.x1 >> sw, r.y1 >> sh, ceilShift(r.x2, sw), ceilShift(r.y2, sh)};
    const int bw = std::max(1, ceilShift(block, sw));
    const int bh = std::max(1, ceilShift(block, sh));
    if (nv12) {
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 2, 0);
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 2, 1);
    } else {
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 1, 0);
      pixelate(frame.data[2], frame.linesize[2], c, bw, bh, 1, 0);
    }
  }
}

}  // namespace vc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='Mosaic.*'`
Expected: `[  PASSED  ] 6 tests.`

- [ ] **Step 5: Commit**

```bash
git add cpp/src/mask cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/mosaic_test.cpp
git commit -m "feat: pixelate mask rectangles on planar YUV and NV12 frames"
```

---

### Task 6: VideoReader 音频透传与 VideoWriter（编码 + 封装 + 音频拷贝）

**Files:**
- Modify: `cpp/src/io/video_reader.h`, `cpp/src/io/video_reader.cpp`
- Create: `cpp/src/io/video_writer.h`, `cpp/src/io/video_writer.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 加 `src/io/video_writer.cpp`）
- Modify: `cpp/tests/test_media.h`, `cpp/tests/test_media.cpp`, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/video_reader_test.cpp`（追加）, `cpp/tests/video_writer_test.cpp`

**Interfaces:**
- Consumes: 现有 `VideoReader`（`next()`、`info()`）、`av_ptr.h`、`VcError`、`checkAv`、`avErrorText`。
- Produces:
  - `VideoReader` 新增：`const AVStream* videoStream() const;`、`std::vector<const AVStream*> audioStreams() const;`、`std::vector<std::string> droppedStreams() const;`（格式 `"#<index> <type> <codec>"`，如 `"#2 subtitle mov_text"`）、`void setAudioSink(std::function<void(const AVPacket&)> sink);`（在 `next()` 内对每个音频包回调，包只在回调期间有效；回调抛出的异常从 `next()` 传出）
  - `struct WriterConfig { std::filesystem::path path; int width; int height; AVPixelFormat pixelFormat; AVRational timeBase; AVRational frameRate; std::string sourceCodec; int64_t bitRate; bool preferHardware; }`
  - `class VideoWriter { VideoWriter(const WriterConfig&, const std::vector<const AVStream*>& audio); const std::string& encoderName() const; void writeVideo(AVFrame& frame); void writeAudio(const AVPacket& packet, size_t ordinal); void finish(); int64_t videoFrames() const; }`：容器固定 MP4（不看扩展名，便于写 `.partial`）；`sourceCodec == "hevc"` 用 `hevc_videotoolbox` → `libx265`，否则 `h264_videotoolbox` → `libx264`；`preferHardware == false` 跳过 VideoToolbox；`bitRate <= 0` 用 8 Mb/s；`writeVideo` 前把 `frame.pict_type` 置为 NONE；编码器不接受 `pixelFormat` 时（如 libx265 不收 NV12）选 yuv420p 并用 swscale 转换；HEVC 的 codec tag 设为 `hvc1`。
  - 测试工具：`fs::path videoWithAudio()`（2 s 640×360@30 H.264 + 2 s AAC）、`fs::path videoWithSubtitles()`（1 s 320×240 H.264 + mov_text 字幕）、`int streamCount(const fs::path&, AVMediaType)`、`double streamDurationSec(const fs::path&, AVMediaType)`

- [ ] **Step 1: Write the failing tests**

`cpp/tests/test_media.h` 增加：

```cpp
// 2 s, 640x360 @ 30 fps H.264 (60 frames) with 2 s of AAC audio.
std::filesystem::path videoWithAudio();
// 1 s, 320x240 @ 30 fps H.264 with a mov_text subtitle stream.
std::filesystem::path videoWithSubtitles();
// Number of streams of `type` in a media file (opened with libavformat).
int streamCount(const std::filesystem::path& path, AVMediaType type);
// Duration in seconds of the first stream of `type`, from the container.
double streamDurationSec(const std::filesystem::path& path, AVMediaType type);
```

`cpp/tests/test_media.cpp` 增加（`#include <fstream>`）：

```cpp
fs::path videoWithAudio() {
  return makeMedia("video_audio.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=2 -f lavfi -i sine=f=440:d=2 "
                   "-c:v libx264 -pix_fmt yuv420p -g 30 -c:a aac -shortest");
}

fs::path videoWithSubtitles() {
  const fs::path srt = tmpDir() / "subs.srt";
  if (!fs::exists(srt)) {
    const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-subs.srt");
    std::ofstream(partial) << "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n";
    fs::rename(partial, srt);
  }
  return makeMedia("video_subs.mp4", "-f lavfi -i testsrc2=s=320x240:r=30:d=1 -i \"" + srt.string() +
                                         "\" -map 0:v -map 1:s -c:v libx264 -pix_fmt yuv420p -c:s mov_text");
}

int streamCount(const fs::path& path, AVMediaType type) {
  AVFormatContext* raw = nullptr;
  if (avformat_open_input(&raw, path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path.string());
  av::InputContextPtr input(raw);
  if (avformat_find_stream_info(input.get(), nullptr) < 0) throw std::runtime_error("probe " + path.string());
  int count = 0;
  for (unsigned i = 0; i < input->nb_streams; ++i) count += input->streams[i]->codecpar->codec_type == type ? 1 : 0;
  return count;
}

double streamDurationSec(const fs::path& path, AVMediaType type) {
  AVFormatContext* raw = nullptr;
  if (avformat_open_input(&raw, path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path.string());
  av::InputContextPtr input(raw);
  if (avformat_find_stream_info(input.get(), nullptr) < 0) throw std::runtime_error("probe " + path.string());
  for (unsigned i = 0; i < input->nb_streams; ++i) {
    const AVStream* s = input->streams[i];
    if (s->codecpar->codec_type == type && s->duration != AV_NOPTS_VALUE) return s->duration * av_q2d(s->time_base);
  }
  return 0.0;
}
```

追加到 `cpp/tests/video_reader_test.cpp`（同一命名空间内）：

```cpp
TEST(VideoReader, ForwardsAudioPacketsToTheSink) {
  VideoReader reader(test::videoWithAudio().string(), HwAccel::None);
  const std::vector<const AVStream*> audio = reader.audioStreams();
  ASSERT_EQ(audio.size(), 1u);
  EXPECT_EQ(reader.videoStream()->codecpar->codec_type, AVMEDIA_TYPE_VIDEO);
  int packets = 0;
  bool otherStream = false;
  reader.setAudioSink([&](const AVPacket& packet) {
    ++packets;
    otherStream = otherStream || packet.stream_index != audio[0]->index;
  });
  int frames = 0;
  while (reader.next() != nullptr) ++frames;
  EXPECT_EQ(frames, 60);
  EXPECT_GT(packets, 50);  // ~94 AAC packets in 2 s at 48 kHz
  EXPECT_FALSE(otherStream);
  EXPECT_TRUE(reader.droppedStreams().empty());
}

TEST(VideoReader, ListsStreamsItDrops) {
  VideoReader reader(test::videoWithSubtitles().string(), HwAccel::None);
  EXPECT_TRUE(reader.audioStreams().empty());
  EXPECT_EQ(reader.droppedStreams(), (std::vector<std::string>{"#1 subtitle mov_text"}));
}
```

`cpp/tests/video_writer_test.cpp`：

```cpp
#include "io/video_writer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/av_ptr.h"
#include "io/video_reader.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

// Decodes `input` and writes every frame (and the audio) through a VideoWriter. Returns the encoder.
std::string reencode(const fs::path& input, const fs::path& output, bool preferHardware) {
  VideoReader reader(input.string(), HwAccel::None);
  const AVStream* stream = reader.videoStream();
  WriterConfig config;
  config.path = output;
  config.width = reader.info().width;
  config.height = reader.info().height;
  config.timeBase = stream->time_base;
  config.frameRate = stream->avg_frame_rate;
  config.sourceCodec = reader.info().codecName;
  config.bitRate = stream->codecpar->bit_rate;
  config.preferHardware = preferHardware;
  std::unique_ptr<VideoWriter> writer;
  std::vector<av::PacketPtr> early;
  reader.setAudioSink([&](const AVPacket& packet) {
    av::PacketPtr copy = av::allocPacket();
    if (av_packet_ref(copy.get(), &packet) < 0) throw std::runtime_error("av_packet_ref");
    if (writer) {
      writer->writeAudio(*copy, 0);
    } else {
      early.push_back(std::move(copy));
    }
  });
  while (const VideoFrame* frame = reader.next()) {
    if (!writer) {
      config.pixelFormat = static_cast<AVPixelFormat>(frame->av->format);
      writer = std::make_unique<VideoWriter>(config, reader.audioStreams());
      for (const av::PacketPtr& p : early) writer->writeAudio(*p, 0);
      early.clear();
    }
    av::FramePtr clone(av_frame_clone(frame->av));
    clone->pts = clone->best_effort_timestamp;
    writer->writeVideo(*clone);
  }
  writer->finish();
  EXPECT_GT(writer->videoFrames(), 0);
  return writer->encoderName();
}

std::vector<double> framePts(const fs::path& path) {
  VideoReader reader(path.string(), HwAccel::None);
  std::vector<double> pts;
  while (const VideoFrame* frame = reader.next()) pts.push_back(frame->ptsMs);
  return pts;
}

TEST(VideoWriter, ReencodesH264WithTheSoftwareEncoder) {
  const fs::path out = test::tmpDir() / "writer_x264.mp4";
  EXPECT_EQ(reencode(test::basicVideo(), out, false), "libx264");
  const std::vector<double> in = framePts(test::basicVideo());
  const std::vector<double> got = framePts(out);
  ASSERT_EQ(got.size(), 60u);
  for (size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], in[i], 1.0) << i;
  EXPECT_EQ(VideoReader(out.string(), HwAccel::None).info().codecName, "h264");
}

TEST(VideoWriter, PrefersVideoToolboxWhenAvailable) {
  const fs::path out = test::tmpDir() / "writer_vt.mp4";
  const std::string encoder = reencode(test::basicVideo(), out, true);
  EXPECT_TRUE(encoder == "h264_videotoolbox" || encoder == "libx264") << encoder;
  EXPECT_EQ(framePts(out).size(), 60u);
}

TEST(VideoWriter, KeepsHevcAsHevc) {
  const fs::path out = test::tmpDir() / "writer_x265.mp4";
  EXPECT_EQ(reencode(test::tenBitVideo(), out, false), "libx265");
  EXPECT_EQ(VideoReader(out.string(), HwAccel::None).info().codecName, "hevc");
  EXPECT_EQ(framePts(out).size(), 30u);
}

TEST(VideoWriter, CopiesAudio) {
  const fs::path out = test::tmpDir() / "writer_audio.mp4";
  reencode(test::videoWithAudio(), out, false);
  EXPECT_EQ(test::streamCount(out, AVMEDIA_TYPE_AUDIO), 1);
  EXPECT_NEAR(test::streamDurationSec(out, AVMEDIA_TYPE_AUDIO),
              test::streamDurationSec(test::videoWithAudio(), AVMEDIA_TYPE_AUDIO), 0.05);
  EXPECT_EQ(framePts(out).size(), 60u);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `video_writer_test.cpp`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `io/video_writer.h` not found / `audioStreams` is not a member of `VideoReader`

- [ ] **Step 3: Write minimal implementation**

`cpp/src/io/video_reader.h`：在 `struct AVFrame;` 旁加 `struct AVPacket;`、`struct AVStream;`，加 `#include <functional>`、`#include <vector>`；`VideoReader` 的 public 部分在 `decodeErrors()` 之后加：

```cpp
  // The decoded video stream (codec parameters, time base, frame rate).
  const AVStream* videoStream() const;
  // Audio streams in stream order; vc_redact copies them unchanged.
  std::vector<const AVStream*> audioStreams() const;
  // Every other stream as "#<index> <type> <codec>" (GPS / data tracks, subtitles, extra video);
  // vc_redact drops them.
  std::vector<std::string> droppedStreams() const;
  // Called from next() with each packet of an audio stream, in demux order. The packet is only
  // valid during the call; an exception thrown by the sink propagates out of next().
  void setAudioSink(std::function<void(const AVPacket&)> sink);
```

`cpp/src/io/video_reader.cpp`：
- `#include <algorithm>`、`#include <functional>`、`#include <vector>`。
- `Impl` 增加成员：

```cpp
  std::vector<int> audioIndices;
  std::vector<std::string> dropped;
  std::function<void(const AVPacket&)> audioSink;
```

- `Impl::open` 在 `startPts = …` 之后加：

```cpp
  for (unsigned i = 0; i < input->nb_streams; ++i) {
    if (static_cast<int>(i) == streamIndex) continue;
    const AVCodecParameters* par = input->streams[i]->codecpar;
    if (par->codec_type == AVMEDIA_TYPE_AUDIO) {
      audioIndices.push_back(static_cast<int>(i));
      continue;
    }
    const char* type = av_get_media_type_string(par->codec_type);
    dropped.push_back("#" + std::to_string(i) + " " + (type ? type : "unknown") + " " +
                      avcodec_get_name(par->codec_id));
  }
```

- `Impl::feedDecoder` 中把非视频包的分支改为：

```cpp
    if (packet->stream_index != streamIndex) {
      const bool audio = std::find(audioIndices.begin(), audioIndices.end(), packet->stream_index) !=
                         audioIndices.end();
      if (audio && audioSink) {
        try {
          audioSink(*packet);
        } catch (...) {
          av_packet_unref(packet.get());
          throw;
        }
      }
      av_packet_unref(packet.get());
      continue;
    }
```

- 新增成员函数：

```cpp
const AVStream* VideoReader::videoStream() const { return impl_->input->streams[impl_->streamIndex]; }

std::vector<const AVStream*> VideoReader::audioStreams() const {
  std::vector<const AVStream*> streams;
  for (int index : impl_->audioIndices) streams.push_back(impl_->input->streams[index]);
  return streams;
}

std::vector<std::string> VideoReader::droppedStreams() const { return impl_->dropped; }

void VideoReader::setAudioSink(std::function<void(const AVPacket&)> sink) {
  impl_->audioSink = std::move(sink);
}
```

`cpp/src/io/video_writer.h`：

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vc {

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
};

// Encodes frames and stream-copies audio into an MP4 file.
class VideoWriter {
 public:
  // Opens the first encoder that works and writes the MP4 header; `audio` streams are copied with
  // their codec parameters and time base. Throws VcError.
  VideoWriter(const WriterConfig& config, const std::vector<const AVStream*>& audio);
  ~VideoWriter();  // closes the file; without finish() it is incomplete
  VideoWriter(const VideoWriter&) = delete;
  VideoWriter& operator=(const VideoWriter&) = delete;

  // e.g. "h264_videotoolbox" or "libx264".
  const std::string& encoderName() const;
  // Encodes one frame whose pts is in config.timeBase. Clears frame.pict_type so the encoder picks
  // frame types itself. Throws VcError.
  void writeVideo(AVFrame& frame);
  // Copies one packet of the audio stream at position `ordinal` of the constructor's `audio`; its
  // timestamps are in that stream's time base. Throws VcError.
  void writeAudio(const AVPacket& packet, size_t ordinal);
  // Flushes the encoder and writes the trailer. Throws VcError.
  void finish();
  int64_t videoFrames() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/io/video_writer.cpp`：

```cpp
#include "io/video_writer.h"

#include <string>
#include <utility>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {
namespace {

constexpr int64_t kDefaultBitRate = 8'000'000;

struct OutputDeleter {
  void operator()(AVFormatContext* p) const {
    if (p == nullptr) return;
    if (p->pb != nullptr && (p->oformat->flags & AVFMT_NOFILE) == 0) avio_closep(&p->pb);
    avformat_free_context(p);
  }
};

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

}  // namespace

struct VideoWriter::Impl {
  WriterConfig config;
  std::unique_ptr<AVFormatContext, OutputDeleter> output;
  av::CodecContextPtr encoder;
  std::string encoderName;
  AVStream* videoStream = nullptr;
  std::vector<AVStream*> audioOut;
  std::vector<AVRational> audioIn;
  av::PacketPtr packet = av::allocPacket();
  av::SwsPtr sws;                              // only when the encoder needs another pixel format
  av::FramePtr converted = av::allocFrame();
  int64_t frames = 0;
  bool finished = false;

  void openEncoder() {
    const AVRational rate = config.frameRate.num > 0 && config.frameRate.den > 0 ? config.frameRate : AVRational{30, 1};
    std::string errors;
    for (const std::string& name : encoderCandidates(config)) {
      const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
      if (codec == nullptr) {
        errors += name + ": not available; ";
        continue;
      }
      av::CodecContextPtr ctx(avcodec_alloc_context3(codec));
      if (!ctx) throw VcError("avcodec_alloc_context3 failed");
      ctx->width = config.width;
      ctx->height = config.height;
      ctx->pix_fmt = encoderFormat(ctx.get(), codec, config.pixelFormat);
      ctx->time_base = config.timeBase;
      ctx->framerate = rate;
      ctx->bit_rate = config.bitRate > 0 ? config.bitRate : kDefaultBitRate;
      if ((output->oformat->flags & AVFMT_GLOBALHEADER) != 0) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      const int ret = avcodec_open2(ctx.get(), codec, nullptr);
      if (ret < 0) {
        errors += name + ": " + avErrorText(ret) + "; ";
        continue;
      }
      encoder = std::move(ctx);
      encoderName = name;
      return;
    }
    throw VcError("no usable video encoder (" + errors + ")");
  }

  void drain() {
    while (true) {
      const int ret = avcodec_receive_packet(encoder.get(), packet.get());
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
      checkAv(ret, "receive encoded packet");
      av_packet_rescale_ts(packet.get(), encoder->time_base, videoStream->time_base);
      packet->stream_index = videoStream->index;
      checkAv(av_interleaved_write_frame(output.get(), packet.get()), "write video packet");
    }
  }
};

VideoWriter::VideoWriter(const WriterConfig& config, const std::vector<const AVStream*>& audio)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.config = config;
  const std::string path = config.path.string();
  AVFormatContext* raw = nullptr;
  checkAv(avformat_alloc_output_context2(&raw, nullptr, "mp4", path.c_str()), "create MP4 muxer");
  s.output.reset(raw);
  s.openEncoder();

  s.videoStream = avformat_new_stream(s.output.get(), nullptr);
  if (s.videoStream == nullptr) throw VcError("avformat_new_stream failed");
  checkAv(avcodec_parameters_from_context(s.videoStream->codecpar, s.encoder.get()), "copy encoder parameters");
  if (s.encoder->codec_id == AV_CODEC_ID_HEVC) s.videoStream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
  s.videoStream->time_base = s.encoder->time_base;
  s.videoStream->avg_frame_rate = s.encoder->framerate;

  for (const AVStream* in : audio) {
    AVStream* out = avformat_new_stream(s.output.get(), nullptr);
    if (out == nullptr) throw VcError("avformat_new_stream failed");
    checkAv(avcodec_parameters_copy(out->codecpar, in->codecpar), "copy audio parameters");
    out->codecpar->codec_tag = 0;
    out->time_base = in->time_base;
    s.audioOut.push_back(out);
    s.audioIn.push_back(in->time_base);
  }

  checkAv(avio_open(&s.output->pb, path.c_str(), AVIO_FLAG_WRITE), "open " + path);
  checkAv(avformat_write_header(s.output.get(), nullptr), "write MP4 header");
}

VideoWriter::~VideoWriter() = default;

const std::string& VideoWriter::encoderName() const { return impl_->encoderName; }

int64_t VideoWriter::videoFrames() const { return impl_->frames; }

void VideoWriter::writeVideo(AVFrame& frame) {
  Impl& s = *impl_;
  if (s.finished) throw VcError("writeVideo after finish");
  AVFrame* toSend = &frame;
  if (frame.format != s.encoder->pix_fmt) {
    if (!s.sws) s.sws = av::allocSws();
    av_frame_unref(s.converted.get());
    s.converted->format = s.encoder->pix_fmt;
    s.converted->width = frame.width;
    s.converted->height = frame.height;
    checkAv(sws_scale_frame(s.sws.get(), s.converted.get(), &frame), "convert frame for the encoder");
    checkAv(av_frame_copy_props(s.converted.get(), &frame), "copy frame properties");
    toSend = s.converted.get();
  }
  toSend->pict_type = AV_PICTURE_TYPE_NONE;
  checkAv(avcodec_send_frame(s.encoder.get(), toSend), "encode frame");
  ++s.frames;
  s.drain();
}

void VideoWriter::writeAudio(const AVPacket& packet, size_t ordinal) {
  Impl& s = *impl_;
  if (ordinal >= s.audioOut.size()) throw VcError("unknown audio stream " + std::to_string(ordinal));
  av::PacketPtr copy = av::allocPacket();
  checkAv(av_packet_ref(copy.get(), &packet), "reference audio packet");
  av_packet_rescale_ts(copy.get(), s.audioIn[ordinal], s.audioOut[ordinal]->time_base);
  copy->stream_index = s.audioOut[ordinal]->index;
  copy->pos = -1;
  checkAv(av_interleaved_write_frame(s.output.get(), copy.get()), "write audio packet");
}

void VideoWriter::finish() {
  Impl& s = *impl_;
  if (s.finished) return;
  checkAv(avcodec_send_frame(s.encoder.get(), nullptr), "flush encoder");
  s.drain();
  checkAv(av_write_trailer(s.output.get()), "write MP4 trailer");
  checkAv(avio_closep(&s.output->pb), "close output");
  s.finished = true;
}

}  // namespace vc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:VideoWriter.*'`
Expected: 全部通过（原有 VideoReader 用例 + 新增 2 个 + VideoWriter 4 个）。

Also run: `./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'`
Expected: 全部通过（VideoReader 的改动不影响 vc_extract）。

- [ ] **Step 5: Commit**

```bash
git add cpp/src/io cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/test_media.h cpp/tests/test_media.cpp cpp/tests/video_reader_test.cpp cpp/tests/video_writer_test.cpp
git commit -m "feat: pass audio through VideoReader and add an MP4 VideoWriter"
```

---

### Task 7: 有界队列、redact 配置与记录格式

**Files:**
- Create: `cpp/src/common/bounded_queue.h`
- Create: `cpp/apps/vc_redact/redact_config.h`, `cpp/apps/vc_redact/redact_config.cpp`
- Create: `cpp/apps/vc_redact/records.h`, `cpp/apps/vc_redact/records.cpp`
- Create: `configs/redact.json`
- Modify: `cpp/CMakeLists.txt`（新增 `vc_redact_lib`）, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/bounded_queue_test.cpp`, `cpp/tests/redact_config_test.cpp`, `cpp/tests/records_test.cpp`

**Interfaces:**
- Consumes: Task 1 `Box`；Task 2 `DecodeParams`；Task 4 `TimelineConfig`、`MaskBox`、`toString(MaskOrigin)`；Task 5 `MosaicConfig`；`kMinBaseFps`（`sampler/frame_sampler.h`）。
- Produces:
  - `template <typename T> class BoundedQueue { explicit BoundedQueue(size_t capacity); bool push(T); std::optional<T> pop(); void close(); }`：`close()` 中止队列——之后 `push` 返回 false、`pop` 返回 `nullopt`、已排队的元素被丢弃。
  - `struct RedactConfig { double detectFps = 5.0; double maxDetectFps = 15.0; double motionThreshold = 0.05; DecodeParams decode; TimelineConfig timeline; MosaicConfig mosaic; }`
  - `RedactConfig parseRedactConfig(const std::string& text, const std::string& origin)`、`RedactConfig readRedactConfig(const std::filesystem::path&)`、`void validateRedactConfig(const RedactConfig&, const std::string& origin)`、`nlohmann::ordered_json redactConfigToJson(const RedactConfig&)`；错误抛 `VcError("<origin>: …")`
  - JSON 键：`detect_fps, max_detect_fps, motion_threshold, score_threshold, nms_iou, match_max_distance, expand_fraction, motion_expand_fraction, min_box_px, mosaic_min_block_px, mosaic_blocks_per_long_side`（均可省略，未知键报错）
  - `std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box, const std::vector<std::string>& classes, const std::string& src)`、`std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes, const std::vector<std::string>& classes)`：各返回一行 JSON（含 `\n`）；类别号越界抛 `VcError`
  - CMake 目标 `vc_redact_lib`（STATIC，include 目录 `apps/`，链接 `vc_detect_lib` 与 `Threads::Threads`）

- [ ] **Step 1: Write the failing tests**

`cpp/tests/bounded_queue_test.cpp`：

```cpp
#include "common/bounded_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

namespace vc {
namespace {

TEST(BoundedQueue, IsFifo) {
  BoundedQueue<int> q(4);
  EXPECT_TRUE(q.push(1));
  EXPECT_TRUE(q.push(2));
  EXPECT_EQ(q.pop(), std::optional<int>(1));
  EXPECT_EQ(q.pop(), std::optional<int>(2));
}

TEST(BoundedQueue, PushBlocksWhileFull) {
  BoundedQueue<int> q(1);
  ASSERT_TRUE(q.push(1));
  std::atomic<bool> pushed{false};
  std::thread producer([&] {
    q.push(2);
    pushed = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(pushed.load());
  EXPECT_EQ(q.pop(), std::optional<int>(1));
  producer.join();
  EXPECT_TRUE(pushed.load());
  EXPECT_EQ(q.pop(), std::optional<int>(2));
}

TEST(BoundedQueue, CloseWakesWaitersAndRejectsPushes) {
  BoundedQueue<int> q(1);
  std::optional<int> got = 7;
  std::thread consumer([&] { got = q.pop(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.close();
  consumer.join();
  EXPECT_FALSE(got.has_value());
  EXPECT_FALSE(q.push(3));
}

}  // namespace
}  // namespace vc
```

`cpp/tests/redact_config_test.cpp`：

```cpp
#include "vc_redact/redact_config.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

void expectDefaults(const RedactConfig& c) {
  const RedactConfig d;
  EXPECT_DOUBLE_EQ(c.detectFps, d.detectFps);
  EXPECT_DOUBLE_EQ(c.maxDetectFps, d.maxDetectFps);
  EXPECT_DOUBLE_EQ(c.motionThreshold, d.motionThreshold);
  EXPECT_FLOAT_EQ(c.decode.scoreThreshold, d.decode.scoreThreshold);
  EXPECT_FLOAT_EQ(c.decode.nmsIou, d.decode.nmsIou);
  EXPECT_DOUBLE_EQ(c.timeline.matchMaxDistance, d.timeline.matchMaxDistance);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, d.timeline.expandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.motionExpandFraction, d.timeline.motionExpandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.minBoxPx, d.timeline.minBoxPx);
  EXPECT_EQ(c.mosaic.minBlockPx, d.mosaic.minBlockPx);
  EXPECT_EQ(c.mosaic.blocksPerLongSide, d.mosaic.blocksPerLongSide);
}

TEST(RedactConfig, RepositoryFileMatchesTheDefaults) {
  expectDefaults(readRedactConfig(std::filesystem::path(VC_REPO_DIR) / "configs" / "redact.json"));
}

TEST(RedactConfig, EmptyObjectGivesDefaultsAndKeysOverride) {
  expectDefaults(parseRedactConfig("{}", "t"));
  const RedactConfig c = parseRedactConfig(R"({"detect_fps": 10, "max_detect_fps": 30, "expand_fraction": 0.2})", "t");
  EXPECT_DOUBLE_EQ(c.detectFps, 10);
  EXPECT_DOUBLE_EQ(c.maxDetectFps, 30);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.2);
}

TEST(RedactConfig, RejectsBadInput) {
  const std::vector<std::string> bad = {
      R"({"detect_fps": 4})",          R"({"detect_fps": 20, "max_detect_fps": 10})",
      R"({"score_threshold": 1.5})",   R"({"nms_iou": 0})",
      R"({"match_max_distance": 0})",  R"({"expand_fraction": -0.1})",
      R"({"mosaic_min_block_px": 0})", R"({"unknown_key": 1})",
      R"({"detect_fps": "5"})",        "[]",
      "{",
  };
  for (const std::string& text : bad) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("cfg: ", 0), 0u) << e.what();
    }
  }
}

TEST(RedactConfig, SerialisesEveryKey) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 11u);
  EXPECT_DOUBLE_EQ(j.at("detect_fps").get<double>(), 5.0);
  EXPECT_EQ(j.at("mosaic_blocks_per_long_side").get<int>(), 8);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/records_test.cpp`：

```cpp
#include "vc_redact/records.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

const std::vector<std::string> kClasses{"face", "plate"};

TEST(Records, DetectionLineFollowsTheContract) {
  const Box b{0, 812.06f, 433.5f, 846.2f, 475.0f, 0.87f};
  EXPECT_EQ(detectionRecord("trip.mp4", 1234, 20566.74, b, kClasses, "student"),
            R"({"video":"trip.mp4","frame":1234,"pts_ms":20566.7,"cls":"face","x1":812.1,"y1":433.5,)"
            R"("x2":846.2,"y2":475.0,"score":0.87,"src":"student"})"
            "\n");
}

TEST(Records, MaskLineListsBoxesWithTheirOrigin) {
  const std::vector<MaskBox> boxes{MaskBox{1, 1.0f, 2.0f, 3.5f, 4.0f, MaskOrigin::Interpolated}};
  EXPECT_EQ(maskRecord("a.mp4", 3, boxes, kClasses),
            R"({"video":"a.mp4","frame":3,"boxes":[{"cls":"plate","x1":1.0,"y1":2.0,"x2":3.5,"y2":4.0,"origin":"interpolated"}]})"
            "\n");
  EXPECT_EQ(maskRecord("a.mp4", 4, {}, kClasses), R"({"video":"a.mp4","frame":4,"boxes":[]})" "\n");
}

TEST(Records, RejectsClassIndicesOutsideTheModel) {
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0, Box{2, 0, 0, 1, 1, 1}, kClasses, "x"), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入三个测试文件；链接改为 `target_link_libraries(vc_tests PRIVATE vc_extract_lib vc_redact_lib vc_detect_lib vc_core GTest::gtest_main)`；`target_compile_definitions` 增加 `VC_REPO_DIR="${PROJECT_SOURCE_DIR}/.."`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4`
Expected: 配置或编译失败（`vc_redact_lib` / `vc_redact/redact_config.h` 不存在）

- [ ] **Step 3: Write minimal implementation**

`configs/redact.json`：

```json
{
  "detect_fps": 5.0,
  "max_detect_fps": 15.0,
  "motion_threshold": 0.05,
  "score_threshold": 0.15,
  "nms_iou": 0.5,
  "match_max_distance": 1.0,
  "expand_fraction": 0.15,
  "motion_expand_fraction": 0.5,
  "min_box_px": 16.0,
  "mosaic_min_block_px": 16,
  "mosaic_blocks_per_long_side": 8
}
```

`cpp/CMakeLists.txt`：在 `vc_extract_lib` 之后加：

```cmake
find_package(Threads REQUIRED)

add_library(vc_redact_lib STATIC
  apps/vc_redact/records.cpp
  apps/vc_redact/redact_config.cpp
)
target_include_directories(vc_redact_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_redact_lib PUBLIC vc_detect_lib Threads::Threads)
```

`cpp/src/common/bounded_queue.h`：

```cpp
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace vc {

// Blocking FIFO between pipeline threads. close() aborts it: waiting and later push() calls return
// false, pop() returns nullopt and queued items are discarded.
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

  bool push(T item) {
    std::unique_lock lock(mutex_);
    notFull_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
    if (closed_) return false;
    items_.push_back(std::move(item));
    notEmpty_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    notEmpty_.wait(lock, [&] { return closed_ || !items_.empty(); });
    if (closed_) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    notFull_.notify_one();
    return item;
  }

  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
      items_.clear();
    }
    notFull_.notify_all();
    notEmpty_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable notFull_;
  std::condition_variable notEmpty_;
  std::deque<T> items_;
  size_t capacity_;
  bool closed_ = false;
};

}  // namespace vc
```

`cpp/apps/vc_redact/redact_config.h`：

```cpp
#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "detect/yolo_decode.h"
#include "mask/mosaic.h"
#include "track/box_timeline.h"

namespace vc {

// configs/redact.json (proxy-validation spec §5.2); the defaults equal the repository file.
struct RedactConfig {
  double detectFps = 5.0;       // base keyframe rate, >= 5
  double maxDetectFps = 15.0;   // keyframe rate while the motion score is high
  double motionThreshold = 0.05;
  DecodeParams decode;
  TimelineConfig timeline;
  MosaicConfig mosaic;
};

// Keys may be omitted; unknown keys, wrong types and out-of-range values throw VcError("<origin>: …").
RedactConfig parseRedactConfig(const std::string& text, const std::string& origin);
RedactConfig readRedactConfig(const std::filesystem::path& path);
void validateRedactConfig(const RedactConfig& config, const std::string& origin);
nlohmann::ordered_json redactConfigToJson(const RedactConfig& config);

}  // namespace vc
```

`cpp/apps/vc_redact/redact_config.cpp`：

```cpp
#include "vc_redact/redact_config.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <string_view>

#include "common/vc_error.h"
#include "sampler/frame_sampler.h"

namespace vc {
namespace {

constexpr std::array<std::string_view, 11> kKeys = {
    "detect_fps",         "max_detect_fps",  "motion_threshold",       "score_threshold",
    "nms_iou",            "match_max_distance", "expand_fraction",     "motion_expand_fraction",
    "min_box_px",         "mosaic_min_block_px", "mosaic_blocks_per_long_side"};

}  // namespace

void validateRedactConfig(const RedactConfig& c, const std::string& origin) {
  const auto require = [&](bool ok, const std::string& what) {
    if (!ok) throw VcError(origin + ": " + what);
  };
  require(c.detectFps >= kMinBaseFps, "detect_fps must be >= 5");
  require(c.maxDetectFps >= c.detectFps, "max_detect_fps must be >= detect_fps");
  require(c.motionThreshold >= 0 && c.motionThreshold <= 1, "motion_threshold must be in [0, 1]");
  require(c.decode.scoreThreshold >= 0 && c.decode.scoreThreshold <= 1, "score_threshold must be in [0, 1]");
  require(c.decode.nmsIou > 0 && c.decode.nmsIou <= 1, "nms_iou must be in (0, 1]");
  require(c.timeline.matchMaxDistance > 0, "match_max_distance must be > 0");
  require(c.timeline.expandFraction >= 0, "expand_fraction must be >= 0");
  require(c.timeline.motionExpandFraction >= 0, "motion_expand_fraction must be >= 0");
  require(c.timeline.minBoxPx >= 0, "min_box_px must be >= 0");
  require(c.mosaic.minBlockPx >= 1, "mosaic_min_block_px must be >= 1");
  require(c.mosaic.blocksPerLongSide >= 1, "mosaic_blocks_per_long_side must be >= 1");
}

RedactConfig parseRedactConfig(const std::string& text, const std::string& origin) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  if (!j.is_object()) throw VcError(origin + ": the root must be a JSON object");
  for (const auto& item : j.items()) {
    if (std::find(kKeys.begin(), kKeys.end(), item.key()) == kKeys.end()) {
      throw VcError(origin + ": unknown key '" + item.key() + "'");
    }
  }
  RedactConfig c;
  try {
    c.detectFps = j.value("detect_fps", c.detectFps);
    c.maxDetectFps = j.value("max_detect_fps", c.maxDetectFps);
    c.motionThreshold = j.value("motion_threshold", c.motionThreshold);
    c.decode.scoreThreshold = j.value("score_threshold", c.decode.scoreThreshold);
    c.decode.nmsIou = j.value("nms_iou", c.decode.nmsIou);
    c.timeline.matchMaxDistance = j.value("match_max_distance", c.timeline.matchMaxDistance);
    c.timeline.expandFraction = j.value("expand_fraction", c.timeline.expandFraction);
    c.timeline.motionExpandFraction = j.value("motion_expand_fraction", c.timeline.motionExpandFraction);
    c.timeline.minBoxPx = j.value("min_box_px", c.timeline.minBoxPx);
    c.mosaic.minBlockPx = j.value("mosaic_min_block_px", c.mosaic.minBlockPx);
    c.mosaic.blocksPerLongSide = j.value("mosaic_blocks_per_long_side", c.mosaic.blocksPerLongSide);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  validateRedactConfig(c, origin);
  return c;
}

RedactConfig readRedactConfig(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parseRedactConfig(text.str(), path.string());
}

nlohmann::ordered_json redactConfigToJson(const RedactConfig& c) {
  nlohmann::ordered_json j;
  j["detect_fps"] = c.detectFps;
  j["max_detect_fps"] = c.maxDetectFps;
  j["motion_threshold"] = c.motionThreshold;
  j["score_threshold"] = c.decode.scoreThreshold;
  j["nms_iou"] = c.decode.nmsIou;
  j["match_max_distance"] = c.timeline.matchMaxDistance;
  j["expand_fraction"] = c.timeline.expandFraction;
  j["motion_expand_fraction"] = c.timeline.motionExpandFraction;
  j["min_box_px"] = c.timeline.minBoxPx;
  j["mosaic_min_block_px"] = c.mosaic.minBlockPx;
  j["mosaic_blocks_per_long_side"] = c.mosaic.blocksPerLongSide;
  return j;
}

}  // namespace vc
```

`cpp/apps/vc_redact/records.h`：

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "detect/box.h"
#include "track/box_timeline.h"

namespace vc {

// One detections.jsonl line (spec §3.2), newline included. Throws VcError when box.cls is not an
// index into `classes`.
std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src);

// One masks.jsonl line for a frame (boxes may be empty), newline included.
std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes,
                       const std::vector<std::string>& classes);

}  // namespace vc
```

`cpp/apps/vc_redact/records.cpp`：

```cpp
#include "vc_redact/records.h"

#include <cmath>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {
namespace {

double round1(double v) { return std::round(v * 10.0) / 10.0; }
double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

const std::string& className(const std::vector<std::string>& classes, int cls) {
  if (cls < 0 || static_cast<size_t>(cls) >= classes.size()) {
    throw VcError("class index " + std::to_string(cls) + " is outside the model's " +
                  std::to_string(classes.size()) + " classes");
  }
  return classes[static_cast<size_t>(cls)];
}

}  // namespace

std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src) {
  nlohmann::ordered_json j;
  j["video"] = video;
  j["frame"] = frame;
  j["pts_ms"] = round1(ptsMs);
  j["cls"] = className(classes, box.cls);
  j["x1"] = round1(box.x1);
  j["y1"] = round1(box.y1);
  j["x2"] = round1(box.x2);
  j["y2"] = round1(box.y2);
  j["score"] = round4(box.score);
  j["src"] = src;
  return j.dump() + "\n";
}

std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes,
                       const std::vector<std::string>& classes) {
  nlohmann::ordered_json j;
  j["video"] = video;
  j["frame"] = frame;
  j["boxes"] = nlohmann::ordered_json::array();
  for (const MaskBox& m : boxes) {
    nlohmann::ordered_json b;
    b["cls"] = className(classes, m.cls);
    b["x1"] = round1(m.x1);
    b["y1"] = round1(m.y1);
    b["x2"] = round1(m.x2);
    b["y2"] = round1(m.y2);
    b["origin"] = toString(m.origin);
    j["boxes"].push_back(b);
  }
  return j.dump() + "\n";
}

}  // namespace vc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='BoundedQueue.*:RedactConfig.*:Records.*'`
Expected: `[  PASSED  ] 10 tests.`

- [ ] **Step 5: Commit**

```bash
git add cpp/src/common/bounded_queue.h cpp/apps/vc_redact configs/redact.json cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/bounded_queue_test.cpp cpp/tests/redact_config_test.cpp cpp/tests/records_test.cpp
git commit -m "feat: add bounded queue, redact config and detection/mask records"
```

---

### Task 8: 脱敏流水线 runRedact（三线程、原子输出、分阶段耗时）

**Files:**
- Modify: `cpp/src/common/vc_error.h`（加入 `OutputExistsError`）
- Modify: `cpp/apps/vc_extract/extract_job.h`（删除其中的 `OutputExistsError` 定义，改用 `common/vc_error.h` 中的同名类）
- Create: `cpp/apps/vc_redact/redact_job.h`, `cpp/apps/vc_redact/redact_job.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_redact_lib` 加 `redact_job.cpp`）
- Modify: `cpp/tests/test_media.h`, `cpp/tests/test_media.cpp`, `cpp/tests/CMakeLists.txt`
- Test: `cpp/tests/redact_job_test.cpp`

**Interfaces:**
- Consumes: `VideoReader`（含 Task 6 新增）、`VideoFrame`、`HwAccel`；`FrameSampler`、`SamplerConfig`、`SampleReason`；`makeLumaThumb`、`meanAbsDiff`、`kThumbWidth`、`kThumbHeight`；Task 2 `Detector`、`FakeDetector`；Task 3 `OnnxYoloDetector`、`ExecutionProvider`；Task 4 `BoxTimeline`、`Keyframe`、`FrameTime`、`MaskBox`；Task 5 `coveringRect`、`applyMosaic`、`PixelRect`；Task 6 `VideoWriter`、`WriterConfig`；Task 7 `BoundedQueue`、`RedactConfig`、`redactConfigToJson`、`detectionRecord`、`maskRecord`。
- Produces:
  - `class OutputExistsError : public VcError`（`src/common/vc_error.h`）
  - `struct RedactOptions { std::filesystem::path input; std::filesystem::path outRoot; std::optional<std::filesystem::path> modelDir; std::optional<std::filesystem::path> fakeDetections; RedactConfig config; bool detectEveryFrame = false; ExecutionProvider provider = ExecutionProvider::CoreML; HwAccel hwAccel = HwAccel::None; bool preferHardwareEncoder = true; bool overwrite = false; }`
  - `struct StageStat { double busySec = 0; int64_t items = 0; double fps() const; }`
  - `struct RedactStats { std::filesystem::path outputDir, video; int64_t frames, keyframes, detections, maskedFrames, maskBoxes, decodeErrors, audioPackets; std::string encoder, detector; std::vector<std::string> droppedStreams; size_t audioStreams; StageStat decode, detect, track, mosaic, encode; double elapsedSec; }`
  - `RedactStats runRedact(const RedactOptions&)`：产物写到 `<outRoot>/<stem>/`：`<stem>.redacted.mp4`、`detections.jsonl`、`masks.jsonl`、`stats.json`。`modelDir` 与 `fakeDetections` 必须恰好给一个（否则 `std::invalid_argument`）。输出目录已存在且未 `overwrite` → `OutputExistsError`；`overwrite` 时只替换空目录或含 `masks.jsonl` 的目录，否则 `OutputExistsError`；输出目录会包含输入视频 → `VcError`。所有检查（打开输入、构造检测器、输出目录）在创建任何文件之前完成。
  - `stats.json` 字段：`input, video, frames, keyframes, detections, masked_frames, mask_boxes, decode_errors, detector, encoder, audio_streams, audio_packets, dropped_streams, detect_every_frame, config, stages{decode, detect, track, mosaic, encode: {busy_s, items, fps}}, elapsed_s`
  - 测试工具：`fs::path noisyVideoWithAudio()`（2 s 640×360@30，testsrc2 叠加时域噪声，H.264 4 Mb/s + AAC）

**流水线要点：**
- 解码线程：`reader.next()` → `av_frame_clone` → `pts = best_effort_timestamp`（无时间戳则 `VcError`）→ 推入队列 1；音频包经 `setAudioSink` 拷贝后也推入队列 1（保持解复用顺序）；结束推 `EndItem`。
- 主线程（`Processor`）：每帧算 160×90 缩略图运动分；`detectEveryFrame` 时每帧都是关键帧，否则由 `FrameSampler{detectFps, maxDetectFps, motionThreshold}` 决定；缓冲帧数达到 120 时强制关键帧（防止时间戳停滞时缓冲无限增长）。关键帧：检测 → 写 `detections.jsonl` → `BoxTimeline::advance(缓冲帧, 关键帧)` → 对缓冲帧逐帧写 `masks.jsonl`、`av_frame_make_writable` 后 `applyMosaic` → 推入队列 2；音频直接转发到队列 2。结束：`finish()` 处理剩余缓冲帧，再推 `EndItem`。
- 编码线程：第一帧到达时用其像素格式创建 `VideoWriter`（之前到达的音频先缓存）；写视频 / 音频；`EndItem` 时 `finish()`。
- 任一线程出错：记录第一个异常、关闭两个队列，其余线程随之退出；主线程 join 后重新抛出。被关闭队列打断的线程用内部的 `PipelineAborted` 异常静默退出。
- 成功：编码帧数必须等于解码帧数，否则 `VcError`；`.partial` 重命名为正式文件，最后写 `stats.json`。失败：删除 `.partial`、正式视频、两个 jsonl 与 `stats.json` 后重新抛出。

- [ ] **Step 1: Write the failing test**

`cpp/tests/test_media.h` 增加：

```cpp
// 2 s, 640x360 @ 30 fps H.264 (4 Mb/s) of testsrc2 with strong temporal noise, plus AAC audio:
// every region has detail, so pixelation is measurable.
std::filesystem::path noisyVideoWithAudio();
```

`cpp/tests/test_media.cpp` 增加：

```cpp
fs::path noisyVideoWithAudio() {
  return makeMedia("noisy_audio.mp4",
                   "-f lavfi -i \"testsrc2=s=640x360:r=30:d=2,noise=alls=40:allf=t\" "
                   "-f lavfi -i sine=f=440:d=2 -c:v libx264 -pix_fmt yuv420p -g 30 -b:v 4M "
                   "-c:a aac -shortest");
}
```

`cpp/tests/redact_job_test.cpp`：

```cpp
#include "vc_redact/redact_job.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/av_ptr.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

constexpr const char* kStem = "noisy_audio";

// detections.jsonl with the same face box on frames 0..frames-1.
fs::path fakeFile(const std::string& name, int frames, const std::string& cls = "face") {
  const fs::path path = test::tmpDir() / name;
  std::ofstream out(path);
  for (int i = 0; i < frames; ++i) {
    out << R"({"video":"noisy_audio.mp4","frame":)" << i << R"(,"pts_ms":0,"cls":")" << cls
        << R"(","x1":100,"y1":100,"x2":200,"y2":180,"score":0.9,"src":"fake"})" << '\n';
  }
  return path;
}

RedactOptions baseOptions(const std::string& outName) {
  RedactOptions o;
  o.input = test::noisyVideoWithAudio();
  o.outRoot = test::tmpDir() / outName;
  fs::remove_all(o.outRoot);
  o.fakeDetections = fakeFile(outName + ".jsonl", 60);
  o.preferHardwareEncoder = false;
  return o;
}

size_t lineCount(const fs::path& path) {
  std::ifstream in(path);
  size_t n = 0;
  for (std::string line; std::getline(in, line);) ++n;
  return n;
}

std::vector<av::FramePtr> decodeAll(const fs::path& path) {
  VideoReader reader(path.string(), HwAccel::None);
  std::vector<av::FramePtr> frames;
  while (const VideoFrame* f = reader.next()) frames.push_back(test::deepCopy(*f->av));
  return frames;
}

double lumaMean(const AVFrame& f, int x1, int y1, int x2, int y2) {
  double sum = 0;
  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) sum += f.data[0][y * f.linesize[0] + x];
  }
  return sum / ((x2 - x1) * (y2 - y1));
}

double lumaStd(const AVFrame& f, int x1, int y1, int x2, int y2) {
  const double mean = lumaMean(f, x1, y1, x2, y2);
  double sq = 0;
  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) {
      const double d = f.data[0][y * f.linesize[0] + x] - mean;
      sq += d * d;
    }
  }
  return std::sqrt(sq / ((x2 - x1) * (y2 - y1)));
}

TEST(RedactJob, MasksEveryFrameAndKeepsTheAudio) {
  const RedactOptions o = baseOptions("redact_basic");
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 60);
  EXPECT_GE(s.keyframes, 10);
  EXPECT_LT(s.keyframes, 60);
  EXPECT_EQ(s.detections, s.keyframes);
  EXPECT_EQ(s.maskedFrames, 60);
  EXPECT_EQ(s.audioStreams, 1u);
  EXPECT_GT(s.audioPackets, 0);
  EXPECT_EQ(s.encoder, "libx264");
  EXPECT_EQ(s.detector, "fake");

  const fs::path dir = o.outRoot / kStem;
  EXPECT_EQ(s.video, dir / "noisy_audio.redacted.mp4");
  EXPECT_TRUE(fs::exists(s.video));
  EXPECT_FALSE(fs::exists(dir / "noisy_audio.redacted.mp4.partial"));
  EXPECT_EQ(lineCount(dir / "masks.jsonl"), 60u);
  EXPECT_EQ(lineCount(dir / "detections.jsonl"), static_cast<size_t>(s.keyframes));
  std::ifstream statsFile(dir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("frames").get<int>(), 60);
  EXPECT_EQ(stats.at("stages").at("decode").at("items").get<int>(), 60);
  EXPECT_EQ(test::streamCount(s.video, AVMEDIA_TYPE_AUDIO), 1);

  const auto in = decodeAll(o.input);
  const auto out = decodeAll(s.video);
  ASSERT_EQ(out.size(), 60u);
  // The mask is the fake box grown by 15 % per side, (85, 88)-(215, 192), in 17 px blocks: the
  // block [102, 119) x [105, 122) is flat in the output and noisy in the input.
  EXPECT_LT(lumaStd(*out[30], 104, 107, 117, 120), 4.0);
  EXPECT_GT(lumaStd(*in[30], 104, 107, 117, 120), 15.0);
  // Far from the mask the picture is kept: same brightness, still detailed.
  EXPECT_NEAR(lumaMean(*out[30], 400, 220, 560, 330), lumaMean(*in[30], 400, 220, 560, 330), 3.0);
  EXPECT_GT(lumaStd(*out[30], 400, 220, 560, 330), 10.0);
}

TEST(RedactJob, DetectEveryFrameMakesEveryFrameAKeyframe) {
  RedactOptions o = baseOptions("redact_every");
  o.detectEveryFrame = true;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 60);
  EXPECT_EQ(s.detections, 60);
}

TEST(RedactJob, FailureLeavesNoOutputFilesBehind) {
  RedactOptions o = baseOptions("redact_fail");
  const fs::path failing = test::tmpDir() / "redact_fail_at_30.jsonl";
  std::ofstream(failing) << R"({"frame":30,"fail":true})" << '\n';
  o.fakeDetections = failing;
  o.detectEveryFrame = true;
  EXPECT_THROW(runRedact(o), VcError);
  const fs::path dir = o.outRoot / kStem;
  for (const char* name : {"noisy_audio.redacted.mp4", "noisy_audio.redacted.mp4.partial", "masks.jsonl",
                           "detections.jsonl", "stats.json"}) {
    EXPECT_FALSE(fs::exists(dir / name)) << name;
  }
}

TEST(RedactJob, RefusesAnExistingOutputUnlessOverwriting) {
  RedactOptions o = baseOptions("redact_exists");
  runRedact(o);
  EXPECT_THROW(runRedact(o), OutputExistsError);
  o.overwrite = true;
  EXPECT_NO_THROW(runRedact(o));
}

TEST(RedactJob, RefusesToReplaceAForeignDirectory) {
  RedactOptions o = baseOptions("redact_foreign");
  fs::create_directories(o.outRoot / kStem);
  std::ofstream(o.outRoot / kStem / "notes.txt") << "keep me";
  o.overwrite = true;
  EXPECT_THROW(runRedact(o), OutputExistsError);
  EXPECT_TRUE(fs::exists(o.outRoot / kStem / "notes.txt"));
}

TEST(RedactJob, RequiresExactlyOneDetector) {
  RedactOptions o = baseOptions("redact_nodetector");
  o.fakeDetections.reset();
  EXPECT_THROW(runRedact(o), std::invalid_argument);
  EXPECT_FALSE(fs::exists(o.outRoot / kStem));
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `redact_job_test.cpp`。

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `vc_redact/redact_job.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/src/common/vc_error.h`：在 `VcError` 之后加：

```cpp
// The output directory already exists and overwrite was not requested, or it is not a directory
// the tool may replace.
class OutputExistsError : public VcError {
 public:
  using VcError::VcError;
};
```

`cpp/apps/vc_extract/extract_job.h`：删除其中的 `class OutputExistsError … {};` 定义（同名类现在来自已包含的 `common/vc_error.h`），注释保留。

`cpp/CMakeLists.txt`：`vc_redact_lib` 源文件加入 `apps/vc_redact/redact_job.cpp`。

`cpp/apps/vc_redact/redact_job.h`：

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "detect/onnx_yolo_detector.h"
#include "io/video_reader.h"
#include "vc_redact/redact_config.h"

namespace vc {

struct RedactOptions {
  std::filesystem::path input;
  std::filesystem::path outRoot;  // outputs go to outRoot / <input stem>
  std::optional<std::filesystem::path> modelDir;        // model.onnx + model.json
  std::optional<std::filesystem::path> fakeDetections;  // detections.jsonl replayed instead
  RedactConfig config;
  bool detectEveryFrame = false;
  ExecutionProvider provider = ExecutionProvider::CoreML;
  HwAccel hwAccel = HwAccel::None;
  bool preferHardwareEncoder = true;
  bool overwrite = false;
};

struct StageStat {
  double busySec = 0;
  int64_t items = 0;

  double fps() const { return busySec > 0 ? static_cast<double>(items) / busySec : 0.0; }
};

struct RedactStats {
  std::filesystem::path outputDir;
  std::filesystem::path video;  // the redacted file
  int64_t frames = 0;
  int64_t keyframes = 0;
  int64_t detections = 0;
  int64_t maskedFrames = 0;
  int64_t maskBoxes = 0;
  int64_t decodeErrors = 0;
  int64_t audioPackets = 0;
  std::string encoder;   // e.g. "h264_videotoolbox", "libx264"
  std::string detector;  // "student/CoreML", "student/CPU" or "fake"
  std::vector<std::string> droppedStreams;
  size_t audioStreams = 0;
  StageStat decode;
  StageStat detect;  // items = keyframes
  StageStat track;   // keyframe selection + BoxTimeline; items = frames
  StageStat mosaic;
  StageStat encode;
  double elapsedSec = 0;
};

// Redacts options.input into <outRoot>/<stem>/: <stem>.redacted.mp4 (written as .partial and
// renamed on success), detections.jsonl, masks.jsonl (one line per frame) and stats.json.
// The input, the detector and the output directory are all checked before anything is created;
// any later failure removes this run's files and rethrows.
//
// Throws std::invalid_argument unless exactly one of modelDir / fakeDetections is set;
// OutputExistsError when the output directory exists without overwrite, or with overwrite is
// neither empty nor holds masks.jsonl; VcError for media, detector and I/O failures.
RedactStats runRedact(const RedactOptions& options);

}  // namespace vc
```

`cpp/apps/vc_redact/redact_job.cpp`：

```cpp
#include "vc_redact/redact_job.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include "common/bounded_queue.h"
#include "detect/fake_detector.h"
#include "io/av_ptr.h"
#include "io/video_writer.h"
#include "mask/mosaic.h"
#include "sampler/frame_sampler.h"
#include "sampler/luma_thumb.h"
#include "track/box_timeline.h"
#include "vc_redact/records.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kQueueCapacity = 64;
// A keyframe is forced after this many buffered frames, so stalled timestamps cannot grow the buffer.
constexpr size_t kMaxBufferedFrames = 120;
constexpr const char* kMasksFile = "masks.jsonl";
constexpr const char* kDetectionsFile = "detections.jsonl";
constexpr const char* kStatsFile = "stats.json";

double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct VideoItem {
  int64_t index = 0;
  double ptsMs = 0;
  av::FramePtr frame;
};
struct AudioItem {
  av::PacketPtr packet;
  size_t ordinal = 0;
};
struct EndItem {};
using Item = std::variant<VideoItem, AudioItem, EndItem>;

// Thrown inside a stage when another stage already failed and closed the queues.
struct PipelineAborted {};

class FirstError {
 public:
  void set(std::exception_ptr error) {
    std::lock_guard lock(mutex_);
    if (!error_) error_ = std::move(error);
  }
  // Call after every thread has been joined.
  void rethrowIfSet() {
    if (error_) std::rethrow_exception(error_);
  }

 private:
  std::mutex mutex_;
  std::exception_ptr error_;
};

std::string stemOf(const fs::path& input) {
  const std::string stem = input.stem().string();
  if (stem.empty() || stem == "." || stem == "..") {
    throw VcError("cannot derive an output directory name from " + input.string());
  }
  return stem;
}

bool isSameOrInside(const fs::path& path, const fs::path& dir) {
  const fs::path relative = path.lexically_relative(dir);
  return !relative.empty() && *relative.begin() != "..";
}

void checkOutputDir(const fs::path& input, const fs::path& dir, bool overwrite) {
  if (fs::exists(dir)) {
    if (!overwrite) {
      throw OutputExistsError("output directory exists: " + dir.string() + " (pass --overwrite to replace it)");
    }
    const bool replaceable = fs::is_directory(dir) && (fs::is_empty(dir) || fs::is_regular_file(dir / kMasksFile));
    if (!replaceable) {
      throw OutputExistsError("refusing to overwrite " + dir.string() + ": not a vc_redact output directory");
    }
  }
  if (isSameOrInside(fs::weakly_canonical(fs::absolute(input)), fs::weakly_canonical(fs::absolute(dir)))) {
    throw VcError("output directory " + dir.string() + " would contain the input video; choose another --out");
  }
}

std::unique_ptr<Detector> makeDetector(const RedactOptions& o, int width, int height) {
  if (o.modelDir.has_value() == o.fakeDetections.has_value()) {
    throw std::invalid_argument("exactly one of a model directory or fake detections is required");
  }
  if (o.fakeDetections) return std::make_unique<FakeDetector>(*o.fakeDetections);
  return std::make_unique<OnnxYoloDetector>(*o.modelDir, o.provider, o.config.decode, width, height);
}

// Keyframe selection, detection, BoxTimeline and mosaic: everything between decode and encode.
class Processor {
 public:
  Processor(const RedactOptions& options, Detector& detector, const VideoInfo& info, std::string video,
            std::ostream& detections, std::ostream& masks, BoundedQueue<Item>& out, RedactStats& stats)
      : options_(options),
        detector_(detector),
        video_(std::move(video)),
        detections_(detections),
        masks_(masks),
        out_(out),
        stats_(stats),
        sampler_(SamplerConfig{options.config.detectFps, options.config.maxDetectFps,
                               options.config.motionThreshold, std::nullopt, std::nullopt}),
        timeline_(options.config.timeline, info.width, info.height),
        thumbWidth_(std::min(kThumbWidth, info.width)),
        thumbHeight_(std::min(kThumbHeight, info.height)) {}

  void onVideo(VideoItem item) {
    const Clock::time_point started = Clock::now();
    const VideoFrame view = viewOf(item);
    LumaThumb thumb = makeLumaThumb(view.luma(), thumbWidth_, thumbHeight_);
    const double motion = previous_ ? meanAbsDiff(thumb, *previous_) : 0.0;
    const double diffToKept = lastKept_ ? meanAbsDiff(thumb, *lastKept_) : 1.0;
    const bool sampled =
        options_.detectEveryFrame || sampler_.offer(item.ptsMs, motion, diffToKept) != SampleReason::None;
    const bool key = sampled || buffer_.size() >= kMaxBufferedFrames;
    if (key) lastKept_ = thumb;
    previous_ = std::move(thumb);
    stats_.track.busySec += secondsSince(started);
    if (!key) {
      buffer_.push_back(std::move(item));
      return;
    }

    const Clock::time_point detectStarted = Clock::now();
    const std::vector<Box> boxes = detector_.detect(view);
    stats_.detect.busySec += secondsSince(detectStarted);
    ++stats_.detect.items;
    ++stats_.keyframes;
    stats_.detections += static_cast<int64_t>(boxes.size());
    for (const Box& b : boxes) {
      detections_ << detectionRecord(video_, item.index, item.ptsMs, b, detector_.classes(), detector_.source());
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
  std::string video_;
  std::ostream& detections_;
  std::ostream& masks_;
  BoundedQueue<Item>& out_;
  RedactStats& stats_;
  FrameSampler sampler_;
  BoxTimeline timeline_;
  int thumbWidth_;
  int thumbHeight_;
  std::optional<LumaThumb> previous_;
  std::optional<LumaThumb> lastKept_;
  std::vector<VideoItem> buffer_;
};

void decodeStage(VideoReader& reader, BoundedQueue<Item>& out, StageStat& decode) {
  std::unordered_map<int, size_t> ordinals;
  const std::vector<const AVStream*> audio = reader.audioStreams();
  for (size_t i = 0; i < audio.size(); ++i) ordinals[audio[i]->index] = i;
  reader.setAudioSink([&](const AVPacket& packet) {
    av::PacketPtr copy = av::allocPacket();
    checkAv(av_packet_ref(copy.get(), &packet), "reference audio packet");
    if (!out.push(Item{AudioItem{std::move(copy), ordinals.at(packet.stream_index)}})) throw PipelineAborted{};
  });
  struct ResetSink {
    VideoReader& reader;
    ~ResetSink() { reader.setAudioSink(nullptr); }
  } resetSink{reader};
  while (true) {
    const Clock::time_point started = Clock::now();
    const VideoFrame* frame = reader.next();
    decode.busySec += secondsSince(started);
    if (frame == nullptr) break;
    av::FramePtr clone(av_frame_clone(frame->av));
    if (!clone) throw VcError("av_frame_clone failed");
    if (clone->best_effort_timestamp != AV_NOPTS_VALUE) clone->pts = clone->best_effort_timestamp;
    if (clone->pts == AV_NOPTS_VALUE) throw VcError("frame " + std::to_string(frame->index) + " has no timestamp");
    ++decode.items;
    if (!out.push(Item{VideoItem{frame->index, frame->ptsMs, std::move(clone)}})) throw PipelineAborted{};
  }
  if (!out.push(Item{EndItem{}})) throw PipelineAborted{};
}

struct EncodeResult {
  StageStat encode;
  std::string encoder;
  int64_t audioPackets = 0;
};

// The writer is created on the first video frame, when its pixel format is known; audio that
// arrives earlier is held until then.
void encodeStage(BoundedQueue<Item>& in, WriterConfig config, const std::vector<const AVStream*>& audio,
                 EncodeResult& result) {
  std::unique_ptr<VideoWriter> writer;
  std::vector<AudioItem> early;
  while (std::optional<Item> item = in.pop()) {
    const Clock::time_point started = Clock::now();
    if (std::holds_alternative<EndItem>(*item)) {
      if (!writer) throw VcError("no video frames to encode");
      writer->finish();
      result.encode.busySec += secondsSince(started);
      return;
    }
    if (auto* a = std::get_if<AudioItem>(&*item)) {
      if (writer) {
        writer->writeAudio(*a->packet, a->ordinal);
        ++result.audioPackets;
      } else {
        early.push_back(std::move(*a));
      }
    } else {
      VideoItem& video = std::get<VideoItem>(*item);
      if (!writer) {
        config.pixelFormat = static_cast<AVPixelFormat>(video.frame->format);
        writer = std::make_unique<VideoWriter>(config, audio);
        result.encoder = writer->encoderName();
        for (AudioItem& e : early) {
          writer->writeAudio(*e.packet, e.ordinal);
          ++result.audioPackets;
        }
        early.clear();
      }
      writer->writeVideo(*video.frame);
      ++result.encode.items;
    }
    result.encode.busySec += secondsSince(started);
  }
}

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
  return c;
}

void runPipeline(const RedactOptions& options, VideoReader& reader, Detector& detector, const fs::path& partial,
                 RedactStats& stats) {
  const fs::path dir = partial.parent_path();
  std::ofstream detections(dir / kDetectionsFile);
  std::ofstream masks(dir / kMasksFile);
  if (!detections || !masks) throw VcError("cannot create output files in " + dir.string());

  BoundedQueue<Item> decoded(kQueueCapacity);
  BoundedQueue<Item> processed(kQueueCapacity);
  FirstError error;
  const auto fail = [&](std::exception_ptr e) {
    error.set(std::move(e));
    decoded.close();
    processed.close();
  };
  const WriterConfig config = writerConfig(reader, partial, options.preferHardwareEncoder);
  const std::vector<const AVStream*> audio = reader.audioStreams();
  StageStat decodeStat;
  EncodeResult encodeResult;

  std::thread decodeThread([&] {
    try {
      decodeStage(reader, decoded, decodeStat);
    } catch (const PipelineAborted&) {
    } catch (...) {
      fail(std::current_exception());
    }
  });
  std::thread encodeThread([&] {
    try {
      encodeStage(processed, config, audio, encodeResult);
    } catch (const PipelineAborted&) {
    } catch (...) {
      fail(std::current_exception());
    }
  });
  try {
    Processor processor(options, detector, reader.info(), options.input.filename().string(), detections, masks,
                        processed, stats);
    while (std::optional<Item> item = decoded.pop()) {
      if (auto* video = std::get_if<VideoItem>(&*item)) {
        processor.onVideo(std::move(*video));
      } else if (std::holds_alternative<AudioItem>(*item)) {
        if (!processed.push(std::move(*item))) throw PipelineAborted{};
      } else {
        processor.onEnd();
        break;
      }
    }
  } catch (const PipelineAborted&) {
  } catch (...) {
    fail(std::current_exception());
  }
  decodeThread.join();
  encodeThread.join();
  error.rethrowIfSet();

  stats.decode = decodeStat;
  stats.encode = encodeResult.encode;
  stats.encoder = encodeResult.encoder;
  stats.audioPackets = encodeResult.audioPackets;
  detections.close();
  masks.close();
  if (!detections || !masks) throw VcError("writing detections.jsonl / masks.jsonl failed");
  if (stats.encode.items != stats.decode.items) {
    throw VcError("encoded " + std::to_string(stats.encode.items) + " of " + std::to_string(stats.decode.items) +
                  " decoded frames");
  }
}

nlohmann::ordered_json stageJson(const StageStat& s) {
  nlohmann::ordered_json j;
  j["busy_s"] = s.busySec;
  j["items"] = s.items;
  j["fps"] = s.fps();
  return j;
}

void writeStats(const fs::path& path, const RedactStats& s, const RedactOptions& o) {
  nlohmann::ordered_json j;
  j["input"] = o.input.string();
  j["video"] = s.video.string();
  j["frames"] = s.frames;
  j["keyframes"] = s.keyframes;
  j["detections"] = s.detections;
  j["masked_frames"] = s.maskedFrames;
  j["mask_boxes"] = s.maskBoxes;
  j["decode_errors"] = s.decodeErrors;
  j["detector"] = s.detector;
  j["encoder"] = s.encoder;
  j["audio_streams"] = s.audioStreams;
  j["audio_packets"] = s.audioPackets;
  j["dropped_streams"] = s.droppedStreams;
  j["detect_every_frame"] = o.detectEveryFrame;
  j["config"] = redactConfigToJson(o.config);
  j["stages"]["decode"] = stageJson(s.decode);
  j["stages"]["detect"] = stageJson(s.detect);
  j["stages"]["track"] = stageJson(s.track);
  j["stages"]["mosaic"] = stageJson(s.mosaic);
  j["stages"]["encode"] = stageJson(s.encode);
  j["elapsed_s"] = s.elapsedSec;
  std::ofstream out(path);
  out << j.dump(2) << '\n';
  out.close();
  if (!out) throw VcError("cannot write " + path.string());
}

}  // namespace

RedactStats runRedact(const RedactOptions& options) {
  const Clock::time_point started = Clock::now();
  VideoReader reader(options.input.string(), options.hwAccel);
  const std::unique_ptr<Detector> detector = makeDetector(options, reader.info().width, reader.info().height);
  const std::string stem = stemOf(options.input);

  RedactStats stats;
  stats.outputDir = options.outRoot / stem;
  checkOutputDir(options.input, stats.outputDir, options.overwrite);
  fs::remove_all(stats.outputDir);
  fs::create_directories(stats.outputDir);
  stats.video = stats.outputDir / (stem + ".redacted.mp4");
  const fs::path partial = stats.outputDir / (stem + ".redacted.mp4.partial");
  const std::vector<fs::path> written{partial, stats.video, stats.outputDir / kDetectionsFile,
                                      stats.outputDir / kMasksFile, stats.outputDir / kStatsFile};
  try {
    runPipeline(options, reader, *detector, partial, stats);
    fs::rename(partial, stats.video);
    stats.frames = stats.decode.items;
    stats.track.items = stats.frames;
    stats.decodeErrors = reader.decodeErrors();
    stats.droppedStreams = reader.droppedStreams();
    stats.audioStreams = reader.audioStreams().size();
    stats.detector = detector->backend().empty() ? detector->source() : detector->source() + "/" + detector->backend();
    stats.elapsedSec = secondsSince(started);
    writeStats(stats.outputDir / kStatsFile, stats, options);
    return stats;
  } catch (...) {
    for (const fs::path& p : written) {
      std::error_code ignored;
      fs::remove(p, ignored);
    }
    throw;
  }
}

}  // namespace vc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*'`
Expected: `[  PASSED  ] 6 tests.`（共约 10 秒）

像素阈值（`< 4.0`、`> 15.0`、`± 3.0`、`> 10.0`）若不通过，先打印实际值查明原因（打码块是否对齐、编码码率），确有依据才调整阈值，并在报告中写明实测值与理由。

Also run: `./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'`
Expected: 全部通过（`OutputExistsError` 移动后 vc_extract 不受影响）。

- [ ] **Step 5: Commit**

```bash
git add cpp/src/common/vc_error.h cpp/apps/vc_extract/extract_job.h cpp/apps/vc_redact/redact_job.h cpp/apps/vc_redact/redact_job.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/test_media.h cpp/tests/test_media.cpp cpp/tests/redact_job_test.cpp
git commit -m "feat: add the threaded vc_redact pipeline with atomic outputs and stage timings"
```

---

### Task 9: vc_redact / vc_detect 命令行与文档

**Files:**
- Modify: `cpp/src/common/vc_error.h`（加入 `UsageError`）
- Modify: `cpp/apps/vc_extract/extract_cli.h`（删除其中的 `UsageError` 定义，改用 `common/vc_error.h`）
- Create: `cpp/apps/vc_redact/redact_cli.h`, `cpp/apps/vc_redact/redact_cli.cpp`, `cpp/apps/vc_redact/main.cpp`
- Create: `cpp/apps/vc_detect/detect_images.h`, `cpp/apps/vc_detect/detect_images.cpp`, `cpp/apps/vc_detect/main.cpp`
- Modify: `cpp/CMakeLists.txt`, `cpp/tests/CMakeLists.txt`, `cpp/README.md`
- Test: `cpp/tests/redact_cli_test.cpp`, `cpp/tests/detect_images_test.cpp`

**Interfaces:**
- Consumes: Task 8 `RedactOptions`、`runRedact`、`RedactStats`、`OutputExistsError`；Task 7 `readRedactConfig`、`validateRedactConfig`、`detectionRecord`；Task 3 `OnnxYoloDetector`、`parseExecutionProvider`；`parseHwAccel`；`VideoReader`。
- Produces:
  - `class UsageError : public std::runtime_error`（`src/common/vc_error.h`）
  - `RedactOptions parseRedactArgs(const std::vector<std::string>& args)`、`std::string redactUsage()`
  - 参数：`--input`（必填）、`--out`（必填）、`--model <dir>` 与 `--fake-detections <jsonl>` 恰好一个、`--config <json>`、`--detect-fps <n>`（覆盖配置；若大于 `max_detect_fps`，后者同步提高）、`--detect-every-frame`、`--ep coreml|cpu`（默认 coreml）、`--hwaccel auto|none|videotoolbox`（默认 none）、`--encoder auto|software`（默认 auto）、`--overwrite`、`-h/--help`；配置文件不可读或非法也报 `UsageError`
  - 退出码：0 成功、1 处理失败、2 参数错误、3 输出已存在
  - `struct DetectImagesOptions { std::filesystem::path modelDir, framesDir, out; ExecutionProvider provider = ExecutionProvider::Cpu; DecodeParams decode; }`；`int64_t runDetectImages(const DetectImagesOptions&)`：对 `framesDir/frames.csv` 中每一行的 `<frame:06d>.jpg` 跑检测器，把 `src = "student"` 的检测记录写到 `out`，返回图片数
  - `vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep cpu|coreml] [--score <x>]`

- [ ] **Step 1: Write the failing tests**

`cpp/tests/redact_cli_test.cpp`：

```cpp
#include "vc_redact/redact_cli.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {

using Args = std::vector<std::string>;

TEST(RedactCli, ParsesAModelRun) {
  const RedactOptions o = parseRedactArgs(
      {"--input", "in.mp4", "--out", "out", "--model", "data/models/r1", "--ep", "cpu", "--encoder", "software"});
  EXPECT_EQ(o.input, "in.mp4");
  EXPECT_EQ(o.outRoot, "out");
  ASSERT_TRUE(o.modelDir.has_value());
  EXPECT_EQ(*o.modelDir, "data/models/r1");
  EXPECT_FALSE(o.fakeDetections.has_value());
  EXPECT_EQ(o.provider, ExecutionProvider::Cpu);
  EXPECT_FALSE(o.preferHardwareEncoder);
  EXPECT_EQ(o.hwAccel, HwAccel::None);
  EXPECT_FALSE(o.detectEveryFrame);
  EXPECT_FALSE(o.overwrite);
}

TEST(RedactCli, DetectFpsOverridesAndRaisesTheMaximum) {
  const RedactOptions o =
      parseRedactArgs({"--detect-fps", "30", "--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 30);
  EXPECT_DOUBLE_EQ(o.config.maxDetectFps, 30);
}

TEST(RedactCli, ReadsTheConfigFile) {
  const std::filesystem::path cfg = test::tmpDir() / "redact_cli.json";
  std::ofstream(cfg) << R"({"detect_fps": 10, "expand_fraction": 0.3})";
  const RedactOptions o = parseRedactArgs(
      {"--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl", "--config", cfg.string(), "--overwrite",
       "--detect-every-frame"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 10);
  EXPECT_DOUBLE_EQ(o.config.timeline.expandFraction, 0.3);
  EXPECT_TRUE(o.overwrite);
  EXPECT_TRUE(o.detectEveryFrame);
}

TEST(RedactCli, RejectsBadCommandLines) {
  const std::filesystem::path bad = test::tmpDir() / "redact_cli_bad.json";
  std::ofstream(bad) << R"({"detect_fps": 1})";
  const std::vector<Args> cases = {
      {"--out", "o", "--model", "m"},
      {"--input", "a.mp4", "--model", "m"},
      {"--input", "a.mp4", "--out", "o"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--fake-detections", "f"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", "cuda"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--encoder", "nvenc"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "fast"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "2"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--config", bad.string()},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--config", "missing.json"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--bogus"},
      {"--input"},
  };
  for (const Args& args : cases) EXPECT_THROW(parseRedactArgs(args), UsageError) << ::testing::PrintToString(args);
}

TEST(RedactCli, UsageMentionsEveryOption) {
  const std::string usage = redactUsage();
  for (const char* option : {"--input", "--out", "--model", "--fake-detections", "--config", "--detect-fps",
                             "--detect-every-frame", "--ep", "--hwaccel", "--encoder", "--overwrite"}) {
    EXPECT_NE(usage.find(option), std::string::npos) << option;
  }
}

}  // namespace
}  // namespace vc
```

`cpp/tests/detect_images_test.cpp`：

```cpp
#include "vc_detect/detect_images.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

std::vector<std::string> lines(const fs::path& path) {
  std::ifstream in(path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

TEST(DetectImages, RunsTheModelOnEveryListedFrame) {
  const fs::path gray = test::makeMedia("gray128.jpg", "-f lavfi -i color=c=gray:s=128x128 -frames:v 1 -update 1");
  const fs::path frames = test::tmpDir() / "detect_frames";
  fs::remove_all(frames);
  fs::create_directories(frames);
  fs::copy_file(gray, frames / "000000.jpg");
  fs::copy_file(gray, frames / "000007.jpg");
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "t.mp4,0,0.0,base,0.0000\n"
                                          "t.mp4,7,233.3,base,0.0100\n";
  DetectImagesOptions o;
  o.modelDir = fs::path(VC_TEST_DATA_DIR) / "const_yolo";
  o.framesDir = frames;
  o.out = test::tmpDir() / "detect_images.jsonl";
  EXPECT_EQ(runDetectImages(o), 2);
  const std::vector<std::string> got = lines(o.out);
  ASSERT_EQ(got.size(), 4u);
  EXPECT_EQ(got[0].rfind(R"({"video":"t.mp4","frame":0,"pts_ms":0.0,"cls":"face",)", 0), 0u) << got[0];
  EXPECT_NE(got[0].find(R"("src":"student")"), std::string::npos);
  EXPECT_EQ(got[2].rfind(R"({"video":"t.mp4","frame":7,"pts_ms":233.3,)", 0), 0u) << got[2];
}

TEST(DetectImages, RejectsAFramesDirWithoutFramesCsv) {
  DetectImagesOptions o;
  o.modelDir = fs::path(VC_TEST_DATA_DIR) / "const_yolo";
  o.framesDir = test::tmpDir() / "no_such_frames";
  o.out = test::tmpDir() / "detect_none.jsonl";
  EXPECT_THROW(runDetectImages(o), VcError);
}

}  // namespace
}  // namespace vc
```

`cpp/tests/CMakeLists.txt`：加入 `redact_cli_test.cpp`、`detect_images_test.cpp`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build/cpp -j 4`
Expected: 编译失败 — `vc_redact/redact_cli.h` / `vc_detect/detect_images.h` not found

- [ ] **Step 3: Write minimal implementation**

`cpp/src/common/vc_error.h`：在 `VcError` 之前加（`<stdexcept>` 已包含）：

```cpp
// Invalid command line.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
```

`cpp/apps/vc_extract/extract_cli.h`：删除其中的 `class UsageError … {};`，改为 `#include "common/vc_error.h"`（`<stdexcept>` 若不再需要可删）。

`cpp/CMakeLists.txt`：`vc_redact_lib` 加入 `apps/vc_redact/redact_cli.cpp` 与 `apps/vc_detect/detect_images.cpp`；文件末尾 tests 之前加：

```cmake
add_executable(vc_redact apps/vc_redact/main.cpp)
target_link_libraries(vc_redact PRIVATE vc_redact_lib)

add_executable(vc_detect apps/vc_detect/main.cpp)
target_link_libraries(vc_detect PRIVATE vc_redact_lib)
```

`cpp/apps/vc_redact/redact_cli.h`：

```cpp
#pragma once

#include <string>
#include <vector>

#include "vc_redact/redact_job.h"

namespace vc {

// Parses the arguments after the program name. Throws UsageError, also for an unreadable or
// invalid --config.
RedactOptions parseRedactArgs(const std::vector<std::string>& args);

std::string redactUsage();

}  // namespace vc
```

`cpp/apps/vc_redact/redact_cli.cpp`：

```cpp
#include "vc_redact/redact_cli.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace vc {
namespace fs = std::filesystem;
namespace {

double parseNumber(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects a number, got '" + text + "'");
}

}  // namespace

RedactOptions parseRedactArgs(const std::vector<std::string>& args) {
  RedactOptions o;
  std::optional<fs::path> config;
  std::optional<double> detectFps;
  bool haveInput = false;
  bool haveOut = false;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    const auto value = [&]() -> const std::string& {
      if (i + 1 >= args.size()) throw UsageError(arg + " needs a value");
      return args[++i];
    };
    if (arg == "--input") {
      o.input = value();
      haveInput = true;
    } else if (arg == "--out") {
      o.outRoot = value();
      haveOut = true;
    } else if (arg == "--model") {
      o.modelDir = fs::path(value());
    } else if (arg == "--fake-detections") {
      o.fakeDetections = fs::path(value());
    } else if (arg == "--config") {
      config = fs::path(value());
    } else if (arg == "--detect-fps") {
      detectFps = parseNumber(arg, value());
    } else if (arg == "--detect-every-frame") {
      o.detectEveryFrame = true;
    } else if (arg == "--ep") {
      try {
        o.provider = parseExecutionProvider(value());
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (arg == "--hwaccel") {
      try {
        o.hwAccel = parseHwAccel(value());
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (arg == "--encoder") {
      const std::string& v = value();
      if (v == "auto") {
        o.preferHardwareEncoder = true;
      } else if (v == "software") {
        o.preferHardwareEncoder = false;
      } else {
        throw UsageError("unknown --encoder '" + v + "' (auto|software)");
      }
    } else if (arg == "--overwrite") {
      o.overwrite = true;
    } else {
      throw UsageError("unknown argument '" + arg + "'");
    }
  }
  if (!haveInput) throw UsageError("--input is required");
  if (!haveOut) throw UsageError("--out is required");
  if (o.modelDir.has_value() == o.fakeDetections.has_value()) {
    throw UsageError("pass exactly one of --model or --fake-detections");
  }
  try {
    if (config) o.config = readRedactConfig(*config);
    if (detectFps) {
      o.config.detectFps = *detectFps;
      o.config.maxDetectFps = std::max(o.config.maxDetectFps, *detectFps);
    }
    validateRedactConfig(o.config, config ? config->string() : "--detect-fps");
  } catch (const VcError& e) {
    throw UsageError(e.what());
  }
  return o;
}

std::string redactUsage() {
  return "Usage: vc_redact --input <video> --out <dir> (--model <dir> | --fake-detections <jsonl>) [options]\n"
         "\n"
         "Masks faces and licence plates. Writes <dir>/<video stem>/: <stem>.redacted.mp4,\n"
         "detections.jsonl, masks.jsonl (one line per frame) and stats.json.\n"
         "\n"
         "Options:\n"
         "  --model <dir>             model.onnx + model.json from vc_train export\n"
         "  --fake-detections <file>  replay a detections.jsonl instead of running a model\n"
         "  --config <json>           parameters (see configs/redact.json; built-in defaults otherwise)\n"
         "  --detect-fps <n>          base keyframe detection rate, >= 5 (overrides the config)\n"
         "  --detect-every-frame      detect on every frame (reference run for temporal coverage)\n"
         "  --ep <provider>           coreml | cpu (default coreml, falls back to cpu)\n"
         "  --hwaccel <mode>          decoder: none | auto | videotoolbox (default none)\n"
         "  --encoder <mode>          auto (VideoToolbox, then software) | software (default auto)\n"
         "  --overwrite               replace a previous vc_redact output directory\n"
         "  -h, --help                show this help\n"
         "\n"
         "Exit codes: 0 ok, 1 processing error, 2 invalid arguments, 3 output exists.\n";
}

}  // namespace vc
```

`cpp/apps/vc_redact/main.cpp`：

```cpp
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "vc_redact/redact_cli.h"
#include "vc_redact/redact_job.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;
constexpr int kExitOutputExists = 3;

std::string joined(const std::vector<std::string>& items) {
  if (items.empty()) return "none";
  std::string out;
  for (const std::string& item : items) out += (out.empty() ? "" : ", ") + item;
  return out;
}

void printStats(const vc::RedactStats& s) {
  const double fps = s.elapsedSec > 0 ? static_cast<double>(s.frames) / s.elapsedSec : 0;
  std::printf("output:        %s\n", s.video.string().c_str());
  std::printf("frames:        %lld (keyframes %lld, detections %lld, masked %lld frames / %lld boxes)\n",
              static_cast<long long>(s.frames), static_cast<long long>(s.keyframes),
              static_cast<long long>(s.detections), static_cast<long long>(s.maskedFrames),
              static_cast<long long>(s.maskBoxes));
  std::printf("detector:      %s\n", s.detector.c_str());
  std::printf("encoder:       %s\n", s.encoder.c_str());
  std::printf("audio:         %zu stream(s), %lld packets copied; dropped streams: %s\n", s.audioStreams,
              static_cast<long long>(s.audioPackets), joined(s.droppedStreams).c_str());
  std::printf("stages (fps):  decode %.1f, detect %.1f, track %.1f, mosaic %.1f, encode %.1f\n", s.decode.fps(),
              s.detect.fps(), s.track.fps(), s.mosaic.fps(), s.encode.fps());
  std::printf("decode errors: %lld\n", static_cast<long long>(s.decodeErrors));
  std::printf("elapsed:       %.2f s (%.1f fps)\n", s.elapsedSec, fps);
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << vc::redactUsage();
      return kExitOk;
    }
  }
  vc::RedactOptions options;
  try {
    options = vc::parseRedactArgs(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_redact: " << e.what() << "\n\n" << vc::redactUsage();
    return kExitUsage;
  }
  av_log_set_level(AV_LOG_ERROR);
  try {
    printStats(vc::runRedact(options));
    return kExitOk;
  } catch (const vc::OutputExistsError& e) {
    std::cerr << "vc_redact: " << e.what() << '\n';
    return kExitOutputExists;
  } catch (const std::exception& e) {
    std::cerr << "vc_redact: " << e.what() << '\n';
    return kExitFailure;
  }
}
```

`cpp/apps/vc_detect/detect_images.h`：

```cpp
#pragma once

#include <cstdint>
#include <filesystem>

#include "detect/onnx_yolo_detector.h"

namespace vc {

struct DetectImagesOptions {
  std::filesystem::path modelDir;   // model.onnx + model.json
  std::filesystem::path framesDir;  // vc_extract output: frames.csv + <frame:06d>.jpg
  std::filesystem::path out;        // detections.jsonl to write
  ExecutionProvider provider = ExecutionProvider::Cpu;
  DecodeParams decode;
};

// Runs the ONNX student model on every frame listed in framesDir/frames.csv and writes their
// detections (src "student") to `out`, for comparing the C++ and Python inference paths. Returns
// the number of images. Throws VcError.
int64_t runDetectImages(const DetectImagesOptions& options);

}  // namespace vc
```

`cpp/apps/vc_detect/detect_images.cpp`：

```cpp
#include "vc_detect/detect_images.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "io/video_reader.h"
#include "vc_redact/records.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

struct Row {
  std::string video;
  int64_t frame = 0;
  double ptsMs = 0;
};

std::vector<Row> readFramesCsv(const fs::path& framesDir) {
  const fs::path csv = framesDir / "frames.csv";
  std::ifstream in(csv);
  if (!in) throw VcError("cannot read " + csv.string());
  std::string line;
  std::getline(in, line);
  if (line != "video,frame,pts_ms,reason,motion_score") throw VcError(csv.string() + ": unexpected header '" + line + "'");
  std::vector<Row> rows;
  int lineNo = 1;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.empty()) continue;
    std::vector<std::string> cells;
    std::stringstream stream(line);
    for (std::string cell; std::getline(stream, cell, ',');) cells.push_back(cell);
    const std::string where = csv.string() + " line " + std::to_string(lineNo);
    if (cells.size() != 5) throw VcError(where + ": expected 5 columns");
    try {
      rows.push_back(Row{cells[0], std::stoll(cells[1]), std::stod(cells[2])});
    } catch (const std::exception&) {
      throw VcError(where + ": bad frame or pts_ms");
    }
  }
  return rows;
}

}  // namespace

int64_t runDetectImages(const DetectImagesOptions& options) {
  const std::vector<Row> rows = readFramesCsv(options.framesDir);
  std::ofstream out(options.out);
  if (!out) throw VcError("cannot write " + options.out.string());
  std::unique_ptr<OnnxYoloDetector> detector;
  int width = 0;
  int height = 0;
  for (const Row& row : rows) {
    char name[32];
    std::snprintf(name, sizeof(name), "%06lld.jpg", static_cast<long long>(row.frame));
    const fs::path image = options.framesDir / name;
    VideoReader reader(image.string(), HwAccel::None);
    const VideoFrame* frame = reader.next();
    if (frame == nullptr) throw VcError("cannot decode " + image.string());
    if (!detector || frame->av->width != width || frame->av->height != height) {
      width = frame->av->width;
      height = frame->av->height;
      detector = std::make_unique<OnnxYoloDetector>(options.modelDir, options.provider, options.decode, width, height);
    }
    VideoFrame view = *frame;
    view.index = row.frame;
    view.ptsMs = row.ptsMs;
    for (const Box& b : detector->detect(view)) {
      out << detectionRecord(row.video, row.frame, row.ptsMs, b, detector->classes(), detector->source());
    }
  }
  out.close();
  if (!out) throw VcError("write " + options.out.string() + " failed");
  return static_cast<int64_t>(rows.size());
}

}  // namespace vc
```

`cpp/apps/vc_detect/main.cpp`：

```cpp
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "vc_detect/detect_images.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

const char* kUsage =
    "Usage: vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep cpu|coreml] [--score <x>]\n"
    "\n"
    "Runs the ONNX student model on every frame listed in <frames>/frames.csv and writes\n"
    "detections.jsonl records (for checking C++ against Python inference).\n";

vc::DetectImagesOptions parse(const std::vector<std::string>& args) {
  vc::DetectImagesOptions o;
  bool model = false, frames = false, out = false;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (i + 1 >= args.size()) throw vc::UsageError(arg + " needs a value");
    const std::string& value = args[++i];
    if (arg == "--model") {
      o.modelDir = value;
      model = true;
    } else if (arg == "--frames") {
      o.framesDir = value;
      frames = true;
    } else if (arg == "--out") {
      o.out = value;
      out = true;
    } else if (arg == "--ep") {
      try {
        o.provider = vc::parseExecutionProvider(value);
      } catch (const std::invalid_argument& e) {
        throw vc::UsageError(e.what());
      }
    } else if (arg == "--score") {
      try {
        o.decode.scoreThreshold = std::stof(value);
      } catch (const std::exception&) {
        throw vc::UsageError("--score expects a number");
      }
    } else {
      throw vc::UsageError("unknown argument '" + arg + "'");
    }
  }
  if (!model || !frames || !out) throw vc::UsageError("--model, --frames and --out are required");
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      return 0;
    }
  }
  vc::DetectImagesOptions options;
  try {
    options = parse(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_detect: " << e.what() << "\n\n" << kUsage;
    return 2;
  }
  av_log_set_level(AV_LOG_ERROR);
  try {
    const int64_t images = vc::runDetectImages(options);
    std::printf("vc_detect: %lld images -> %s\n", static_cast<long long>(images), options.out.string().c_str());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "vc_detect: " << e.what() << '\n';
    return 1;
  }
}
```

`cpp/README.md`：
- 开头一句改为同时介绍 `vc_extract`、`vc_redact`、`vc_detect`。
- “依赖”一节加入第三方包（不在 CMake 配置阶段下载，需事先解压到共享缓存）：

````markdown
- onnxruntime 1.30.0 官方预编译包（macOS arm64，含 CoreML EP）与 nlohmann/json 3.12.0 单头文件，解压到所有 checkout 共享的缓存（路径可用 `-DVC_THIRD_PARTY_DIR=…` 覆盖）：

  ```bash
  C=~/.cache/video-compliance/third_party && mkdir -p "$C/nlohmann_json-3.12.0/include/nlohmann"
  curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz
  shasum -a 256 /tmp/ort.tgz   # 6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012
  tar -xzf /tmp/ort.tgz -C "$C"
  curl -L -o "$C/nlohmann_json-3.12.0/include/nlohmann/json.hpp" https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
  shasum -a 256 "$C/nlohmann_json-3.12.0/include/nlohmann/json.hpp"   # aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
  ```
  这两步会联网（约 43MB），只需做一次。
````

- “测试”一节的过滤命令补充：`'Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*'`、`'BoxTimeline.*:Mosaic.*:BoundedQueue.*:RedactConfig.*:Records.*'`、`'VideoReader.*:VideoWriter.*'`、`'RedactJob.*:RedactCli.*:DetectImages.*'`。
- 新增 “vc_redact” 一节：用法示例（`./build/cpp/vc_redact --input data/videos/sg_drive_2100s.mp4 --out data/redacted --model data/models/r1 --config configs/redact.json`）、产物列表（视频、`detections.jsonl`、`masks.jsonl` 每帧一行、`stats.json`）、参数表（同 `redactUsage()`）、要点：检测器出错整段失败且不留半成品；音频直接拷贝；GPS / 字幕等其他流丢弃并在统计中列出；编码格式跟随原片（H.264 / HEVC），VideoToolbox 不可用时退到 libx264 / libx265；`--detect-every-frame` 用于评测时间覆盖率；退出码。
- 新增 “vc_detect” 一节：用法与用途（C++ / Python 推理一致性检查）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build/cpp -j 4 && ./build/cpp/tests/vc_tests --gtest_filter='RedactCli.*:DetectImages.*:ExtractCli.*'`
Expected: 全部通过。

然后按 README 的四组过滤器各跑一次（加上原有的 `'FrameSampler.*:LumaThumb.*'`、`'VideoReader.*:JpegWriter.*'`、`'ExtractCli.*:ExtractJob.*'`），全部通过；`./build/cpp/vc_redact --help` 与 `./build/cpp/vc_detect --help` 退出码 0。

再用 `FakeDetector` 跑一次端到端冒烟（不需要模型，几秒钟）：

```bash
printf '{"video":"basic.mp4","frame":0,"pts_ms":0,"cls":"face","x1":100,"y1":100,"x2":200,"y2":180,"score":0.9,"src":"fake"}\n' > build/cpp/tests/tmp/smoke.jsonl
./build/cpp/vc_redact --input build/cpp/tests/tmp/basic.mp4 --out build/cpp/tests/tmp/smoke --fake-detections build/cpp/tests/tmp/smoke.jsonl --overwrite
```

Expected: 退出码 0，打印各阶段 fps；`build/cpp/tests/tmp/smoke/basic/basic.redacted.mp4` 存在。把输出贴进报告。

- [ ] **Step 5: Commit**

```bash
git add cpp/src/common/vc_error.h cpp/apps/vc_extract/extract_cli.h cpp/apps/vc_redact cpp/apps/vc_detect cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/redact_cli_test.cpp cpp/tests/detect_images_test.cpp cpp/README.md
git commit -m "feat: add vc_redact and vc_detect command lines with docs"
```
