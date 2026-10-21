# 减少误打：复核模型（方案 A）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `vc_redact` 的检测和打码之间加一个轻量复核模型，把尾灯、信号灯、车轮、路牌这类误打拦下来；召回线（人脸 ≥ 95%、车牌 ≥ 98%）不动。

**Architecture:**
- **C++（`vc_redact`）**：对"没把握的框"裁出 64×64 RGB 小图，可选地写入转储文件（训练数据），并交给 ONNX 复核模型打分，低于放行门槛的框不进 `BoxTimeline`。
- **Python 新包 `vc_hardneg`**：从转储和打码记录里挖候选，用"整帧老师 → 本地老师放大复看 → 可选 LLM → 人工"依次判定，先试判 100 张作为放量关卡。
- **Python `vc_train verifier`**：直接用 C++ 转储的小图训练小 CNN 并导出 ONNX，保证训练和运行时看到的像素完全一致。
- **Python `vc_eval`**：新增误打计数命令，`run` 透传复核模型。

**Tech Stack:** C++20、FFmpeg（libswscale）、onnxruntime（CoreML / CPU）、GoogleTest；Python 3.12、PyTorch（MPS）、onnx / onnxruntime、OpenCV、transformers（OWLv2）、pytest。

**设计文档：** `docs/superpowers/specs/2026-10-21-false-positive-filter-design.md`。本计划覆盖它的 §3、§4、§6、§7；§5（重训学生模型 r4）在本计划验收后另写计划。

## 与设计文档的两处细化（已同步写进设计文档 §3.2）

- **训练用的小图由 C++ 直接转储**：`vc_redact --dump-verify-crops` 把复核模型在运行时看到的那张 64×64 RGB 小图原样写出，`vc_train verifier` 直接用它们训练，避免 Python 和 C++ 在裁图、缩放、色彩转换上出现差异。
- **只在标注帧上挖候选**：训练段每 12 帧有老师标注，转储时取每 48 帧一次（`--dump-verify-every 48`），这样每个候选都有整帧老师对照可用。原分辨率小图仍由 Python 从 `data/frames/` 裁出，只用于判定和人工复核。

## Global Constraints

- 召回硬门槛：三个留出段（`sg_drive_1200s`、`2400s`、`60s`）合计人脸 ≥ 95%、车牌 ≥ 98%；基线遮住、新版没遮住的真目标为 0；逐帧参照各分数段覆盖比基线低不超过 1 个百分点。
- 误打目标：对不上老师框的打码，人脸至少减少 50%，车牌至少减少 30%。
- 速度：10 档在 `sg_drive_2400s` 上的吞吐比同时段基线慢不超过 10%。
- 屏上面积 p99 ≤ 10%，帧间跳变 ≤ 5%。
- 挖反例、训练只用 10 个训练段：`sg_drive_300s`、`600s`、`900s`、`1500s`、`1800s`、`2100s`、`2700s`、`3000s`、`3300s`、`3600s`；不用 `sg_full.mp4`。
- "没把握的框"：车牌分数 < `verify_plate_below`（默认 0.5）；人脸只有一个检测器检出（`src` 里没有 `+`），或分数 < `verify_face_unless_both_above`（默认 0.5）。
- 小图：以框为中心的正方形，边长 = max(2.5 × 框长边, 32 px)，超出画面部分填 114，缩放到 64×64 RGB。
- 判定格式：`{"id", "verdict": "yes"|"no"|"unsure", "judge", "score", "detail"}`，追加写入 `judgments.jsonl`；同一 `id` 以最后一条为准，人工优先于模型。
- 100 张试判关卡：判 `no` 的里面真目标 ≤ 2%。
- LLM 裁判缺省拒跑：必须设 `VC_HARDNEG_ALLOW_LLM=1`。
- 放行门槛：验证集里判 `yes` 的小图至少 99.5% 放行。
- 给了 `--verifier-model` 但模型有问题：启动时报错退出，不悄悄跳过复核。不给模型时行为与现在完全一致。
- 本机只跑与改动相关的测试（`vc_tests --gtest_filter=...`、`pytest <文件>`），不跑全量；不联网下载（计费流量）；训练、批量判定这类重任务开始前告知用户，放后台跑。
- 提交信息格式 `<type>: <description>`，结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。所有工作在分支 `feat/fp-filter` 上。

## 文件结构

**C++（`cpp/`）**

| 文件 | 职责 |
|---|---|
| `src/detect/verifier_crop.h/.cpp`（新，`vc_core`） | 小图窗口几何；从 YUV 帧裁出 size×size 的 RGB 字节；字节转平面 float |
| `src/detect/verifier.h/.cpp`（新，`vc_detect_lib`） | 读 `model.json`（`kind: "verifier"`）；ORT 会话；按固定批量给小图打分 |
| `apps/vc_redact/verify_policy.h/.cpp`（新） | "没把握的框"判断、放行门槛选择、`VerifyResult` |
| `apps/vc_redact/verify_dump.h/.cpp`（新） | 转储写入 `verify_crops.bin` + `verify_crops.jsonl` |
| `apps/vc_redact/redact_config.h/.cpp` | 4 个新配置键 |
| `apps/vc_redact/records.h/.cpp` | `detectionRecord` 写 `verify` / `verified` |
| `apps/vc_redact/redact_job.h/.cpp` | 选项、复核与转储接入、统计 |
| `apps/vc_redact/redact_cli.cpp` | `--verifier-model`、`--dump-verify-crops`、`--dump-verify-every` |
| `tests/data/make_const_verifier.py`、`tests/data/const_verifier/` | 常量复核模型夹具 |
| `tests/verifier_crop_test.cpp`、`tests/verifier_test.cpp`、`tests/verify_policy_test.cpp`、`tests/verify_dump_test.cpp`（新） | 单元测试 |

**Python（`py/`）**

| 文件 | 职责 |
|---|---|
| `vc_hardneg/candidates.py` | 读转储与打码记录，对轨迹，去重，整帧老师对照，裁原分辨率小图 |
| `vc_hardneg/judgments.py` | 判定记录的读写与合并 |
| `vc_hardneg/judges.py` | 整帧老师、本地老师复看、命令行 LLM 三种裁判 |
| `vc_hardneg/config.py` | `configs/hardneg.json` 的读取与校验 |
| `vc_hardneg/review.py` | 本地小图墙网页、导入人工结果 |
| `vc_hardneg/pilot.py` | 100 张分层抽样与关卡报告 |
| `vc_hardneg/cli.py`、`__main__.py`、`__init__.py` | `python -m vc_hardneg mine | judge | review | import-human | pilot | pilot-report` |
| `vc_train/verifier.py` | 复核模型的数据、网络、训练、门槛、导出 |
| `vc_train/cli.py` | 新子命令 `verifier` |
| `vc_eval/falsepos.py`、`vc_eval/cli.py`、`vc_eval/run.py` | 误打计数命令；`run --verifier-model` |
| `tests/test_hardneg_*.py`、`tests/test_train_verifier.py`、`tests/test_eval_falsepos.py` | 单元测试 |

**其他：** `configs/hardneg.json`（新）、`configs/redact.json`、`scripts/redact.sh`、`cpp/README.md`、`py/README.md`、`docs/USAGE.md`、`docs/reports/fp-filter-<日期>.md`。

## 构建与测试命令

- **C++ 调试构建**：
  - 构建：`cmake --build build/cpp --target vc_tests -j4`
  - 运行：`./build/cpp/tests/vc_tests --gtest_filter='<Suite>.*'`
- **C++ Release**：`cmake --build build/cpp-release --target vc_redact -j4`
- **Python**：在 `py/` 下运行 `../.venv/bin/python -m pytest tests/<文件> -q`
- **新增 C++ 源文件后**：把它加进 `cpp/CMakeLists.txt` 对应的库，测试文件加进 `cpp/tests/CMakeLists.txt` 的 `vc_tests`，再重新运行 `cmake -S cpp -B build/cpp` 一次。

---

### Task 1: 小图裁取（C++ `verifier_crop`）

**Files:**
- Create: `cpp/src/detect/verifier_crop.h`, `cpp/src/detect/verifier_crop.cpp`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 加 `src/detect/verifier_crop.cpp`）、`cpp/tests/CMakeLists.txt`（加 `verifier_crop_test.cpp`）
- Test: `cpp/tests/verifier_crop_test.cpp`

**Interfaces:**
- Produces:
  - `struct CropWindow { double x; double y; double side; };`
  - `struct CropParams { int size = 64; double factor = 2.5; double minPx = 32; int padValue = 114; };`
  - `CropWindow verifierWindow(const Box& box, double factor, double minPx);`
  - `class CropSampler { explicit CropSampler(const CropParams&); void fill(const AVFrame& frame, const CropWindow& window, uint8_t* rgb); const CropParams& params() const; };`，`rgb` 为 size*size*3 字节，交错 RGB。
  - `void toPlanarFloat(const uint8_t* rgb, int size, float* out);`，`out` 为 3*size*size，R、G、B 三个平面，值 /255。

- [ ] **Step 1: 写失败的测试**

`cpp/tests/verifier_crop_test.cpp`:

```cpp
#include "detect/verifier_crop.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "io/av_ptr.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace vc {
namespace {

// The RGB24 conversion of the whole frame, the reference a crop at scale 1 must reproduce.
std::vector<uint8_t> fullRgb(const AVFrame& frame) {
  av::SwsPtr sws = av::allocSws();
  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = frame.width;
  rgb->height = frame.height;
  EXPECT_GE(sws_scale_frame(sws.get(), rgb.get(), &frame), 0);
  std::vector<uint8_t> out(static_cast<size_t>(frame.width) * frame.height * 3);
  for (int y = 0; y < frame.height; ++y) {
    std::copy_n(rgb->data[0] + static_cast<ptrdiff_t>(y) * rgb->linesize[0], frame.width * 3,
                out.data() + static_cast<size_t>(y) * frame.width * 3);
  }
  return out;
}

TEST(VerifierCrop, WindowIsASquareAroundTheBoxCentre) {
  const CropWindow w = verifierWindow(Box{0, 100, 200, 140, 210, 0.5f}, 2.5, 32);  // 40 x 10
  EXPECT_DOUBLE_EQ(w.side, 100);  // 2.5 * 40
  EXPECT_DOUBLE_EQ(w.x, 70);      // centre 120 - 50
  EXPECT_DOUBLE_EQ(w.y, 155);     // centre 205 - 50
}

TEST(VerifierCrop, WindowIsAtLeastMinPx) {
  const CropWindow w = verifierWindow(Box{0, 10, 10, 16, 14, 0.5f}, 2.5, 32);  // 6 x 4: 2.5 * 6 = 15 < 32
  EXPECT_DOUBLE_EQ(w.side, 32);
  EXPECT_DOUBLE_EQ(w.x, 13 - 16);
  EXPECT_DOUBLE_EQ(w.y, 12 - 16);
}

// A window inside the frame at scale 1 (side == size) is the frame's own RGB conversion, pixel for pixel.
TEST(VerifierCrop, AtScaleOneTheCropIsTheFramesRgb) {
  auto frame = test::makeFrame(128, 96, AV_PIX_FMT_YUV420P, 0, 128, 128);
  for (int y = 0; y < 96; ++y) {  // a luma ramp, chroma neutral: no chroma edge to blur
    for (int x = 0; x < 128; ++x) frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>(16 + x + y);
  }
  const std::vector<uint8_t> ref = fullRgb(*frame);
  CropSampler sampler(CropParams{.size = 32});
  std::vector<uint8_t> rgb(32 * 32 * 3);
  sampler.fill(*frame, CropWindow{40, 20, 32}, rgb.data());
  for (int v = 0; v < 32; ++v) {
    for (int u = 0; u < 32; ++u) {
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(rgb[(v * 32 + u) * 3 + c], ref[((20 + v) * 128 + 40 + u) * 3 + c]) << u << "," << v << "," << c;
      }
    }
  }
}

// What lies outside the frame is the pad value; a solid frame gives its own colour everywhere else.
TEST(VerifierCrop, OutsideTheFrameIsPadded) {
  auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 180, 128, 128);
  const std::vector<uint8_t> ref = fullRgb(*frame);
  CropSampler sampler(CropParams{.size = 16, .padValue = 114});
  std::vector<uint8_t> rgb(16 * 16 * 3);
  sampler.fill(*frame, CropWindow{-32, -32, 64}, rgb.data());  // the frame covers the bottom-right quarter
  for (int v = 0; v < 16; ++v) {
    for (int u = 0; u < 16; ++u) {
      const bool inside = u >= 8 && v >= 8;
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(rgb[(v * 16 + u) * 3 + c], inside ? ref[c] : 114) << u << "," << v;
      }
    }
  }
}

TEST(VerifierCrop, AWindowWhollyOutsideIsAllPad) {
  auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 180, 128, 128);
  CropSampler sampler(CropParams{.size = 8});
  std::vector<uint8_t> rgb(8 * 8 * 3, 0);
  sampler.fill(*frame, CropWindow{200, 200, 40}, rgb.data());
  for (uint8_t b : rgb) ASSERT_EQ(b, 114);
}

TEST(VerifierCrop, ToPlanarFloatSplitsTheChannels) {
  const std::vector<uint8_t> rgb = {255, 0, 51, 0, 255, 102};  // two pixels of a 1 x 2 "image" read as size 1 twice
  std::vector<float> out(3);
  toPlanarFloat(rgb.data(), 1, out.data());
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 0.0f);
  EXPECT_FLOAT_EQ(out[2], 51.0f / 255.0f);
}

}  // namespace
}  // namespace vc
```

- [ ] **Step 2: 加进构建，运行，确认失败**

在 `cpp/tests/CMakeLists.txt` 的 `vc_tests` 列表里加 `verifier_crop_test.cpp`（按字母序放在 `video_writer_test.cpp` 前）。然后运行：

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -m3 error
```

Expected: 编译错误 `'detect/verifier_crop.h' file not found`。

- [ ] **Step 3: 写实现**

`cpp/src/detect/verifier_crop.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include "detect/box.h"

struct AVFrame;

namespace vc {

// The square of frame pixels a verifier crop shows (spec 2026-10-21 §3.2): centred on the box, side
// max(factor * the box's long side, minPx). It may reach past the frame.
struct CropWindow {
  double x = 0;  // left edge, frame pixels
  double y = 0;  // top edge
  double side = 0;
};

// What a verifier crop is: size x size RGB, the window grown by `factor`, at least `minPx` frame pixels, the
// part outside the frame filled with padValue. The defaults are spec 2026-10-21 §3.2's.
struct CropParams {
  int size = 64;
  double factor = 2.5;
  double minPx = 32;
  int padValue = 114;
};

CropWindow verifierWindow(const Box& box, double factor, double minPx);

// Cuts verifier crops out of 8-bit YUV frames: the window's part inside the frame is converted to RGB24 at the
// frame's own resolution (libswscale, the frame's colour space and range, as the detector's input), then
// resampled bilinearly to size x size; output pixels whose centre falls outside the frame get padValue. Not
// thread-safe.
class CropSampler {
 public:
  explicit CropSampler(const CropParams& params);
  ~CropSampler();
  CropSampler(const CropSampler&) = delete;
  CropSampler& operator=(const CropSampler&) = delete;

  const CropParams& params() const;
  // rgb: size * size * 3 bytes, interleaved R, G, B, row by row. Throws VcError when the conversion fails.
  void fill(const AVFrame& frame, const CropWindow& window, uint8_t* rgb);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// rgb (size * size * 3, interleaved) -> out (3 * size * size: planes R, G, B), each byte / 255: the verifier's input.
void toPlanarFloat(const uint8_t* rgb, int size, float* out);

}  // namespace vc
```

`cpp/src/detect/verifier_crop.cpp`:

```cpp
#include "detect/verifier_crop.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace vc {

CropWindow verifierWindow(const Box& box, double factor, double minPx) {
  const double longSide = std::max(box.width(), box.height());
  const double side = std::max(factor * longSide, minPx);
  const double cx = (static_cast<double>(box.x1) + box.x2) / 2.0;
  const double cy = (static_cast<double>(box.y1) + box.y2) / 2.0;
  return CropWindow{cx - side / 2.0, cy - side / 2.0, side};
}

struct CropSampler::Impl {
  CropParams params;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr view = av::allocFrame();
  av::FramePtr rgb = av::allocFrame();
};

CropSampler::CropSampler(const CropParams& params) : impl_(std::make_unique<Impl>()) {
  if (params.size < 1 || params.padValue < 0 || params.padValue > 255 || !(params.factor > 0) ||
      !(params.minPx >= 0)) {
    throw std::invalid_argument("invalid verifier crop parameters");
  }
  impl_->params = params;
  impl_->sws->flags = SWS_BILINEAR;
}

CropSampler::~CropSampler() = default;

const CropParams& CropSampler::params() const { return impl_->params; }

void CropSampler::fill(const AVFrame& frame, const CropWindow& window, uint8_t* rgb) {
  Impl& s = *impl_;
  const int size = s.params.size;
  const auto pad = static_cast<uint8_t>(s.params.padValue);
  std::fill(rgb, rgb + static_cast<size_t>(size) * size * 3, pad);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
  if (desc == nullptr) throw VcError("verifier crop: unknown pixel format");
  // The frame pixels the window covers, grown to the chroma grid so the sub-frame starts on a chroma sample.
  const int alignX = 1 << desc->log2_chroma_w;
  const int alignY = 1 << desc->log2_chroma_h;
  const auto clampI = [](double v, int hi) { return static_cast<int>(std::clamp(v, 0.0, static_cast<double>(hi))); };
  int x0 = clampI(std::floor(window.x), frame.width) / alignX * alignX;
  int y0 = clampI(std::floor(window.y), frame.height) / alignY * alignY;
  const int x1 = clampI(std::ceil(window.x + window.side), frame.width);
  const int y1 = clampI(std::ceil(window.y + window.side), frame.height);
  if (x1 <= x0 || y1 <= y0) return;  // wholly outside: all pad
  const int w = x1 - x0;
  const int h = y1 - y0;

  // A view of the frame's [x0, x1) x [y0, y1) without copying: plane pointers moved to the corner.
  AVFrame& v = *s.view;
  av_frame_unref(&v);
  v.format = frame.format;
  v.width = w;
  v.height = h;
  v.color_range = frame.color_range;
  v.colorspace = frame.colorspace;
  v.color_primaries = frame.color_primaries;
  v.color_trc = frame.color_trc;
  for (int p = 0; p < AV_NUM_DATA_POINTERS && frame.data[p] != nullptr; ++p) {
    const bool chroma = p == 1 || p == 2;
    const int px = chroma ? x0 >> desc->log2_chroma_w : x0;
    const int py = chroma ? y0 >> desc->log2_chroma_h : y0;
    const int step = desc->comp[std::min(p, desc->nb_components - 1)].step;  // bytes per sample on that plane
    v.data[p] = frame.data[p] + static_cast<ptrdiff_t>(py) * frame.linesize[p] + static_cast<ptrdiff_t>(px) * step;
    v.linesize[p] = frame.linesize[p];
  }
  av_frame_unref(s.rgb.get());
  s.rgb->format = AV_PIX_FMT_RGB24;
  s.rgb->width = w;
  s.rgb->height = h;
  checkAv(sws_scale_frame(s.sws.get(), s.rgb.get(), &v), "convert verifier crop to RGB");

  // Bilinear resampling: output pixel (u, v) shows frame point (window.x + (u + 0.5) * scale - 0.5, ...).
  const double scale = window.side / size;
  const uint8_t* src = s.rgb->data[0];
  const int stride = s.rgb->linesize[0];
  for (int oy = 0; oy < size; ++oy) {
    const double fy = window.y + (oy + 0.5) * scale - 0.5;
    if (fy < -0.5 || fy >= frame.height - 0.5) continue;
    const double ly = std::clamp(fy - y0, 0.0, static_cast<double>(h - 1));
    const int ya = static_cast<int>(ly);
    const int yb = std::min(ya + 1, h - 1);
    const double ty = ly - ya;
    for (int ox = 0; ox < size; ++ox) {
      const double fx = window.x + (ox + 0.5) * scale - 0.5;
      if (fx < -0.5 || fx >= frame.width - 0.5) continue;
      const double lx = std::clamp(fx - x0, 0.0, static_cast<double>(w - 1));
      const int xa = static_cast<int>(lx);
      const int xb = std::min(xa + 1, w - 1);
      const double tx = lx - xa;
      for (int c = 0; c < 3; ++c) {
        const double a = src[ya * stride + xa * 3 + c] * (1 - tx) + src[ya * stride + xb * 3 + c] * tx;
        const double b = src[yb * stride + xa * 3 + c] * (1 - tx) + src[yb * stride + xb * 3 + c] * tx;
        rgb[(static_cast<size_t>(oy) * size + ox) * 3 + c] = static_cast<uint8_t>(std::lround(a * (1 - ty) + b * ty));
      }
    }
  }
}

void toPlanarFloat(const uint8_t* rgb, int size, float* out) {
  const size_t plane = static_cast<size_t>(size) * size;
  for (size_t i = 0; i < plane; ++i) {
    out[i] = rgb[i * 3] / 255.0f;
    out[plane + i] = rgb[i * 3 + 1] / 255.0f;
    out[2 * plane + i] = rgb[i * 3 + 2] / 255.0f;
  }
}

}  // namespace vc
```

在 `cpp/CMakeLists.txt` 的 `vc_core` 列表里，`src/detect/letterbox.cpp` 下面加一行 `src/detect/verifier_crop.cpp`。

- [ ] **Step 4: 运行测试，确认通过**

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -E "error|warning"; ./build/cpp/tests/vc_tests --gtest_filter='VerifierCrop.*'
```

Expected: `[  PASSED  ] 6 tests.`

`AtScaleOneTheCropIsTheFramesRgb` 若在边缘差 1：说明 swscale 对子区域边缘的色度插值与整帧不同，而这个测试的色度是常量，不应该出现。此时检查 `v.data[p]` 的偏移（`step`），不要放宽断言。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/detect/verifier_crop.h cpp/src/detect/verifier_crop.cpp cpp/tests/verifier_crop_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt
git commit -m "feat: cut verifier crops from YUV frames" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 复核模型的读取与推理（C++ `Verifier`）

**Files:**
- Create: `cpp/src/detect/verifier.h`, `cpp/src/detect/verifier.cpp`, `cpp/tests/data/make_const_verifier.py`, `cpp/tests/data/const_verifier/model.json`, `cpp/tests/data/const_verifier/model.onnx`（脚本生成）
- Modify: `cpp/CMakeLists.txt`（`vc_detect_lib` 加 `src/detect/verifier.cpp`）、`cpp/tests/CMakeLists.txt`（加 `verifier_test.cpp`）
- Test: `cpp/tests/verifier_test.cpp`

**Interfaces:**
- Consumes: `createOrtSession`, `SessionCaches`, `coreMlCacheDirectory`, `tensorRtCacheDirectory`, `readFloatTensorSpec`, `shapeText`（`detect/ort_session.h`），`ExecutionProvider`、`SessionSettings`（`detect/execution_provider.h`）。
- Produces:
  - `struct VerifierInfo { std::string name; std::vector<std::string> classes; int inputSize; int batch; int padValue; double cropFactor; double minCropPx; std::vector<float> passThreshold; CropParams crop() const; };`
  - `VerifierInfo parseVerifierInfo(const std::string& text, const std::string& origin);`、`VerifierInfo readVerifierInfo(const std::filesystem::path&);`
  - `struct VerifierInput { int cls; const float* pixels; };`（`cls` 是 `classes` 的下标，`pixels` 为 3*size*size）
  - `class Verifier { Verifier(const fs::path& modelDir, ExecutionProvider, const SessionSettings&); const VerifierInfo& info() const; std::string backend() const; std::vector<float> score(const std::vector<VerifierInput>&); };`

常量夹具的输出：logit 固定为人脸 −2（sigmoid ≈ 0.119）、车牌 +2（≈ 0.881）；`model.json` 两类门槛都是 0.5，批量 4，尺寸 64。

- [ ] **Step 1: 写夹具脚本并生成模型**

`cpp/tests/data/make_const_verifier.py`:

```python
"""Writes <out>/model.onnx: crops [4,3,64,64] -> logits [4,2] = (-2, +2) per row + 0 * mean(crops).

The face logit -2 (sigmoid 0.119) fails and the plate logit +2 (0.881) passes the 0.5 thresholds of the
model.json next to it, so tests can tell a rejected face from a kept plate.

Run once with the repo venv; the output is committed:
    .venv/bin/python -I cpp/tests/data/make_const_verifier.py cpp/tests/data/const_verifier
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BATCH, SIZE = 4, 64


def main(out_dir: Path) -> None:
    logits = np.tile(np.array([[-2.0, 2.0]], np.float32), (BATCH, 1))
    nodes = [
        helper.make_node("ReduceMean", ["crops"], ["mean"], axes=[1, 2, 3], keepdims=1),  # [4, 1, 1, 1]
        helper.make_node("Reshape", ["mean", "shape"], ["flat"]),  # [4, 1]
        helper.make_node("Mul", ["flat", "zero"], ["zeroed"]),
        helper.make_node("Add", ["logits_const", "zeroed"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "const_verifier",
        [helper.make_tensor_value_info("crops", TensorProto.FLOAT, [BATCH, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, 2])],
        initializer=[
            numpy_helper.from_array(logits, "logits_const"),
            numpy_helper.from_array(np.zeros(1, np.float32), "zero"),
            numpy_helper.from_array(np.array([BATCH, 1], np.int64), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_dir / "model.onnx"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_const_verifier.py <out_dir>")
    main(Path(sys.argv[1]))
```

`cpp/tests/data/const_verifier/model.json`:

```json
{
  "name": "const_verifier",
  "kind": "verifier",
  "classes": ["face", "plate"],
  "input": {"size": 64, "batch": 4, "color": "RGB", "scale": 0.00392156862745098, "pad_value": 114},
  "crop": {"factor": 2.5, "min_px": 32},
  "pass_threshold": {"face": 0.5, "plate": 0.5},
  "metrics": {}
}
```

Run: `.venv/bin/python -I cpp/tests/data/make_const_verifier.py cpp/tests/data/const_verifier`
Expected: `cpp/tests/data/const_verifier/model.onnx` 生成，无输出。

- [ ] **Step 2: 写失败的测试**

`cpp/tests/verifier_test.cpp`:

```cpp
#include "detect/verifier.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path constVerifier() { return fs::path(VC_TEST_DATA_DIR) / "const_verifier"; }

std::string readText(const fs::path& p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

TEST(VerifierInfo, ReadsTheConstModel) {
  const VerifierInfo info = readVerifierInfo(constVerifier() / "model.json");
  EXPECT_EQ(info.name, "const_verifier");
  EXPECT_EQ(info.classes, (std::vector<std::string>{"face", "plate"}));
  EXPECT_EQ(info.inputSize, 64);
  EXPECT_EQ(info.batch, 4);
  EXPECT_EQ(info.padValue, 114);
  EXPECT_DOUBLE_EQ(info.cropFactor, 2.5);
  EXPECT_DOUBLE_EQ(info.minCropPx, 32);
  EXPECT_EQ(info.passThreshold, (std::vector<float>{0.5f, 0.5f}));
  EXPECT_EQ(info.crop().size, 64);
}

TEST(VerifierInfo, RefusesAnotherKindOrMissingThresholds) {
  std::string text = readText(constVerifier() / "model.json");
  const std::string kind = R"("kind": "verifier")";
  const std::string other = R"("kind": "detector")";
  std::string wrongKind = text;
  wrongKind.replace(wrongKind.find(kind), kind.size(), other);
  EXPECT_THROW(parseVerifierInfo(wrongKind, "m.json"), VcError);
  std::string noPlate = text;
  const std::string plate = R"(, "plate": 0.5)";
  noPlate.erase(noPlate.find(plate), plate.size());
  EXPECT_THROW(parseVerifierInfo(noPlate, "m.json"), VcError);
  EXPECT_THROW(parseVerifierInfo("[]", "m.json"), VcError);
}

TEST(Verifier, ScoresEachCropByItsClass) {
  Verifier v(constVerifier(), ExecutionProvider::Cpu, SessionSettings{});
  EXPECT_EQ(v.backend(), "CPU");
  std::vector<float> pixels(3 * 64 * 64, 0.5f);
  // 6 crops: more than one batch of 4, so the second batch is padded
  std::vector<VerifierInput> in;
  for (int i = 0; i < 6; ++i) in.push_back(VerifierInput{i % 2, pixels.data()});
  const std::vector<float> s = v.score(in);
  ASSERT_EQ(s.size(), 6u);
  const float face = 1.0f / (1.0f + std::exp(2.0f));
  const float plate = 1.0f / (1.0f + std::exp(-2.0f));
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(s[i], i % 2 == 0 ? face : plate, 1e-5) << i;
  EXPECT_TRUE(v.score({}).empty());
}

TEST(Verifier, RefusesAMissingModelOrAShapeThatDisagreesWithModelJson) {
  EXPECT_THROW(Verifier(test::tmpDir() / "no_such_verifier", ExecutionProvider::Cpu, SessionSettings{}), VcError);
  const fs::path dir = test::tmpDir() / "verifier_wrong_batch";
  fs::create_directories(dir);
  fs::copy_file(constVerifier() / "model.onnx", dir / "model.onnx", fs::copy_options::overwrite_existing);
  std::string text = readText(constVerifier() / "model.json");
  text.replace(text.find(R"("batch": 4)"), 10, R"("batch": 8)");
  std::ofstream(dir / "model.json") << text;
  EXPECT_THROW(Verifier(dir, ExecutionProvider::Cpu, SessionSettings{}), VcError);
}

}  // namespace
}  // namespace vc
```

把 `verifier_test.cpp` 加进 `cpp/tests/CMakeLists.txt`（放在 `verifier_crop_test.cpp` 后），然后运行：

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -m3 error
```

Expected: 编译错误 `'detect/verifier.h' file not found`。

- [ ] **Step 3: 写实现**

`cpp/src/detect/verifier.h`:

```cpp
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "detect/execution_provider.h"
#include "detect/verifier_crop.h"

namespace vc {

// The model.json of a verifier (`vc_train verifier`, spec 2026-10-21 §4.1).
struct VerifierInfo {
  std::string name;
  std::vector<std::string> classes;  // the order of the model's logits
  int inputSize = 0;                 // crops are inputSize x inputSize RGB
  int batch = 0;                     // the model's fixed batch
  int padValue = 114;
  double cropFactor = 2.5;
  double minCropPx = 32;
  std::vector<float> passThreshold;  // per class, in `classes` order, in [0, 1]

  CropParams crop() const { return CropParams{inputSize, cropFactor, minCropPx, padValue}; }
};

// Throws VcError("<origin>: …") for invalid JSON, kind other than "verifier", color other than "RGB", scale
// other than 1/255, sizes that are not positive JSON integers, a pad_value outside [0, 255], a crop factor
// not > 0, min_px < 0, or a class without a pass_threshold in [0, 1].
VerifierInfo parseVerifierInfo(const std::string& text, const std::string& origin);
VerifierInfo readVerifierInfo(const std::filesystem::path& path);

struct VerifierInput {
  int cls = 0;                    // index into VerifierInfo::classes
  const float* pixels = nullptr;  // 3 * inputSize * inputSize floats (toPlanarFloat)
};

// The verifier's onnxruntime session: model.onnx with input [batch, 3, size, size] and output [batch, n classes],
// both float32. score() is not thread-safe.
class Verifier {
 public:
  // Throws VcError when the files are missing or invalid, when the ONNX shapes disagree with model.json, or
  // when onnxruntime fails (also on the CPU fallback).
  Verifier(const std::filesystem::path& modelDir, ExecutionProvider provider, const SessionSettings& settings);
  ~Verifier();
  Verifier(const Verifier&) = delete;
  Verifier& operator=(const Verifier&) = delete;

  const VerifierInfo& info() const;
  std::string backend() const;  // providerLabel of the session's provider
  // The pass probability of each input: sigmoid of its class's logit, in input order. Runs ceil(n / batch)
  // batches, the last one padded with pad-value crops. Throws VcError when a class index is out of range or
  // inference fails.
  std::vector<float> score(const std::vector<VerifierInput>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
```

`cpp/src/detect/verifier.cpp`:

```cpp
#include "detect/verifier.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/ort_session.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

const nlohmann::json& member(const nlohmann::json& j, const char* key, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) throw VcError(origin + ": missing " + key);
  return *it;
}

int positiveInt(const nlohmann::json& j, const char* key, const std::string& origin) {
  const nlohmann::json& v = member(j, key, origin);
  if (!v.is_number_integer() || v.get<int64_t>() < 1 || v.get<int64_t>() > std::numeric_limits<int>::max()) {
    throw VcError(origin + ": " + key + " must be a positive integer");
  }
  return static_cast<int>(v.get<int64_t>());
}

double number(const nlohmann::json& j, const char* key, const std::string& origin) {
  const nlohmann::json& v = member(j, key, origin);
  if (!v.is_number() || !std::isfinite(v.get<double>())) throw VcError(origin + ": " + key + " must be a number");
  return v.get<double>();
}

}  // namespace

VerifierInfo parseVerifierInfo(const std::string& text, const std::string& origin) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  if (!j.is_object()) throw VcError(origin + ": the root must be a JSON object");
  if (member(j, "kind", origin) != "verifier") throw VcError(origin + ": kind must be \"verifier\"");
  VerifierInfo info;
  const nlohmann::json& name = member(j, "name", origin);
  if (!name.is_string()) throw VcError(origin + ": name must be a string");
  info.name = name.get<std::string>();
  const nlohmann::json& classes = member(j, "classes", origin);
  if (!classes.is_array() || classes.empty()) throw VcError(origin + ": classes must be a non-empty array");
  for (const auto& c : classes) {
    if (!c.is_string()) throw VcError(origin + ": classes must be strings");
    info.classes.push_back(c.get<std::string>());
  }
  const nlohmann::json& input = member(j, "input", origin);
  info.inputSize = positiveInt(input, "size", origin);
  info.batch = positiveInt(input, "batch", origin);
  if (member(input, "color", origin) != "RGB") throw VcError(origin + ": input color must be \"RGB\"");
  if (std::abs(number(input, "scale", origin) - 1.0 / 255.0) > 1e-9) throw VcError(origin + ": input scale must be 1/255");
  const nlohmann::json& pad = member(input, "pad_value", origin);
  if (!pad.is_number_integer() || pad.get<int64_t>() < 0 || pad.get<int64_t>() > 255) {
    throw VcError(origin + ": pad_value must be an integer in [0, 255]");
  }
  info.padValue = static_cast<int>(pad.get<int64_t>());
  const nlohmann::json& crop = member(j, "crop", origin);
  info.cropFactor = number(crop, "factor", origin);
  info.minCropPx = number(crop, "min_px", origin);
  if (!(info.cropFactor > 0) || info.minCropPx < 0) throw VcError(origin + ": crop factor must be > 0, min_px >= 0");
  const nlohmann::json& thresholds = member(j, "pass_threshold", origin);
  for (const std::string& c : info.classes) {
    const double t = number(thresholds, c.c_str(), origin);
    if (t < 0 || t > 1) throw VcError(origin + ": pass_threshold of " + c + " must be in [0, 1]");
    info.passThreshold.push_back(static_cast<float>(t));
  }
  return info;
}

VerifierInfo readVerifierInfo(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parseVerifierInfo(text.str(), path.string());
}

struct Verifier::Impl {
  VerifierInfo info;
  Ort::Session session{nullptr};
  std::string backend = "CPU";
  std::string inputName;
  std::string outputName;
  std::vector<float> input;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  void checkShapes(const fs::path& onnx) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw VcError(onnx.string() + ": expected one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
    const std::vector<int64_t> in = readFloatTensorSpec(session.GetInputTypeInfo(0), "input", onnx).shape;
    const std::vector<int64_t> wantIn{info.batch, 3, info.inputSize, info.inputSize};
    if (in != wantIn) {
      throw VcError(onnx.string() + ": input shape " + shapeText(in) + " does not match model.json " + shapeText(wantIn));
    }
    const std::vector<int64_t> out = readFloatTensorSpec(session.GetOutputTypeInfo(0), "output", onnx).shape;
    const std::vector<int64_t> wantOut{info.batch, static_cast<int64_t>(info.classes.size())};
    if (out != wantOut) {
      throw VcError(onnx.string() + ": output shape " + shapeText(out) + " does not match model.json " +
                    shapeText(wantOut));
    }
  }
};

Verifier::Verifier(const fs::path& modelDir, ExecutionProvider provider, const SessionSettings& settings)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.info = readVerifierInfo(modelDir / "model.json");
  const fs::path onnx = modelDir / "model.onnx";
  if (!fs::is_regular_file(onnx)) throw VcError("missing " + onnx.string());
  const int size = s.info.inputSize;
  try {
    const SessionCaches caches{[&] { return coreMlCacheDirectory(onnx); },
                               [&] { return tensorRtCacheDirectory(onnx, size, size); }};
    CreatedSession created = createOrtSession(onnx, provider, caches, settings);
    s.session = std::move(created.session);
    s.backend = created.backend;
    s.checkShapes(onnx);
  } catch (const Ort::Exception& e) {
    throw VcError("onnxruntime: " + onnx.string() + ": " + e.what());
  }
  s.input.resize(static_cast<size_t>(s.info.batch) * 3 * size * size);
}

Verifier::~Verifier() = default;

const VerifierInfo& Verifier::info() const { return impl_->info; }

std::string Verifier::backend() const { return impl_->backend; }

std::vector<float> Verifier::score(const std::vector<VerifierInput>& inputs) {
  Impl& s = *impl_;
  const size_t per = 3 * static_cast<size_t>(s.info.inputSize) * s.info.inputSize;
  const size_t n = s.info.classes.size();
  const auto batch = static_cast<size_t>(s.info.batch);
  std::vector<float> scores;
  scores.reserve(inputs.size());
  for (size_t start = 0; start < inputs.size(); start += batch) {
    const size_t count = std::min(batch, inputs.size() - start);
    std::fill(s.input.begin(), s.input.end(), static_cast<float>(s.info.padValue) / 255.0f);
    for (size_t i = 0; i < count; ++i) {
      const VerifierInput& in = inputs[start + i];
      if (in.cls < 0 || static_cast<size_t>(in.cls) >= n) throw VcError("verifier: class index out of range");
      std::copy_n(in.pixels, per, s.input.data() + i * per);
    }
    try {
      const std::array<int64_t, 4> shape{s.info.batch, 3, s.info.inputSize, s.info.inputSize};
      Ort::Value tensor =
          Ort::Value::CreateTensor<float>(s.memory, s.input.data(), s.input.size(), shape.data(), shape.size());
      const char* inNames[] = {s.inputName.c_str()};
      const char* outNames[] = {s.outputName.c_str()};
      std::vector<Ort::Value> out = s.session.Run(Ort::RunOptions{nullptr}, inNames, &tensor, 1, outNames, 1);
      const float* logits = out[0].GetTensorData<float>();
      for (size_t i = 0; i < count; ++i) {
        const float logit = logits[i * n + static_cast<size_t>(inputs[start + i].cls)];
        scores.push_back(1.0f / (1.0f + std::exp(-logit)));
      }
    } catch (const Ort::Exception& e) {
      throw VcError(std::string("onnxruntime: verifier inference failed: ") + e.what());
    }
  }
  return scores;
}

}  // namespace vc
```

在 `cpp/CMakeLists.txt` 的 `vc_detect_lib` 列表末尾（`src/detect/yunet_detector.cpp` 后）加 `src/detect/verifier.cpp`。

- [ ] **Step 4: 运行测试，确认通过**

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -E "error|warning"; ./build/cpp/tests/vc_tests --gtest_filter='Verifier*'
```

Expected: `[  PASSED  ] 10 tests.`（`VerifierCrop` 6 个 + `VerifierInfo` 2 个 + `Verifier` 2 个）

- [ ] **Step 5: 提交**

```bash
git add cpp/src/detect/verifier.h cpp/src/detect/verifier.cpp cpp/tests/verifier_test.cpp cpp/tests/data/make_const_verifier.py cpp/tests/data/const_verifier cpp/CMakeLists.txt cpp/tests/CMakeLists.txt
git commit -m "feat: load and run a verifier model" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 配置键与"没把握的框"判断

**Files:**
- Create: `cpp/apps/vc_redact/verify_policy.h`, `cpp/apps/vc_redact/verify_policy.cpp`, `cpp/tests/verify_policy_test.cpp`
- Modify: `cpp/apps/vc_redact/redact_config.h`, `cpp/apps/vc_redact/redact_config.cpp`, `configs/redact.json`, `cpp/tests/redact_config_test.cpp`, `cpp/CMakeLists.txt`（`vc_redact_lib` 加 `apps/vc_redact/verify_policy.cpp`）、`cpp/tests/CMakeLists.txt`（加 `verify_policy_test.cpp`）

**Interfaces:**
- Consumes: `SourcedBox`（`detect/detector.h`），`VerifierInfo`（Task 2）。
- Produces:
  - `RedactConfig` 新字段：`float verifyPlateBelow = 0.5f; float verifyFaceUnlessBothAbove = 0.5f; std::optional<float> verifyThresholdFace; std::optional<float> verifyThresholdPlate;`
  - `struct VerifyResult { float score = 0; bool passed = true; };`
  - `bool needsVerification(const SourcedBox& box, const std::string& className, const RedactConfig& config);`
  - `float passThreshold(const VerifierInfo& info, int verifierClass, const RedactConfig& config);`

- [ ] **Step 1: 写失败的测试**

`cpp/tests/verify_policy_test.cpp`:

```cpp
#include "vc_redact/verify_policy.h"

#include <gtest/gtest.h>

namespace vc {
namespace {

SourcedBox box(float score, const std::string& src) { return SourcedBox{Box{0, 0, 0, 10, 10, score}, src}; }

TEST(VerifyPolicy, PlatesBelowTheBandAreChecked) {
  const RedactConfig c;
  EXPECT_TRUE(needsVerification(box(0.49f, "student"), "plate", c));
  EXPECT_FALSE(needsVerification(box(0.5f, "student"), "plate", c));
}

TEST(VerifyPolicy, FacesAreCheckedUnlessBothDetectorsFoundThemConfidently) {
  const RedactConfig c;
  EXPECT_TRUE(needsVerification(box(0.95f, "yunet"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.95f, "student"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.95f, "fake"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.49f, "student+yunet"), "face", c));
  EXPECT_FALSE(needsVerification(box(0.5f, "student+yunet"), "face", c));
}

TEST(VerifyPolicy, OtherClassesAreNeverChecked) {
  EXPECT_FALSE(needsVerification(box(0.01f, "student"), "vehicle", RedactConfig{}));
}

TEST(VerifyPolicy, TheConfigOverridesTheModelsThreshold) {
  VerifierInfo info;
  info.classes = {"face", "plate"};
  info.passThreshold = {0.2f, 0.3f};
  RedactConfig c;
  EXPECT_FLOAT_EQ(passThreshold(info, 0, c), 0.2f);
  EXPECT_FLOAT_EQ(passThreshold(info, 1, c), 0.3f);
  c.verifyThresholdFace = 0.6f;
  c.verifyThresholdPlate = 0.7f;
  EXPECT_FLOAT_EQ(passThreshold(info, 0, c), 0.6f);
  EXPECT_FLOAT_EQ(passThreshold(info, 1, c), 0.7f);
}

}  // namespace
}  // namespace vc
```

在 `cpp/tests/redact_config_test.cpp` 末尾（`}  // namespace` 之前）加：

```cpp
TEST(RedactConfig, VerifySettingsDefaultToTheSpec) {
  const RedactConfig c;
  EXPECT_FLOAT_EQ(c.verifyPlateBelow, 0.5f);
  EXPECT_FLOAT_EQ(c.verifyFaceUnlessBothAbove, 0.5f);
  EXPECT_FALSE(c.verifyThresholdFace.has_value());
  EXPECT_FALSE(c.verifyThresholdPlate.has_value());
}

TEST(RedactConfig, ReadsAndWritesTheVerifySettings) {
  const RedactConfig c = parseRedactConfig(
      R"({"verify_plate_below": 0.4, "verify_face_unless_both_above": 0.6, "verify_threshold_face": 0.25,
          "verify_threshold_plate": null})",
      "t");
  EXPECT_FLOAT_EQ(c.verifyPlateBelow, 0.4f);
  EXPECT_FLOAT_EQ(c.verifyFaceUnlessBothAbove, 0.6f);
  ASSERT_TRUE(c.verifyThresholdFace.has_value());
  EXPECT_FLOAT_EQ(*c.verifyThresholdFace, 0.25f);
  EXPECT_FALSE(c.verifyThresholdPlate.has_value());
  const auto j = redactConfigToJson(c);
  EXPECT_DOUBLE_EQ(j.at("verify_threshold_face").get<double>(), 0.25);
  EXPECT_TRUE(j.at("verify_threshold_plate").is_null());
}

TEST(RedactConfig, RefusesVerifySettingsOutOfRange) {
  EXPECT_THROW(parseRedactConfig(R"({"verify_plate_below": 1.5})", "t"), VcError);
  EXPECT_THROW(parseRedactConfig(R"({"verify_face_unless_both_above": -0.1})", "t"), VcError);
  EXPECT_THROW(parseRedactConfig(R"({"verify_threshold_face": 2})", "t"), VcError);
  EXPECT_THROW(parseRedactConfig(R"({"verify_threshold_plate": "0.5"})", "t"), VcError);
}
```

把 `verify_policy_test.cpp` 加进 `cpp/tests/CMakeLists.txt`，然后运行：

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -m3 error
```

Expected: 编译错误，`verify_policy.h` 不存在。

- [ ] **Step 2: 写实现**

`cpp/apps/vc_redact/redact_config.h`：在 `struct RedactConfig` 里 `largeFaceMinScore` 之后加字段，文件头部加 `#include <optional>`：

```cpp
  // The verifier (spec 2026-10-21 §3.1, §4.3): plates scoring below verifyPlateBelow and faces that only one
  // detector found or that score below verifyFaceUnlessBothAbove go through the verifier model when one is given
  // (vc_redact --verifier-model). The pass thresholds are the model's (model.json pass_threshold) unless set here.
  float verifyPlateBelow = 0.5f;
  float verifyFaceUnlessBothAbove = 0.5f;
  std::optional<float> verifyThresholdFace;
  std::optional<float> verifyThresholdPlate;
```

`cpp/apps/vc_redact/redact_config.cpp`：

1. `kKeys` 的大小 24 改为 28，末尾加 `"verify_plate_below", "verify_face_unless_both_above", "verify_threshold_face", "verify_threshold_plate"`。
2. 在 `boolSetting` 之后加：

```cpp
// The setting as an optional float: absent or null: nullopt; otherwise it must be a number.
std::optional<float> optionalFloatSetting(const nlohmann::json& j, const char* key, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return std::nullopt;
  return floatSetting(j, key, 0.0f, origin);
}
```

3. `parseRedactConfig` 中 `c.largeFaceMinScore = ...` 之后加：

```cpp
  c.verifyPlateBelow = floatSetting(j, "verify_plate_below", c.verifyPlateBelow, origin);
  c.verifyFaceUnlessBothAbove = floatSetting(j, "verify_face_unless_both_above", c.verifyFaceUnlessBothAbove, origin);
  c.verifyThresholdFace = optionalFloatSetting(j, "verify_threshold_face", origin);
  c.verifyThresholdPlate = optionalFloatSetting(j, "verify_threshold_plate", origin);
```

4. `validateRedactConfig` 末尾加：

```cpp
  require(c.verifyPlateBelow >= 0 && c.verifyPlateBelow <= 1, "verify_plate_below must be in [0, 1]");
  require(c.verifyFaceUnlessBothAbove >= 0 && c.verifyFaceUnlessBothAbove <= 1,
          "verify_face_unless_both_above must be in [0, 1]");
  require(!c.verifyThresholdFace || (*c.verifyThresholdFace >= 0 && *c.verifyThresholdFace <= 1),
          "verify_threshold_face must be in [0, 1]");
  require(!c.verifyThresholdPlate || (*c.verifyThresholdPlate >= 0 && *c.verifyThresholdPlate <= 1),
          "verify_threshold_plate must be in [0, 1]");
```

5. `redactConfigToJson` 中 `large_face_min_score` 之后加：

```cpp
  j["verify_plate_below"] = jsonFloat(c.verifyPlateBelow);
  j["verify_face_unless_both_above"] = jsonFloat(c.verifyFaceUnlessBothAbove);
  j["verify_threshold_face"] = c.verifyThresholdFace ? nlohmann::ordered_json(jsonFloat(*c.verifyThresholdFace)) : nullptr;
  j["verify_threshold_plate"] = c.verifyThresholdPlate ? nlohmann::ordered_json(jsonFloat(*c.verifyThresholdPlate)) : nullptr;
```

6. `configs/redact.json`：在 `"large_face_min_score": 0.6,` 之后加：

```json
  "verify_plate_below": 0.5,
  "verify_face_unless_both_above": 0.5,
  "verify_threshold_face": null,
  "verify_threshold_plate": null,
```

`cpp/apps/vc_redact/verify_policy.h`:

```cpp
#pragma once

#include <string>

#include "detect/detector.h"
#include "detect/verifier.h"
#include "vc_redact/redact_config.h"

namespace vc {

// A box the verifier scored: its pass probability, and whether it reached the pass threshold.
struct VerifyResult {
  float score = 0;
  bool passed = true;
};

// Whether `box` (of class `className`) goes through the verifier (spec 2026-10-21 §3.1): a plate scoring below
// verify_plate_below; a face that only one detector found (no '+' in its src) or that scores below
// verify_face_unless_both_above. Other classes never do.
bool needsVerification(const SourcedBox& box, const std::string& className, const RedactConfig& config);

// The pass threshold of the verifier's class `verifierClass`: verify_threshold_face / verify_threshold_plate
// when set and the class is "face" / "plate", otherwise the model's pass_threshold.
float passThreshold(const VerifierInfo& info, int verifierClass, const RedactConfig& config);

}  // namespace vc
```

`cpp/apps/vc_redact/verify_policy.cpp`:

```cpp
#include "vc_redact/verify_policy.h"

namespace vc {

bool needsVerification(const SourcedBox& box, const std::string& className, const RedactConfig& config) {
  if (className == "plate") return box.box.score < config.verifyPlateBelow;
  if (className == "face") {
    const bool both = box.src.find('+') != std::string::npos;
    return !both || box.box.score < config.verifyFaceUnlessBothAbove;
  }
  return false;
}

float passThreshold(const VerifierInfo& info, int verifierClass, const RedactConfig& config) {
  const std::string& name = info.classes.at(static_cast<size_t>(verifierClass));
  if (name == "face" && config.verifyThresholdFace) return *config.verifyThresholdFace;
  if (name == "plate" && config.verifyThresholdPlate) return *config.verifyThresholdPlate;
  return info.passThreshold.at(static_cast<size_t>(verifierClass));
}

}  // namespace vc
```

在 `cpp/CMakeLists.txt` 的 `vc_redact_lib` 列表里 `apps/vc_redact/redact_job.cpp` 后加 `apps/vc_redact/verify_policy.cpp`。

- [ ] **Step 3: 运行测试，确认通过**

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -E "error|warning"; ./build/cpp/tests/vc_tests --gtest_filter='VerifyPolicy.*:RedactConfig.*'
```

Expected: 全部 PASSED。`RedactConfig` 里原有逐键列举或往返（round trip）的测试若因新键失败，按新键补全它的期望值，不要删测试。

- [ ] **Step 4: 提交**

```bash
git add cpp/apps/vc_redact/verify_policy.h cpp/apps/vc_redact/verify_policy.cpp cpp/apps/vc_redact/redact_config.h cpp/apps/vc_redact/redact_config.cpp configs/redact.json cpp/tests/verify_policy_test.cpp cpp/tests/redact_config_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt
git commit -m "feat: verifier settings and which boxes it checks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 检测记录字段与小图转储

**Files:**
- Create: `cpp/apps/vc_redact/verify_dump.h`, `cpp/apps/vc_redact/verify_dump.cpp`, `cpp/tests/verify_dump_test.cpp`
- Modify: `cpp/apps/vc_redact/records.h`, `cpp/apps/vc_redact/records.cpp`, `cpp/tests/records_test.cpp`, `cpp/CMakeLists.txt`（`vc_redact_lib` 加 `apps/vc_redact/verify_dump.cpp`）、`cpp/tests/CMakeLists.txt`（加 `verify_dump_test.cpp`）

**Interfaces:**
- Consumes: `VerifyResult`（Task 3）。
- Produces:
  - `detectionRecord(video, frame, ptsMs, box, classes, src, std::optional<bool> confirmed = std::nullopt, std::optional<VerifyResult> verify = std::nullopt)`。字段顺序为 `..., "src", "verify", "verified"(仅在为 false 时写), "confirmed"`。
  - `class VerifyDump { VerifyDump(const fs::path& dir, int size); void add(const std::string& video, int64_t frame, const Box& box, const std::string& cls, const std::string& src, const uint8_t* rgb); int64_t count() const; void close(); };`，写 `dir/verify_crops.bin`（每张 size*size*3 字节，按 `index` 顺序紧挨着存）和 `dir/verify_crops.jsonl`（每行 `{"index","video","frame","cls","x1","y1","x2","y2","score","src","size"}`）。

- [ ] **Step 1: 写失败的测试**

在 `cpp/tests/records_test.cpp` 加：

```cpp
TEST(Records, DetectionLineCarriesTheVerifierScoreAndRejection) {
  const Box b{0, 1.0f, 2.0f, 3.0f, 4.0f, 0.4f};
  EXPECT_EQ(detectionRecord("a.mp4", 6, 100.0, b, kClasses, "yunet", false, VerifyResult{0.12341f, false}),
            R"({"video":"a.mp4","frame":6,"pts_ms":100.0,"cls":"face","x1":1.0,"y1":2.0,"x2":3.0,"y2":4.0,)"
            R"("score":0.4,"src":"yunet","verify":0.1234,"verified":false,"confirmed":false})"
            "\n");
  const std::string passed = detectionRecord("a.mp4", 6, 100.0, b, kClasses, "yunet", true, VerifyResult{0.9f, true});
  EXPECT_NE(passed.find(R"("verify":0.9,"confirmed":true})"), std::string::npos) << passed;
  EXPECT_EQ(passed.find("verified"), std::string::npos);
}
```

`cpp/tests/verify_dump_test.cpp`:

```cpp
#include "vc_redact/verify_dump.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

TEST(VerifyDump, WritesTheCropsAndOneLineEachInOrder) {
  const fs::path dir = test::tmpDir() / "verify_dump";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    VerifyDump dump(dir, 2);
    const std::vector<uint8_t> a(12, 7), b(12, 9);
    dump.add("v.mp4", 12, Box{1, 10.04f, 20.0f, 30.0f, 40.0f, 0.33333f}, "plate", "student", a.data());
    dump.add("v.mp4", 24, Box{0, 1.0f, 2.0f, 3.0f, 4.0f, 0.5f}, "face", "yunet", b.data());
    EXPECT_EQ(dump.count(), 2);
    dump.close();
  }
  std::ifstream bin(dir / "verify_crops.bin", std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(bin)), {});
  ASSERT_EQ(bytes.size(), 24u);
  EXPECT_EQ(bytes[0], 7);
  EXPECT_EQ(bytes[12], 9);
  std::ifstream jsonl(dir / "verify_crops.jsonl");
  std::string line;
  ASSERT_TRUE(std::getline(jsonl, line));
  const auto first = nlohmann::json::parse(line);
  EXPECT_EQ(first.at("index"), 0);
  EXPECT_EQ(first.at("frame"), 12);
  EXPECT_EQ(first.at("cls"), "plate");
  EXPECT_DOUBLE_EQ(first.at("x1").get<double>(), 10.0);
  EXPECT_DOUBLE_EQ(first.at("score").get<double>(), 0.3333);
  EXPECT_EQ(first.at("src"), "student");
  EXPECT_EQ(first.at("size"), 2);
  ASSERT_TRUE(std::getline(jsonl, line));
  EXPECT_EQ(nlohmann::json::parse(line).at("index"), 1);
  EXPECT_FALSE(std::getline(jsonl, line));
}

}  // namespace
}  // namespace vc
```

把 `verify_dump_test.cpp` 加进 `cpp/tests/CMakeLists.txt`，然后运行：

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -m3 error
```

Expected: 编译错误，`verify_dump.h` 不存在，`detectionRecord` 也不接受 `VerifyResult` 参数。

- [ ] **Step 2: 写实现**

`cpp/apps/vc_redact/records.h`：加 `#include "vc_redact/verify_policy.h"`，`detectionRecord` 声明改为：

```cpp
// One detections.jsonl line (spec §3.2), newline included; with `verify` the verifier's pass probability
// ("verify", 4 decimals) and, when it failed, "verified": false (spec 2026-10-21 §4.3); with `confirmed` a last
// field saying whether the box was masked (false: a low-score box seen on one keyframe only, spec 2026-10-19
// §3.4, or one the verifier rejected). Throws VcError when box.cls is not an index into `classes`, or when
// ptsMs, a box coordinate or a score is not finite (or overflows when rounded). Invalid UTF-8 in `video` or
// `src` is written as U+FFFD.
std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src,
                            std::optional<bool> confirmed = std::nullopt,
                            std::optional<VerifyResult> verify = std::nullopt);
```

`cpp/apps/vc_redact/records.cpp`：函数签名同步改，`j["src"] = src;` 之后加：

```cpp
  if (verify) {
    j["verify"] = round4(verify->score, "verify");
    if (!verify->passed) j["verified"] = false;
  }
```

`cpp/apps/vc_redact/verify_dump.h`:

```cpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "detect/box.h"

namespace vc {

// The verifier crops of a run, for training (spec 2026-10-21 §3.2; vc_redact --dump-verify-crops):
// verify_crops.bin holds the crops one after another, size * size * 3 bytes each (RGB, interleaved, the
// verifier's exact input before / 255), and verify_crops.jsonl one line per crop in the same order. Not
// thread-safe.
class VerifyDump {
 public:
  // Creates (truncates) both files in `dir`. Throws VcError when they cannot be created.
  VerifyDump(const std::filesystem::path& dir, int size);
  void add(const std::string& video, int64_t frame, const Box& box, const std::string& cls, const std::string& src,
           const uint8_t* rgb);
  int64_t count() const { return count_; }
  // Flushes and closes both files. Throws VcError when writing failed.
  void close();

 private:
  std::ofstream bin_;
  std::ofstream jsonl_;
  int size_;
  int64_t count_ = 0;
};

}  // namespace vc
```

`cpp/apps/vc_redact/verify_dump.cpp`:

```cpp
#include "vc_redact/verify_dump.h"

#include <cmath>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {
namespace {

double rounded(double v, double scale) { return std::round(v * scale) / scale; }

}  // namespace

VerifyDump::VerifyDump(const std::filesystem::path& dir, int size)
    : bin_(dir / "verify_crops.bin", std::ios::binary | std::ios::trunc),
      jsonl_(dir / "verify_crops.jsonl", std::ios::trunc),
      size_(size) {
  if (!bin_ || !jsonl_) throw VcError("cannot create verify_crops.bin / verify_crops.jsonl in " + dir.string());
}

void VerifyDump::add(const std::string& video, int64_t frame, const Box& box, const std::string& cls,
                     const std::string& src, const uint8_t* rgb) {
  bin_.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(size_) * size_ * 3);
  nlohmann::ordered_json j;
  j["index"] = count_;
  j["video"] = video;
  j["frame"] = frame;
  j["cls"] = cls;
  j["x1"] = rounded(box.x1, 10);
  j["y1"] = rounded(box.y1, 10);
  j["x2"] = rounded(box.x2, 10);
  j["y2"] = rounded(box.y2, 10);
  j["score"] = rounded(box.score, 10000);
  j["src"] = src;
  j["size"] = size_;
  jsonl_ << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
  ++count_;
}

void VerifyDump::close() {
  bin_.close();
  jsonl_.close();
  if (!bin_ || !jsonl_) throw VcError("writing verify_crops.bin / verify_crops.jsonl failed");
}

}  // namespace vc
```

把 `apps/vc_redact/verify_dump.cpp` 加进 `vc_redact_lib`。

- [ ] **Step 3: 运行测试，确认通过**

```bash
cmake -S cpp -B build/cpp > /dev/null && cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -E "error|warning"; ./build/cpp/tests/vc_tests --gtest_filter='Records.*:VerifyDump.*'
```

Expected: 全部 PASSED。

- [ ] **Step 4: 提交**

```bash
git add cpp/apps/vc_redact/records.h cpp/apps/vc_redact/records.cpp cpp/apps/vc_redact/verify_dump.h cpp/apps/vc_redact/verify_dump.cpp cpp/tests/records_test.cpp cpp/tests/verify_dump_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt
git commit -m "feat: record verifier scores and dump verifier crops" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 接入 `vc_redact`（复核、转储、统计、命令行、脚本）

**Files:**
- Modify: `cpp/apps/vc_redact/redact_job.h`, `cpp/apps/vc_redact/redact_job.cpp`, `cpp/apps/vc_redact/redact_cli.cpp`, `scripts/redact.sh`, `cpp/tests/redact_job_test.cpp`, `cpp/tests/redact_cli_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–4 的全部产出。
- Produces:
  - `RedactOptions` 新字段：`std::optional<std::filesystem::path> verifierModelDir; bool dumpVerifyCrops = false; int64_t dumpVerifyEvery = 1;`
  - `RedactStats` 新字段：`std::string verifier; int64_t verifyChecked = 0; int64_t verifyRejectedFaces = 0; int64_t verifyRejectedPlates = 0; int64_t verifyDumped = 0; StageStat verify;`
  - `stats.json` 新增：`"verifier"`（字符串或 null）、`"verify": {"checked", "rejected": {"face", "plate"}, "dumped"}`、`"stages"."verify"`。
  - 命令行：`--verifier-model <dir>`、`--dump-verify-crops`、`--dump-verify-every <n>`；`scripts/redact.sh` 读 `VC_VERIFIER`（默认 `data/models/verifier-v1`，设为空则不复核；目录不存在时缺省值自动视为空，显式给的不存在则报错）。

**行为（写进 `redact_job.h` 的 `runRedact` 注释）：**
- 关键帧上，大人脸规则之后，对每个 `needsVerification` 为真的框裁小图。
- 有复核模型、且该类在模型里时打分；分数 < `passThreshold` 的框不交给 `BoxTimeline`，在 `detections.jsonl` 里写 `"verify"`、`"verified": false`、`"confirmed": false`。
- 开了转储、且关键帧帧号 % `dumpVerifyEvery` == 0 时，把小图写进转储（不论是否有模型）。
- `stats.detections` 改为"进入 `BoxTimeline` 的框数"，即不含被拦下的框。
- 复核模型在 `runRedact` 中与检测器一起、在创建任何输出之前构建，坏模型就此报错退出。

- [ ] **Step 1: 写失败的测试**

在 `cpp/tests/redact_job_test.cpp` 加（`fakeFile`、`baseOptions`、`lineCount` 已在文件里）：

```cpp
fs::path constVerifierDir() { return fs::path(VC_TEST_DATA_DIR) / "const_verifier"; }

// The fake face (src "fake": one detector) is checked; the const verifier gives faces 0.119 < 0.5: nothing is
// masked, every detection line says why, and the stats count it.
TEST(RedactJob, TheVerifierRejectsAFaceItDoesNotPass) {
  RedactOptions o = baseOptions("redact_verifier_face");
  o.verifierModelDir = constVerifierDir();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.verifier, "const_verifier");
  EXPECT_EQ(s.verifyChecked, s.keyframes);
  EXPECT_EQ(s.verifyRejectedFaces, s.keyframes);
  EXPECT_EQ(s.verifyRejectedPlates, 0);
  EXPECT_EQ(s.detections, 0);
  EXPECT_EQ(s.maskedFrames, 0);
  std::ifstream det(o.outRoot / kStem / "detections.jsonl");
  for (std::string line; std::getline(det, line);) {
    const auto j = nlohmann::json::parse(line);
    EXPECT_FALSE(j.at("verified").get<bool>()) << line;
    EXPECT_FALSE(j.at("confirmed").get<bool>()) << line;
    EXPECT_NEAR(j.at("verify").get<double>(), 0.1192, 1e-4) << line;
  }
  std::ifstream statsFile(o.outRoot / kStem / "stats.json");
  const auto stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("verifier"), "const_verifier");
  EXPECT_EQ(stats.at("verify").at("rejected").at("face").get<int64_t>(), s.keyframes);
}

// A plate scoring 0.9 is above verify_plate_below: never checked, masked as before.
TEST(RedactJob, ConfidentPlatesSkipTheVerifier) {
  RedactOptions o = baseOptions("redact_verifier_plate");
  o.fakeDetections = fakeFile("redact_verifier_plate.jsonl", 60, "plate");
  o.verifierModelDir = constVerifierDir();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.verifyChecked, 0);
  EXPECT_EQ(s.maskedFrames, 60);
}

// The dump holds the crop of every checked box on the dumped keyframes, with or without a verifier model: every
// keyframe with --dump-verify-every 1, only frame 0 of this 60-frame video with 1000.
TEST(RedactJob, DumpsTheCropsOfEveryNthFrame) {
  RedactOptions all = baseOptions("redact_verifier_dump_all");
  all.dumpVerifyCrops = true;
  const RedactStats a = runRedact(all);
  EXPECT_EQ(a.verifyDumped, a.keyframes);  // one fake face per keyframe
  EXPECT_EQ(a.maskedFrames, 60);           // no model: masking is unchanged
  const fs::path dirA = all.outRoot / kStem;
  EXPECT_EQ(lineCount(dirA / "verify_crops.jsonl"), static_cast<size_t>(a.verifyDumped));
  EXPECT_EQ(fs::file_size(dirA / "verify_crops.bin"), static_cast<uintmax_t>(a.verifyDumped) * 64 * 64 * 3);

  RedactOptions sparse = baseOptions("redact_verifier_dump_sparse");
  sparse.dumpVerifyCrops = true;
  sparse.dumpVerifyEvery = 1000;
  const RedactStats s = runRedact(sparse);
  EXPECT_EQ(s.verifyDumped, 1);
  std::ifstream jsonl(sparse.outRoot / kStem / "verify_crops.jsonl");
  std::string line;
  ASSERT_TRUE(std::getline(jsonl, line));
  EXPECT_EQ(nlohmann::json::parse(line).at("frame").get<int64_t>(), 0);
}

TEST(RedactJob, ABrokenVerifierModelStopsTheRunBeforeAnyOutput) {
  RedactOptions o = baseOptions("redact_verifier_broken");
  o.verifierModelDir = test::tmpDir() / "no_such_verifier";
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(o.outRoot / kStem));
}
```

在 `cpp/tests/redact_cli_test.cpp` 加：

```cpp
TEST(RedactCli, ParsesTheVerifierOptions) {
  const RedactOptions o = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--model", "m", "--verifier-model",
                                           "v", "--dump-verify-crops", "--dump-verify-every", "12"});
  ASSERT_TRUE(o.verifierModelDir.has_value());
  EXPECT_EQ(*o.verifierModelDir, "v");
  EXPECT_TRUE(o.dumpVerifyCrops);
  EXPECT_EQ(o.dumpVerifyEvery, 12);
  const RedactOptions d = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--model", "m"});
  EXPECT_FALSE(d.verifierModelDir.has_value());
  EXPECT_FALSE(d.dumpVerifyCrops);
  EXPECT_EQ(d.dumpVerifyEvery, 1);
  EXPECT_THROW(parseRedactArgs({"--input", "i", "--out", "o", "--model", "m", "--dump-verify-every", "0"}),
               UsageError);
}
```

运行 `cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -m3 error`。
Expected: 编译错误，`RedactOptions` 没有 `verifierModelDir`。

- [ ] **Step 2: 选项与统计字段**

`cpp/apps/vc_redact/redact_job.h`：
- 在 `RedactOptions` 的 `faceModelDir` 之后加：

```cpp
  // The verifier model (`vc_train verifier`, spec 2026-10-21 §4.3): boxes needsVerification() picks are scored on
  // each keyframe, those below the pass threshold are not masked. Without it nothing is verified.
  std::optional<std::filesystem::path> verifierModelDir;
  // Writes verify_crops.bin / verify_crops.jsonl (VerifyDump) next to detections.jsonl: the crop of every box
  // needsVerification() picks on keyframes whose frame index is a multiple of dumpVerifyEvery (>= 1).
  bool dumpVerifyCrops = false;
  int64_t dumpVerifyEvery = 1;
```

- 在 `RedactStats` 的 `largeFacesDropped` 之后加：

```cpp
  std::string verifier;               // the verifier model's name; empty without one
  int64_t verifyChecked = 0;          // boxes the verifier scored
  int64_t verifyRejectedFaces = 0;    // ... and rejected, by class
  int64_t verifyRejectedPlates = 0;
  int64_t verifyDumped = 0;           // crops written with dumpVerifyCrops
```

- 在 `detectFace` 之后加 `StageStat verify;  // items = keyframes with boxes to check; cropping + scoring`。
- `detections` 的注释改为 `// the boxes passed to the timeline: not counting largeFacesDropped or the verifier's rejections`。

- [ ] **Step 3: `Processor` 里的复核与转储**

`cpp/apps/vc_redact/redact_job.cpp`：
- 加 include：`detect/verifier.h`、`detect/verifier_crop.h`、`vc_redact/verify_dump.h`、`vc_redact/verify_policy.h`。
- `Processor` 构造函数增加参数 `Verifier* verifier, VerifyDump* dump`（可为 nullptr），保存为成员 `verifier_`、`dump_`。
- 初始化 `sampler_`：`verifier ? verifier->info().crop() : CropParams{}`。
- 初始化 `verifierClass_`：按检测器类别名查复核模型的类别下标，查不到为 −1；没有模型时全为 −1。

```cpp
  Verifier* verifier_;
  VerifyDump* dump_;
  CropSampler sampler_;
  std::vector<int> verifierClass_;   // per detector class: index into the verifier's classes, -1 when not verified
  std::vector<uint8_t> cropBytes_;   // size * size * 3 per checked box of the current keyframe
  std::vector<float> cropFloats_;    // 3 * size * size per scored box
```

构造函数初始化列表里加：

```cpp
        verifier_(verifier),
        dump_(dump),
        sampler_(verifier != nullptr ? verifier->info().crop() : CropParams{}),
        verifierClass_(verifierClassesFor(detector.classes(), verifier))
```

匿名命名空间里加：

```cpp
// Per detector class, its index in the verifier's classes (by name), -1 when the verifier has none or there is
// no verifier.
std::vector<int> verifierClassesFor(const std::vector<std::string>& classes, const Verifier* verifier) {
  std::vector<int> out(classes.size(), -1);
  if (verifier == nullptr) return out;
  for (size_t i = 0; i < classes.size(); ++i) out[i] = classIndex(verifier->info().classes, classes[i]);
  return out;
}
```

（`classIndex` 已在同一文件中定义，放在它后面。）

在 `onVideo` 中，`const std::vector<SourcedBox>& found = filtered.kept;` 之后、构造 `boxes` 之前，换成：

```cpp
    const std::vector<std::optional<VerifyResult>> verified = verify(view, found);
    std::vector<Box> boxes;
    boxes.reserve(found.size());
    for (size_t i = 0; i < found.size(); ++i) {
      if (!verified[i] || verified[i]->passed) boxes.push_back(found[i].box);
    }
    stats_.detections += static_cast<int64_t>(boxes.size());
    stats_.largeFacesDropped += filtered.dropped;
```

删掉原来的 `stats_.detections += ...found.size()` 那一行和原来构造 `boxes` 的循环。`pending_ = KeyframeDetections{item.index, item.ptsMs, found};` 改为 `pending_ = KeyframeDetections{item.index, item.ptsMs, found, verified};`。

`KeyframeDetections` 加字段 `std::vector<std::optional<VerifyResult>> verified{};`。

`writeDetections` 改为：`confirmed` 只对应通过的框；被拦下的框写 `confirmed: false`。

```cpp
  void writeDetections(const std::vector<bool>& confirmed) {
    if (!pending_) return;
    size_t next = 0;
    for (size_t i = 0; i < pending_->boxes.size(); ++i) {
      const SourcedBox& b = pending_->boxes[i];
      const std::optional<VerifyResult>& v = pending_->verified[i];
      bool masked = false;
      if (!v || v->passed) {
        if (next >= confirmed.size()) throw VcError("internal error: confirmation flags do not match the keyframe's detections");
        masked = confirmed[next++];
      }
      detections_ << detectionRecord(video_, pending_->frame, pending_->ptsMs, b.box, detector_.classes(), b.src,
                                     masked, v);
    }
    if (next != confirmed.size()) throw VcError("internal error: confirmation flags do not match the keyframe's detections");
    pending_.reset();
  }
```

`Processor` 加私有方法：

```cpp
  // Crops every box needsVerification() picks; dumps them on dumped frames; scores those of a verified class.
  // Returns one entry per box: nullopt for boxes the verifier did not score.
  std::vector<std::optional<VerifyResult>> verify(const VideoFrame& frame, const std::vector<SourcedBox>& found) {
    std::vector<std::optional<VerifyResult>> out(found.size());
    if (verifier_ == nullptr && dump_ == nullptr) return out;
    const Clock::time_point started = Clock::now();
    const bool dumpThis = dump_ != nullptr && frame.index % options_.dumpVerifyEvery == 0;
    const int size = sampler_.params().size;
    const size_t bytes = static_cast<size_t>(size) * size * 3;
    std::vector<size_t> scored;  // indices into found
    cropBytes_.clear();
    for (size_t i = 0; i < found.size(); ++i) {
      const SourcedBox& b = found[i];
      const std::string& name = detector_.classes().at(static_cast<size_t>(b.box.cls));
      if (!needsVerification(b, name, options_.config)) continue;
      const bool score = verifierClass_.at(static_cast<size_t>(b.box.cls)) >= 0;
      if (!score && !dumpThis) continue;
      const size_t at = cropBytes_.size();
      cropBytes_.resize(at + bytes);
      const CropParams& p = sampler_.params();
      sampler_.fill(*frame.av, verifierWindow(b.box, p.factor, p.minPx), cropBytes_.data() + at);
      if (dumpThis) {
        dump_->add(video_, frame.index, b.box, name, b.src, cropBytes_.data() + at);
        ++stats_.verifyDumped;
      }
      if (score) {
        scored.push_back(i);
      } else {
        cropBytes_.resize(at);  // dumped only
      }
    }
    if (!scored.empty()) {
      const size_t per = 3 * static_cast<size_t>(size) * size;
      cropFloats_.resize(scored.size() * per);
      std::vector<VerifierInput> inputs;
      inputs.reserve(scored.size());
      for (size_t k = 0; k < scored.size(); ++k) {
        toPlanarFloat(cropBytes_.data() + k * bytes, size, cropFloats_.data() + k * per);
        inputs.push_back(VerifierInput{verifierClass_[static_cast<size_t>(found[scored[k]].box.cls)],
                                       cropFloats_.data() + k * per});
      }
      const std::vector<float> scores = verifier_->score(inputs);
      for (size_t k = 0; k < scored.size(); ++k) {
        const SourcedBox& b = found[scored[k]];
        const int vc = verifierClass_[static_cast<size_t>(b.box.cls)];
        const bool passed = scores[k] >= passThreshold(verifier_->info(), vc, options_.config);
        out[scored[k]] = VerifyResult{scores[k], passed};
        ++stats_.verifyChecked;
        if (!passed) {
          const std::string& name = detector_.classes()[static_cast<size_t>(b.box.cls)];
          if (name == "face") ++stats_.verifyRejectedFaces;
          if (name == "plate") ++stats_.verifyRejectedPlates;
        }
      }
    }
    stats_.verify.busySec += secondsSince(started);
    ++stats_.verify.items;
    return out;
  }
```

注意：上面"只转储不打分"时用 `resize(at)` 回收空间；而打分的小图按 `scored` 的顺序依次存放，所以 `cropBytes_` 里第 k 个小图正好对应 `scored[k]`。

- [ ] **Step 4: 在 `runRedact` / `runPipeline` 里构建复核模型与转储**

- `checkOptions` 里加：`if (o.dumpVerifyEvery < 1) throw std::invalid_argument("dumpVerifyEvery must be >= 1");`
- `runRedact` 中 `makeDetector(...)` 那一行之后，在创建任何输出之前，加：

```cpp
  std::unique_ptr<Verifier> verifier;
  if (options.verifierModelDir) {
    verifier = std::make_unique<Verifier>(*options.verifierModelDir, options.provider,
                                          SessionSettings{options.config.ortIntraThreads, options.config.trtFp16});
  }
```

把 `verifier.get()` 传给 `runPipeline`：签名加 `Verifier* verifier` 参数，所有调用处同步。

- `runPipeline` 中，`detections` / `masks` 文件创建之后加：

```cpp
  std::unique_ptr<VerifyDump> dump;
  if (options.dumpVerifyCrops) {
    dump = std::make_unique<VerifyDump>(dir, verifier != nullptr ? verifier->info().inputSize : CropParams{}.size);
  }
```

`Processor` 构造时传 `verifier, dump.get()`。在 `detections.close();` 旁边加 `if (dump) dump->close();`。

- `stats.verifier = verifier ? verifier->info().name : "";`，与 `stats.detector` 在同一处赋值。

- [ ] **Step 5: `stats.json`、摘要和命令行**

`stats.json` 的写出函数：在 `j["large_faces_dropped"]` 之后加：

```cpp
  j["verifier"] = s.verifier.empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(s.verifier);
  j["verify"]["checked"] = s.verifyChecked;
  j["verify"]["rejected"]["face"] = s.verifyRejectedFaces;
  j["verify"]["rejected"]["plate"] = s.verifyRejectedPlates;
  j["verify"]["dumped"] = s.verifyDumped;
```

`j["stages"]["detect_face"]` 之后加 `j["stages"]["verify"] = stageJson(s.verify);`。

`redactSummary`：在 detector 那一行之后，若 `!s.verifier.empty()` 加一行：

```
verifier:      <name>: <checked> checked, <face> faces and <plate> plates rejected
```

`redact_cli.cpp` 的参数解析中加：

```cpp
    } else if (arg == "--verifier-model") {
      o.verifierModelDir = fs::path(value());
    } else if (arg == "--dump-verify-crops") {
      o.dumpVerifyCrops = true;
    } else if (arg == "--dump-verify-every") {
      const double n = parseNumber(arg, value());
      if (n < 1 || n != std::floor(n) || n > 1e9) throw UsageError("--dump-verify-every expects an integer >= 1");
      o.dumpVerifyEvery = static_cast<int64_t>(n);
```

（文件头部加 `#include <cmath>`。）

`redactUsage()` 在 `--face-model` 之后加：

```
  --verifier-model <dir>    verifier model (model.onnx + model.json from vc_train verifier): plates and
                            faces the detectors are unsure of are checked and not masked below its pass
                            threshold (configs/redact.json verify_*)
  --dump-verify-crops       write the verifier crops of checked boxes to verify_crops.bin / .jsonl
  --dump-verify-every <n>   dump only keyframes whose frame index is a multiple of n (default 1)
```

`scripts/redact.sh`：
- 注释的环境变量表里加：`#   VC_VERIFIER     verifier model directory      (default data/models/verifier-v1 when it exists; empty = no verifier)`
- 在 `FACE=...` 之后加：

```bash
if [ "${VC_VERIFIER+set}" = set ]; then
  VERIFIER="$VC_VERIFIER"
elif [ -d "$ROOT/data/models/verifier-v1" ]; then
  VERIFIER="$ROOT/data/models/verifier-v1"
else
  VERIFIER=""
fi
```

- 模型目录检查循环改为 `for dir in "$MODEL" ${FACE:+"$FACE"} ${VERIFIER:+"$VERIFIER"}; do`。
- 在 `if [ -n "$FACE" ]; then ARGS+=(--face-model "$FACE"); fi` 之后加 `if [ -n "$VERIFIER" ]; then ARGS+=(--verifier-model "$VERIFIER"); fi`。

- [ ] **Step 6: 运行相关测试**

```bash
cmake --build build/cpp --target vc_tests -j4 2>&1 | grep -E "error|warning"; ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:Records.*:Verif*'
```

Expected: 全部 PASSED，包括原有的 `RedactJob` 测试。不给复核模型时它们的行为必须不变；如果原有测试因 `detections` 的新口径失败，说明有框被拦了，而这不应该发生，先查原因。

- [ ] **Step 7: 提交**

```bash
git add cpp/apps/vc_redact scripts/redact.sh cpp/tests/redact_job_test.cpp cpp/tests/redact_cli_test.cpp
git commit -m "feat: vc_redact checks unsure boxes with a verifier model and can dump their crops" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 6: 挖候选（`vc_hardneg` 包骨架与 `candidates`）

**Files:**
- Create: `py/vc_hardneg/__init__.py`, `py/vc_hardneg/__main__.py`, `py/vc_hardneg/config.py`, `py/vc_hardneg/candidates.py`, `configs/hardneg.json`
- Test: `py/tests/test_hardneg_candidates.py`

**Interfaces:**
- Consumes: `vc_redact` 运行目录 `<root>/<stem>/` 里的 `verify_crops.jsonl`、`masks.jsonl`（Task 5）；`data/labels/<stem>/merged.jsonl`、`processed.txt`、`excluded_refs.csv`；`data/frames/<stem>/<frame:06d>.jpg`。
- Produces:
  - `HardnegConfig`（`config.py`）及 `load_config(path: Path) -> HardnegConfig`，字段见下方代码。
  - `Candidate`（frozen dataclass），以及 `Candidate.to_record() -> dict`、`Candidate.from_record(dict) -> Candidate`。
  - `candidate_id(video: str, frame: int, cls: str, box: tuple[float, float, float, float]) -> str`。
  - `crop_window(box, factor, min_px) -> tuple[float, float, float]`，返回 (x, y, side)。
  - `native_crop(image: np.ndarray, window) -> tuple[np.ndarray, tuple[int, int]]`，返回小图和小图左上角在整帧中的坐标。
  - `box_in_crop(c: Candidate, cfg: HardnegConfig) -> Box`。
  - `read_candidates(path: Path) -> list[Candidate]`、`write_candidates(path: Path, cands: list[Candidate]) -> None`。
  - `mine(runs: list[Path], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> MineSummary`；`out_dir` 下写 `candidates.jsonl` 和 `crops/<id>.png`。

- [ ] **Step 1: 配置文件与配置读取**

`configs/hardneg.json`:

```json
{
  "crop_factor": 2.5,
  "min_crop_px": 32,
  "pad_value": 114,
  "per_track": 3,
  "frame_teacher_yes": 0.5,
  "frame_teacher_iou": 0.3,
  "upscale": 4,
  "yes_iou": 0.3,
  "overlap_iou": 0.1,
  "plate": {"yes_score": 0.3, "no_score": 0.1, "max_side": 960},
  "face": {"yes_score": 0.7, "no_score": 0.3, "max_side": 640},
  "llm": {
    "name": "codex",
    "command": [],
    "timeout_s": 120,
    "prompt": "这张小图来自行车记录仪画面，红框里的东西是不是一块{cls_zh}？只回答一个 JSON：{\"verdict\": \"yes\" | \"no\" | \"unsure\", \"reason\": \"一句话\"}。看不清就答 unsure。"
  }
}
```

`py/vc_hardneg/__init__.py`：一行 docstring `"""Hard negatives for the verifier: mining, judging, review (spec 2026-10-21 §3)."""`。

`py/vc_hardneg/config.py`:

```python
"""configs/hardneg.json: crop geometry, judge thresholds and the optional LLM judge (spec 2026-10-21 §3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("configs/hardneg.json")
_KEYS = {
    "crop_factor", "min_crop_px", "pad_value", "per_track", "frame_teacher_yes", "frame_teacher_iou", "upscale",
    "yes_iou", "overlap_iou", "plate", "face", "llm",
}


@dataclass(frozen=True)
class ClassThresholds:
    yes_score: float  # a teacher box at the candidate (IoU >= yes_iou) scoring this much: "yes"
    no_score: float  # no overlapping teacher box scoring this much: "no"
    max_side: int  # the upscaled crop's longest side is capped here


@dataclass(frozen=True)
class LlmConfig:
    name: str
    command: tuple[str, ...]  # argv; {image}, {cls}, {cls_zh} and {prompt} are filled in; empty: no LLM judge
    timeout_s: float
    prompt: str


@dataclass(frozen=True)
class HardnegConfig:
    crop_factor: float
    min_crop_px: float
    pad_value: int
    per_track: int
    frame_teacher_yes: float
    frame_teacher_iou: float
    upscale: float
    yes_iou: float
    overlap_iou: float
    plate: ClassThresholds
    face: ClassThresholds
    llm: LlmConfig

    def thresholds(self, cls: str) -> ClassThresholds:
        return {"plate": self.plate, "face": self.face}[cls]


def _unit(name: str, v: float) -> float:
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return float(v)


def _class(name: str, d: dict) -> ClassThresholds:
    t = ClassThresholds(_unit(f"{name}.yes_score", d["yes_score"]), _unit(f"{name}.no_score", d["no_score"]), int(d["max_side"]))
    if t.no_score > t.yes_score:
        raise ValueError(f"{name}.no_score must not exceed {name}.yes_score")
    if t.max_side < 32:
        raise ValueError(f"{name}.max_side must be at least 32")
    return t


def load_config(path: Path = DEFAULT_CONFIG) -> HardnegConfig:
    """Raises ValueError for unknown or missing keys and values out of range."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or set(d) != _KEYS:
        raise ValueError(f"{path}: keys must be exactly {sorted(_KEYS)}")
    llm = d["llm"]
    cfg = HardnegConfig(
        crop_factor=float(d["crop_factor"]),
        min_crop_px=float(d["min_crop_px"]),
        pad_value=int(d["pad_value"]),
        per_track=int(d["per_track"]),
        frame_teacher_yes=_unit("frame_teacher_yes", d["frame_teacher_yes"]),
        frame_teacher_iou=_unit("frame_teacher_iou", d["frame_teacher_iou"]),
        upscale=float(d["upscale"]),
        yes_iou=_unit("yes_iou", d["yes_iou"]),
        overlap_iou=_unit("overlap_iou", d["overlap_iou"]),
        plate=_class("plate", d["plate"]),
        face=_class("face", d["face"]),
        llm=LlmConfig(str(llm["name"]), tuple(str(a) for a in llm["command"]), float(llm["timeout_s"]), str(llm["prompt"])),
    )
    if cfg.crop_factor <= 0 or cfg.min_crop_px < 0 or not 0 <= cfg.pad_value <= 255 or cfg.per_track < 1 or cfg.upscale < 1:
        raise ValueError(f"{path}: crop_factor > 0, min_crop_px >= 0, pad_value in [0, 255], per_track >= 1, upscale >= 1")
    return cfg
```

- [ ] **Step 2: 写失败的测试**

`py/tests/test_hardneg_candidates.py`:

```python
import json

import cv2
import numpy as np
import pytest

from vc_hardneg.candidates import (
    Candidate, box_in_crop, candidate_id, crop_window, mine, native_crop, read_candidates,
)
from vc_hardneg.config import load_config

CFG = load_config()


def test_crop_window_is_the_spec_square():
    assert crop_window((100, 200, 140, 210), 2.5, 32) == (70.0, 155.0, 100.0)
    assert crop_window((10, 10, 16, 14), 2.5, 32) == (-3.0, -4.0, 32.0)


def test_native_crop_pads_outside_the_frame():
    image = np.full((50, 60, 3), 200, np.uint8)
    crop, origin = native_crop(image, (-10.0, -5.0, 20.0), 114)
    assert origin == (-10, -5)
    assert crop.shape == (20, 20, 3)
    assert (crop[:5, :, :] == 114).all() and (crop[:, :10, :] == 114).all()
    assert (crop[5:, 10:, :] == 200).all()


def test_candidate_ids_are_stable_and_distinct():
    a = candidate_id("v.mp4", 12, "plate", (1.0, 2.0, 3.0, 4.0))
    assert a == candidate_id("v.mp4", 12, "plate", (1.04, 2.0, 3.0, 4.0))  # 0.1 px rounding, as the records
    assert a != candidate_id("v.mp4", 24, "plate", (1.0, 2.0, 3.0, 4.0))
    assert len(a) == 16


def _run(tmp_path, stem, rows, masks):
    run = tmp_path / "runs" / stem
    run.mkdir(parents=True)
    (run / "verify_crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "masks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in masks))
    return run


def _labels(tmp_path, stem, frames, teacher):
    d = tmp_path / "labels" / stem
    d.mkdir(parents=True)
    (d / "processed.txt").write_text("".join(f"{f}\n" for f in frames))
    (d / "merged.jsonl").write_text("".join(json.dumps(t) + "\n" for t in teacher))
    fr = tmp_path / "frames" / stem
    fr.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(fr / f"{f:06d}.jpg"), np.full((120, 160, 3), 90, np.uint8))


def _row(i, frame, box, cls="plate", score=0.1, src="student"):
    return {"index": i, "video": "s.mp4", "frame": frame, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
            "y2": box[3], "score": score, "src": src, "size": 64}


def _mask(frame, boxes):
    return {"video": "s.mp4", "frame": frame, "boxes": [
        {"cls": c, "x1": b[0] - 2, "y1": b[1] - 2, "x2": b[2] + 2, "y2": b[3] + 2, "track": t, "origin": "detected"}
        for c, b, t in boxes]}


def test_mine_keeps_masked_boxes_on_labelled_frames_at_most_per_track(tmp_path):
    box = (40.0, 40.0, 60.0, 50.0)
    frames = [0, 12, 24, 36, 48]
    rows = [_row(i, f, box) for i, f in enumerate(frames)]
    rows.append(_row(5, 12, (100.0, 10.0, 110.0, 20.0)))  # not masked (no track): left out
    rows.append(_row(6, 60, box))  # frame 60 is not labelled: left out
    masks = [_mask(f, [("plate", box, 7)]) for f in frames]
    run = _run(tmp_path, "s", rows, masks)
    teacher = [{"video": "s.mp4", "frame": 24, "pts_ms": 0, "cls": "plate", "x1": 41, "y1": 40, "x2": 60, "y2": 50,
                "score": 0.8, "src": "teacher_owlv2"}]
    _labels(tmp_path, "s", frames, teacher)
    out = tmp_path / "out"
    summary = mine([run], tmp_path / "labels", tmp_path / "frames", out, CFG)
    cands = read_candidates(out / "candidates.jsonl")
    assert summary.candidates == len(cands) == 3  # per_track 3 of the 5 frames of track 7
    assert [c.frame for c in cands] == [0, 24, 48]  # spread evenly along the track
    assert all(c.track == 7 and c.dump_index == c.frame // 12 for c in cands)
    assert {c.frame: c.frame_teacher for c in cands}[24] == pytest.approx(0.8)
    assert {c.frame: c.frame_teacher for c in cands}[0] == 0.0
    for c in cands:
        assert (out / c.image).is_file()
        b = box_in_crop(c, CFG)
        assert b.x1 == pytest.approx(40 - c.crop_origin[0]) and b.y1 == pytest.approx(40 - c.crop_origin[1])
```

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_candidates.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_hardneg.candidates'`。

- [ ] **Step 3: 写实现**

`py/vc_hardneg/candidates.py`:

```python
"""Verifier candidates (spec 2026-10-21 §3.2): unsure boxes vc_redact dumped and masked, a few per track."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_eval.records import read_excluded
from vc_hardneg.config import HardnegConfig
from vc_label.boxes import iou
from vc_label.contracts import Box, read_jsonl

CANDIDATES = "candidates.jsonl"
CROPS_DIR = "crops"
EXCLUDED_IOU = 0.3  # a teacher box overlapping an excluded_refs.csv row this much is a known teacher false positive


@dataclass(frozen=True)
class Candidate:
    id: str
    video: str
    stem: str
    frame: int
    cls: str
    box: tuple[float, float, float, float]  # frame pixels
    score: float
    src: str
    track: int
    dump_dir: str  # the vc_redact run directory holding verify_crops.bin
    dump_index: int  # the crop's index in it
    frame_teacher: float  # best same-class teacher box at IoU >= frame_teacher_iou on the frame, 0 if none
    image: str  # native-resolution crop, relative to the candidates file's directory
    crop_origin: tuple[int, int]  # the crop's top-left corner in frame pixels

    def to_record(self) -> dict:
        d = asdict(self)
        d["box"], d["crop_origin"] = list(self.box), list(self.crop_origin)
        return d

    @staticmethod
    def from_record(r: dict) -> Candidate:
        return Candidate(**{**r, "box": tuple(r["box"]), "crop_origin": tuple(r["crop_origin"])})


@dataclass(frozen=True)
class MineSummary:
    rows: int  # dump rows read
    masked: int  # ... on labelled frames that a track masked
    candidates: int  # ... kept after the per-track limit


def candidate_id(video: str, frame: int, cls: str, box: tuple[float, float, float, float]) -> str:
    key = f"{video}:{frame}:{cls}:" + ",".join(f"{round(v, 1):.1f}" for v in box)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def crop_window(box: tuple[float, float, float, float], factor: float, min_px: float) -> tuple[float, float, float]:
    """(x, y, side) of the verifier crop's square, as vc_redact's verifierWindow."""
    x1, y1, x2, y2 = box
    side = max(factor * max(x2 - x1, y2 - y1), min_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return cx - side / 2, cy - side / 2, side


def native_crop(image: np.ndarray, window: tuple[float, float, float], pad: int) -> tuple[np.ndarray, tuple[int, int]]:
    """The window's whole pixels at the image's own resolution, pad outside the image; and its top-left corner."""
    x, y, side = window
    x0, y0 = math.floor(x), math.floor(y)
    w, h = math.ceil(x + side) - x0, math.ceil(y + side) - y0
    out = np.full((h, w, 3), pad, np.uint8)
    ih, iw = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(iw, x0 + w), min(ih, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out, (x0, y0)


def box_in_crop(c: Candidate, cfg: HardnegConfig) -> Box:
    ox, oy = c.crop_origin
    x1, y1, x2, y2 = c.box
    return Box(c.cls, x1 - ox, y1 - oy, x2 - ox, y2 - oy, score=c.score, src=c.src)


def read_candidates(path: Path) -> list[Candidate]:
    with Path(path).open(encoding="utf-8") as f:
        return [Candidate.from_record(json.loads(line)) for line in f if line.strip()]


def write_candidates(path: Path, cands: list[Candidate]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")


def _masked_tracks(masks_path: Path, frames: set[int]) -> dict[int, list[tuple[str, Box, int]]]:
    """frame -> (cls, mask box, track) of the masks a keyframe detection made ("origin": "detected")."""
    out: dict[int, list[tuple[str, Box, int]]] = defaultdict(list)
    with masks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["frame"] not in frames:
                continue
            for b in rec["boxes"]:
                if b.get("origin") == "detected" and "track" in b:
                    box = Box(b["cls"], b["x1"], b["y1"], b["x2"], b["y2"])
                    out[rec["frame"]].append((b["cls"], box, int(b["track"])))
    return out


def _track_of(cls: str, box: Box, masks: list[tuple[str, Box, int]]) -> int | None:
    """The track whose detected mask on that frame contains the box's centre, the nearest centre first."""
    cx, cy = box.center
    inside = [(m, t) for c, m, t in masks if c == cls and m.x1 <= cx <= m.x2 and m.y1 <= cy <= m.y2]
    if not inside:
        return None
    return min(inside, key=lambda mt: (mt[0].center[0] - cx) ** 2 + (mt[0].center[1] - cy) ** 2)[1]


def _spread(items: list, k: int) -> list:
    """k items spread evenly over the list (both ends), all of them when there are at most k."""
    if len(items) <= k:
        return items
    if k == 1:
        return [items[(len(items) - 1) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _frame_teacher(cls: str, box: Box, teacher: list[Box], excluded: list[Box], min_iou: float) -> float:
    real = [t for t in teacher if t.cls == cls and not any(e.cls == cls and iou(e, t) >= EXCLUDED_IOU for e in excluded)]
    return max((t.score for t in real if iou(t, box) >= min_iou), default=0.0)


def mine(runs: list[Path], labels_root: Path, frames_root: Path, out_dir: Path, cfg: HardnegConfig) -> MineSummary:
    """Candidates from vc_redact runs made with --dump-verify-crops (each run dir named after its stem)."""
    found: list[Candidate] = []
    rows_read = 0
    for run in runs:
        stem = run.name
        labels = labels_root / stem
        processed = {int(v) for v in (labels / "processed.txt").read_text().split()}
        with (run / "verify_crops.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows_read += len(rows)
        frames = {r["frame"] for r in rows if r["frame"] in processed and (frames_root / stem / f"{r['frame']:06d}.jpg").is_file()}
        masks = _masked_tracks(run / "masks.jsonl", frames)
        teacher: dict[int, list[Box]] = defaultdict(list)
        for d in read_jsonl(labels / "merged.jsonl"):
            teacher[d.frame].append(d.box)
        excluded: dict[int, list[Box]] = defaultdict(list)
        if (labels / "excluded_refs.csv").is_file():
            for e in read_excluded(labels):
                excluded[e.frame].append(e.box)
        for r in rows:
            if r["frame"] not in frames:
                continue
            box = Box(r["cls"], r["x1"], r["y1"], r["x2"], r["y2"], score=r["score"], src=r["src"])
            track = _track_of(r["cls"], box, masks[r["frame"]])
            if track is None:
                continue
            coords = (box.x1, box.y1, box.x2, box.y2)
            found.append(Candidate(
                id=candidate_id(r["video"], r["frame"], r["cls"], coords), video=r["video"], stem=stem, frame=r["frame"],
                cls=r["cls"], box=coords, score=r["score"], src=r["src"], track=track, dump_dir=str(run),
                dump_index=r["index"], frame_teacher=_frame_teacher(r["cls"], box, teacher[r["frame"]], excluded[r["frame"]], cfg.frame_teacher_iou),
                image="", crop_origin=(0, 0),
            ))
    by_track: dict[tuple[str, int, str], list[Candidate]] = defaultdict(list)
    for c in found:
        by_track[(c.stem, c.track, c.cls)].append(c)
    picked = [c for key in sorted(by_track) for c in _spread(sorted(by_track[key], key=lambda c: c.frame), cfg.per_track)]
    (out_dir / CROPS_DIR).mkdir(parents=True, exist_ok=True)
    final: list[Candidate] = []
    for c in sorted(picked, key=lambda c: (c.stem, c.frame, c.id)):
        image = cv2.imread(str(frames_root / c.stem / f"{c.frame:06d}.jpg"))
        crop, origin = native_crop(image, crop_window(c.box, cfg.crop_factor, cfg.min_crop_px), cfg.pad_value)
        rel = f"{CROPS_DIR}/{c.id}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        final.append(Candidate(**{**c.__dict__, "image": rel, "crop_origin": origin}))
    write_candidates(out_dir / CANDIDATES, final)
    return MineSummary(rows=rows_read, masked=len(found), candidates=len(final))
```

`py/vc_hardneg/__main__.py`：

```python
import sys

from vc_hardneg.cli import main

sys.exit(main())
```

（`cli.py` 在 Task 8 写；在那之前 `__main__` 不会被测试导入。）

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_candidates.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add py/vc_hardneg configs/hardneg.json py/tests/test_hardneg_candidates.py
git commit -m "feat: mine verifier candidates from dumped crops" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 判定记录与三种裁判

**Files:**
- Create: `py/vc_hardneg/judgments.py`, `py/vc_hardneg/judges.py`
- Test: `py/tests/test_hardneg_judges.py`

**Interfaces:**
- Consumes: `Candidate`、`box_in_crop`、`HardnegConfig`（Task 6）；`Owlv2PlateTeacher`、`RetinaFaceTeacher`、`RETINAFACE_WEIGHTS`、`offline_model_env`（vc_label）。
- Produces:
  - `Judgment(id: str, verdict: str, judge: str, score: float | None = None, detail: str = "")`；`VERDICTS = ("yes", "no", "unsure")`；`HUMAN = "human"`。
  - `append_judgments(path: Path, items: Iterable[Judgment]) -> None`、`read_judgments(path: Path) -> list[Judgment]`、`resolve(judgments: Iterable[Judgment], include_human: bool = True) -> dict[str, Judgment]`。
  - `verdict_from_boxes(target: Box, found: list[Box], cfg: HardnegConfig) -> tuple[str, float]`。
  - `FrameTeacherJudge(cfg).judge(c) -> Judgment | None`；`TeacherJudge(cfg, plate_detect, face_detect).judge(c, crop_bgr) -> Judgment`；`make_teacher_judge(cfg, device) -> TeacherJudge`。
  - `CommandJudge(cfg.llm, env=os.environ).judge(c, image_path: Path) -> Judgment`；`LLM_ENV = "VC_HARDNEG_ALLOW_LLM"`。
  - `run_judges(cands, judgments_path, crops_root, frame_judge, teacher_judge, llm_judge=None, rejudge=False) -> Counter`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_hardneg_judges.py`:

```python
import json
import sys

import numpy as np
import pytest

from vc_hardneg.candidates import Candidate
from vc_hardneg.config import load_config
from vc_hardneg.judges import (
    LLM_ENV, CommandJudge, FrameTeacherJudge, TeacherJudge, run_judges, verdict_from_boxes,
)
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments, resolve
from vc_label.contracts import Box

CFG = load_config()


def _cand(cid="a", cls="plate", frame_teacher=0.0, image="crops/a.png"):
    return Candidate(id=cid, video="v.mp4", stem="v", frame=0, cls=cls, box=(110.0, 110.0, 130.0, 118.0), score=0.1,
                     src="student", track=1, dump_dir="d", dump_index=0, frame_teacher=frame_teacher, image=image,
                     crop_origin=(100, 100))


def test_resolve_keeps_the_last_and_prefers_people(tmp_path):
    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("a", "no", "teacher"), Judgment("a", "yes", HUMAN), Judgment("a", "unsure", "teacher"),
                            Judgment("b", "no", "teacher"), Judgment("b", "unsure", "llm:codex")])
    r = resolve(read_judgments(path))
    assert r["a"].verdict == "yes" and r["a"].judge == HUMAN
    assert r["b"].verdict == "unsure"
    assert resolve(read_judgments(path), include_human=False)["a"].verdict == "unsure"
    with pytest.raises(ValueError):
        append_judgments(path, [Judgment("c", "maybe", "teacher")])


def test_verdict_from_boxes_uses_the_class_thresholds():
    target = Box("plate", 10, 10, 30, 18)
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.5)], CFG)[0] == "yes"
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.2)], CFG)[0] == "unsure"
    assert verdict_from_boxes(target, [Box("plate", 10, 10, 30, 18, score=0.05)], CFG)[0] == "no"
    assert verdict_from_boxes(target, [Box("plate", 200, 200, 230, 210, score=0.9)], CFG)[0] == "no"  # elsewhere
    assert verdict_from_boxes(target, [], CFG) == ("no", 0.0)


def test_frame_teacher_says_yes_only_when_confident():
    assert FrameTeacherJudge(CFG).judge(_cand(frame_teacher=0.6)).verdict == "yes"
    assert FrameTeacherJudge(CFG).judge(_cand(frame_teacher=0.4)) is None


def test_teacher_judge_upscales_and_maps_the_box_back():
    seen = {}

    def plate_detect(img):
        seen["shape"] = img.shape
        return [Box("plate", 40, 40, 120, 72, score=0.9)]  # (110..130, 110..118) - origin 100, times 4

    judge = TeacherJudge(CFG, plate_detect, lambda img: [])
    j = judge.judge(_cand(), np.zeros((50, 50, 3), np.uint8))
    assert seen["shape"] == (200, 200, 3)
    assert j.verdict == "yes" and j.judge == "teacher"
    assert judge.judge(_cand(cls="face"), np.zeros((50, 50, 3), np.uint8)).verdict == "no"


def test_the_llm_judge_refuses_to_run_without_the_switch():
    cfg = load_config().llm.__class__(name="t", command=(sys.executable, "-c", "print(1)"), timeout_s=5, prompt="p")
    with pytest.raises(PermissionError):
        CommandJudge(cfg, env={})


def test_the_llm_judge_reads_the_last_json_and_falls_back_to_unsure(tmp_path):
    llm = load_config().llm.__class__
    ok = llm("t", (sys.executable, "-c", 'print("thinking..."); print(\'{"verdict": "no", "reason": "tail light"}\')'), 5, "p")
    assert CommandJudge(ok, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png").verdict == "no"
    garbage = llm("t", (sys.executable, "-c", "print('no json here')"), 5, "p")
    assert CommandJudge(garbage, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png").verdict == "unsure"
    slow = llm("t", (sys.executable, "-c", "import time; time.sleep(5)"), 0.2, "p")
    j = CommandJudge(slow, env={LLM_ENV: "1"}).judge(_cand(), tmp_path / "a.png")
    assert j.verdict == "unsure" and "timeout" in j.detail


def test_run_judges_skips_judged_and_asks_the_llm_only_when_unsure(tmp_path):
    import cv2

    (tmp_path / "crops").mkdir()
    cands = [_cand("a", frame_teacher=0.9, image="crops/a.png"), _cand("b", image="crops/b.png"), _cand("c", image="crops/c.png")]
    for c in cands:
        cv2.imwrite(str(tmp_path / c.image), np.zeros((50, 50, 3), np.uint8))
    path = tmp_path / "j.jsonl"
    append_judgments(path, [Judgment("c", "no", HUMAN)])
    teacher = TeacherJudge(CFG, lambda img: [Box("plate", 40, 40, 120, 72, score=0.2)], lambda img: [])  # unsure
    asked = []

    class FakeLlm:
        def judge(self, c, image_path):
            asked.append(c.id)
            return Judgment(c.id, "no", "llm:fake")

    counts = run_judges(cands, path, tmp_path, FrameTeacherJudge(CFG), teacher, FakeLlm())
    assert asked == ["b"]
    r = resolve(read_judgments(path))
    assert r["a"].judge == "frame_teacher" and r["b"].verdict == "no" and r["c"].judge == HUMAN
    assert counts["frame_teacher:yes"] == 1 and counts["teacher:unsure"] == 1 and counts["llm:fake:no"] == 1
```

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_judges.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_hardneg.judges'`。

- [ ] **Step 2: 写实现**

`py/vc_hardneg/judgments.py`:

```python
"""judgments.jsonl (spec 2026-10-21 §3.3): one verdict per line, appended; the last per id wins, people over models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

VERDICTS = ("yes", "no", "unsure")
HUMAN = "human"


@dataclass(frozen=True)
class Judgment:
    id: str
    verdict: str  # one of VERDICTS
    judge: str  # "frame_teacher" | "teacher" | "llm:<name>" | "human"
    score: float | None = None
    detail: str = ""


def append_judgments(path: Path, items: Iterable[Judgment]) -> None:
    items = list(items)
    for j in items:
        if j.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {j.verdict!r} for {j.id}")
    with Path(path).open("a", encoding="utf-8") as f:
        for j in items:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")


def read_judgments(path: Path) -> list[Judgment]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Judgment(**json.loads(line)) for line in f if line.strip()]


def resolve(judgments: Iterable[Judgment], include_human: bool = True) -> dict[str, Judgment]:
    """Per id the last judgment, except that a person's last word beats any later model's."""
    out: dict[str, Judgment] = {}
    for j in judgments:
        if j.judge == HUMAN and not include_human:
            continue
        if j.id in out and out[j.id].judge == HUMAN and j.judge != HUMAN:
            continue
        out[j.id] = j
    return out
```

`py/vc_hardneg/judges.py`:

```python
"""The judges of spec 2026-10-21 §3.3, tried in order: the frame's own teacher boxes, the local teachers on the
upscaled crop, an optional LLM command for what is still unsure. People come last, through review.py."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np

from vc_hardneg.candidates import Candidate, box_in_crop
from vc_hardneg.config import HardnegConfig, LlmConfig
from vc_hardneg.judgments import Judgment, append_judgments, read_judgments, resolve
from vc_label.boxes import iou
from vc_label.contracts import Box

LLM_ENV = "VC_HARDNEG_ALLOW_LLM"
CLS_ZH = {"plate": "车牌", "face": "人脸"}
Detect = Callable[[np.ndarray], list[Box]]


def verdict_from_boxes(target: Box, found: list[Box], cfg: HardnegConfig) -> tuple[str, float]:
    """"yes" when a box at the target (IoU >= yes_iou) reaches yes_score, "no" when no box overlapping it
    (IoU >= overlap_iou) reaches no_score, else "unsure"; and the best overlapping score."""
    t = cfg.thresholds(target.cls)
    overlapping = [b for b in found if iou(b, target) >= cfg.overlap_iou]
    best = max((b.score for b in overlapping), default=0.0)
    if any(iou(b, target) >= cfg.yes_iou and b.score >= t.yes_score for b in found):
        return "yes", best
    if best < t.no_score:
        return "no", best
    return "unsure", best


class FrameTeacherJudge:
    name = "frame_teacher"

    def __init__(self, cfg: HardnegConfig) -> None:
        self.cfg = cfg

    def judge(self, c: Candidate) -> Judgment | None:
        if c.frame_teacher >= self.cfg.frame_teacher_yes:
            return Judgment(c.id, "yes", self.name, c.frame_teacher, "teacher box on the full frame")
        return None


class TeacherJudge:
    name = "teacher"

    def __init__(self, cfg: HardnegConfig, plate_detect: Detect, face_detect: Detect) -> None:
        self.cfg = cfg
        self.detect = {"plate": plate_detect, "face": face_detect}

    def judge(self, c: Candidate, crop_bgr: np.ndarray) -> Judgment:
        t = self.cfg.thresholds(c.cls)
        h, w = crop_bgr.shape[:2]
        scale = max(1.0, min(self.cfg.upscale, t.max_side / max(h, w)))
        img = cv2.resize(crop_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
        b = box_in_crop(c, self.cfg)
        target = Box(c.cls, b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale)
        found = [f for f in self.detect[c.cls](img) if f.cls == c.cls]
        verdict, best = verdict_from_boxes(target, found, self.cfg)
        return Judgment(c.id, verdict, self.name, best, f"x{scale:.2f}, {len(found)} boxes")


def make_teacher_judge(cfg: HardnegConfig, device) -> TeacherJudge:
    """The local teachers (cached weights only; offline_model_env() must have run)."""
    from vc_label.paths import RETINAFACE_WEIGHTS
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher
    from vc_label.teachers.retinaface import RetinaFaceTeacher

    owl = Owlv2PlateTeacher(device, score_threshold=min(0.05, cfg.plate.no_score))
    face = RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device, score_threshold=min(0.1, cfg.face.no_score))

    def plate_detect(img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        return owl.detect(img, [Box("vehicle", 0, 0, w, h)])

    return TeacherJudge(cfg, plate_detect, face.detect)


def _fill(text: str, fields: dict[str, str]) -> str:
    """The placeholders replaced literally: str.format would choke on the JSON braces a prompt or command holds."""
    for key, value in fields.items():
        text = text.replace(key, value)
    return text


class CommandJudge:
    """An external command (an LLM CLI) given the crop; it must print a JSON object with "verdict". It sends the
    crop wherever that command sends it, so it only runs with VC_HARDNEG_ALLOW_LLM=1."""

    def __init__(self, cfg: LlmConfig, env: Mapping[str, str] = os.environ) -> None:
        if env.get(LLM_ENV) != "1":
            raise PermissionError(f"the LLM judge sends crops to an outside service; set {LLM_ENV}=1 to allow it")
        if not cfg.command:
            raise ValueError("configs/hardneg.json llm.command is empty")
        self.cfg = cfg
        self.name = f"llm:{cfg.name}"

    def judge(self, c: Candidate, image_path: Path) -> Judgment:
        prompt = _fill(self.cfg.prompt, {"{image}": str(image_path), "{cls}": c.cls, "{cls_zh}": CLS_ZH[c.cls]})
        fields = {"{image}": str(image_path), "{cls}": c.cls, "{cls_zh}": CLS_ZH[c.cls], "{prompt}": prompt}
        argv = [_fill(a, fields) for a in self.cfg.command]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=self.cfg.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return Judgment(c.id, "unsure", self.name, None, f"timeout after {self.cfg.timeout_s:g} s")
        except OSError as e:
            return Judgment(c.id, "unsure", self.name, None, f"cannot run: {e}")
        for text in reversed(re.findall(r"\{[^{}]*\}", done.stdout)):
            try:
                answer = json.loads(text)
            except ValueError:
                continue
            if isinstance(answer, dict) and answer.get("verdict") in ("yes", "no", "unsure"):
                return Judgment(c.id, answer["verdict"], self.name, None, str(answer.get("reason", ""))[:200])
        return Judgment(c.id, "unsure", self.name, None, f"no verdict in the output (exit {done.returncode})")


def run_judges(cands: list[Candidate], judgments_path: Path, crops_root: Path, frame_judge: FrameTeacherJudge,
               teacher_judge: TeacherJudge, llm_judge=None, rejudge: bool = False) -> Counter:
    """Judges every candidate without a judgment yet (all with rejudge, except those people judged); appends each
    verdict as it comes, so an interrupted run keeps what it did. Returns counts of "<judge>:<verdict>"."""
    done = resolve(read_judgments(judgments_path))
    counts: Counter = Counter()
    for c in cands:
        prior = done.get(c.id)
        if prior is not None and (not rejudge or prior.judge == "human"):
            continue
        j = frame_judge.judge(c)
        if j is None:
            crop = cv2.imread(str(crops_root / c.image))
            if crop is None:
                raise FileNotFoundError(f"{crops_root / c.image} cannot be read")
            j = teacher_judge.judge(c, crop)
        append_judgments(judgments_path, [j])
        counts[f"{j.judge}:{j.verdict}"] += 1
        if j.verdict == "unsure" and llm_judge is not None:
            k = llm_judge.judge(c, crops_root / c.image)
            append_judgments(judgments_path, [k])
            counts[f"{k.judge}:{k.verdict}"] += 1
    return counts
```

- [ ] **Step 3: 运行测试，确认通过**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_judges.py -q`
Expected: `7 passed`

- [ ] **Step 4: 提交**

```bash
git add py/vc_hardneg/judgments.py py/vc_hardneg/judges.py py/tests/test_hardneg_judges.py
git commit -m "feat: judgments and the frame-teacher, teacher and LLM judges" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 小图墙、试判与命令行

**Files:**
- Create: `py/vc_hardneg/review.py`, `py/vc_hardneg/pilot.py`, `py/vc_hardneg/cli.py`
- Test: `py/tests/test_hardneg_review.py`

**Interfaces:**
- Consumes: Tasks 6–7。
- Produces:
  - `write_review_page(cands: list[Candidate], crops_root: Path, out_dir: Path, verdicts: dict[str, list[Judgment]] | None = None, title: str = "") -> Path`：生成 `out_dir/index.html`，小图复制到 `out_dir/img/`。
  - `import_human(judgments_path: Path, export_file: Path, known_ids: set[str]) -> int`。
  - `STRATA`、`stratum(c) -> str | None`、`sample_pilot(cands, per_stratum=25, seed=0) -> list[Candidate]`、`pilot_report(ids: list[str], judgments: list[Judgment], max_no_false: float = 0.02) -> dict`。
  - CLI：`python -m vc_hardneg mine | judge | review | import-human | pilot | pilot-report`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_hardneg_review.py`:

```python
import json
import re

import cv2
import numpy as np

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.cli import main
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments
from vc_hardneg.pilot import STRATA, pilot_report, sample_pilot, stratum
from vc_hardneg.review import import_human, write_review_page


def _cand(cid, cls="plate", score=0.1, src="student"):
    return Candidate(id=cid, video="v.mp4", stem="v", frame=0, cls=cls, box=(110.0, 110.0, 130.0, 118.0), score=score,
                     src=src, track=1, dump_dir="d", dump_index=0, frame_teacher=0.0, image=f"crops/{cid}.png",
                     crop_origin=(100, 100))


def _crops(root, cands):
    (root / "crops").mkdir(exist_ok=True)
    for c in cands:
        cv2.imwrite(str(root / c.image), np.zeros((50, 50, 3), np.uint8))


def test_the_page_lists_every_crop_and_carries_no_outside_resource(tmp_path):
    cands = [_cand("a"), _cand("b", cls="face", src="yunet")]
    _crops(tmp_path, cands)
    page = write_review_page(cands, tmp_path, tmp_path / "page", {"a": [Judgment("a", "no", "teacher")]}, "试判")
    text = page.read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S).group(1))
    assert [d["id"] for d in data["items"]] == ["a", "b"]
    assert data["items"][0]["verdicts"] == [{"judge": "teacher", "verdict": "no"}]
    assert (tmp_path / "page" / "img" / "a.png").is_file()
    assert "http://" not in text and "https://" not in text


def test_import_human_appends_known_ids_only(tmp_path):
    path = tmp_path / "j.jsonl"
    export = tmp_path / "human.json"
    export.write_text(json.dumps([{"id": "a", "verdict": "yes"}, {"id": "b", "verdict": "no"}]))
    assert import_human(path, export, {"a", "b"}) == 2
    assert [j.judge for j in read_judgments(path)] == [HUMAN, HUMAN]
    export.write_text(json.dumps([{"id": "zzz", "verdict": "yes"}]))
    try:
        import_human(path, export, {"a"})
        raise AssertionError("an unknown id must be refused")
    except ValueError:
        pass


def test_pilot_sampling_is_stratified_and_reproducible():
    cands = [_cand(f"p{i}", score=0.05) for i in range(40)] + [_cand(f"q{i}", score=0.3) for i in range(10)] + \
            [_cand(f"y{i}", cls="face", src="yunet") for i in range(30)] + [_cand("s0", cls="face", src="student")] + \
            [_cand("both", cls="face", src="student+yunet", score=0.3)]
    a = sample_pilot(cands, 25, seed=0)
    assert a == sample_pilot(cands, 25, seed=0)
    counts = {s: sum(stratum(c) == s for c in a) for s in STRATA}
    assert counts == {"plate_lt02": 25, "plate_02_05": 10, "face_yunet": 25, "face_student": 1}
    assert stratum(cands[-1]) is None


def test_pilot_report_gates_on_real_targets_judged_no():
    ids = [f"c{i}" for i in range(100)]
    judged = [Judgment(i, "no", "teacher") for i in ids[:60]] + [Judgment(i, "yes", "teacher") for i in ids[60:]]
    truth = [Judgment(i, "no", HUMAN) for i in ids[:59]] + [Judgment(ids[59], "yes", HUMAN)] + \
            [Judgment(i, "yes", HUMAN) for i in ids[60:]]
    r = pilot_report(ids, judged + truth)
    assert r["pipeline"]["no"] == 60 and r["pipeline"]["no_but_real"] == 1
    assert r["passed"] is True  # 1 / 60 = 1.7 % <= 2 %
    truth[0] = Judgment(ids[0], "yes", HUMAN)
    assert pilot_report(ids, judged + truth)["passed"] is False  # 2 / 60 = 3.3 %


def test_cli_review_and_import_round_trip(tmp_path):
    cands = [_cand("a"), _cand("b")]
    _crops(tmp_path, cands)
    write_candidates(tmp_path / "candidates.jsonl", cands)
    append_judgments(tmp_path / "j.jsonl", [Judgment("a", "unsure", "teacher"), Judgment("b", "no", "teacher")])
    assert main(["review", "--candidates", str(tmp_path / "candidates.jsonl"), "--judgments", str(tmp_path / "j.jsonl"),
                 "--out", str(tmp_path / "page"), "--only", "unsure"]) == 0
    text = (tmp_path / "page" / "index.html").read_text(encoding="utf-8")
    assert '"id": "a"' in text and '"id": "b"' not in text
    (tmp_path / "h.json").write_text(json.dumps([{"id": "a", "verdict": "yes"}]))
    assert main(["import-human", "--candidates", str(tmp_path / "candidates.jsonl"), "--judgments",
                 str(tmp_path / "j.jsonl"), "--file", str(tmp_path / "h.json")]) == 0
    assert read_judgments(tmp_path / "j.jsonl")[-1] == Judgment("a", "yes", HUMAN)
```

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_review.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_hardneg.cli'`。

- [ ] **Step 2: 写 `review.py` 与 `pilot.py`**

`py/vc_hardneg/review.py`:

```python
"""A local review page for the crops people judge (spec 2026-10-21 §3.3, §3.4): no external resources; the
verdicts leave the page as a downloaded JSON file that import_human reads back."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from vc_hardneg.candidates import Candidate
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--yes:#1f7a4d;--no:#b3261e;--line:#dedbd3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#22221f;--ink:#ecebe6;--muted:#9a988f;--line:#3a3934}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,"PingFang SC",sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}
.card.yes{border-color:var(--yes)}.card.no{border-color:var(--no)}
.pic{position:relative;width:100%;aspect-ratio:1;background:#777;overflow:hidden}
.pic img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}
.box{position:absolute;border:2px solid #ff3b30;pointer-events:none}
.meta{color:var(--muted);font-size:12px;margin:6px 0}
button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button:focus-visible{outline:2px solid #3478f6;outline-offset:2px}
.yesb{color:var(--yes)}.nob{color:var(--no)}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span>
<button id="export">导出结果（human_judgments.json）</button><span class="meta">红框是被打码的框。点"是"/"不是"，看不清就跳过。</span></header>
<main id="grid"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_hardneg:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const grid = document.getElementById('grid');
const progress = () => { document.getElementById('progress').textContent = Object.keys(state).length + ' / ' + data.items.length; };
for (const it of data.items) {
  const card = document.createElement('div');
  card.className = 'card' + (state[it.id] ? ' ' + state[it.id] : '');
  const v = (it.verdicts || []).map(x => x.judge + ': ' + x.verdict).join('，');
  card.innerHTML = '<div class="pic"><img loading="lazy" alt=""><div class="box"></div></div>' +
    '<div class="meta"></div><button class="yesb">是</button> <button class="nob">不是</button>';
  card.querySelector('img').src = it.img;
  const b = card.querySelector('.box');
  b.style.left = it.box[0] + '%'; b.style.top = it.box[1] + '%';
  b.style.width = (it.box[2] - it.box[0]) + '%'; b.style.height = (it.box[3] - it.box[1]) + '%';
  card.querySelector('.meta').textContent = it.cls + ' ' + it.score.toFixed(2) + ' ' + it.src + (v ? ' ｜ ' + v : '');
  const set = (val) => { state[it.id] = val; card.className = 'card ' + val; save(); progress(); };
  card.querySelector('.yesb').onclick = () => set('yes');
  card.querySelector('.nob').onclick = () => set('no');
  grid.appendChild(card);
}
progress();
document.getElementById('export').onclick = () => {
  const out = Object.entries(state).map(([id, verdict]) => ({id, verdict}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'human_judgments.json';
  a.click();
};
</script></body></html>
"""


def write_review_page(cands: list[Candidate], crops_root: Path, out_dir: Path,
                      verdicts: dict[str, list[Judgment]] | None = None, title: str = "") -> Path:
    """out_dir/index.html with every candidate's crop (copied to out_dir/img), its box and the models' verdicts."""
    import cv2

    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for c in cands:
        src = crops_root / c.image
        shutil.copyfile(src, out_dir / "img" / f"{c.id}.png")
        h, w = cv2.imread(str(src)).shape[:2]
        ox, oy = c.crop_origin
        x1, y1, x2, y2 = c.box
        items.append({
            "id": c.id, "img": f"img/{c.id}.png", "cls": c.cls, "score": c.score, "src": c.src,
            "box": [100 * (x1 - ox) / w, 100 * (y1 - oy) / h, 100 * (x2 - ox) / w, 100 * (y2 - oy) / h],
            "verdicts": [{"judge": j.judge, "verdict": j.verdict} for j in (verdicts or {}).get(c.id, [])],
        })
    data = json.dumps({"title": title or out_dir.name, "items": items}, ensure_ascii=False, indent=1).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(_PAGE.replace("__TITLE__", html.escape(title or out_dir.name)).replace("__DATA__", data), encoding="utf-8")
    return page


def import_human(judgments_path: Path, export_file: Path, known_ids: set[str]) -> int:
    """Appends the page's export as human judgments. Raises ValueError for unknown ids or verdicts other than yes / no."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{export_file} is not a list of {{id, verdict}}")
    items = []
    for r in rows:
        if r.get("id") not in known_ids:
            raise ValueError(f"{export_file}: unknown id {r.get('id')!r}")
        if r.get("verdict") not in ("yes", "no"):
            raise ValueError(f"{export_file}: verdict of {r['id']} must be yes or no")
        items.append(Judgment(r["id"], r["verdict"], HUMAN))
    append_judgments(judgments_path, items)
    return len(items)
```

`py/vc_hardneg/pilot.py`:

```python
"""The 100-crop pilot (spec 2026-10-21 §3.4): a stratified sample, and the gate on what the models judged "no"."""

from __future__ import annotations

import random

from vc_hardneg.candidates import Candidate
from vc_hardneg.judgments import HUMAN, Judgment, resolve

STRATA = ("plate_lt02", "plate_02_05", "face_yunet", "face_student")


def stratum(c: Candidate) -> str | None:
    if c.cls == "plate":
        return "plate_lt02" if c.score < 0.2 else "plate_02_05" if c.score < 0.5 else None
    if c.cls == "face":
        return {"yunet": "face_yunet", "student": "face_student"}.get(c.src)
    return None


def sample_pilot(cands: list[Candidate], per_stratum: int = 25, seed: int = 0) -> list[Candidate]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    for s in STRATA:
        pool = sorted((c for c in cands if stratum(c) == s), key=lambda c: c.id)
        out.extend(rng.sample(pool, min(per_stratum, len(pool))))
    return out


def _confusion(model: dict[str, Judgment], truth: dict[str, str], ids: list[str]) -> dict:
    rows = {v: {"yes": 0, "no": 0} for v in ("yes", "no", "unsure")}
    for i in ids:
        if i in model and i in truth:
            rows[model[i].verdict][truth[i]] += 1
    no = rows["no"]["yes"] + rows["no"]["no"]
    return {"confusion": rows, "no": no, "no_but_real": rows["no"]["yes"],
            "no_but_real_rate": rows["no"]["yes"] / no if no else 0.0}


def pilot_report(ids: list[str], judgments: list[Judgment], max_no_false: float = 0.02) -> dict:
    """Per model judge and for the models together (the last non-human word per id): how their verdicts compare
    with the people's. Passes when, of what the models together judged "no", at most max_no_false is real."""
    truth = {j.id: j.verdict for j in judgments if j.judge == HUMAN}
    judged = [i for i in ids if i in truth]
    out = {"ids": len(ids), "judged_by_people": len(judged), "max_no_false": max_no_false, "judges": {}}
    for name in sorted({j.judge for j in judgments if j.judge != HUMAN}):
        out["judges"][name] = _confusion(resolve([j for j in judgments if j.judge == name]), truth, judged)
    out["pipeline"] = _confusion(resolve(judgments, include_human=False), truth, judged)
    out["passed"] = len(judged) == len(ids) and out["pipeline"]["no_but_real_rate"] <= max_no_false
    return out
```

- [ ] **Step 3: 写 `cli.py`**

`py/vc_hardneg/cli.py`:

```python
"""python -m vc_hardneg mine | judge | review | import-human | pilot | pilot-report (spec 2026-10-21 §3)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from vc_hardneg.candidates import CANDIDATES, mine, read_candidates
from vc_hardneg.config import DEFAULT_CONFIG, load_config
from vc_hardneg.judgments import read_judgments, resolve


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_hardneg")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine", help="candidates from vc_redact --dump-verify-crops runs")
    m.add_argument("--run", type=Path, action="append", required=True, help="a run dir <out>/<stem>; repeat")
    m.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    m.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name, help_ in (("judge", "judge candidates with the local teachers (and the LLM with --llm)"),
                        ("pilot", "sample the pilot, judge it and write its review page")):
        j = sub.add_parser(name, help=help_)
        j.add_argument("--candidates", type=Path, required=True)
        j.add_argument("--judgments", type=Path, required=True)
        j.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        j.add_argument("--device", default="mps")
        j.add_argument("--llm", action="store_true", help="ask the LLM command about what stays unsure (needs VC_HARDNEG_ALLOW_LLM=1)")
        if name == "judge":
            j.add_argument("--ids", type=Path, help="only the ids listed in this file, one per line")
            j.add_argument("--rejudge", action="store_true", help="judge again what models judged before (people's verdicts stay)")
        else:
            j.add_argument("--out", type=Path, required=True)
            j.add_argument("--per-stratum", type=int, default=25)
            j.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("review", help="local review page")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--judgments", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--only", choices=("unsure", "all"), default="unsure")
    r.add_argument("--ids", type=Path)
    i = sub.add_parser("import-human", help="append a review page's export as human judgments")
    i.add_argument("--candidates", type=Path, required=True)
    i.add_argument("--judgments", type=Path, required=True)
    i.add_argument("--file", type=Path, required=True)
    pr = sub.add_parser("pilot-report", help="compare the pilot's model verdicts with the people's")
    pr.add_argument("--pilot", type=Path, required=True)
    pr.add_argument("--judgments", type=Path, required=True)
    pr.add_argument("--max-no-false", type=float, default=0.02)
    return p


def _ids(path: Path | None) -> set[str] | None:
    return None if path is None else {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _judges(args):
    from vc_hardneg.judges import CommandJudge, FrameTeacherJudge, make_teacher_judge
    from vc_label.paths import offline_model_env

    offline_model_env()
    import torch

    cfg = load_config(args.config)
    llm = CommandJudge(cfg.llm) if args.llm else None  # PermissionError without VC_HARDNEG_ALLOW_LLM=1, before any model loads
    return FrameTeacherJudge(cfg), make_teacher_judge(cfg, torch.device(args.device)), llm


def _mine(args) -> int:
    s = mine(args.run, args.labels_root, args.frames_root, args.out, load_config(args.config))
    print(f"mine: {s.rows} dumped crops, {s.masked} masked on labelled frames, {s.candidates} candidates in {args.out / CANDIDATES}")
    return 0


def _judge(args) -> int:
    from vc_hardneg.judges import run_judges

    cands = read_candidates(args.candidates)
    wanted = _ids(args.ids)
    if wanted is not None:
        cands = [c for c in cands if c.id in wanted]
    frame, teacher, llm = _judges(args)
    if llm is not None:
        print(f"judge: the LLM judge may send up to {len(cands)} crops to {llm.name}", file=sys.stderr)
    counts = run_judges(cands, args.judgments, args.candidates.parent, frame, teacher, llm, args.rejudge)
    print("judge: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def _review(args) -> int:
    from vc_hardneg.review import write_review_page

    cands = read_candidates(args.candidates)
    judgments = read_judgments(args.judgments)
    final = resolve(judgments)
    wanted = _ids(args.ids)
    picked = [c for c in cands if (wanted is None or c.id in wanted)
              and (args.only == "all" or final.get(c.id) is None or final[c.id].verdict == "unsure")]
    by_id = defaultdict(list)
    for j in judgments:
        by_id[j.id].append(j)
    page = write_review_page(picked, args.candidates.parent, args.out, by_id, args.out.name)
    print(f"review: {len(picked)} crops in {page}")
    return 0


def _import(args) -> int:
    from vc_hardneg.review import import_human

    n = import_human(args.judgments, args.file, {c.id for c in read_candidates(args.candidates)})
    print(f"import-human: {n} verdicts appended to {args.judgments}")
    return 0


def _pilot(args) -> int:
    from vc_hardneg.judges import run_judges
    from vc_hardneg.pilot import sample_pilot
    from vc_hardneg.review import write_review_page

    cands = sample_pilot(read_candidates(args.candidates), args.per_stratum, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_ids.txt").write_text("".join(f"{c.id}\n" for c in cands))
    frame, teacher, llm = _judges(args)
    counts = run_judges(cands, args.judgments, args.candidates.parent, frame, teacher, llm)
    by_id = defaultdict(list)
    for j in read_judgments(args.judgments):
        by_id[j.id].append(j)
    page = write_review_page(cands, args.candidates.parent, args.out, by_id, "试判 100 张")
    print(f"pilot: {len(cands)} crops; " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"; review {page}")
    return 0


def _pilot_report(args) -> int:
    from vc_hardneg.pilot import pilot_report

    ids = sorted(_ids(args.pilot / "pilot_ids.txt"))
    r = pilot_report(ids, read_judgments(args.judgments), args.max_no_false)
    (args.pilot / "report.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p = r["pipeline"]
    print(f"pilot-report: {r['judged_by_people']}/{r['ids']} judged by people; models said no {p['no']} times, "
          f"{p['no_but_real']} of them real ({p['no_but_real_rate']:.1%}, limit {args.max_no_false:.0%}): "
          + ("PASS" if r["passed"] else "FAIL"))
    return 0 if r["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"mine": _mine, "judge": _judge, "review": _review, "import-human": _import, "pilot": _pilot,
                "pilot-report": _pilot_report}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError, PermissionError) as e:
        print(f"vc_hardneg: {e}", file=sys.stderr)
        return 2
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_hardneg_review.py tests/test_hardneg_judges.py tests/test_hardneg_candidates.py -q`
Expected: 全部 passed（16 个）。

- [ ] **Step 5: 提交**

```bash
git add py/vc_hardneg/review.py py/vc_hardneg/pilot.py py/vc_hardneg/cli.py py/vc_hardneg/__main__.py py/tests/test_hardneg_review.py
git commit -m "feat: review page, pilot sampling and report, vc_hardneg CLI" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 转储训练段、挖候选、100 张试判（操作步骤，含用户关卡）

**Files:** 不改代码。产出在 `data/`（不进仓库）：`data/redacted/fp-dump/<stem>/`、`data/hardneg/v1/`。

- [ ] **Step 1: 告知用户并编译 Release**

先在对话里告诉用户：接下来要用 Release 版 `vc_redact` 跑 10 个训练段，每段约 40 秒，合计约 7 分钟，期间占满本机。然后：

```bash
cmake --build build/cpp-release --target vc_redact -j4
```

- [ ] **Step 2: 带转储跑 10 个训练段（不复核）**

每 48 帧转储一次，和老师标注的帧对齐：

```bash
for s in sg_drive_300s sg_drive_600s sg_drive_900s sg_drive_1500s sg_drive_1800s sg_drive_2100s sg_drive_2700s sg_drive_3000s sg_drive_3300s sg_drive_3600s; do
  ORT_DISABLE_TELEMETRY=1 ./build/cpp-release/vc_redact --input data/videos/$s.mp4 --out data/redacted/fp-dump \
    --model data/models/r3 --face-model data/models/yunet --config configs/redact.json \
    --dump-verify-crops --dump-verify-every 48 --overwrite || break
done
```

Expected：每段打印摘要，`data/redacted/fp-dump/<stem>/verify_crops.jsonl` 非空。

- [ ] **Step 3: 挖候选**

```bash
cd py && ../.venv/bin/python -m vc_hardneg mine $(for s in 300 600 900 1500 1800 2100 2700 3000 3300 3600; do echo --run ../data/redacted/fp-dump/sg_drive_${s}s; done) \
  --labels-root ../data/labels --frames-root ../data/frames --out ../data/hardneg/v1 --config ../configs/hardneg.json
```

Expected：`mine: N dumped crops, M masked on labelled frames, K candidates`，K 在几千量级。若 K < 500，先查 `processed.txt` 的帧号与 48 的对齐，再继续。

- [ ] **Step 4: 试判 100 张**

先告诉用户：本地老师（OWLv2、RetinaFace）在 MPS 上判 100 张，预计几分钟。

```bash
cd py && ../.venv/bin/python -m vc_hardneg pilot --candidates ../data/hardneg/v1/candidates.jsonl \
  --judgments ../data/hardneg/v1/judgments.jsonl --out ../data/hardneg/v1/pilot --config ../configs/hardneg.json
```

若用户已提供 Codex 的调用方式：先把 `configs/hardneg.json` 的 `llm.command` 填好，再加 `--llm`，并设置 `VC_HARDNEG_ALLOW_LLM=1`。

- [ ] **Step 5: 用户关卡（停下等用户）**

请用户打开 `data/hardneg/v1/pilot/index.html`，逐张点"是 / 不是"，导出 `human_judgments.json` 后告诉你文件位置。然后：

```bash
cd py && ../.venv/bin/python -m vc_hardneg import-human --candidates ../data/hardneg/v1/candidates.jsonl \
  --judgments ../data/hardneg/v1/judgments.jsonl --file <导出文件>
../.venv/bin/python -m vc_hardneg pilot-report --pilot ../data/hardneg/v1/pilot --judgments ../data/hardneg/v1/judgments.jsonl
```

- **PASS**：把 `report.json` 的混淆矩阵报给用户，继续 Task 10。
- **FAIL**：看判"no"却是真目标的那几张，属于哪一类、放大后老师分数多少。据此调高 `configs/hardneg.json` 里对应类的 `no_score`，把边界情况推给 `unsure`。然后用 `judge --ids ../data/hardneg/v1/pilot/pilot_ids.txt --rejudge` 重判、重出报告（人工结论保留）。把调整和结果报给用户，等用户确认后再继续。

---

### Task 10: 放量判定与人工复核（操作步骤）

- [ ] **Step 1: 判定全部候选**

先告诉用户预计耗时：按试判的速度乘以候选数，通常几十分钟。放后台跑：

```bash
cd py && ../.venv/bin/python -m vc_hardneg judge --candidates ../data/hardneg/v1/candidates.jsonl \
  --judgments ../data/hardneg/v1/judgments.jsonl --config ../configs/hardneg.json
```

中断后重跑会跳过已判的。

- [ ] **Step 2: 人工复核"拿不准"的**

```bash
cd py && ../.venv/bin/python -m vc_hardneg review --candidates ../data/hardneg/v1/candidates.jsonl \
  --judgments ../data/hardneg/v1/judgments.jsonl --out ../data/hardneg/v1/review-unsure --only unsure
```

把张数告诉用户，请用户点完并导出，再用 `import-human` 读回。张数太多（超过 1,000）时，先问用户是只复核一部分（剩下的 `unsure` 不进训练），还是启用 LLM 先过一遍。

- [ ] **Step 3: 统计**

```bash
cd py && ../.venv/bin/python -c "
from vc_hardneg.candidates import read_candidates; from vc_hardneg.judgments import read_judgments, resolve
from collections import Counter
c={x.id:x for x in read_candidates('../data/hardneg/v1/candidates.jsonl')}; r=resolve(read_judgments('../data/hardneg/v1/judgments.jsonl'))
print(Counter((c[i].cls, j.verdict) for i, j in r.items() if i in c))"
```

把每类的 yes / no / unsure 数量报给用户。每类 yes 和 no 都至少要有 200 张才往下训练；不够时停下来和用户商量。

---

### Task 11: 复核模型的训练与导出（`vc_train verifier`）

**Files:**
- Create: `py/vc_train/verifier.py`, `py/tests/test_train_verifier.py`
- Modify: `py/vc_train/cli.py`

**Interfaces:**
- Consumes: `read_candidates`（Task 6）、`read_judgments`、`resolve`（Task 7）、`cpu_session`（`vc_train/ort_cpu.py`）；转储格式见 Task 4。
- Produces:
  - `VERIFIER_CLASSES = ("face", "plate")`。
  - `Sample(id, stem, cls_index, label, dump_dir, dump_index)`。
  - `load_samples(candidates: Path, judgments: Path) -> list[Sample]`、`load_crops(samples) -> np.ndarray`（[N, S, S, 3] uint8）。
  - `VerifierNet(n_classes=2)`。
  - `train_verifier(samples, crops, val_stems, epochs, batch, lr, device, seed) -> TrainResult`。
  - `pass_thresholds(probs, labels, cls_index, keep=0.995) -> dict[str, float]`。
  - `export_verifier(model, out_dir, size, batch, meta) -> Path`。
  - CLI：`python -m vc_train verifier ...`。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_train_verifier.py`:

```python
import json

import numpy as np
import pytest

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments
from vc_train.verifier import (
    VERIFIER_CLASSES, VerifierNet, export_verifier, load_crops, load_samples, pass_thresholds, train_verifier,
)

SIZE = 16


def _dump(dir_, crops):
    dir_.mkdir(parents=True)
    (dir_ / "verify_crops.bin").write_bytes(np.stack(crops).astype(np.uint8).tobytes())
    (dir_ / "verify_crops.jsonl").write_text("".join(json.dumps({"index": i, "size": SIZE}) + "\n" for i in range(len(crops))))


def _toy(tmp_path, n=120):
    """Bright crops are real, dark ones are not; half plates, half faces; two stems."""
    rng = np.random.default_rng(0)
    cands, judged = [], []
    for stem in ("a", "b"):
        crops = []
        for i in range(n):
            real = i % 2 == 0
            crops.append(np.clip(rng.normal(200 if real else 50, 20, (SIZE, SIZE, 3)), 0, 255))
            cid = f"{stem}{i}"
            cands.append(Candidate(id=cid, video="v", stem=stem, frame=i, cls=VERIFIER_CLASSES[(i // 2) % 2],
                                   box=(0, 0, 1, 1), score=0.1, src="student", track=i, dump_dir=str(tmp_path / stem),
                                   dump_index=i, frame_teacher=0.0, image="", crop_origin=(0, 0)))
            judged.append(Judgment(cid, "yes" if real else "no", "teacher"))
        _dump(tmp_path / stem, crops)
    judged.append(Judgment("a0", "unsure", HUMAN))  # a person's unsure: left out
    write_candidates(tmp_path / "c.jsonl", cands)
    append_judgments(tmp_path / "j.jsonl", judged)
    return tmp_path / "c.jsonl", tmp_path / "j.jsonl"


def test_samples_take_yes_and_no_and_read_the_right_crop(tmp_path):
    c, j = _toy(tmp_path)
    samples = load_samples(c, j)
    assert len(samples) == 239 and "a0" not in {s.id for s in samples}
    crops = load_crops(samples)
    assert crops.shape == (239, SIZE, SIZE, 3) and crops.dtype == np.uint8
    first = next(i for i, s in enumerate(samples) if s.id == "a2")
    assert crops[first].mean() > 150 and samples[first].label == 1


def test_pass_thresholds_keep_the_asked_share_of_real_crops():
    probs = np.linspace(0.01, 1.0, 100)
    labels = np.ones(100, int)
    cls = np.zeros(100, int)
    t = pass_thresholds(probs, labels, cls, keep=0.95)
    assert (probs >= t["face"]).mean() >= 0.95
    assert t["plate"] == 0.0  # no real plate in the set: pass everything


def test_training_separates_the_toy_and_the_export_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    c, j = _toy(tmp_path)
    samples = load_samples(c, j)
    result = train_verifier(samples, load_crops(samples), ["b"], epochs=15, batch=32, lr=1e-2, device="cpu", seed=0)
    acc = ((result.val_probs >= 0.5).astype(int) == result.val_labels).mean()
    assert acc > 0.9
    out = tmp_path / "model"
    onnx_path = export_verifier(result.model, out, SIZE, 4, {"name": "toy", "pass_threshold": {"face": 0.5, "plate": 0.5}})
    meta = json.loads((out / "model.json").read_text())
    assert meta["kind"] == "verifier" and meta["input"] == {"size": SIZE, "batch": 4, "color": "RGB",
                                                             "scale": 1 / 255, "pad_value": 114}
    from vc_train.ort_cpu import cpu_session

    x = np.random.default_rng(1).random((4, 3, SIZE, SIZE), dtype=np.float32)
    ort_out = cpu_session(onnx_path).run(None, {"crops": x})[0]
    with torch.no_grad():
        torch_out = result.model.cpu().eval()(torch.from_numpy(x)).numpy()
    assert ort_out.shape == (4, 2) and np.abs(ort_out - torch_out).max() < 1e-4
```

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_verifier.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_train.verifier'`。

- [ ] **Step 2: 写实现**

`py/vc_train/verifier.py`:

```python
"""The verifier (spec 2026-10-21 §4): a small CNN scoring vc_redact's dumped crops, trained on judged candidates."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vc_hardneg.candidates import read_candidates
from vc_hardneg.judgments import read_judgments, resolve

VERIFIER_CLASSES = ("face", "plate")
PAD_VALUE = 114
CROP_FACTOR = 2.5
MIN_CROP_PX = 32


@dataclass(frozen=True)
class Sample:
    id: str
    stem: str
    cls_index: int  # into VERIFIER_CLASSES
    label: int  # 1 real, 0 not
    dump_dir: str
    dump_index: int


def load_samples(candidates: Path, judgments: Path) -> list[Sample]:
    """Candidates whose final verdict is yes or no (unsure ones are left out), in candidates-file order."""
    final = resolve(read_judgments(judgments))
    out = []
    for c in read_candidates(candidates):
        j = final.get(c.id)
        if j is None or j.verdict == "unsure" or c.cls not in VERIFIER_CLASSES:
            continue
        out.append(Sample(c.id, c.stem, VERIFIER_CLASSES.index(c.cls), int(j.verdict == "yes"), c.dump_dir, c.dump_index))
    return out


def load_crops(samples: list[Sample]) -> np.ndarray:
    """[N, S, S, 3] uint8, the dumped bytes of each sample (all dumps must share one size)."""
    sizes, maps = {}, {}
    for d in sorted({s.dump_dir for s in samples}):
        with (Path(d) / "verify_crops.jsonl").open(encoding="utf-8") as f:
            sizes[d] = json.loads(f.readline())["size"]
        maps[d] = np.memmap(Path(d) / "verify_crops.bin", dtype=np.uint8, mode="r").reshape(-1, sizes[d], sizes[d], 3)
    if len(set(sizes.values())) > 1:
        raise ValueError(f"the dumps have different crop sizes: {sizes}")
    return np.stack([np.asarray(maps[s.dump_dir][s.dump_index]) for s in samples])


def _torch():
    import torch
    from torch import nn

    return torch, nn


def VerifierNet(n_classes: int = 2):  # noqa: N802 (a class factory: torch is imported lazily)
    torch, nn = _torch()

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers, cin = [], 3
            for cout in (16, 32, 64, 128):
                layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                           nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
                cin = cout
            self.features = nn.Sequential(*layers)
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):
            return self.head(self.features(x).mean(dim=(2, 3)))

    return _Net()


@dataclass
class TrainResult:
    model: object
    val_probs: np.ndarray  # pass probability of each validation sample (its own class's head)
    val_labels: np.ndarray
    val_cls: np.ndarray
    train_count: int


def _augment(torch, x):
    """Random horizontal flip, brightness +-15 %, shift up to 4 px with pad-value borders."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = (x * (0.85 + 0.3 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))).clamp(0, 1)
    pad = torch.nn.functional.pad(x, (4, 4, 4, 4), value=PAD_VALUE / 255)
    dx, dy = np.random.randint(0, 9, 2)
    return pad[:, :, dy : dy + x.shape[2], dx : dx + x.shape[3]]


def train_verifier(samples: list[Sample], crops: np.ndarray, val_stems: list[str], epochs: int = 30, batch: int = 256,
                   lr: float = 1e-3, device: str = "mps", seed: int = 0) -> TrainResult:
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    val = np.array([s.stem in val_stems for s in samples])
    if val.all() or not val.any():
        raise ValueError(f"--val-stem {val_stems} must leave both training and validation samples")
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255
    y = torch.tensor([s.label for s in samples], dtype=torch.float32)
    k = torch.tensor([s.cls_index for s in samples])
    tr = np.flatnonzero(~val)
    model = VerifierNet(len(VERIFIER_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + batch - 1) // batch))
    # balance yes / no within each class: each sample weighs 1 / (count of its (class, label))
    groups = k[tr] * 2 + y[tr].long()
    weight = 1.0 / torch.bincount(groups, minlength=4).float().clamp(min=1)[groups]
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        model.train()
        order = tr[torch.multinomial(weight, len(tr), replacement=True).numpy()]
        for start in range(0, len(order), batch):
            idx = torch.from_numpy(order[start : start + batch])
            xb = _augment(torch, x[idx].to(device))
            logits = model(xb).gather(1, k[idx].to(device)[:, None])[:, 0]
            loss = loss_fn(logits, y[idx].to(device)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    vi = np.flatnonzero(val)
    with torch.no_grad():
        logits = torch.cat([model(x[vi[s : s + batch]].to(device)).cpu() for s in range(0, len(vi), batch)])
    probs = torch.sigmoid(logits.gather(1, k[vi][:, None])[:, 0]).numpy()
    return TrainResult(model, probs, y[vi].numpy().astype(int), k[vi].numpy(), len(tr))


def pass_thresholds(probs: np.ndarray, labels: np.ndarray, cls_index: np.ndarray, keep: float = 0.995) -> dict[str, float]:
    """Per class the largest threshold that still passes `keep` of its real crops (0 when it has none)."""
    out = {}
    for i, name in enumerate(VERIFIER_CLASSES):
        real = np.sort(probs[(cls_index == i) & (labels == 1)])
        out[name] = 0.0 if len(real) == 0 else float(real[int(np.floor((1 - keep) * len(real)))])
    return out


def export_verifier(model, out_dir: Path, size: int, batch: int, meta: dict) -> Path:
    """model.onnx ([batch, 3, size, size] -> [batch, 2]) + model.json into out_dir (absent or empty), all at once."""
    torch, _ = _torch()
    from vc_train.ort_cpu import cpu_session

    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} is not empty; choose a new --out")
    model = model.cpu().eval()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        onnx_path = staging / "model.onnx"
        dummy = torch.rand(batch, 3, size, size)
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["crops"], output_names=["logits"], opset_version=17,
                          dynamo=False)
        session = cpu_session(onnx_path)
        if [list(i.shape) for i in session.get_inputs()] != [[batch, 3, size, size]]:
            raise ValueError("the exported verifier does not have a static input")
        with torch.no_grad():
            diff = np.abs(session.run(None, {"crops": dummy.numpy()})[0] - model(dummy).numpy()).max()
        if diff > 1e-4:
            raise ValueError(f"onnxruntime and torch disagree by {diff}")
        info = {
            "name": meta.get("name", out_dir.name), "kind": "verifier", "classes": list(VERIFIER_CLASSES),
            "input": {"size": size, "batch": batch, "color": "RGB", "scale": 1 / 255, "pad_value": PAD_VALUE},
            "crop": {"factor": CROP_FACTOR, "min_px": MIN_CROP_PX},
            **{k: v for k, v in meta.items() if k != "name"},
        }
        (staging / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # empty, checked above
        os.replace(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir / "model.onnx"
```

`py/vc_train/cli.py`：

- 在 `_parser()` 的 `yunet` 子命令之后加：

```python
    v = sub.add_parser("verifier", help="train the verifier on judged candidates (spec 2026-10-21 §4)")
    v.add_argument("--candidates", type=Path, required=True)
    v.add_argument("--judgments", type=Path, required=True)
    v.add_argument("--val-stem", action="append", required=True, help="stems held out for validation; repeat")
    v.add_argument("--out", type=Path, required=True, help="new or empty directory for model.onnx + model.json")
    v.add_argument("--epochs", type=int, default=30)
    v.add_argument("--batch", type=int, default=256)
    v.add_argument("--lr", type=float, default=1e-3)
    v.add_argument("--device", default="mps")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--keep", type=float, default=0.995, help="share of real validation crops each class's threshold passes")
    v.add_argument("--onnx-batch", type=int, default=16)
```

- 加处理函数：

```python
def _verifier(args) -> int:
    import hashlib

    import numpy as np

    from vc_train.verifier import VERIFIER_CLASSES, export_verifier, load_crops, load_samples, pass_thresholds, train_verifier

    samples = load_samples(args.candidates, args.judgments)
    crops = load_crops(samples)
    r = train_verifier(samples, crops, args.val_stem, args.epochs, args.batch, args.lr, args.device, args.seed)
    thresholds = pass_thresholds(r.val_probs, r.val_labels, r.val_cls, args.keep)
    metrics = {}
    for i, name in enumerate(VERIFIER_CLASSES):
        m = r.val_cls == i
        real, fake = m & (r.val_labels == 1), m & (r.val_labels == 0)
        metrics[name] = {
            "val_real": int(real.sum()), "val_not_real": int(fake.sum()),
            "real_passed": float((r.val_probs[real] >= thresholds[name]).mean()) if real.any() else None,
            "not_real_rejected": float((r.val_probs[fake] < thresholds[name]).mean()) if fake.any() else None,
        }
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    meta = {"name": args.out.name, "pass_threshold": thresholds, "metrics": metrics,
            "train": {"samples": r.train_count, "val_stems": args.val_stem, "epochs": args.epochs, "keep": args.keep,
                      "candidates_sha256": sha(args.candidates), "judgments_sha256": sha(args.judgments)}}
    onnx_path = export_verifier(r.model, args.out, crops.shape[1], args.onnx_batch, meta)
    print(f"verifier: {onnx_path}; thresholds {thresholds}")
    for name, m in metrics.items():
        print(f"  {name}: {m}")
    return 0
```

- `main` 的分派表加 `"verifier": _verifier`。

- [ ] **Step 3: 运行测试，确认通过**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_train_verifier.py -q`
Expected: `3 passed`

- [ ] **Step 4: 提交**

```bash
git add py/vc_train/verifier.py py/vc_train/cli.py py/tests/test_train_verifier.py
git commit -m "feat: train and export the verifier from judged candidates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: 训练 verifier-v1（操作步骤）

- [ ] **Step 1: 告知并训练**

告诉用户：训练在 MPS 上跑，30 轮几千张 64×64 小图，预计 5–15 分钟。然后：

```bash
cd py && ../.venv/bin/python -m vc_train verifier --candidates ../data/hardneg/v1/candidates.jsonl \
  --judgments ../data/hardneg/v1/judgments.jsonl --val-stem sg_drive_3000s --val-stem sg_drive_3600s \
  --out ../data/models/verifier-v1
```

Expected：打印门槛和每类的 `real_passed`（应 ≥ 0.995）、`not_real_rejected`。

- [ ] **Step 2: 判断能否往下**

- `not_real_rejected` 两类都 < 0.3：小模型拦不住多少，先报给用户；可以问用户是否同意下载 MobileNetV3-Small 预训练权重（约 10 MB）后再试。
- 否则进入 Task 13。

---

### Task 13: 误打计数、逐帧覆盖分段、`run` 透传复核模型、新漏核查

**Files:**
- Create: `py/vc_eval/falsepos.py`, `py/vc_eval/newmisses.py`, `py/tests/test_eval_falsepos.py`
- Modify: `py/vc_eval/temporal.py`, `py/vc_eval/run.py`, `py/vc_eval/cli.py`, `py/tests/test_eval_run.py`

**Interfaces:**
- Produces:
  - `count_false_positives(detections: Path, labels_dir: Path, ref_iou=0.1, excluded_iou=0.3) -> dict`，返回 `{cls: {"masked": n, "unreferenced": n, "by_src": {src: [masked, unreferenced]}}}`。
  - `TemporalReport.bins: dict[str, Rate]`，键为 `face>=0.8`、`face0.5-0.8`、`face<0.5`、`plate>=0.5`、`plate0.2-0.5`、`plate<0.2`。
  - `run --verifier-model <dir>`：只传给稀疏运行，不传给 `--detect-every-frame` 的逐帧参照运行；`summary.json` 记录 `verifier_model`，每段的 `temporal_bins`。
  - `new_misses(frames: Path, labels: Path, baseline_masks: Path, candidate_masks: Path, recall_targets) -> list[Miss]`；CLI `vc_eval falsepos`、`vc_eval newmisses`（后者把新漏的框裁图存进 `--out`）。

- [ ] **Step 1: 写失败的测试**

`py/tests/test_eval_falsepos.py`:

```python
import json

from vc_eval.falsepos import count_false_positives


def _det(frame, box, cls="plate", src="student", confirmed=True, verified=None):
    d = {"video": "v.mp4", "frame": frame, "pts_ms": 0, "cls": cls, "x1": box[0], "y1": box[1], "x2": box[2],
         "y2": box[3], "score": 0.1, "src": src, "confirmed": confirmed}
    if verified is not None:
        d["verified"] = verified
    return d


def test_counts_masked_boxes_without_a_teacher_box(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "processed.txt").write_text("0\n48\n")
    teacher = [
        {"video": "v.mp4", "frame": 0, "pts_ms": 0, "cls": "plate", "x1": 10, "y1": 10, "x2": 30, "y2": 20, "score": 0.15, "src": "t"},
        {"video": "v.mp4", "frame": 48, "pts_ms": 0, "cls": "plate", "x1": 100, "y1": 100, "x2": 120, "y2": 110, "score": 0.9, "src": "t"},
    ]
    (labels / "merged.jsonl").write_text("".join(json.dumps(t) + "\n" for t in teacher))
    (labels / "excluded_refs.csv").write_text("frame,cls,x1,y1,x2,y2,reason\n48,plate,100,100,120,110,尾灯\n")
    dets = tmp_path / "d.jsonl"
    dets.write_text("".join(json.dumps(d) + "\n" for d in [
        _det(0, (11, 10, 30, 20)),                        # referenced
        _det(0, (200, 200, 220, 210)),                    # not referenced
        _det(0, (300, 300, 320, 310), confirmed=False),   # never masked: not counted
        _det(0, (400, 400, 420, 410), verified=False),    # rejected by the verifier: not counted
        _det(48, (100, 100, 120, 110), src="yunet", cls="plate"),  # on an excluded teacher box: not referenced
        _det(12, (0, 0, 5, 5)),                           # frame not labelled: not counted
    ]))
    r = count_false_positives(dets, labels)
    assert r["plate"]["masked"] == 3 and r["plate"]["unreferenced"] == 2
    assert r["plate"]["by_src"] == {"student": [2, 1], "yunet": [1, 1]}
    assert r["face"] == {"masked": 0, "unreferenced": 0, "by_src": {}}
```

在 `py/tests/test_eval_run.py` 加一个测试：仿照该文件里已有的、用假 `runner` 捕获 `vc_redact` 命令的测试（搜索 `face_model=` 找到它），把 `verifier_model` 设为一个含 `model.onnx` 和 `model.json` 的临时目录，然后断言：
- 每条不含 `--detect-every-frame` 的命令都包含 `["--verifier-model", str(dir)]`；
- 含 `--detect-every-frame` 的命令都不包含 `--verifier-model`；
- `summary.json` 的 `verifier_model` 等于 `str(dir)`。

写法与该文件现有的 `face_model` 测试保持一致。

Run: `cd py && ../.venv/bin/python -m pytest tests/test_eval_falsepos.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vc_eval.falsepos'`。

- [ ] **Step 2: 写 `falsepos.py`**

`py/vc_eval/falsepos.py`:

```python
"""Masks no teacher box explains (spec 2026-10-21 §1, §6): on labelled frames, masked detections that overlap no
same-class teacher box (IoU >= ref_iou), teacher boxes on an excluded_refs.csv row not counting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vc_eval.records import EXCLUDED_REFS, read_excluded
from vc_label.boxes import iou
from vc_label.contracts import LABEL_CLASSES, Box, read_jsonl


def count_false_positives(detections: Path, labels_dir: Path, ref_iou: float = 0.1, excluded_iou: float = 0.3) -> dict:
    frames = {int(v) for v in (labels_dir / "processed.txt").read_text().split()}
    teacher: dict[int, list[Box]] = defaultdict(list)
    for d in read_jsonl(labels_dir / "merged.jsonl"):
        teacher[d.frame].append(d.box)
    excluded: dict[int, list[Box]] = defaultdict(list)
    if (labels_dir / EXCLUDED_REFS).is_file():
        for e in read_excluded(labels_dir):
            excluded[e.frame].append(e.box)
    out = {c: {"masked": 0, "unreferenced": 0, "by_src": {}} for c in LABEL_CLASSES}
    with Path(detections).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] not in frames or r.get("confirmed") is False or r.get("verified") is False or r["cls"] not in out:
                continue
            box = Box(r["cls"], r["x1"], r["y1"], r["x2"], r["y2"])
            refs = [t for t in teacher[r["frame"]] if t.cls == box.cls
                    and not any(e.cls == box.cls and iou(e, t) >= excluded_iou for e in excluded[r["frame"]])]
            unref = not any(iou(t, box) >= ref_iou for t in refs)
            o = out[box.cls]
            o["masked"] += 1
            o["unreferenced"] += int(unref)
            s = o["by_src"].setdefault(r["src"], [0, 0])
            s[0] += 1
            s[1] += int(unref)
    return out
```

- [ ] **Step 3: 逐帧覆盖按分数段统计**

`py/vc_eval/temporal.py`：
- `TemporalReport` 加字段 `bins: dict[str, Rate] = field(default_factory=dict)`。
- 加函数：

```python
def score_bin(cls: str, score: float) -> str:
    """The score bands of docs/reports/stable-masks-2026-10-19.md."""
    if cls == "face":
        return "face>=0.8" if score >= 0.8 else "face0.5-0.8" if score >= 0.5 else "face<0.5"
    return "plate>=0.5" if score >= 0.5 else "plate0.2-0.5" if score >= 0.2 else "plate<0.2"
```

- 在 `evaluate_temporal` 的循环里，`report.classes[box.cls] = ...` 之后加：

```python
            key = score_bin(box.cls, box.score)
            report.bins[key] = report.bins.get(key, Rate()).add(hit)
```

- `run.py` 写每段 `"temporal"` 的地方旁边加：`"temporal_bins": {k: rate_json(v) for k, v in sorted(temporal.bins.items())},`。

- [ ] **Step 4: `run --verifier-model` 与新漏核查**

`py/vc_eval/run.py`：
- `RunConfig` 加 `verifier_model: Path | None = None  # vc_train verifier model dir, passed to the sparse vc_redact runs only`。
- 读文件的检查列表里与 `face_model` 同样处理：存在时加进 `reads`，并检查目录里有 `model.onnx` 和 `model.json`。
- `_redact` 改为：

```python
    face = ["--face-model", str(cfg.face_model)] if cfg.face_model is not None else []
    # the dense run is the reference of the temporal check: it must see every box, verified or not
    verifier = (["--verifier-model", str(cfg.verifier_model)]
                if cfg.verifier_model is not None and "--detect-every-frame" not in extra else [])
    command = [
        str(cfg.vc_redact), "--input", str(cfg.videos / f"{stem}.mp4"), "--out", str(out), "--model", str(cfg.model),
        *face, *verifier, "--config", str(cfg.config), *extra, "--overwrite",
    ]
```

- `summary` 里 `"face_model"` 旁边加 `"verifier_model": None if cfg.verifier_model is None else str(cfg.verifier_model),`。

`py/vc_eval/newmisses.py`:

```python
"""Reference boxes the candidate masks miss but the baseline masks covered (spec 2026-10-21 §6), with their crops."""

from __future__ import annotations

from pathlib import Path

import cv2

from vc_eval.coverage import Miss, evaluate_coverage
from vc_label.boxes import iou


def new_misses(frames: Path, labels: Path, baseline_masks: Path, candidate_masks: Path, recall_targets) -> list[Miss]:
    base = evaluate_coverage(frames, labels, baseline_masks, recall_targets)
    cand = evaluate_coverage(frames, labels, candidate_masks, recall_targets)
    old = [(m.frame, m.box) for m in base.misses + base.uncertain_misses]
    return [m for m in cand.misses + cand.uncertain_misses
            if not any(f == m.frame and b.cls == m.box.cls and iou(b, m.box) >= 0.9 for f, b in old)]


def write_crops(misses: list[Miss], frames: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(misses):
        image = cv2.imread(str(frames / f"{m.frame:06d}.jpg"))
        b = m.box
        side = max(96.0, 3 * max(b.width, b.height))
        cx, cy = b.center
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        crop = image[y0 : int(cy + side / 2), x0 : int(cx + side / 2)].copy()
        cv2.rectangle(crop, (int(b.x1 - x0), int(b.y1 - y0)), (int(b.x2 - x0), int(b.y2 - y0)), (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{i:03d}_{m.frame:06d}_{b.cls}_{b.score:.2f}.jpg"), crop)
```

`py/vc_eval/cli.py`：
- `_parser()` 加三处：

```python
    f = sub.add_parser("falsepos", help="masks no teacher box explains, by class and source (labelled frames)")
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--detections", type=Path, required=True)
    f.add_argument("--json", type=Path)
    n = sub.add_parser("newmisses", help="reference boxes the candidate masks miss but the baseline's covered")
    n.add_argument("--frames", type=Path, required=True)
    n.add_argument("--labels", type=Path, required=True)
    n.add_argument("--baseline", type=Path, required=True, help="masks.jsonl")
    n.add_argument("--candidate", type=Path, required=True, help="masks.jsonl")
    n.add_argument("--out", type=Path, required=True, help="crops of the new misses go here")
    n.add_argument("--recall-target", default="")
```

  `run` 子命令加 `r.add_argument("--verifier-model", type=Path, default=None, help="vc_train verifier model dir, passed to the sparse vc_redact runs as --verifier-model")`，并在构造 `RunConfig` 处传 `verifier_model=args.verifier_model`。

- 加处理函数并加入分派表：

```python
def _falsepos(args) -> int:
    from vc_eval.falsepos import count_false_positives

    r = count_false_positives(args.detections, args.labels)
    for cls, c in r.items():
        share = c["unreferenced"] / c["masked"] if c["masked"] else 0.0
        print(f"{cls}: {c['unreferenced']}/{c['masked']} masked boxes match no teacher box ({share:.1%}); by source {c['by_src']}")
    _write_json(args.json, r)
    return 0


def _newmisses(args) -> int:
    from vc_eval.coverage import parse_recall_targets
    from vc_eval.newmisses import new_misses, write_crops

    misses = new_misses(args.frames, args.labels, args.baseline, args.candidate, parse_recall_targets(args.recall_target))
    write_crops(misses, args.frames, args.out)
    print(f"newmisses: {len(misses)} reference boxes newly missed; crops in {args.out}")
    return 0 if not misses else 1
```

- [ ] **Step 5: 运行相关测试**

Run: `cd py && ../.venv/bin/python -m pytest tests/test_eval_falsepos.py tests/test_eval_run.py tests/test_eval_viewing.py -q`
Expected: 全部 passed。`test_eval_run.py` 里原有的测试若检查 `summary.json` 的完整键集合，就把 `verifier_model` 和 `temporal_bins` 补进期望值。

- [ ] **Step 6: 提交**

```bash
git add py/vc_eval/falsepos.py py/vc_eval/newmisses.py py/vc_eval/temporal.py py/vc_eval/run.py py/vc_eval/cli.py py/tests/test_eval_falsepos.py py/tests/test_eval_run.py
git commit -m "feat: vc_eval counts unexplained masks, temporal coverage by score band, new misses; run takes a verifier" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: 验收、报告与文档（操作步骤 + 文档）

**Files:**
- Create: `docs/reports/fp-filter-<日期>.md`
- Modify: `cpp/README.md`, `py/README.md`, `docs/USAGE.md`

- [ ] **Step 1: 告知并跑基线与复核两组**

告诉用户：两组各跑 3 个留出段，含逐帧参照，每组约 10–15 分钟，放后台。

```bash
cmake --build build/cpp-release --target vc_redact vc_detect -j4
cd py
../.venv/bin/python -m vc_eval run --stems sg_drive_1200s,sg_drive_2400s,sg_drive_60s --model ../data/models/r3 \
  --face-model ../data/models/yunet --fps 10 --recall-target face=0.95,plate=0.98 \
  --vc-redact ../build/cpp-release/vc_redact --vc-detect ../build/cpp-release/vc_detect --config ../configs/redact.json \
  --out ../data/reports/fp-baseline --redacted-root ../data/redacted/fp-baseline --overwrite
../.venv/bin/python -m vc_eval run --stems sg_drive_1200s,sg_drive_2400s,sg_drive_60s --model ../data/models/r3 \
  --face-model ../data/models/yunet --verifier-model ../data/models/verifier-v1 --fps 10 --recall-target face=0.95,plate=0.98 \
  --vc-redact ../build/cpp-release/vc_redact --vc-detect ../build/cpp-release/vc_detect --config ../configs/redact.json \
  --out ../data/reports/fp-verifier --redacted-root ../data/redacted/fp-verifier --overwrite
```

（`vc_eval run` 在 `py/` 下运行时，相对路径的默认值以 `py/` 为基准，所以上面把路径都写全了。若 `run` 要求在仓库根目录运行，就去掉 `../`，在根目录执行 `py` 的模块：`.venv/bin/python -m vc_eval ...`，并设 `PYTHONPATH=py`。）

- [ ] **Step 2: 误打计数**

```bash
for s in sg_drive_1200s sg_drive_2400s sg_drive_60s; do for v in fp-baseline fp-verifier; do
  echo "== $s $v"; PYTHONPATH=py .venv/bin/python -m vc_eval falsepos --labels data/labels/$s \
    --detections data/redacted/$v/fps10/$s/detections.jsonl --json data/reports/$v/falsepos_$s.json
done; done
```

（`fps10` 这一层目录名以 `vc_eval run` 实际的输出布局为准，`ls data/redacted/fp-baseline` 看一下。）

- [ ] **Step 3: 新漏核查**

```bash
for s in sg_drive_1200s sg_drive_2400s sg_drive_60s; do
  PYTHONPATH=py .venv/bin/python -m vc_eval newmisses --frames data/frames/$s --labels data/labels/$s \
    --baseline data/redacted/fp-baseline/fps10/$s/masks.jsonl --candidate data/redacted/fp-verifier/fps10/$s/masks.jsonl \
    --out data/reports/fp-verifier/newmisses/$s --recall-target face=0.95,plate=0.98
done
```

逐张看 `newmisses/<段>/` 的裁图（用 Read 看图），记下每张是真目标还是老师误检。**只要有一个真车牌或真人脸，验收就不过**：报告给用户，并提出调高该类放行门槛（`verify_threshold_*`）后重跑 Step 1–3 的方案，等用户确认。

- [ ] **Step 4: 判定其余各项**

从两份 `summary.json` 和 Step 2 的 JSON 读数，逐项核对 Global Constraints：
- 召回：`pooled[0].coverage` 的人脸、车牌。
- 逐帧分段：每段的 `temporal_bins` 与基线逐段比较，各段差 ≤ 1 个百分点。
- 误打：两组 `unreferenced` 合计的降幅，人脸 ≥ 50%，车牌 ≥ 30%。
- 观感：`viewing` 的屏上面积 p99 ≤ 10%，跳变 ≤ 5%。
- 速度：两组 `sg_drive_2400s` 的 `stats.json` 中 `frames / elapsed_s`，慢不超过 10%。

另外抽 40 张被拦下的框（`detections.jsonl` 里 `"verified": false` 的）做拼图，参照 `data/reports/false-positives/sheets.py`，并亲眼看一遍。

- [ ] **Step 5: 写报告**

`docs/reports/fp-filter-<日期>.md`，格式参照 `docs/reports/stable-masks-2026-10-19.md`：
- 结论；
- 100 张试判的混淆矩阵与选定门槛；
- 判定数量（每类 yes / no / unsure，人工几张）；
- 复核模型的验证指标与门槛；
- 验收表（目标 / 实测 / 是否通过）；
- 新漏核查结果；
- 拦截与残留误打的拼图路径；
- 已知问题。

- [ ] **Step 6: 更新文档**

- `cpp/README.md`：`vc_redact` 参数表加 `--verifier-model`、`--dump-verify-crops`、`--dump-verify-every`；配置表加 4 个 `verify_*` 键；`detections.jsonl` 字段说明加 `verify`、`verified`；`stats.json` 说明加 `verifier`、`verify`、`stages.verify`。
- `py/README.md`：新增 `vc_hardneg` 一节（`mine` → `pilot` → `pilot-report` → `judge` → `review` / `import-human` 的流程、`configs/hardneg.json`、LLM 缺省拒跑与 `VC_HARDNEG_ALLOW_LLM`）；`vc_train verifier`；`vc_eval falsepos`、`newmisses`、`run --verifier-model`。
- `docs/USAGE.md`：`scripts/redact.sh` 的 `VC_VERIFIER`，以及"不想复核就设 `VC_VERIFIER=`"。

- [ ] **Step 7: 提交**

```bash
git add docs/reports/fp-filter-*.md cpp/README.md py/README.md docs/USAGE.md
git commit -m "docs: verifier acceptance report and usage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: 交给用户决定**

把验收表和新漏核查结果报给用户。是否把 `feat/fp-filter` 合并到 main，由用户决定。合并后按全局约定，把本地 checkout 更新到最新。
