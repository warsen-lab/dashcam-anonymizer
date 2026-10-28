# 按轨迹打码（子项目 2）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vc_redact 多看 W 个关键帧再输出，把已确认目标连续漏检的空档连上，并在它第一次检出之前往回补几个间隔，让 `sg_drive_60s` 上确认的 3 处车牌泄露消失。

**Architecture:** 新增 `KeyframeLinker`（跨空档配对、确认、连上、往回补，晚 W 个关键帧吐出带轨迹号的关键帧）。`BoxTimeline` 增加"按轨迹号连框"模式（`TimelineConfig::linked`），插值、保持、平滑都不动。`TrackMasker` 把两者包起来，接口和 `BoxTimeline` 一样；W = 0 时直接用 `BoxTimeline`，行为与现在逐字节一致。`redact_job` 的帧缓存改成队列，按 `TrackMasker` 吐出的掩码数从队头发帧。

**Tech Stack:** C++20、GoogleTest（`build/cpp/tests/vc_tests`）、nlohmann/json；验收用 Python（`py/vc_eval`、`py/vc_gt`，`.venv`）。

## Global Constraints

- 设计文档：`docs/superpowers/specs/2026-10-27-track-masking-design.md`（本计划对它的三处修正见 Task 8）。
- 新配置键 `track_lookahead_keyframes` 默认 5，整数 0–30；`track_backfill_keyframes` 默认 2，整数 0 到 max(0, W − 1)。W = 0 = 现在的行为。
- `masks.jsonl` 的 `origin` 新增 `bridged`、`backfilled`；`detected`、`interpolated`、`held` 不变。
- 补出来的框不写进 `detections.jsonl`、不参与确认、不更新速度。
- **本机禁止全量测试**：只跑 `--gtest_filter` 指定的套件，不跑 `ctest`、不跑不带过滤的 `vc_tests`。
- **计费流量**：调用多模态模型（问题 B）之前必须停下来，把要发的组数报给用户，用户同意后才能带 `VC_HARDNEG_ALLOW_LLM=1` 运行。
- 杀进程只能按 PID 或 `pkill -x`；不要用 `git stash`。
- 每个任务一个提交，提交信息末尾带 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- 构建：测试用 `cmake --build build/cpp -j 4 --target vc_tests`；运行用 `cmake --build build/cpp-release -j 4 --target vc_redact`。新增源文件后 CMake 会自动重新配置。

## File Structure

| 文件 | 职责 |
|---|---|
| `cpp/src/track/box_geometry.h`（新） | 中心、宽高、平移、插值等框运算（从 `box_timeline.cpp` 搬出，三个单元共用） |
| `cpp/src/track/box_matching.{h,cpp}`（新） | 两轮贪心配对 `matchBoxes` 和 `needsConfirmation`（从 `BoxTimeline::match` 原样搬出） |
| `cpp/src/track/box_timeline.{h,cpp}` | 加 `linked` 模式：按 `BoxLink` 连框，新 `origin` |
| `cpp/src/track/keyframe_linker.{h,cpp}`（新） | 窗口内配对、确认、连上、往回补 |
| `cpp/src/track/track_masker.{h,cpp}`（新） | 把 linker 和 timeline 串起来，处理延后输出 |
| `cpp/apps/vc_redact/redact_config.{h,cpp}` | 两个新键、`trackMaskerFor` |
| `cpp/apps/vc_redact/redact_job.{h,cpp}` | 队列缓存、按关键帧写 detections、统计、内存上限 |
| `configs/redact.json`、`docs/USAGE.md`、`cpp/README.md` | 新键与行为说明 |
| `cpp/tests/box_matching_test.cpp`、`keyframe_linker_test.cpp`、`track_masker_test.cpp`（新） | 单元测试 |

---

### Task 0: 改动前的基线输出

**Files:** 无代码改动；输出在 `data/redacted/track-base/`（不进仓库）。

改动前用当前代码重新跑一遍 `sg_drive_60s`，作为 Task 7 逐行比对的基准。CoreML 的结果可能随编译变化，所以不直接拿旧的 `demo-r3-pw16` 比。

- [ ] **Step 1: 编译当前代码的 vc_redact**

Run: `cmake --build build/cpp-release -j 4 --target vc_redact`
Expected: 以 `[100%] Built target vc_redact` 结束。

- [ ] **Step 2: 跑基线（断网沙箱）**

Run:
```bash
VC_VERIFIER= VC_REDACT_ARGS=--overwrite scripts/redact.sh data/redacted/track-base data/videos/sg_drive_60s.mp4
```
Expected: 退出码 0，生成 `data/redacted/track-base/sg_drive_60s/masks.jsonl`（7194 行）。

- [ ] **Step 3: 确认基线可复现**

再跑一遍到 `data/redacted/track-base2`，比较两次的 `masks.jsonl`：
```bash
VC_VERIFIER= VC_REDACT_ARGS=--overwrite scripts/redact.sh data/redacted/track-base2 data/videos/sg_drive_60s.mp4
cmp data/redacted/track-base/sg_drive_60s/masks.jsonl data/redacted/track-base2/sg_drive_60s/masks.jsonl && echo identical
```
Expected: `identical`。如果不一致，说明检测本身不可复现，Task 7 的逐行比对要改成"差异不超过两次基线之间的差异"，先停下来告诉用户。

---

### Task 1: 把配对搬成独立单元（纯重构）

**Files:**
- Create: `cpp/src/track/box_geometry.h`
- Create: `cpp/src/track/box_matching.h`
- Create: `cpp/src/track/box_matching.cpp`
- Modify: `cpp/src/track/box_timeline.h`
- Modify: `cpp/src/track/box_timeline.cpp:10-57,59-62,142-209,233-236`
- Modify: `cpp/CMakeLists.txt`（`vc_core` 的源文件列表）
- Modify: `cpp/tests/CMakeLists.txt`
- Modify: `cpp/README.md:60`（测试过滤组）
- Test: `cpp/tests/box_matching_test.cpp`

**Interfaces:**
- Produces:
  - `struct Velocity { double x; double y; }`
  - `struct MatchTrack { Box box; Box anchor; std::optional<Velocity> velocity; }`
  - `struct MatchParams { double maxDistance = 1.0; double maxDistanceLoose = 6.0; std::vector<float> confirmScore; }`
  - `struct MatchPair { size_t track; size_t next; }`
  - `std::vector<MatchPair> matchBoxes(const std::vector<MatchTrack>&, const std::vector<Box>& next, double steps, const MatchParams&)`
  - `bool needsConfirmation(const Box&, const std::vector<float>& confirmScore)`
  - `box_geometry.h`：`Point`、`centerOf`、`widthOf`、`heightOf`、`toFloat`、`shifted`、`lerp`（全部 `inline`，命名空间 `vc`）

- [ ] **Step 1: 写失败的测试**

`cpp/tests/box_matching_test.cpp`:
```cpp
#include "track/box_matching.h"

#include <gtest/gtest.h>

#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

MatchParams params() { return MatchParams{1.0, 6.0, {0.5f}}; }

TEST(BoxMatching, TakesTheCheapestPairsFirstOneToOne) {
  const std::vector<MatchTrack> tracks{{face(100, 180), face(100, 180), std::nullopt},
                                       {face(130, 210), face(130, 210), std::nullopt}};
  // box 1 is 5 px from track 0, box 0 6 px from track 1: the cheaper pair is taken first
  const std::vector<MatchPair> pairs = matchBoxes(tracks, {face(136, 216), face(95, 175)}, 6, params());
  ASSERT_EQ(pairs.size(), 2u);
  EXPECT_EQ(pairs[0].track, 0u);
  EXPECT_EQ(pairs[0].next, 1u);
  EXPECT_EQ(pairs[1].track, 1u);
  EXPECT_EQ(pairs[1].next, 0u);
}

TEST(BoxMatching, FollowsTheVelocityInTheStrictPass) {
  // 120 px on in 6 frames is 1.5 mean sides: too far for the strict pass unless the track keeps its velocity of
  // 20 px per frame, and the low scores keep the pair out of the loose pass
  const std::vector<MatchTrack> moving{{face(100, 180, 0.3f), face(100, 180, 0.3f), Velocity{20, 0}}};
  const std::vector<MatchTrack> still{{face(100, 180, 0.3f), face(100, 180, 0.3f), std::nullopt}};
  EXPECT_EQ(matchBoxes(moving, {face(220, 300, 0.3f)}, 6, params()).size(), 1u);
  EXPECT_TRUE(matchBoxes(still, {face(220, 300, 0.3f)}, 6, params()).empty());
}

TEST(BoxMatching, LeavesLowScorePairsToTheStrictPass) {
  const std::vector<MatchTrack> strong{{face(100, 180), face(100, 180), std::nullopt}};
  const std::vector<MatchTrack> weakAnchor{{face(100, 180), face(100, 180, 0.3f), std::nullopt}};
  // 240 px away: 3 sides, too far for the strict pass, near enough for the loose one
  EXPECT_EQ(matchBoxes(strong, {face(340, 420)}, 6, params()).size(), 1u);
  EXPECT_TRUE(matchBoxes(weakAnchor, {face(340, 420)}, 6, params()).empty());
  EXPECT_TRUE(matchBoxes(strong, {face(340, 420, 0.3f)}, 6, params()).empty());
}

TEST(BoxMatching, NeedsConfirmationGoesByClass) {
  EXPECT_TRUE(needsConfirmation(face(0, 10, 0.3f), {0.5f}));
  EXPECT_FALSE(needsConfirmation(face(0, 10, 0.5f), {0.5f}));
  EXPECT_FALSE(needsConfirmation(Box{1, 0, 0, 10, 10, 0.01f}, {0.5f}));  // a class without an entry
}

}  // namespace
}  // namespace vc
```

把 `box_matching_test.cpp` 加进 `cpp/tests/CMakeLists.txt` 的 `add_executable(vc_tests ...)` 列表（紧跟 `box_timeline_test.cpp`）。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`'track/box_matching.h' file not found`。

- [ ] **Step 3: 写 `box_geometry.h`**

```cpp
#pragma once

#include <algorithm>
#include <limits>

#include "detect/box.h"

namespace vc {

// Box arithmetic shared by the track code. Sizes are computed in double: the float difference of two finite
// coordinates can overflow to inf, and inf later turns into a NaN margin (0 * inf) that would drop a mask silently.

struct Point {
  double x = 0;
  double y = 0;
};

inline Point centerOf(const Box& b) {
  return {(static_cast<double>(b.x1) + b.x2) / 2.0, (static_cast<double>(b.y1) + b.y2) / 2.0};
}

inline double widthOf(const Box& b) { return b.x2 > b.x1 ? static_cast<double>(b.x2) - b.x1 : 0.0; }
inline double heightOf(const Box& b) { return b.y2 > b.y1 ? static_cast<double>(b.y2) - b.y1 : 0.0; }

// The nearest float, saturating instead of overflowing to inf.
inline float toFloat(double v) {
  constexpr double kMax = std::numeric_limits<float>::max();
  return static_cast<float>(std::clamp(v, -kMax, kMax));
}

inline Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = toFloat(b.x1 + dx);
  s.x2 = toFloat(b.x2 + dx);
  s.y1 = toFloat(b.y1 + dy);
  s.y2 = toFloat(b.y2 + dy);
  return s;
}

// Each coordinate a fraction t of its way from a's to b's; the score is the higher of the two.
inline Box lerp(const Box& a, const Box& b, double t) {
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

}  // namespace vc
```

- [ ] **Step 4: 写 `box_matching.h` / `box_matching.cpp`**

`box_matching.h`:
```cpp
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "detect/box.h"

namespace vc {

// A velocity in px per frame index.
struct Velocity {
  double x = 0;
  double y = 0;
};

// What matching needs of a track: where it is on the previous keyframe, its last real detection (whose score
// decides whether the loose pass may take it, spec 2026-10-19 §3.8) and its velocity, when it has one.
struct MatchTrack {
  Box box{};
  Box anchor{};
  std::optional<Velocity> velocity{};
};

struct MatchParams {
  double maxDistance = 1.0;       // first pass: max |predicted centre - centre| / mean box side
  double maxDistanceLoose = 6.0;  // second pass: max |dx| and |dy| / the largest side of the two boxes
  // Per class index: boxes scoring below their class's entry need confirming; a class without an entry does not.
  std::vector<float> confirmScore{};
};

struct MatchPair {
  size_t track = 0;  // index into the tracks
  size_t next = 0;   // index into the next keyframe's boxes
};

// Matches tracks to the boxes of the next keyframe, `steps` frames after the previous one, in two greedy,
// one-to-one passes over same-class pairs, cheapest first; both try the target staying put and keeping its
// velocity. The first pass takes pairs whose centres are at most maxDistance mean box sides apart. The second
// pass, over what is left, takes pairs of similar size (each side within a factor of 3 of the other's) whose
// centres are at most maxDistanceLoose times the largest side of the two boxes apart along each axis, and only if
// neither the track's anchor nor the box is below its class's confirmScore. Pairs come in the order they were taken.
std::vector<MatchPair> matchBoxes(const std::vector<MatchTrack>& tracks, const std::vector<Box>& next, double steps,
                                  const MatchParams& params);

// Whether `box` scores below its class's entry of confirmScore (a class without an entry needs no confirmation).
bool needsConfirmation(const Box& box, const std::vector<float>& confirmScore);

}  // namespace vc
```

`box_matching.cpp`（函数体就是 `BoxTimeline::match` 原样，只把 `tracks_[i]` 换成 `tracks[i]`、`config_` 换成 `params`）:
```cpp
#include "track/box_matching.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "track/box_geometry.h"

namespace vc {
namespace {

// The second matching pass pairs boxes only when each side is within this factor of the other's.
constexpr double kMaxLooseSizeRatio = 3.0;

bool similarSizes(const Box& a, const Box& b) {
  const auto within = [](double p, double q) { return p <= kMaxLooseSizeRatio * q && q <= kMaxLooseSizeRatio * p; };
  return within(widthOf(a), widthOf(b)) && within(heightOf(a), heightOf(b));
}

}  // namespace

bool needsConfirmation(const Box& box, const std::vector<float>& confirmScore) {
  if (box.cls < 0 || static_cast<size_t>(box.cls) >= confirmScore.size()) return false;
  return box.score < confirmScore[static_cast<size_t>(box.cls)];
}

std::vector<MatchPair> matchBoxes(const std::vector<MatchTrack>& tracks, const std::vector<Box>& next, double steps,
                                  const MatchParams& params) {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  // The cheaper of "stayed put" and "kept its velocity" (a target that stops still matches).
  const auto cheapest = [&](const MatchTrack& track, const auto& costFrom) {
    const Point from = centerOf(track.box);
    double cost = costFrom(from);
    if (const auto& v = track.velocity) {
      cost = std::min(cost, costFrom(Point{from.x + v->x * steps, from.y + v->y * steps}));
    }
    return cost;
  };
  std::vector<bool> usedTrack(tracks.size(), false);
  std::vector<bool> usedNext(next.size(), false);
  std::vector<MatchPair> pairs;
  const auto takeGreedily = [&](std::vector<Candidate> candidates) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
    for (const Candidate& c : candidates) {
      if (usedTrack[c.i] || usedNext[c.j]) continue;
      usedTrack[c.i] = true;
      usedNext[c.j] = true;
      pairs.push_back(MatchPair{c.i, c.j});
    }
  };

  // First pass: centre distance in mean box sides.
  std::vector<Candidate> strict;
  for (size_t i = 0; i < tracks.size(); ++i) {
    const Box& a = tracks[i].box;
    for (size_t j = 0; j < next.size(); ++j) {
      const Box& b = next[j];
      if (a.cls != b.cls) continue;
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (widthOf(a) + heightOf(a) + widthOf(b) + heightOf(b)) / 4.0);
      const double cost =
          cheapest(tracks[i], [&](Point p) { return std::hypot(actual.x - p.x, actual.y - p.y) / size; });
      if (cost <= params.maxDistance) strict.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(strict));

  // Second pass, over what is left: both axes in the largest side of the two boxes, similar sizes only. A pair
  // with either side below its class's confirm score is left to the first pass (spec 2026-10-19 §3.8): two
  // far-apart low-score boxes are more likely two false positives than one target, and joining them sweeps a mask
  // between them. A track's side is its anchor, its last real detection.
  std::vector<Candidate> loose;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (usedTrack[i]) continue;
    const Box& a = tracks[i].box;
    for (size_t j = 0; j < next.size(); ++j) {
      const Box& b = next[j];
      if (usedNext[j] || a.cls != b.cls || !similarSizes(a, b)) continue;
      if (needsConfirmation(tracks[i].anchor, params.confirmScore) || needsConfirmation(b, params.confirmScore)) {
        continue;
      }
      const Point actual = centerOf(b);
      const double side = std::max({1.0, widthOf(a), heightOf(a), widthOf(b), heightOf(b)});
      const double cost = cheapest(tracks[i], [&](Point p) {
        return std::max(std::abs(actual.x - p.x), std::abs(actual.y - p.y)) / side;
      });
      if (cost <= params.maxDistanceLoose) loose.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(loose));
  return pairs;
}

}  // namespace vc
```

- [ ] **Step 5: `BoxTimeline` 改用它们**

`box_timeline.h`：
- 在 `#include "detect/box.h"` 后加 `#include "track/box_matching.h"`。
- 删除私有的 `struct Match { size_t track; size_t next; };`。
- `std::vector<Match> match(const Keyframe& next, double steps) const;` 改为 `std::vector<MatchPair> match(const Keyframe& next, double steps) const;`。
- 类注释里"Matching runs in two greedy ... in the first pass only." 那一段改成一句：`// Boxes are matched to the tracks with matchBoxes (box_matching.h); a ghost's anchor is its last real detection.`

`box_timeline.cpp`：
- `#include "track/box_geometry.h"`。
- 匿名命名空间里删掉 `kMaxLooseSizeRatio`、`Point`、`centerOf`、`widthOf`、`heightOf`、`toFloat`、`shifted`、`lerp`、`similarSizes`（`growTo`、`fractions`、`requireIncreasing`、`requireFinite`、`validConfig`、`kMaxHeldShift` 保留）。
- 整个 `BoxTimeline::match` 换成：
```cpp
std::vector<MatchPair> BoxTimeline::match(const Keyframe& next, double steps) const {
  std::vector<MatchTrack> tracks;
  tracks.reserve(tracks_.size());
  for (const Track& t : tracks_) {
    std::optional<Velocity> velocity;
    if (t.velocity) velocity = Velocity{t.velocity->x, t.velocity->y};
    tracks.push_back(MatchTrack{t.box, t.anchor, velocity});
  }
  return matchBoxes(tracks, next.boxes, steps,
                    MatchParams{config_.matchMaxDistance, config_.matchMaxDistanceLoose, config_.confirmScore});
}
```
- `BoxTimeline::needsConfirmation` 的函数体换成 `return ::vc::needsConfirmation(box, config_.confirmScore);`。
- `advance()` 里 `for (const Match& m : match(next, steps))` 改为 `for (const MatchPair& m : match(next, steps))`。

`cpp/CMakeLists.txt`：`vc_core` 里 `src/track/box_timeline.cpp` 前加一行 `src/track/box_matching.cpp`。

`cpp/README.md:60` 的过滤组改为：
```
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:BoxMatching.*:Mosaic.*:MosaicGrid.*:BoundedQueue.*:RedactConfig.*:Records.*'
```

- [ ] **Step 6: 运行测试，确认新旧都通过**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='BoxMatching.*:BoxTimeline.*'`
Expected: `[  PASSED  ]`，BoxMatching 4 个、BoxTimeline 69 个全部通过（BoxTimeline 的测试一行没改）。

- [ ] **Step 7: 提交**

```bash
git add cpp/src/track/box_geometry.h cpp/src/track/box_matching.h cpp/src/track/box_matching.cpp cpp/src/track/box_timeline.h cpp/src/track/box_timeline.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/box_matching_test.cpp cpp/README.md
git commit -m "refactor: move keyframe box matching out of BoxTimeline into box_matching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `BoxTimeline` 按轨迹号连框

**Files:**
- Modify: `cpp/src/track/box_timeline.h`
- Modify: `cpp/src/track/box_timeline.cpp`
- Test: `cpp/tests/box_timeline_test.cpp`（文件末尾追加）

**Interfaces:**
- Consumes: Task 1 的 `MatchPair`、`matchBoxes`。
- Produces:
  - `enum class MaskOrigin { Detected, Interpolated, Held, Bridged, Backfilled }`；`toString` 返回 `"bridged"`、`"backfilled"`
  - `enum class LinkKind { Real, Bridged, Backfilled }`
  - `struct BoxLink { int64_t track = 0; LinkKind kind = LinkKind::Real; bool masked = true; }`
  - `Keyframe` 加 `std::vector<BoxLink> links{}`
  - `TimelineConfig` 加 `bool linked = false`

- [ ] **Step 1: 写失败的测试**

追加到 `cpp/tests/box_timeline_test.cpp` 末尾（`}  // namespace` 之前）：
```cpp
// ---- Linked keyframes (spec 2026-10-27 §3.2): KeyframeLinker's track ids and kinds instead of matching.

TimelineConfig linkedExact() {
  TimelineConfig c = exact();
  c.linked = true;
  return c;
}

Keyframe linkedKey(int64_t index, std::vector<Box> boxes, std::vector<BoxLink> links) {
  Keyframe k = key(index, std::move(boxes));
  k.links = std::move(links);
  return k;
}

TEST(BoxTimeline, LinkedTracksFollowTheirIdsWhereverTheBoxesAre) {
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80), face(1000, 0, 1080, 80)}, {{1}, {2}}));
  // the ids cross: track 1 goes to the right, track 2 to the left; half way both are in the middle
  const TimelineOutput out =
      t.advance(frames(0, 6), linkedKey(6, {face(0, 0, 80, 80), face(1000, 0, 1080, 80)}, {{2}, {1}}));
  ASSERT_EQ(out.masks[3].size(), 2u);
  for (const MaskBox& m : out.masks[3]) expectBox(m, 500, 0, 580, 80, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FramesNextToABridgedBoxAreBridged) {
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1}}));
  const TimelineOutput a = t.advance(frames(0, 6), linkedKey(6, {face(60, 0, 140, 80)}, {{1, LinkKind::Bridged}}));
  const TimelineOutput b = t.advance(frames(6, 12), linkedKey(12, {face(120, 0, 200, 80)}, {{1}}));
  expectBox(a.masks[0][0], 0, 0, 80, 80, MaskOrigin::Detected);
  expectBox(a.masks[3][0], 30, 0, 110, 80, MaskOrigin::Bridged);
  expectBox(b.masks[0][0], 60, 0, 140, 80, MaskOrigin::Bridged);
  expectBox(b.masks[5][0], 110, 0, 190, 80, MaskOrigin::Bridged);
}

TEST(BoxTimeline, FramesFromABackfilledBoxAreBackfilled) {
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1, LinkKind::Backfilled}}));
  const TimelineOutput out = t.advance(frames(0, 6), linkedKey(6, {face(60, 0, 140, 80)}, {{1}}));
  expectBox(out.masks[0][0], 0, 0, 80, 80, MaskOrigin::Backfilled);
  expectBox(out.masks[5][0], 50, 0, 130, 80, MaskOrigin::Backfilled);
}

TEST(BoxTimeline, AnUnmaskedLinkIsNotMaskedEvenWhenItsTrackGoesOn) {
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1, LinkKind::Real, false}}));
  const TimelineOutput out =
      t.advance(frames(0, 6), linkedKey(6, {face(0, 0, 80, 80)}, {{1, LinkKind::Real, false}}));
  for (const auto& m : out.masks) EXPECT_TRUE(m.empty());
  EXPECT_EQ(out.confirmed, std::vector<bool>{false});
}

TEST(BoxTimeline, ALinkedTrackThatEndsIsHeldForOneIntervalAsBefore) {
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1}}));
  const TimelineOutput held = t.advance(frames(0, 6), linkedKey(6, {}, {}));
  const TimelineOutput after = t.advance(frames(6, 12), linkedKey(12, {}, {}));
  expectBox(held.masks[0][0], 0, 0, 80, 80, MaskOrigin::Detected);
  expectBox(held.masks[5][0], 0, 0, 80, 80, MaskOrigin::Held);
  for (const auto& m : after.masks) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, OnlyRealBoxesSetALinkedTracksVelocity) {
  // the track starts on a backfilled box: its first real box takes no velocity from it, so the hold that follows
  // stays where the real box was instead of moving on
  BoxTimeline t(linkedExact(), 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1, LinkKind::Backfilled}}));
  t.advance(frames(0, 6), linkedKey(6, {face(60, 0, 140, 80)}, {{1}}));
  const TimelineOutput held = t.advance(frames(6, 12), linkedKey(12, {}, {}));
  expectBox(held.masks[5][0], 60, 0, 140, 80, MaskOrigin::Held);
}

TEST(BoxTimeline, ALinkedKeyframeNeedsOneUniqueLinkPerBox) {
  BoxTimeline linked(linkedExact(), 1920, 1080);
  EXPECT_THROW(linked.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {})), std::invalid_argument);
  EXPECT_THROW(linked.advance({}, linkedKey(0, {face(0, 0, 80, 80), face(200, 0, 280, 80)}, {{1}, {1}})),
               std::invalid_argument);
  BoxTimeline plain(exact(), 1920, 1080);
  EXPECT_THROW(plain.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1}})), std::invalid_argument);
}

TEST(BoxTimeline, TheNewOriginsHaveTheirNames) {
  EXPECT_STREQ(toString(MaskOrigin::Bridged), "bridged");
  EXPECT_STREQ(toString(MaskOrigin::Backfilled), "backfilled");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`no member named 'linked' in 'vc::TimelineConfig'` / `unknown type name 'BoxLink'`。

- [ ] **Step 3: 改头文件**

`box_timeline.h`：
```cpp
enum class MaskOrigin { Detected, Interpolated, Held, Bridged, Backfilled };

// "detected" | "interpolated" | "held" | "bridged" | "backfilled" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

// What a box on a linked keyframe is (KeyframeLinker, spec 2026-10-27 §2): a detection, or filled in where its
// track was missed (bridged) or before it was first detected (backfilled).
enum class LinkKind { Real, Bridged, Backfilled };

// A box's place in KeyframeLinker's tracks.
struct BoxLink {
  int64_t track = 0;  // the linker's track id, from 1
  LinkKind kind = LinkKind::Real;
  bool masked = true;  // the linker's confirmation: false for a low-score box it did not confirm
};
```
`Keyframe` 改为：
```cpp
struct Keyframe {
  FrameTime time;
  std::vector<Box> boxes;
  // With TimelineConfig::linked, one per box (unique track ids); otherwise empty.
  std::vector<BoxLink> links{};
};
```
`TimelineConfig` 末尾加：
```cpp
  // Keyframes come from KeyframeLinker (spec 2026-10-27 §3.2): a track follows its BoxLink::track instead of being
  // matched, and is masked as BoxLink::masked says instead of by confirmScore.
  bool linked = false;
```
私有 `Track` 末尾加三个字段：
```cpp
    int64_t link = 0;                // linked: KeyframeLinker's track id
    LinkKind kind = LinkKind::Real;  // linked: what `box` is
    bool anchorReal = true;          // `anchor` is a real detection (false while the track has only filled-in boxes)
```
私有方法加：
```cpp
  // Linked: pairs each track with the box of `next` that has its link id.
  std::vector<MatchPair> matchLinks(const Keyframe& next) const;
  // The track box j of `next` starts (a new id).
  Track startTrack(const Keyframe& next, size_t j);
  // `old` continued by box j of `next`, `steps` frames later: a real box becomes the anchor and gives the velocity
  // from the last real one; a filled-in box changes neither.
  Track continueTrack(const Track& old, const Keyframe& next, size_t j, double steps) const;
```
类注释末尾加一段：
```cpp
// With config.linked the keyframes come from KeyframeLinker: tracks follow the boxes' link ids instead of matching,
// a box is masked as its link says (no confirmScore), and frames between a bridged or backfilled box and its
// neighbours get that origin (backfilled before bridged before interpolated). Filled-in boxes never set a velocity.
```

- [ ] **Step 4: 改实现**

`box_timeline.cpp`：

`#include <unordered_set>`。匿名命名空间加：
```cpp
// Linked keyframes carry one link per box, with unique ids; unlinked ones none (spec 2026-10-27 §3.2).
void requireLinks(const Keyframe& key, bool linked) {
  if (!linked) {
    if (!key.links.empty()) throw std::invalid_argument("BoxTimeline: links on a keyframe of an unlinked timeline");
    return;
  }
  if (key.links.size() != key.boxes.size()) {
    throw std::invalid_argument("BoxTimeline: a linked keyframe needs one link per box");
  }
  std::unordered_set<int64_t> ids;
  for (const BoxLink& l : key.links) {
    if (!ids.insert(l.track).second) throw std::invalid_argument("BoxTimeline: two boxes of a keyframe share a link");
  }
}

// The origin of a mask on a keyframe box of this kind; `real` for a detection.
MaskOrigin originOf(LinkKind kind, MaskOrigin real) {
  switch (kind) {
    case LinkKind::Bridged:
      return MaskOrigin::Bridged;
    case LinkKind::Backfilled:
      return MaskOrigin::Backfilled;
    case LinkKind::Real:
      break;
  }
  return real;
}

// Between two keyframe boxes: backfilled if either end is, else bridged if either end is, else interpolated.
MaskOrigin betweenOrigin(LinkKind from, LinkKind to) {
  if (from == LinkKind::Backfilled || to == LinkKind::Backfilled) return MaskOrigin::Backfilled;
  if (from == LinkKind::Bridged || to == LinkKind::Bridged) return MaskOrigin::Bridged;
  return MaskOrigin::Interpolated;
}
```
`toString` 的 switch 加：
```cpp
    case MaskOrigin::Bridged:
      return "bridged";
    case MaskOrigin::Backfilled:
      return "backfilled";
```
`match()` 开头加 `if (config_.linked) return matchLinks(next);`，并新增：
```cpp
std::vector<MatchPair> BoxTimeline::matchLinks(const Keyframe& next) const {
  std::vector<MatchPair> pairs;
  for (size_t j = 0; j < next.boxes.size(); ++j) {
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (tracks_[i].link == next.links[j].track) {
        pairs.push_back(MatchPair{i, j});
        break;
      }
    }
  }
  return pairs;
}

BoxTimeline::Track BoxTimeline::startTrack(const Keyframe& next, size_t j) {
  const Box& box = next.boxes[j];
  Track t{nextId_++, box, std::nullopt, false, box, 0.0, static_cast<int>(j), !needsConfirmation(box)};
  if (config_.linked) {
    const BoxLink& link = next.links[j];
    t.confirmed = link.masked;
    t.link = link.track;
    t.kind = link.kind;
    t.anchorReal = link.kind == LinkKind::Real;
  }
  return t;
}

BoxTimeline::Track BoxTimeline::continueTrack(const Track& old, const Keyframe& next, size_t j, double steps) const {
  const Box& box = next.boxes[j];
  Track t{old.id, box, old.velocity, false, old.anchor, old.anchorSteps + steps, static_cast<int>(j), true};
  t.link = old.link;
  t.anchorReal = old.anchorReal;
  if (config_.linked) {
    t.confirmed = next.links[j].masked;
    t.kind = next.links[j].kind;
  }
  if (t.kind == LinkKind::Real) {
    if (old.anchorReal) {
      const Point a = centerOf(old.anchor);
      const Point b = centerOf(box);
      const double span = old.anchorSteps + steps;
      t.velocity = Offset{(b.x - a.x) / span, (b.y - a.y) / span};
    }
    t.anchor = box;
    t.anchorSteps = 0.0;
    t.anchorReal = true;
  }
  return t;
}
```
`advance()`：
1. `requireFinite(next);` 之后加 `requireLinks(next, config_.linked);`。
2. 第一次调用建轨迹的循环改为 `for (size_t j = 0; j < next.boxes.size(); ++j) tracks_.push_back(startTrack(next, j));`。
3. 打码循环换成：
```cpp
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    // Unlinked, a match confirms a track: two consecutive keyframes, or a ghost (always confirmed) found again.
    // Linked, the linker has decided already (BoxLink::masked).
    const bool masked = track.confirmed || (!config_.linked && matchedTo[i].has_value());
    if (track.keyBox >= 0) out.confirmed[static_cast<size_t>(track.keyBox)] = masked;
    if (matchedTo[i]) {
      if (!masked) continue;
      const Box& to = next.boxes[*matchedTo[i]];
      const LinkKind toKind = config_.linked ? next.links[*matchedTo[i]].kind : LinkKind::Real;
      const Point a = centerOf(track.box);
      const Point b = centerOf(to);
      const Offset motion{std::abs(b.x - a.x), std::abs(b.y - a.y)};
      for (size_t k = 0; k < frames.size(); ++k) {
        const MaskOrigin origin = k > 0       ? betweenOrigin(track.kind, toKind)
                                  : track.ghost ? MaskOrigin::Held
                                                : originOf(track.kind, MaskOrigin::Detected);
        emit(out.masks[k], track, lerp(track.box, to, t[k]), motion, origin);
      }
    } else if (!track.ghost && masked) {
      hold(out.masks, frames, prevTime, track, steps);
    } else if (track.ghost && config_.holdLostGhosts) {
      holdGhost(out.masks, frames, track);  // lost for the second keyframe in a row: held, then it ends (spec §3.7)
    }
    // Otherwise an unmatched ghost, or an unconfirmed track seen once, ends here unmasked.
  }
```
4. 建下一批轨迹的循环换成：
```cpp
  for (size_t j = 0; j < next.boxes.size(); ++j) {
    const auto from = matchedFrom[j];
    tracks.push_back(from ? continueTrack(tracks_[*from], next, j, steps) : startTrack(next, j));
  }
```
5. 幽灵那行改为先建再补字段：
```cpp
    Track ghost{track.id, shifted(track.box, s.x, s.y), track.velocity, true, track.anchor,
                track.anchorSteps + steps, -1, true};
    ghost.link = track.link;
    ghost.anchorReal = track.anchorReal;
    tracks.push_back(ghost);
```

- [ ] **Step 5: 运行测试**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='BoxMatching.*:BoxTimeline.*'`
Expected: 全部通过（BoxTimeline 69 + 8 = 77 个）。

- [ ] **Step 6: 提交**

```bash
git add cpp/src/track/box_timeline.h cpp/src/track/box_timeline.cpp cpp/tests/box_timeline_test.cpp
git commit -m "feat: BoxTimeline follows KeyframeLinker's track ids on linked keyframes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `KeyframeLinker`

**Files:**
- Create: `cpp/src/track/keyframe_linker.h`
- Create: `cpp/src/track/keyframe_linker.cpp`
- Modify: `cpp/CMakeLists.txt`、`cpp/tests/CMakeLists.txt`、`cpp/README.md:60`
- Test: `cpp/tests/keyframe_linker_test.cpp`

**Interfaces:**
- Consumes: `matchBoxes`、`needsConfirmation`、`MatchParams`、`MatchTrack`、`Velocity`（Task 1）；`Keyframe`、`BoxLink`、`LinkKind`（Task 2）；`box_geometry.h`。
- Produces:
  - `struct LinkerConfig { MatchParams match; int lookahead = 1; int backfill = 0; }`
  - `class KeyframeLinker { explicit KeyframeLinker(const LinkerConfig&); std::vector<Keyframe> push(const Keyframe&); std::vector<Keyframe> finish(); }`

- [ ] **Step 1: 写失败的测试**

`cpp/tests/keyframe_linker_test.cpp`:
```cpp
#include "track/keyframe_linker.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

// Faces scoring below 0.5 need confirming (and plates below 0.15), as in BoxTimeline's tests.
LinkerConfig linker(int lookahead, int backfill) { return LinkerConfig{MatchParams{1.0, 6.0, {0.5f, 0.15f}}, lookahead, backfill}; }

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

// Keyframe n at frame 6 n, 60 fps.
Keyframe key(int64_t n, std::vector<Box> boxes) { return Keyframe{{6 * n, 6 * n * 1000.0 / 60.0}, std::move(boxes)}; }

// Every keyframe that comes out, in order: keyframes[n] is keyframe n, then finish().
std::vector<Keyframe> run(const LinkerConfig& c, const std::vector<std::vector<Box>>& keyframes) {
  KeyframeLinker l(c);
  std::vector<Keyframe> out;
  for (size_t n = 0; n < keyframes.size(); ++n) {
    for (Keyframe& k : l.push(key(static_cast<int64_t>(n), keyframes[n]))) out.push_back(std::move(k));
  }
  for (Keyframe& k : l.finish()) out.push_back(std::move(k));
  return out;
}

TEST(KeyframeLinker, ReturnsEachKeyframeLookaheadKeyframesLater) {
  KeyframeLinker l(linker(3, 0));
  std::vector<size_t> counts;
  for (int n = 0; n < 6; ++n) counts.push_back(l.push(key(n, {face(100, 180)})).size());
  EXPECT_EQ(counts, (std::vector<size_t>{0, 0, 0, 1, 1, 1}));
  const std::vector<Keyframe> rest = l.finish();
  ASSERT_EQ(rest.size(), 3u);
  EXPECT_EQ(rest.front().time.index, 18);
}

TEST(KeyframeLinker, ATargetSeenOnEveryKeyframeIsOneMaskedTrack) {
  const std::vector<Keyframe> out = run(linker(3, 0), std::vector<std::vector<Box>>(6, {face(100, 180)}));
  ASSERT_EQ(out.size(), 6u);
  for (size_t n = 0; n < out.size(); ++n) {
    EXPECT_EQ(out[n].time.index, static_cast<int64_t>(6 * n));
    ASSERT_EQ(out[n].links.size(), 1u);
    EXPECT_EQ(out[n].links[0].track, 1);
    EXPECT_EQ(out[n].links[0].kind, LinkKind::Real);
    EXPECT_TRUE(out[n].links[0].masked);
  }
}

TEST(KeyframeLinker, BridgesUpToLookaheadMissedKeyframes) {
  // missed on keyframes 1-3 (W = 3), found again on keyframe 4, 40 px to the right
  const std::vector<Keyframe> out = run(linker(3, 0), {{face(100, 180)}, {}, {}, {}, {face(140, 220)}});
  for (int n = 1; n <= 3; ++n) {
    ASSERT_EQ(out[n].boxes.size(), 1u) << n;
    EXPECT_EQ(out[n].links[0].kind, LinkKind::Bridged);
    EXPECT_EQ(out[n].links[0].track, 1);
    EXPECT_TRUE(out[n].links[0].masked);
    EXPECT_NEAR(out[n].boxes[0].x1, 100 + 10 * n, 1e-3);
  }
  EXPECT_EQ(out[4].links[0].track, 1);
}

TEST(KeyframeLinker, DoesNotBridgeMoreThanLookahead) {
  const std::vector<Keyframe> out = run(linker(3, 0), {{face(100, 180)}, {}, {}, {}, {}, {face(140, 220)}});
  for (int n = 1; n <= 4; ++n) EXPECT_TRUE(out[n].boxes.empty()) << n;
  EXPECT_EQ(out[5].links[0].track, 2);
}

TEST(KeyframeLinker, AFoundAgainTrackIsPredictedAlongItsVelocity) {
  // low scores keep the loose pass out: only the velocity (10 px per frame) brings keyframe 4 within reach
  const float weak = 0.3f;
  const std::vector<Keyframe> out =
      run(linker(3, 0), {{face(0, 80, weak)}, {face(60, 140, weak)}, {}, {}, {face(240, 320, weak)}});
  EXPECT_EQ(out[4].links[0].track, out[0].links[0].track);
  ASSERT_EQ(out[2].boxes.size(), 1u);
  EXPECT_NEAR(out[2].boxes[0].x1, 120, 1e-3);  // frame 12, a third of the way from frame 6 to frame 24
}

TEST(KeyframeLinker, AnUnconfirmedTrackIsNeitherBridgedNorMasked) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out = run(linker(3, 2), {{face(100, 180, weak)}, {}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[1].boxes.empty());
  EXPECT_FALSE(out[0].links[0].masked);
  EXPECT_FALSE(out[2].links[0].masked);
  EXPECT_NE(out[2].links[0].track, out[0].links[0].track);
}

TEST(KeyframeLinker, ALowScoreTrackSeenOnTwoConsecutiveKeyframesIsMaskedOnBoth) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out = run(linker(3, 0), {{}, {face(100, 180, weak)}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[1].links[0].masked);
  EXPECT_TRUE(out[2].links[0].masked);
}

TEST(KeyframeLinker, BackfillsAHighScoreTrackAlongItsVelocity) {
  // first seen on keyframe 3 (frame 18), then 12 px further right on keyframe 4: 2 px per frame
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {}, {}, {face(100, 180)}, {face(112, 192)}});
  EXPECT_TRUE(out[0].boxes.empty());
  ASSERT_EQ(out[1].boxes.size(), 1u);
  ASSERT_EQ(out[2].boxes.size(), 1u);
  EXPECT_EQ(out[2].links[0].kind, LinkKind::Backfilled);
  EXPECT_EQ(out[2].links[0].track, out[3].links[0].track);
  EXPECT_NEAR(out[2].boxes[0].x1, 88, 1e-3);  // 6 frames earlier
  EXPECT_NEAR(out[1].boxes[0].x1, 76, 1e-3);  // 12 frames earlier
}

TEST(KeyframeLinker, ABackfilledBoxMovesAtMostHalfASidePerInterval) {
  // 20 px per frame: 120 px back on keyframe 2 is capped at 40 (half of 80), 240 px on keyframe 1 at 80
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {}, {}, {face(400, 480)}, {face(520, 600)}});
  EXPECT_NEAR(out[2].boxes[0].x1, 360, 1e-3);
  EXPECT_NEAR(out[1].boxes[0].x1, 320, 1e-3);
}

TEST(KeyframeLinker, BackfillsALowScoreTrackOnceConfirmed) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out =
      run(linker(3, 2), {{}, {}, {}, {face(100, 180, weak)}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[0].boxes.empty());
  ASSERT_EQ(out[1].boxes.size(), 1u);
  EXPECT_EQ(out[1].links[0].kind, LinkKind::Backfilled);
  EXPECT_NEAR(out[1].boxes[0].x1, 100, 1e-3);  // standing still
  EXPECT_TRUE(out[3].links[0].masked);
}

TEST(KeyframeLinker, DoesNotBackfillBeforeTheFirstKeyframe) {
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {face(100, 180)}});
  ASSERT_EQ(out[0].boxes.size(), 1u);
  EXPECT_EQ(out[0].links[0].kind, LinkKind::Backfilled);
}

TEST(KeyframeLinker, FinishStartsAfresh) {
  KeyframeLinker l(linker(2, 0));
  l.push(key(0, {face(100, 180)}));
  l.push(key(1, {face(500, 580)}));
  EXPECT_EQ(l.finish().size(), 2u);
  l.push(key(0, {face(100, 180)}));
  const std::vector<Keyframe> out = l.finish();
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].links[0].track, 1);
}

TEST(KeyframeLinker, RejectsBadSettingsAndKeyframes) {
  EXPECT_THROW(KeyframeLinker{linker(0, 0)}, std::invalid_argument);
  EXPECT_THROW(KeyframeLinker{linker(3, 3)}, std::invalid_argument);
  EXPECT_THROW(KeyframeLinker{linker(3, -1)}, std::invalid_argument);
  KeyframeLinker l(linker(2, 0));
  l.push(key(1, {}));
  EXPECT_THROW(l.push(key(1, {})), std::invalid_argument);
  EXPECT_THROW(l.push(key(2, {Box{kFace, NAN, 0, 1, 1, 0.9f}})), std::invalid_argument);
  EXPECT_NO_THROW(l.push(key(2, {})));
}

}  // namespace
}  // namespace vc
```
把 `keyframe_linker_test.cpp` 加进 `cpp/tests/CMakeLists.txt`，`src/track/keyframe_linker.cpp` 加进 `cpp/CMakeLists.txt` 的 `vc_core`，`cpp/README.md:60` 的过滤组加上 `:KeyframeLinker.*`。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`'track/keyframe_linker.h' file not found`。

- [ ] **Step 3: 写 `keyframe_linker.h`**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "track/box_matching.h"
#include "track/box_timeline.h"

namespace vc {

struct LinkerConfig {
  MatchParams match{};
  int lookahead = 1;  // W >= 1: keyframes come out W keyframes late; a track is found again across up to W misses
  int backfill = 0;   // B in [0, W - 1]: keyframes filled in before a masked track's first detection
};

// Links the detections on keyframes into tracks, looking W keyframes ahead (spec 2026-10-27 §2), for a BoxTimeline
// with TimelineConfig::linked.
//
// Boxes are matched to the tracks with matchBoxes, each track where it would be on the previous keyframe: its last
// detection moved along its velocity. A track is confirmed (masked) at once when its first box scores at least its
// class's confirm score, otherwise when it is matched on the very next keyframe; an unconfirmed track that misses a
// keyframe ends. A confirmed track lives on unmatched for up to W keyframes: matched again, each keyframe it missed
// gets a box interpolated by frame index between its last detection and the new one (LinkKind::Bridged). When a
// track is confirmed, up to B keyframes before its first detection get its first box (LinkKind::Backfilled, never
// before the first keyframe), moved back along its velocity once it has one (in place until then) by at most half
// its width and height per keyframe interval. Filled-in boxes never set a velocity or confirm anything; a velocity
// runs between a track's last two detections.
//
// Keyframes come out once nothing can change them any more: real boxes first, in the order given, then the
// filled-in ones, with one BoxLink per box (track ids from 1).
class KeyframeLinker {
 public:
  // Throws std::invalid_argument for lookahead < 1 or backfill outside [0, lookahead - 1].
  explicit KeyframeLinker(const LinkerConfig& config);

  // Takes the next keyframe; returns the keyframes now settled, oldest first: none for the first W calls, then
  // one per call. Throws std::invalid_argument when its index does not follow the previous keyframe's, or its
  // timestamp or a box coordinate is not finite (the linker is left untouched).
  std::vector<Keyframe> push(const Keyframe& key);

  // Every keyframe not returned yet, oldest first; resets the linker (ids start from 1 again).
  std::vector<Keyframe> finish();

 private:
  struct Track {
    int64_t id = 0;
    bool confirmed = false;
    Box first{};             // its first detection
    int64_t firstSeq = 0;    // ... on this keyframe (pushes counted from 0)
    size_t firstBox = 0;     // ... at this index of that keyframe's boxes
    int64_t firstIndex = 0;  // ... at this frame index
    Box last{};              // its last detection
    int64_t lastSeq = 0;
    int64_t lastIndex = 0;
    std::optional<Velocity> velocity{};  // px per frame, between its last two detections
    int missed = 0;                      // keyframes since lastSeq
    std::vector<std::pair<int64_t, size_t>> backfills{};  // (keyframe, box index) of its backfilled boxes
  };

  Keyframe& at(int64_t seq);
  // Fills the keyframes `track` missed with boxes on the way from its last detection to `to` at frame `toIndex`.
  void bridge(const Track& track, const Box& to, int64_t toIndex);
  // Marks `track` and its first box masked and backfills before it.
  void confirm(Track& track);
  // Moves the backfilled boxes of `track` still pending to where its velocity puts them.
  void placeBackfills(const Track& track);
  Box backfillBox(const Track& track, int64_t seq);

  LinkerConfig config_;
  std::deque<Keyframe> pending_;  // pushed, not returned yet, oldest first
  int64_t baseSeq_ = 0;           // the push number of pending_.front()
  int64_t nextSeq_ = 0;
  std::optional<FrameTime> prevTime_;  // the keyframe pushed last
  std::vector<Track> tracks_;
  int64_t nextId_ = 1;
};

}  // namespace vc
```

- [ ] **Step 4: 写 `keyframe_linker.cpp`**

```cpp
#include "track/keyframe_linker.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include "track/box_geometry.h"

namespace vc {
namespace {

// A backfilled box moves back along its track's velocity by at most this fraction of its width (x) and height (y)
// per keyframe interval, like a held one (spec 2026-10-27 §2.2).
constexpr double kMaxShiftPerInterval = 0.5;

bool finiteBox(const Box& b) {
  return std::isfinite(b.x1) && std::isfinite(b.y1) && std::isfinite(b.x2) && std::isfinite(b.y2);
}

}  // namespace

KeyframeLinker::KeyframeLinker(const LinkerConfig& config) : config_(config) {
  if (config.lookahead < 1) throw std::invalid_argument("KeyframeLinker: lookahead must be >= 1");
  if (config.backfill < 0 || config.backfill > config.lookahead - 1) {
    throw std::invalid_argument("KeyframeLinker: backfill must be in [0, lookahead - 1]");
  }
}

Keyframe& KeyframeLinker::at(int64_t seq) { return pending_.at(static_cast<size_t>(seq - baseSeq_)); }

std::vector<Keyframe> KeyframeLinker::push(const Keyframe& key) {
  if (!std::isfinite(key.time.ptsMs) || !std::all_of(key.boxes.begin(), key.boxes.end(), finiteBox)) {
    throw std::invalid_argument("KeyframeLinker: keyframe timestamp or box is not finite");
  }
  if (prevTime_ && key.time.index <= prevTime_->index) {
    throw std::invalid_argument("KeyframeLinker: keyframe indices must increase");
  }
  const int64_t seq = nextSeq_++;
  pending_.push_back(Keyframe{key.time, key.boxes, std::vector<BoxLink>(key.boxes.size())});

  // Each track where it would be on the previous keyframe: its last detection moved along its velocity.
  std::vector<MatchTrack> candidates;
  candidates.reserve(tracks_.size());
  for (const Track& t : tracks_) {
    Box box = t.last;
    if (t.velocity && prevTime_) {
      const double since = static_cast<double>(prevTime_->index - t.lastIndex);
      box = shifted(t.last, t.velocity->x * since, t.velocity->y * since);
    }
    candidates.push_back(MatchTrack{box, t.last, t.velocity});
  }
  const double steps = prevTime_ ? static_cast<double>(key.time.index - prevTime_->index) : 0.0;
  std::vector<std::optional<size_t>> matchedTo(tracks_.size());
  std::vector<bool> taken(key.boxes.size(), false);
  for (const MatchPair& m : matchBoxes(candidates, key.boxes, steps, config_.match)) {
    matchedTo[m.track] = m.next;
    taken[m.next] = true;
  }

  std::vector<Track> tracks;
  tracks.reserve(tracks_.size() + key.boxes.size());
  for (size_t i = 0; i < tracks_.size(); ++i) {
    Track t = tracks_[i];
    if (!matchedTo[i]) {
      ++t.missed;
      // a confirmed track waits up to W keyframes to be found again; an unconfirmed one ends at its first miss
      if (t.confirmed && t.missed <= config_.lookahead) tracks.push_back(std::move(t));
      continue;
    }
    const size_t j = *matchedTo[i];
    const Box& box = key.boxes[j];
    if (t.missed > 0) bridge(t, box, key.time.index);
    const Point a = centerOf(t.last);
    const Point b = centerOf(box);
    const double span = static_cast<double>(key.time.index - t.lastIndex);
    const bool firstVelocity = !t.velocity;
    t.velocity = Velocity{(b.x - a.x) / span, (b.y - a.y) / span};
    t.last = box;
    t.lastSeq = seq;
    t.lastIndex = key.time.index;
    t.missed = 0;
    pending_.back().links[j] = BoxLink{t.id, LinkKind::Real, true};
    if (!t.confirmed) {
      confirm(t);  // matched on the very next keyframe: an unconfirmed track never survives a miss
    } else if (firstVelocity) {
      placeBackfills(t);
    }
    tracks.push_back(std::move(t));
  }
  for (size_t j = 0; j < key.boxes.size(); ++j) {
    if (taken[j]) continue;
    Track t;
    t.id = nextId_++;
    t.first = key.boxes[j];
    t.firstSeq = seq;
    t.firstBox = j;
    t.firstIndex = key.time.index;
    t.last = key.boxes[j];
    t.lastSeq = seq;
    t.lastIndex = key.time.index;
    pending_.back().links[j] = BoxLink{t.id, LinkKind::Real, false};
    if (!needsConfirmation(key.boxes[j], config_.match.confirmScore)) confirm(t);
    tracks.push_back(std::move(t));
  }
  tracks_ = std::move(tracks);
  prevTime_ = key.time;

  std::vector<Keyframe> out;
  while (pending_.size() > static_cast<size_t>(config_.lookahead)) {
    out.push_back(std::move(pending_.front()));
    pending_.pop_front();
    ++baseSeq_;
  }
  return out;
}

void KeyframeLinker::bridge(const Track& track, const Box& to, int64_t toIndex) {
  // the keyframes after its last detection, up to the one just pushed (nextSeq_ - 1)
  for (int64_t s = track.lastSeq + 1; s < nextSeq_ - 1; ++s) {
    Keyframe& k = at(s);
    const double f =
        static_cast<double>(k.time.index - track.lastIndex) / static_cast<double>(toIndex - track.lastIndex);
    k.boxes.push_back(lerp(track.last, to, f));
    k.links.push_back(BoxLink{track.id, LinkKind::Bridged, true});
  }
}

void KeyframeLinker::confirm(Track& track) {
  track.confirmed = true;
  at(track.firstSeq).links.at(track.firstBox).masked = true;
  for (int b = 1; b <= config_.backfill; ++b) {
    const int64_t s = track.firstSeq - b;
    if (s < baseSeq_) break;  // returned already, or before the first keyframe
    Keyframe& k = at(s);
    k.boxes.push_back(backfillBox(track, s));
    k.links.push_back(BoxLink{track.id, LinkKind::Backfilled, true});
    track.backfills.emplace_back(s, k.boxes.size() - 1);
  }
}

void KeyframeLinker::placeBackfills(const Track& track) {
  for (const auto& [s, j] : track.backfills) {
    if (s >= baseSeq_) at(s).boxes.at(j) = backfillBox(track, s);
  }
}

Box KeyframeLinker::backfillBox(const Track& track, int64_t seq) {
  if (!track.velocity) return track.first;
  const double since = static_cast<double>(track.firstIndex - at(seq).time.index);
  const double intervals = static_cast<double>(track.firstSeq - seq);
  const double capX = kMaxShiftPerInterval * intervals * widthOf(track.first);
  const double capY = kMaxShiftPerInterval * intervals * heightOf(track.first);
  return shifted(track.first, std::clamp(-track.velocity->x * since, -capX, capX),
                 std::clamp(-track.velocity->y * since, -capY, capY));
}

std::vector<Keyframe> KeyframeLinker::finish() {
  std::vector<Keyframe> out(std::make_move_iterator(pending_.begin()), std::make_move_iterator(pending_.end()));
  pending_.clear();
  baseSeq_ = 0;
  nextSeq_ = 0;
  prevTime_.reset();
  tracks_.clear();
  nextId_ = 1;
  return out;
}

}  // namespace vc
```

- [ ] **Step 5: 运行测试**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='KeyframeLinker.*'`
Expected: 13 个全部通过。

- [ ] **Step 6: 提交**

```bash
git add cpp/src/track/keyframe_linker.h cpp/src/track/keyframe_linker.cpp cpp/tests/keyframe_linker_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/README.md
git commit -m "feat: KeyframeLinker bridges missed keyframes and backfills confirmed tracks within a lookahead

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `TrackMasker`

**Files:**
- Create: `cpp/src/track/track_masker.h`
- Create: `cpp/src/track/track_masker.cpp`
- Modify: `cpp/CMakeLists.txt`、`cpp/tests/CMakeLists.txt`、`cpp/README.md:60`
- Test: `cpp/tests/track_masker_test.cpp`

**Interfaces:**
- Consumes: `BoxTimeline`、`TimelineConfig::linked`、`Keyframe::links`（Task 2）；`KeyframeLinker`、`LinkerConfig`（Task 3）。
- Produces:
  - `struct TrackMaskerConfig { TimelineConfig timeline; int lookahead = 0; int backfill = 0; }`
  - `struct TrackingOutput { std::vector<std::vector<MaskBox>> masks; std::vector<std::vector<bool>> confirmed; }`
  - `class TrackMasker { TrackMasker(const TrackMaskerConfig&, int w, int h); TrackingOutput advance(const std::vector<FrameTime>&, const Keyframe&); TrackingOutput finish(const std::vector<FrameTime>&); }`

- [ ] **Step 1: 写失败的测试**

`cpp/tests/track_masker_test.cpp`:
```cpp
#include "track/track_masker.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

// No margins and no smoothing (each mask is its box); faces below 0.5 and plates below 0.15 need confirming.
TrackMaskerConfig masker(int lookahead, int backfill) {
  TrackMaskerConfig c;
  c.timeline.expandFraction = 0;
  c.timeline.motionExpandFraction = 0;
  c.timeline.minBoxPx = 0;
  c.timeline.shrinkPerFrame = 1.0;
  c.timeline.confirmScore = {0.5f, 0.15f};
  c.lookahead = lookahead;
  c.backfill = backfill;
  return c;
}

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

struct Run {
  std::vector<std::vector<MaskBox>> masks;
  std::vector<std::vector<bool>> confirmed;
};

FrameTime at(int64_t i) { return {i, i * 1000.0 / 60.0}; }

// keyframes[n] on frame 6 n, then `tail` frames after the last keyframe; with a TrackMasker, or with a bare
// BoxTimeline when `bare`.
Run run(const TrackMaskerConfig& c, const std::vector<std::vector<Box>>& keyframes, int tail = 6, bool bare = false) {
  TrackMasker m(c, 1920, 1080);
  BoxTimeline t(c.timeline, 1920, 1080);
  bool started = false;
  Run r;
  const auto take = [&r](std::vector<std::vector<MaskBox>> masks, std::vector<std::vector<bool>> confirmed) {
    for (auto& x : masks) r.masks.push_back(std::move(x));
    for (auto& x : confirmed) r.confirmed.push_back(std::move(x));
  };
  std::vector<FrameTime> interval;
  for (size_t n = 0; n < keyframes.size(); ++n) {
    const auto index = static_cast<int64_t>(6 * n);
    const Keyframe k{at(index), keyframes[n]};
    if (bare) {
      TimelineOutput o = t.advance(interval, k);
      take(std::move(o.masks), started ? std::vector<std::vector<bool>>{o.confirmed} : std::vector<std::vector<bool>>{});
      started = true;
    } else {
      TrackingOutput o = m.advance(interval, k);
      take(std::move(o.masks), std::move(o.confirmed));
    }
    interval.clear();
    const int length = n + 1 < keyframes.size() ? 6 : tail;
    for (int64_t i = index; i < index + length; ++i) interval.push_back(at(i));
  }
  if (bare) {
    TimelineOutput o = t.finish(interval);
    take(std::move(o.masks), {o.confirmed});
  } else {
    TrackingOutput o = m.finish(interval);
    take(std::move(o.masks), std::move(o.confirmed));
  }
  return r;
}

TEST(TrackMasker, WithoutLookaheadItIsBoxTimeline) {
  const std::vector<std::vector<Box>> script{
      {face(100, 180)}, {face(130, 210), face(900, 980, 0.3f)}, {}, {}, {face(200, 280)}, {face(600, 680, 0.3f)}, {}};
  const Run masked = run(masker(0, 0), script);
  const Run bare = run(masker(0, 0), script, 6, true);
  ASSERT_EQ(masked.masks.size(), bare.masks.size());
  for (size_t f = 0; f < bare.masks.size(); ++f) {
    ASSERT_EQ(masked.masks[f].size(), bare.masks[f].size()) << f;
    for (size_t k = 0; k < bare.masks[f].size(); ++k) {
      const MaskBox& a = masked.masks[f][k];
      const MaskBox& b = bare.masks[f][k];
      EXPECT_EQ(a.x1, b.x1);
      EXPECT_EQ(a.x2, b.x2);
      EXPECT_EQ(a.origin, b.origin);
      EXPECT_EQ(a.track, b.track);
    }
  }
  EXPECT_EQ(masked.confirmed, bare.confirmed);
}

TEST(TrackMasker, WithLookaheadEveryFrameGetsItsMasksOnceInOrder) {
  const Run r = run(masker(3, 0), std::vector<std::vector<Box>>(10, {face(100, 180)}), 4);
  ASSERT_EQ(r.masks.size(), 58u);  // 9 intervals of 6 and a tail of 4
  for (const auto& m : r.masks) EXPECT_EQ(m.size(), 1u);
  EXPECT_EQ(r.confirmed.size(), 10u);
}

TEST(TrackMasker, BridgesTheGapBoxTimelineLeaves) {
  const std::vector<std::vector<Box>> script{{face(100, 180)}, {}, {}, {face(100, 180)}};
  const Run before = run(masker(0, 0), script);
  for (int f = 6; f < 18; ++f) EXPECT_TRUE(before.masks[f].empty()) << f;
  const Run after = run(masker(3, 0), script);
  for (int f = 0; f < 24; ++f) ASSERT_EQ(after.masks[f].size(), 1u) << f;
  EXPECT_EQ(after.masks[0][0].origin, MaskOrigin::Detected);
  for (int f = 1; f < 18; ++f) EXPECT_EQ(after.masks[f][0].origin, MaskOrigin::Bridged) << f;
  EXPECT_EQ(after.masks[18][0].origin, MaskOrigin::Detected);
}

TEST(TrackMasker, BackfillsBeforeTheFirstDetection) {
  const Run r = run(masker(3, 2), {{}, {}, {}, {face(100, 180)}});
  for (int f = 0; f < 6; ++f) EXPECT_TRUE(r.masks[f].empty()) << f;
  for (int f = 6; f < 18; ++f) {
    ASSERT_EQ(r.masks[f].size(), 1u) << f;
    EXPECT_EQ(r.masks[f][0].origin, MaskOrigin::Backfilled) << f;
  }
  EXPECT_EQ(r.masks[18][0].origin, MaskOrigin::Detected);
}

TEST(TrackMasker, ConfirmedListsTheRealBoxesOfEachKeyframe) {
  const Run r = run(masker(2, 1), {{face(100, 180), face(900, 980, 0.3f)}, {face(100, 180)}, {}, {face(500, 580, 0.3f)}});
  EXPECT_EQ(r.confirmed, (std::vector<std::vector<bool>>{{true, false}, {true}, {}, {false}}));
}

TEST(TrackMasker, RejectsALookaheadOrBackfillOutOfRange) {
  EXPECT_THROW(TrackMasker(masker(0, 1), 1920, 1080), std::invalid_argument);
  EXPECT_THROW(TrackMasker(masker(3, 3), 1920, 1080), std::invalid_argument);
  EXPECT_THROW(TrackMasker(masker(-1, 0), 1920, 1080), std::invalid_argument);
}

}  // namespace
}  // namespace vc
```
把 `track_masker_test.cpp` 加进 `cpp/tests/CMakeLists.txt`，`src/track/track_masker.cpp` 加进 `cpp/CMakeLists.txt` 的 `vc_core`，`cpp/README.md:60` 的过滤组加上 `:TrackMasker.*`。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`'track/track_masker.h' file not found`。

- [ ] **Step 3: 写 `track_masker.h`**

```cpp
#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "track/box_timeline.h"
#include "track/keyframe_linker.h"

namespace vc {

struct TrackMaskerConfig {
  TimelineConfig timeline{};  // its `linked` is set by TrackMasker
  int lookahead = 0;  // W: 0 = BoxTimeline alone (as before spec 2026-10-27); >= 1 = KeyframeLinker in front of it
  int backfill = 0;   // B in [0, max(0, W - 1)]
};

struct TrackingOutput {
  // The masks of the oldest frames passed in that had none yet, in order: over a whole video, advance() and
  // finish() return one entry per frame, in the order the frames came.
  std::vector<std::vector<MaskBox>> masks{};
  // One entry per keyframe now settled, oldest first: whether each box it came with is masked, in order.
  std::vector<std::vector<bool>> confirmed{};
};

// Keyframes to masks (spec 2026-10-27 §3). With lookahead 0 this is BoxTimeline: the masks of each interval come
// with the next keyframe and its keyframe's flags with the one after. With lookahead W >= 1 the keyframes go through
// KeyframeLinker first and into a linked BoxTimeline W keyframes later, so masks come out W keyframes later too
// (the caller keeps those frames); a keyframe's flags come out when it leaves the linker.
class TrackMasker {
 public:
  // Throws std::invalid_argument for a negative lookahead, a backfill outside [0, max(0, lookahead - 1)], and what
  // BoxTimeline's constructor throws.
  TrackMasker(const TrackMaskerConfig& config, int frameWidth, int frameHeight);

  // As BoxTimeline::advance: `frames` from the previous keyframe (inclusive) to `next` (exclusive), empty on the
  // first call.
  TrackingOutput advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // As BoxTimeline::finish: the frames from the last keyframe to the end of the video. Returns every mask and flag
  // still owed and resets.
  TrackingOutput finish(const std::vector<FrameTime>& frames);

 private:
  static TimelineConfig timelineConfig(const TrackMaskerConfig& config);
  // Puts a keyframe the linker settled into the timeline, adding what comes out to `out`.
  void feed(Keyframe key, TrackingOutput& out);

  BoxTimeline timeline_;
  std::optional<KeyframeLinker> linker_;
  std::optional<int64_t> lastKey_;  // the index of the keyframe passed in last
  bool fed_ = false;                // linked: a keyframe went into the timeline
  // Linked: the frames of each keyframe from the last one fed to the timeline on, oldest first (the newest
  // keyframe's come with the next advance() or finish()).
  std::deque<std::vector<FrameTime>> intervals_;
};

}  // namespace vc
```

- [ ] **Step 4: 写 `track_masker.cpp`**

```cpp
#include "track/track_masker.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vc {

TimelineConfig TrackMasker::timelineConfig(const TrackMaskerConfig& c) {
  if (c.lookahead < 0 || c.backfill < 0 || c.backfill > std::max(0, c.lookahead - 1)) {
    throw std::invalid_argument("TrackMasker: lookahead must be >= 0 and backfill in [0, max(0, lookahead - 1)]");
  }
  TimelineConfig t = c.timeline;
  t.linked = c.lookahead > 0;
  return t;
}

TrackMasker::TrackMasker(const TrackMaskerConfig& c, int frameWidth, int frameHeight)
    : timeline_(timelineConfig(c), frameWidth, frameHeight) {
  if (c.lookahead > 0) {
    linker_.emplace(LinkerConfig{
        MatchParams{c.timeline.matchMaxDistance, c.timeline.matchMaxDistanceLoose, c.timeline.confirmScore},
        c.lookahead, c.backfill});
  }
}

TrackingOutput TrackMasker::advance(const std::vector<FrameTime>& frames, const Keyframe& next) {
  TrackingOutput out;
  if (!linker_) {
    TimelineOutput step = timeline_.advance(frames, next);
    out.masks = std::move(step.masks);
    if (lastKey_) out.confirmed.push_back(std::move(step.confirmed));
    lastKey_ = next.time.index;
    return out;
  }
  if (!lastKey_ && !frames.empty()) throw std::invalid_argument("TrackMasker: frames before the first keyframe");
  if (lastKey_ && (frames.empty() || frames.front().index != *lastKey_)) {
    throw std::invalid_argument("TrackMasker: frames must start at the previous keyframe");
  }
  std::vector<Keyframe> settled = linker_->push(next);  // throws before anything here changes
  if (lastKey_) intervals_.back() = frames;
  intervals_.emplace_back();
  lastKey_ = next.time.index;
  for (Keyframe& k : settled) feed(std::move(k), out);
  return out;
}

TrackingOutput TrackMasker::finish(const std::vector<FrameTime>& frames) {
  TrackingOutput out;
  if (!linker_) {
    TimelineOutput step = timeline_.finish(frames);
    out.masks = std::move(step.masks);
    if (lastKey_) out.confirmed.push_back(std::move(step.confirmed));
    lastKey_.reset();
    return out;
  }
  if (!lastKey_) {
    if (!frames.empty()) throw std::invalid_argument("TrackMasker: frames before the first keyframe");
    return out;
  }
  if (!frames.empty() && frames.front().index != *lastKey_) {
    throw std::invalid_argument("TrackMasker: frames must start at the previous keyframe");
  }
  intervals_.back() = frames;
  for (Keyframe& k : linker_->finish()) feed(std::move(k), out);
  TimelineOutput step = timeline_.finish(intervals_.front());
  for (auto& m : step.masks) out.masks.push_back(std::move(m));
  intervals_.clear();
  fed_ = false;
  lastKey_.reset();
  return out;
}

void TrackMasker::feed(Keyframe key, TrackingOutput& out) {
  std::vector<bool> flags;
  for (const BoxLink& link : key.links) {
    if (link.kind == LinkKind::Real) flags.push_back(link.masked);  // the real boxes come first, in order
  }
  if (!fed_) {
    timeline_.advance({}, key);
    fed_ = true;
  } else {
    const std::vector<FrameTime> frames = std::move(intervals_.front());
    intervals_.pop_front();
    TimelineOutput step = timeline_.advance(frames, key);
    for (auto& m : step.masks) out.masks.push_back(std::move(m));
  }
  out.confirmed.push_back(std::move(flags));
}

}  // namespace vc
```

- [ ] **Step 5: 运行测试**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='TrackMasker.*:KeyframeLinker.*:BoxTimeline.*:BoxMatching.*'`
Expected: 全部通过（TrackMasker 6 个）。

- [ ] **Step 6: 提交**

```bash
git add cpp/src/track/track_masker.h cpp/src/track/track_masker.cpp cpp/tests/track_masker_test.cpp cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/README.md
git commit -m "feat: TrackMasker runs KeyframeLinker in front of BoxTimeline with delayed output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 配置键

**Files:**
- Modify: `cpp/apps/vc_redact/redact_config.h`
- Modify: `cpp/apps/vc_redact/redact_config.cpp`
- Modify: `configs/redact.json`
- Modify: `docs/USAGE.md:127`（`hold_lost_ghosts` 那行之后）
- Test: `cpp/tests/redact_config_test.cpp`

**Interfaces:**
- Consumes: `TrackMaskerConfig`（Task 4）。
- Produces: `RedactConfig::trackLookaheadKeyframes`（int，默认 5）、`RedactConfig::trackBackfillKeyframes`（int，默认 2）；`TrackMaskerConfig trackMaskerFor(const RedactConfig&, const std::vector<std::string>& classes)`。

- [ ] **Step 1: 写失败的测试**

`redact_config_test.cpp`：
- `kAllKeys` 末尾加 `"track_lookahead_keyframes", "track_backfill_keyframes"`。
- `SerialisesEveryKey` 里 `EXPECT_EQ(j.size(), 29u);` 改为 `31u`。
- `DefaultsAreTheValidatedSettings` 里 `EXPECT_FALSE(c.timeline.holdLostGhosts);` 后加：
```cpp
  EXPECT_EQ(c.trackLookaheadKeyframes, 5);
  EXPECT_EQ(c.trackBackfillKeyframes, 2);
```
- 在 `ReadsThePlateMinimumWidth` 之后加：
```cpp
TEST(RedactConfig, ReadsTheTrackLookahead) {
  const RedactConfig c = parseRedactConfig(R"({"track_lookahead_keyframes": 8, "track_backfill_keyframes": 7})", "t");
  EXPECT_EQ(c.trackLookaheadKeyframes, 8);
  EXPECT_EQ(c.trackBackfillKeyframes, 7);
  const RedactConfig off = parseRedactConfig(R"({"track_lookahead_keyframes": 0, "track_backfill_keyframes": 0})", "t");
  EXPECT_EQ(off.trackLookaheadKeyframes, 0);
  for (const std::string text :
       {R"({"track_lookahead_keyframes": 31})", R"({"track_lookahead_keyframes": -1})",
        R"({"track_lookahead_keyframes": 2.5})", R"({"track_backfill_keyframes": "2"})",
        R"({"track_backfill_keyframes": 5})", R"({"track_lookahead_keyframes": 0})"}) {
    SCOPED_TRACE(text);
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted";
    } catch (const VcError& e) {
      const std::string message = e.what();
      EXPECT_EQ(message.rfind("cfg: ", 0), 0u) << message;
      EXPECT_NE(message.find("track_"), std::string::npos) << message;
    }
  }
  const TrackMaskerConfig m = trackMaskerFor(c, {"face", "plate"});
  EXPECT_EQ(m.lookahead, 8);
  EXPECT_EQ(m.backfill, 7);
  EXPECT_EQ(m.timeline.confirmScore, (std::vector<float>{0.5f, 0.2f}));
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`no member named 'trackLookaheadKeyframes'`。

- [ ] **Step 3: 实现**

`redact_config.h`：`#include "track/box_timeline.h"` 后加 `#include "track/track_masker.h"`。`RedactConfig` 里 `plateMinWidthPx` 之后加：
```cpp
  // Track-level masking (spec 2026-10-27): the output runs trackLookaheadKeyframes keyframes behind detection, so a
  // confirmed target missed on up to that many keyframes in a row is masked across the gap, and a target is masked
  // trackBackfillKeyframes keyframe intervals before it was first detected. Lookahead in [0, 30], 0 = off (masks
  // exactly as before); backfill in [0, lookahead - 1].
  int trackLookaheadKeyframes = 5;
  int trackBackfillKeyframes = 2;
```
`parseRedactConfig` 的注释里加一句：`track_lookahead_keyframes must be a JSON integer in [0, 30] and track_backfill_keyframes one in [0, track_lookahead_keyframes - 1] (0 when the lookahead is 0);`。在 `timelineFor` 声明后加：
```cpp
// The masking settings for a detector with these classes: timelineFor() plus the lookahead and backfill.
TrackMaskerConfig trackMaskerFor(const RedactConfig& config, const std::vector<std::string>& classes);
```

`redact_config.cpp`：
- `kKeys` 改为 31 个，末尾加 `"track_lookahead_keyframes", "track_backfill_keyframes"`。
- 常量区加 `constexpr int kMaxLookaheadKeyframes = 30;  // half a minute at 1 keyframe per second: past it, a typo`。
- `validateRedactConfig` 末尾加：
```cpp
  require(c.trackLookaheadKeyframes >= 0 && c.trackLookaheadKeyframes <= kMaxLookaheadKeyframes,
          "track_lookahead_keyframes must be in [0, 30]");
  require(c.trackBackfillKeyframes >= 0 && c.trackBackfillKeyframes <= std::max(0, c.trackLookaheadKeyframes - 1),
          "track_backfill_keyframes must be in [0, track_lookahead_keyframes - 1] (0 when track_lookahead_keyframes "
          "is 0), got " + std::to_string(c.trackBackfillKeyframes) + " with track_lookahead_keyframes " +
              std::to_string(c.trackLookaheadKeyframes));
```
- `parseRedactConfig` 里 `ort_intra_threads` 那行之后加：
```cpp
  c.trackLookaheadKeyframes = integerSetting(j, "track_lookahead_keyframes", c.trackLookaheadKeyframes, 0,
                                             kMaxLookaheadKeyframes, origin);
  c.trackBackfillKeyframes = integerSetting(j, "track_backfill_keyframes", c.trackBackfillKeyframes, 0,
                                            kMaxLookaheadKeyframes, origin);
```
- `redactConfigToJson` 里 `hold_lost_ghosts` 之后加：
```cpp
  j["track_lookahead_keyframes"] = c.trackLookaheadKeyframes;
  j["track_backfill_keyframes"] = c.trackBackfillKeyframes;
```
- 文件末尾（`timelineFor` 之后）加：
```cpp
TrackMaskerConfig trackMaskerFor(const RedactConfig& config, const std::vector<std::string>& classes) {
  return TrackMaskerConfig{timelineFor(config, classes), config.trackLookaheadKeyframes,
                           config.trackBackfillKeyframes};
}
```

`configs/redact.json`：`"hold_lost_ghosts": false,` 之后加两行：
```json
  "track_lookahead_keyframes": 5,
  "track_backfill_keyframes": 2,
```

`docs/USAGE.md`：`hold_lost_ghosts` 那一行之后加：
```markdown
| `track_lookahead_keyframes` | 5 | 按轨迹打码（设计 2026-10-27）：输出比检测晚这么多个关键帧（10 档时 5 个约 0.5 秒）。已确认的目标在这个窗口里连续漏检、又被检出时，中间的空档整段打码（`masks.jsonl` 的 `origin` 为 `bridged`）。0 关闭，打码与以前完全一样。整数 0–30；窗口越大，帧缓存越多（60 fps 1080p、5 个约 110 MB） |
| `track_backfill_keyframes` | 2 | 目标确认要打码后，往它第一次检出之前补这么多个关键帧间隔（`origin` 为 `backfilled`），按它前两次检出的速度往回推，每个间隔最多移半个框宽 / 框高。整数 0 到 `track_lookahead_keyframes` − 1；`track_lookahead_keyframes` 为 0 时只能是 0 |
```

- [ ] **Step 4: 运行测试**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='RedactConfig.*'`
Expected: 全部通过（含 `RepositoryFileMatchesTheDefaults`）。

- [ ] **Step 5: 提交**

```bash
git add cpp/apps/vc_redact/redact_config.h cpp/apps/vc_redact/redact_config.cpp cpp/tests/redact_config_test.cpp configs/redact.json docs/USAGE.md
git commit -m "feat: track_lookahead_keyframes and track_backfill_keyframes config keys

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `vc_redact` 接上 `TrackMasker`

**Files:**
- Modify: `cpp/apps/vc_redact/redact_job.h`
- Modify: `cpp/apps/vc_redact/redact_job.cpp`（`Processor` 类 242–481、`writeStats`、`redactSummary`、`runRedact`）
- Test: `cpp/tests/redact_job_test.cpp`

**Interfaces:**
- Consumes: `TrackMasker`、`TrackingOutput`（Task 4）；`trackMaskerFor`、两个新配置字段（Task 5）；`MaskOrigin::Bridged/Backfilled`（Task 2）。
- Produces:
  - `RedactStats` 新字段：`int lookaheadKeyframes`、`int backfillKeyframes`、`int64_t bridgedMaskBoxes`、`int64_t backfilledMaskBoxes`、`int64_t bufferedFramesPeak`
  - `stats.json` 新键：`lookahead_keyframes`、`backfill_keyframes`、`bridged_mask_boxes`、`backfilled_mask_boxes`、`buffered_frames_peak`
  - `int64_t lookaheadBufferFrames(int lookahead, int64_t stride)`、`void checkLookaheadBuffer(int lookahead, int64_t stride, int width, int height)`

- [ ] **Step 1: 写失败的测试**

`redact_job_test.cpp`：

`baseOptions` 里 `o.config.timeline = TimelineConfig{};` 之后加：
```cpp
  // Without lookahead: these tests were written for BoxTimeline alone (spec 2026-10-27 has its own below).
  o.config.trackLookaheadKeyframes = 0;
  o.config.trackBackfillKeyframes = 0;
```

在 `TheSummaryCountsTheLargeFacesDropped` 之后加（视频 30 fps、60 帧，5 档每 6 帧一个关键帧）：
```cpp
// ---- Track-level masking (spec 2026-10-27).

// Fake detections: a face (score `score`) at x `x1` on each of `frames`.
fs::path faceFrames(const std::string& name, const std::vector<int>& frames, double score = 0.9) {
  const fs::path path = test::tmpDir() / (name + ".jsonl");
  std::ofstream out(path);
  for (int f : frames) {
    out << R"({"video":"noisy_audio.mp4","frame":)" << f
        << R"(,"pts_ms":0,"cls":"face","x1":100,"y1":100,"x2":180,"y2":180,"score":)" << score
        << R"(,"src":"fake"})" << '\n';
  }
  return path;
}

// frame -> its masks.jsonl boxes.
std::map<int, nlohmann::json> masksByFrame(const RedactStats& s) {
  std::map<int, nlohmann::json> out;
  std::ifstream in(s.outputDir / "masks.jsonl");
  for (std::string line; std::getline(in, line);) {
    const nlohmann::json record = nlohmann::json::parse(line);
    out[record.at("frame").get<int>()] = record.at("boxes");
  }
  return out;
}

TEST(RedactJob, BridgesATargetMissedOnTwoKeyframesInARow) {
  // seen on keyframe 12, missed on 18 and 24, seen on 30
  RedactOptions off = baseOptions("redact_bridge_off");
  off.fakeDetections = faceFrames("redact_bridge_off", {12, 30});
  const std::map<int, nlohmann::json> before = masksByFrame(runRedact(off));
  for (int f = 18; f < 30; ++f) EXPECT_TRUE(before.at(f).empty()) << f;

  RedactOptions o = baseOptions("redact_bridge");
  o.fakeDetections = faceFrames("redact_bridge", {12, 30});
  o.config.trackLookaheadKeyframes = 3;
  const RedactStats s = runRedact(o);
  const std::map<int, nlohmann::json> masks = masksByFrame(s);
  for (int f = 12; f < 36; ++f) ASSERT_EQ(masks.at(f).size(), 1u) << f;
  for (int f = 13; f < 30; ++f) EXPECT_EQ(masks.at(f)[0].at("origin"), "bridged") << f;
  EXPECT_EQ(s.bridgedMaskBoxes, 17);
  EXPECT_EQ(s.frames, 60);
  EXPECT_EQ(lineCount(s.outputDir / "masks.jsonl"), 60u);
}

TEST(RedactJob, BackfillsBeforeTheFirstDetection) {
  RedactOptions o = baseOptions("redact_backfill");
  o.fakeDetections = faceFrames("redact_backfill", {30});
  o.config.trackLookaheadKeyframes = 3;
  o.config.trackBackfillKeyframes = 2;
  const RedactStats s = runRedact(o);
  const std::map<int, nlohmann::json> masks = masksByFrame(s);
  for (int f = 0; f < 18; ++f) EXPECT_TRUE(masks.at(f).empty()) << f;
  for (int f = 18; f < 30; ++f) {
    ASSERT_EQ(masks.at(f).size(), 1u) << f;
    EXPECT_EQ(masks.at(f)[0].at("origin"), "backfilled") << f;
  }
  EXPECT_EQ(s.backfilledMaskBoxes, 12);
  const std::vector<nlohmann::json> lines = detectionLines(s);
  ASSERT_EQ(lines.size(), 1u);  // filled-in boxes are not detections
  EXPECT_EQ(lines[0].at("frame").get<int>(), 30);
  EXPECT_TRUE(lines[0].at("confirmed").get<bool>());
}

TEST(RedactJob, TheLookaheadKeepsTheConfirmationOfEveryDetection) {
  // MasksALowScoreFaceOnlyOnceItIsConfirmed with a lookahead: the same flags, each keyframe written once
  RedactOptions o = baseOptions("redact_confirm_lookahead");
  o.config.faceConfirmScore = 0.5f;
  o.config.trackLookaheadKeyframes = 3;
  const fs::path fake = test::tmpDir() / "redact_confirm_lookahead.jsonl";
  {
    std::ofstream out(fake);
    for (const auto& [frame, x1] : std::vector<std::pair<int, int>>{{12, 100}, {24, 300}, {30, 300}}) {
      out << R"({"video":"noisy_audio.mp4","frame":)" << frame << R"(,"pts_ms":0,"cls":"face","x1":)" << x1
          << R"(,"y1":100,"x2":)" << x1 + 80 << R"(,"y2":180,"score":0.3,"src":"fake"})" << '\n';
    }
  }
  o.fakeDetections = fake;
  const RedactStats s = runRedact(o);
  std::map<int, bool> confirmed;
  for (const nlohmann::json& d : detectionLines(s)) confirmed[d.at("frame").get<int>()] = d.at("confirmed").get<bool>();
  EXPECT_EQ(confirmed, (std::map<int, bool>{{12, false}, {24, true}, {30, true}}));
  EXPECT_EQ(s.maskedFrames, 12);
}

TEST(RedactJob, StatsRecordTheLookahead) {
  RedactOptions o = baseOptions("redact_lookahead_stats");
  o.config.trackLookaheadKeyframes = 3;
  o.config.trackBackfillKeyframes = 2;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.bufferedFramesPeak, 24);  // 4 intervals of 6 frames
  const nlohmann::json stats = statsJson(s);
  EXPECT_EQ(stats.at("lookahead_keyframes").get<int>(), 3);
  EXPECT_EQ(stats.at("backfill_keyframes").get<int>(), 2);
  EXPECT_EQ(stats.at("bridged_mask_boxes").get<int>(), 0);
  EXPECT_EQ(stats.at("backfilled_mask_boxes").get<int>(), 0);  // the face is there from frame 0
  EXPECT_EQ(stats.at("buffered_frames_peak").get<int>(), 24);
  EXPECT_NE(redactSummary(s).find("tracks:        lookahead 3 keyframes, backfill 2; bridged 0, backfilled 0 mask "
                                  "boxes; up to 24 frames buffered\n"),
            std::string::npos)
      << redactSummary(s);
}

TEST(RedactJob, RefusesALookaheadThatWouldBufferTooMuch) {
  EXPECT_EQ(lookaheadBufferFrames(5, 6), 36);
  EXPECT_EQ(lookaheadBufferFrames(5, 1000), 720);  // an interval never exceeds 120 frames
  EXPECT_NO_THROW(checkLookaheadBuffer(5, 6, 1920, 1080));
  EXPECT_NO_THROW(checkLookaheadBuffer(0, 1000, 7680, 4320));  // no lookahead, no new limit
  try {
    checkLookaheadBuffer(30, 24, 3840, 2160);
    ADD_FAILURE() << "accepted";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("track_lookahead_keyframes"), std::string::npos) << e.what();
  }
}
```
文件头部加 `#include <utility>`（`std::pair`）。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`no member named 'bridgedMaskBoxes' in 'vc::RedactStats'`。

- [ ] **Step 3: 头文件**

`redact_job.h` 的 `RedactStats` 里 `narrowPlatesDropped` 之后加：
```cpp
  // Track-level masking (spec 2026-10-27): the lookahead and backfill in keyframes, the mask boxes they added
  // (masks.jsonl origin "bridged" / "backfilled") and the most frames held at once.
  int lookaheadKeyframes = 0;
  int backfillKeyframes = 0;
  int64_t bridgedMaskBoxes = 0;
  int64_t backfilledMaskBoxes = 0;
  int64_t bufferedFramesPeak = 0;
```
`pipelineQueueCapacity` 声明之后加：
```cpp
// The most frames vc_redact holds at once with this lookahead (spec 2026-10-27 §3.3): lookahead + 1 keyframe
// intervals of `stride` frames, an interval being at most the 120 frames after which a keyframe is forced.
int64_t lookaheadBufferFrames(int lookahead, int64_t stride);

// Throws VcError (naming track_lookahead_keyframes) when, with a lookahead, lookaheadBufferFrames of a width x height
// video at 3 bytes per pixel exceed 2 GiB. Without a lookahead the buffer is what it always was: no check.
void checkLookaheadBuffer(int lookahead, int64_t stride, int width, int height);
```
`redactSummary` 的注释里 "the frame and mask counts," 后加 "the track masking (lookahead, backfill, the mask boxes they added, the frames buffered),"。

- [ ] **Step 4: `Processor`**

`redact_job.cpp`：`#include <deque>`；`#include "track/box_timeline.h"` 换成 `#include "track/track_masker.h"`。常量区加：
```cpp
// The frames held for the lookahead may take at most this much memory (spec 2026-10-27 §3.3).
constexpr uint64_t kMaxLookaheadBufferBytes = 2ull * 1024 * 1024 * 1024;
```
`Processor`：
- 构造函数初始化列表里 `timeline_(timelineFor(options.config, detector.classes()), info.width, info.height),` 换成 `masker_(trackMaskerFor(options.config, detector.classes()), info.width, info.height),`。
- `onVideo` 换成：
```cpp
  void onVideo(VideoItem item) {
    const Clock::time_point started = Clock::now();
    // A keyframe every schedule_.stride frames by frame index (spec 2026-10-19 §2), the first frame always
    // (BoxTimeline starts on a keyframe), and one whenever kMaxBufferedFrames frames have come since the last.
    const bool key = options_.detectEveryFrame || !sawKeyframe_ || onSchedule(schedule_, item.index) ||
                     interval_.size() >= kMaxBufferedFrames;
    stats_.track.busySec += secondsSince(started);
    if (!key) {
      keep(std::move(item));
      return;
    }
    sawKeyframe_ = true;

    const VideoFrame view = viewOf(item);
    const Clock::time_point detectStarted = Clock::now();
    std::vector<SourcedBox> detected = detector_.detectSourced(view);
    stats_.detect.busySec += secondsSince(detectStarted);
    ++stats_.detect.items;
    ++stats_.keyframes;
    // Giant low-score faces (spec 2026-10-19 §3.6) and plates too narrow to read never reach the timeline,
    // detections.jsonl or the masks.
    const KeyframeFilter filtered =
        filterKeyframe(std::move(detected), detector_.classes(), options_.config, width_, height_);
    const std::vector<SourcedBox>& found = filtered.kept;
    // Boxes the verifier rejects never reach the timeline (spec 2026-10-21 §4.3).
    std::vector<std::optional<VerifyResult>> verified = verify(view, found);
    std::vector<Box> boxes;
    boxes.reserve(found.size());
    for (size_t i = 0; i < found.size(); ++i) {
      if (!verified[i] || verified[i]->passed) boxes.push_back(found[i].box);
    }
    stats_.detections += static_cast<int64_t>(boxes.size());
    stats_.largeFacesDropped += filtered.largeFaces;
    stats_.narrowPlatesDropped += filtered.narrowPlates;
    pending_.push_back(KeyframeDetections{item.index, item.ptsMs, found, std::move(verified)});

    const Clock::time_point trackStarted = Clock::now();
    const TrackingOutput step = masker_.advance(interval_, Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    // The keyframes the masker has settled are written now that it has said which of their boxes it masked.
    writeDetections(step.confirmed);
    emit(step.masks);
    interval_.clear();
    keep(std::move(item));
  }

  void onEnd() {
    const Clock::time_point started = Clock::now();
    const TrackingOutput step = masker_.finish(interval_);
    stats_.track.busySec += secondsSince(started);
    writeDetections(step.confirmed);
    emit(step.masks);
    if (!buffer_.empty() || !pending_.empty()) {
      throw VcError("internal error: frames or detections were left without masks at the end");
    }
    if (!out_.push(Item{EndItem{}})) throw PipelineAborted{};
  }
```
- `writeDetections` 换成：
```cpp
  // Writes the detections.jsonl lines of the oldest pending keyframes, one keyframe per entry of `confirmed` (its
  // flags: one per box the masker got, in order); a box the verifier rejected is written unconfirmed.
  void writeDetections(const std::vector<std::vector<bool>>& confirmed) {
    for (const std::vector<bool>& flags : confirmed) {
      if (pending_.empty()) throw VcError("internal error: confirmation flags for a keyframe that is not pending");
      const KeyframeDetections& d = pending_.front();
      size_t next = 0;
      for (size_t i = 0; i < d.boxes.size(); ++i) {
        const SourcedBox& b = d.boxes[i];
        const std::optional<VerifyResult>& v = d.verified[i];
        bool masked = false;
        if (!v || v->passed) {
          if (next >= flags.size()) {
            throw VcError("internal error: confirmation flags do not match the keyframe's detections");
          }
          masked = flags[next++];
        }
        detections_ << detectionRecord(video_, d.frame, d.ptsMs, b.box, detector_.classes(), b.src, masked, v);
      }
      if (next != flags.size()) throw VcError("internal error: confirmation flags do not match the keyframe's detections");
      pending_.pop_front();
    }
  }
```
- 删除 `times()`；加：
```cpp
  // Holds a frame until its masks come out; it belongs to the interval of the last keyframe.
  void keep(VideoItem item) {
    interval_.push_back(FrameTime{item.index, item.ptsMs});
    buffer_.push_back(std::move(item));
    stats_.bufferedFramesPeak = std::max(stats_.bufferedFramesPeak, static_cast<int64_t>(buffer_.size()));
  }
```
- `emit` 换成：
```cpp
  // Writes masks.jsonl, pixelates and sends the oldest masks.size() held frames downstream, in order. A mask's block
  // comes from its track (TrackBlocks), so a target keeps its mosaic from frame to frame (spec 2026-10-19 §4); a
  // plate's mosaic stays inside its mask instead of snapping out to the grid.
  void emit(const std::vector<std::vector<MaskBox>>& masks) {
    if (masks.size() > buffer_.size()) throw VcError("internal error: more masks than held frames");
    for (const std::vector<MaskBox>& frameMasks : masks) {
      VideoItem frame = std::move(buffer_.front());
      buffer_.pop_front();
      const Clock::time_point started = Clock::now();
      std::vector<MaskBox> boxes = frameMasks;
      std::vector<MosaicRect> rects;
      rects.reserve(boxes.size());
      for (MaskBox& m : boxes) {
        const PixelRect rect = coveringRect(m.x1, m.y1, m.x2, m.y2, frame.frame->width, frame.frame->height);
        m.block = blocks_.blockFor(m.track, rect);
        m.snap = m.cls != plateClass_;
        rects.push_back(MosaicRect{rect, m.block, m.snap});
        if (m.origin == MaskOrigin::Bridged) ++stats_.bridgedMaskBoxes;
        if (m.origin == MaskOrigin::Backfilled) ++stats_.backfilledMaskBoxes;
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
  }
```
- 成员：`BoxTimeline timeline_;` 换成 `TrackMasker masker_;`；`std::vector<VideoItem> buffer_;` 换成：
```cpp
  std::deque<VideoItem> buffer_;     // decoded, waiting for their masks, oldest first
  std::vector<FrameTime> interval_;  // the frames from the last keyframe (inclusive) on
```
`std::optional<KeyframeDetections> pending_;` 换成 `std::deque<KeyframeDetections> pending_;  // keyframes waiting for their flags, oldest first`。
- `KeyframeDetections` 的注释改为 `// A keyframe's detections, written once the masker says which were masked.`

- [ ] **Step 5: 统计、汇总、内存上限**

`writeStats`：`j["mask_boxes"] = s.maskBoxes;` 之后加：
```cpp
  j["lookahead_keyframes"] = s.lookaheadKeyframes;
  j["backfill_keyframes"] = s.backfillKeyframes;
  j["bridged_mask_boxes"] = s.bridgedMaskBoxes;
  j["backfilled_mask_boxes"] = s.backfilledMaskBoxes;
  j["buffered_frames_peak"] = s.bufferedFramesPeak;
```
`redactSummary`：`frames:` 那行之后加：
```cpp
  out += formatted("tracks:        lookahead %d keyframes, backfill %d; bridged %lld, backfilled %lld mask boxes; up to "
                   "%lld frames buffered\n",
                   s.lookaheadKeyframes, s.backfillKeyframes, static_cast<long long>(s.bridgedMaskBoxes),
                   static_cast<long long>(s.backfilledMaskBoxes), static_cast<long long>(s.bufferedFramesPeak));
```
`pipelineQueueCapacity` 之后加：
```cpp
int64_t lookaheadBufferFrames(int lookahead, int64_t stride) {
  const int64_t interval = std::clamp<int64_t>(stride, 1, static_cast<int64_t>(kMaxBufferedFrames));
  return (static_cast<int64_t>(std::max(lookahead, 0)) + 1) * interval;
}

void checkLookaheadBuffer(int lookahead, int64_t stride, int width, int height) {
  if (lookahead <= 0) return;
  const auto frames = static_cast<uint64_t>(lookaheadBufferFrames(lookahead, stride));
  const uint64_t bytes = frames * static_cast<uint64_t>(std::max(width, 1)) * static_cast<uint64_t>(std::max(height, 1)) *
                         kBytesPerPixel;
  if (bytes > kMaxLookaheadBufferBytes) {
    throw VcError(formatted("track_lookahead_keyframes %d would hold about %llu frames (%.1f GiB) of this %dx%d video, "
                            "over the 2 GiB limit: lower track_lookahead_keyframes",
                            lookahead, static_cast<unsigned long long>(frames),
                            static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), width, height));
  }
}
```
`runRedact`：`stats.detectFpsAboveVideoRate = schedule.aboveVideoRate;` 之后加：
```cpp
  stats.lookaheadKeyframes = options.config.trackLookaheadKeyframes;
  stats.backfillKeyframes = options.config.trackBackfillKeyframes;
  // Before the model is loaded: a lookahead this video cannot afford is refused at once (spec 2026-10-27 §3.3).
  checkLookaheadBuffer(options.config.trackLookaheadKeyframes, schedule.stride, reader.info().width,
                       reader.info().height);
```

- [ ] **Step 6: 运行测试**

Run: `cmake --build build/cpp -j 4 --target vc_tests && ./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*'`
Expected: 全部通过（含原有的 `ForcesAKeyframeWhenTheBufferGrowsTooLong`、`MasksALowScoreFaceOnlyOnceItIsConfirmed`）。

- [ ] **Step 7: 提交**

```bash
git add cpp/apps/vc_redact/redact_job.h cpp/apps/vc_redact/redact_job.cpp cpp/tests/redact_job_test.cpp
git commit -m "feat: vc_redact masks by track with a keyframe lookahead and reports what it filled in

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 真实视频上的回归与 3 处泄露

**Files:** 无代码改动（除非发现问题）；配置放在 `data/track-configs/`，输出放在 `data/redacted/track-*`（都不进仓库）。

- [ ] **Step 1: 编译 release 并跑相关测试组**

```bash
cmake --build build/cpp-release -j 4 --target vc_redact
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:BoxMatching.*:KeyframeLinker.*:TrackMasker.*:Mosaic.*:MosaicGrid.*:BoundedQueue.*:RedactConfig.*:Records.*'
./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```
Expected: 两组都 `[  PASSED  ]`。

- [ ] **Step 2: W = 0 与改动前逐行一致**

```bash
mkdir -p data/track-configs
.venv/bin/python -c "import json; c=json.load(open('configs/redact.json')); c['track_lookahead_keyframes']=0; c['track_backfill_keyframes']=0; json.dump(c, open('data/track-configs/w0.json','w'), indent=2)"
VC_VERIFIER= VC_REDACT_ARGS="--overwrite --config data/track-configs/w0.json" scripts/redact.sh data/redacted/track-w0 data/videos/sg_drive_60s.mp4
cmp data/redacted/track-base/sg_drive_60s/masks.jsonl data/redacted/track-w0/sg_drive_60s/masks.jsonl && echo identical
cmp data/redacted/track-base/sg_drive_60s/detections.jsonl data/redacted/track-w0/sg_drive_60s/detections.jsonl && echo identical
```
Expected: 两个 `identical`。不一致就停下：用 `diff <(head -c ...)` 找第一处差异，回到 Task 6 查原因，不要往下走。

- [ ] **Step 3: 新默认值（W = 5、B = 2）**

```bash
VC_VERIFIER= VC_REDACT_ARGS=--overwrite scripts/redact.sh data/redacted/track-w5 data/videos/sg_drive_60s.mp4
.venv/bin/python -c "import json; s=json.load(open('data/redacted/track-w5/sg_drive_60s/stats.json')); print({k: s[k] for k in ('lookahead_keyframes','backfill_keyframes','bridged_mask_boxes','backfilled_mask_boxes','buffered_frames_peak','masked_frames','mask_boxes')})"
```
Expected: `lookahead_keyframes` 5、`backfill_keyframes` 2、`buffered_frames_peak` 36，`bridged_mask_boxes` 和 `backfilled_mask_boxes` 都大于 0。

- [ ] **Step 4: 3 处泄露帧是否已被盖住（不调用模型）**

脚本写到 scratchpad，参数传路径：
```python
# check_leaks.py <masks.jsonl>
import sys
from pathlib import Path

from vc_eval.records import read_masks
from vc_gt.crops import leak_frames
from vc_gt.tracks import read_tracks

_, masks = read_masks(Path(sys.argv[1]))
tracks = {str(t.id): t for t in read_tracks(Path("data/gt/sg_drive_60s/tracks.jsonl"))}
for tid, frame in [("16042", 1770), ("19940", 2172), ("25810", 4068)]:
    print(f"o{tid}@{frame}:", "still unmasked" if frame in leak_frames(tracks[tid], masks, 1080) else "masked")
```
Run:
```bash
PYTHONPATH=py .venv/bin/python <scratchpad>/check_leaks.py data/redacted/track-base/sg_drive_60s/masks.jsonl
PYTHONPATH=py .venv/bin/python <scratchpad>/check_leaks.py data/redacted/track-w5/sg_drive_60s/masks.jsonl
```
Expected: 基线三行都是 `still unmasked`；W = 5 三行都是 `masked`。若 W = 5 仍有 `still unmasked`，把那条轨迹前后关键帧的 `detections.jsonl` 和 `masks.jsonl`（`origin`）列出来，交给用户判断是调大 W / B 还是另有原因，不要自行改设计。

- [ ] **Step 5: 记录结果**

把 Step 2–4 的输出（含 stats 摘要）贴进本任务的执行记录，供 Task 9 的报告引用。本任务无提交。

---

### Task 8: 文档

**Files:**
- Modify: `cpp/README.md`（"补框：一个目标一个框"那段之后、"已知代价"列表）
- Modify: `docs/superpowers/specs/2026-10-27-track-masking-design.md`（文末加 §7）

- [ ] **Step 1: `cpp/README.md`**

在 `cpp/README.md:137`（"补框：一个目标一个框"那一段）之后加一段：
```markdown
- **按轨迹打码**（设计见 `docs/superpowers/specs/2026-10-27-track-masking-design.md`）：`track_lookahead_keyframes`（默认 5）大于 0 时，关键帧先经过 `KeyframeLinker`（`src/track/keyframe_linker.h`），晚这么多个关键帧再交给 `BoxTimeline`。连接器用同样的两轮规则配对（`src/track/box_matching.h`），但每条轨迹以"最后一次检出 + 速度 × 经过的帧数"去配。已确认的轨迹连续漏检不超过窗口、又被配上时，漏掉的每个关键帧补一个从最后一次检出到重新检出按帧号插值的框（`masks.jsonl` 的 `origin` 为 `bridged`）。轨迹一经确认（高分当场；低分要在下一个关键帧又被检出），往它第一次检出之前补 `track_backfill_keyframes`（默认 2）个关键帧：框按前两次检出的速度往回推（还没有速度时原地），每个间隔每轴最多移半个框宽 / 框高（`origin` 为 `backfilled`），不早于视频第一帧。补出来的框不写进 `detections.jsonl`、不参与确认、不更新速度。`BoxTimeline` 在这种模式下（`TimelineConfig::linked`）不再自己配对和确认，而是按框上的轨迹号连框、按连接器的结论打码；插值、保持、平滑、`hold_lost_ghosts` 都不变。帧缓存相应加长到 W + 1 个关键帧间隔（`stats.json` 的 `buffered_frames_peak`），超过 2 GiB 时启动即报错。`track_lookahead_keyframes` 为 0 时，`BoxTimeline` 照旧自己配对，输出与以前逐字节相同。
```
并把"已知代价"里"同一个目标连续两个关键帧都漏检"那一条的末尾加一句：`打开按轨迹打码（默认）后，窗口内的空档会被连上，这一条只在漏检超过 track_lookahead_keyframes 个关键帧时才会发生。`

- [ ] **Step 2: 设计文档补 §7**

在 `docs/superpowers/specs/2026-10-27-track-masking-design.md` 末尾加：
```markdown
## 7. 实施时的修正（2026-10-27，写计划时）

- **§3.2 的结构**：`BoxTimeline` 的配对没有删掉，而是搬进 `track/box_matching`，由连接器和 `BoxTimeline` 共用。`BoxTimeline` 加 `linked` 模式：W ≥ 1 时按连接器的轨迹号连框、按 `BoxLink::masked` 打码；W = 0 时完全照旧，自己配对和确认。外面再包一层 `TrackMasker`，`vc_redact` 只和它打交道。这样 W = 0 不改动任何既有路径，原有 69 个 `BoxTimeline` 测试一行不改。
- **§2.2 往回补的上限**：连接器在处理完新关键帧之后才吐出最老的一个，所以确认时 k_{a−B} 一定还在窗口里。高分轨迹在 k_a 确认，低分轨迹在 k_{a+1} 确认，都能补满 `track_backfill_keyframes` ≤ W − 1 个，只有视频开头会少补。§2.2 里"高分最多 W − 1、低分最多 W − 2"作废。
- **§5 的基线**：召回、误打面积和观感直接用已有报告 `data/reports/fp-r3-pw16` 的数字作对照，不再同一时段重跑一遍三段视频，省下约 20 分钟的重负载；速度对照只在 `sg_drive_2400s` 上用 W = 0 和默认值各跑一次 `vc_redact`。`vc_eval falsepos` 数的是检测框，补出来的框不算检测，所以不用它衡量误打，改看 `vc_eval run` 报告里的"误打面积"。
```

- [ ] **Step 3: 提交**

```bash
git add cpp/README.md docs/superpowers/specs/2026-10-27-track-masking-design.md
git commit -m "docs: track-level masking in the C++ README and the spec's implementation notes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 验收（设计 §5）

**Files:**
- Create: `docs/reports/track-masking-2026-10-27.md`
- 数据输出：`data/redacted/track-eval/`、`data/reports/track-eval/`、`data/gt-frames-redacted/track-w5/`、`data/reports/gt/sg_drive_60s-track-w5/`（都不进仓库）

- [ ] **Step 1: 三段视频的召回、误打面积、观感**

`vc_eval run` 会对每段先跑一遍逐帧检测（`--detect-every-frame`）作参照，再跑 10 档，三段合计约 20 分钟，后台运行：
```bash
PYTHONPATH=py .venv/bin/python -m vc_eval run --stems sg_drive_1200s,sg_drive_2400s,sg_drive_60s \
  --model data/models/r3 --face-model data/models/yunet --fps 10 --recall-target face=0.95,plate=0.98 \
  --config configs/redact.json --redacted-root data/redacted/track-eval --out data/reports/track-eval
```
Expected: 退出码 0，生成 `data/reports/track-eval/summary.md`。

- [ ] **Step 2: 与基线对比**

从 `data/reports/track-eval/summary.md` 和 `data/reports/fp-r3-pw16/summary.md` 里取三段合计的：P1 覆盖率（车牌、人脸）、误打面积均值和 95 分位、屏上面积均值和 99 分位、帧间跳变。按设计 §5 判：
- 召回：车牌 ≥ 98.2%，人脸 ≥ 100%（不低于基线）。
- 误打面积均值、95 分位：各 ≤ 基线 × 1.10。
- 屏上面积均值、99 分位：各 ≤ 基线 × 1.10。
- 帧间跳变 ≤ 基线 + 0.5 个百分点。

把对比表写进执行记录。

- [ ] **Step 3: 速度**

```bash
VC_VERIFIER= VC_REDACT_ARGS="--overwrite --config data/track-configs/w0.json" scripts/redact.sh data/redacted/track-speed-w0 data/videos/sg_drive_2400s.mp4
VC_VERIFIER= VC_REDACT_ARGS=--overwrite scripts/redact.sh data/redacted/track-speed-w5 data/videos/sg_drive_2400s.mp4
```
从两个 `stats.json` 取 `frames / elapsed_s` 和 `stages.track` 的 fps，写进执行记录。

- [ ] **Step 4: 新的未打码帧要问多少组（不调用模型）**

先抽出新打码视频的 10 fps 帧：
```bash
build/cpp-release/vc_extract --input data/redacted/track-w5/sg_drive_60s/sg_drive_60s.redacted.mp4 \
  --out data/gt-frames-redacted/track-w5 --fps 10 --max-fps 10 --hwaccel none
diff <(ls data/gt-frames/sg_drive_60s | grep jpg) <(ls data/gt-frames-redacted/track-w5/sg_drive_60s.redacted | grep jpg) && echo same-frames
```
Expected: `same-frames`（帧号与答案的帧一致）。

再数问题 B 还没判过的帧（脚本写到 scratchpad）：
```python
# count_pending.py <masks.jsonl>
import math
import sys
from collections import Counter
from pathlib import Path

from vc_eval.records import read_masks
from vc_gt.config import load_config
from vc_gt.crops import leak_frames
from vc_gt.judge import FRAMES, frame_id, is_candidate
from vc_gt.tracks import read_tracks
from vc_hardneg.judgments import read_judgments, resolve

gt = Path("data/gt/sg_drive_60s")
cfg = load_config(Path("data/gt/gt-parallel2.json"))
_, masks = read_masks(Path(sys.argv[1]))
known = resolve(read_judgments(gt / FRAMES))
todo = Counter()
for t in read_tracks(gt / "tracks.jsonl"):
    if is_candidate(t, 1080, cfg):
        todo[t.cls] += sum(frame_id(t.id, f) not in known for f in leak_frames(t, masks, 1080))
print("frames to ask:", dict(todo), "groups:", sum(math.ceil(n / cfg.llm.batch) for n in todo.values()))
```
Run: `PYTHONPATH=py .venv/bin/python <scratchpad>/count_pending.py data/redacted/track-w5/sg_drive_60s/masks.jsonl`

- [ ] **Step 5: 停下来问用户**

把 Step 2 的对比表、Step 3 的速度和 Step 4 的组数报给用户，问是否同意调用模型。**用户明确同意之前不要进行 Step 6。** 组数为 0 时跳过 Step 6，直接到 Step 7。

- [ ] **Step 6: 问题 B 与人工核对（用户同意后）**

```bash
VC_HARDNEG_ALLOW_LLM=1 PYTHONPATH=py .venv/bin/python -m vc_gt judge --gt data/gt/sg_drive_60s \
  --frames data/gt-frames/sg_drive_60s --masks data/redacted/track-w5/sg_drive_60s/masks.jsonl \
  --redacted-frames data/gt-frames-redacted/track-w5/sg_drive_60s.redacted --stage frames --config data/gt/gt-parallel2.json
PYTHONPATH=py .venv/bin/python -m vc_gt review --gt data/gt/sg_drive_60s --frames data/gt-frames/sg_drive_60s \
  --masks data/redacted/track-w5/sg_drive_60s/masks.jsonl --set h1 --out data/review/gt-h1-track-w5
```
`review` 的帧取的是哪一份打码帧，以 `vc_gt review` 的实现为准；如果它用的是答案目录里缓存的截图（`crops/frames`），先确认这些截图来自 `track-w5` 的打码帧，不是旧的。核对页要给用户看时，在 `data/review` 下起本地服务（`python -m http.server 8766 --bind 127.0.0.1`，后台运行），等用户导出后用 `vc_gt import-human --kind frames` 导入，用完按 PID 停掉服务。

- [ ] **Step 7: 泄露报告**

```bash
rm -rf data/reports/gt/sg_drive_60s-track-w5
PYTHONPATH=py .venv/bin/python -m vc_eval objects --gt data/gt/sg_drive_60s --frames data/gt-frames/sg_drive_60s \
  --redacted-frames data/gt-frames-redacted/track-w5/sg_drive_60s.redacted \
  --masks data/redacted/track-w5/sg_drive_60s/masks.jsonl --out data/reports/gt/sg_drive_60s-track-w5
```
Expected: 首行 `leaks found on any candidate: plate 0, face 0`。有新泄露的，逐个写明原因（漏检超过窗口、从未检出、检出了但框偏），作为报告的一部分。

- [ ] **Step 8: 不达标时的调参（设计 §5 第 5 条）**

只有 Step 2 或 Step 7 不达标时才做：在 W ∈ {3, 5, 8}、B ∈ {1, 2, 3}（B ≤ W − 1）里，用 `data/track-configs/w<W>b<B>.json` 逐个在 `sg_drive_60s` 上跑 `vc_redact`，再用 Task 7 Step 4 的脚本查 3 处泄露；对能清零的组合再跑 Step 1 比误打和观感。都不满足就停下，把各组合的数字表给用户定。

- [ ] **Step 9: 写报告并提交**

`docs/reports/track-masking-2026-10-27.md` 写：目标与默认值；Task 7 的回归结果（W = 0 逐行一致）；3 处泄露在基线和新版下的情况；Step 2 的对比表和是否达标；Step 3 的速度；Step 4–7 的新问组数、核对结果与泄露数；结论（是否保留默认 W = 5、B = 2）。附 3 处泄露帧在新版下的打码截图（从 `data/gt-frames-redacted/track-w5` 裁，图片放 `docs/reports/img/track-masking/`）。

```bash
git add docs/reports/track-masking-2026-10-27.md docs/reports/img/track-masking
git commit -m "docs: track-level masking acceptance report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## 修订（2026-10-27，Task 7 试跑后，用户决定；见设计 §8）

以下覆盖前文：
- **默认值**：`track_lookahead_keyframes` 2、`track_backfill_keyframes` 1；新键 `hold_lost_min_score` 默认 0.5（Task 10）。
- **Task 7**：Step 2 的 W = 0 配置还要设 `track_backfill_keyframes` 0、`hold_lost_min_score` null（`data/track-configs/w0.json`）；Step 3 改为新默认值，输出到 `data/redacted/track-w2`，期望 `lookahead_keyframes` 2、`backfill_keyframes` 1、`buffered_frames_peak` 18；Step 4 期望新默认值下三行都是 `masked`。
- **Task 8**：设计文档 §7、§8 已由控制者写好，Task 8 只改 `cpp/README.md`，段落里的默认值写 2 和 1，并加一句 `hold_lost_min_score` 的规则（见设计 §8）。
- **Task 9**：所有 `track-w5` 路径改为 `track-w2`；报告结论写"是否保留默认 W = 2、B = 1、`hold_lost_min_score` 0.5"。Step 8 的调参范围改为 W ∈ {2, 3}、B ∈ {0, 1}、`hold_lost_min_score` ∈ {0.5, 0.7, null}。
- **执行顺序**：Task 10 → Task 7（重跑）→ Task 8 → Task 9。

### Task 10: 新默认值与高分轨迹丢失后多保持一个间隔

**Files:**
- Modify: `cpp/src/track/box_timeline.h`、`cpp/src/track/box_timeline.cpp`
- Modify: `cpp/apps/vc_redact/redact_config.h`、`cpp/apps/vc_redact/redact_config.cpp`
- Modify: `configs/redact.json`、`docs/USAGE.md`（按轨迹打码的两行，及新增一行）
- Test: `cpp/tests/box_timeline_test.cpp`、`cpp/tests/redact_config_test.cpp`

**Interfaces:**
- Produces:
  - `TimelineConfig::holdLostMinScore`（`std::optional<float>`，默认 `std::nullopt` = 关闭）
  - `RedactConfig` 默认 `timeline.holdLostMinScore = 0.5f`、`trackLookaheadKeyframes = 2`、`trackBackfillKeyframes = 1`
  - 配置键 `hold_lost_min_score`：[0, 1] 内的数，或 `null`（关闭）；缺省 0.5

- [ ] **Step 1: 写失败的测试**

`cpp/tests/box_timeline_test.cpp` 末尾（`}  // namespace` 之前）加：
```cpp
// ---- A lost track whose best real detection scored high enough is held for one more interval (spec 2026-10-27 §8).

TEST(BoxTimeline, ALostTrackWhoseBestScoreReachesTheMinimumIsHeldOneMoreInterval) {
  TimelineConfig c = confirming();
  c.holdLostMinScore = 0.5f;
  BoxTimeline t(c, 1920, 1080);
  t.advance({}, key(0, {face(0, 0, 80, 80), weakFace(500, 0, 580, 80)}));
  t.advance(frames(0, 6), key(6, {face(0, 0, 80, 80), weakFace(500, 0, 580, 80)}));  // both confirmed
  t.advance(frames(6, 12), key(12, {}));                                              // both lost: held
  const TimelineOutput out = t.advance(frames(12, 18), key(18, {}));
  ASSERT_EQ(out.masks[0].size(), 1u);  // the face scoring 0.9, not the one scoring 0.3
  expectBox(out.masks[0][0], 0, 0, 80, 80, MaskOrigin::Held);
  expectBox(out.masks[5][0], 0, 0, 80, 80, MaskOrigin::Held);
  const TimelineOutput after = t.advance(frames(18, 24), key(24, {}));
  for (const auto& m : after.masks) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, TheMinimumScoreAlsoHoldsAGhostLeftAtTheEnd) {
  for (const bool on : {false, true}) {
    TimelineConfig c = exact();
    if (on) c.holdLostMinScore = 0.5f;
    BoxTimeline t(c, 1920, 1080);
    t.advance({}, key(0, {face(0, 0, 80, 80)}));
    t.advance(frames(0, 6), key(6, {face(0, 0, 80, 80)}));
    t.advance(frames(6, 12), key(12, {}));
    const TimelineOutput out = t.finish(frames(12, 18));
    for (const auto& m : out.masks) EXPECT_EQ(m.size(), on ? 1u : 0u) << on;
  }
}

TEST(BoxTimeline, OnlyRealBoxesCountTowardsTheBestScore) {
  // the backfilled box scores 0.9, the only real one 0.3: the track is not held once more after it is lost
  TimelineConfig c = linkedExact();
  c.holdLostMinScore = 0.5f;
  BoxTimeline t(c, 1920, 1080);
  t.advance({}, linkedKey(0, {face(0, 0, 80, 80)}, {{1, LinkKind::Backfilled}}));
  t.advance(frames(0, 6), linkedKey(6, {weakFace(0, 0, 80, 80)}, {{1}}));
  t.advance(frames(6, 12), linkedKey(12, {}, {}));
  const TimelineOutput out = t.advance(frames(12, 18), linkedKey(18, {}, {}));
  for (const auto& m : out.masks) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, RejectsAMinimumHoldScoreOutsideZeroToOne) {
  for (const float bad : {-0.1f, 1.5f, std::numeric_limits<float>::quiet_NaN()}) {
    TimelineConfig c = exact();
    c.holdLostMinScore = bad;
    EXPECT_THROW(BoxTimeline(c, 1920, 1080), std::invalid_argument) << bad;
  }
}
```

`cpp/tests/redact_config_test.cpp`：
- `kAllKeys` 加 `"hold_lost_min_score"`；`SerialisesEveryKey` 的 `31u` 改为 `32u`。
- `DefaultsAreTheValidatedSettings`：`trackLookaheadKeyframes` 期望改为 2、`trackBackfillKeyframes` 改为 1，并加 `ASSERT_TRUE(c.timeline.holdLostMinScore.has_value()); EXPECT_FLOAT_EQ(*c.timeline.holdLostMinScore, 0.5f); EXPECT_FALSE(TimelineConfig{}.holdLostMinScore.has_value());`。
- `ReadsTheTrackLookahead` 里 `{"track_backfill_keyframes": 5}` 那条在新默认 W = 2 下仍被拒（B ≤ 1），保留。
- `expectSameConfig`（往返比较的辅助函数）加上 `holdLostMinScore` 的比较；非默认往返配置里设 `hold_lost_min_score` 为 0.7。
- 新测试：
```cpp
TEST(RedactConfig, ReadsTheMinimumHoldScore) {
  EXPECT_FLOAT_EQ(*parseRedactConfig(R"({"hold_lost_min_score": 0.7})", "t").timeline.holdLostMinScore, 0.7f);
  EXPECT_FALSE(parseRedactConfig(R"({"hold_lost_min_score": null})", "t").timeline.holdLostMinScore.has_value());
  EXPECT_FLOAT_EQ(*parseRedactConfig("{}", "t").timeline.holdLostMinScore, 0.5f);  // absent: the default
  for (const std::string value : {"-0.1", "1.5", R"("0.5")", "true"}) {
    const std::string text = R"({"hold_lost_min_score": )" + value + "}";
    SCOPED_TRACE(text);
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted";
    } catch (const VcError& e) {
      EXPECT_NE(std::string(e.what()).find("hold_lost_min_score"), std::string::npos) << e.what();
    }
  }
  const nlohmann::ordered_json off = redactConfigToJson(parseRedactConfig(R"({"hold_lost_min_score": null})", "t"));
  EXPECT_TRUE(off.at("hold_lost_min_score").is_null());
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cmake --build build/cpp -j 4 --target vc_tests`
Expected: 编译失败，`no member named 'holdLostMinScore'`。

- [ ] **Step 3: `BoxTimeline`**

`box_timeline.h`：`TimelineConfig` 末尾（`linked` 之后）加：
```cpp
  // A confirmed track whose best real detection scored at least this is held for one more interval when it is lost
  // again at the next keyframe, as holdLostGhosts does for every track (spec 2026-10-27 §8: targets leaving the
  // frame). Filled-in boxes do not count. nullopt: off.
  std::optional<float> holdLostMinScore{};
```
私有 `Track` 末尾加 `float bestScore = 0.0f;  // the highest score of its real detections`；私有方法加：
```cpp
  // Whether the ghost `track`, lost again, is held for one more interval: holdLostGhosts, or its best real
  // detection reaches holdLostMinScore.
  bool holdsLostGhost(const Track& track) const;
```
类注释里讲 holdLostGhosts 的那句后面加：`holdLostMinScore does the same for the tracks whose best real detection scored at least that much.`

`box_timeline.cpp`：
- `validConfig` 的返回值再与上 `(!c.holdLostMinScore || (*c.holdLostMinScore >= 0.0f && *c.holdLostMinScore <= 1.0f))`（NaN 两个比较都不成立，被拒）。
- `startTrack` 末尾（`return t;` 之前）加 `t.bestScore = t.kind == LinkKind::Real ? box.score : 0.0f;`。
- `continueTrack` 在 `if (config_.linked) {...}` 之后加 `t.bestScore = std::max(old.bestScore, t.kind == LinkKind::Real ? box.score : 0.0f);`。
- 建幽灵处加 `ghost.bestScore = track.bestScore;`。
- 新增：
```cpp
bool BoxTimeline::holdsLostGhost(const Track& track) const {
  return config_.holdLostGhosts || (config_.holdLostMinScore && track.bestScore >= *config_.holdLostMinScore);
}
```
- `advance()` 和 `finish()` 里两处 `track.ghost && config_.holdLostGhosts` 改为 `track.ghost && holdsLostGhost(track)`。

- [ ] **Step 4: 配置**

`redact_config.h`：
- `TimelineConfig timeline{...}` 的指定初始化末尾加 `.holdLostMinScore = 0.5f`（字段按声明顺序，它在 `linked` 之后）。
- `trackLookaheadKeyframes` 改为 `2`、`trackBackfillKeyframes` 改为 `1`，上方注释末尾加一句：`The defaults are the cheapest that fixed the bridgeable leaks of the pilot (spec 2026-10-27 §8): W 5, B 2 cost 32 % more unexplained mask area on sg_drive_60s, W 2, B 1 12 %.`
- 加注释说明 `timeline.holdLostMinScore`：`// hold_lost_min_score (spec 2026-10-27 §8): a confirmed track whose best real detection scored at least this is held one more interval when lost, e.g. a plate leaving the frame; null switches it off.`
- `parseRedactConfig` 的规则注释里加 `hold_lost_min_score is a number in [0, 1] or null (off);`。

`redact_config.cpp`：
- `kKeys` 改为 32 个，加 `"hold_lost_min_score"`。
- `validateRedactConfig` 加：
```cpp
  require(!c.timeline.holdLostMinScore ||
              (*c.timeline.holdLostMinScore >= 0 && *c.timeline.holdLostMinScore <= 1),
          "hold_lost_min_score must be in [0, 1] or null");
```
- `parseRedactConfig` 里 `hold_lost_ghosts` 那行之后加（缺省保留默认值，null 关闭）：
```cpp
  if (j.contains("hold_lost_min_score")) {
    c.timeline.holdLostMinScore = optionalFloatSetting(j, "hold_lost_min_score", origin);
  }
```
- `redactConfigToJson` 里 `hold_lost_ghosts` 之后加：
```cpp
  j["hold_lost_min_score"] = c.timeline.holdLostMinScore ? nlohmann::ordered_json(jsonFloat(*c.timeline.holdLostMinScore))
                                                         : nlohmann::ordered_json(nullptr);
```

`configs/redact.json`：`"hold_lost_ghosts": false,` 之后加 `"hold_lost_min_score": 0.5,`；`track_lookahead_keyframes` 改为 2、`track_backfill_keyframes` 改为 1。

`docs/USAGE.md`：`track_lookahead_keyframes` 一行的默认值改为 2，"（10 档时 5 个约 0.5 秒）"改为"（10 档时 2 个约 0.2 秒）"，"60 fps 1080p、5 个约 110 MB"改为"60 fps 1080p、2 个约 55 MB"，并加一句"默认 2、`track_backfill_keyframes` 1：`sg_drive_60s` 试跑中能连上的泄露都补上了，误打面积多 12%；5 / 2 多 32%（设计 2026-10-27 §8）"；`track_backfill_keyframes` 一行默认值改为 1。在 `hold_lost_ghosts` 一行之后加：
```markdown
| `hold_lost_min_score` | 0.5 | 已确认的轨迹丢失后，若它真实检出里的最高分不低于这个值，就像 `hold_lost_ghosts` 那样在最后停留的位置再多打一个间隔的码，然后结束（主要为开出画面、之后再没检出的车牌，设计 2026-10-27 §8）。补出来的框的分数不算。[0, 1] 内的数，`null` 关闭；与 `hold_lost_ghosts` 是"或"的关系 |
```

- [ ] **Step 5: 运行测试**

```bash
cmake --build build/cpp -j 4 --target vc_tests
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:BoxMatching.*:KeyframeLinker.*:TrackMasker.*:RedactConfig.*:Records.*'
./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*'
```
Expected: 全部通过（`baseOptions` 把 `timeline` 设成 `TimelineConfig{}`，其中 `holdLostMinScore` 为空，所以原有 RedactJob 测试不受影响）。

- [ ] **Step 6: 提交**

```bash
git add cpp/src/track/box_timeline.h cpp/src/track/box_timeline.cpp cpp/apps/vc_redact/redact_config.h cpp/apps/vc_redact/redact_config.cpp cpp/tests/box_timeline_test.cpp cpp/tests/redact_config_test.cpp configs/redact.json docs/USAGE.md
git commit -m "feat: hold_lost_min_score holds high-score lost tracks one more interval; lookahead defaults 2 and 1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
