#include "vc_redact/redact_job.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
  // The pixel windows and keyframe counts in these tests were worked out for 5 fps and the
  // lower-level margin and score defaults, not for vc_redact's validated defaults.
  o.config.detectFps = 5.0;
  o.config.decode = DecodeParams{};
  o.config.timeline = TimelineConfig{};
  // Without lookahead: these tests were written for BoxTimeline alone (spec 2026-10-27 has its own below).
  o.config.trackLookaheadKeyframes = 0;
  o.config.trackBackfillKeyframes = 0;
  return o;
}

// 2 s, 320x240 @ 25 fps (50 frames): a frame rate the 60 fps detection rates do not divide.
fs::path video25fps() {
  return test::makeMedia("basic_25fps.mp4", "-f lavfi -i testsrc2=s=320x240:r=25:d=2 -c:v libx264 -pix_fmt yuv420p -g 25");
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

// Fraction of neighbouring pixel pairs (left-right and top-bottom) inside the pixel box [x1, x2) x
// [y1, y2) whose luma differs by more than 12 levels: one half on test::checkerboardVideo(), close to
// none inside a mosaic (its blocks all average the checkerboard to the same grey).
double roughness(const AVFrame& f, int x1, int y1, int x2, int y2) {
  int pairs = 0;
  int rough = 0;
  const auto at = [&](int x, int y) { return static_cast<int>(f.data[0][y * f.linesize[0] + x]); };
  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) {
      if (x + 1 < x2) {
        ++pairs;
        rough += std::abs(at(x + 1, y) - at(x, y)) > 12 ? 1 : 0;
      }
      if (y + 1 < y2) {
        ++pairs;
        rough += std::abs(at(x, y + 1) - at(x, y)) > 12 ? 1 : 0;
      }
    }
  }
  return pairs == 0 ? 0.0 : static_cast<double>(rough) / pairs;
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
  ASSERT_EQ(in.size(), 60u);
  // The mask is the fake box grown by 15 % per side, (85, 88)-(215, 192); its long side asks for ceil(130 / 8) = 17 px
  // blocks, the ladder's 24, on the frame's grid: the cell [96, 120) x [96, 120) is flat in the output and noisy in
  // the input, on every frame.
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_LT(lumaStd(*out[i], 96, 96, 120, 120), 4.0) << "frame " << i;
    EXPECT_GT(lumaStd(*in[i], 96, 96, 120, 120), 15.0) << "frame " << i;
    // Far from the mask the picture is kept: same brightness, still detailed.
    EXPECT_NEAR(lumaMean(*out[i], 400, 220, 560, 330), lumaMean(*in[i], 400, 220, 560, 330), 3.0) << "frame " << i;
    EXPECT_GT(lumaStd(*out[i], 400, 220, 560, 330), 10.0) << "frame " << i;
  }

  // Every mask records the block it was pixelated with; one static target keeps one block.
  std::ifstream masksFile(dir / "masks.jsonl");
  for (std::string line; std::getline(masksFile, line);) {
    for (const auto& box : nlohmann::json::parse(line).at("boxes")) {
      EXPECT_EQ(box.at("block").get<int>(), 24) << line;
      EXPECT_TRUE(box.at("snap").get<bool>()) << line;
    }
  }
  // A face's mosaic snaps out to the grid: the rows [72, 88) above the mask are pixelated too.
  for (size_t i = 0; i < out.size(); ++i) EXPECT_LT(lumaStd(*out[i], 96, 74, 120, 86), 4.0) << "frame " << i;
}

// A plate's mosaic stays inside its mask (spec 2026-10-19 §4, plates): the same box as above as a plate, mask
// (85, 88)-(215, 192) in 24 px cells, leaves the rows [72, 88) that the face's mosaic snaps out to as they were.
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
  size_t lines = 0;
  for (std::string line; std::getline(det, line); ++lines) {
    const auto j = nlohmann::json::parse(line);
    EXPECT_FALSE(j.at("verified").get<bool>()) << line;
    EXPECT_FALSE(j.at("confirmed").get<bool>()) << line;
    EXPECT_NEAR(j.at("verify").get<double>(), 0.1192, 1e-4) << line;
  }
  EXPECT_EQ(lines, static_cast<size_t>(s.keyframes));
  std::ifstream statsFile(o.outRoot / kStem / "stats.json");
  const auto stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("verifier"), "const_verifier");
  EXPECT_EQ(stats.at("verify").at("rejected").at("face").get<int64_t>(), s.keyframes);
  EXPECT_EQ(stats.at("stages").at("verify").at("items").get<int64_t>(), s.keyframes);
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

TEST(RedactJob, PixelatesAPlateOnlyInsideItsMask) {
  RedactOptions o = baseOptions("redact_plate_clipped");
  o.fakeDetections = fakeFile("redact_plate_clipped.jsonl", 60, "plate");
  runRedact(o);
  const fs::path dir = o.outRoot / kStem;
  const auto in = decodeAll(o.input);
  const auto out = decodeAll(dir / "noisy_audio.redacted.mp4");
  ASSERT_EQ(out.size(), 60u);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_LT(lumaStd(*out[i], 96, 96, 120, 120), 4.0) << "frame " << i;
    EXPECT_GT(lumaStd(*out[i], 96, 74, 200, 86), 10.0) << "frame " << i;  // outside the mask: still detailed
    EXPECT_NEAR(lumaMean(*out[i], 96, 74, 200, 86), lumaMean(*in[i], 96, 74, 200, 86), 3.0) << "frame " << i;
  }
  std::ifstream masksFile(dir / "masks.jsonl");
  for (std::string line; std::getline(masksFile, line);) {
    for (const auto& box : nlohmann::json::parse(line).at("boxes")) EXPECT_FALSE(box.at("snap").get<bool>()) << line;
  }
}

// A face and a plate moving 2.5 and 2 box widths per keyframe interval (200 ms = 6 frames), turning
// round at frame 30 and coming back. The fake detections list every frame (the pipeline asks only
// for the keyframes), so they are also the ground truth: the box is pixelated on every output frame up to the
// last keyframe.
TEST(RedactJob, MasksTargetsThatMoveSeveralBoxWidthsBetweenKeyframes) {
  struct Target {
    const char* cls;
    double x0;
    double y0;
    double w;
    double h;
    double vx;  // px per frame
  };
  const std::vector<Target> targets = {{"face", 100, 60, 24, 24, 2.5 * 24 / 6},
                                       {"plate", 560, 250, 40, 16, -2.0 * 40 / 6}};
  // Integer pixel bounds of the target on frame f (out to the next pixel edge).
  struct Rect {
    int x1, y1, x2, y2;
  };
  const auto rectAt = [](const Target& t, int f) {
    const double x = t.x0 + t.vx * (f <= 30 ? f : 60 - f);
    return Rect{static_cast<int>(std::floor(x)), static_cast<int>(t.y0), static_cast<int>(std::ceil(x + t.w)),
                static_cast<int>(t.y0 + t.h)};
  };

  RedactOptions o = baseOptions("redact_fast_movers");
  o.input = test::checkerboardVideo();
  const fs::path fake = test::tmpDir() / "redact_fast_movers.jsonl";
  {
    std::ofstream out(fake);
    for (int f = 0; f < 60; ++f) {
      for (const Target& t : targets) {
        const Rect r = rectAt(t, f);
        out << R"({"video":"checkerboard.mp4","frame":)" << f << R"(,"pts_ms":0,"cls":")" << t.cls << R"(","x1":)"
            << r.x1 << R"(,"y1":)" << r.y1 << R"(,"x2":)" << r.x2 << R"(,"y2":)" << r.y2
            << R"(,"score":0.9,"src":"fake"})" << '\n';
      }
    }
  }
  o.fakeDetections = fake;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 10);
  EXPECT_EQ(s.maskedFrames, 60);

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

// ---- Large faces need a higher score (spec 2026-10-19 §3.6). The video is 640 x 360 (230,400 px), keyframes every
// 6 frames; every box is detected on keyframe 12 only, so a kept box is masked on frames 12 to 17 (detected, then
// held).

struct FakeBox {
  const char* cls;
  double x1;
  double y1;
  double x2;
  double y2;
  double score;
};

// Runs the job over fake detections that hold `boxes` on frame 12 and nothing else.
RedactStats runWithBoxesOnOneKeyframe(const std::string& name, const std::vector<FakeBox>& boxes,
                                      const std::function<void(RedactConfig&)>& configure = {}) {
  RedactOptions o = baseOptions(name);
  if (configure) configure(o.config);
  const fs::path fake = test::tmpDir() / (name + ".jsonl");
  {
    std::ofstream out(fake);
    for (const FakeBox& b : boxes) {
      out << R"({"video":"noisy_audio.mp4","frame":12,"pts_ms":0,"cls":")" << b.cls << R"(","x1":)" << b.x1
          << R"(,"y1":)" << b.y1 << R"(,"x2":)" << b.x2 << R"(,"y2":)" << b.y2 << R"(,"score":)" << b.score
          << R"(,"src":"fake"})" << '\n';
    }
  }
  o.fakeDetections = fake;
  return runRedact(o);
}

nlohmann::json statsJson(const RedactStats& s) {
  std::ifstream in(s.outputDir / "stats.json");
  return nlohmann::json::parse(in);
}

// The cls and frame of every detections.jsonl line.
std::vector<nlohmann::json> detectionLines(const RedactStats& s) {
  std::vector<nlohmann::json> out;
  std::ifstream in(s.outputDir / "detections.jsonl");
  for (std::string line; std::getline(in, line);) out.push_back(nlohmann::json::parse(line));
  return out;
}

size_t maskedBoxCount(const RedactStats& s) {
  size_t n = 0;
  std::ifstream in(s.outputDir / "masks.jsonl");
  for (std::string line; std::getline(in, line);) n += nlohmann::json::parse(line).at("boxes").size();
  return n;
}

// A face of 30 % of the frame (320 x 216 px of 640 x 360) scoring 0.5 is a giant false positive: dropped before the
// timeline, so it is not masked, not in detections.jsonl and not in stats.detections; it is counted.
TEST(RedactJob, DropsALargeFaceBelowTheMinimumScore) {
  const RedactStats s = runWithBoxesOnOneKeyframe("redact_large_face_dropped", {{"face", 100, 50, 420, 266, 0.5}});
  EXPECT_EQ(s.largeFacesDropped, 1);
  EXPECT_EQ(s.detections, 0);
  EXPECT_EQ(s.maskedFrames, 0);
  EXPECT_EQ(s.maskBoxes, 0);
  EXPECT_EQ(maskedBoxCount(s), 0u);
  EXPECT_TRUE(detectionLines(s).empty());
  const nlohmann::json stats = statsJson(s);
  EXPECT_EQ(stats.at("large_faces_dropped").get<int>(), 1);
  EXPECT_EQ(stats.at("detections").get<int>(), 0);
  EXPECT_DOUBLE_EQ(stats.at("config").at("large_face_area_fraction").get<double>(), 0.05);
  EXPECT_DOUBLE_EQ(stats.at("config").at("large_face_min_score").get<double>(), 0.6);
}

// The same face scoring 0.7, or exactly the minimum (below is strict), is kept and masked.
TEST(RedactJob, KeepsALargeFaceAtOrAboveTheMinimumScore) {
  for (const double score : {0.7, 0.6}) {
    SCOPED_TRACE("score " + std::to_string(score));
    const RedactStats s = runWithBoxesOnOneKeyframe("redact_large_face_kept", {{"face", 100, 50, 420, 266, score}});
    EXPECT_EQ(s.largeFacesDropped, 0);
    EXPECT_EQ(s.detections, 1);
    EXPECT_EQ(s.maskedFrames, 6);
    EXPECT_EQ(detectionLines(s).size(), 1u);
    EXPECT_EQ(statsJson(s).at("large_faces_dropped").get<int>(), 0);
  }
}

// Only large faces are held to the higher score: a small face at 0.5 and a plate of 30 % of the frame at 0.5 stay.
TEST(RedactJob, KeepsASmallFaceAndALargePlateAtScoreHalf) {
  const RedactStats s = runWithBoxesOnOneKeyframe(
      "redact_large_face_other_boxes", {{"face", 100, 100, 200, 180, 0.5}, {"plate", 300, 50, 620, 266, 0.5}});
  EXPECT_EQ(s.largeFacesDropped, 0);
  EXPECT_EQ(s.detections, 2);
  EXPECT_EQ(s.maskedFrames, 6);
  EXPECT_EQ(s.maskBoxes, 12);
  const std::vector<nlohmann::json> lines = detectionLines(s);
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0].at("cls").get<std::string>(), "face");
  EXPECT_EQ(lines[1].at("cls").get<std::string>(), "plate");
}

// "Exceeds" is strict, and the area is the box's own: at a quarter of the frame, 320 x 180 is kept and 321 x 180 is
// dropped.
TEST(RedactJob, ABoxExactlyAtTheAreaLimitIsKept) {
  const RedactStats s = runWithBoxesOnOneKeyframe(
      "redact_large_face_limit", {{"face", 0, 0, 320, 180, 0.5}, {"face", 0, 180, 321, 360, 0.5}},
      [](RedactConfig& c) { c.largeFaceAreaFraction = 0.25; });
  EXPECT_EQ(s.largeFacesDropped, 1);
  EXPECT_EQ(s.detections, 1);
  const std::vector<nlohmann::json> lines = detectionLines(s);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].at("x2").get<int>(), 320);
  EXPECT_EQ(lines[0].at("y1").get<int>(), 0);
  EXPECT_EQ(s.maskedFrames, 6);
}

// large_face_min_score 0 switches the rule off; so does a bigger area fraction.
TEST(RedactJob, TheLargeFaceRuleCanBeSwitchedOff) {
  const std::vector<FakeBox> giant = {{"face", 100, 50, 420, 266, 0.31}};
  const RedactStats off =
      runWithBoxesOnOneKeyframe("redact_large_face_off", giant, [](RedactConfig& c) {
        c.largeFaceMinScore = 0.0f;
        c.faceConfirmScore = 0.0f;
      });
  EXPECT_EQ(off.largeFacesDropped, 0);
  EXPECT_EQ(off.detections, 1);
  EXPECT_EQ(off.maskedFrames, 6);
  const RedactStats wide = runWithBoxesOnOneKeyframe("redact_large_face_wide", giant, [](RedactConfig& c) {
    c.largeFaceAreaFraction = 1.0;
    c.faceConfirmScore = 0.0f;
  });
  EXPECT_EQ(wide.largeFacesDropped, 0);
  EXPECT_EQ(wide.maskedFrames, 6);
}

// With plate_min_width_px 24, a plate 23.5 px wide is dropped before the timeline like a giant face (not masked, not in
// detections.jsonl, counted); one exactly 24 px wide and a face 10 px wide stay.
TEST(RedactJob, DropsPlatesNarrowerThanTheMinimumWidth) {
  const RedactStats s = runWithBoxesOnOneKeyframe(
      "redact_narrow_plate",
      {{"plate", 100, 100, 123.5, 108, 0.9}, {"plate", 300, 100, 324, 108, 0.9}, {"face", 500, 100, 510, 112, 0.9}},
      [](RedactConfig& c) { c.plateMinWidthPx = 24.0; });
  EXPECT_EQ(s.narrowPlatesDropped, 1);
  EXPECT_EQ(s.detections, 2);
  const std::vector<nlohmann::json> lines = detectionLines(s);
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0].at("x1").get<int>(), 300);
  EXPECT_EQ(lines[1].at("cls").get<std::string>(), "face");
  EXPECT_EQ(s.maskBoxes, 12);
  const nlohmann::json stats = statsJson(s);
  EXPECT_EQ(stats.at("narrow_plates_dropped").get<int>(), 1);
  EXPECT_DOUBLE_EQ(stats.at("config").at("plate_min_width_px").get<double>(), 24.0);
}

// plate_min_width_px 0 masks plates of any width.
TEST(RedactJob, ZeroPlateMinimumWidthKeepsNarrowPlates) {
  const RedactStats s = runWithBoxesOnOneKeyframe("redact_narrow_plate_off", {{"plate", 100, 100, 110, 108, 0.9}},
                                                  [](RedactConfig& c) { c.plateMinWidthPx = 0.0; });
  EXPECT_EQ(s.narrowPlatesDropped, 0);
  EXPECT_EQ(s.detections, 1);
  EXPECT_EQ(s.maskedFrames, 6);
  EXPECT_EQ(statsJson(s).at("narrow_plates_dropped").get<int>(), 0);
}

TEST(RedactJob, TheSummaryCountsTheLargeFacesDropped) {
  RedactStats s;
  s.frames = 600;
  s.keyframes = 100;
  s.detections = 250;
  s.largeFacesDropped = 3;
  s.maskedFrames = 400;
  s.maskBoxes = 520;
  s.narrowPlatesDropped = 7;
  const std::string summary = redactSummary(s);
  EXPECT_NE(summary.find("frames:        600 (keyframes 100, detections 250, large faces dropped 3, "
                         "narrow plates dropped 7, masked 400 frames / 520 boxes)\n"),
            std::string::npos)
      << summary;
  s.largeFacesDropped = 0;
  EXPECT_NE(redactSummary(s).find("large faces dropped 0,"), std::string::npos);  // shown even when 0
}

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

TEST(RedactJob, DetectEveryFrameMakesEveryFrameAKeyframe) {
  RedactOptions o = baseOptions("redact_every");
  o.detectEveryFrame = true;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.keyframes, 60);
  EXPECT_EQ(s.detections, 60);
  EXPECT_EQ(s.keyframeStride, 1);
  EXPECT_NEAR(s.detectFpsActual, 30.0, 1e-6);
}

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

// A frame rate above 1e6 fps in the header is a broken stream, not a video: the run is refused with a message that
// names the file and the rate (detectSchedule's std::invalid_argument does not), before anything is created.
TEST(RedactJob, AnImplausibleFrameRateIsRefusedNamingTheVideo) {
  RedactOptions o = baseOptions("redact_huge_fps");
  // 100 frames half a microsecond apart: avg_frame_rate 400000000/201, about 1.99 million fps.
  o.input = test::makeMedia("huge_fps.mp4",
                            "-f lavfi -i \"testsrc2=s=64x64:r=1000000:d=0.0001,setpts=PTS/2\" -fps_mode passthrough "
                            "-c:v libx264 -pix_fmt yuv420p -video_track_timescale 4000000");
  try {
    runRedact(o);
    ADD_FAILURE() << "accepted a 1.99 million fps video";
  } catch (const VcError& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("huge_fps.mp4"), std::string::npos) << what;
    EXPECT_NE(what.find("(0, 1e6]"), std::string::npos) << what;
    EXPECT_NE(what.find("1.99005e+06"), std::string::npos) << what;
  }
  EXPECT_FALSE(fs::exists(o.outRoot));
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

// A failed run leaves nothing behind, not even the empty directory, so a retry needs no --overwrite.
TEST(RedactJob, AFailedRunCanBeRetriedWithoutOverwriting) {
  RedactOptions o = baseOptions("redact_retry");
  const fs::path good = *o.fakeDetections;
  const fs::path failing = test::tmpDir() / "redact_retry_fail.jsonl";
  std::ofstream(failing) << R"({"frame":10,"fail":true})" << '\n';
  o.fakeDetections = failing;
  o.detectEveryFrame = true;
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(o.outRoot / kStem));
  o.fakeDetections = good;
  EXPECT_NO_THROW(runRedact(o));
  EXPECT_TRUE(fs::exists(o.outRoot / kStem / "stats.json"));
}

TEST(RedactJob, KeepsTheTimingOfVideoAndAudio) {
  const RedactOptions o = baseOptions("redact_timing");
  const RedactStats s = runRedact(o);
  EXPECT_NEAR(test::streamDurationSec(s.video, AVMEDIA_TYPE_VIDEO), test::streamDurationSec(o.input, AVMEDIA_TYPE_VIDEO), 0.05);
  EXPECT_NEAR(test::streamDurationSec(s.video, AVMEDIA_TYPE_AUDIO), test::streamDurationSec(o.input, AVMEDIA_TYPE_AUDIO), 0.05);
  // Each stream starts where it did in the input, so the audio / video offset is unchanged.
  for (const AVMediaType type : {AVMEDIA_TYPE_VIDEO, AVMEDIA_TYPE_AUDIO}) {
    const test::StreamTiming in = test::streamTiming(o.input, type);
    const test::StreamTiming out = test::streamTiming(s.video, type);
    EXPECT_NEAR(out.startSec, in.startSec, 1.0 / 30) << av_get_media_type_string(type);
    EXPECT_NEAR(out.firstPtsSec, in.firstPtsSec, 1.0 / 30) << av_get_media_type_string(type);
    EXPECT_NEAR(out.lastPtsSec, in.lastPtsSec, 1.0 / 30) << av_get_media_type_string(type);
  }
  const double sourceOffset = test::streamTiming(o.input, AVMEDIA_TYPE_AUDIO).startSec - test::streamTiming(o.input, AVMEDIA_TYPE_VIDEO).startSec;
  const double outputOffset = test::streamTiming(s.video, AVMEDIA_TYPE_AUDIO).startSec - test::streamTiming(s.video, AVMEDIA_TYPE_VIDEO).startSec;
  EXPECT_NEAR(outputOffset, sourceOffset, 1.0 / 30);
}

// The picture is shown like the input: same pixel aspect ratio, colour tags and rotation.
TEST(RedactJob, KeepsThePresentationOfTheInput) {
  struct Case {
    const char* name;
    fs::path input;
  };
  for (const Case& c : {Case{"redact_present_sar", test::anamorphicVideo()}, Case{"redact_present_rot", test::rotatedVideo()},
                        Case{"redact_present_709", test::bt709Video()}, Case{"redact_present_full", test::bt601FullRangeVideo()}}) {
    RedactOptions o = baseOptions(c.name);
    o.input = c.input;
    const RedactStats s = runRedact(o);
    const test::VideoPresentation in = test::videoPresentation(c.input);
    const test::VideoPresentation out = test::videoPresentation(s.video);
    EXPECT_EQ(out.sampleAspectRatio.num * in.sampleAspectRatio.den, in.sampleAspectRatio.num * out.sampleAspectRatio.den) << c.name;
    EXPECT_EQ(out.hasDisplayMatrix, in.hasDisplayMatrix) << c.name;
    EXPECT_EQ(out.displayMatrix, in.displayMatrix) << c.name;
    EXPECT_EQ(out.colorRange, in.colorRange) << c.name;
    EXPECT_EQ(out.colorSpace, in.colorSpace) << c.name;
    EXPECT_EQ(out.colorPrimaries, in.colorPrimaries) << c.name;
    EXPECT_EQ(out.colorTrc, in.colorTrc) << c.name;
  }
}

// With a face model, every keyframe runs both detectors. On a cyan picture the constant YuNet model
// (cpp/tests/data/const_yunet) finds every full 32x32 cell (640x360 pads to 640x384: 20 x 11 cells
// lie fully inside); the fake face is exactly one of them, so that one is a single box found by both.
TEST(RedactJob, AFaceModelAddsItsBoxesAndMergesTheOnesBothDetectorsFind) {
  RedactOptions o = baseOptions("redact_face_model");
  o.input = test::makeMedia("cyan.mp4", "-f lavfi -i color=c=cyan:s=640x360:r=30:d=2 -c:v libx264 -pix_fmt yuv420p");
  {
    std::ofstream fake(*o.fakeDetections);
    for (int i = 0; i < 60; ++i) {
      fake << R"({"video":"cyan.mp4","frame":)" << i
           << R"(,"pts_ms":0,"cls":"face","x1":96,"y1":96,"x2":128,"y2":128,"score":0.5,"src":"fake"})" << '\n';
    }
  }
  o.faceModelDir = fs::path(VC_TEST_DATA_DIR) / "const_yunet";
  o.provider = ExecutionProvider::Cpu;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.detector, "fake + yunet/CPU");
  EXPECT_EQ(s.detections, s.keyframes * 220);

  std::ifstream in(s.outputDir / "detections.jsonl");
  std::map<int64_t, std::map<std::string, int>> sources;  // frame -> src -> boxes
  for (std::string line; std::getline(in, line);) {
    const nlohmann::json d = nlohmann::json::parse(line);
    const std::string src = d.at("src").get<std::string>();
    ++sources[d.at("frame").get<int64_t>()][src];
    EXPECT_EQ(d.at("cls").get<std::string>(), "face");
    if (src == "fake+yunet") {
      EXPECT_DOUBLE_EQ(d.at("x1").get<double>(), 96);
      EXPECT_DOUBLE_EQ(d.at("y2").get<double>(), 128);
      EXPECT_GT(d.at("score").get<double>(), 0.9);  // the higher of the two scores
    }
  }
  EXPECT_EQ(static_cast<int64_t>(sources.size()), s.keyframes);
  for (const auto& [frame, counts] : sources) {
    EXPECT_EQ(counts, (std::map<std::string, int>{{"fake+yunet", 1}, {"yunet", 219}})) << "frame " << frame;
  }

  // The YuNet boxes are masked, not just detected: on every frame each of the 220 full cells (the
  // 20 x 11 grid of 32x32 cells) lies inside one mask, and nothing but faces is masked.
  std::ifstream masks(s.outputDir / "masks.jsonl");
  int maskedFrames = 0;
  for (std::string line; std::getline(masks, line);) {
    const nlohmann::json record = nlohmann::json::parse(line);
    const int frame = record.at("frame").get<int>();
    ++maskedFrames;
    for (const auto& box : record.at("boxes")) EXPECT_EQ(box.at("cls").get<std::string>(), "face") << "frame " << frame;
    for (int row = 0; row < 11; ++row) {
      for (int col = 0; col < 20; ++col) {
        bool covered = false;
        for (const auto& box : record.at("boxes")) {
          covered = covered || (box.at("x1").get<double>() <= 32 * col + 0.1 && box.at("y1").get<double>() <= 32 * row + 0.1 &&
                                box.at("x2").get<double>() >= 32 * (col + 1) - 0.1 &&
                                box.at("y2").get<double>() >= 32 * (row + 1) - 0.1);
        }
        EXPECT_TRUE(covered) << "frame " << frame << " cell row " << row << " col " << col;
      }
    }
  }
  EXPECT_EQ(maskedFrames, 60);
  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("detector").get<std::string>(), "fake + yunet/CPU");
  // Both detectors' own busy times, one item per keyframe each; detect is the wall time of the two together.
  EXPECT_EQ(s.detectStudent.items, s.keyframes);
  EXPECT_EQ(s.detectFace.items, s.keyframes);
  EXPECT_EQ(stats.at("stages").at("detect_student").at("items").get<int64_t>(), s.keyframes);
  EXPECT_EQ(stats.at("stages").at("detect_face").at("items").get<int64_t>(), s.keyframes);
  EXPECT_GT(s.detectFace.busySec, 0.0);
}

TEST(RedactJob, ChecksTheFaceModelBeforeCreatingAnything) {
  RedactOptions o = baseOptions("redact_bad_face_model");
  o.faceModelDir = fs::path(VC_TEST_DATA_DIR) / "const_yolo";  // a student model, not YuNet
  o.provider = ExecutionProvider::Cpu;
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(o.outRoot));
}

TEST(RedactJob, RejectsAnInvalidConfigBeforeCreatingAnything) {
  RedactOptions o = baseOptions("redact_badconfig");
  o.config.detectFps = 1.0;  // below the 5 fps floor
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(o.outRoot));
}

TEST(RedactJob, ChecksTheDetectorBeforeCreatingAnything) {
  RedactOptions o = baseOptions("redact_baddetector");
  o.fakeDetections = test::tmpDir() / "redact_does_not_exist.jsonl";
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(o.outRoot));
}

TEST(RedactJob, DoesNotDeleteAnInputInsideTheOutputDirectory) {
  RedactOptions o = baseOptions("redact_contains");
  const fs::path dir = o.outRoot / kStem;
  fs::create_directories(dir);
  std::ofstream(dir / "masks.jsonl") << "{}\n";  // a replaceable previous output ...
  o.input = dir / "noisy_audio.mp4";             // ... that holds the input
  fs::copy_file(test::noisyVideoWithAudio(), o.input);
  o.overwrite = true;
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_TRUE(fs::exists(o.input));
}

TEST(RedactJob, WorksWithoutAudio) {
  RedactOptions o = baseOptions("redact_noaudio");
  o.input = test::basicVideo();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 60);
  EXPECT_EQ(s.audioStreams, 0u);
  EXPECT_EQ(s.audioPackets, 0);
  EXPECT_EQ(test::streamCount(s.video, AVMEDIA_TYPE_AUDIO), 0);
  EXPECT_EQ(lineCount(s.outputDir / "masks.jsonl"), 60u);
}

TEST(RedactJob, ReportsAndDropsNonMediaStreams) {
  RedactOptions o = baseOptions("redact_subs");
  o.input = test::videoWithSubtitles();
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.droppedStreams, (std::vector<std::string>{"#1 subtitle mov_text"}));
  EXPECT_EQ(test::streamCount(s.video, AVMEDIA_TYPE_SUBTITLE), 0);
  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("dropped_streams").size(), 1u);
}

// 500 frames at 1000 fps: 5 fps is a 200-frame stride, which would leave 200 frames buffered, so the
// 120-frame limit has to force keyframes (at 120 and 320).
TEST(RedactJob, ForcesAKeyframeWhenTheBufferGrowsTooLong) {
  RedactOptions o = baseOptions("redact_buffer");
  o.input = test::makeMedia("fast_1000fps.mp4",
                            "-f lavfi -i testsrc2=s=160x90:r=1000:d=0.5 -c:v libx264 -pix_fmt yuv420p -g 100");
  const fs::path fake = test::tmpDir() / "redact_buffer.jsonl";
  {
    std::ofstream out(fake);
    for (int i = 0; i < 500; ++i) {
      out << R"({"video":"fast_1000fps.mp4","frame":)" << i
          << R"(,"pts_ms":0,"cls":"face","x1":20,"y1":20,"x2":60,"y2":50,"score":0.9,"src":"fake"})" << '\n';
    }
  }
  o.fakeDetections = fake;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.frames, 500);
  EXPECT_EQ(s.maskedFrames, 500);
  EXPECT_GE(s.keyframes, 5);  // frames 0, 120, 200, 320, 400; the stride alone would pick 0, 200, 400
  EXPECT_EQ(lineCount(s.outputDir / "masks.jsonl"), 500u);
}

// A detector that takes 40 ms per frame keeps the processing stage busy and fills the queue in front
// of it: the decode stage then waits for room, and that waiting is not decoding time (it used to
// count when it happened while the reader was delivering an audio packet).
TEST(RedactJob, DecodeTimingExcludesWaitingForASlowConsumer) {
  RedactOptions o = baseOptions("redact_slow");
  const fs::path slow = test::tmpDir() / "redact_slow.jsonl";
  {
    std::ofstream out(slow);
    for (int i = 0; i < 60; ++i) {
      out << R"({"frame":)" << i << R"(,"delay_ms":40})" << '\n';
      out << R"({"video":"noisy_audio.mp4","frame":)" << i
          << R"(,"pts_ms":0,"cls":"face","x1":100,"y1":100,"x2":200,"y2":180,"score":0.9,"src":"fake"})" << '\n';
    }
  }
  o.fakeDetections = slow;
  o.detectEveryFrame = true;
  const RedactStats s = runRedact(o);
  ASSERT_GT(s.audioPackets, 80);
  ASSERT_GE(s.detect.busySec, 2.3);  // 60 x 40 ms
  // Decoding 60 frames takes ~0.03 s; with the waiting counted it came to ~0.3 x the detect time.
  EXPECT_LT(s.decode.busySec, 0.15 * s.detect.busySec) << "decode busy " << s.decode.busySec << " s, detect busy "
                                                      << s.detect.busySec << " s";
}

// Failures on the decode and the encode thread: the run fails with the stage's own message and
// leaves nothing behind, not even the output root this run created.
TEST(RedactJob, ADecodeThreadFailureFailsTheRunAndLeavesNothing) {
  RedactOptions o = baseOptions("redact_decode_fail");
  o.input = test::rawH264Stream();  // frames without timestamps
  try {
    runRedact(o);
    FAIL() << "expected VcError";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("has no timestamp"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(o.outRoot / "raw"));
  EXPECT_FALSE(fs::exists(o.outRoot));
}

TEST(RedactJob, AnEncodeThreadFailureFailsTheRunAndLeavesNothing) {
  RedactOptions o = baseOptions("redact_encode_fail");
  o.input = test::resolutionChangeVideo();  // the writer refuses the first frame of the new size
  try {
    runRedact(o);
    FAIL() << "expected VcError";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("frame is 160x120, the writer was set up for 320x240"), std::string::npos)
        << e.what();
  }
  EXPECT_FALSE(fs::exists(o.outRoot / "size_change"));
  EXPECT_FALSE(fs::exists(o.outRoot));
}

// Audio that MP4 cannot store is refused before the output is touched: a previous output that
// --overwrite would replace is still there afterwards.
TEST(RedactJob, RefusesAnAudioCodecThatMp4CannotStoreBeforeTouchingTheOutput) {
  RedactOptions o = baseOptions("redact_mulaw");
  o.input = test::videoWithMulawAudio();
  const fs::path previous = o.outRoot / "video_mulaw";
  fs::create_directories(previous);
  std::ofstream(previous / "masks.jsonl") << "{}\n";
  o.overwrite = true;
  try {
    runRedact(o);
    FAIL() << "expected VcError";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("audio codec pcm_mulaw cannot be stored in MP4"), std::string::npos) << e.what();
  }
  EXPECT_TRUE(fs::exists(previous / "masks.jsonl"));
}

// The output directory is checked before the detector is opened (loading a model can take long).
TEST(RedactJob, ChecksTheOutputDirectoryBeforeOpeningTheDetector) {
  RedactOptions o = baseOptions("redact_exists_first");
  fs::create_directories(o.outRoot / kStem);
  o.fakeDetections = test::tmpDir() / "redact_exists_first_missing.jsonl";  // would fail to open
  EXPECT_THROW(runRedact(o), OutputExistsError);
}

// Loading the detector (CoreML compiles the model) sits between the first check of the output directory
// and its replacement, so the directory is checked again right before it is removed: something that
// appeared there meanwhile is not deleted. No sleeps: the detections "file" is a FIFO, and opening it
// blocks the run until this test opens the other end, which it does only once the run has got past the
// first check and is loading the detector; the foreign file is created after that, and the FakeDetector
// reads to the end of the FIFO (the writer closing it) before the run goes on to replace the directory.
TEST(RedactJob, ChecksTheOutputDirectoryAgainBeforeReplacingIt) {
  RedactOptions o = baseOptions("redact_recheck");
  const fs::path dir = o.outRoot / kStem;
  fs::create_directories(dir);  // empty: --overwrite may replace it
  o.overwrite = true;
  // Named like the final video, so that the failure handler (which deletes the files a run writes) would
  // remove it if the second check were moved inside the try that the handler belongs to.
  const std::string foreign = std::string(kStem) + ".redacted.mp4";
  const fs::path fifo = test::tmpDir() / "redact_recheck.fifo";
  fs::remove(fifo);
  ASSERT_EQ(mkfifo(fifo.c_str(), 0600), 0);
  o.fakeDetections = fifo;

  std::thread writer([&] {
    const int fd = open(fifo.c_str(), O_WRONLY);  // returns when the run opens the detections
    if (fd < 0) return;
    std::ofstream(dir / foreign) << "keep me";
    const char blankLine = '\n';
    if (write(fd, &blankLine, 1) < 0) std::perror("write");
    close(fd);
  });
  bool refused = false;
  std::string failure;
  try {
    runRedact(o);
  } catch (const OutputExistsError&) {
    refused = true;
  } catch (const std::exception& e) {
    failure = e.what();
  }
  // A run that failed before it opened the FIFO would leave the writer blocked in open(): open the
  // read end (and hold it, so that the writer's write cannot hit a closed pipe) to let it through.
  const int reader = open(fifo.c_str(), O_RDONLY | O_NONBLOCK);
  writer.join();
  if (reader >= 0) close(reader);

  EXPECT_TRUE(refused) << "the run was not refused; failure: " << failure;
  EXPECT_TRUE(fs::exists(dir / foreign));
  EXPECT_FALSE(fs::exists(dir / "masks.jsonl"));
}

fs::path failingDetections(const std::string& name) {
  const fs::path path = test::tmpDir() / name;
  std::ofstream(path) << R"({"frame":10,"fail":true})" << '\n';
  return path;
}

TEST(RedactJob, AFailedRunRemovesTheOutputRootsItCreated) {
  RedactOptions o = baseOptions("redact_nested_unused");
  const fs::path top = test::tmpDir() / "redact_nested";
  fs::remove_all(top);
  o.outRoot = top / "a" / "b";
  o.fakeDetections = failingDetections("redact_nested_fail.jsonl");
  o.detectEveryFrame = true;
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_FALSE(fs::exists(top));
}

TEST(RedactJob, AFailedRunKeepsAnOutputRootThatExisted) {
  RedactOptions o = baseOptions("redact_kept_root");
  fs::create_directories(o.outRoot);
  std::ofstream(o.outRoot / "notes.txt") << "keep me";
  o.fakeDetections = failingDetections("redact_kept_root_fail.jsonl");
  o.detectEveryFrame = true;
  EXPECT_THROW(runRedact(o), VcError);
  EXPECT_TRUE(fs::exists(o.outRoot / "notes.txt"));
  EXPECT_FALSE(fs::exists(o.outRoot / kStem));

  RedactOptions empty = baseOptions("redact_kept_empty_root");
  fs::create_directories(empty.outRoot);
  empty.fakeDetections = o.fakeDetections;
  empty.detectEveryFrame = true;
  EXPECT_THROW(runRedact(empty), VcError);
  EXPECT_TRUE(fs::is_directory(empty.outRoot));  // it was there before the run
}

TEST(RedactJob, ReportsDecoderErrorsInTheStatsAndAsAWarning) {
  RedactOptions o = baseOptions("redact_corrupt");
  o.input = test::corruptFirstNalVideo();
  const RedactStats s = runRedact(o);
  EXPECT_GE(s.decodeErrors, 1);
  EXPECT_GE(s.frames, 1);
  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  EXPECT_EQ(stats.at("decode_errors").get<int64_t>(), s.decodeErrors);
  const std::vector<std::string> warnings = redactWarnings(s);
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].rfind("warning:", 0), 0u) << warnings[0];
  EXPECT_NE(warnings[0].find(std::to_string(s.decodeErrors)), std::string::npos) << warnings[0];

  RedactStats clean;
  EXPECT_TRUE(redactWarnings(clean).empty());
}

// The source video stream's bitrate as the container states it.
int64_t sourceBitRate(const fs::path& video) {
  VideoReader reader(video.string(), HwAccel::None);
  return reader.videoStream()->codecpar->bit_rate;
}

nlohmann::json statsOf(const RedactStats& s) {
  std::ifstream statsFile(s.outputDir / "stats.json");
  return nlohmann::json::parse(statsFile);
}

// The default is constant quality: no target bit rate, and stats.json says so.
TEST(RedactJob, EncodesAtConstantQualityByDefault) {
  const RedactOptions o = baseOptions("redact_quality");
  EXPECT_EQ(o.config.rateControl, RateControl::Quality);
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.rateControl, RateControl::Quality);
  EXPECT_EQ(s.outputBitRate, 0);
  const nlohmann::json stats = statsOf(s);
  EXPECT_EQ(stats.at("rate_control").get<std::string>(), "quality");
  EXPECT_TRUE(stats.at("output_bit_rate").is_null());
  EXPECT_EQ(stats.at("config").at("rate_control").get<std::string>(), "quality");
  EXPECT_EQ(test::videoPresentation(s.video).codec, AV_CODEC_ID_H264);
}

// spec 2026-10-20 §3 with rate_control "bitrate": a 10-bit HEVC source comes out as 8-bit 4:2:0 H.264 at 1.5 times its
// bitrate, masked like any other, with a note on the conversion.
TEST(RedactJob, AnHevcSourceComesOutAsH264AtTheFactorsBitrate) {
  RedactOptions o = baseOptions("redact_hevc");
  o.input = test::tenBitVideo();
  o.config.rateControl = RateControl::Bitrate;
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
  EXPECT_EQ(stats.at("rate_control").get<std::string>(), "bitrate");
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
  o.config.rateControl = RateControl::Bitrate;
  o.config.outputBitrateFactorHevc = 2.0;
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.outputBitRate, sourceBitRate(o.input) * 2);
}

// An H.264 source keeps its bitrate (with rate_control "bitrate") and needs no note.
TEST(RedactJob, AnH264SourceKeepsItsBitrate) {
  RedactOptions o = baseOptions("redact_h264_rate");
  o.config.rateControl = RateControl::Bitrate;
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
TEST(RedactJob, ReportsTheMainStageAndTheSetupTime) {
  const RedactOptions o = baseOptions("redact_stages");
  const RedactStats s = runRedact(o);
  EXPECT_EQ(s.main.items, 60);
  EXPECT_DOUBLE_EQ(s.main.busySec, s.detect.busySec + s.track.busySec + s.mosaic.busySec);
  EXPECT_GT(s.setupSec, 0.0);
  EXPECT_LT(s.setupSec, s.elapsedSec);

  std::ifstream statsFile(s.outputDir / "stats.json");
  const nlohmann::json stats = nlohmann::json::parse(statsFile);
  const nlohmann::json& main = stats.at("stages").at("main");
  EXPECT_EQ(main.at("items").get<int>(), 60);
  EXPECT_DOUBLE_EQ(main.at("busy_s").get<double>(), s.main.busySec);
  EXPECT_DOUBLE_EQ(main.at("fps").get<double>(), s.main.fps());
  EXPECT_DOUBLE_EQ(stats.at("setup_s").get<double>(), s.setupSec);
  for (const char* stage : {"decode", "detect", "track", "mosaic", "encode"}) {
    EXPECT_TRUE(stats.at("stages").contains(stage)) << stage;
  }
  // Without a face model the student is the only detector: its busy time is detect's, the face model has none.
  EXPECT_DOUBLE_EQ(s.detectStudent.busySec, s.detect.busySec);
  EXPECT_EQ(s.detectStudent.items, s.detect.items);
  EXPECT_EQ(s.detectFace.items, 0);
  EXPECT_TRUE(stats.at("stages").contains("detect_student"));
  EXPECT_TRUE(stats.at("stages").contains("detect_face"));
}

TEST(RedactJob, TheSummaryShowsTheSetupAndTheParallelStages) {
  RedactStats s;
  s.frames = 600;
  s.elapsedSec = 3.0;
  s.setupSec = 0.75;
  s.decode = StageStat{0.5, 600};
  s.main = StageStat{2.0, 600};
  s.detect = StageStat{1.5, 50};
  s.detectStudent = StageStat{1.2, 50};
  s.detectFace = StageStat{0.6, 50};
  s.track = StageStat{0.25, 600};
  s.mosaic = StageStat{0.25, 600};
  s.encode = StageStat{2.4, 600};
  s.videoFps = 60000.0 / 1001.0;
  s.keyframeStride = 6;
  s.detectFpsActual = s.videoFps / 6;
  const std::string summary = redactSummary(s);
  EXPECT_NE(summary.find("setup:         0.75 s"), std::string::npos) << summary;
  EXPECT_NE(summary.find("stages (fps):  decode 1200.0, main 300.0 (detect 33.3, track 2400.0, mosaic 2400.0), "
                         "encode 250.0"),
            std::string::npos)
      << summary;
  EXPECT_NE(summary.find("detect rate:   9.99 /s (every 6 frames of a 59.94 fps video)\n"), std::string::npos) << summary;
  EXPECT_NE(summary.find("elapsed:       3.00 s (200.0 fps)"), std::string::npos) << summary;
  EXPECT_NE(summary.find("detectors:     student 24.0 ms + face 12.0 ms per keyframe, run in parallel: 30.0 ms\n"),
            std::string::npos)
      << summary;
  s.detectFace = StageStat{};
  EXPECT_EQ(redactSummary(s).find("detectors:"), std::string::npos);  // one detector: no such line
}

TEST(RedactJob, QueueCapacityIsBoundedByBytesPerFrame) {
  EXPECT_EQ(pipelineQueueCapacity(640, 360), 64u);
  EXPECT_EQ(pipelineQueueCapacity(1920, 1080), 43u);  // 256 MiB / (1920 x 1080 x 3 bytes)
  EXPECT_EQ(pipelineQueueCapacity(3840, 2160), 10u);
  EXPECT_EQ(pipelineQueueCapacity(16384, 16384), 4u);  // never below 4
  EXPECT_EQ(pipelineQueueCapacity(1, 1), 64u);         // never above 64
}

}  // namespace
}  // namespace vc
