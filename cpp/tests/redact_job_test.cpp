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

// 500 frames in 0.5 s: the sampler's 200 ms interval would leave 200 frames buffered, so
// the 120-frame limit has to force keyframes.
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
  EXPECT_GE(s.keyframes, 5);  // frames 0, 120, 200, 320, 400; the sampler alone would pick 0, 200, 400
  EXPECT_EQ(lineCount(s.outputDir / "masks.jsonl"), 500u);
}

}  // namespace
}  // namespace vc
