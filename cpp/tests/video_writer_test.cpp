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
