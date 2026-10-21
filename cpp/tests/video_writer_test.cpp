#include "io/video_writer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "io/av_ptr.h"
#include "io/video_reader.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

// The writer configuration vc_redact builds for `reader`'s video stream.
WriterConfig configFor(const VideoReader& reader, const fs::path& output, bool preferHardware) {
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
  copyPresentation(*stream, config);
  return config;
}

// Decodes `input` and writes every frame (and the audio) through a VideoWriter, its configuration changed by
// `tweak`. Returns the encoder.
std::string reencode(const fs::path& input, const fs::path& output, bool preferHardware,
                     const std::function<void(WriterConfig&)>& tweak = {}) {
  VideoReader reader(input.string(), HwAccel::None);
  WriterConfig config = configFor(reader, output, preferHardware);
  if (tweak) tweak(config);
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
      writer = std::make_unique<VideoWriter>(config, snapshotAudioStreams(reader.audioStreams()));
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

// Absolute timestamps in seconds (the container's own, not relative to the stream start).
std::vector<double> frameTimesSec(const fs::path& path) {
  VideoReader reader(path.string(), HwAccel::None);
  const AVRational tb = reader.videoStream()->time_base;
  std::vector<double> times;
  while (const VideoFrame* frame = reader.next()) {
    times.push_back(static_cast<double>(frame->av->best_effort_timestamp) * av_q2d(tb));
  }
  return times;
}

constexpr double kVideoFrameSec = 1.0 / 30.0;
constexpr double kAacFrameSec = 1024.0 / 44100.0;

// Start offset, first and last packet of one stream agree with the source within `tolerance`.
void expectSameTiming(const fs::path& source, const fs::path& output, AVMediaType type, double tolerance) {
  const test::StreamTiming in = test::streamTiming(source, type);
  const test::StreamTiming out = test::streamTiming(output, type);
  EXPECT_NEAR(out.startSec, in.startSec, tolerance) << "start_time of " << av_get_media_type_string(type);
  EXPECT_NEAR(out.firstPtsSec, in.firstPtsSec, tolerance) << "first pts of " << av_get_media_type_string(type);
  EXPECT_NEAR(out.lastPtsSec, in.lastPtsSec, tolerance) << "last pts of " << av_get_media_type_string(type);
}

// Same frames at the same absolute times.
void expectSameFrameTimes(const fs::path& source, const fs::path& output) {
  const std::vector<double> in = frameTimesSec(source);
  const std::vector<double> got = frameTimesSec(output);
  ASSERT_EQ(got.size(), in.size());
  ASSERT_FALSE(in.empty());
  for (size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], in[i], kVideoFrameSec) << "frame " << i;
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

// Constant quality ignores the bit rate: asked for 50 kb/s, a bit-rate encode stays small, a constant-quality one
// spends what the picture needs, several times more. Both encoders the writer can pick.
TEST(VideoWriter, ConstantQualityIgnoresTheBitRate) {
  for (const bool hardware : {false, true}) {
    const fs::path rated = test::tmpDir() / (hardware ? "writer_rate_hw.mp4" : "writer_rate_sw.mp4");
    const fs::path quality = test::tmpDir() / (hardware ? "writer_quality_hw.mp4" : "writer_quality_sw.mp4");
    const std::string a = reencode(test::basicVideo(), rated, hardware, [](WriterConfig& c) { c.bitRate = 50'000; });
    const std::string b = reencode(test::basicVideo(), quality, hardware, [](WriterConfig& c) {
      c.bitRate = 50'000;
      c.constantQuality = true;
    });
    EXPECT_EQ(a, b);
    EXPECT_GT(fs::file_size(quality), 3 * fs::file_size(rated)) << b;
    EXPECT_EQ(framePts(quality).size(), 60u) << b;
  }
}

// VideoToolbox's H.264 encoder takes at most 4096 px per side: the writer falls back to libx264 and
// says so on stderr, naming the encoder and the reason.
TEST(VideoWriter, SaysWhyItFallsBackToTheSoftwareEncoder) {
  WriterConfig config;
  config.path = test::tmpDir() / "writer_fallback.mp4";
  config.width = 5000;
  config.height = 200;
  config.timeBase = AVRational{1, 30};
  config.frameRate = AVRational{30, 1};
  config.preferHardware = true;
  testing::internal::CaptureStderr();
  std::string encoder;
  {
    VideoWriter writer(config, {});
    encoder = writer.encoderName();
  }
  const std::string err = testing::internal::GetCapturedStderr();
  ASSERT_EQ(encoder, "libx264");
  EXPECT_NE(err.find("vc: video encoder h264_videotoolbox"), std::string::npos) << err;
  EXPECT_NE(err.find("using libx264"), std::string::npos) << err;
  // FFmpeg's own messages aside, the writer prints exactly one line.
  size_t ours = 0;
  for (size_t at = err.find("vc: "); at != std::string::npos; at = err.find("vc: ", at + 1)) ++ours;
  EXPECT_EQ(ours, 1u) << err;

  // No line when the first choice opens.
  config.preferHardware = false;
  config.width = 640;
  testing::internal::CaptureStderr();
  { VideoWriter writer(config, {}); }
  EXPECT_EQ(testing::internal::GetCapturedStderr().find("vc: "), std::string::npos);
}

// Every frame sent comes out as exactly one packet; anything else fails finish().
TEST(VideoWriter, CountsOnePacketPerFrame) {
  for (const bool hardware : {false, true}) {
    VideoReader reader(test::basicVideo().string(), HwAccel::None);
    WriterConfig config = configFor(reader, test::tmpDir() / "writer_packets.mp4", hardware);
    std::unique_ptr<VideoWriter> writer;
    while (const VideoFrame* frame = reader.next()) {
      if (!writer) {
        config.pixelFormat = static_cast<AVPixelFormat>(frame->av->format);
        writer = std::make_unique<VideoWriter>(config, std::vector<AudioStreamInfo>{});
      }
      av::FramePtr clone(av_frame_clone(frame->av));
      clone->pts = clone->best_effort_timestamp;
      writer->writeVideo(*clone);
    }
    writer->finish();
    EXPECT_EQ(writer->videoFrames(), 60) << writer->encoderName();
    EXPECT_EQ(writer->videoPackets(), 60) << writer->encoderName();
  }
  EXPECT_NO_THROW(checkEncodedPackets(60, 60));
  EXPECT_THROW(checkEncodedPackets(60, 59), VcError);
  EXPECT_THROW(checkEncodedPackets(60, 61), VcError);
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

TEST(VideoWriter, KeepsTheTimingOfACopyCutSource) {
  const fs::path source = test::copyCutVideo();
  const fs::path out = test::tmpDir() / "writer_cut.mp4";
  reencode(source, out, false);
  expectSameFrameTimes(source, out);
  expectSameTiming(source, out, AVMEDIA_TYPE_VIDEO, kVideoFrameSec);
  expectSameTiming(source, out, AVMEDIA_TYPE_AUDIO, kAacFrameSec);
  EXPECT_EQ(test::streamTiming(out, AVMEDIA_TYPE_AUDIO).packets, test::streamTiming(source, AVMEDIA_TYPE_AUDIO).packets);
}

// MPEG-TS keeps both streams in 1/90000 and starts at ~1.4 s; the MP4 muxer picks its own audio
// time base, so the audio timestamps go through a rescale.
TEST(VideoWriter, KeepsTheTimingOfAnMpegTsSource) {
  const fs::path source = test::mpegTsVideo();
  ASSERT_GT(test::streamTiming(source, AVMEDIA_TYPE_VIDEO).startSec, 1.0);
  const fs::path out = test::tmpDir() / "writer_ts.mp4";
  reencode(source, out, false);
  expectSameFrameTimes(source, out);
  expectSameTiming(source, out, AVMEDIA_TYPE_VIDEO, kVideoFrameSec);
  expectSameTiming(source, out, AVMEDIA_TYPE_AUDIO, kAacFrameSec);
  EXPECT_EQ(test::streamTiming(out, AVMEDIA_TYPE_AUDIO).packets, test::streamTiming(source, AVMEDIA_TYPE_AUDIO).packets);
}

TEST(VideoWriter, CopiesEveryAudioPacketWithTheSourceCodec) {
  const fs::path out = test::tmpDir() / "writer_audio_packets.mp4";
  reencode(test::videoWithAudio(), out, false);
  EXPECT_EQ(test::streamCodec(out, AVMEDIA_TYPE_AUDIO), test::streamCodec(test::videoWithAudio(), AVMEDIA_TYPE_AUDIO));
  const int expected = test::streamTiming(test::videoWithAudio(), AVMEDIA_TYPE_AUDIO).packets;
  EXPECT_GT(expected, 80);
  EXPECT_EQ(test::streamTiming(out, AVMEDIA_TYPE_AUDIO).packets, expected);
}

TEST(VideoWriter, TagsHevcAsHvc1) {
  const fs::path out = test::tmpDir() / "writer_hvc1.mp4";
  EXPECT_EQ(reencode(test::tenBitVideo(), out, false), "libx265");
  EXPECT_EQ(test::videoPresentation(out).codecTag, static_cast<uint32_t>(MKTAG('h', 'v', 'c', '1')));
}

TEST(VideoWriter, KeepsTheSampleAspectRatio) {
  const AVRational sar = test::videoPresentation(test::anamorphicVideo()).sampleAspectRatio;
  ASSERT_EQ(sar.num * 3, sar.den * 4) << sar.num << ":" << sar.den;  // 4:3 in the source
  for (const bool hardware : {false, true}) {
    const fs::path out = test::tmpDir() / "writer_sar.mp4";
    fs::remove(out);
    const std::string encoder = reencode(test::anamorphicVideo(), out, hardware);
    const AVRational got = test::videoPresentation(out).sampleAspectRatio;
    EXPECT_EQ(got.num * 3, got.den * 4) << encoder << ": " << got.num << ":" << got.den;
  }
}

TEST(VideoWriter, KeepsTheRotation) {
  const test::VideoPresentation source = test::videoPresentation(test::rotatedVideo());
  ASSERT_TRUE(source.hasDisplayMatrix);
  ASSERT_NEAR(std::abs(source.rotationDegrees), 90.0, 0.01);
  for (const bool hardware : {false, true}) {
    const fs::path out = test::tmpDir() / "writer_rotated.mp4";
    fs::remove(out);
    const std::string encoder = reencode(test::rotatedVideo(), out, hardware);
    const test::VideoPresentation got = test::videoPresentation(out);
    ASSERT_TRUE(got.hasDisplayMatrix) << encoder;
    EXPECT_EQ(got.displayMatrix, source.displayMatrix) << encoder;
  }
}

TEST(VideoWriter, KeepsTheColourTags) {
  for (const fs::path& source : {test::bt709Video(), test::bt601FullRangeVideo()}) {
    const test::VideoPresentation in = test::videoPresentation(source);
    ASSERT_NE(in.colorSpace, AVCOL_SPC_UNSPECIFIED) << source;
    ASSERT_NE(in.colorRange, AVCOL_RANGE_UNSPECIFIED) << source;
    for (const bool hardware : {false, true}) {
      const fs::path out = test::tmpDir() / "writer_colours.mp4";
      fs::remove(out);
      const std::string encoder = reencode(source, out, hardware);
      const test::VideoPresentation got = test::videoPresentation(out);
      const std::string where = source.filename().string() + " via " + encoder;
      EXPECT_EQ(got.colorRange, in.colorRange) << where;
      EXPECT_EQ(got.colorSpace, in.colorSpace) << where;
      EXPECT_EQ(got.colorPrimaries, in.colorPrimaries) << where;
      EXPECT_EQ(got.colorTrc, in.colorTrc) << where;
    }
  }
}

TEST(VideoWriter, LeavesUntaggedSourcesUntagged) {
  const fs::path out = test::tmpDir() / "writer_untagged.mp4";
  reencode(test::basicVideo(), out, false);
  const test::VideoPresentation got = test::videoPresentation(out);
  EXPECT_EQ(got.colorSpace, AVCOL_SPC_UNSPECIFIED);
  EXPECT_EQ(got.colorPrimaries, AVCOL_PRI_UNSPECIFIED);
  EXPECT_EQ(got.colorTrc, AVCOL_TRC_UNSPECIFIED);
  EXPECT_FALSE(got.hasDisplayMatrix);
}

// The encoder takes the frame's size from the configuration: a smaller frame would be read past its end.
TEST(VideoWriter, RejectsAFrameOfAnotherSize) {
  WriterConfig config;
  config.path = test::tmpDir() / "writer_size.mp4";
  config.width = 64;
  config.height = 64;
  config.preferHardware = false;
  VideoWriter writer(config, {});
  av::FramePtr good = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 100, 128, 128);
  good->pts = 0;
  EXPECT_NO_THROW(writer.writeVideo(*good));
  for (const auto& [w, h] : {std::pair{32, 32}, std::pair{64, 32}, std::pair{128, 64}}) {
    av::FramePtr bad = test::makeFrame(w, h, AV_PIX_FMT_YUV420P, 100, 128, 128);
    bad->pts = 1;
    EXPECT_THROW(writer.writeVideo(*bad), VcError) << w << "x" << h;
  }
  EXPECT_EQ(writer.videoFrames(), 1);
}

// The snapshot is what the encode thread works with: it must not depend on the demuxer's streams.
TEST(VideoWriter, AudioSnapshotsOutliveTheReaderAndDriveTheWriter) {
  std::vector<AudioStreamInfo> audio;
  av::PacketPtr packet = av::allocPacket();
  AVRational timeBase{0, 1};
  AVCodecID codec = AV_CODEC_ID_NONE;
  int sampleRate = 0;
  {
    VideoReader reader(test::videoWithAudio().string(), HwAccel::None);
    audio = snapshotAudioStreams(reader.audioStreams());
    ASSERT_EQ(audio.size(), 1u);
    reader.setAudioSink([&](const AVPacket& p) {
      if (packet->size == 0 && av_packet_ref(packet.get(), &p) < 0) throw std::runtime_error("av_packet_ref");
    });
    while (packet->size == 0 && reader.next() != nullptr) {
    }
    ASSERT_GT(packet->size, 0);
    timeBase = reader.audioStreams()[0]->time_base;
    codec = reader.audioStreams()[0]->codecpar->codec_id;
    sampleRate = reader.audioStreams()[0]->codecpar->sample_rate;
  }
  EXPECT_EQ(audio[0].timeBase.num, timeBase.num);
  EXPECT_EQ(audio[0].timeBase.den, timeBase.den);
  EXPECT_EQ(audio[0].parameters->codec_id, codec);
  EXPECT_EQ(audio[0].parameters->sample_rate, sampleRate);
  EXPECT_GT(sampleRate, 0);

  WriterConfig config;
  config.path = test::tmpDir() / "writer_snapshot.mp4";
  config.width = 64;
  config.height = 64;
  config.preferHardware = false;
  {
    VideoWriter writer(config, audio);
    av::FramePtr frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 100, 128, 128);
    frame->pts = 0;
    writer.writeVideo(*frame);
    writer.writeAudio(*packet, 0);  // an MP4 track without samples is not written
    writer.finish();
  }
  EXPECT_EQ(test::streamCount(config.path, AVMEDIA_TYPE_AUDIO), 1);
  EXPECT_EQ(test::streamCodec(config.path, AVMEDIA_TYPE_AUDIO), codec);
}

TEST(VideoWriter, RejectsAudioAndVideoAfterFinish) {
  VideoReader reader(test::videoWithAudio().string(), HwAccel::None);
  WriterConfig config = configFor(reader, test::tmpDir() / "writer_after_finish.mp4", false);
  av::PacketPtr packet = av::allocPacket();
  reader.setAudioSink([&](const AVPacket& p) {
    if (packet->size == 0 && av_packet_ref(packet.get(), &p) < 0) throw std::runtime_error("av_packet_ref");
  });
  const VideoFrame* first = reader.next();
  ASSERT_NE(first, nullptr);
  av::FramePtr frame(av_frame_clone(first->av));
  frame->pts = frame->best_effort_timestamp;
  config.pixelFormat = static_cast<AVPixelFormat>(frame->format);
  while (packet->size == 0 && reader.next() != nullptr) {
  }
  ASSERT_GT(packet->size, 0);
  VideoWriter writer(config, snapshotAudioStreams(reader.audioStreams()));
  writer.writeVideo(*frame);
  writer.writeAudio(*packet, 0);
  writer.finish();
  try {
    writer.writeAudio(*packet, 0);
    ADD_FAILURE() << "writeAudio after finish did not throw";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("after finish"), std::string::npos) << e.what();
  }
  EXPECT_THROW(writer.writeVideo(*frame), VcError);
  EXPECT_NO_THROW(writer.finish());  // a second finish stays harmless
}

// Frames of an I/P/B source carry the decoder's picture type; the encoder must choose its own.
TEST(VideoWriter, ClearsThePictType) {
  for (const auto& [input, codecLabel] : {std::pair{test::basicVideo(), "h264, no conversion"},
                                          std::pair{test::tenBitVideo(), "hevc, converted"}}) {
    VideoReader reader(input.string(), HwAccel::None);
    WriterConfig config = configFor(reader, test::tmpDir() / "writer_pict_type.mp4", false);
    std::unique_ptr<VideoWriter> writer;
    int checked = 0;
    while (const VideoFrame* frame = reader.next()) {
      if (!writer) {
        config.pixelFormat = static_cast<AVPixelFormat>(frame->av->format);
        writer = std::make_unique<VideoWriter>(config, snapshotAudioStreams(reader.audioStreams()));
      }
      av::FramePtr clone(av_frame_clone(frame->av));
      clone->pts = clone->best_effort_timestamp;
      clone->pict_type = checked % 2 == 0 ? AV_PICTURE_TYPE_B : AV_PICTURE_TYPE_I;
      writer->writeVideo(*clone);
      EXPECT_EQ(clone->pict_type, AV_PICTURE_TYPE_NONE) << codecLabel << " frame " << checked;
      if (++checked == 8) break;
    }
    EXPECT_EQ(checked, 8) << codecLabel;
    writer->finish();
  }
}

// Frames that need a pixel format conversion keep their range: full-range luma 235 must not be
// squeezed into the limited range (where it would come out near 218).
TEST(VideoWriter, KeepsTheRangeOfFramesItConverts) {
  WriterConfig config;
  config.path = test::tmpDir() / "writer_full_range.mp4";
  config.width = 128;
  config.height = 128;
  config.pixelFormat = AV_PIX_FMT_NV12;  // libx265 takes no NV12: converted to yuv420p
  config.sourceCodec = "hevc";
  config.colorRange = AVCOL_RANGE_JPEG;
  config.preferHardware = false;
  fs::remove(config.path);
  {
    VideoWriter writer(config, {});
    for (int i = 0; i < 4; ++i) {
      av::FramePtr frame = test::makeFrame(128, 128, AV_PIX_FMT_NV12, 235, 128, 128);
      frame->color_range = AVCOL_RANGE_JPEG;
      frame->pts = i * 33;
      writer.writeVideo(*frame);
    }
    writer.finish();
  }
  VideoReader reader(config.path.string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  EXPECT_NEAR(frame->av->data[0][64 * frame->av->linesize[0] + 64], 235, 4);
}

TEST(VideoWriter, RefusesAnAudioCodecThatMp4CannotStoreBeforeCreatingTheFile) {
  VideoReader reader(test::videoWithMulawAudio().string(), HwAccel::None);
  WriterConfig config = configFor(reader, test::tmpDir() / "writer_mulaw.mp4", false);
  fs::remove(config.path);
  config.pixelFormat = AV_PIX_FMT_YUV420P;
  try {
    VideoWriter writer(config, snapshotAudioStreams(reader.audioStreams()));
    ADD_FAILURE() << "no throw";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("audio codec pcm_mulaw cannot be stored in MP4"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(config.path));
}

}  // namespace
}  // namespace vc
