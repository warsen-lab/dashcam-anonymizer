#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include "io/av_ptr.h"

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
// 1 s, 640x360 @ 30 fps 8-bit 4:2:2 H.264 (High 4:2:2, 30 frames).
std::filesystem::path yuv422Video();
// 1 s AAC audio, no video stream.
std::filesystem::path audioOnlyFile();
// 2 s, 640x360 @ 30 fps H.264 (60 frames) with 2 s of AAC audio.
std::filesystem::path videoWithAudio();
// 2 s, 640x360 @ 30 fps H.264 (4 Mb/s) of testsrc2 with strong temporal noise, plus AAC audio:
// every region has detail, so pixelation is measurable.
std::filesystem::path noisyVideoWithAudio();
// 2 s, 640x360 @ 30 fps lossless H.264 of a still 2 px luma checkerboard (0 / 255): half of all
// neighbouring pixel pairs differ by 255 on every frame, so any region left unpixelated shows.
std::filesystem::path checkerboardVideo();
// basicVideo() with the length prefix of its first NAL unit overwritten, so the decoder rejects the
// first access unit and the rest of the stream still decodes.
std::filesystem::path corruptFirstNalVideo();
// 1 s, 320x240 @ 30 fps H.264 with a mov_text subtitle stream.
std::filesystem::path videoWithSubtitles();
// 2 s of videoWithAudio() cut with `ffmpeg -ss 0.7 -c copy`: the video starts on the keyframe before
// the cut, so its first packets have negative timestamps (hidden by an MP4 edit list).
std::filesystem::path copyCutVideo();
// videoWithAudio() remuxed to MPEG-TS: 90 kHz time bases for both streams and a start offset of ~1.4 s.
std::filesystem::path mpegTsVideo();
// 1 s, 320x240 @ 30 fps H.264 with a 4:3 sample aspect ratio.
std::filesystem::path anamorphicVideo();
// basic.mp4 (video only) with a 90 degree rotation in the display matrix.
std::filesystem::path rotatedVideo();
// 1 s, 320x240 @ 30 fps H.264 tagged BT.709 / limited range.
std::filesystem::path bt709Video();
// 1 s, 320x240 @ 30 fps H.264 tagged BT.601 (smpte170m / bt470bg) / full range.
std::filesystem::path bt601FullRangeVideo();
// 1 s, 320x240 @ 30 fps H.264 with PCM mu-law audio in a MOV: that audio cannot be stored in MP4.
std::filesystem::path videoWithMulawAudio();
// MPEG-TS: 0.5 s of 320x240 H.264, then 0.5 s of 160x120 (the frame size changes mid-stream).
std::filesystem::path resolutionChangeVideo();
// 1 s, 320x240 H.264 as a raw elementary stream (.h264): frames carry no timestamps.
std::filesystem::path rawH264Stream();

// Timing of one stream, from the container and its packets (packets flagged discard, which the
// decoder never shows, are left out of first / last).
struct StreamTiming {
  double startSec = 0;     // AVStream::start_time
  double firstPtsSec = 0;  // smallest packet pts
  double lastPtsSec = 0;   // largest packet pts
  int packets = 0;
};
// First stream of `type`. Throws std::runtime_error when there is none.
StreamTiming streamTiming(const std::filesystem::path& path, AVMediaType type);
// Codec id of the first stream of `type`.
AVCodecID streamCodec(const std::filesystem::path& path, AVMediaType type);

// How the first video stream is presented.
struct VideoPresentation {
  AVRational sampleAspectRatio{0, 1};
  AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
  AVColorSpace colorSpace = AVCOL_SPC_UNSPECIFIED;
  AVColorPrimaries colorPrimaries = AVCOL_PRI_UNSPECIFIED;
  AVColorTransferCharacteristic colorTrc = AVCOL_TRC_UNSPECIFIED;
  bool hasDisplayMatrix = false;
  std::array<int32_t, 9> displayMatrix{};
  double rotationDegrees = 0;  // av_display_rotation_get of the matrix, 0 without one
  uint32_t codecTag = 0;
  AVCodecID codec = AV_CODEC_ID_NONE;
  int level = 0;          // the codec parameters' level (H.264: level_idc, 52 for 5.2)
  int pixelFormat = -1;   // the codec parameters' format (an AVPixelFormat)
};
VideoPresentation videoPresentation(const std::filesystem::path& path);

// Number of streams of `type` in a media file (opened with libavformat).
int streamCount(const std::filesystem::path& path, AVMediaType type);
// Duration in seconds of the first stream of `type`, from the container.
double streamDurationSec(const std::filesystem::path& path, AVMediaType type);

// Frame whose luma bytes are all y and chroma bytes u / v (8-bit planar YUV or NV12), limited range.
av::FramePtr makeFrame(int width, int height, AVPixelFormat format, uint8_t y, uint8_t u, uint8_t v);
// Same layout filled with deterministic pseudo-random bytes.
av::FramePtr makeNoiseFrame(int width, int height, AVPixelFormat format, unsigned seed);
// Independent copy with its own writable buffers.
av::FramePtr deepCopy(const AVFrame& frame);

}  // namespace vc::test
