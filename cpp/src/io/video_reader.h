#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/luma_plane.h"
#include "common/platform.h"

struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vc {

enum class HwAccel { None, VideoToolbox, Cuda, Auto };

// Parses "none" | "videotoolbox" | "cuda" | "auto"; throws std::invalid_argument.
HwAccel parseHwAccel(const std::string& text);

// The hardware decoder a requested mode means on a platform (spec 2026-10-20 §3).
enum class HwDevice { None, VideoToolbox, Cuda };
struct HwDecodePlan {
  HwDevice device = HwDevice::None;  // the device to try; None decodes in software
  bool required = false;             // a device that cannot be opened is an error instead of software decoding
  bool noteOnFallback = false;       // software decoding instead of the device gets a line on stderr
};
// none: software. videotoolbox: VideoToolbox, required. cuda: CUDA (NVDEC), software with a note when it cannot be
// opened (on macOS it never can). auto: CUDA with that note on Linux, VideoToolbox silently falling back on macOS,
// software elsewhere.
HwDecodePlan hwDecodePlan(HwAccel requested, Platform platform);

struct VideoInfo {
  int width = 0;
  int height = 0;
  double fps = 0;         // average frame rate, 0 when unknown
  double durationMs = 0;  // container duration, 0 when unknown
  std::string codecName;
};

// A decoded frame in CPU memory, always 8-bit YUV (planar or NV12), so
// plane 0 is the luma plane. Hardware frames (VideoToolbox, CUDA) are copied back to memory.
struct VideoFrame {
  // 0-based count of the frames the decoder has returned, in decoder output
  // (display) order. Frames lost to corrupt packets are not counted, so this can
  // drift from the container's own frame numbering.
  int64_t index = 0;
  double ptsMs = 0;  // relative to the stream start
  const AVFrame* av = nullptr;

  LumaPlane luma() const;
};

// Sequential demux + decode of the best video stream in a file.
class VideoReader {
 public:
  // Throws VcError when the file cannot be opened, has no decodable video
  // stream, or VideoToolbox was requested explicitly and is unavailable.
  // HwAccel::Auto and HwAccel::Cuda fall back to software decoding (hwDecodePlan).
  VideoReader(const std::string& path, HwAccel hwAccel);
  ~VideoReader();
  VideoReader(const VideoReader&) = delete;
  VideoReader& operator=(const VideoReader&) = delete;

  const VideoInfo& info() const;
  // True once at least one frame came out of the hardware decoder.
  bool usingHardware() const;
  // Packets/frames the decoder rejected as corrupt; decoding continues past them.
  int64_t decodeErrors() const;
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

  // Next frame, or nullptr at end of stream. The pointer and its AVFrame stay
  // valid until the next call. The frame's buffers are reference counted and a
  // later call never reuses or overwrites them in place, so a caller may keep
  // an av_frame_clone() of the frame for as long as it likes (vc_redact does).
  // Throws VcError on unrecoverable errors.
  const VideoFrame* next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
