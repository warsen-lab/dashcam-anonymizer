#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/luma_plane.h"

struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vc {

enum class HwAccel { None, VideoToolbox, Auto };

// Parses "none" | "videotoolbox" | "auto"; throws std::invalid_argument.
HwAccel parseHwAccel(const std::string& text);

struct VideoInfo {
  int width = 0;
  int height = 0;
  double fps = 0;         // average frame rate, 0 when unknown
  double durationMs = 0;  // container duration, 0 when unknown
  std::string codecName;
};

// A decoded frame in CPU memory, always 8-bit YUV (planar or NV12), so
// plane 0 is the luma plane.
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
  // HwAccel::Auto falls back to software decoding.
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
  // valid until the next call. Throws VcError on unrecoverable errors.
  const VideoFrame* next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
