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
