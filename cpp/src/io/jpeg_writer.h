#pragma once

#include <filesystem>
#include <memory>

struct AVFrame;

namespace vc {

inline constexpr int kBestJpegQscale = 2;
inline constexpr int kWorstJpegQscale = 31;

// Encodes decoded frames as baseline JPEG files with FFmpeg's MJPEG encoder.
class JpegWriter {
 public:
  // qscale is FFmpeg's MJPEG quantiser: 2 (best, ~quality 95) .. 31 (worst).
  // Throws std::invalid_argument outside that range.
  explicit JpegWriter(int qscale = kBestJpegQscale);
  ~JpegWriter();
  JpegWriter(const JpegWriter&) = delete;
  JpegWriter& operator=(const JpegWriter&) = delete;

  // Writes `frame` (any software pixel format) to `path`. Throws VcError.
  void write(const AVFrame& frame, const std::filesystem::path& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
