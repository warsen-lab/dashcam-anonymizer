#pragma once

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
// 1 s AAC audio, no video stream.
std::filesystem::path audioOnlyFile();

// Frame whose luma bytes are all y and chroma bytes u / v (8-bit planar YUV or NV12), limited range.
av::FramePtr makeFrame(int width, int height, AVPixelFormat format, uint8_t y, uint8_t u, uint8_t v);
// Same layout filled with deterministic pseudo-random bytes.
av::FramePtr makeNoiseFrame(int width, int height, AVPixelFormat format, unsigned seed);
// Independent copy with its own writable buffers.
av::FramePtr deepCopy(const AVFrame& frame);

}  // namespace vc::test
