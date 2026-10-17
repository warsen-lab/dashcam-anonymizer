#pragma once

#include <filesystem>

#include "io/jpeg_writer.h"
#include "io/video_reader.h"
#include "sampler/frame_sampler.h"

namespace vc {

struct ExtractOptions {
  std::filesystem::path input;    // video file
  std::filesystem::path outRoot;  // frames go to outRoot / <input stem>
  SamplerConfig sampler;
  HwAccel hwAccel = HwAccel::Auto;
  int jpegQscale = kBestJpegQscale;
  bool overwrite = false;
};

}  // namespace vc
