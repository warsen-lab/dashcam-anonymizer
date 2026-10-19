#include "detect/letterbox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {

Letterbox makeLetterbox(int frameWidth, int frameHeight, int inputWidth, int inputHeight) {
  if (frameWidth <= 0 || frameHeight <= 0 || inputWidth <= 0 || inputHeight <= 0) {
    throw std::invalid_argument("letterbox sizes must be positive");
  }
  Letterbox lb;
  lb.frameWidth = frameWidth;
  lb.frameHeight = frameHeight;
  lb.inputWidth = inputWidth;
  lb.inputHeight = inputHeight;
  lb.r = std::min(static_cast<double>(inputWidth) / frameWidth,
                  static_cast<double>(inputHeight) / frameHeight);
  lb.scaledWidth = std::min(inputWidth, static_cast<int>(std::lround(frameWidth * lb.r)));
  lb.scaledHeight = std::min(inputHeight, static_cast<int>(std::lround(frameHeight * lb.r)));
  if (lb.scaledWidth <= 0 || lb.scaledHeight <= 0) {
    throw std::invalid_argument("a " + std::to_string(frameWidth) + "x" + std::to_string(frameHeight) +
                                " frame scales to " + std::to_string(lb.scaledWidth) + "x" +
                                std::to_string(lb.scaledHeight) + " in a " + std::to_string(inputWidth) + "x" +
                                std::to_string(inputHeight) + " model input");
  }
  return lb;
}

struct InputTensorFiller::Impl {
  Letterbox lb;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr rgb = av::allocFrame();
};

InputTensorFiller::InputTensorFiller(const Letterbox& letterbox)
    : impl_(std::make_unique<Impl>()) {
  impl_->lb = letterbox;
  impl_->sws->flags = SWS_BILINEAR;
}

InputTensorFiller::~InputTensorFiller() = default;

void InputTensorFiller::fill(const AVFrame& frame, float* out) {
  Impl& s = *impl_;
  const Letterbox& lb = s.lb;
  if (frame.width != lb.frameWidth || frame.height != lb.frameHeight) {
    throw VcError("frame is " + std::to_string(frame.width) + "x" + std::to_string(frame.height) +
                  ", the detector was set up for " + std::to_string(lb.frameWidth) + "x" +
                  std::to_string(lb.frameHeight));
  }
  av_frame_unref(s.rgb.get());
  s.rgb->format = AV_PIX_FMT_RGB24;
  s.rgb->width = lb.scaledWidth;
  s.rgb->height = lb.scaledHeight;
  checkAv(sws_scale_frame(s.sws.get(), s.rgb.get(), &frame), "convert frame to RGB");

  const size_t plane = static_cast<size_t>(lb.inputWidth) * lb.inputHeight;
  std::fill(out, out + 3 * plane, static_cast<float>(kPadValue) / 255.0f);
  for (int y = 0; y < lb.scaledHeight; ++y) {
    const uint8_t* row = s.rgb->data[0] + static_cast<ptrdiff_t>(y) * s.rgb->linesize[0];
    for (int x = 0; x < lb.scaledWidth; ++x) {
      const size_t at = static_cast<size_t>(y) * lb.inputWidth + x;
      out[at] = row[3 * x] / 255.0f;
      out[plane + at] = row[3 * x + 1] / 255.0f;
      out[2 * plane + at] = row[3 * x + 2] / 255.0f;
    }
  }
}

}  // namespace vc
