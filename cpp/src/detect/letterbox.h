#pragma once

#include <memory>

struct AVFrame;

namespace vc {

inline constexpr int kPadValue = 114;

// Frame -> model input mapping (proxy-validation spec §4.1): the frame is scaled by r, placed at
// the top-left corner and the rest of the input is filled with kPadValue.
struct Letterbox {
  int frameWidth = 0;
  int frameHeight = 0;
  int inputWidth = 0;
  int inputHeight = 0;
  int scaledWidth = 0;   // round(frameWidth * r)
  int scaledHeight = 0;  // round(frameHeight * r)
  double r = 1.0;        // min(inputWidth / frameWidth, inputHeight / frameHeight)
};

// Throws std::invalid_argument for non-positive sizes and when a scaled size rounds to 0 (an extreme
// aspect ratio would otherwise fail late, inside the conversion).
Letterbox makeLetterbox(int frameWidth, int frameHeight, int inputWidth, int inputHeight);

// Converts 8-bit YUV frames of one size into the model's [1, 3, H, W] float input (RGB / 255).
class InputTensorFiller {
 public:
  explicit InputTensorFiller(const Letterbox& letterbox);
  ~InputTensorFiller();
  InputTensorFiller(const InputTensorFiller&) = delete;
  InputTensorFiller& operator=(const InputTensorFiller&) = delete;

  // out must hold 3 * inputWidth * inputHeight floats (planes R, G, B). Throws VcError when the
  // frame size differs from the letterbox's or the conversion fails.
  void fill(const AVFrame& frame, float* out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
