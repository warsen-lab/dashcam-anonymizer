#pragma once

#include <cstdint>
#include <memory>

#include "detect/box.h"

struct AVFrame;

namespace vc {

// The square of frame pixels a verifier crop shows (spec 2026-10-21 §3.2): centred on the box, side
// max(factor * the box's long side, minPx). It may reach past the frame.
struct CropWindow {
  double x = 0;  // left edge, frame pixels
  double y = 0;  // top edge
  double side = 0;
};

// What a verifier crop is: size x size RGB, the window grown by `factor`, at least `minPx` frame pixels, the
// part outside the frame filled with padValue. The defaults are spec 2026-10-21 §3.2's.
struct CropParams {
  int size = 64;
  double factor = 2.5;
  double minPx = 32;
  int padValue = 114;
};

CropWindow verifierWindow(const Box& box, double factor, double minPx);

// Cuts verifier crops out of 8-bit YUV frames: the window's part inside the frame is converted to RGB24 at the
// frame's own resolution (libswscale, the frame's colour space and range, as the detector's input), then
// resampled bilinearly to size x size; output pixels whose centre falls outside the frame get padValue. Not
// thread-safe.
class CropSampler {
 public:
  // Throws std::invalid_argument for a size < 1, a padValue outside [0, 255], a factor not > 0 or minPx < 0.
  explicit CropSampler(const CropParams& params);
  ~CropSampler();
  CropSampler(const CropSampler&) = delete;
  CropSampler& operator=(const CropSampler&) = delete;

  const CropParams& params() const;
  // rgb: size * size * 3 bytes, interleaved R, G, B, row by row. Throws VcError when the conversion fails.
  void fill(const AVFrame& frame, const CropWindow& window, uint8_t* rgb);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// rgb (size * size * 3, interleaved) -> out (3 * size * size: planes R, G, B), each byte / 255: the verifier's input.
void toPlanarFloat(const uint8_t* rgb, int size, float* out);

}  // namespace vc
