#pragma once

#include <cstdint>
#include <vector>

#include "detect/box.h"
#include "detect/letterbox.h"

namespace vc {

struct DecodeParams {
  float scoreThreshold = 0.15f;
  float nmsIou = 0.5f;
};

// Decodes a raw YOLOv8 output laid out as [4 + numClasses][numAnchors] (the [1, 4 + nc, N] ONNX
// output): rows cx, cy, w, h in model-input pixels, then one score row per class. Each anchor
// takes its best class; anchors scoring below the threshold are dropped; boxes are divided by
// lb.r, clipped to the frame and suppressed per class. Same rules as vc_train.geometry.decode_yolov8.
std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors,
                              const Letterbox& lb, const DecodeParams& params);

}  // namespace vc
