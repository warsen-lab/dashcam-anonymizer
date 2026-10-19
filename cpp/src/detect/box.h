#pragma once

#include <vector>

namespace vc {

// Axis-aligned box in frame pixels; cls indexes the detector's class list.
struct Box {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  float score = 0;

  float width() const { return x2 > x1 ? x2 - x1 : 0.0f; }
  float height() const { return y2 > y1 ? y2 - y1 : 0.0f; }
  float area() const { return width() * height(); }
};

float iou(const Box& a, const Box& b);

// Greedy non-maximum suppression within each class, highest score first (ties keep input
// order). A box is suppressed when its IoU with a kept box of its class is >= iouThreshold.
std::vector<Box> nmsPerClass(std::vector<Box> boxes, float iouThreshold);

}  // namespace vc
