#pragma once

#include <array>
#include <vector>

#include "detect/box.h"

namespace vc {

struct YuNetParams {
  float scoreThreshold = 0.3f;
  float nmsIou = 0.3f;
  int topK = 5000;  // candidates NMS looks at, highest scores first (cv2.FaceDetectorYN's default)
};

// The strides of YuNet's three output levels.
inline constexpr std::array<int, 3> kYuNetStrides = {8, 16, 32};

// The raw outputs of one level (the [1, N, k] tensors cls_<s>, obj_<s>, bbox_<s>): the anchor at row
// r, column c of a stride-s level has index idx = r * (padWidth / s) + c and reads cls[idx],
// obj[idx] and bbox[4 idx .. 4 idx + 3].
struct YuNetLevel {
  const float* cls = nullptr;
  const float* obj = nullptr;
  const float* bbox = nullptr;
};

// Decodes YuNet's outputs for a padWidth x padHeight input (levels[i] is stride kYuNetStrides[i]) into
// face boxes (cls 0) in a frameWidth x frameHeight frame, highest score first. Same rules as
// cv2.FaceDetectorYN and vc_train.yunet.decode_yunet: an anchor scores sqrt(clamp(cls, 0, 1) *
// clamp(obj, 0, 1)) and is kept when the score is >= the threshold; its box is centred at
// ((c + b0) s, (r + b1) s) and is exp(b2) s wide, exp(b3) s high. NMS (cv::dnn::NMSBoxes) sorts by
// score (stable), keeps the topK best and suppresses a box whose overlap with a kept one is above
// nmsIou, measured on the boxes truncated to integer rectangles as OpenCV's cv::Rect2i does. Boxes are
// clipped to the frame after NMS. One deliberate difference: NMSBoxes also drops a score exactly equal
// to the threshold when there are several candidates; this keeps it.
//
// A broken model must stop the run, not lose a mask: throws VcError when ANY cls or obj value is not
// finite, and when a kept anchor has a non-finite bbox value or box (exp overflowing to infinity).
//
// Preconditions (the caller checks the output shapes): padWidth and padHeight are positive multiples
// of 32 and every level holds (padWidth / s) * (padHeight / s) anchors.
std::vector<Box> decodeYuNet(const std::array<YuNetLevel, 3>& levels, int padWidth, int padHeight, int frameWidth,
                             int frameHeight, const YuNetParams& params);

}  // namespace vc
