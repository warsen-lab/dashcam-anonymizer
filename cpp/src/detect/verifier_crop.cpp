#include "detect/verifier_crop.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace vc {

CropWindow verifierWindow(const Box& box, double factor, double minPx) {
  const double longSide = std::max(box.width(), box.height());
  const double side = std::max(factor * longSide, minPx);
  const double cx = (static_cast<double>(box.x1) + box.x2) / 2.0;
  const double cy = (static_cast<double>(box.y1) + box.y2) / 2.0;
  return CropWindow{cx - side / 2.0, cy - side / 2.0, side};
}

struct CropSampler::Impl {
  CropParams params;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr view = av::allocFrame();
  av::FramePtr rgb = av::allocFrame();
};

CropSampler::CropSampler(const CropParams& params) : impl_(std::make_unique<Impl>()) {
  if (params.size < 1 || params.padValue < 0 || params.padValue > 255 || !(params.factor > 0) ||
      !(params.minPx >= 0)) {
    throw std::invalid_argument("invalid verifier crop parameters");
  }
  impl_->params = params;
  impl_->sws->flags = SWS_BILINEAR;
}

CropSampler::~CropSampler() = default;

const CropParams& CropSampler::params() const { return impl_->params; }

void CropSampler::fill(const AVFrame& frame, const CropWindow& window, uint8_t* rgb) {
  Impl& s = *impl_;
  const int size = s.params.size;
  std::fill(rgb, rgb + static_cast<size_t>(size) * size * 3, static_cast<uint8_t>(s.params.padValue));
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
  if (desc == nullptr) throw VcError("verifier crop: unknown pixel format");
  // The frame pixels the window covers, the corner moved back onto the chroma grid so the view starts on a
  // chroma sample.
  const int alignX = 1 << desc->log2_chroma_w;
  const int alignY = 1 << desc->log2_chroma_h;
  const auto clampTo = [](double v, int hi) { return static_cast<int>(std::clamp(v, 0.0, static_cast<double>(hi))); };
  const int x0 = clampTo(std::floor(window.x), frame.width) / alignX * alignX;
  const int y0 = clampTo(std::floor(window.y), frame.height) / alignY * alignY;
  const int x1 = clampTo(std::ceil(window.x + window.side), frame.width);
  const int y1 = clampTo(std::ceil(window.y + window.side), frame.height);
  if (x1 <= x0 || y1 <= y0) return;  // wholly outside: all pad
  const int w = x1 - x0;
  const int h = y1 - y0;

  // A view of the frame's [x0, x1) x [y0, y1) without copying: the plane pointers moved to the corner.
  AVFrame& v = *s.view;
  av_frame_unref(&v);
  v.format = frame.format;
  v.width = w;
  v.height = h;
  v.color_range = frame.color_range;
  v.colorspace = frame.colorspace;
  v.color_primaries = frame.color_primaries;
  v.color_trc = frame.color_trc;
  for (int c = 0; c < desc->nb_components; ++c) {
    const AVComponentDescriptor& comp = desc->comp[c];
    const bool chroma = c == 1 || c == 2;
    const int px = chroma ? x0 >> desc->log2_chroma_w : x0;
    const int py = chroma ? y0 >> desc->log2_chroma_h : y0;
    const int p = comp.plane;
    // Interleaved components (NV12's U and V) share a plane: point at the first sample of the pair, not the
    // component's own byte.
    v.data[p] = frame.data[p] + static_cast<ptrdiff_t>(py) * frame.linesize[p] + static_cast<ptrdiff_t>(px) * comp.step;
    v.linesize[p] = frame.linesize[p];
  }
  av_frame_unref(s.rgb.get());
  s.rgb->format = AV_PIX_FMT_RGB24;
  s.rgb->width = w;
  s.rgb->height = h;
  checkAv(sws_scale_frame(s.sws.get(), s.rgb.get(), &v), "convert verifier crop to RGB");

  // Bilinear resampling: output pixel (ox, oy) shows frame point (window.x + (ox + 0.5) * scale - 0.5, ...).
  const double scale = window.side / size;
  const uint8_t* src = s.rgb->data[0];
  const int stride = s.rgb->linesize[0];
  for (int oy = 0; oy < size; ++oy) {
    const double fy = window.y + (oy + 0.5) * scale - 0.5;
    if (fy < -0.5 || fy >= frame.height - 0.5) continue;
    const double ly = std::clamp(fy - y0, 0.0, static_cast<double>(h - 1));
    const int ya = static_cast<int>(ly);
    const int yb = std::min(ya + 1, h - 1);
    const double ty = ly - ya;
    for (int ox = 0; ox < size; ++ox) {
      const double fx = window.x + (ox + 0.5) * scale - 0.5;
      if (fx < -0.5 || fx >= frame.width - 0.5) continue;
      const double lx = std::clamp(fx - x0, 0.0, static_cast<double>(w - 1));
      const int xa = static_cast<int>(lx);
      const int xb = std::min(xa + 1, w - 1);
      const double tx = lx - xa;
      for (int c = 0; c < 3; ++c) {
        const double a = src[ya * stride + xa * 3 + c] * (1 - tx) + src[ya * stride + xb * 3 + c] * tx;
        const double b = src[yb * stride + xa * 3 + c] * (1 - tx) + src[yb * stride + xb * 3 + c] * tx;
        rgb[(static_cast<size_t>(oy) * size + ox) * 3 + c] = static_cast<uint8_t>(std::lround(a * (1 - ty) + b * ty));
      }
    }
  }
}

void toPlanarFloat(const uint8_t* rgb, int size, float* out) {
  const size_t plane = static_cast<size_t>(size) * size;
  for (size_t i = 0; i < plane; ++i) {
    out[i] = rgb[i * 3] / 255.0f;
    out[plane + i] = rgb[i * 3 + 1] / 255.0f;
    out[2 * plane + i] = rgb[i * 3 + 2] / 255.0f;
  }
}

}  // namespace vc
