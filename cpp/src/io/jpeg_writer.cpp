#include "io/jpeg_writer.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {

struct JpegWriter::Impl {
  int qscale = kBestJpegQscale;
  av::CodecContextPtr encoder;
  av::SwsPtr sws = av::allocSws();
  av::FramePtr yuv = av::allocFrame();
  av::PacketPtr packet = av::allocPacket();

  void ensureEncoder(int width, int height);
};

void JpegWriter::Impl::ensureEncoder(int width, int height) {
  if (encoder && encoder->width == width && encoder->height == height) return;
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  if (codec == nullptr) throw VcError("MJPEG encoder not available");
  av::CodecContextPtr fresh(avcodec_alloc_context3(codec));
  if (!fresh) throw VcError("avcodec_alloc_context3 failed");
  fresh->width = width;
  fresh->height = height;
  fresh->pix_fmt = AV_PIX_FMT_YUV420P;
  fresh->color_range = AVCOL_RANGE_JPEG;
  fresh->time_base = AVRational{1, 25};
  fresh->flags |= AV_CODEC_FLAG_QSCALE;
  fresh->global_quality = qscale * FF_QP2LAMBDA;
  checkAv(avcodec_open2(fresh.get(), codec, nullptr), "open MJPEG encoder");
  // Publish only a fully opened encoder so a failed open never leaves a stale context behind.
  encoder = std::move(fresh);
}

JpegWriter::JpegWriter(int qscale) : impl_(std::make_unique<Impl>()) {
  if (qscale < kBestJpegQscale || qscale > kWorstJpegQscale) {
    throw std::invalid_argument("JPEG qscale must be in [2, 31], got " + std::to_string(qscale));
  }
  impl_->qscale = qscale;
}

JpegWriter::~JpegWriter() = default;

void JpegWriter::write(const AVFrame& frame, const std::filesystem::path& path) {
  Impl& s = *impl_;
  s.ensureEncoder(frame.width, frame.height);

  av_frame_unref(s.yuv.get());
  s.yuv->format = AV_PIX_FMT_YUV420P;
  s.yuv->width = frame.width;
  s.yuv->height = frame.height;
  s.yuv->color_range = AVCOL_RANGE_JPEG;
  checkAv(sws_scale_frame(s.sws.get(), s.yuv.get(), &frame), "convert frame for JPEG");
  // MJPEG with AV_CODEC_FLAG_QSCALE takes its quantiser from the frame.
  s.yuv->quality = s.encoder->global_quality;
  // Source P/B picture types must not leak into the intra-only encoder.
  s.yuv->pict_type = AV_PICTURE_TYPE_NONE;

  checkAv(avcodec_send_frame(s.encoder.get(), s.yuv.get()), "encode JPEG");
  checkAv(avcodec_receive_packet(s.encoder.get(), s.packet.get()), "receive JPEG packet");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(s.packet->data), s.packet->size);
  av_packet_unref(s.packet.get());
  // Close explicitly so a flush failure is reported here rather than swallowed in the destructor.
  out.close();
  if (!out) throw VcError("write " + path.string() + " failed");
}

}  // namespace vc
