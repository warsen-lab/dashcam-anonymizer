#pragma once

#include <memory>

#include "common/vc_error.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

// RAII owners for FFmpeg objects. Internal to src/io.
namespace vc::av {

struct FrameDeleter {
  void operator()(AVFrame* p) const { av_frame_free(&p); }
};
struct PacketDeleter {
  void operator()(AVPacket* p) const { av_packet_free(&p); }
};
struct CodecContextDeleter {
  void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
};
struct InputContextDeleter {
  void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
};
struct SwsDeleter {
  void operator()(SwsContext* p) const { sws_freeContext(p); }
};
struct BufferDeleter {
  void operator()(AVBufferRef* p) const { av_buffer_unref(&p); }
};

using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using InputContextPtr = std::unique_ptr<AVFormatContext, InputContextDeleter>;
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferDeleter>;

inline FramePtr allocFrame() {
  FramePtr frame(av_frame_alloc());
  if (!frame) throw VcError("av_frame_alloc failed");
  return frame;
}

inline PacketPtr allocPacket() {
  PacketPtr packet(av_packet_alloc());
  if (!packet) throw VcError("av_packet_alloc failed");
  return packet;
}

inline SwsPtr allocSws() {
  SwsPtr sws(sws_alloc_context());
  if (!sws) throw VcError("sws_alloc_context failed");
  return sws;
}

}  // namespace vc::av
