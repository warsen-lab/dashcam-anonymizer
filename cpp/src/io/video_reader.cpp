#include "io/video_reader.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/hwcontext.h>
}

namespace vc {
namespace {

// 8-bit YUV layouts whose plane 0 can be read directly as luma.
bool isDirect8BitYuv(int format) {
  switch (format) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_NV12:
      return true;
    default:
      return false;
  }
}

AVPixelFormat preferVideoToolbox(AVCodecContext* ctx, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    if (*f == AV_PIX_FMT_VIDEOTOOLBOX) return *f;
  }
  return avcodec_default_get_format(ctx, formats);
}

}  // namespace

HwAccel parseHwAccel(const std::string& text) {
  if (text == "none") return HwAccel::None;
  if (text == "videotoolbox") return HwAccel::VideoToolbox;
  if (text == "auto") return HwAccel::Auto;
  throw std::invalid_argument("unknown hwaccel '" + text + "' (none|videotoolbox|auto)");
}

LumaPlane VideoFrame::luma() const {
  return LumaPlane{av->data[0], av->width, av->height, av->linesize[0]};
}

struct VideoReader::Impl {
  av::InputContextPtr input;
  av::CodecContextPtr decoder;
  av::BufferPtr hwDevice;
  av::SwsPtr sws;
  av::PacketPtr packet = av::allocPacket();
  av::FramePtr decoded = av::allocFrame();
  av::FramePtr downloaded = av::allocFrame();
  av::FramePtr converted = av::allocFrame();
  int streamIndex = -1;
  AVRational timeBase{0, 1};
  int64_t startPts = 0;
  VideoInfo info;
  VideoFrame current;
  int64_t nextIndex = 0;
  int64_t decodeErrors = 0;
  bool sawHardwareFrame = false;
  bool inputDrained = false;
  bool decoderDrained = false;
  std::vector<int> audioIndices;
  std::vector<std::string> dropped;
  std::function<void(const AVPacket&)> audioSink;

  void open(const std::string& path, HwAccel hwAccel);
  void attachHardware(HwAccel hwAccel);
  void feedDecoder();
  const VideoFrame* publish();
  const AVFrame* toCpu8Bit();
};

void VideoReader::Impl::open(const std::string& path, HwAccel hwAccel) {
  AVFormatContext* rawInput = nullptr;
  checkAv(avformat_open_input(&rawInput, path.c_str(), nullptr, nullptr), "open " + path);
  input.reset(rawInput);
  checkAv(avformat_find_stream_info(input.get(), nullptr), "read stream info of " + path);

  const AVCodec* codec = nullptr;
  streamIndex = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (streamIndex < 0 || codec == nullptr) {
    throw VcError("no decodable video stream in " + path);
  }
  const AVStream* stream = input->streams[streamIndex];
  timeBase = stream->time_base;
  startPts = stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time;
  for (unsigned i = 0; i < input->nb_streams; ++i) {
    if (static_cast<int>(i) == streamIndex) continue;
    const AVCodecParameters* par = input->streams[i]->codecpar;
    if (par->codec_type == AVMEDIA_TYPE_AUDIO) {
      audioIndices.push_back(static_cast<int>(i));
      continue;
    }
    const char* type = av_get_media_type_string(par->codec_type);
    dropped.push_back("#" + std::to_string(i) + " " + (type ? type : "unknown") + " " +
                      avcodec_get_name(par->codec_id));
  }

  decoder.reset(avcodec_alloc_context3(codec));
  if (!decoder) throw VcError("avcodec_alloc_context3 failed");
  checkAv(avcodec_parameters_to_context(decoder.get(), stream->codecpar),
          "copy codec parameters");
  decoder->pkt_timebase = timeBase;
  decoder->thread_count = 0;  // FFmpeg picks the thread count
  attachHardware(hwAccel);
  checkAv(avcodec_open2(decoder.get(), codec, nullptr), "open decoder " + std::string(codec->name));

  info.width = decoder->width;
  info.height = decoder->height;
  const AVRational rate = stream->avg_frame_rate;
  info.fps = rate.num > 0 && rate.den > 0 ? av_q2d(rate) : 0.0;
  info.durationMs = input->duration > 0 ? static_cast<double>(input->duration) / 1000.0 : 0.0;
  info.codecName = codec->name;
}

void VideoReader::Impl::attachHardware(HwAccel hwAccel) {
  if (hwAccel == HwAccel::None) return;
  AVBufferRef* device = nullptr;
  const int ret =
      av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_VIDEOTOOLBOX, nullptr, nullptr, 0);
  if (ret < 0) {
    if (hwAccel == HwAccel::VideoToolbox) {
      throw VcError("VideoToolbox unavailable: " + avErrorText(ret));
    }
    return;  // Auto: software decoding
  }
  hwDevice.reset(device);
  decoder->hw_device_ctx = av_buffer_ref(device);
  decoder->get_format = preferVideoToolbox;
}

void VideoReader::Impl::feedDecoder() {
  while (true) {
    const int ret = av_read_frame(input.get(), packet.get());
    if (ret == AVERROR_EOF) {
      checkAv(avcodec_send_packet(decoder.get(), nullptr), "flush decoder");
      inputDrained = true;
      return;
    }
    checkAv(ret, "read packet");
    if (packet->stream_index != streamIndex) {
      const bool audio = std::find(audioIndices.begin(), audioIndices.end(), packet->stream_index) !=
                         audioIndices.end();
      if (audio && audioSink) {
        try {
          audioSink(*packet);
        } catch (...) {
          av_packet_unref(packet.get());
          throw;
        }
      }
      av_packet_unref(packet.get());
      continue;
    }
    const int sent = avcodec_send_packet(decoder.get(), packet.get());
    av_packet_unref(packet.get());
    if (sent == AVERROR_INVALIDDATA) {
      ++decodeErrors;
      return;
    }
    checkAv(sent, "send packet");
    return;
  }
}

const VideoFrame* VideoReader::Impl::publish() {
  const int64_t ts = decoded->best_effort_timestamp;
  current.index = nextIndex++;
  if (ts == AV_NOPTS_VALUE) {
    // Without a frame rate every such frame would get ptsMs 0 and freeze the sampler.
    if (!(info.fps > 0)) {
      throw VcError("frame " + std::to_string(current.index) +
                    " has no timestamp and the stream frame rate is unknown");
    }
    current.ptsMs = static_cast<double>(current.index) * 1000.0 / info.fps;
  } else {
    current.ptsMs = static_cast<double>(ts - startPts) * av_q2d(timeBase) * 1000.0;
  }
  current.av = toCpu8Bit();
  return &current;
}

const AVFrame* VideoReader::Impl::toCpu8Bit() {
  const AVFrame* frame = decoded.get();
  if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
    sawHardwareFrame = true;
    av_frame_unref(downloaded.get());
    checkAv(av_hwframe_transfer_data(downloaded.get(), frame, 0), "download hardware frame");
    checkAv(av_frame_copy_props(downloaded.get(), frame), "copy frame properties");
    frame = downloaded.get();
  }
  if (isDirect8BitYuv(frame->format)) return frame;

  if (!sws) sws = av::allocSws();
  av_frame_unref(converted.get());
  converted->format = AV_PIX_FMT_NV12;
  converted->width = frame->width;
  converted->height = frame->height;
  checkAv(sws_scale_frame(sws.get(), converted.get(), frame), "convert frame to NV12");
  checkAv(av_frame_copy_props(converted.get(), frame), "copy frame properties");
  return converted.get();
}

VideoReader::VideoReader(const std::string& path, HwAccel hwAccel)
    : impl_(std::make_unique<Impl>()) {
  impl_->open(path, hwAccel);
}

VideoReader::~VideoReader() = default;

const VideoInfo& VideoReader::info() const { return impl_->info; }

bool VideoReader::usingHardware() const { return impl_->sawHardwareFrame; }

int64_t VideoReader::decodeErrors() const { return impl_->decodeErrors; }

const AVStream* VideoReader::videoStream() const { return impl_->input->streams[impl_->streamIndex]; }

std::vector<const AVStream*> VideoReader::audioStreams() const {
  std::vector<const AVStream*> streams;
  for (int index : impl_->audioIndices) streams.push_back(impl_->input->streams[index]);
  return streams;
}

std::vector<std::string> VideoReader::droppedStreams() const { return impl_->dropped; }

void VideoReader::setAudioSink(std::function<void(const AVPacket&)> sink) {
  impl_->audioSink = std::move(sink);
}

const VideoFrame* VideoReader::next() {
  Impl& s = *impl_;
  while (!s.decoderDrained) {
    const int ret = avcodec_receive_frame(s.decoder.get(), s.decoded.get());
    if (ret == AVERROR(EAGAIN)) {
      if (s.inputDrained) {
        s.decoderDrained = true;
        break;
      }
      s.feedDecoder();
      continue;
    }
    if (ret == AVERROR_EOF) {
      s.decoderDrained = true;
      break;
    }
    if (ret == AVERROR_INVALIDDATA) {
      ++s.decodeErrors;
      continue;
    }
    checkAv(ret, "decode frame");
    return s.publish();
  }
  return nullptr;
}

}  // namespace vc
