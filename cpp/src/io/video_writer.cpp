#include "io/video_writer.h"

#include <cstring>
#include <string>
#include <utility>

#include "common/vc_error.h"
#include "io/av_ptr.h"

namespace vc {
namespace {

constexpr int64_t kDefaultBitRate = 8'000'000;

struct OutputDeleter {
  void operator()(AVFormatContext* p) const {
    if (p == nullptr) return;
    if (p->pb != nullptr && (p->oformat->flags & AVFMT_NOFILE) == 0) avio_closep(&p->pb);
    avformat_free_context(p);
  }
};

// `wanted` when the encoder takes it, else yuv420p, else the encoder's first format (libx265 has no NV12).
AVPixelFormat encoderFormat(const AVCodecContext* ctx, const AVCodec* codec, AVPixelFormat wanted) {
  const void* configs = nullptr;
  int count = 0;
  if (avcodec_get_supported_config(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0 ||
      configs == nullptr || count == 0) {
    return wanted;  // the encoder accepts any format
  }
  const auto* formats = static_cast<const AVPixelFormat*>(configs);
  for (int i = 0; i < count; ++i) {
    if (formats[i] == wanted) return wanted;
  }
  for (int i = 0; i < count; ++i) {
    if (formats[i] == AV_PIX_FMT_YUV420P) return AV_PIX_FMT_YUV420P;
  }
  return formats[0];
}

std::vector<std::string> encoderCandidates(const WriterConfig& config) {
  const bool hevc = config.sourceCodec == "hevc";
  std::vector<std::string> names;
  if (config.preferHardware) names.emplace_back(hevc ? "hevc_videotoolbox" : "h264_videotoolbox");
  names.emplace_back(hevc ? "libx265" : "libx264");
  return names;
}

bool validRatio(AVRational r) { return r.num > 0 && r.den > 0; }

}  // namespace

void copyPresentation(const AVStream& videoStream, WriterConfig& config) {
  const AVCodecParameters& par = *videoStream.codecpar;
  config.sampleAspectRatio = validRatio(videoStream.sample_aspect_ratio) ? videoStream.sample_aspect_ratio
                             : validRatio(par.sample_aspect_ratio)       ? par.sample_aspect_ratio
                                                                         : AVRational{0, 1};
  config.colorRange = par.color_range;
  config.colorSpace = par.color_space;
  config.colorPrimaries = par.color_primaries;
  config.colorTrc = par.color_trc;
  config.displayMatrix.reset();
  const AVPacketSideData* matrix =
      av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
  if (matrix != nullptr && matrix->size >= sizeof(int32_t) * 9) {
    std::array<int32_t, 9> values{};
    std::memcpy(values.data(), matrix->data, sizeof(int32_t) * 9);
    config.displayMatrix = values;
  }
}

struct VideoWriter::Impl {
  WriterConfig config;
  std::unique_ptr<AVFormatContext, OutputDeleter> output;
  av::CodecContextPtr encoder;
  std::string encoderName;
  AVStream* videoStream = nullptr;
  std::vector<AVStream*> audioOut;
  std::vector<AVRational> audioIn;
  av::PacketPtr packet = av::allocPacket();
  av::SwsPtr sws;                              // only when the encoder needs another pixel format
  av::FramePtr converted = av::allocFrame();
  int64_t frames = 0;
  bool finished = false;

  void openEncoder() {
    const AVRational rate = config.frameRate.num > 0 && config.frameRate.den > 0 ? config.frameRate : AVRational{30, 1};
    std::string errors;
    for (const std::string& name : encoderCandidates(config)) {
      const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
      if (codec == nullptr) {
        errors += name + ": not available; ";
        continue;
      }
      av::CodecContextPtr ctx(avcodec_alloc_context3(codec));
      if (!ctx) throw VcError("avcodec_alloc_context3 failed");
      ctx->width = config.width;
      ctx->height = config.height;
      ctx->pix_fmt = encoderFormat(ctx.get(), codec, config.pixelFormat);
      ctx->time_base = config.timeBase;
      ctx->framerate = rate;
      ctx->bit_rate = config.bitRate > 0 ? config.bitRate : kDefaultBitRate;
      if (config.sampleAspectRatio.num > 0 && config.sampleAspectRatio.den > 0) {
        ctx->sample_aspect_ratio = config.sampleAspectRatio;
      }
      ctx->color_range = config.colorRange;
      ctx->colorspace = config.colorSpace;
      ctx->color_primaries = config.colorPrimaries;
      ctx->color_trc = config.colorTrc;
      if ((output->oformat->flags & AVFMT_GLOBALHEADER) != 0) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      const int ret = avcodec_open2(ctx.get(), codec, nullptr);
      if (ret < 0) {
        errors += name + ": " + avErrorText(ret) + "; ";
        continue;
      }
      encoder = std::move(ctx);
      encoderName = name;
      return;
    }
    throw VcError("no usable video encoder (" + errors + ")");
  }

  void drain() {
    while (true) {
      const int ret = avcodec_receive_packet(encoder.get(), packet.get());
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
      checkAv(ret, "receive encoded packet");
      av_packet_rescale_ts(packet.get(), encoder->time_base, videoStream->time_base);
      packet->stream_index = videoStream->index;
      checkAv(av_interleaved_write_frame(output.get(), packet.get()), "write video packet");
    }
  }
};


VideoWriter::VideoWriter(const WriterConfig& config, const std::vector<const AVStream*>& audio)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.config = config;
  const std::string path = config.path.string();
  AVFormatContext* raw = nullptr;
  checkAv(avformat_alloc_output_context2(&raw, nullptr, "mp4", path.c_str()), "create MP4 muxer");
  s.output.reset(raw);
  // Before anything is opened or created: a codec MP4 cannot hold would only fail at the header.
  for (const AVStream* in : audio) {
    const AVCodecID id = in->codecpar->codec_id;
    if (avformat_query_codec(s.output->oformat, id, FF_COMPLIANCE_NORMAL) == 0) {
      throw VcError(std::string("audio codec ") + avcodec_get_name(id) + " cannot be stored in MP4");
    }
  }
  s.openEncoder();

  s.videoStream = avformat_new_stream(s.output.get(), nullptr);
  if (s.videoStream == nullptr) throw VcError("avformat_new_stream failed");
  checkAv(avcodec_parameters_from_context(s.videoStream->codecpar, s.encoder.get()), "copy encoder parameters");
  if (s.encoder->codec_id == AV_CODEC_ID_HEVC) s.videoStream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
  s.videoStream->time_base = s.encoder->time_base;
  s.videoStream->avg_frame_rate = s.encoder->framerate;
  s.videoStream->sample_aspect_ratio = s.encoder->sample_aspect_ratio;
  if (config.displayMatrix) {
    AVCodecParameters& par = *s.videoStream->codecpar;
    AVPacketSideData* matrix = av_packet_side_data_new(&par.coded_side_data, &par.nb_coded_side_data,
                                                       AV_PKT_DATA_DISPLAYMATRIX, sizeof(int32_t) * 9, 0);
    if (matrix == nullptr) throw VcError("cannot store the display matrix");
    std::memcpy(matrix->data, config.displayMatrix->data(), sizeof(int32_t) * 9);
  }

  for (const AVStream* in : audio) {
    AVStream* out = avformat_new_stream(s.output.get(), nullptr);
    if (out == nullptr) throw VcError("avformat_new_stream failed");
    checkAv(avcodec_parameters_copy(out->codecpar, in->codecpar), "copy audio parameters");
    out->codecpar->codec_tag = 0;
    out->time_base = in->time_base;
    s.audioOut.push_back(out);
    s.audioIn.push_back(in->time_base);
  }

  checkAv(avio_open(&s.output->pb, path.c_str(), AVIO_FLAG_WRITE), "open " + path);
  checkAv(avformat_write_header(s.output.get(), nullptr), "write MP4 header");
}

VideoWriter::~VideoWriter() = default;

const std::string& VideoWriter::encoderName() const { return impl_->encoderName; }

int64_t VideoWriter::videoFrames() const { return impl_->frames; }

void VideoWriter::writeVideo(AVFrame& frame) {
  Impl& s = *impl_;
  if (s.finished) throw VcError("writeVideo after finish");
  // The encoder reads width x height pixels from the frame whatever its real size is.
  if (frame.width != s.config.width || frame.height != s.config.height) {
    throw VcError("frame is " + std::to_string(frame.width) + "x" + std::to_string(frame.height) +
                  ", the writer was set up for " + std::to_string(s.config.width) + "x" +
                  std::to_string(s.config.height));
  }
  frame.pict_type = AV_PICTURE_TYPE_NONE;
  AVFrame* toSend = &frame;
  if (frame.format != s.encoder->pix_fmt) {
    if (!s.sws) s.sws = av::allocSws();
    av_frame_unref(s.converted.get());
    s.converted->format = s.encoder->pix_fmt;
    s.converted->width = frame.width;
    s.converted->height = frame.height;
    // Only the pixel format changes: without these the conversion would assume the limited range
    // and squeeze full-range pictures, while the output is tagged with the source's range.
    s.converted->color_range = frame.color_range;
    s.converted->colorspace = frame.colorspace;
    s.converted->color_primaries = frame.color_primaries;
    s.converted->color_trc = frame.color_trc;
    checkAv(sws_scale_frame(s.sws.get(), s.converted.get(), &frame), "convert frame for the encoder");
    checkAv(av_frame_copy_props(s.converted.get(), &frame), "copy frame properties");
    toSend = s.converted.get();
  }
  checkAv(avcodec_send_frame(s.encoder.get(), toSend), "encode frame");
  ++s.frames;
  s.drain();
}

void VideoWriter::writeAudio(const AVPacket& packet, size_t ordinal) {
  Impl& s = *impl_;
  if (s.finished) throw VcError("writeAudio after finish");
  if (ordinal >= s.audioOut.size()) throw VcError("unknown audio stream " + std::to_string(ordinal));
  av::PacketPtr copy = av::allocPacket();
  checkAv(av_packet_ref(copy.get(), &packet), "reference audio packet");
  av_packet_rescale_ts(copy.get(), s.audioIn[ordinal], s.audioOut[ordinal]->time_base);
  copy->stream_index = s.audioOut[ordinal]->index;
  copy->pos = -1;
  checkAv(av_interleaved_write_frame(s.output.get(), copy.get()), "write audio packet");
}

void VideoWriter::finish() {
  Impl& s = *impl_;
  if (s.finished) return;
  checkAv(avcodec_send_frame(s.encoder.get(), nullptr), "flush encoder");
  s.drain();
  checkAv(av_write_trailer(s.output.get()), "write MP4 trailer");
  checkAv(avio_closep(&s.output->pb), "close output");
  s.finished = true;
}

}  // namespace vc
