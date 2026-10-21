#include "io/video_writer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "io/av_ptr.h"

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

constexpr int64_t kDefaultBitRate = 8'000'000;
constexpr int kMacroblockPx = 16;
// h264_videotoolbox's level option starts at 3.0 (1.3 is Baseline only) and ends at 5.2.
constexpr int kVideoToolboxMinLevel = 30;
constexpr int kVideoToolboxMaxLevel = 52;

// Constant-quality settings (RateControl::Quality). VideoToolbox and libx264 are the values measured on a 1080p
// dashcam video: SSIM Y 0.985 / 0.986 against the source, where re-encoding at the source's own bit rate gave
// VideoToolbox 0.92, visibly soft.
constexpr int kVideoToolboxQuality = 65;  // ffmpeg's -q:v scale, 1-100
constexpr double kSoftwareCrf = 18;
// h264_nvenc: variable bit rate steered by a constant quantizer (cq, 0-51, lower is better) with no bit rate. 19 is
// PROVISIONAL, picked to land near libx264 crf 18; to be calibrated against SSIM on the target machine (GPU plan 1c).
constexpr double kNvencCq = 19;

// ITU-T H.264 Table A-1: level_idc, MaxMBPS (macroblocks per second), MaxFS (macroblocks per frame). Level 1b has
// the limits of level 1 and is left out.
struct H264Level {
  int idc = 0;
  double maxMacroblocksPerSecond = 0;
  int64_t maxFrameMacroblocks = 0;
};
constexpr std::array<H264Level, 19> kH264Levels = {{
    {10, 1485, 99},       {11, 3000, 396},       {12, 6000, 396},       {13, 11880, 396},     {20, 11880, 396},
    {21, 19800, 792},     {22, 20250, 1620},     {30, 40500, 1620},     {31, 108000, 3600},   {32, 216000, 5120},
    {40, 245760, 8192},   {41, 245760, 8192},    {42, 522240, 8704},    {50, 589824, 22080},  {51, 983040, 36864},
    {52, 2073600, 36864}, {60, 4177920, 139264}, {61, 8355840, 139264}, {62, 16711680, 139264},
}};

bool is8Bit420(AVPixelFormat format) {
  return format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUVJ420P || format == AV_PIX_FMT_NV12;
}

struct OutputDeleter {
  void operator()(AVFormatContext* p) const {
    if (p == nullptr) return;
    if (p->pb != nullptr && (p->oformat->flags & AVFMT_NOFILE) == 0) avio_closep(&p->pb);
    avformat_free_context(p);
  }
};

// The 8-bit 4:2:0 format the encoder gets: `wanted` when it is one and the encoder takes it, else NV12, else
// yuv420p; AV_PIX_FMT_NONE when the encoder takes neither.
AVPixelFormat encoderFormat(const AVCodecContext* ctx, const AVCodec* codec, AVPixelFormat wanted) {
  const void* configs = nullptr;
  int count = 0;
  if (avcodec_get_supported_config(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0 ||
      configs == nullptr || count == 0) {
    return is8Bit420(wanted) ? wanted : AV_PIX_FMT_YUV420P;  // the encoder accepts any format
  }
  const auto* formats = static_cast<const AVPixelFormat*>(configs);
  const auto takes = [&](AVPixelFormat f) { return std::find(formats, formats + count, f) != formats + count; };
  if (is8Bit420(wanted) && takes(wanted)) return wanted;
  if (takes(AV_PIX_FMT_NV12)) return AV_PIX_FMT_NV12;
  if (takes(AV_PIX_FMT_YUV420P)) return AV_PIX_FMT_YUV420P;
  return AV_PIX_FMT_NONE;
}

// Encoder-specific options; the reason it cannot be used, or an empty string. The rate control comes first: config.bitRate
// as the bit rate (RateControl::Bitrate), or constant quality, which leaves the bit rate unset (RateControl::Quality).
// h264_videotoolbox picks levels below what the frame rate needs (4.0 for 1080p at 59.94 fps, 5.1 for 4K60), so High
// profile and the level are set; h264_nvenc gets the preset and the level; libx264 chooses a sufficient level by itself.
std::string setEncoderOptions(AVCodecContext& ctx, const std::string& name, const WriterConfig& config, int level) {
  const auto failed = [&](int ret, const char* what) { return std::string(what) + ": " + avErrorText(ret); };
  if (config.rateControl == RateControl::Bitrate) {
    ctx.bit_rate = config.bitRate > 0 ? config.bitRate : kDefaultBitRate;
  } else if (name == "h264_videotoolbox") {
    ctx.flags |= AV_CODEC_FLAG_QSCALE;
    ctx.global_quality = kVideoToolboxQuality * FF_QP2LAMBDA;
  } else if (name == "h264_nvenc") {
    ctx.bit_rate = 0;  // cq governs
    if (const int ret = av_opt_set(ctx.priv_data, "rc", "vbr", 0); ret < 0) return failed(ret, "rc vbr");
    if (const int ret = av_opt_set_double(ctx.priv_data, "cq", kNvencCq, 0); ret < 0) return failed(ret, "cq");
  } else if (const int ret = av_opt_set_double(ctx.priv_data, "crf", kSoftwareCrf, 0); ret < 0) {
    return failed(ret, "crf");
  }
  if (name == "h264_videotoolbox") {
    if (const int ret = av_opt_set(ctx.priv_data, "profile", "high", 0); ret < 0) return failed(ret, "profile high");
    const int vtLevel =
        std::clamp(level > 0 ? level : kVideoToolboxMaxLevel, kVideoToolboxMinLevel, kVideoToolboxMaxLevel);
    if (const int ret = av_opt_set_int(ctx.priv_data, "level", vtLevel, 0); ret < 0) return failed(ret, "level");
  } else if (name == "h264_nvenc") {
    if (const int ret = av_opt_set(ctx.priv_data, "preset", config.nvencPreset.c_str(), 0); ret < 0) {
      return failed(ret, "preset");
    }
    if (level > 0) {
      if (const int ret = av_opt_set_int(ctx.priv_data, "level", level, 0); ret < 0) return failed(ret, "level");
    }
  }
  return {};
}

bool validRatio(AVRational r) { return r.num > 0 && r.den > 0; }

}  // namespace

std::vector<std::string> h264EncoderCandidates(bool preferHardware, Platform platform) {
  std::vector<std::string> names;
  if (preferHardware) {
    if (platform != Platform::MacOS) names.emplace_back("h264_nvenc");
    if (platform == Platform::MacOS) names.emplace_back("h264_videotoolbox");
  }
  names.emplace_back("libx264");
  return names;
}

int64_t h264OutputBitRate(bool sourceIsH264, int64_t sourceBitRate, double nonH264Factor) {
  const int64_t base = sourceBitRate > 0 ? sourceBitRate : kDefaultBitRate;
  if (sourceIsH264) return base;
  return std::llround(static_cast<double>(base) * nonH264Factor);
}

int h264LevelIdc(int width, int height, double fps) {
  const int64_t frameMacroblocks = static_cast<int64_t>((width + kMacroblockPx - 1) / kMacroblockPx) *
                                   static_cast<int64_t>((height + kMacroblockPx - 1) / kMacroblockPx);
  for (const H264Level& level : kH264Levels) {
    if (frameMacroblocks <= level.maxFrameMacroblocks &&
        static_cast<double>(frameMacroblocks) * fps <= level.maxMacroblocksPerSecond) {
      return level.idc;
    }
  }
  return 0;
}

std::optional<std::string> h264FormatNote(AVPixelFormat source) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(source);
  if (desc == nullptr || is8Bit420(source)) return std::nullopt;
  std::string layout;
  if ((desc->flags & AV_PIX_FMT_FLAG_RGB) != 0) {
    layout = "RGB";
  } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
    layout = "4:2:0";
  } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
    layout = "4:2:2";
  } else if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
    layout = "4:4:4";
  } else {
    layout = "subsampled " + std::to_string(1 << desc->log2_chroma_w) + "x" + std::to_string(1 << desc->log2_chroma_h);
  }
  return std::string("the source video is ") + desc->name + " (" + std::to_string(desc->comp[0].depth) + "-bit " +
         layout + "); the H.264 output is 8-bit 4:2:0";
}

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

AudioStreamInfo snapshotAudioStream(const AVStream& stream) {
  AVCodecParameters* copy = avcodec_parameters_alloc();
  if (copy == nullptr) throw VcError("avcodec_parameters_alloc failed");
  std::shared_ptr<AVCodecParameters> owned(copy, [](AVCodecParameters* p) { avcodec_parameters_free(&p); });
  checkAv(avcodec_parameters_copy(copy, stream.codecpar), "copy audio parameters");
  return AudioStreamInfo{std::move(owned), stream.time_base};
}

std::vector<AudioStreamInfo> snapshotAudioStreams(const std::vector<const AVStream*>& streams) {
  std::vector<AudioStreamInfo> out;
  out.reserve(streams.size());
  for (const AVStream* stream : streams) out.push_back(snapshotAudioStream(*stream));
  return out;
}

void requireMp4Audio(const std::vector<AudioStreamInfo>& audio) {
  const AVOutputFormat* mp4 = av_guess_format("mp4", nullptr, nullptr);
  if (mp4 == nullptr) throw VcError("the MP4 muxer is not available");
  for (const AudioStreamInfo& in : audio) {
    const AVCodecID id = in.parameters->codec_id;
    if (avformat_query_codec(mp4, id, FF_COMPLIANCE_NORMAL) == 0) {
      throw VcError(std::string("audio codec ") + avcodec_get_name(id) + " cannot be stored in MP4");
    }
  }
}

void checkEncodedPackets(int64_t frames, int64_t packets) {
  if (packets != frames) {
    throw VcError("the video encoder returned " + std::to_string(packets) + " packets for " + std::to_string(frames) +
                  " frames");
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
  int64_t packets = 0;
  bool finished = false;

  void openEncoder() {
    const AVRational rate = config.frameRate.num > 0 && config.frameRate.den > 0 ? config.frameRate : AVRational{30, 1};
    const int level = h264LevelIdc(config.width, config.height, av_q2d(rate));
    std::vector<std::pair<std::string, std::string>> failed;  // encoder, reason
    for (const std::string& name : h264EncoderCandidates(config.preferHardware, buildPlatform())) {
      const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
      if (codec == nullptr) {
        failed.emplace_back(name, "not available");
        continue;
      }
      av::CodecContextPtr ctx(avcodec_alloc_context3(codec));
      if (!ctx) throw VcError("avcodec_alloc_context3 failed");
      ctx->width = config.width;
      ctx->height = config.height;
      ctx->pix_fmt = encoderFormat(ctx.get(), codec, config.pixelFormat);
      if (ctx->pix_fmt == AV_PIX_FMT_NONE) {
        failed.emplace_back(name, "takes no 8-bit 4:2:0 input");
        continue;
      }
      ctx->time_base = config.timeBase;
      ctx->framerate = rate;
      if (config.sampleAspectRatio.num > 0 && config.sampleAspectRatio.den > 0) {
        ctx->sample_aspect_ratio = config.sampleAspectRatio;
      }
      ctx->color_range = config.colorRange;
      ctx->colorspace = config.colorSpace;
      ctx->color_primaries = config.colorPrimaries;
      ctx->color_trc = config.colorTrc;
      if ((output->oformat->flags & AVFMT_GLOBALHEADER) != 0) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      if (const std::string why = setEncoderOptions(*ctx, name, config, level); !why.empty()) {
        failed.emplace_back(name, why);
        continue;
      }
      const int ret = avcodec_open2(ctx.get(), codec, nullptr);
      if (ret < 0) {
        failed.emplace_back(name, avErrorText(ret));
        continue;
      }
      encoder = std::move(ctx);
      encoderName = name;
      for (const auto& [other, reason] : failed) {
        std::fprintf(stderr, "vc: video encoder %s could not be opened (%s); using %s\n", other.c_str(),
                     reason.c_str(), name.c_str());
      }
      return;
    }
    std::string errors;
    for (const auto& [other, reason] : failed) errors += other + ": " + reason + "; ";
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
      ++packets;
    }
  }
};


VideoWriter::VideoWriter(const WriterConfig& config, const std::vector<AudioStreamInfo>& audio)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.config = config;
  const std::string path = config.path.string();
  AVFormatContext* raw = nullptr;
  checkAv(avformat_alloc_output_context2(&raw, nullptr, "mp4", path.c_str()), "create MP4 muxer");
  s.output.reset(raw);
  // Before anything is opened or created: a codec MP4 cannot hold would only fail at the header.
  requireMp4Audio(audio);
  s.openEncoder();

  s.videoStream = avformat_new_stream(s.output.get(), nullptr);
  if (s.videoStream == nullptr) throw VcError("avformat_new_stream failed");
  checkAv(avcodec_parameters_from_context(s.videoStream->codecpar, s.encoder.get()), "copy encoder parameters");
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

  for (const AudioStreamInfo& in : audio) {
    AVStream* out = avformat_new_stream(s.output.get(), nullptr);
    if (out == nullptr) throw VcError("avformat_new_stream failed");
    checkAv(avcodec_parameters_copy(out->codecpar, in.parameters.get()), "copy audio parameters");
    out->codecpar->codec_tag = 0;
    out->time_base = in.timeBase;
    s.audioOut.push_back(out);
    s.audioIn.push_back(in.timeBase);
  }

  checkAv(avio_open(&s.output->pb, path.c_str(), AVIO_FLAG_WRITE), "open " + path);
  checkAv(avformat_write_header(s.output.get(), nullptr), "write MP4 header");
}

VideoWriter::~VideoWriter() = default;

const std::string& VideoWriter::encoderName() const { return impl_->encoderName; }

int64_t VideoWriter::videoFrames() const { return impl_->frames; }

int64_t VideoWriter::videoPackets() const { return impl_->packets; }

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
  checkEncodedPackets(s.frames, s.packets);
  checkAv(av_write_trailer(s.output.get()), "write MP4 trailer");
  checkAv(avio_closep(&s.output->pb), "close output");
  s.finished = true;
}

}  // namespace vc
