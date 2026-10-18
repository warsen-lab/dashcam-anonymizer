#include "test_media.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

extern "C" {
#include <libavutil/common.h>
#include <libavutil/display.h>
#include <libavutil/pixdesc.h>
}

namespace vc::test {
namespace fs = std::filesystem;

fs::path tmpDir() {
  const fs::path dir = VC_TEST_TMP_DIR;
  fs::create_directories(dir);
  return dir;
}

fs::path makeMedia(const std::string& name, const std::string& args) {
  const fs::path output = tmpDir() / name;
  if (fs::exists(output)) return output;
  // ctest may run test processes in parallel: write a private file, then rename.
  const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-" + name);
  const std::string command = std::string("\"") + VC_FFMPEG + "\" -v error -y " + args + " \"" +
                              partial.string() + "\"";
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("ffmpeg failed: " + command);
  }
  fs::rename(partial, output);
  return output;
}

fs::path basicVideo() {
  return makeMedia("basic.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=2 -c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path staticThenNoiseVideo() {
  return makeMedia(
      "static_then_noise.mp4",
      "-f lavfi -i color=c=gray:s=640x360:r=30:d=1,format=yuv420p "
      "-f lavfi -i \"nullsrc=s=640x360:r=30:d=1,geq=lum='random(1)*255':cb=128:cr=128,"
      "format=yuv420p\" "
      "-filter_complex \"[0:v][1:v]concat=n=2:v=1:a=0[v]\" -map \"[v]\" "
      "-c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path tenBitVideo() {
  return makeMedia("ten_bit.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=1 -c:v libx265 -pix_fmt yuv420p10le "
                   "-x265-params log-level=error");
}

fs::path audioOnlyFile() {
  return makeMedia("audio_only.m4a", "-f lavfi -i sine=f=440:d=1 -c:a aac");
}

fs::path videoWithAudio() {
  return makeMedia("video_audio.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=2 -f lavfi -i sine=f=440:d=2 "
                   "-c:v libx264 -pix_fmt yuv420p -g 30 -c:a aac -shortest");
}

fs::path noisyVideoWithAudio() {
  return makeMedia("noisy_audio.mp4",
                   "-f lavfi -i \"testsrc2=s=640x360:r=30:d=2,noise=alls=40:allf=t\" "
                   "-f lavfi -i sine=f=440:d=2 -c:v libx264 -pix_fmt yuv420p -g 30 -b:v 4M "
                   "-c:a aac -shortest");
}

fs::path checkerboardVideo() {
  return makeMedia("checkerboard.mp4",
                   "-f lavfi -i \"nullsrc=s=640x360:r=30:d=2,geq=lum='255*mod(floor(X/2)+floor(Y/2),2)':cb=128:cr=128,"
                   "format=yuv420p\" -c:v libx264 -qp 0 -pix_fmt yuv420p -g 30");
}

fs::path copyCutVideo() {
  return makeMedia("cut_copy.mp4", "-ss 0.7 -i \"" + videoWithAudio().string() + "\" -c copy");
}

fs::path mpegTsVideo() {
  return makeMedia("video_audio.ts", "-i \"" + videoWithAudio().string() + "\" -c copy");
}

fs::path anamorphicVideo() {
  return makeMedia("anamorphic.mp4",
                   "-f lavfi -i testsrc2=s=320x240:r=30:d=1 -vf setsar=4/3 -c:v libx264 -pix_fmt yuv420p");
}

fs::path rotatedVideo() {
  return makeMedia("rotated.mp4", "-display_rotation 90 -i \"" + basicVideo().string() + "\" -c copy");
}

fs::path bt709Video() {
  return makeMedia("bt709.mp4",
                   "-f lavfi -i testsrc2=s=320x240:r=30:d=1 "
                   "-vf setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv "
                   "-c:v libx264 -pix_fmt yuv420p");
}

fs::path bt601FullRangeVideo() {
  return makeMedia("bt601_full.mp4",
                   "-f lavfi -i testsrc2=s=320x240:r=30:d=1 "
                   "-vf setparams=colorspace=bt470bg:color_primaries=smpte170m:color_trc=smpte170m:range=pc "
                   "-c:v libx264 -pix_fmt yuv420p");
}

fs::path videoWithMulawAudio() {
  return makeMedia("video_mulaw.mov",
                   "-f lavfi -i testsrc2=s=320x240:r=30:d=1 -f lavfi -i sine=f=440:d=1 "
                   "-c:v libx264 -pix_fmt yuv420p -c:a pcm_mulaw -ar 8000 -shortest");
}

fs::path rawH264Stream() {
  return makeMedia("raw.h264",
                   "-f lavfi -i testsrc2=s=320x240:r=30:d=1 -c:v libx264 -pix_fmt yuv420p -f h264");
}

fs::path corruptFirstNalVideo() {
  const fs::path output = tmpDir() / "basic_corrupt_first_nal.mp4";
  if (fs::exists(output)) return output;
  std::vector<char> bytes;
  {
    std::ifstream in(basicVideo(), std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  // The first NAL unit's 4-byte length prefix follows the "mdat" fourcc.
  const std::string fourcc = "mdat";
  const auto mdat = std::search(bytes.begin(), bytes.end(), fourcc.begin(), fourcc.end());
  if (mdat == bytes.end()) throw std::runtime_error("no mdat box in basic.mp4");
  const size_t prefix = static_cast<size_t>(mdat - bytes.begin()) + fourcc.size();
  if (prefix + 4 >= bytes.size()) throw std::runtime_error("basic.mp4 is truncated");
  std::fill_n(bytes.begin() + static_cast<std::ptrdiff_t>(prefix), 4, static_cast<char>(0xFF));
  const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-basic_corrupt_first_nal.mp4");
  std::ofstream(partial, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  fs::rename(partial, output);
  return output;
}

fs::path videoWithSubtitles() {
  const fs::path srt = tmpDir() / "subs.srt";
  if (!fs::exists(srt)) {
    const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-subs.srt");
    std::ofstream(partial) << "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n";
    fs::rename(partial, srt);
  }
  return makeMedia("video_subs.mp4", "-f lavfi -i testsrc2=s=320x240:r=30:d=1 -i \"" + srt.string() +
                                         "\" -map 0:v -map 1:s -c:v libx264 -pix_fmt yuv420p -c:s mov_text");
}

namespace {

av::InputContextPtr openProbed(const fs::path& path) {
  AVFormatContext* raw = nullptr;
  if (avformat_open_input(&raw, path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path.string());
  av::InputContextPtr input(raw);
  if (avformat_find_stream_info(input.get(), nullptr) < 0) throw std::runtime_error("probe " + path.string());
  return input;
}

int firstStreamOf(const AVFormatContext& input, AVMediaType type, const fs::path& path) {
  for (unsigned i = 0; i < input.nb_streams; ++i) {
    if (input.streams[i]->codecpar->codec_type == type) return static_cast<int>(i);
  }
  throw std::runtime_error("no " + std::string(av_get_media_type_string(type)) + " stream in " + path.string());
}

}  // namespace

StreamTiming streamTiming(const fs::path& path, AVMediaType type) {
  const av::InputContextPtr input = openProbed(path);
  const int index = firstStreamOf(*input, type, path);
  const AVStream* stream = input->streams[index];
  StreamTiming t;
  t.startSec = stream->start_time == AV_NOPTS_VALUE ? 0.0 : static_cast<double>(stream->start_time) * av_q2d(stream->time_base);
  int64_t first = std::numeric_limits<int64_t>::max();
  int64_t last = std::numeric_limits<int64_t>::min();
  const av::PacketPtr packet = av::allocPacket();
  while (av_read_frame(input.get(), packet.get()) >= 0) {
    if (packet->stream_index == index) {
      ++t.packets;
      if ((packet->flags & AV_PKT_FLAG_DISCARD) == 0 && packet->pts != AV_NOPTS_VALUE) {
        first = std::min(first, packet->pts);
        last = std::max(last, packet->pts);
      }
    }
    av_packet_unref(packet.get());
  }
  if (first > last) throw std::runtime_error("no packets with timestamps in " + path.string());
  t.firstPtsSec = static_cast<double>(first) * av_q2d(stream->time_base);
  t.lastPtsSec = static_cast<double>(last) * av_q2d(stream->time_base);
  return t;
}

AVCodecID streamCodec(const fs::path& path, AVMediaType type) {
  const av::InputContextPtr input = openProbed(path);
  return input->streams[firstStreamOf(*input, type, path)]->codecpar->codec_id;
}

VideoPresentation videoPresentation(const fs::path& path) {
  const av::InputContextPtr input = openProbed(path);
  const AVStream* stream = input->streams[firstStreamOf(*input, AVMEDIA_TYPE_VIDEO, path)];
  const AVCodecParameters* par = stream->codecpar;
  VideoPresentation p;
  p.sampleAspectRatio = stream->sample_aspect_ratio.num > 0 ? stream->sample_aspect_ratio : par->sample_aspect_ratio;
  p.colorRange = par->color_range;
  p.colorSpace = par->color_space;
  p.colorPrimaries = par->color_primaries;
  p.colorTrc = par->color_trc;
  p.codecTag = par->codec_tag;
  const AVPacketSideData* sd = av_packet_side_data_get(par->coded_side_data, par->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
  if (sd != nullptr && sd->size >= sizeof(int32_t) * 9) {
    p.hasDisplayMatrix = true;
    std::memcpy(p.displayMatrix.data(), sd->data, sizeof(int32_t) * 9);
    p.rotationDegrees = av_display_rotation_get(p.displayMatrix.data());
  }
  return p;
}

int streamCount(const fs::path& path, AVMediaType type) {
  AVFormatContext* raw = nullptr;
  if (avformat_open_input(&raw, path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path.string());
  av::InputContextPtr input(raw);
  if (avformat_find_stream_info(input.get(), nullptr) < 0) throw std::runtime_error("probe " + path.string());
  int count = 0;
  for (unsigned i = 0; i < input->nb_streams; ++i) count += input->streams[i]->codecpar->codec_type == type ? 1 : 0;
  return count;
}

double streamDurationSec(const fs::path& path, AVMediaType type) {
  AVFormatContext* raw = nullptr;
  if (avformat_open_input(&raw, path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path.string());
  av::InputContextPtr input(raw);
  if (avformat_find_stream_info(input.get(), nullptr) < 0) throw std::runtime_error("probe " + path.string());
  for (unsigned i = 0; i < input->nb_streams; ++i) {
    const AVStream* s = input->streams[i];
    if (s->codecpar->codec_type == type && s->duration != AV_NOPTS_VALUE) return s->duration * av_q2d(s->time_base);
  }
  return 0.0;
}

namespace {

av::FramePtr allocated(int width, int height, AVPixelFormat format) {
  av::FramePtr frame = av::allocFrame();
  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->color_range = AVCOL_RANGE_MPEG;
  if (av_frame_get_buffer(frame.get(), 0) < 0) throw std::runtime_error("av_frame_get_buffer failed");
  return frame;
}

// fill(plane, x, y) gives the byte for luma (plane 0) or chroma U (1) / V (2) sample (x, y).
template <typename Fill>
void fillPlanes(AVFrame& frame, Fill fill) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
  const int cw = AV_CEIL_RSHIFT(frame.width, desc->log2_chroma_w);
  const int ch = AV_CEIL_RSHIFT(frame.height, desc->log2_chroma_h);
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) frame.data[0][y * frame.linesize[0] + x] = fill(0, x, y);
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      if (frame.format == AV_PIX_FMT_NV12) {
        frame.data[1][y * frame.linesize[1] + 2 * x] = fill(1, x, y);
        frame.data[1][y * frame.linesize[1] + 2 * x + 1] = fill(2, x, y);
      } else {
        frame.data[1][y * frame.linesize[1] + x] = fill(1, x, y);
        frame.data[2][y * frame.linesize[2] + x] = fill(2, x, y);
      }
    }
  }
}

}  // namespace

av::FramePtr makeFrame(int width, int height, AVPixelFormat format, uint8_t y, uint8_t u, uint8_t v) {
  av::FramePtr frame = allocated(width, height, format);
  const uint8_t values[3] = {y, u, v};
  fillPlanes(*frame, [&](int plane, int, int) { return values[plane]; });
  return frame;
}

av::FramePtr makeNoiseFrame(int width, int height, AVPixelFormat format, unsigned seed) {
  av::FramePtr frame = allocated(width, height, format);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  fillPlanes(*frame, [&](int, int, int) { return static_cast<uint8_t>(byte(rng)); });
  return frame;
}

av::FramePtr deepCopy(const AVFrame& source) {
  av::FramePtr frame =
      allocated(source.width, source.height, static_cast<AVPixelFormat>(source.format));
  if (av_frame_copy(frame.get(), &source) < 0 || av_frame_copy_props(frame.get(), &source) < 0) {
    throw std::runtime_error("av_frame_copy failed");
  }
  return frame;
}

}  // namespace vc::test
