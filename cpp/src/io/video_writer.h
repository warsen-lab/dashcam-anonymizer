#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/platform.h"

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

struct AVCodecParameters;
struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vc {

// The output is always H.264 (spec 2026-10-20 §3), whatever the source codec.
struct WriterConfig {
  std::filesystem::path path;  // created as MP4 whatever the extension (so ".partial" works)
  int width = 0;
  int height = 0;
  AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;  // of the frames passed to writeVideo(); anything but 8-bit
                                                   // 4:2:0 is converted to it (H.264 encoders all take that)
  AVRational timeBase{1, 1000};                    // of the frames' pts
  AVRational frameRate{0, 1};                      // 0/1 when unknown (the encoder assumes 30 fps)
  int64_t bitRate = 0;                             // <= 0: 8 Mb/s (h264OutputBitRate gives vc_redact's)
  bool preferHardware = true;                      // h264EncoderCandidates
  std::string nvencPreset = "p4";                  // h264_nvenc's preset (configs/redact.json nvenc_preset)

  // Presentation metadata of the source video, written to the encoder and the MP4 so that the
  // output is shown like the input (aspect ratio, colours, rotation). The defaults mean "not
  // specified" and write nothing extra; copyPresentation() fills them from an input stream.
  AVRational sampleAspectRatio{0, 1};  // 0/1 when unknown
  AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
  AVColorSpace colorSpace = AVCOL_SPC_UNSPECIFIED;
  AVColorPrimaries colorPrimaries = AVCOL_PRI_UNSPECIFIED;
  AVColorTransferCharacteristic colorTrc = AVCOL_TRC_UNSPECIFIED;
  std::optional<std::array<int32_t, 9>> displayMatrix;  // AV_PKT_DATA_DISPLAYMATRIX (rotation / flip)
};

// Copies the presentation metadata of an input video stream into `config`: the sample aspect ratio
// (the stream's, else the codec parameters'), the colour range / space / primaries / transfer from
// the codec parameters and the display matrix from the stream's coded side data. Reads the stream,
// so call it where the demuxer is not in use on another thread.
void copyPresentation(const AVStream& videoStream, WriterConfig& config);

// Codec parameters and time base of an input audio stream, copied so that the writer never reads
// the demuxer's own streams (the demuxer may be running on another thread). The copy stays valid
// after the reader that produced it is gone.
struct AudioStreamInfo {
  std::shared_ptr<const AVCodecParameters> parameters;
  AVRational timeBase{0, 1};
};
// Throws VcError when the copy fails.
AudioStreamInfo snapshotAudioStream(const AVStream& stream);
std::vector<AudioStreamInfo> snapshotAudioStreams(const std::vector<const AVStream*>& streams);

// Throws VcError naming the first audio codec that an MP4 file cannot store (VideoWriter would
// refuse it); call it before creating anything for a run.
void requireMp4Audio(const std::vector<AudioStreamInfo>& audio);

// Throws VcError unless the encoder returned exactly one packet per frame it was sent: a dropped or
// extra frame would put the output out of step with the source and masks.jsonl.
void checkEncodedPackets(int64_t frames, int64_t packets);

// The H.264 encoders to try, in order (spec 2026-10-20 §3): with preferHardware h264_nvenc (not on macOS), then
// h264_videotoolbox (macOS only), then libx264; without it libx264 alone.
std::vector<std::string> h264EncoderCandidates(bool preferHardware, Platform platform);

// The output bitrate: the source's (8 Mb/s when it is unknown, i.e. <= 0), times nonH264Factor unless the source is
// H.264 (configs/redact.json output_bitrate_factor_hevc), rounded to whole bits per second.
int64_t h264OutputBitRate(bool sourceIsH264, int64_t sourceBitRate, double nonH264Factor);

// level_idc (52 for level 5.2) of the lowest H.264 level whose frame size and macroblock rate (ITU-T H.264 Table
// A-1) hold width x height at fps: 3840x2160 at 60 fps needs 52, 1920x1080 at 59.94 fps 42. 0 beyond level 6.2.
// The same table as vc_eval's required_h264_level (py/vc_eval/integrity.py).
int h264LevelIdc(int width, int height, double fps);

// What vc_redact notes on stderr when the source is not 8-bit 4:2:0 (spec 2026-10-20 §3: it is encoded as 8-bit
// 4:2:0), e.g. "the source video is yuv420p10le (10-bit 4:2:0); the H.264 output is 8-bit 4:2:0"; nullopt for an
// 8-bit 4:2:0 or unknown format.
std::optional<std::string> h264FormatNote(AVPixelFormat source);

// Encodes frames and stream-copies audio into an MP4 file.
//
// Not thread-safe, and not tied to one thread by construction either: create it, call every method
// (writeVideo / writeAudio / finish / the accessors) and destroy it from ONE thread. vc_redact
// builds it on its encode thread and uses it only there.
class VideoWriter {
 public:
  // Opens the first encoder that works (one line on stderr naming each encoder that could not be
  // opened and why, when a later one is used instead) and writes the MP4 header; `audio` streams are
  // copied with their codec parameters and time base. Throws VcError, also when an audio codec cannot
  // be stored in MP4 (checked before the file is created).
  VideoWriter(const WriterConfig& config, const std::vector<AudioStreamInfo>& audio);
  ~VideoWriter();  // closes the file; without finish() it is incomplete
  VideoWriter(const VideoWriter&) = delete;
  VideoWriter& operator=(const VideoWriter&) = delete;

  // e.g. "h264_nvenc", "h264_videotoolbox" or "libx264".
  const std::string& encoderName() const;
  // Encodes one frame whose pts is in config.timeBase. Clears frame.pict_type so the encoder picks
  // frame types itself. Throws VcError, also when the frame is not config.width x config.height
  // (the encoder would read past the end of a smaller frame) or after finish().
  void writeVideo(AVFrame& frame);
  // Copies one packet of the audio stream at position `ordinal` of the constructor's `audio`; its
  // timestamps are in that stream's time base. Throws VcError, also after finish().
  void writeAudio(const AVPacket& packet, size_t ordinal);
  // Flushes the encoder and writes the trailer. Throws VcError, also when the encoder returned more
  // or fewer packets than frames it was sent (checkEncodedPackets).
  void finish();
  int64_t videoFrames() const;   // frames sent to the encoder
  int64_t videoPackets() const;  // encoded video packets written

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
