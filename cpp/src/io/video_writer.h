#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

struct AVCodecParameters;
struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vc {

struct WriterConfig {
  std::filesystem::path path;  // created as MP4 whatever the extension (so ".partial" works)
  int width = 0;
  int height = 0;
  AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;  // of the frames passed to writeVideo(); converted when
                                                   // the encoder does not take it
  AVRational timeBase{1, 1000};                    // of the frames' pts
  AVRational frameRate{0, 1};                      // 0/1 when unknown (the encoder assumes 30 fps)
  std::string sourceCodec = "h264";                // "hevc" stays HEVC, anything else becomes H.264
  int64_t bitRate = 0;                             // <= 0: 8 Mb/s
  bool preferHardware = true;                      // VideoToolbox first, then libx264 / libx265

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

  // e.g. "h264_videotoolbox" or "libx264".
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
