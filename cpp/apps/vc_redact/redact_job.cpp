#include "vc_redact/redact_job.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include "common/bounded_queue.h"
#include "detect/fake_detector.h"
#include "detect/merged_detector.h"
#include "detect/yunet_detector.h"
#include "io/av_ptr.h"
#include "io/video_writer.h"
#include "mask/mosaic.h"
#include "track/box_timeline.h"
#include "vc_redact/detect_rate.h"
#include "vc_redact/records.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMaxQueueCapacity = 64;
constexpr size_t kMinQueueCapacity = 4;
constexpr uint64_t kQueueBudgetBytes = 256ull * 1024 * 1024;
constexpr uint64_t kBytesPerPixel = 3;  // the largest layout the reader emits: 8-bit 4:4:4
// A keyframe is forced after this many buffered frames, whatever the stride (for very high frame rates).
constexpr size_t kMaxBufferedFrames = 120;
constexpr const char* kMasksFile = "masks.jsonl";
constexpr const char* kDetectionsFile = "detections.jsonl";
constexpr const char* kStatsFile = "stats.json";

double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// printf-style formatting into a string of any length.
template <typename... Args>
std::string formatted(const char* format, Args... args) {
  const int size = std::snprintf(nullptr, 0, format, args...);
  if (size <= 0) return {};
  std::string out(static_cast<size_t>(size), '\0');
  std::snprintf(out.data(), out.size() + 1, format, args...);
  return out;
}

std::string joined(const std::vector<std::string>& items) {
  if (items.empty()) return "none";
  std::string out;
  for (const std::string& item : items) out += (out.empty() ? "" : ", ") + item;
  return out;
}

struct VideoItem {
  int64_t index = 0;
  double ptsMs = 0;
  av::FramePtr frame;
};
struct AudioItem {
  av::PacketPtr packet;
  size_t ordinal = 0;
};
struct EndItem {};
using Item = std::variant<VideoItem, AudioItem, EndItem>;

// Thrown inside a stage when another stage already failed and closed the queues.
struct PipelineAborted {};

class FirstError {
 public:
  void set(std::exception_ptr error) {
    std::lock_guard lock(mutex_);
    if (!error_) error_ = std::move(error);
  }
  // Call after every thread has been joined.
  void rethrowIfSet() {
    if (error_) std::rethrow_exception(error_);
  }

 private:
  std::mutex mutex_;
  std::exception_ptr error_;
};

// Stage threads that are always joined, also when starting a later one throws: the queues are
// closed first so that a running stage cannot block forever (a joinable std::thread that is
// destroyed would terminate the process).
class StageThreads {
 public:
  explicit StageThreads(std::function<void()> closeQueues) : closeQueues_(std::move(closeQueues)) {}
  ~StageThreads() {
    closeQueues_();
    join();
  }
  StageThreads(const StageThreads&) = delete;
  StageThreads& operator=(const StageThreads&) = delete;

  void start(std::function<void()> body) { threads_.emplace_back(std::move(body)); }
  void join() {
    for (std::thread& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

 private:
  std::function<void()> closeQueues_;
  std::vector<std::thread> threads_;
};

std::string stemOf(const fs::path& input) {
  const std::string stem = input.stem().string();
  if (stem.empty() || stem == "." || stem == "..") {
    throw VcError("cannot derive an output directory name from " + input.string());
  }
  return stem;
}

bool isSameOrInside(const fs::path& path, const fs::path& dir) {
  const fs::path relative = path.lexically_relative(dir);
  return !relative.empty() && *relative.begin() != "..";
}

void checkOutputDir(const fs::path& input, const fs::path& dir, bool overwrite) {
  if (fs::exists(dir)) {
    if (!overwrite) {
      throw OutputExistsError("output directory exists: " + dir.string() + " (pass --overwrite to replace it)");
    }
    const bool replaceable = fs::is_directory(dir) && (fs::is_empty(dir) || fs::is_regular_file(dir / kMasksFile));
    if (!replaceable) {
      throw OutputExistsError("refusing to overwrite " + dir.string() + ": not a vc_redact output directory");
    }
  }
  if (isSameOrInside(fs::weakly_canonical(fs::absolute(input)), fs::weakly_canonical(fs::absolute(dir)))) {
    throw VcError("output directory " + dir.string() + " would contain the input video; choose another --out");
  }
}

void checkOptions(const RedactOptions& o) {
  if (o.modelDir.has_value() == o.fakeDetections.has_value()) {
    throw std::invalid_argument("exactly one of a model directory or fake detections is required");
  }
  validateRedactConfig(o.config, "redact options");
}

// The keyframe schedule for the input's video stream (spec 2026-10-19 §2). streamFrameRate throws a VcError that
// names the video; the std::invalid_argument of detectSchedule / everyFrameSchedule (a header frame rate outside
// (0, 1e6] fps) would reach the user without the file or the rate, so it is rethrown as a VcError that names both.
DetectSchedule keyframeSchedule(const RedactOptions& o, const AVStream& stream) {
  const std::string video = o.input.string();
  const double videoFps = streamFrameRate(stream.avg_frame_rate, stream.r_frame_rate, video);
  try {
    return o.detectEveryFrame ? everyFrameSchedule(videoFps) : detectSchedule(videoFps, o.config.detectFps);
  } catch (const std::invalid_argument& e) {
    throw VcError("cannot space the keyframes of " + video + ": " + e.what() +
                  formatted(" (its video stream says %g fps)", videoFps));
  }
}

std::unique_ptr<Detector> makeDetector(const RedactOptions& o, int width, int height) {
  std::unique_ptr<Detector> detector;
  if (o.fakeDetections) {
    detector = std::make_unique<FakeDetector>(*o.fakeDetections);
  } else {
    detector = std::make_unique<OnnxYoloDetector>(*o.modelDir, o.provider, o.config.decode, width, height);
  }
  if (!o.faceModelDir) return detector;
  auto face = std::make_unique<YuNetDetector>(*o.faceModelDir, o.provider, o.config.face, width, height);
  return std::make_unique<MergedDetector>(std::move(detector), std::move(face));
}

// The index of `name` in `classes`, -1 when it is not there.
int classIndex(const std::vector<std::string>& classes, const std::string& name) {
  const auto it = std::find(classes.begin(), classes.end(), name);
  return it == classes.end() ? -1 : static_cast<int>(it - classes.begin());
}

// A keyframe's detections after the large-face rule, and how many it dropped.
struct LargeFaceFilter {
  std::vector<SourcedBox> kept;
  int64_t dropped = 0;
};

// Drops every face box whose own area exceeds config.largeFaceAreaFraction of the width x height frame and whose
// score is below config.largeFaceMinScore (spec 2026-10-19 §3.6: giant false positives on road and buildings). Faces
// are the boxes whose class name in `classes` is "face", whichever detector found them; other classes, smaller or
// higher-scoring faces are all kept, in order. "Exceeds" and "below" are strict.
LargeFaceFilter dropLargeLowScoreFaces(std::vector<SourcedBox> found, const std::vector<std::string>& classes,
                                       const RedactConfig& config, int width, int height) {
  const auto faceClass = std::find(classes.begin(), classes.end(), "face");
  if (faceClass == classes.end()) return {std::move(found), 0};
  const auto face = static_cast<int>(faceClass - classes.begin());
  const double limit = config.largeFaceAreaFraction * static_cast<double>(width) * static_cast<double>(height);
  const auto isDropped = [&](const SourcedBox& b) {
    const double w = std::max(0.0, static_cast<double>(b.box.x2) - static_cast<double>(b.box.x1));
    const double h = std::max(0.0, static_cast<double>(b.box.y2) - static_cast<double>(b.box.y1));
    return b.box.cls == face && w * h > limit && b.box.score < config.largeFaceMinScore;
  };
  LargeFaceFilter out;
  out.kept.reserve(found.size());
  for (SourcedBox& b : found) {
    if (isDropped(b)) {
      ++out.dropped;
    } else {
      out.kept.push_back(std::move(b));
    }
  }
  return out;
}

// Keyframe choice, detection, BoxTimeline and mosaic: everything between decode and encode.
class Processor {
 public:
  Processor(const RedactOptions& options, Detector& detector, const VideoInfo& info, const DetectSchedule& schedule,
            std::string video, std::ostream& detections, std::ostream& masks, BoundedQueue<Item>& out,
            RedactStats& stats)
      : options_(options),
        detector_(detector),
        schedule_(schedule),
        video_(std::move(video)),
        detections_(detections),
        masks_(masks),
        out_(out),
        stats_(stats),
        width_(info.width),
        height_(info.height),
        timeline_(timelineFor(options.config, detector.classes()), info.width, info.height),
        blocks_(options.config.mosaic),
        plateClass_(classIndex(detector.classes(), "plate")) {}

  void onVideo(VideoItem item) {
    const Clock::time_point started = Clock::now();
    // A keyframe every schedule_.stride frames by frame index (spec 2026-10-19 §2), the first frame always
    // (BoxTimeline starts on a keyframe), and one whenever kMaxBufferedFrames frames are waiting.
    const bool key = options_.detectEveryFrame || !sawKeyframe_ || onSchedule(schedule_, item.index) ||
                     buffer_.size() >= kMaxBufferedFrames;
    stats_.track.busySec += secondsSince(started);
    if (!key) {
      buffer_.push_back(std::move(item));
      return;
    }
    sawKeyframe_ = true;

    const VideoFrame view = viewOf(item);
    const Clock::time_point detectStarted = Clock::now();
    std::vector<SourcedBox> detected = detector_.detectSourced(view);
    stats_.detect.busySec += secondsSince(detectStarted);
    ++stats_.detect.items;
    ++stats_.keyframes;
    // Giant low-score faces never reach the timeline, detections.jsonl or the masks (spec 2026-10-19 §3.6).
    const LargeFaceFilter filtered =
        dropLargeLowScoreFaces(std::move(detected), detector_.classes(), options_.config, width_, height_);
    const std::vector<SourcedBox>& found = filtered.kept;
    stats_.detections += static_cast<int64_t>(found.size());
    stats_.largeFacesDropped += filtered.dropped;
    std::vector<Box> boxes;
    boxes.reserve(found.size());
    for (const SourcedBox& b : found) boxes.push_back(b.box);

    const Clock::time_point trackStarted = Clock::now();
    const TimelineOutput step = timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    // The previous keyframe's boxes are written now that the timeline has said which of them it masked.
    writeDetections(step.confirmed);
    pending_ = KeyframeDetections{item.index, item.ptsMs, found};
    emit(step.masks);
    buffer_.push_back(std::move(item));
  }

  void onEnd() {
    const Clock::time_point started = Clock::now();
    const TimelineOutput step = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    writeDetections(step.confirmed);
    emit(step.masks);
    if (!out_.push(Item{EndItem{}})) throw PipelineAborted{};
  }

 private:
  // A keyframe's detections, written once the next keyframe (or the end) says which were masked.
  struct KeyframeDetections {
    int64_t frame = 0;
    double ptsMs = 0;
    std::vector<SourcedBox> boxes{};
  };

  // Writes the pending keyframe's detections.jsonl lines with their confirmation flags.
  void writeDetections(const std::vector<bool>& confirmed) {
    if (!pending_) return;
    if (confirmed.size() != pending_->boxes.size()) {
      throw VcError("internal error: confirmation flags do not match the keyframe's detections");
    }
    for (size_t i = 0; i < confirmed.size(); ++i) {
      const SourcedBox& b = pending_->boxes[i];
      detections_ << detectionRecord(video_, pending_->frame, pending_->ptsMs, b.box, detector_.classes(), b.src,
                                     static_cast<bool>(confirmed[i]));
    }
    pending_.reset();
  }

  static VideoFrame viewOf(const VideoItem& item) {
    VideoFrame v;
    v.index = item.index;
    v.ptsMs = item.ptsMs;
    v.av = item.frame.get();
    return v;
  }

  std::vector<FrameTime> times() const {
    std::vector<FrameTime> t;
    t.reserve(buffer_.size());
    for (const VideoItem& f : buffer_) t.push_back(FrameTime{f.index, f.ptsMs});
    return t;
  }

  // Writes masks.jsonl, pixelates and sends every buffered frame downstream, in order. A mask's block comes
  // from its track (TrackBlocks), so a target keeps its mosaic from frame to frame (spec 2026-10-19 §4); a
  // plate's mosaic stays inside its mask instead of snapping out to the grid.
  void emit(const std::vector<std::vector<MaskBox>>& masks) {
    if (masks.size() != buffer_.size()) throw VcError("internal error: masks do not match the buffered frames");
    for (size_t k = 0; k < buffer_.size(); ++k) {
      VideoItem& frame = buffer_[k];
      const Clock::time_point started = Clock::now();
      std::vector<MaskBox> boxes = masks[k];
      std::vector<MosaicRect> rects;
      rects.reserve(boxes.size());
      for (MaskBox& m : boxes) {
        const PixelRect rect = coveringRect(m.x1, m.y1, m.x2, m.y2, frame.frame->width, frame.frame->height);
        m.block = blocks_.blockFor(m.track, rect);
        m.snap = m.cls != plateClass_;
        rects.push_back(MosaicRect{rect, m.block, m.snap});
      }
      blocks_.endFrame();
      if (!boxes.empty()) {
        checkAv(av_frame_make_writable(frame.frame.get()), "make frame writable");
        applyMosaic(*frame.frame, rects);
        ++stats_.maskedFrames;
        stats_.maskBoxes += static_cast<int64_t>(boxes.size());
      }
      stats_.mosaic.busySec += secondsSince(started);
      ++stats_.mosaic.items;
      masks_ << maskRecord(video_, frame.index, boxes, detector_.classes());
      if (!out_.push(Item{std::move(frame)})) throw PipelineAborted{};
    }
    buffer_.clear();
  }

  const RedactOptions& options_;
  Detector& detector_;
  DetectSchedule schedule_;
  std::string video_;
  std::ostream& detections_;
  std::ostream& masks_;
  BoundedQueue<Item>& out_;
  RedactStats& stats_;
  int width_;
  int height_;
  BoxTimeline timeline_;
  TrackBlocks blocks_;
  int plateClass_;  // -1 when the detector has no plate class
  bool sawKeyframe_ = false;
  std::vector<VideoItem> buffer_;
  std::optional<KeyframeDetections> pending_;
};

void decodeStage(VideoReader& reader, BoundedQueue<Item>& out, StageStat& decode) {
  std::unordered_map<int, size_t> ordinals;
  const std::vector<const AVStream*> audio = reader.audioStreams();
  for (size_t i = 0; i < audio.size(); ++i) ordinals[audio[i]->index] = i;
  // The reader hands audio packets over from inside next(). Pushing them there would count the
  // time spent waiting for room in the queue as decoding time, so they are only collected there
  // and pushed once next() has returned, in the order they were read: before the video frame that
  // call returned. (One next() call reads packets only until it has a frame, so this stays small.)
  std::vector<AudioItem> delivered;
  reader.setAudioSink([&](const AVPacket& packet) {
    av::PacketPtr copy = av::allocPacket();
    checkAv(av_packet_ref(copy.get(), &packet), "reference audio packet");
    delivered.push_back(AudioItem{std::move(copy), ordinals.at(packet.stream_index)});
  });
  struct ResetSink {
    VideoReader& reader;
    ~ResetSink() { reader.setAudioSink(nullptr); }
  } resetSink{reader};
  while (true) {
    const Clock::time_point started = Clock::now();
    const VideoFrame* frame = reader.next();
    decode.busySec += secondsSince(started);
    for (AudioItem& packet : delivered) {
      if (!out.push(Item{std::move(packet)})) throw PipelineAborted{};
    }
    delivered.clear();
    if (frame == nullptr) break;
    av::FramePtr clone(av_frame_clone(frame->av));
    if (!clone) throw VcError("av_frame_clone failed");
    if (clone->best_effort_timestamp != AV_NOPTS_VALUE) clone->pts = clone->best_effort_timestamp;
    if (clone->pts == AV_NOPTS_VALUE) throw VcError("frame " + std::to_string(frame->index) + " has no timestamp");
    ++decode.items;
    if (!out.push(Item{VideoItem{frame->index, frame->ptsMs, std::move(clone)}})) throw PipelineAborted{};
  }
  if (!out.push(Item{EndItem{}})) throw PipelineAborted{};
}

struct EncodeResult {
  StageStat encode;
  std::string encoder;
  int64_t audioPackets = 0;
};

// The writer is created on the first video frame, when its pixel format is known; audio that
// arrives earlier is held until then.
void encodeStage(BoundedQueue<Item>& in, WriterConfig config, const std::vector<AudioStreamInfo>& audio,
                 EncodeResult& result) {
  std::unique_ptr<VideoWriter> writer;
  std::vector<AudioItem> early;
  while (std::optional<Item> item = in.pop()) {
    const Clock::time_point started = Clock::now();
    if (std::holds_alternative<EndItem>(*item)) {
      if (!writer) throw VcError("no video frames to encode");
      writer->finish();
      result.encode.busySec += secondsSince(started);
      return;
    }
    if (auto* a = std::get_if<AudioItem>(&*item)) {
      if (writer) {
        writer->writeAudio(*a->packet, a->ordinal);
        ++result.audioPackets;
      } else {
        early.push_back(std::move(*a));
      }
    } else {
      VideoItem& video = std::get<VideoItem>(*item);
      if (!writer) {
        config.pixelFormat = static_cast<AVPixelFormat>(video.frame->format);
        writer = std::make_unique<VideoWriter>(config, audio);
        result.encoder = writer->encoderName();
        for (AudioItem& e : early) {
          writer->writeAudio(*e.packet, e.ordinal);
          ++result.audioPackets;
        }
        early.clear();
      }
      writer->writeVideo(*video.frame);
      ++result.encode.items;
    }
    result.encode.busySec += secondsSince(started);
  }
}

// Everything the encode thread needs to know about the input video, copied by value here on the
// main thread, so that it never reads the demuxer's streams (the decode thread is using them).
WriterConfig writerConfig(const VideoReader& reader, const fs::path& path, bool preferHardware) {
  const AVStream* stream = reader.videoStream();
  WriterConfig c;
  c.path = path;
  c.width = reader.info().width;
  c.height = reader.info().height;
  c.timeBase = stream->time_base;
  c.frameRate = stream->avg_frame_rate;
  c.sourceCodec = reader.info().codecName;
  c.bitRate = stream->codecpar->bit_rate;
  c.constantQuality = true;  // at the source's bit rate a re-encode comes out visibly soft
  c.preferHardware = preferHardware;
  copyPresentation(*stream, c);  // aspect ratio, colour tags and rotation stay as in the input
  return c;
}

// `audio`: the reader's audio streams, snapshotted on the main thread. `started`: the start of the
// run, for stats.setupSec.
void runPipeline(const RedactOptions& options, VideoReader& reader, Detector& detector,
                 const DetectSchedule& schedule, const std::vector<AudioStreamInfo>& audio, const fs::path& partial, Clock::time_point started,
                 RedactStats& stats) {
  const fs::path dir = partial.parent_path();
  std::ofstream detections(dir / kDetectionsFile);
  std::ofstream masks(dir / kMasksFile);
  if (!detections || !masks) throw VcError("cannot create output files in " + dir.string());

  const size_t capacity = pipelineQueueCapacity(reader.info().width, reader.info().height);
  BoundedQueue<Item> decoded(capacity);
  BoundedQueue<Item> processed(capacity);
  FirstError error;
  const auto closeQueues = [&] {
    decoded.close();
    processed.close();
  };
  const auto fail = [&](std::exception_ptr e) {
    error.set(std::move(e));
    closeQueues();
  };
  // Runs a stage body; a stage stopped by another stage's failure exits silently.
  const auto guarded = [&](const std::function<void()>& body) {
    try {
      body();
    } catch (const PipelineAborted&) {
    } catch (...) {
      fail(std::current_exception());
    }
  };

  // Everything that can fail before a thread exists is set up first, so the threads never see a
  // half-built pipeline; the reader's info and the video stream's metadata are copied (like the
  // audio streams' codec parameters and time bases) because the decode thread owns the reader.
  const VideoInfo info = reader.info();
  const WriterConfig config = writerConfig(reader, partial, options.preferHardwareEncoder);
  Processor processor(options, detector, info, schedule, options.input.filename().string(), detections, masks,
                      processed, stats);
  StageStat decodeStat;
  EncodeResult encodeResult;
  bool finished = false;

  {
    StageThreads threads(closeQueues);
    threads.start([&] { guarded([&] { decodeStage(reader, decoded, decodeStat); }); });
    threads.start([&] { guarded([&] { encodeStage(processed, config, audio, encodeResult); }); });
    guarded([&] {
      bool first = true;
      while (std::optional<Item> item = decoded.pop()) {
        if (auto* video = std::get_if<VideoItem>(&*item)) {
          if (first) stats.setupSec = secondsSince(started);
          first = false;
          processor.onVideo(std::move(*video));
        } else if (std::holds_alternative<AudioItem>(*item)) {
          if (!processed.push(std::move(*item))) throw PipelineAborted{};
        } else {
          processor.onEnd();
          finished = true;
          break;
        }
      }
    });
    threads.join();
  }
  error.rethrowIfSet();
  if (!finished) throw VcError("internal error: the pipeline stopped before the end of the video");

  stats.decode = decodeStat;
  stats.encode = encodeResult.encode;
  stats.encoder = encodeResult.encoder;
  stats.audioPackets = encodeResult.audioPackets;
  detections.close();
  masks.close();
  if (!detections || !masks) throw VcError("writing detections.jsonl / masks.jsonl failed");
  if (stats.encode.items != stats.decode.items) {
    throw VcError("encoded " + std::to_string(stats.encode.items) + " of " + std::to_string(stats.decode.items) +
                  " decoded frames");
  }
}

nlohmann::ordered_json stageJson(const StageStat& s) {
  nlohmann::ordered_json j;
  j["busy_s"] = s.busySec;
  j["items"] = s.items;
  j["fps"] = s.fps();
  return j;
}

void writeStats(const fs::path& path, const RedactStats& s, const RedactOptions& o) {
  nlohmann::ordered_json j;
  j["input"] = o.input.string();
  j["video"] = s.video.string();
  j["frames"] = s.frames;
  j["keyframes"] = s.keyframes;
  j["detections"] = s.detections;
  j["large_faces_dropped"] = s.largeFacesDropped;
  j["masked_frames"] = s.maskedFrames;
  j["mask_boxes"] = s.maskBoxes;
  j["decode_errors"] = s.decodeErrors;
  j["detector"] = s.detector;
  j["encoder"] = s.encoder;
  j["audio_streams"] = s.audioStreams;
  j["audio_packets"] = s.audioPackets;
  j["dropped_streams"] = s.droppedStreams;
  j["detect_every_frame"] = o.detectEveryFrame;
  j["video_fps"] = s.videoFps;
  j["keyframe_stride"] = s.keyframeStride;
  j["detect_fps_actual"] = s.detectFpsActual;
  j["config"] = redactConfigToJson(o.config);
  // decode, main and encode run in parallel; main is detect + track + mosaic, one after another.
  j["stages"]["decode"] = stageJson(s.decode);
  j["stages"]["main"] = stageJson(s.main);
  j["stages"]["encode"] = stageJson(s.encode);
  j["stages"]["detect"] = stageJson(s.detect);
  j["stages"]["detect_student"] = stageJson(s.detectStudent);
  j["stages"]["detect_face"] = stageJson(s.detectFace);
  j["stages"]["track"] = stageJson(s.track);
  j["stages"]["mosaic"] = stageJson(s.mosaic);
  j["setup_s"] = s.setupSec;
  j["elapsed_s"] = s.elapsedSec;
  std::ofstream out(path);
  out << j.dump(2) << '\n';
  out.close();
  if (!out) throw VcError("cannot write " + path.string());
}

}  // namespace

size_t pipelineQueueCapacity(int width, int height) {
  const uint64_t pixels = width > 0 && height > 0 ? static_cast<uint64_t>(width) * static_cast<uint64_t>(height) : 1;
  const uint64_t frames = kQueueBudgetBytes / (pixels * kBytesPerPixel);
  return static_cast<size_t>(std::clamp<uint64_t>(frames, kMinQueueCapacity, kMaxQueueCapacity));
}

std::vector<std::string> redactWarnings(const RedactStats& stats) {
  std::vector<std::string> warnings;
  if (stats.detectFpsAboveVideoRate) {
    warnings.push_back(formatted("note: detect_fps %g is above the video's %.2f fps, so every frame is a keyframe",
                                 stats.detectFpsRequested, stats.videoFps));
  }
  if (stats.decodeErrors > 0) {
    warnings.push_back("warning: the decoder rejected " + std::to_string(stats.decodeErrors) +
                       " corrupt packet(s) or frame(s) and skipped them, so the output lacks those frames "
                       "(decode_errors in stats.json)");
  }
  return warnings;
}

std::string redactSummary(const RedactStats& s) {
  const double fps = s.elapsedSec > 0 ? static_cast<double>(s.frames) / s.elapsedSec : 0;
  std::string out = "output:        " + s.video.string() + "\n";
  out += formatted(
      "frames:        %lld (keyframes %lld, detections %lld, large faces dropped %lld, "
      "masked %lld frames / %lld boxes)\n",
      static_cast<long long>(s.frames), static_cast<long long>(s.keyframes), static_cast<long long>(s.detections),
      static_cast<long long>(s.largeFacesDropped), static_cast<long long>(s.maskedFrames),
      static_cast<long long>(s.maskBoxes));
  out += formatted("detect rate:   %.2f /s (every %lld frames of a %.2f fps video)\n", s.detectFpsActual,
                   static_cast<long long>(s.keyframeStride), s.videoFps);
  out += "detector:      " + s.detector + "\n";
  out += "encoder:       " + s.encoder + "\n";
  out += formatted("audio:         %zu stream(s), %lld packets copied; dropped streams: ", s.audioStreams,
                   static_cast<long long>(s.audioPackets)) +
         joined(s.droppedStreams) + "\n";
  out += formatted("setup:         %.2f s (input, output checks, model load / CoreML compile, first frame)\n",
                   s.setupSec);
  out += formatted("stages (fps):  decode %.1f, main %.1f (detect %.1f, track %.1f, mosaic %.1f), encode %.1f\n",
                   s.decode.fps(), s.main.fps(), s.detect.fps(), s.track.fps(), s.mosaic.fps(), s.encode.fps());
  if (s.detectFace.items > 0) {
    const auto perKeyframeMs = [](const StageStat& stage) {
      return stage.items > 0 ? 1000.0 * stage.busySec / static_cast<double>(stage.items) : 0.0;
    };
    out += formatted("detectors:     student %.1f ms + face %.1f ms per keyframe, run in parallel: %.1f ms\n",
                     perKeyframeMs(s.detectStudent), perKeyframeMs(s.detectFace), perKeyframeMs(s.detect));
  }
  out += formatted("decode errors: %lld\n", static_cast<long long>(s.decodeErrors));
  out += formatted("elapsed:       %.2f s (%.1f fps)\n", s.elapsedSec, fps);
  return out;
}

RedactStats runRedact(const RedactOptions& options) {
  const Clock::time_point started = Clock::now();
  checkOptions(options);
  const std::string stem = stemOf(options.input);
  RedactStats stats;
  stats.outputDir = options.outRoot / stem;
  // Cheap checks first: a refused run must not pay for loading the model (CoreML compiles it).
  checkOutputDir(options.input, stats.outputDir, options.overwrite);
  VideoReader reader(options.input.string(), options.hwAccel);
  // Copied here because the decode thread will own the reader; audio MP4 cannot store is refused
  // now, before the previous output is removed, not on the encode thread.
  const std::vector<AudioStreamInfo> audio = snapshotAudioStreams(reader.audioStreams());
  requireMp4Audio(audio);
  // The keyframe spacing comes from the video's own frame rate (spec 2026-10-19 §2); a stream without one
  // is refused here, before the model is loaded.
  const DetectSchedule schedule = keyframeSchedule(options, *reader.videoStream());
  stats.videoFps = schedule.videoFps;
  stats.keyframeStride = schedule.stride;
  stats.detectFpsActual = schedule.actualFps;
  stats.detectFpsRequested = options.detectEveryFrame ? 0.0 : options.config.detectFps;
  stats.detectFpsAboveVideoRate = schedule.aboveVideoRate;
  const std::unique_ptr<Detector> detector = makeDetector(options, reader.info().width, reader.info().height);

  // The directories this run is about to create (outRoot and parents that do not exist yet), deepest
  // first: a failed run removes the ones that are still empty, so that it leaves nothing behind.
  std::vector<fs::path> createdRoots;
  for (fs::path dir = options.outRoot; !dir.empty() && !fs::exists(dir); dir = dir.parent_path()) {
    createdRoots.push_back(dir);
  }
  stats.video = stats.outputDir / (stem + ".redacted.mp4");
  const fs::path partial = stats.outputDir / (stem + ".redacted.mp4.partial");
  const std::vector<fs::path> written{partial, stats.video, stats.outputDir / kDetectionsFile,
                                      stats.outputDir / kMasksFile, stats.outputDir / kStatsFile};
  // The first check is long ago (loading the detector took its time, CoreML compiles the model there):
  // check again right before anything is removed, and outside the try below, whose handler deletes
  // `written` and the directory, which something else may own by now. What is left is the
  // time of two filesystem calls.
  checkOutputDir(options.input, stats.outputDir, options.overwrite);
  try {
    fs::remove_all(stats.outputDir);
    fs::create_directories(stats.outputDir);
    runPipeline(options, reader, *detector, schedule, audio, partial, started, stats);
    fs::rename(partial, stats.video);
    stats.frames = stats.decode.items;
    stats.track.items = stats.frames;
    stats.main = StageStat{stats.detect.busySec + stats.track.busySec + stats.mosaic.busySec, stats.frames};
    stats.decodeErrors = reader.decodeErrors();
    stats.droppedStreams = reader.droppedStreams();
    stats.audioStreams = reader.audioStreams().size();
    stats.detector = detector->description();
    if (const auto* merged = dynamic_cast<const MergedDetector*>(detector.get())) {
      stats.detectStudent = StageStat{merged->firstBusySec(), stats.detect.items};
      stats.detectFace = StageStat{merged->secondBusySec(), stats.detect.items};
    } else {
      stats.detectStudent = stats.detect;
    }
    stats.elapsedSec = secondsSince(started);
    writeStats(stats.outputDir / kStatsFile, stats, options);
    return stats;
  } catch (...) {
    // Everything in the directory is this run's own (it was empty or replaced above); the
    // directory goes too, so that a retry without --overwrite is not refused.
    std::error_code ignored;
    for (const fs::path& p : written) fs::remove(p, ignored);
    fs::remove(stats.outputDir, ignored);
    for (const fs::path& dir : createdRoots) {
      if (fs::is_empty(dir, ignored)) fs::remove(dir, ignored);
    }
    throw;
  }
}

}  // namespace vc
