#include "vc_redact/redact_job.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include "common/bounded_queue.h"
#include "detect/fake_detector.h"
#include "io/av_ptr.h"
#include "io/video_writer.h"
#include "mask/mosaic.h"
#include "sampler/frame_sampler.h"
#include "sampler/luma_thumb.h"
#include "track/box_timeline.h"
#include "vc_redact/records.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMaxQueueCapacity = 64;
constexpr size_t kMinQueueCapacity = 4;
constexpr uint64_t kQueueBudgetBytes = 256ull * 1024 * 1024;
constexpr uint64_t kBytesPerPixel = 3;  // the largest layout the reader emits: 8-bit 4:4:4
// A keyframe is forced after this many buffered frames, so stalled timestamps cannot grow the buffer.
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

std::unique_ptr<Detector> makeDetector(const RedactOptions& o, int width, int height) {
  if (o.fakeDetections) return std::make_unique<FakeDetector>(*o.fakeDetections);
  return std::make_unique<OnnxYoloDetector>(*o.modelDir, o.provider, o.config.decode, width, height);
}

// Keyframe selection, detection, BoxTimeline and mosaic: everything between decode and encode.
class Processor {
 public:
  Processor(const RedactOptions& options, Detector& detector, const VideoInfo& info, std::string video,
            std::ostream& detections, std::ostream& masks, BoundedQueue<Item>& out, RedactStats& stats)
      : options_(options),
        detector_(detector),
        video_(std::move(video)),
        detections_(detections),
        masks_(masks),
        out_(out),
        stats_(stats),
        sampler_(SamplerConfig{options.config.detectFps, options.config.maxDetectFps,
                               options.config.motionThreshold, std::nullopt, std::nullopt}),
        timeline_(options.config.timeline, info.width, info.height),
        thumbWidth_(std::min(kThumbWidth, info.width)),
        thumbHeight_(std::min(kThumbHeight, info.height)) {}

  void onVideo(VideoItem item) {
    const Clock::time_point started = Clock::now();
    const VideoFrame view = viewOf(item);
    LumaThumb thumb = makeLumaThumb(view.luma(), thumbWidth_, thumbHeight_);
    const double motion = previous_ ? meanAbsDiff(thumb, *previous_) : 0.0;
    const double diffToKept = lastKept_ ? meanAbsDiff(thumb, *lastKept_) : 1.0;
    const bool sampled =
        options_.detectEveryFrame || sampler_.offer(item.ptsMs, motion, diffToKept) != SampleReason::None;
    const bool key = sampled || buffer_.size() >= kMaxBufferedFrames;
    if (key) lastKept_ = thumb;
    previous_ = std::move(thumb);
    stats_.track.busySec += secondsSince(started);
    if (!key) {
      buffer_.push_back(std::move(item));
      return;
    }

    const Clock::time_point detectStarted = Clock::now();
    const std::vector<Box> boxes = detector_.detect(view);
    stats_.detect.busySec += secondsSince(detectStarted);
    ++stats_.detect.items;
    ++stats_.keyframes;
    stats_.detections += static_cast<int64_t>(boxes.size());
    for (const Box& b : boxes) {
      detections_ << detectionRecord(video_, item.index, item.ptsMs, b, detector_.classes(), detector_.source());
    }

    const Clock::time_point trackStarted = Clock::now();
    const std::vector<std::vector<MaskBox>> masks =
        timeline_.advance(times(), Keyframe{FrameTime{item.index, item.ptsMs}, boxes});
    stats_.track.busySec += secondsSince(trackStarted);
    emit(masks);
    buffer_.push_back(std::move(item));
  }

  void onEnd() {
    const Clock::time_point started = Clock::now();
    const std::vector<std::vector<MaskBox>> masks = timeline_.finish(times());
    stats_.track.busySec += secondsSince(started);
    emit(masks);
    if (!out_.push(Item{EndItem{}})) throw PipelineAborted{};
  }

 private:
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

  // Writes masks.jsonl, pixelates and sends every buffered frame downstream, in order.
  void emit(const std::vector<std::vector<MaskBox>>& masks) {
    if (masks.size() != buffer_.size()) throw VcError("internal error: masks do not match the buffered frames");
    for (size_t k = 0; k < buffer_.size(); ++k) {
      VideoItem& frame = buffer_[k];
      masks_ << maskRecord(video_, frame.index, masks[k], detector_.classes());
      if (!masks[k].empty()) {
        const Clock::time_point started = Clock::now();
        checkAv(av_frame_make_writable(frame.frame.get()), "make frame writable");
        std::vector<PixelRect> rects;
        rects.reserve(masks[k].size());
        for (const MaskBox& m : masks[k]) {
          rects.push_back(coveringRect(m.x1, m.y1, m.x2, m.y2, frame.frame->width, frame.frame->height));
        }
        applyMosaic(*frame.frame, rects, options_.config.mosaic);
        stats_.mosaic.busySec += secondsSince(started);
        ++stats_.maskedFrames;
        stats_.maskBoxes += static_cast<int64_t>(masks[k].size());
      }
      ++stats_.mosaic.items;
      if (!out_.push(Item{std::move(frame)})) throw PipelineAborted{};
    }
    buffer_.clear();
  }

  const RedactOptions& options_;
  Detector& detector_;
  std::string video_;
  std::ostream& detections_;
  std::ostream& masks_;
  BoundedQueue<Item>& out_;
  RedactStats& stats_;
  FrameSampler sampler_;
  BoxTimeline timeline_;
  int thumbWidth_;
  int thumbHeight_;
  std::optional<LumaThumb> previous_;
  std::optional<LumaThumb> lastKept_;
  std::vector<VideoItem> buffer_;
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
  c.preferHardware = preferHardware;
  copyPresentation(*stream, c);  // aspect ratio, colour tags and rotation stay as in the input
  return c;
}

// `audio`: the reader's audio streams, snapshotted on the main thread. `started`: the start of the
// run, for stats.setupSec.
void runPipeline(const RedactOptions& options, VideoReader& reader, Detector& detector,
                 const std::vector<AudioStreamInfo>& audio, const fs::path& partial, Clock::time_point started,
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
  Processor processor(options, detector, info, options.input.filename().string(), detections, masks, processed,
                      stats);
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
  j["masked_frames"] = s.maskedFrames;
  j["mask_boxes"] = s.maskBoxes;
  j["decode_errors"] = s.decodeErrors;
  j["detector"] = s.detector;
  j["encoder"] = s.encoder;
  j["audio_streams"] = s.audioStreams;
  j["audio_packets"] = s.audioPackets;
  j["dropped_streams"] = s.droppedStreams;
  j["detect_every_frame"] = o.detectEveryFrame;
  j["config"] = redactConfigToJson(o.config);
  // decode, main and encode run in parallel; main is detect + track + mosaic, one after another.
  j["stages"]["decode"] = stageJson(s.decode);
  j["stages"]["main"] = stageJson(s.main);
  j["stages"]["encode"] = stageJson(s.encode);
  j["stages"]["detect"] = stageJson(s.detect);
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
  out += formatted("frames:        %lld (keyframes %lld, detections %lld, masked %lld frames / %lld boxes)\n",
                   static_cast<long long>(s.frames), static_cast<long long>(s.keyframes),
                   static_cast<long long>(s.detections), static_cast<long long>(s.maskedFrames),
                   static_cast<long long>(s.maskBoxes));
  out += "detector:      " + s.detector + "\n";
  out += "encoder:       " + s.encoder + "\n";
  out += formatted("audio:         %zu stream(s), %lld packets copied; dropped streams: ", s.audioStreams,
                   static_cast<long long>(s.audioPackets)) +
         joined(s.droppedStreams) + "\n";
  out += formatted("setup:         %.2f s (input, output checks, model load / CoreML compile, first frame)\n",
                   s.setupSec);
  out += formatted("stages (fps):  decode %.1f, main %.1f (detect %.1f, track %.1f, mosaic %.1f), encode %.1f\n",
                   s.decode.fps(), s.main.fps(), s.detect.fps(), s.track.fps(), s.mosaic.fps(), s.encode.fps());
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
  try {
    fs::remove_all(stats.outputDir);
    fs::create_directories(stats.outputDir);
    runPipeline(options, reader, *detector, audio, partial, started, stats);
    fs::rename(partial, stats.video);
    stats.frames = stats.decode.items;
    stats.track.items = stats.frames;
    stats.main = StageStat{stats.detect.busySec + stats.track.busySec + stats.mosaic.busySec, stats.frames};
    stats.decodeErrors = reader.decodeErrors();
    stats.droppedStreams = reader.droppedStreams();
    stats.audioStreams = reader.audioStreams().size();
    stats.detector = detector->backend().empty() ? detector->source() : detector->source() + "/" + detector->backend();
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
