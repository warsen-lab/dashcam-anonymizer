#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "detect/onnx_yolo_detector.h"
#include "io/video_reader.h"
#include "vc_redact/redact_config.h"

namespace vc {

struct RedactOptions {
  std::filesystem::path input;
  std::filesystem::path outRoot;  // outputs go to outRoot / <input stem>
  std::optional<std::filesystem::path> modelDir;        // model.onnx + model.json
  std::optional<std::filesystem::path> fakeDetections;  // detections.jsonl replayed instead
  RedactConfig config;
  bool detectEveryFrame = false;
  ExecutionProvider provider = ExecutionProvider::CoreML;
  HwAccel hwAccel = HwAccel::None;
  bool preferHardwareEncoder = true;
  bool overwrite = false;
};

struct StageStat {
  double busySec = 0;
  int64_t items = 0;

  double fps() const { return busySec > 0 ? static_cast<double>(items) / busySec : 0.0; }
};

struct RedactStats {
  std::filesystem::path outputDir;
  std::filesystem::path video;  // the redacted file
  int64_t frames = 0;
  int64_t keyframes = 0;
  int64_t detections = 0;
  int64_t maskedFrames = 0;
  int64_t maskBoxes = 0;
  int64_t decodeErrors = 0;
  int64_t audioPackets = 0;
  std::string encoder;   // e.g. "h264_videotoolbox", "libx264"
  std::string detector;  // "student/CoreML", "student/CPU" or "fake"
  std::vector<std::string> droppedStreams;
  size_t audioStreams = 0;
  // decode, main and encode run in parallel on three threads; main is the processing thread, which
  // runs detect, track and mosaic one after another (its busy time is their sum).
  StageStat decode;
  StageStat main;    // items = frames
  StageStat detect;  // items = keyframes
  StageStat track;   // keyframe selection + BoxTimeline; items = frames
  StageStat mosaic;
  StageStat encode;
  // From the start of runRedact until the first frame reaches the processing thread: opening the
  // input, the output checks, loading the model (CoreML compiles it here), starting the threads and
  // decoding the first frame. The encoder is opened later, on the encode thread (in encode's time).
  double setupSec = 0;
  double elapsedSec = 0;
};

// Capacity of each queue between the pipeline stages: clamp(256 MiB / bytes per frame, 4, 64). The
// frame size assumes 3 bytes per pixel, the largest layout the reader emits (8-bit 4:4:4), so a
// queue never holds much more than 256 MiB of frames.
size_t pipelineQueueCapacity(int width, int height);

// Lines for stderr after a successful run (empty when there is nothing to warn about): currently
// decoder errors, i.e. corrupt packets that were skipped.
std::vector<std::string> redactWarnings(const RedactStats& stats);

// The summary printed after a successful run: one "name: value" line each for the output, the frame
// and mask counts, detector, encoder, audio, setup time, stage rates, decode errors and elapsed time.
std::string redactSummary(const RedactStats& stats);

// Redacts options.input into <outRoot>/<stem>/: <stem>.redacted.mp4 (written as .partial and
// renamed on success), detections.jsonl, masks.jsonl (one line per frame) and stats.json.
// The options, the output directory, the input (including whether MP4 can store its audio) and the
// detector are checked, in that order, before anything is created or removed (with overwrite, the
// previous output is removed once they pass); any later failure removes this run's files (and the output root and its new parent
// directories when this run created them and they are empty) and rethrows. Decoding, detection plus
// masking, and encoding run on three threads; the first error stops all of them.
//
// Throws std::invalid_argument unless exactly one of modelDir / fakeDetections is set;
// OutputExistsError when the output directory exists without overwrite, or with overwrite is
// neither empty nor holds masks.jsonl; VcError for an invalid config, media, detector and I/O
// failures (a frame without a timestamp, an encoder that cannot be opened, fewer frames encoded
// than decoded, ...). Errors from the sampler and BoxTimeline (std::invalid_argument) propagate as
// they are.
RedactStats runRedact(const RedactOptions& options);

}  // namespace vc
