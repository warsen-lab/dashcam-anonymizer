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
  StageStat decode;
  StageStat detect;  // items = keyframes
  StageStat track;   // keyframe selection + BoxTimeline; items = frames
  StageStat mosaic;
  StageStat encode;
  double elapsedSec = 0;
};

// Redacts options.input into <outRoot>/<stem>/: <stem>.redacted.mp4 (written as .partial and
// renamed on success), detections.jsonl, masks.jsonl (one line per frame) and stats.json.
// The options, the input, the detector and the output directory are all checked before anything is
// created; any later failure removes this run's files and rethrows. Decoding, detection plus
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
