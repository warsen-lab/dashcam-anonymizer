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

// vc_redact's --hwaccel default (spec 2026-10-20 §3): auto on Linux (CUDA, else software), none elsewhere (on macOS
// software decoding is faster than VideoToolbox for 1080p H.264).
HwAccel defaultRedactHwAccel(Platform platform);

struct RedactOptions {
  std::filesystem::path input;
  std::filesystem::path outRoot;  // outputs go to outRoot / <input stem>
  std::optional<std::filesystem::path> modelDir;        // model.onnx + model.json
  std::optional<std::filesystem::path> fakeDetections;  // detections.jsonl replayed instead
  // The YuNet face model (`vc_train yunet`), run on every keyframe next to the detector above; its
  // boxes are merged with the detector's (MergedDetector). Without it only that detector runs.
  std::optional<std::filesystem::path> faceModelDir;
  // The verifier model (`vc_train verifier`, spec 2026-10-21 §4.3): the boxes needsVerification() picks are scored
  // on each keyframe and those below the pass threshold are not masked. Without it nothing is verified.
  std::optional<std::filesystem::path> verifierModelDir;
  // Writes verify_crops.bin / verify_crops.jsonl (VerifyDump) next to detections.jsonl: the crop of every box
  // needsVerification() picks on the keyframes whose frame index is a multiple of dumpVerifyEvery (>= 1).
  bool dumpVerifyCrops = false;
  int64_t dumpVerifyEvery = 1;
  RedactConfig config;
  bool detectEveryFrame = false;
  ExecutionProvider provider = ExecutionProvider::Auto;  // providerOrder: CoreML then CPU on macOS
  HwAccel hwAccel = defaultRedactHwAccel(buildPlatform());
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
  int64_t detections = 0;  // the boxes passed to the timeline: not counting the dropped ones or verifier rejections
  int64_t largeFacesDropped = 0;  // large faces scoring below large_face_min_score (spec 2026-10-19 §3.6)
  int64_t narrowPlatesDropped = 0;  // plates narrower than plate_min_width_px (2026-10-22)
  // Track-level masking (spec 2026-10-27): the lookahead and backfill in keyframes, the mask boxes they added
  // (masks.jsonl origin "bridged" / "backfilled") and the most frames held at once.
  int lookaheadKeyframes = 0;
  int backfillKeyframes = 0;
  int64_t bridgedMaskBoxes = 0;
  int64_t backfilledMaskBoxes = 0;
  int64_t bufferedFramesPeak = 0;
  std::string verifier;             // the verifier model's name; empty without one
  int64_t verifyChecked = 0;        // boxes the verifier scored
  int64_t verifyRejectedFaces = 0;  // ... and rejected, by class
  int64_t verifyRejectedPlates = 0;
  int64_t verifyDumped = 0;         // crops written with dumpVerifyCrops
  int64_t maskedFrames = 0;
  int64_t maskBoxes = 0;
  int64_t decodeErrors = 0;
  int64_t audioPackets = 0;
  // Keyframe spacing (spec 2026-10-19 §2): a keyframe every keyframeStride frames of a videoFps video, i.e.
  // detectFpsActual detections per second of video. detectFpsRequested is the configured detect_fps (0 with
  // --detect-every-frame); detectFpsAboveVideoRate when it exceeded the video frame rate.
  double videoFps = 0;
  int64_t keyframeStride = 0;
  double detectFpsActual = 0;
  double detectFpsRequested = 0;
  bool detectFpsAboveVideoRate = false;
  std::string encoder;   // e.g. "h264_nvenc", "h264_videotoolbox", "libx264"; the output is always H.264
  RateControl rateControl = RateControl::Quality;  // configs/redact.json rate_control
  // The encoder's target with rate_control "bitrate": h264OutputBitRate of the source (spec 2026-10-20 §3); 0 (null
  // in stats.json) with "quality".
  int64_t outputBitRate = 0;
  // Set when the source is not 8-bit 4:2:0 and is converted (h264FormatNote); redactWarnings prints it.
  std::string formatNote{};
  // "cpu-frames": frames go through main memory between decode, detection, mosaic and encode (the GPU path of
  // spec 2026-10-20 §4, "gpu", comes with stage 2b).
  std::string pipeline = "cpu-frames";
  // "student/<backend>" (providerLabel: "student/CoreML", "student/CPU", on Linux + NVIDIA "student/TensorRT") or
  // "fake"; with a face model both, e.g. "student/CoreML + yunet/CoreML"
  std::string detector;
  std::vector<std::string> droppedStreams;
  size_t audioStreams = 0;
  // decode, main and encode run in parallel on three threads; main is the processing thread, which
  // runs detect, track and mosaic one after another (its busy time is their sum).
  StageStat decode;
  StageStat main;    // items = frames
  StageStat detect;         // items = keyframes; wall time of each keyframe's detection (with a face model both
                            // detectors at once, plus the merge)
  StageStat detectStudent;  // the student (or fake) detector's own busy time; equals detect without a face model
  StageStat detectFace;     // the face model's own busy time (it overlaps detectStudent); {0, 0} without one
  StageStat verify;         // items = keyframes with something to verify or dump: cropping + scoring
  StageStat track;   // keyframe choice + BoxTimeline; items = frames
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

// The most frames vc_redact holds at once with this lookahead (spec 2026-10-27 §3.3): lookahead + 1 keyframe
// intervals of `stride` frames, an interval being at most the 120 frames after which a keyframe is forced.
int64_t lookaheadBufferFrames(int lookahead, int64_t stride);

// Throws VcError (naming track_lookahead_keyframes) when, with a lookahead, lookaheadBufferFrames of a width x height
// video at 3 bytes per pixel exceed 2 GiB. Without a lookahead the buffer is what it always was: no check.
void checkLookaheadBuffer(int lookahead, int64_t stride, int width, int height);

// Lines for stderr after a successful run (empty when there is nothing to warn about): a note when detect_fps
// is above the video's frame rate (so every frame is a keyframe), a note when the source was not 8-bit 4:2:0
// (formatNote), and a warning for decoder errors, i.e. corrupt packets that were skipped.
std::vector<std::string> redactWarnings(const RedactStats& stats);

// The summary printed after a successful run: one "name: value" line each for the output, the frame
// and mask counts, the track masking (lookahead, backfill, the mask boxes they added, the frames buffered),
// the detect rate (every how many frames of what frame rate), detector, encoder, audio,
// setup time, stage rates, decode errors and elapsed time.
std::string redactSummary(const RedactStats& stats);

// Redacts options.input into <outRoot>/<stem>/: <stem>.redacted.mp4 (written as .partial and
// renamed on success), detections.jsonl, masks.jsonl (one line per frame) and stats.json, with dumpVerifyCrops
// also verify_crops.bin / verify_crops.jsonl.
// On each keyframe, after the large-face rule, the boxes needsVerification() picks are cropped (verifier_crop.h):
// the crops of dumped keyframes go to the dump, and with a verifier model the boxes of a class it knows are scored;
// one below its pass threshold never reaches BoxTimeline and is written with "verify", "verified": false and
// "confirmed": false (spec 2026-10-21 §4.3).
// The options, the output directory, the input (including whether MP4 can store its audio) and the
// detector (with the face model and the verifier, if any) are checked, in that order, before anything is created or removed (with overwrite, the
// previous output is removed once they pass); any later failure removes this run's files (and the output root and its new parent
// directories when this run created them and they are empty) and rethrows. Decoding, detection plus
// masking, and encoding run on three threads; the first error stops all of them.
//
// Throws std::invalid_argument unless exactly one of modelDir / fakeDetections is set or when dumpVerifyEvery < 1;
// OutputExistsError when the output directory exists without overwrite, or with overwrite is
// neither empty nor holds masks.jsonl; VcError for an invalid config, media, detector and I/O
// failures (a frame without a timestamp, an encoder that cannot be opened, fewer frames encoded
// than decoded, a video stream without a usable frame rate, ...). Errors from BoxTimeline
// (std::invalid_argument) propagate as they are.
RedactStats runRedact(const RedactOptions& options);

}  // namespace vc
