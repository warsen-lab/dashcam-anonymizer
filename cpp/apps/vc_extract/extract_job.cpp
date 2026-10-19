#include "vc_extract/extract_job.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "io/frame_file_name.h"
#include "io/jpeg_writer.h"
#include "io/video_reader.h"
#include "sampler/frame_sampler.h"
#include "sampler/luma_thumb.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

// <outRoot>/<input stem>; refuses names that resolve to the output root or its parent.
fs::path outputDirFor(const ExtractOptions& options) {
  const fs::path stem = options.input.stem();
  if (stem.empty() || stem == "." || stem == "..") {
    throw VcError("cannot derive an output directory name from " + options.input.string());
  }
  return options.outRoot / stem;
}

// A directory vc_extract may replace: empty, or holding the frames.csv of a previous run.
bool looksLikeExtractOutput(const fs::path& dir) {
  return fs::is_directory(dir) &&
         (fs::is_empty(dir) || fs::is_regular_file(dir / "frames.csv"));
}

// True when path is dir itself or lies below it. Both must be weakly canonical.
bool isSameOrInside(const fs::path& path, const fs::path& dir) {
  const fs::path relative = path.lexically_relative(dir);
  return !relative.empty() && *relative.begin() != "..";
}

// Validates everything that could make creating `dir` destructive, without touching the
// filesystem.
void checkOutputDir(const fs::path& input, const fs::path& dir, bool overwrite) {
  if (fs::exists(dir)) {
    if (!overwrite) {
      throw OutputExistsError("output directory exists: " + dir.string() +
                              " (pass --overwrite to replace it)");
    }
    if (!looksLikeExtractOutput(dir)) {
      throw OutputExistsError("refusing to overwrite " + dir.string() +
                              ": not a vc_extract output directory");
    }
  }
  if (isSameOrInside(fs::weakly_canonical(fs::absolute(input)),
                     fs::weakly_canonical(fs::absolute(dir)))) {
    throw VcError("output directory " + dir.string() +
                  " would contain the input video; choose another --out");
  }
}

void prepareOutputDir(const fs::path& dir) {
  fs::remove_all(dir);  // no-op when absent
  fs::create_directories(dir);
}

std::string csvRow(const std::string& video, const VideoFrame& frame, SampleReason reason,
                   double motion) {
  char numbers[96];
  std::snprintf(numbers, sizeof(numbers), "%lld,%.1f,%s,%.4f",
                static_cast<long long>(frame.index), frame.ptsMs, toString(reason), motion);
  return video + "," + numbers + "\n";
}

void countReason(ExtractStats& stats, SampleReason reason) {
  ++stats.keptFrames;
  if (reason == SampleReason::Base) ++stats.keptBase;
  if (reason == SampleReason::Motion) ++stats.keptMotion;
  if (reason == SampleReason::Clip) ++stats.keptClip;
}

}  // namespace

ExtractStats runExtract(const ExtractOptions& options) {
  const auto started = std::chrono::steady_clock::now();
  VideoReader reader(options.input.string(), options.hwAccel);
  FrameSampler sampler(options.sampler);
  JpegWriter jpeg(options.jpegQscale);

  ExtractStats stats;
  stats.outputDir = outputDirFor(options);
  checkOutputDir(options.input, stats.outputDir, options.overwrite);
  prepareOutputDir(stats.outputDir);
  const fs::path csvPath = stats.outputDir / "frames.csv";
  std::ofstream csv(csvPath);
  csv << "video,frame,pts_ms,reason,motion_score\n";
  if (!csv) throw VcError("cannot write " + csvPath.string());

  const std::string video = options.input.filename().string();
  const int thumbWidth = std::min(kThumbWidth, reader.info().width);
  const int thumbHeight = std::min(kThumbHeight, reader.info().height);
  const double clipEndMs = options.sampler.clip
                               ? options.sampler.clip->startMs + options.sampler.clip->durationMs
                               : std::numeric_limits<double>::infinity();
  std::optional<LumaThumb> previous;
  std::optional<LumaThumb> lastKept;

  while (const VideoFrame* frame = reader.next()) {
    ++stats.decodedFrames;
    if (frame->ptsMs >= clipEndMs) break;
    LumaThumb thumb = makeLumaThumb(frame->luma(), thumbWidth, thumbHeight);
    const double motion = previous ? meanAbsDiff(thumb, *previous) : 0.0;
    const double diffToKept = lastKept ? meanAbsDiff(thumb, *lastKept) : 1.0;
    const SampleReason reason = sampler.offer(frame->ptsMs, motion, diffToKept);
    if (reason != SampleReason::None) {
      jpeg.write(*frame->av, stats.outputDir / frameFileName(frame->index));
      csv << csvRow(video, *frame, reason, motion);
      countReason(stats, reason);
      lastKept = thumb;
    }
    previous = std::move(thumb);
  }

  if (stats.decodedFrames == 0) {
    throw VcError("no frames could be decoded from " + options.input.string());
  }
  csv.close();
  if (!csv) throw VcError("write " + csvPath.string() + " failed");
  stats.decodeErrors = reader.decodeErrors();
  stats.usedHardware = reader.usingHardware();
  stats.elapsedSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return stats;
}

}  // namespace vc
