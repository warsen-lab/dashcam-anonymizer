#include "vc_extract/extract_job.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "io/jpeg_writer.h"
#include "io/video_reader.h"
#include "sampler/frame_sampler.h"
#include "sampler/luma_thumb.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

void prepareOutputDir(const fs::path& dir, bool overwrite) {
  if (fs::exists(dir)) {
    if (!overwrite) {
      throw OutputExistsError("output directory exists: " + dir.string() +
                              " (pass --overwrite to replace it)");
    }
    fs::remove_all(dir);
  }
  fs::create_directories(dir);
}

std::string frameFileName(int64_t index) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06lld.jpg", static_cast<long long>(index));
  return name;
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
  stats.outputDir = options.outRoot / options.input.stem();
  prepareOutputDir(stats.outputDir, options.overwrite);
  std::ofstream csv(stats.outputDir / "frames.csv");
  csv << "video,frame,pts_ms,reason,motion_score\n";

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

  csv.close();
  if (!csv) throw VcError("write " + (stats.outputDir / "frames.csv").string() + " failed");
  stats.decodeErrors = reader.decodeErrors();
  stats.usedHardware = reader.usingHardware();
  stats.elapsedSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return stats;
}

}  // namespace vc
