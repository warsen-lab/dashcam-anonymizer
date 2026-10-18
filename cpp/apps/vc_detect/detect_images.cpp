#include "vc_detect/detect_images.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "common/vc_error.h"
#include "io/frame_file_name.h"
#include "io/video_reader.h"
#include "vc_redact/records.h"

extern "C" {
#include <libavutil/frame.h>
}

namespace vc {
namespace fs = std::filesystem;
namespace {

constexpr const char* kFramesHeader = "video,frame,pts_ms,reason,motion_score";
// Columns after the video name: frame, pts_ms, reason, motion_score.
constexpr std::size_t kColumnsAfterVideo = 4;

struct Row {
  std::string video;
  int64_t frame = 0;
  double ptsMs = 0;
};

std::vector<std::string> splitOnCommas(const std::string& line) {
  std::vector<std::string> cells(1);
  for (const char c : line) {
    if (c == ',') {
      cells.emplace_back();
    } else {
      cells.back().push_back(c);
    }
  }
  return cells;
}

// vc_extract writes the video name unquoted, so a comma in it adds cells at the front: the last
// four columns are located from the right and the rest is the name.
Row parseRow(const std::string& line, const std::string& where) {
  const std::vector<std::string> cells = splitOnCommas(line);
  if (cells.size() < kColumnsAfterVideo + 1) throw VcError(where + ": expected 5 columns");
  const std::size_t frameColumn = cells.size() - kColumnsAfterVideo;
  Row row;
  for (std::size_t i = 0; i < frameColumn; ++i) row.video += (i == 0 ? "" : ",") + cells[i];
  try {
    std::size_t used = 0;
    row.frame = std::stoll(cells[frameColumn], &used);
    if (used != cells[frameColumn].size() || row.frame < 0) throw std::invalid_argument("frame");
    row.ptsMs = std::stod(cells[frameColumn + 1], &used);
    if (used != cells[frameColumn + 1].size() || !std::isfinite(row.ptsMs)) throw std::invalid_argument("pts_ms");
  } catch (const std::exception&) {
    throw VcError(where + ": bad frame or pts_ms");
  }
  return row;
}

std::vector<Row> readFramesCsv(const fs::path& framesDir) {
  const fs::path csv = framesDir / "frames.csv";
  std::ifstream in(csv);
  if (!in) throw VcError("cannot read " + csv.string());
  std::string line;
  std::getline(in, line);
  if (line != kFramesHeader) throw VcError(csv.string() + ": unexpected header '" + line + "'");
  std::vector<Row> rows;
  int lineNo = 1;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.empty()) continue;
    rows.push_back(parseRow(line, csv.string() + " line " + std::to_string(lineNo)));
  }
  // An empty list is a wrong directory or a failed vc_extract; an empty detections file would pass
  // for "nothing detected".
  if (rows.empty()) throw VcError(csv.string() + " lists no frames");
  return rows;
}

void detectAll(const DetectImagesOptions& options, const std::vector<Row>& rows, const fs::path& target) {
  std::ofstream out(target);
  if (!out) throw VcError("cannot write " + target.string());
  std::unique_ptr<OnnxYoloDetector> detector;
  int width = 0;
  int height = 0;
  for (const Row& row : rows) {
    const fs::path image = options.framesDir / frameFileName(row.frame);
    VideoReader reader(image.string(), HwAccel::None);
    const VideoFrame* decoded = reader.next();
    if (decoded == nullptr) throw VcError("cannot decode " + image.string());
    if (!detector || decoded->av->width != width || decoded->av->height != height) {
      width = decoded->av->width;
      height = decoded->av->height;
      detector = std::make_unique<OnnxYoloDetector>(options.modelDir, options.provider, options.decode, width, height);
    }
    VideoFrame frame = *decoded;
    frame.index = row.frame;
    frame.ptsMs = row.ptsMs;
    for (const Box& box : detector->detect(frame)) {
      out << detectionRecord(row.video, row.frame, row.ptsMs, box, detector->classes(), detector->source());
    }
  }
  out.close();
  if (!out) throw VcError("write " + target.string() + " failed");
}

}  // namespace

int64_t runDetectImages(const DetectImagesOptions& options) {
  const std::vector<Row> rows = readFramesCsv(options.framesDir);
  const fs::path partial = fs::path(options.out).concat(".partial");
  try {
    detectAll(options, rows, partial);
    std::error_code ec;
    fs::rename(partial, options.out, ec);
    if (ec) throw VcError("cannot rename " + partial.string() + " to " + options.out.string() + ": " + ec.message());
  } catch (...) {
    std::error_code ignored;
    fs::remove(partial, ignored);
    throw;
  }
  return static_cast<int64_t>(rows.size());
}

}  // namespace vc
