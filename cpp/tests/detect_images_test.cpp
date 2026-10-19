#include "vc_detect/detect_images.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

std::vector<std::string> lines(const fs::path& path) {
  std::ifstream in(path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

fs::path constModel() { return fs::path(VC_TEST_DATA_DIR) / "const_yolo"; }

fs::path gray(int size) {
  return test::makeMedia("gray" + std::to_string(size) + ".jpg",
                         "-f lavfi -i color=c=gray:s=" + std::to_string(size) + "x" + std::to_string(size) +
                             " -frames:v 1 -update 1");
}

// A fresh frames directory holding one grey 128x128 JPEG per name in `frameFiles`.
fs::path framesDirWith(const std::string& name, const std::vector<std::string>& frameFiles) {
  const fs::path dir = test::tmpDir() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  for (const std::string& file : frameFiles) fs::copy_file(gray(128), dir / file);
  return dir;
}

DetectImagesOptions optionsFor(const fs::path& framesDir, const std::string& outName) {
  DetectImagesOptions o;
  o.modelDir = constModel();
  o.framesDir = framesDir;
  o.out = test::tmpDir() / outName;
  fs::remove(o.out);
  return o;
}

TEST(DetectImages, RunsTheModelOnEveryListedFrame) {
  const fs::path gray = test::makeMedia("gray128.jpg", "-f lavfi -i color=c=gray:s=128x128 -frames:v 1 -update 1");
  const fs::path frames = test::tmpDir() / "detect_frames";
  fs::remove_all(frames);
  fs::create_directories(frames);
  fs::copy_file(gray, frames / "000000.jpg");
  fs::copy_file(gray, frames / "000007.jpg");
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "t.mp4,0,0.0,base,0.0000\n"
                                          "t.mp4,7,233.3,base,0.0100\n";
  DetectImagesOptions o;
  o.modelDir = fs::path(VC_TEST_DATA_DIR) / "const_yolo";
  o.framesDir = frames;
  o.out = test::tmpDir() / "detect_images.jsonl";
  EXPECT_EQ(runDetectImages(o), 2);
  const std::vector<std::string> got = lines(o.out);
  ASSERT_EQ(got.size(), 4u);
  EXPECT_EQ(got[0].rfind(R"({"video":"t.mp4","frame":0,"pts_ms":0.0,"cls":"face",)", 0), 0u) << got[0];
  EXPECT_NE(got[0].find(R"("src":"student")"), std::string::npos);
  EXPECT_EQ(got[2].rfind(R"({"video":"t.mp4","frame":7,"pts_ms":233.3,)", 0), 0u) << got[2];
}

TEST(DetectImages, KeepsACommaInTheVideoName) {
  const fs::path frames = framesDirWith("detect_comma", {"000003.jpg"});
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "a,b.mp4,3,100.0,base,0.0000\n";
  const DetectImagesOptions o = optionsFor(frames, "detect_comma.jsonl");
  EXPECT_EQ(runDetectImages(o), 1);
  const std::vector<std::string> got = lines(o.out);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0].rfind(R"({"video":"a,b.mp4","frame":3,"pts_ms":100.0,)", 0), 0u) << got[0];
}

// A frames.csv without a single frame is a wrong directory (or a failed vc_extract), not "no
// detections": an empty detections file would be taken for a clean result.
TEST(DetectImages, RejectsAnEmptyFramesList) {
  const fs::path frames = framesDirWith("detect_empty", {});
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n";
  const DetectImagesOptions o = optionsFor(frames, "detect_empty.jsonl");
  try {
    runDetectImages(o);
    ADD_FAILURE() << "no throw for an empty frames.csv";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("lists no frames"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(o.out));
  EXPECT_FALSE(fs::exists(fs::path(o.out).concat(".partial")));
}

// A frames.csv saved with Windows line endings (Python's csv module writes them by default) is the
// same list of frames.
TEST(DetectImages, AcceptsAFramesCsvWithWindowsLineEndings) {
  const fs::path frames = framesDirWith("detect_crlf", {"000000.jpg"});
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\r\n"
                                          "t.mp4,0,0.0,base,0.0\r\n";
  DetectImagesOptions o = optionsFor(frames, "detect_crlf.jsonl");
  EXPECT_EQ(runDetectImages(o), 1);
  EXPECT_EQ(lines(o.out).size(), 2u);  // the constant model's face and plate
}

TEST(DetectImages, TheScoreThresholdChangesTheOutput) {
  const fs::path frames = framesDirWith("detect_score", {"000000.jpg"});
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "t.mp4,0,0.0,base,0.0000\n";
  // The constant model reports a face at 0.9 and a plate at 0.7 on every image.
  const struct {
    float score;
    size_t records;
  } cases[] = {{0.15f, 2}, {0.8f, 1}, {0.95f, 0}};
  for (const auto& c : cases) {
    DetectImagesOptions o = optionsFor(frames, "detect_score.jsonl");
    o.decode.scoreThreshold = c.score;
    EXPECT_EQ(runDetectImages(o), 1);
    const std::vector<std::string> got = lines(o.out);
    EXPECT_EQ(got.size(), c.records) << "score " << c.score;
    if (c.score == 0.8f) {
      ASSERT_EQ(got.size(), 1u);
      EXPECT_NE(got[0].find(R"("cls":"face")"), std::string::npos) << got[0];
    }
  }
}

// The letterbox scale depends on the image size, so a detector built for one size cannot serve
// another: mixed sizes in one frames.csv need the detector rebuilt.
TEST(DetectImages, RebuildsTheDetectorWhenImageSizesChange) {
  const fs::path frames = framesDirWith("detect_sizes", {});
  fs::copy_file(gray(128), frames / "000000.jpg");
  fs::copy_file(gray(64), frames / "000001.jpg");
  fs::copy_file(gray(128), frames / "000002.jpg");
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "t.mp4,0,0.0,base,0.0000\n"
                                          "t.mp4,1,33.3,base,0.0000\n"
                                          "t.mp4,2,66.7,base,0.0000\n";
  const DetectImagesOptions o = optionsFor(frames, "detect_sizes.jsonl");
  EXPECT_EQ(runDetectImages(o), 3);
  const std::vector<std::string> got = lines(o.out);
  ASSERT_EQ(got.size(), 6u);
  // The model input is 64x64: a 128x128 image is scaled by 0.5 (the face is at x 24..40), a 64x64
  // image by 1 (x 12..20).
  EXPECT_NE(got[0].find(R"("x1":24.0)"), std::string::npos) << got[0];
  EXPECT_NE(got[2].find(R"("x1":12.0)"), std::string::npos) << got[2];
  EXPECT_NE(got[4].find(R"("x1":24.0)"), std::string::npos) << got[4];
}

TEST(DetectImages, RejectsAFramesDirWithoutFramesCsv) {
  DetectImagesOptions o;
  o.modelDir = fs::path(VC_TEST_DATA_DIR) / "const_yolo";
  o.framesDir = test::tmpDir() / "no_such_frames";
  o.out = test::tmpDir() / "detect_none.jsonl";
  EXPECT_THROW(runDetectImages(o), VcError);
}

TEST(DetectImages, RejectsABadFramesCsv) {
  const std::vector<std::string> bodies = {
      "frame,video\n",
      "video,frame,pts_ms,reason,motion_score\nt.mp4,0,0.0,base\n",
      "video,frame,pts_ms,reason,motion_score\nt.mp4,zero,0.0,base,0.0\n",
      "video,frame,pts_ms,reason,motion_score\nt.mp4,0,soon,base,0.0\n",
      "video,frame,pts_ms,reason,motion_score\nt.mp4,0,nan,base,0.0\n",
      "video,frame,pts_ms,reason,motion_score\nt.mp4,-1,0.0,base,0.0\n",
  };
  for (const std::string& body : bodies) {
    const fs::path frames = framesDirWith("detect_bad_csv", {"000000.jpg"});
    std::ofstream(frames / "frames.csv") << body;
    const DetectImagesOptions o = optionsFor(frames, "detect_bad_csv.jsonl");
    EXPECT_THROW(runDetectImages(o), VcError) << body;
    EXPECT_FALSE(fs::exists(o.out)) << body;
  }
}

TEST(DetectImages, AFailureLeavesNoOutputFile) {
  const fs::path frames = framesDirWith("detect_missing_jpg", {"000000.jpg"});  // 000001.jpg is listed but absent
  std::ofstream(frames / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n"
                                          "t.mp4,0,0.0,base,0.0000\n"
                                          "t.mp4,1,33.3,base,0.0000\n";
  const DetectImagesOptions o = optionsFor(frames, "detect_missing_jpg.jsonl");
  EXPECT_THROW(runDetectImages(o), VcError);
  EXPECT_FALSE(fs::exists(o.out));
  EXPECT_FALSE(fs::exists(fs::path(o.out).concat(".partial")));
}

}  // namespace
}  // namespace vc
