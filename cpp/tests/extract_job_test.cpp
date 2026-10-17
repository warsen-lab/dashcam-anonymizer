#include "vc_extract/extract_job.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

// A clean output root per test.
fs::path freshOutRoot(const std::string& name) {
  const fs::path root = test::tmpDir() / "extract" / name;
  fs::remove_all(root);
  return root;
}

ExtractOptions optionsFor(const fs::path& input, const fs::path& outRoot) {
  ExtractOptions options;
  options.input = input;
  options.outRoot = outRoot;
  options.hwAccel = HwAccel::None;
  return options;
}

std::vector<std::string> readLines(const fs::path& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

// Frame numbers of the frames.csv rows (header skipped) with the given reason.
std::vector<int> framesWithReason(const std::vector<std::string>& lines,
                                  const std::string& reason) {
  std::vector<int> frames;
  for (size_t i = 1; i < lines.size(); ++i) {
    std::stringstream row(lines[i]);
    std::string video, frame, pts, rowReason;
    std::getline(row, video, ',');
    std::getline(row, frame, ',');
    std::getline(row, pts, ',');
    std::getline(row, rowReason, ',');
    if (rowReason == reason) frames.push_back(std::stoi(frame));
  }
  return frames;
}

TEST(ExtractJob, SamplesAtBaseRateAndWritesCsv) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("base_rate"));
  options.sampler.motionThreshold = 1.0;  // nothing counts as motion

  const ExtractStats stats = runExtract(options);

  EXPECT_EQ(stats.outputDir, options.outRoot / "basic");
  EXPECT_EQ(stats.decodedFrames, 60);
  EXPECT_EQ(stats.keptFrames, 10);
  EXPECT_EQ(stats.keptBase, 10);
  EXPECT_EQ(stats.decodeErrors, 0);
  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  ASSERT_EQ(lines.size(), 11u);
  EXPECT_EQ(lines[0], "video,frame,pts_ms,reason,motion_score");
  EXPECT_EQ(lines[1], "basic.mp4,0,0.0,base,0.0000");
  EXPECT_EQ(framesWithReason(lines, "base"),
            (std::vector<int>{0, 6, 12, 18, 24, 30, 36, 42, 48, 54}));
  EXPECT_TRUE(fs::exists(stats.outputDir / "000000.jpg"));
  EXPECT_TRUE(fs::exists(stats.outputDir / "000054.jpg"));
  EXPECT_FALSE(fs::exists(stats.outputDir / "000001.jpg"));
}

TEST(ExtractJob, SamplesFasterWhileMotionIsHigh) {
  ExtractOptions options = optionsFor(test::staticThenNoiseVideo(), freshOutRoot("motion"));
  options.sampler.motionThreshold = 0.03;  // static half scores 0, noise half ~0.08

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  EXPECT_EQ(framesWithReason(lines, "base"), (std::vector<int>{0, 6, 12, 18, 24, 30}));
  std::vector<int> motion;
  for (int i = 32; i < 60; i += 2) motion.push_back(i);
  EXPECT_EQ(framesWithReason(lines, "motion"), motion);
  EXPECT_EQ(stats.keptMotion, 14);
}

TEST(ExtractJob, DedupDropsStaticFrames) {
  ExtractOptions options = optionsFor(test::staticThenNoiseVideo(), freshOutRoot("dedup"));
  options.sampler.motionThreshold = 1.0;
  options.sampler.dedupThreshold = 0.01;

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  EXPECT_EQ(framesWithReason(lines, "base"), (std::vector<int>{0, 30, 36, 42, 48, 54}));
  EXPECT_EQ(stats.keptFrames, 6);
}

TEST(ExtractJob, ClipExportsEveryFrameInWindow) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("clip"));
  options.sampler.clip = ClipWindow{500.0, 500.0};

  const ExtractStats stats = runExtract(options);

  const std::vector<std::string> lines = readLines(stats.outputDir / "frames.csv");
  std::vector<int> expected;
  for (int i = 15; i < 30; ++i) expected.push_back(i);
  EXPECT_EQ(framesWithReason(lines, "clip"), expected);
  EXPECT_EQ(stats.keptClip, 15);
  EXPECT_EQ(stats.decodedFrames, 31);  // stops at the first frame past the window
}

TEST(ExtractJob, RefusesExistingOutputUnlessOverwrite) {
  ExtractOptions options = optionsFor(test::basicVideo(), freshOutRoot("overwrite"));
  const ExtractStats first = runExtract(options);
  std::ofstream(first.outputDir / "stale.txt") << "old";

  EXPECT_THROW(runExtract(options), OutputExistsError);
  EXPECT_TRUE(fs::exists(first.outputDir / "stale.txt"));

  options.overwrite = true;
  runExtract(options);
  EXPECT_FALSE(fs::exists(first.outputDir / "stale.txt"));
}

TEST(ExtractJob, OverwriteRefusesDirectoryWithoutFramesCsv) {
  const fs::path outRoot = freshOutRoot("overwrite_foreign_dir");
  fs::create_directories(outRoot / "basic");
  std::ofstream(outRoot / "basic" / "notes.txt") << "not ours";
  ExtractOptions options = optionsFor(test::basicVideo(), outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), OutputExistsError);

  EXPECT_TRUE(fs::exists(outRoot / "basic" / "notes.txt"));
}

TEST(ExtractJob, OverwriteRefusesWhenOutputWouldContainInput) {
  const fs::path outRoot = freshOutRoot("overwrite_contains_input");
  fs::create_directories(outRoot / "basic");
  const fs::path input = outRoot / "basic" / "basic.mp4";
  fs::copy_file(test::basicVideo(), input);
  std::ofstream(outRoot / "basic" / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n";
  ExtractOptions options = optionsFor(input, outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), VcError);

  EXPECT_TRUE(fs::is_regular_file(input));
  EXPECT_EQ(fs::file_size(input), fs::file_size(test::basicVideo()));
}

TEST(ExtractJob, RefusesOutputPathThatIsTheInputFile) {
  const fs::path outRoot = freshOutRoot("output_is_input");
  fs::create_directories(outRoot);
  const fs::path input = outRoot / "trip01";  // no extension: stem == file name
  fs::copy_file(test::basicVideo(), input);
  ExtractOptions options = optionsFor(input, outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), VcError);

  EXPECT_TRUE(fs::is_regular_file(input));
  EXPECT_EQ(fs::file_size(input), fs::file_size(test::basicVideo()));
}

TEST(ExtractJob, RefusesInputWhoseStemCannotNameADirectory) {
  const fs::path root = freshOutRoot("dot_stem");
  const fs::path outRoot = root / "out";
  fs::create_directories(root / "in");
  fs::create_directories(outRoot);
  const fs::path input = root / "in" / "..mp4";  // stem is "." -> outRoot / "." is outRoot itself
  fs::copy_file(test::basicVideo(), input);
  // Make outRoot look like a previous output so only the stem check can stop the wipe.
  std::ofstream(outRoot / "frames.csv") << "video,frame,pts_ms,reason,motion_score\n";
  std::ofstream(outRoot / "other.txt") << "another video's frames";
  ExtractOptions options = optionsFor(input, outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), VcError);

  EXPECT_TRUE(fs::is_regular_file(input));
  EXPECT_TRUE(fs::exists(outRoot / "other.txt"));
}

TEST(ExtractJob, BadInputLeavesExistingOutputAlone) {
  const fs::path outRoot = freshOutRoot("bad_input");
  fs::create_directories(outRoot / "missing");
  std::ofstream(outRoot / "missing" / "keep.txt") << "keep";
  ExtractOptions options = optionsFor(test::tmpDir() / "missing.mp4", outRoot);
  options.overwrite = true;

  EXPECT_THROW(runExtract(options), VcError);

  EXPECT_TRUE(fs::exists(outRoot / "missing" / "keep.txt"));
}

}  // namespace
}  // namespace vc
