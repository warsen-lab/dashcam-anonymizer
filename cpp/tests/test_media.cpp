#include "test_media.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace vc::test {
namespace fs = std::filesystem;

fs::path tmpDir() {
  const fs::path dir = VC_TEST_TMP_DIR;
  fs::create_directories(dir);
  return dir;
}

fs::path makeMedia(const std::string& name, const std::string& args) {
  const fs::path output = tmpDir() / name;
  if (fs::exists(output)) return output;
  // ctest may run test processes in parallel: write a private file, then rename.
  const fs::path partial = tmpDir() / (std::to_string(getpid()) + "-" + name);
  const std::string command = std::string("\"") + VC_FFMPEG + "\" -v error -y " + args + " \"" +
                              partial.string() + "\"";
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("ffmpeg failed: " + command);
  }
  fs::rename(partial, output);
  return output;
}

fs::path basicVideo() {
  return makeMedia("basic.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=2 -c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path staticThenNoiseVideo() {
  return makeMedia(
      "static_then_noise.mp4",
      "-f lavfi -i color=c=gray:s=640x360:r=30:d=1,format=yuv420p "
      "-f lavfi -i \"nullsrc=s=640x360:r=30:d=1,geq=lum='random(1)*255':cb=128:cr=128,"
      "format=yuv420p\" "
      "-filter_complex \"[0:v][1:v]concat=n=2:v=1:a=0[v]\" -map \"[v]\" "
      "-c:v libx264 -pix_fmt yuv420p -g 30");
}

fs::path tenBitVideo() {
  return makeMedia("ten_bit.mp4",
                   "-f lavfi -i testsrc2=s=640x360:r=30:d=1 -c:v libx265 -pix_fmt yuv420p10le "
                   "-x265-params log-level=error");
}

fs::path audioOnlyFile() {
  return makeMedia("audio_only.m4a", "-f lavfi -i sine=f=440:d=1 -c:a aac");
}

}  // namespace vc::test
