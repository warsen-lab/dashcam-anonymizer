#pragma once

#include <stdexcept>
#include <string>

namespace vc {

// Invalid command line.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Failure while reading, decoding or writing media.
class VcError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The output directory already exists and overwrite was not requested, or it is not a directory
// the tool may replace.
class OutputExistsError : public VcError {
 public:
  using VcError::VcError;
};

// Human-readable text for an FFmpeg AVERROR code.
std::string avErrorText(int errnum);

// Throws VcError("<what>: <ffmpeg message>") when ret < 0, otherwise returns ret.
int checkAv(int ret, const std::string& what);

}  // namespace vc
