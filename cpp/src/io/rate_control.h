#pragma once

namespace vc {

// How the video encoder spends bits (configs/redact.json rate_control). Quality: constant quality, a file as large
// as the picture needs (h264_videotoolbox q 65, libx264 crf 18, h264_nvenc vbr cq 19 with no bit rate). Bitrate: the
// bit rate in WriterConfig::bitRate.
enum class RateControl { Quality, Bitrate };

}  // namespace vc
