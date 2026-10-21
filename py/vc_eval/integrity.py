"""Output integrity of a redacted video against its source (proxy-validation spec §6.4)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


@dataclass(frozen=True)
class StreamInfo:
    """What ffprobe reported for one stream; a field it did not report (or reported as N/A) is None, never 0."""

    codec: str
    frames: int | None
    duration: float | None
    start: float | None
    width: int | None = None
    height: int | None = None
    level: int | None = None  # H.264 / HEVC level as ffprobe reports it (level_idc: 52 for 5.2); None when unknown


@dataclass(frozen=True)
class MediaInfo:
    video: StreamInfo
    audio: StreamInfo | None
    fps: float
    duration: float | None  # the container's; None when it reports none
    errors: tuple[str, ...] = ()  # decode errors ffprobe printed to stderr while still exiting 0


def _number(value, cast=float):
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return None
    return None if isinstance(number, float) and not math.isfinite(number) else number


def _rate(text: str) -> float:
    num, _, den = (text or "0/1").partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def _stream(s: dict) -> StreamInfo:
    return StreamInfo(
        codec=s.get("codec_name", ""),
        frames=_number(s.get("nb_read_frames"), int),
        duration=_number(s.get("duration")),
        start=_number(s.get("start_time")),
        width=_number(s.get("width"), int),
        height=_number(s.get("height"), int),
        level=_level(s.get("level")),
    )


def _level(value) -> int | None:
    """ffprobe's level, None for a missing or unknown one (it prints -99 when the stream has none)."""
    level = _number(value, int)
    return level if level is not None and level > 0 else None


# H.264 Table A-1 (ITU-T H.264 Annex A): level_idc, MaxMBPS (macroblocks per second), MaxFS (macroblocks per frame).
# Level 1b has the limits of level 1 and is left out.
H264_LEVELS = (
    (10, 1485, 99), (11, 3000, 396), (12, 6000, 396), (13, 11880, 396), (20, 11880, 396), (21, 19800, 792),
    (22, 20250, 1620), (30, 40500, 1620), (31, 108000, 3600), (32, 216000, 5120), (40, 245760, 8192),
    (41, 245760, 8192), (42, 522240, 8704), (50, 589824, 22080), (51, 983040, 36864), (52, 2073600, 36864),
    (60, 4177920, 139264), (61, 8355840, 139264), (62, 16711680, 139264),
)


def required_h264_level(width: int, height: int, fps: float) -> int | None:
    """level_idc of the lowest H.264 level whose frame size and macroblock rate hold width x height at fps
    (3840x2160 at 60 fps: 52, i.e. 5.2); None beyond level 6.2. Mirrors h264LevelIdc in cpp/src/io/video_writer.cpp."""
    frame_mbs = math.ceil(width / 16) * math.ceil(height / 16)
    for level, max_mbps, max_fs in H264_LEVELS:
        if frame_mbs <= max_fs and frame_mbs * fps <= max_mbps:
            return level
    return None


def _level_text(level: int) -> str:
    return f"{level // 10}.{level % 10}"


_probe_cache: dict[tuple, MediaInfo] | None = None


@contextmanager
def probe_cache():
    """Inside the block probe() remembers its answer per file, so a file is decoded for counting once per run.

    A file is the same file only while its path, size and modification time are; a rewritten file is probed again.
    """
    global _probe_cache
    outer = _probe_cache
    _probe_cache = {}
    try:
        yield
    finally:
        _probe_cache = outer


def _cache_key(path: Path) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None  # let ffprobe report the file as unreadable
    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)


def probe(path: Path) -> MediaInfo:
    """Streams of a media file; decodes them to count frames (ffprobe -count_frames).

    A non-zero exit raises ValueError. Decode errors on an otherwise readable file come with exit 0, so the
    stderr lines are kept in `MediaInfo.errors`. See probe_cache.
    """
    key = None if _probe_cache is None else _cache_key(path)
    if key is not None and key in _probe_cache:
        return _probe_cache[key]
    info = _probe(path)
    if key is not None:
        _probe_cache[key] = info
    return info


def _probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise ValueError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"ffprobe found no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return MediaInfo(
        video=_stream(video),
        audio=_stream(audio) if audio else None,
        fps=_rate(video.get("avg_frame_rate")),
        duration=_number(data.get("format", {}).get("duration")),
        errors=tuple(line.strip() for line in out.stderr.splitlines() if line.strip()),
    )


@dataclass
class IntegrityReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    redacted_frames: int | None = None  # frames decoded from the redacted video; None when it could not be probed or reports none

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync", "h264", "h264_level")


def _unreadable(reason: str) -> IntegrityReport:
    """Every check failed: the redacted file could not be probed at all."""
    return IntegrityReport(
        checks={name: False for name in _CHECKS},
        details={name: reason if name == "decodable" else "not checked: output unreadable" for name in _CHECKS},
    )


def _size(stream: StreamInfo) -> str:
    return f"{stream.width}x{stream.height}" if stream.width and stream.height else "unknown"


def _seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f} s"


def _check_h264(report: IntegrityReport, r: MediaInfo) -> None:
    """The output is H.264 (spec 2026-10-20 §3) and its level is at least the one its size and frame rate need."""
    codec = r.video.codec or "unknown"
    report.checks["h264"] = r.video.codec == "h264"
    report.details["h264"] = f"redacted video codec {codec}"
    if r.video.codec != "h264":
        report.checks["h264_level"] = False
        report.details["h264_level"] = f"not checked: the redacted video is {codec}, not H.264"
        return
    if not (r.video.width and r.video.height and r.fps > 0):
        report.checks["h264_level"] = False
        report.details["h264_level"] = f"unknown size or frame rate ({_size(r.video)}, {r.fps:g} fps)"
        return
    need = required_h264_level(r.video.width, r.video.height, r.fps)
    have = "unknown" if r.video.level is None else _level_text(r.video.level)
    wanted = "beyond 6.2" if need is None else _level_text(need)
    report.checks["h264_level"] = need is not None and r.video.level is not None and r.video.level >= need
    report.details["h264_level"] = f"level {have}; {_size(r.video)} at {r.fps:.2f} fps needs {wanted}"


def check_integrity(source: Path, redacted: Path) -> IntegrityReport:
    """Source and redacted must agree, and the output must be H.264 at a sufficient level; a source that cannot be
    probed raises ValueError, a broken output fails."""
    s = probe(source)
    try:
        r = probe(redacted)
    except ValueError as exc:
        return _unreadable(str(exc))
    report = _compare(s, r)
    _check_h264(report, r)
    return report


def _compare(s: MediaInfo, r: MediaInfo) -> IntegrityReport:
    """The checks of the redacted video `r` against its source `s`: decoding, frames, size, duration and audio."""
    frame = 1.0 / s.fps if s.fps > 0 else 1.0 / 30
    report = IntegrityReport(redacted_frames=r.video.frames)
    report.checks["decodable"] = bool(r.video.frames) and len(r.errors) <= len(s.errors)
    report.details["decodable"] = (
        f"{r.video.frames} frames decoded; decode errors: source {len(s.errors)}, redacted {len(r.errors)}"
        + (f": {r.errors[0]}" if r.errors else "")
    )
    report.checks["frame_count"] = (
        r.video.frames is not None and s.video.frames is not None and abs(r.video.frames - s.video.frames) <= 1
    )
    report.details["frame_count"] = f"source {s.video.frames}, redacted {r.video.frames}"
    sizes = (_size(s.video), _size(r.video))
    report.checks["resolution"] = "unknown" not in sizes and sizes[0] == sizes[1]
    report.details["resolution"] = f"source {sizes[0]}, redacted {sizes[1]}" + (" (unknown size)" if "unknown" in sizes else "")
    if s.duration is None or r.duration is None:
        # a container that reports no duration is no evidence of a match; never read it as 0.0 s
        report.checks["duration"] = False
        report.details["duration"] = f"unknown duration: source {_seconds(s.duration)}, redacted {_seconds(r.duration)}"
    else:
        report.checks["duration"] = abs(r.duration - s.duration) <= frame
        report.details["duration"] = f"source {s.duration:.3f} s, redacted {r.duration:.3f} s"
    if s.audio is None:
        report.checks["audio_kept"] = True
        report.checks["av_sync"] = True
        report.details["audio_kept"] = report.details["av_sync"] = "source has no audio"
        return report
    a = r.audio
    if a is None:
        report.checks["audio_kept"] = report.checks["av_sync"] = False
        report.details["audio_kept"] = f"source {s.audio.codec} {s.audio.duration} s, redacted none"
        report.details["av_sync"] = "redacted has no audio"
        return report
    report.details["audio_kept"] = f"source {s.audio.codec} {s.audio.duration} s, redacted {a.codec} {a.duration} s"
    if s.audio.duration is None or a.duration is None:
        # a missing duration is no evidence of a match (mkv, damaged headers); never compare None with None
        report.checks["audio_kept"] = False
        report.details["audio_kept"] += " (unknown duration)"
    else:
        report.checks["audio_kept"] = a.codec == s.audio.codec and abs(a.duration - s.audio.duration) <= frame
    starts = {"source video": s.video.start, "source audio": s.audio.start, "redacted video": r.video.start, "redacted audio": a.start}
    unknown = [name for name, start in starts.items() if start is None]
    if unknown:
        report.checks["av_sync"] = False
        report.details["av_sync"] = f"unknown start_time of {', '.join(unknown)}; the audio offset cannot be compared"
        return report
    shift = (a.start - r.video.start) - (s.audio.start - s.video.start)
    report.checks["av_sync"] = abs(shift) <= frame
    report.details["av_sync"] = f"audio offset changed by {shift * 1000:.1f} ms (limit {frame * 1000:.1f} ms)"
    return report
