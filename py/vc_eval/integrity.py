"""Output integrity of a redacted video against its source (proxy-validation spec §6.4)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
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
    )


def probe(path: Path) -> MediaInfo:
    """Streams of a media file; decodes them to count frames (ffprobe -count_frames).

    A non-zero exit raises ValueError. Decode errors on an otherwise readable file come with exit 0, so the
    stderr lines are kept in `MediaInfo.errors`.
    """
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

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


_CHECKS = ("decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync")


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


def check_integrity(source: Path, redacted: Path) -> IntegrityReport:
    """Source and redacted must agree; a source that cannot be probed raises ValueError, a broken output fails."""
    s = probe(source)
    try:
        r = probe(redacted)
    except ValueError as exc:
        return _unreadable(str(exc))
    frame = 1.0 / s.fps if s.fps > 0 else 1.0 / 30
    report = IntegrityReport()
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
