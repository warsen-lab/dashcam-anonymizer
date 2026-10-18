"""Output integrity of a redacted video against its source (proxy-validation spec §6.4)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


@dataclass(frozen=True)
class StreamInfo:
    codec: str
    frames: int | None
    duration: float | None
    start: float


@dataclass(frozen=True)
class MediaInfo:
    video: StreamInfo
    audio: StreamInfo | None
    fps: float
    duration: float


def _number(value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _rate(text: str) -> float:
    num, _, den = (text or "0/1").partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def _stream(s: dict) -> StreamInfo:
    return StreamInfo(
        codec=s.get("codec_name", ""),
        frames=_number(s.get("nb_read_frames"), int),
        duration=_number(s.get("duration")),
        start=_number(s.get("start_time")) or 0.0,
    )


def probe(path: Path) -> MediaInfo:
    """Streams of a media file; decodes the video to count frames (ffprobe -count_frames)."""
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
        duration=_number(data.get("format", {}).get("duration")) or 0.0,
    )


@dataclass
class IntegrityReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def check_integrity(source: Path, redacted: Path) -> IntegrityReport:
    s, r = probe(source), probe(redacted)
    frame = 1.0 / s.fps if s.fps > 0 else 1.0 / 30
    report = IntegrityReport()
    report.checks["decodable"] = bool(r.video.frames)
    report.details["decodable"] = f"{r.video.frames} frames decoded"
    report.checks["frame_count"] = (
        r.video.frames is not None and s.video.frames is not None and abs(r.video.frames - s.video.frames) <= 1
    )
    report.details["frame_count"] = f"source {s.video.frames}, redacted {r.video.frames}"
    report.checks["duration"] = abs(r.duration - s.duration) <= frame
    report.details["duration"] = f"source {s.duration:.3f} s, redacted {r.duration:.3f} s"
    if s.audio is None:
        report.checks["audio_kept"] = True
        report.checks["av_sync"] = True
        report.details["audio_kept"] = report.details["av_sync"] = "source has no audio"
        return report
    a = r.audio
    report.checks["audio_kept"] = (
        a is not None and a.codec == s.audio.codec and abs((a.duration or 0.0) - (s.audio.duration or 0.0)) <= frame
    )
    report.details["audio_kept"] = (
        f"source {s.audio.codec} {s.audio.duration} s, redacted "
        + (f"{a.codec} {a.duration} s" if a else "none")
    )
    if a is None:
        report.checks["av_sync"] = False
        report.details["av_sync"] = "redacted has no audio"
        return report
    shift = (a.start - r.video.start) - (s.audio.start - s.video.start)
    report.checks["av_sync"] = abs(shift) <= frame
    report.details["av_sync"] = f"audio offset changed by {shift * 1000:.1f} ms (limit {frame * 1000:.1f} ms)"
    return report
