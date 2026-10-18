import json
import subprocess

import pytest

from vc_eval import integrity
from vc_eval.integrity import FFMPEG, check_integrity, probe


def _video(path, seconds=1, audio=True, rate=30, faststart=False, size="160x120"):
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if faststart:
        args += ["-movflags", "+faststart"]
    args.append(str(path))
    subprocess.run(args, check=True)
    return path


def _two_length_video(path, video_seconds, audio_seconds, audio_delay=0.0):
    """160x120 video and sine audio of independent lengths; the audio can start `audio_delay` s late."""
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x120:r=30:d={video_seconds}"]
    if audio_delay:
        args += ["-itsoffset", str(audio_delay)]
    args += ["-f", "lavfi", "-i", f"sine=f=440:d={audio_seconds}"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    subprocess.run(args, check=True)
    return path


def _corrupt_middle(src, dst, length=400):
    """Copy of `src` with `length` bytes in the middle of the file overwritten (a faststart mp4 keeps its moov in front)."""
    data = bytearray(src.read_bytes())
    mid = len(data) // 2
    data[mid:mid + length] = bytes((i * 37 + 11) % 256 for i in range(length))
    dst.write_bytes(bytes(data))
    return dst


def test_probe_reads_streams(tmp_path):
    info = probe(_video(tmp_path / "a.mp4"))
    assert info.video.codec == "h264" and info.video.frames == 30
    assert info.fps == pytest.approx(30)
    assert info.audio is not None and info.audio.codec == "aac"


def test_identical_files_pass(tmp_path):
    v = _video(tmp_path / "a.mp4")
    r = check_integrity(v, v)
    assert r.passed and set(r.checks) == {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync"}


def test_missing_audio_and_wrong_length_fail(tmp_path):
    src = _video(tmp_path / "src.mp4", seconds=2)
    out = _video(tmp_path / "out.mp4", seconds=1, audio=False)
    r = check_integrity(src, out)
    assert not r.passed
    assert not r.checks["frame_count"] and not r.checks["audio_kept"]


def test_probe_rejects_a_non_media_file(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_text("not a video")
    with pytest.raises(ValueError, match="ffprobe"):
        probe(bad)


CHECKS = {"decodable", "frame_count", "resolution", "duration", "audio_kept", "av_sync"}


def test_probe_of_a_clean_file_has_no_decode_errors(tmp_path):
    assert probe(_video(tmp_path / "a.mp4")).errors == ()


def test_corrupted_bytes_fail_decodable_although_ffprobe_exits_zero(tmp_path):
    src = _video(tmp_path / "src.mp4", seconds=3, faststart=True)
    corrupt = _corrupt_middle(src, tmp_path / "corrupt.mp4")
    r = check_integrity(src, corrupt)
    assert r.checks["decodable"] is False and not r.passed
    assert "decode error" in r.details["decodable"]
    # the damage is only visible as decode errors: the container still reads and every frame is counted
    info = probe(corrupt)
    assert info.errors and info.video.frames == probe(src).video.frames


def test_decode_errors_already_in_the_source_are_not_held_against_the_output(tmp_path):
    src = _corrupt_middle(_video(tmp_path / "clean.mp4", seconds=3, faststart=True), tmp_path / "src.mp4")
    r = check_integrity(src, src)
    assert r.checks["decodable"] is True
    n = len(probe(src).errors)
    assert n > 0 and f"source {n}" in r.details["decodable"] and f"redacted {n}" in r.details["decodable"]


def test_unknown_audio_duration_fails_audio_kept(tmp_path):
    src = _two_length_video(tmp_path / "src.mkv", video_seconds=3, audio_seconds=3)
    out = _two_length_video(tmp_path / "out.mkv", video_seconds=3, audio_seconds=1)
    assert probe(src).audio.duration is None and probe(out).audio.duration is None  # what mkv reports
    r = check_integrity(src, out)
    assert r.checks["audio_kept"] is False
    assert "unknown duration" in r.details["audio_kept"]
    assert not r.passed


def test_an_empty_output_reports_every_check_failed_without_raising(tmp_path):
    src = _video(tmp_path / "src.mp4")
    out = tmp_path / "out.mp4"
    out.write_bytes(b"")
    r = check_integrity(src, out)
    assert set(r.checks) == CHECKS and not any(r.checks.values()) and not r.passed
    assert "ffprobe" in r.details["decodable"]


def test_a_missing_output_reports_every_check_failed_without_raising(tmp_path):
    r = check_integrity(_video(tmp_path / "src.mp4"), tmp_path / "nope.mp4")
    assert set(r.checks) == CHECKS and not any(r.checks.values())


def test_a_source_that_cannot_be_probed_still_raises(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"")
    with pytest.raises(ValueError, match="ffprobe"):
        check_integrity(src, _video(tmp_path / "out.mp4"))


def test_delayed_audio_fails_av_sync(tmp_path):
    src = _two_length_video(tmp_path / "src.mp4", video_seconds=2, audio_seconds=2)
    out = _two_length_video(tmp_path / "out.mp4", video_seconds=2, audio_seconds=2, audio_delay=0.5)
    r = check_integrity(src, out)
    assert r.checks["av_sync"] is False
    assert "audio offset changed by" in r.details["av_sync"]


def test_a_source_without_audio_passes_the_audio_checks(tmp_path):
    src = _video(tmp_path / "src.mp4", audio=False)
    r = check_integrity(src, src)
    assert r.checks["audio_kept"] is True and r.checks["av_sync"] is True
    assert r.passed


def _drop_from_probe(monkeypatch, path, edit):
    """ffprobe output for `path` (only) goes through `edit(json)` first, as if the file's headers lacked those fields."""
    real = subprocess.run

    def run(cmd, *args, **kwargs):
        out = real(cmd, *args, **kwargs)
        if cmd[0] == integrity.FFPROBE and cmd[-1] == str(path):
            data = json.loads(out.stdout)
            edit(data)
            return subprocess.CompletedProcess(cmd, out.returncode, json.dumps(data), out.stderr)
        return out

    monkeypatch.setattr(integrity.subprocess, "run", run)


def _no_format_duration(data):
    del data["format"]["duration"]


def _no_start_time(codec_type):
    def edit(data):
        for stream in data["streams"]:
            if stream["codec_type"] == codec_type:
                del stream["start_time"]

    return edit


def test_probe_reports_an_unknown_duration_and_start_time_as_none_not_zero(tmp_path, monkeypatch):
    v = _video(tmp_path / "a.mp4")
    assert probe(v).duration == pytest.approx(1.0, abs=0.1) and probe(v).video.start == 0.0  # known: a real 0.0 stays 0.0
    _drop_from_probe(monkeypatch, v, _no_format_duration)
    assert probe(v).duration is None
    _drop_from_probe(monkeypatch, v, _no_start_time("video"))
    assert probe(v).video.start is None and probe(v).audio.start is not None


@pytest.mark.parametrize("which", ["source", "redacted"])
def test_an_unknown_container_duration_fails_the_duration_check(tmp_path, monkeypatch, which):
    src, out = _video(tmp_path / "src.mp4"), _video(tmp_path / "out.mp4")
    _drop_from_probe(monkeypatch, {"source": src, "redacted": out}[which], _no_format_duration)
    r = check_integrity(src, out)
    assert r.checks["duration"] is False and not r.passed
    assert "unknown" in r.details["duration"] and "0.000" not in r.details["duration"]
    assert r.checks["frame_count"] is True  # only the affected check fails


@pytest.mark.parametrize("which, stream", [("source", "video"), ("source", "audio"), ("redacted", "video"), ("redacted", "audio")])
def test_an_unknown_stream_start_time_fails_av_sync(tmp_path, monkeypatch, which, stream):
    src, out = _video(tmp_path / "src.mp4"), _video(tmp_path / "out.mp4")
    _drop_from_probe(monkeypatch, {"source": src, "redacted": out}[which], _no_start_time(stream))
    r = check_integrity(src, out)
    assert r.checks["av_sync"] is False and not r.passed
    assert "unknown" in r.details["av_sync"] and "start_time" in r.details["av_sync"]
    assert r.checks["audio_kept"] is True and r.checks["duration"] is True


def test_an_unknown_start_time_does_not_matter_without_audio(tmp_path, monkeypatch):
    src = _video(tmp_path / "src.mp4", audio=False)
    _drop_from_probe(monkeypatch, src, _no_start_time("video"))
    r = check_integrity(src, src)
    assert r.checks["av_sync"] is True and r.passed


def test_a_different_resolution_fails_the_resolution_check(tmp_path):
    src = _video(tmp_path / "src.mp4")
    out = _video(tmp_path / "out.mp4", size="320x240")
    r = check_integrity(src, out)
    assert r.checks["resolution"] is False and not r.passed
    assert "160x120" in r.details["resolution"] and "320x240" in r.details["resolution"]
    assert r.checks["frame_count"] is True and r.checks["duration"] is True
    same = check_integrity(src, src)
    assert same.checks["resolution"] is True and "160x120" in same.details["resolution"]


def test_an_unknown_resolution_fails_the_resolution_check(tmp_path, monkeypatch):
    src, out = _video(tmp_path / "src.mp4"), _video(tmp_path / "out.mp4")

    def no_width(data):
        del data["streams"][0]["width"]

    _drop_from_probe(monkeypatch, out, no_width)
    r = check_integrity(src, out)
    assert r.checks["resolution"] is False and "unknown" in r.details["resolution"]


def _count_ffprobe_runs(monkeypatch):
    """A dict path -> number of ffprobe runs made from now on."""
    real = subprocess.run
    counts = {}

    def run(cmd, *args, **kwargs):
        if cmd[0] == integrity.FFPROBE:
            counts[cmd[-1]] = counts.get(cmd[-1], 0) + 1
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(integrity.subprocess, "run", run)
    return counts


def test_probe_decodes_a_file_once_per_cache_block_and_every_time_outside_one(tmp_path, monkeypatch):
    v = _video(tmp_path / "a.mp4")
    counts = _count_ffprobe_runs(monkeypatch)
    probe(v), probe(v)
    assert counts[str(v)] == 2  # no cache outside a block
    with integrity.probe_cache():
        first = probe(v)
        assert probe(v) is first and counts[str(v)] == 3
        check_integrity(v, v)
        assert counts[str(v)] == 3  # both files of check_integrity came from the cache
    probe(v)
    assert counts[str(v)] == 4


def test_a_file_that_changes_inside_a_cache_block_is_probed_again(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v, seconds=1)
    with integrity.probe_cache():
        assert probe(v).video.frames == 30
        _video(v, seconds=2)  # replaced by a longer clip: the size and mtime differ
        assert probe(v).video.frames == 60


def test_a_failed_probe_is_not_cached(tmp_path):
    v = tmp_path / "a.mp4"
    with integrity.probe_cache():
        with pytest.raises(ValueError):
            probe(v)
        _video(v)
        assert probe(v).video.frames == 30


def test_the_report_carries_the_frame_count_of_the_redacted_video(tmp_path):
    v = _video(tmp_path / "a.mp4")
    assert check_integrity(v, v).redacted_frames == 30
    assert check_integrity(v, tmp_path / "nope.mp4").redacted_frames is None
