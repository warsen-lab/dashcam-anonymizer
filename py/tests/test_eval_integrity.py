import subprocess

import pytest

from vc_eval.integrity import FFMPEG, check_integrity, probe


def _video(path, seconds=1, audio=True, rate=30, faststart=False):
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x120:r={rate}:d={seconds}"]
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
    assert r.passed and set(r.checks) == {"decodable", "frame_count", "duration", "audio_kept", "av_sync"}


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


CHECKS = {"decodable", "frame_count", "duration", "audio_kept", "av_sync"}


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
