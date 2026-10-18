import subprocess

import pytest

from vc_eval.integrity import FFMPEG, check_integrity, probe


def _video(path, seconds=1, audio=True, rate=30):
    args = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x120:r={rate}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path


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
