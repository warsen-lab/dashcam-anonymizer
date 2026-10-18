import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box
from vc_eval import spotcheck
from vc_eval.integrity import FFMPEG
from vc_eval.spotcheck import BATCH, Item, Section, extract_frames, miss_tile, thumbnail, write_spotcheck


def _clip(path, seconds=2):
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x120:r=30:d={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_extract_frames_decodes_requested_frame_numbers(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=30:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    got = extract_frames(video, [45, 3, 3, 59], tmp_path / "out")
    assert sorted(got) == [3, 45, 59]
    assert got[45].name == "000045.jpg" and cv2.imread(str(got[45])).shape == (120, 160, 3)
    assert extract_frames(video, [], tmp_path / "none") == {}


def test_extract_frames_handles_more_than_one_batch(tmp_path):
    video = _clip(tmp_path / "v.mp4")
    got = extract_frames(video, list(range(60)), tmp_path / "out")
    assert sorted(got) == list(range(60))
    assert cv2.imread(str(got[59])).shape == (120, 160, 3) and got[51].name == "000051.jpg"


def test_extract_frames_rejects_frames_past_the_end_even_with_leftovers(tmp_path):
    video = _clip(tmp_path / "v.mp4")
    out = tmp_path / "out"
    out.mkdir()
    cv2.imwrite(str(out / "batch_00002.jpg"), np.full((120, 160, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="no frame 100"):
        extract_frames(video, [3, 100], out)


def test_miss_tile_puts_original_and_redacted_side_by_side(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    cv2.imwrite(str(a), np.full((400, 600, 3), 200, np.uint8))
    cv2.imwrite(str(b), np.full((400, 600, 3), 50, np.uint8))
    out = miss_tile(a, b, Box("face", 100, 100, 120, 130), tmp_path / "tile.jpg")
    tile = cv2.imread(str(out))
    assert tile.shape[0] == 160 and tile.shape[1] > 300
    only = miss_tile(a, None, Box("face", 590, 390, 600, 400), tmp_path / "only.jpg")
    assert cv2.imread(str(only)).shape[0] == 160


def test_thumbnail_keeps_the_aspect_ratio(tmp_path):
    src = tmp_path / "f.jpg"
    cv2.imwrite(str(src), np.zeros((1080, 1920, 3), np.uint8))
    assert cv2.imread(str(thumbnail(src, tmp_path / "t.jpg"))).shape == (360, 640, 3)


def test_write_spotcheck_escapes_text_and_uses_relative_paths(tmp_path):
    page = write_spotcheck(
        tmp_path,
        "抽查 <test>",
        "intro & notes",
        [Section("未遮住", "note", [Item("img/a.jpg", "frame 3 <face>")], [("视频", "../../redacted/a.mp4")])],
    )
    html = page.read_text(encoding="utf-8")
    assert page == tmp_path / "index.html"
    assert "抽查 &lt;test&gt;" in html and "intro &amp; notes" in html and "frame 3 &lt;face&gt;" in html
    assert 'src="img/a.jpg"' in html and 'href="../../redacted/a.mp4"' in html
    assert "http://" not in html and "https://" not in html


def _numbered_clip(path, frames=100):
    """160x120 clip of `frames` (<= 100) frames whose top band is the frame number: rows 0-15 show the tens digit,
    rows 16-31 the units digit, each as luma 16 + 20 * digit. B-frames and a short GOP make the decode order differ
    from the display order inside every group of pictures."""
    expr = "if(lt(Y,16),16+20*floor(N/10),if(lt(Y,32),16+20*mod(N,10),128))"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=160x120:r=30,geq=lum='{expr}':cb=128:cr=128",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-g", "12", "-keyint_min", "12",
         "-sc_threshold", "0", "-bf", "2", str(path)],
        check=True,
    )
    return path


def _number_in(image_path):
    """The frame number encoded in the top band of a decoded frame (the JPEG is full range, the clip limited range)."""
    gray = cv2.imread(str(image_path))[..., 1].astype(float)

    def digit(rows):
        return round(gray[rows, 20:140].mean() / 255 * 219 / 20)

    return 10 * digit(slice(2, 12)) + digit(slice(18, 28))


def test_every_extracted_frame_holds_the_number_it_was_requested_by_across_batches(tmp_path):
    video = _numbered_clip(tmp_path / "v.mp4")
    got = extract_frames(video, list(range(100)), tmp_path / "out")  # two batches: 0-49 and 50-99
    assert BATCH == 50 and sorted(got) == list(range(100))
    assert {n: _number_in(path) for n, path in got.items()} == {n: n for n in range(100)}


def test_sparse_requests_around_the_batch_boundary_get_their_own_frames(tmp_path):
    video = _numbered_clip(tmp_path / "v.mp4")
    wanted = [99, 0, 49, 50, 51, 7, 13, 48]  # unsorted, with duplicates
    got = extract_frames(video, wanted + [49, 50], tmp_path / "out")
    assert {n: _number_in(path) for n, path in got.items()} == {n: n for n in set(wanted)}


def test_frames_of_a_stream_copy_cut_are_numbered_by_the_decode_order_of_the_cut(tmp_path):
    clip = _numbered_clip(tmp_path / "v.mp4")
    cut = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "1.0", "-i", str(clip), "-c", "copy", str(cut)], check=True)
    decoded = []  # an independent decode of the cut: what a reader numbering frames by decode order sees
    capture = cv2.VideoCapture(str(cut))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert 50 < len(decoded) < 100  # the batch boundary lies inside the cut
    got = extract_frames(cut, list(range(len(decoded))), tmp_path / "out")
    first = _number_in(got[0])
    assert 0 < first <= 30  # the cut starts at or before the requested second, on a keyframe
    assert [_number_in(got[k]) for k in range(len(decoded))] == list(range(first, first + len(decoded)))
    for k in (0, 1, 49, 50, len(decoded) - 1):
        cv2.imwrite(str(tmp_path / "ref.png"), decoded[k])
        assert _number_in(tmp_path / "ref.png") == _number_in(got[k])


def test_extract_frames_works_in_a_directory_whose_name_contains_percent_signs(tmp_path):
    (tmp_path / "100%d").mkdir()
    video = _numbered_clip(tmp_path / "100%d" / "v.mp4")
    got = extract_frames(video, [3, 60], tmp_path / "100%d" / "50%" / "redacted")
    assert {n: _number_in(path) for n, path in got.items()} == {3: 3, 60: 60}
    assert all("%" in str(path) and path.is_file() for path in got.values())
    assert not list(tmp_path.glob("*b_*")) and not list((tmp_path / "100%d").glob("*.jpg"))  # nothing written elsewhere


def test_extract_frames_runs_ffmpeg_without_stdin_in_the_output_dir_and_stops_after_the_batch(tmp_path, monkeypatch):
    video = _numbered_clip(tmp_path / "v.mp4", frames=60)
    real = subprocess.run
    calls = []

    def spy(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(spotcheck.subprocess, "run", spy)
    out = tmp_path / "out"
    extract_frames(video, list(range(0, 60, 2)), out)  # 30 frames: one batch
    cmd, kwargs = calls[0]
    assert "-nostdin" in cmd and kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["cwd"] == out and cmd[-1] == "batch_%05d.jpg" and Path(cmd[cmd.index("-i") + 1]).is_absolute()
    assert cmd[cmd.index("-frames:v") + 1] == "30"


def test_missing_frames_are_still_reported_when_the_batch_is_cut_short(tmp_path):
    video = _numbered_clip(tmp_path / "v.mp4", frames=60)
    with pytest.raises(ValueError, match="no frame 70"):
        extract_frames(video, [3, 59, 70], tmp_path / "out")


def test_miss_tile_refuses_originals_and_redacted_frames_of_different_sizes(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    cv2.imwrite(str(a), np.full((400, 600, 3), 200, np.uint8))
    cv2.imwrite(str(b), np.full((200, 300, 3), 50, np.uint8))
    with pytest.raises(ValueError, match=r"600x400.*300x200"):
        miss_tile(a, b, Box("face", 100, 100, 120, 130), tmp_path / "tile.jpg")
    assert not (tmp_path / "tile.jpg").exists()


def test_miss_tile_refuses_a_box_outside_the_image(tmp_path):
    a = tmp_path / "a.jpg"
    cv2.imwrite(str(a), np.full((400, 600, 3), 200, np.uint8))
    with pytest.raises(ValueError, match="outside"):
        miss_tile(a, None, Box("face", 700, 500, 720, 520), tmp_path / "tile.jpg")
    assert not (tmp_path / "tile.jpg").exists()


def test_miss_tile_and_thumbnail_name_the_file_they_could_not_write(tmp_path, monkeypatch):
    a = tmp_path / "a.jpg"
    cv2.imwrite(str(a), np.full((400, 600, 3), 200, np.uint8))
    monkeypatch.setattr(spotcheck.cv2, "imwrite", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="tile.jpg"):
        miss_tile(a, None, Box("face", 100, 100, 120, 130), tmp_path / "tile.jpg")
    with pytest.raises(ValueError, match="thumb.jpg"):
        thumbnail(a, tmp_path / "thumb.jpg")
