import subprocess

import cv2
import numpy as np
import pytest

from vc_label.contracts import Box
from vc_eval.integrity import FFMPEG
from vc_eval.spotcheck import Item, Section, extract_frames, miss_tile, thumbnail, write_spotcheck


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
