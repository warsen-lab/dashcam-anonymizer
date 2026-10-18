"""Local static spot-check page for the user (proxy-validation spec §6.5): no external resources."""

from __future__ import annotations

import html
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vc_label.contracts import Box
from vc_eval.integrity import FFMPEG

BATCH = 50
GREEN = (0, 200, 0)


def extract_frames(video: Path, frames: list[int], out_dir: Path) -> dict[int, Path]:
    """Decodes 0-based frame numbers of `video` into out_dir/<frame:06d>.jpg.

    ffmpeg runs inside out_dir with a relative output pattern: a directory name holding a `%` would otherwise be read
    as part of the pattern. It never reads stdin, and stops once a batch is complete instead of decoding to the end.
    """
    wanted = sorted(set(frames))
    if not wanted:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    source = os.path.abspath(video)  # the working directory changes below
    got: dict[int, Path] = {}
    for start in range(0, len(wanted), BATCH):
        batch = wanted[start : start + BATCH]
        expr = "+".join(f"eq(n\\,{n})" for n in batch)
        for leftover in out_dir.glob("batch_*.jpg"):  # from an interrupted run: must not be mistaken for output
            leftover.unlink()
        subprocess.run(
            [FFMPEG, "-nostdin", "-v", "error", "-y", "-i", source, "-vf", f"select={expr}", "-fps_mode", "passthrough",
             "-frames:v", str(len(batch)), "-q:v", "3", "batch_%05d.jpg"],
            check=True, cwd=out_dir, stdin=subprocess.DEVNULL,
        )
        for i, frame in enumerate(batch, 1):
            produced = out_dir / f"batch_{i:05d}.jpg"
            if not produced.exists():
                raise ValueError(f"{video} has no frame {frame}")
            target = out_dir / f"{frame:06d}.jpg"
            produced.replace(target)
            got[frame] = target
    return got


def _crop(image: np.ndarray, box: Box, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    side = max(96.0, 3 * max(box.width, box.height))
    cx, cy = box.center
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError(f"{path}: the box ({box.x1:g}, {box.y1:g})-({box.x2:g}, {box.y2:g}) is outside the {w}x{h} image")
    return crop, (x1, y1)


def _to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def _write(path: Path, image: np.ndarray, params: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params):  # imwrite reports a full disk or an unwritable path by returning False
        raise ValueError(f"cannot write {path}")


def miss_tile(original: Path, redacted: Path | None, box: Box, out: Path, height: int = 160) -> Path:
    """Crop around `box` from the original (box drawn in green) and, when given, from the redacted frame beside it.

    The two frames must have the same size and the box must lie at least partly inside them; otherwise ValueError.
    """
    src = _read(original)
    red = None if redacted is None else _read(redacted)
    if red is not None and red.shape[:2] != src.shape[:2]:
        raise ValueError(
            f"{original} is {src.shape[1]}x{src.shape[0]} but {redacted} is {red.shape[1]}x{red.shape[0]}; they are not the same frame size"
        )
    crop, (ox, oy) = _crop(src, box, original)
    cv2.rectangle(crop, (int(box.x1 - ox), int(box.y1 - oy)), (int(box.x2 - ox), int(box.y2 - oy)), GREEN, 1)
    parts = [_to_height(crop, height)]
    if red is not None:
        parts.append(_to_height(_crop(red, box, redacted)[0], height))
    _write(out, cv2.hconcat(parts), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def thumbnail(image: Path, out: Path, width: int = 640) -> Path:
    src = _read(image)
    scale = width / src.shape[1]
    _write(out, cv2.resize(src, (width, round(src.shape[0] * scale)), interpolation=cv2.INTER_AREA), [])
    return out


@dataclass
class Item:
    image: str  # relative to the page
    caption: str


@dataclass
class Section:
    title: str
    note: str
    items: list[Item]
    links: list[tuple[str, str]] = field(default_factory=list)


_STYLE = """
body{margin:0;padding:16px;font:14px/1.5 -apple-system,"PingFang SC",sans-serif;background:#f6f5f2;color:#222}
h1{font-size:22px;margin:0 0 8px}h2{font-size:17px;margin:28px 0 6px;border-bottom:1px solid #ccc}
.note{color:#555;max-width:72em}.grid{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;background:#fff;border:1px solid #ddd;padding:6px;max-width:100%}
figure img{display:block;max-width:100%;height:auto}figcaption{font-size:12px;color:#444;margin-top:4px}
@media (prefers-color-scheme:dark){body{background:#1d1d1f;color:#eee}figure{background:#2a2a2c;border-color:#444}
.note,figcaption{color:#bbb}h2{border-color:#555}a{color:#8ab4f8}}
"""


def write_spotcheck(out_dir: Path, title: str, intro: str, sections: list[Section]) -> Path:
    e = html.escape
    parts = [
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>{e(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{e(title)}</h1><p class=\"note\">{e(intro)}</p>",
    ]
    for s in sections:
        parts.append(f"<h2>{e(s.title)}</h2><p class=\"note\">{e(s.note)}</p>")
        if s.links:
            parts.append("<p>" + " · ".join(f"<a href=\"{e(href)}\">{e(text)}</a>" for text, href in s.links) + "</p>")
        parts.append("<div class=\"grid\">")
        for item in s.items:
            parts.append(
                f"<figure><img src=\"{e(item.image)}\" loading=\"lazy\" alt=\"{e(item.caption)}\">"
                f"<figcaption>{e(item.caption)}</figcaption></figure>"
            )
        parts.append("</div>")
    parts.append("</body></html>\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "index.html"
    page.write_text("".join(parts), encoding="utf-8")
    return page
