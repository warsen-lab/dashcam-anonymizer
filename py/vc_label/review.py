"""Human review round trip through X-AnyLabeling (LabelMe JSON), kept outside the frames dir."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from vc_label.contracts import LABEL_CLASSES, Box, FrameRef, to_yolo_lines
from vc_label.rules import Flag

SCENE_FLAGS = ("day", "night", "rain", "tunnel", "city", "highway")
REVIEWED_FLAG = "reviewed"
LABELME_VERSION = "5.4.1"
SOURCE_KEY = "vc_source"


def ensure_outside(frames_dir: Path, path: Path) -> None:
    """Refuses any output inside the frames dir: vc_extract --overwrite deletes that whole dir."""
    frames = frames_dir.resolve()
    target = path.resolve()
    message = f"{path} is inside the frames directory {frames_dir}; write labels elsewhere"
    if target == frames or frames in target.parents:
        raise ValueError(message)
    # resolve() keeps the spelling it is given, and APFS is case-insensitive: data/frames/TRIP/labels
    # is inside data/frames/trip yet compares different. The filesystem knows whether two paths are one directory.
    if frames_dir.exists():
        for candidate in (target, *target.parents):
            if candidate.exists() and os.path.samefile(candidate, frames_dir):
                raise ValueError(message)


def labelme_document(ref: FrameRef, boxes: list[Box], flags: list[Flag], review_dir: Path, width: int, height: int) -> dict:
    notes: dict[Box, list[str]] = defaultdict(list)
    for f in flags:
        notes[f.box].append(f.rule)
    shapes = [
        {
            "label": b.cls,
            "points": [[round(b.x1, 1), round(b.y1, 1)], [round(b.x2, 1), round(b.y2, 1)]],
            "group_id": None,
            "description": " ".join([f"score={b.score:.2f}", f"src={b.src}", *notes.get(b, [])]),
            "shape_type": "rectangle",
            "flags": {},
        }
        for b in boxes
    ]
    return {
        "version": LABELME_VERSION,
        "flags": {REVIEWED_FLAG: False, **{name: False for name in SCENE_FLAGS}},
        "shapes": shapes,
        "imagePath": os.path.relpath(ref.image.resolve(), review_dir.resolve()),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        # X-AnyLabeling rewrites imagePath to the bare file name on save and keeps unknown top-level keys,
        # so this is what ties the file to its video: every video's frames are named NNNNNN.jpg.
        SOURCE_KEY: {"video": ref.video, "frame": ref.frame},
    }


def write_review_queue(path: Path, flags: list[Flag]) -> None:
    """One row per flagged frame, most flags first; `where` lists rule@x1,y1,x2,y2 per flag."""
    per_frame: dict[tuple[str, int], list[Flag]] = defaultdict(list)
    for f in flags:
        per_frame[(f.video, f.frame)].append(f)
    rows = sorted(per_frame.items(), key=lambda item: (-len(item[1]), item[0][1]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "frame", "flags", "rules", "where"])
        for (video, frame), frame_flags in rows:
            where = " ".join(
                f"{f.rule}@{f.box.x1:.0f},{f.box.y1:.0f},{f.box.x2:.0f},{f.box.y2:.0f}" for f in frame_flags
            )
            rules = ";".join(sorted({f.rule for f in frame_flags}))
            writer.writerow([video, frame, len(frame_flags), rules, where])


def read_labelme_boxes(path: Path) -> tuple[list[Box], list[str], bool, int, int]:
    """(boxes, true scene flags, reviewed, width, height).

    `reviewed` is the human's "this frame is done" checkbox and is never part of the scene tags.
    Non-rectangles, unknown labels, empty boxes and unknown true image-level flags are errors.
    """
    return boxes_from_labelme(path, load_labelme(path))


def load_labelme(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: a LabelMe file must be a JSON object, got {type(doc).__name__}")
    return doc


def boxes_from_labelme(path: Path, doc: dict) -> tuple[list[Box], list[str], bool, int, int]:
    """read_labelme_boxes for a document that load_labelme already parsed (`path` only names errors)."""
    width, height = int(doc["imageWidth"]), int(doc["imageHeight"])
    boxes = []
    for i, shape in enumerate(doc.get("shapes", [])):
        label = shape.get("label")
        if label not in LABEL_CLASSES:
            raise ValueError(f"{path}: unknown label {label!r} (expected one of {LABEL_CLASSES})")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"{path}: shape {label!r} is {shape.get('shape_type')!r}, not a rectangle")
        points = shape.get("points") or []
        if len(points) < 2:
            raise ValueError(f"{path}: shape {i} {label!r} has {len(points)} points, need at least 2")
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        box = Box(label, min(xs), min(ys), max(xs), max(ys), src="human").clipped(width, height)
        if box.area <= 0:
            raise ValueError(f"{path}: shape {i} {label!r} is empty after clipping to the image")
        boxes.append(box)
    flags = doc.get("flags") or {}
    if not isinstance(flags, dict):
        raise ValueError(f"{path}: image-level flags must be an object, got {type(flags).__name__}")
    on = [name for name, value in flags.items() if value]
    for name in on:
        if name != REVIEWED_FLAG and name not in SCENE_FLAGS:
            raise ValueError(f"{path}: unknown image flag {name!r} (expected {REVIEWED_FLAG!r} or one of {SCENE_FLAGS})")
    reviewed = flags.get(REVIEWED_FLAG) is True  # the checkbox is a JSON boolean: a string "false" is not a tick
    return boxes, sorted(name for name in on if name in SCENE_FLAGS), reviewed, width, height


MAX_LISTED_FRAMES = 20


def check_source(path: Path, doc: dict, ref: FrameRef) -> None:
    """The JSON must carry the `vc_source` that export-review wrote, naming this frame of this video."""
    if SOURCE_KEY not in doc:
        raise ValueError(f"{path}: no {SOURCE_KEY}; re-export it with vc_label export-review")
    source = doc[SOURCE_KEY]
    if not isinstance(source, dict):
        raise ValueError(
            f"{path}: {SOURCE_KEY} must be an object with video and frame, got {source!r}; "
            f"re-export it with vc_label export-review"
        )
    video, frame = source.get("video"), source.get("frame")
    if video != ref.video or type(frame) is not int or frame != ref.frame:  # type(): True == 1 is not frame 1
        raise ValueError(
            f"{path}: {SOURCE_KEY} records {video}#{frame!r}, but the frames.csv row for this file is "
            f"{ref.video}#{ref.frame}; the file does not belong to that row"
        )


def count_without_source(review_dir: Path) -> tuple[int, int]:
    """(`*.json` files with no `vc_source` key, all `*.json` files) in review_dir.

    These are files exported before `vc_source` existed (export-review keeps existing files), which import-review
    refuses. A file that does not parse, or has a wrong `vc_source`, is not counted: import-review reports it.
    """
    paths = sorted(review_dir.glob("*.json"))
    without = 0
    for path in paths:
        try:
            doc = load_labelme(path)
        except (ValueError, OSError):
            continue
        if SOURCE_KEY not in doc:
            without += 1
    return without, len(paths)


def check_image_path(path: Path, doc: dict, ref: FrameRef) -> None:
    """Secondary to `check_source`: the JSON's `imagePath` must lead to the frame image it is filed under.

    Frame numbers repeat across videos. `imagePath` is resolved against the JSON's directory; if that file
    exists it must be the frame image itself, otherwise (a review dir moved to another machine, or X-AnyLabeling
    having cut it down to a bare file name) the file names must agree.
    """
    image_path = doc.get("imagePath")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"{path}: no imagePath, so it cannot be tied to the frame image {ref.image}")
    spelled = image_path.replace("\\", "/")  # saved by X-AnyLabeling on Windows
    target = path.parent / spelled
    if target.exists():
        if not os.path.samefile(target, ref.image):
            raise ValueError(
                f"{path}: imagePath {image_path!r} is {target}, not the frame image {ref.image}; "
                f"this review dir was exported from another video or frames dir"
            )
    elif PurePosixPath(spelled).name != ref.image.name:
        raise ValueError(
            f"{path}: imagePath {image_path!r} names {PurePosixPath(spelled).name}, not {ref.image.name} "
            f"(the frame image is {ref.image}); this review dir was exported from another video or frames dir"
        )


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:  # header read only
        return img.size


def import_review(review_dir: Path, refs: list[FrameRef], gold_dir: Path, allow_unreviewed: bool = False) -> int:
    """Converts reviewed LabelMe JSONs to gold YOLO labels plus gold_meta.csv; returns frame count.

    Every frame must carry the `reviewed` checkbox unless `allow_unreviewed` (training videos only:
    their pre-labels pass through, recorded as reviewed=false in gold_meta.csv).
    Every JSON must carry the `vc_source` (video, frame) that `labelme_document` wrote: X-AnyLabeling rewrites
    `imagePath` to a bare file name, which every video's frames share, so `imagePath` alone cannot tell videos apart.
    All JSONs are parsed and validated first; only then is the gold dir touched, so a bad file leaves an
    earlier import intact. Label files of frames no longer in the review dir are removed, so a review
    dir that is missing or holds no JSON is refused: it would otherwise empty the gold dir.
    """
    if (gold_dir / "processed.txt").exists():
        raise ValueError(f"{gold_dir} is a labels dir (it has processed.txt): fuse would overwrite its yolo/; use another --gold")
    if not review_dir.is_dir():
        raise ValueError(f"--review {review_dir} is not a directory")
    json_files = sorted(review_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"--review {review_dir} has no *.json files to import")
    by_name = {f"{r.frame:06d}": r for r in refs}
    parsed = []
    for path in json_files:
        ref = by_name.get(path.stem)
        if ref is None:
            raise ValueError(f"{path}: no frame {path.stem} in frames.csv")
        doc = load_labelme(path)
        check_source(path, doc, ref)
        check_image_path(path, doc, ref)
        boxes, tags, reviewed, width, height = boxes_from_labelme(path, doc)
        actual = image_size(ref.image)
        if (width, height) != actual:
            raise ValueError(f"{path}: imageWidth x imageHeight is {width}x{height} but {ref.image} is {actual[0]}x{actual[1]}")
        parsed.append((ref, boxes, tags, reviewed, width, height))
    pending = [ref.frame for ref, _, _, reviewed, _, _ in parsed if not reviewed]
    if pending and not allow_unreviewed:
        listed = ", ".join(str(n) for n in pending[:MAX_LISTED_FRAMES])
        more = f" (+{len(pending) - MAX_LISTED_FRAMES} more)" if len(pending) > MAX_LISTED_FRAMES else ""
        raise ValueError(
            f"{len(pending)} of {len(parsed)} frames are not marked reviewed: {listed}{more}; "
            f"tick 'reviewed' in X-AnyLabeling or pass --allow-unreviewed (training videos only)"
        )

    yolo_dir = gold_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref, boxes, tags, reviewed, width, height in parsed:
        yolo_text = "".join(f"{line}\n" for line in to_yolo_lines(boxes, width, height))
        (yolo_dir / f"{ref.frame:06d}.txt").write_text(yolo_text, encoding="utf-8")
        rows.append([ref.image.name, ref.video, ref.frame, "true" if reviewed else "false", ";".join(tags)])
    imported = {f"{ref.frame:06d}.txt" for ref, *_ in parsed}
    for stale in yolo_dir.glob("*.txt"):
        if stale.name != "classes.txt" and stale.name not in imported:
            stale.unlink()
    (yolo_dir / "classes.txt").write_text("".join(f"{c}\n" for c in LABEL_CLASSES), encoding="utf-8")
    meta = gold_dir / "gold_meta.csv"
    tmp = meta.with_name(meta.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "video", "frame", "reviewed", "scene_tags"])
        writer.writerows(rows)
    os.replace(tmp, meta)
    return len(rows)
