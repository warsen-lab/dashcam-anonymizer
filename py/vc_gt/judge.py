"""The LLM's two questions (spec 2026-10-22 §2 step 4): A, per object, real or not and readable at its best; B, per
frame vc_redact left a real object unmasked, readable there or not. Verdicts go to objects.jsonl and frames.jsonl in
vc_hardneg's judgment format, so people's verdicts (judge "human") override them. Only objects that reach the
must-mask size and that some detector scored judge_min_score on are asked about (spec §8): the rest are too small to
read or almost surely not plates and faces; the H3 audit counts any real one among them as missed by all."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vc_eval.metrics import must_mask
from vc_gt.config import GtConfig
from vc_gt.crops import (CROPS_INDEX, Crop, append_crop_index, read_crop_index, unmasked_frames, write_frame_crop,
                         write_object_crops)
from vc_gt.tracks import Track
from vc_hardneg.judges import CLS_ZH, CommandJudge, _fill
from vc_hardneg.judgments import Judgment, append_judgments, read_judgments, resolve
from vc_label.contracts import Box
from vc_label.review import image_size

OBJECT_VERDICTS = ("real_readable", "real_unreadable", "not_real", "unsure")
FRAME_VERDICTS = ("readable", "unreadable", "unsure")
REAL = ("real_readable", "real_unreadable")
OBJECTS = "objects.jsonl"
FRAMES = "frames.jsonl"
Runner = Callable[[Path, str, str], "str | None"]  # (grid image, prompt, class) -> the command's stdout


def object_id(track_id: int) -> str:
    return f"o{track_id}"


def frame_id(track_id: int, frame: int) -> str:
    return f"o{track_id}@{frame}"


def ever_must_mask(track: Track, frame_height: int) -> bool:
    """Whether the track's box reaches the must-mask size (vc_eval.metrics) on at least one frame."""
    return any(must_mask(b, frame_height) for b in track.boxes)


@dataclass(frozen=True)
class Item:
    id: str
    cls: str
    crops: tuple[Crop, ...]


def parse_answers(out: str | None, allowed: tuple[str, ...]) -> dict[int, str]:
    """{n: verdict} from the last JSON array of objects in the output; rows with another verdict are left out."""
    for text in reversed(re.findall(r"\[[^\[\]]*\]", out or "")):
        try:
            rows = json.loads(text)
        except ValueError:
            continue
        if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
            return {r["n"]: r["verdict"] for r in rows if isinstance(r.get("n"), int) and r.get("verdict") in allowed}
    return {}


def grid_image(items: list[Item], tile: int, out: Path) -> Path:
    """Every item in one cell, its crops side by side, each fitted into tile x tile with the target in red and the
    item's number (1, 2, ...) in yellow at the cell's top left; two cells per row when items have several crops,
    else four."""
    per = max(len(it.crops) for it in items)
    cols = min(len(items), 2 if per > 1 else 4)
    rows = (len(items) + cols - 1) // cols
    cell = per * tile
    grid = np.full((rows * tile, cols * cell, 3), 40, np.uint8)
    for k, it in enumerate(items):
        r, q = divmod(k, cols)
        for j, crop in enumerate(it.crops):
            img = cv2.imread(str(crop.path))
            if img is None:
                raise FileNotFoundError(f"{crop.path} cannot be read")
            s = (tile - 4) / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            x1, y1, x2, y2 = crop.box
            cv2.rectangle(img, (round(x1 * s), round(y1 * s)), (round(x2 * s), round(y2 * s)), (0, 0, 255), 2)
            oy, ox = r * tile + 2, q * cell + j * tile + 2
            grid[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
        for color, width in (((0, 0, 0), 5), ((0, 255, 255), 2)):
            cv2.putText(grid, str(k + 1), (q * cell + 8, r * tile + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, width)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), grid):
        raise ValueError(f"cannot write {out}")
    return out


def ask(items: list[Item], prompt: str, allowed: tuple[str, ...], runner: Runner, judge: str, judgments: Path,
        work_dir: Path, batch: int, tile: int, parallel: int = 1) -> Counter:
    """Asks about every item without a judgment in `judgments` yet: batches of up to `batch` items of one class on a
    numbered grid (written to work_dir), `parallel` calls at once. Each batch's verdicts are appended as it comes, so an
    interrupted run keeps them; an item the answer leaves out is "unsure". Returns counts per verdict."""
    known = resolve(read_judgments(judgments))
    todo = [it for it in items if it.id not in known]
    batches = [same[i : i + batch] for cls in sorted({it.cls for it in todo})
               for same in [[it for it in todo if it.cls == cls]] for i in range(0, len(same), batch)]

    def one(b: list[Item]) -> list[Judgment]:
        cls = b[0].cls
        image = grid_image(b, tile, Path(work_dir) / f"batch_{b[0].id}.jpg")  # JPEG: a seventh of the PNG upload
        out = runner(image, _fill(prompt, {"{count}": str(len(b)), "{cls_zh}": CLS_ZH[cls], "{cls}": cls}), cls)
        answers = parse_answers(out, allowed)
        detail = f"grid of {len(b)}" if answers else ("no answer" if out is None else "no verdicts in the output")
        return [Judgment(it.id, answers.get(k + 1, "unsure"), judge, None, detail) for k, it in enumerate(b)]

    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for got in pool.map(one, batches):
            append_judgments(judgments, got, verdicts=allowed)
            counts.update(j.verdict for j in got)
    return counts


def _cached(index: dict[str, tuple[Crop, ...]], gt_dir: Path, item_id: str, make) -> tuple[Crop, ...]:
    if item_id in index:
        return index[item_id]
    crops = tuple(make())
    append_crop_index(gt_dir / CROPS_INDEX, item_id, crops)
    return crops


def judge_objects(gt_dir: Path, tracks: list[Track], images: dict[int, Path], cfg: GtConfig, runner: Runner,
                  judge: str) -> Counter:
    """Question A about every track not judged yet that reaches the must-mask size somewhere and scores
    cfg.judge_min_score on some frame."""
    gt_dir = Path(gt_dir)
    if not tracks:
        return Counter()
    frame_height = image_size(next(iter(images.values())))[1]
    known = resolve(read_judgments(gt_dir / OBJECTS))
    index = read_crop_index(gt_dir / CROPS_INDEX)
    items = [Item(object_id(t.id), t.cls,
                  _cached(index, gt_dir, object_id(t.id),
                          lambda t=t: write_object_crops(t, images, gt_dir / "crops" / "objects", cfg)))
             for t in tracks if object_id(t.id) not in known and ever_must_mask(t, frame_height)
             and max(b.score for b in t.boxes) >= cfg.judge_min_score]
    return ask(items, cfg.llm.object_prompt, OBJECT_VERDICTS, runner, judge, gt_dir / OBJECTS, gt_dir / "llm",
               cfg.llm.batch, cfg.llm.tile, cfg.llm.parallel)


def judge_frames(gt_dir: Path, tracks: list[Track], images: dict[int, Path], masks_by_frame: dict[int, list[Box]],
                 cfg: GtConfig, runner: Runner, judge: str) -> Counter:
    """Question B about every unmasked frame of every track judged real (by the latest verdict) not judged yet."""
    gt_dir = Path(gt_dir)
    objects = resolve(read_judgments(gt_dir / OBJECTS))
    known = resolve(read_judgments(gt_dir / FRAMES))
    index = read_crop_index(gt_dir / CROPS_INDEX)
    items = []
    for t in tracks:
        j = objects.get(object_id(t.id))
        if j is None or j.verdict not in REAL:
            continue
        for f in unmasked_frames(t, masks_by_frame):
            fid = frame_id(t.id, f)
            if fid not in known:
                crops = _cached(index, gt_dir, fid,
                                lambda t=t, f=f: [write_frame_crop(t, f, images, gt_dir / "crops" / "frames", cfg)])
                items.append(Item(fid, t.cls, crops))
    return ask(items, cfg.llm.frame_prompt, FRAME_VERDICTS, runner, judge, gt_dir / FRAMES, gt_dir / "llm",
               cfg.llm.batch, cfg.llm.tile, cfg.llm.parallel)


def make_runner(cfg: GtConfig) -> tuple[Runner, str]:
    """(the runner, its judge name "llm:<name>"); raises PermissionError without VC_HARDNEG_ALLOW_LLM=1."""
    command = CommandJudge(cfg.llm.command_config())
    return (lambda image, prompt, cls: command.run(image, prompt, cls)[0]), command.name
