"""python -m vc_gt candidates | judge | review | annotate | import-human (spec 2026-10-22 §2): build the answer key of
one held-out segment in data/gt/<stem>/; `python -m vc_eval objects` measures a vc_redact run against it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from vc_gt.audit import (AUDIT, AUDIT_FRAMES, FRAME_CHOICES, H2_SAMPLE, OBJECT_CHOICES, PageItem, audit_sample,
                         h1_frame_ids, h1_object_ids, h2_sample, import_annotations, import_choices,
                         write_annotate_page, write_audit, write_choice_page)
from vc_gt.candidates import TRACKS, read_persons
from vc_gt.config import DEFAULT_CONFIG, load_config
from vc_gt.crops import CROPS_INDEX, leak_frames, read_crop_index
from vc_gt.judge import (FRAME_VERDICTS, FRAMES, OBJECT_VERDICTS, OBJECTS, REAL, frame_id, judge_frames,
                         judge_objects, make_runner, object_id)
from vc_gt.tracks import Track, read_tracks
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments, resolve
from vc_label.contracts import read_frames_csv

REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_gt")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates", help="every plausible plate and face, linked into tracks (slow, resumable)")
    c.add_argument("--frames", type=Path, required=True, help="10 fps frames of one segment (vc_extract)")
    c.add_argument("--out", type=Path, required=True, help="data/gt/<stem>")
    c.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    c.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    c.add_argument("--vc-detect", type=Path, default=REPO / "build/cpp-release/vc_detect")
    c.add_argument("--student-model", type=Path, default=REPO / "data/models/r3")
    c.add_argument("--limit", type=int, help="process at most N new frames in this run")
    j = sub.add_parser("judge", help="ask the LLM questions A and B (needs VC_HARDNEG_ALLOW_LLM=1)")
    j.add_argument("--gt", type=Path, required=True)
    j.add_argument("--frames", type=Path, required=True)
    j.add_argument("--masks", type=Path, required=True, help="masks.jsonl of the vc_redact run under test")
    j.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    j.add_argument("--stage", choices=["objects", "frames", "both"], default="both")
    r = sub.add_parser("review", help="local pages for people: h1 (leaks and unsure) or h2 (sample of model verdicts)")
    r.add_argument("--gt", type=Path, required=True)
    r.add_argument("--frames", type=Path, required=True)
    r.add_argument("--masks", type=Path, required=True)
    r.add_argument("--set", choices=["h1", "h2"], required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = sub.add_parser("annotate", help="H3: a page to box every readable plate (face) on sampled whole frames")
    a.add_argument("--gt", type=Path, required=True)
    a.add_argument("--frames", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    i = sub.add_parser("import-human", help="append a page's export as people's verdicts (or the audit boxes)")
    i.add_argument("--gt", type=Path, required=True)
    i.add_argument("--file", type=Path, required=True)
    i.add_argument("--kind", choices=["objects", "frames", "audit"], required=True)
    return p


def _candidates(args) -> int:
    from vc_eval.consistency import run_vc_detect
    from vc_gt.candidates import TeacherFinder, build_candidates, relink

    cfg = load_config(args.config)
    refs = read_frames_csv(args.frames)
    if refs and set(read_persons(args.out)) >= {r.frame for r in refs}:  # every frame found: link again only
        print(f"relinked: {relink(args.frames, args.out, cfg)} tracks")
        return 0
    student = run_vc_detect(args.vc_detect, args.student_model, refs, args.out / "student", score=cfg.student_score)
    n = build_candidates(args.frames, args.out, TeacherFinder(cfg, args.device), student, cfg, args.limit)
    done = (args.out / TRACKS).exists()
    print(f"candidates: {n} frames processed in this run; tracks {'written' if done else 'not yet (frames left)'}")
    return 0


def _judge(args) -> int:
    from vc_eval.records import read_masks

    cfg = load_config(args.config)
    runner, name = make_runner(cfg)
    tracks = read_tracks(args.gt / TRACKS)
    images = {r.frame: r.image for r in read_frames_csv(args.frames)}
    if args.stage in ("objects", "both"):
        print(f"question A: {dict(judge_objects(args.gt, tracks, images, cfg, runner, name))}")
    if args.stage in ("frames", "both"):
        _, masks = read_masks(args.masks)
        print(f"question B: {dict(judge_frames(args.gt, tracks, images, masks, cfg, runner, name))}")
    return 0


def _unmasked_ids(gt: Path, masks_path: Path, tracks: list[Track], frame_height: int) -> set[str]:
    """Frame ids of every frame the masks leave a real object (by the latest verdict) uncovered at must-mask size."""
    from vc_eval.records import read_masks

    _, masks = read_masks(masks_path)
    objects = resolve(read_judgments(gt / OBJECTS))
    return {frame_id(t.id, f) for t in tracks
            if (j := objects.get(object_id(t.id))) is not None and j.verdict in REAL
            for f in leak_frames(t, masks, frame_height)}


def _model_ids(judgments: list[Judgment]) -> list[str]:
    return [i for i, j in resolve(judgments, include_human=False).items() if j.judge != HUMAN]


def _review(args) -> int:
    cfg = load_config(args.config)
    tracks = read_tracks(args.gt / TRACKS)
    cls_of = {object_id(t.id): t.cls for t in tracks}
    objects = resolve(read_judgments(args.gt / OBJECTS))
    frames = resolve(read_judgments(args.gt / FRAMES))
    index = read_crop_index(args.gt / CROPS_INDEX)
    if args.set == "h1":
        object_ids = h1_object_ids(objects)
        from vc_label.review import image_size

        height = image_size(read_frames_csv(args.frames)[0].image)[1]
        frame_ids = h1_frame_ids(frames, _unmasked_ids(args.gt, args.masks, tracks, height))
    else:
        path = args.gt / H2_SAMPLE
        if not path.exists():
            path.write_text(json.dumps({
                "objects": h2_sample(_model_ids(read_judgments(args.gt / OBJECTS)), cfg.audit.objects, cfg.audit.seed),
                "frames": h2_sample(_model_ids(read_judgments(args.gt / FRAMES)), cfg.audit.frames, cfg.audit.seed)}))
        sample = json.loads(path.read_text())
        object_ids, frame_ids = sample["objects"], sample["frames"]
    for kind, ids, choices, judged in (("objects", object_ids, OBJECT_CHOICES, objects),
                                       ("frames", frame_ids, FRAME_CHOICES, frames)):
        items = [PageItem(i, index[i], f"{cls_of[i.split('@')[0]]} 模型：{judged[i].verdict}" if i in judged else "")
                 for i in ids if i in index]
        page = write_choice_page(items, args.out / kind, f"{args.set.upper()} {kind} {args.gt.name}", choices)
        print(f"{kind}: {len(items)} to judge, {page}")
    return 0


def _annotate(args) -> int:
    cfg = load_config(args.config)
    path = args.gt / AUDIT_FRAMES
    if not path.exists():
        persons = read_persons(args.gt)
        path.write_text(json.dumps(audit_sample(sorted(persons), persons, cfg.audit.frames_per_class, cfg.audit.seed)))
    sample = json.loads(path.read_text())
    images = {r.frame: r.image for r in read_frames_csv(args.frames)}
    frames = [(f"{cls}:{f}", images[f], cls) for cls in ("plate", "face") for f in sample[cls]]
    page = write_annotate_page(frames, args.out, f"H3 {args.gt.name}")
    print(f"annotate: {len(frames)} frames, {page}")
    return 0


def _import(args) -> int:
    if args.kind == "audit":
        sample = json.loads((args.gt / AUDIT_FRAMES).read_text())
        keys = {f"{cls}:{f}": (cls, f) for cls in sample for f in sample[cls]}
        rows = import_annotations(args.file, keys)
        write_audit(args.gt / AUDIT, rows)
        print(f"audit: {len(rows)} finished frames, {sum(len(r['boxes']) for r in rows)} boxes")
        return 0
    tracks = read_tracks(args.gt / TRACKS)
    if args.kind == "objects":
        known = {object_id(t.id) for t in tracks}
        allowed, path, verdicts = tuple(v for v, _ in OBJECT_CHOICES), args.gt / OBJECTS, OBJECT_VERDICTS
    else:
        known = {frame_id(t.id, f) for t in tracks for f in t.frames}
        allowed, path, verdicts = tuple(v for v, _ in FRAME_CHOICES), args.gt / FRAMES, FRAME_VERDICTS
    items = import_choices(args.file, known, allowed)
    append_judgments(path, items, verdicts=verdicts)
    print(f"import-human: {len(items)} {args.kind} verdicts")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"candidates": _candidates, "judge": _judge, "review": _review, "annotate": _annotate,
                "import-human": _import}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:  # OSError covers PermissionError (no LLM switch)
        print(f"vc_gt: {e}", file=sys.stderr)
        return 2
