"""python -m vc_eval coverage | temporal | integrity | viewing | run | falsepos | newmisses | objects (proxy-validation
spec §6, spec 2026-10-19 §5, spec 2026-10-21 §6, spec 2026-10-22 §4)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from vc_label.contracts import LABEL_CLASSES


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_eval")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("coverage", help="final coverage of masks.jsonl against teacher labels")
    c.add_argument("--frames", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True)
    c.add_argument("--masks", type=Path, required=True)
    c.add_argument("--recall-target", default="", help="P1 recall per class, e.g. face=0.95,plate=0.98 (default 0.99 each)")
    c.add_argument("--json", type=Path)
    t = sub.add_parser("temporal", help="sparse masks against a --detect-every-frame run")
    t.add_argument("--dense", type=Path, required=True, help="detections.jsonl of the dense run")
    t.add_argument("--sparse", type=Path, required=True, help="masks.jsonl of the sparse run")
    t.add_argument("--frame-height", type=int, required=True)
    t.add_argument("--json", type=Path)
    i = sub.add_parser("integrity", help="frame count, duration, audio, A/V sync, H.264 and its level of the output")
    i.add_argument("--source", type=Path, required=True)
    i.add_argument("--redacted", type=Path, required=True)
    i.add_argument("--json", type=Path)
    v = sub.add_parser("viewing", help="masked area, flicker and boxes per frame of a masks.jsonl (every frame)")
    v.add_argument("--masks", type=Path, required=True)
    v.add_argument("--width", type=int, required=True, help="coded frame width in px, as ffprobe reports it for the video stream (not swapped for rotation)")
    v.add_argument("--height", type=int, required=True, help="coded frame height in px, as ffprobe reports it for the video stream (not swapped for rotation)")
    v.add_argument("--json", type=Path)
    f = sub.add_parser("falsepos", help="masks no teacher box explains, by class and source (labelled frames)")
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--detections", type=Path, required=True)
    f.add_argument("--json", type=Path)
    n = sub.add_parser("newmisses", help="reference boxes the candidate masks miss but the baseline's covered")
    n.add_argument("--frames", type=Path, required=True)
    n.add_argument("--labels", type=Path, required=True)
    n.add_argument("--baseline", type=Path, required=True, help="masks.jsonl")
    n.add_argument("--candidate", type=Path, required=True, help="masks.jsonl")
    n.add_argument("--out", type=Path, required=True, help="crops of the new misses go here")
    n.add_argument("--recall-target", default="")
    o = sub.add_parser("objects", help="per-object leaks and per-frame recall against a vc_gt answer key")
    o.add_argument("--gt", type=Path, required=True, help="data/gt/<stem>")
    o.add_argument("--frames", type=Path, required=True, help="the 10 fps frames the answer key was built on")
    o.add_argument("--masks", type=Path, required=True, help="masks.jsonl of the vc_redact run under test")
    o.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="detection-rate sweep, summary and spot-check page")
    r.add_argument("--stems", required=True, help="comma-separated held-out stems")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument(
        "--face-model", type=Path, default=None,
        help="YuNet face model dir (python -m vc_train yunet) passed to every vc_redact run as --face-model",
    )
    r.add_argument(
        "--verifier-model", type=Path, default=None,
        help="vc_train verifier model dir, passed to the sparse vc_redact runs (not the dense reference) as --verifier-model",
    )
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--recall-target", default="", help="P1 recall per class, e.g. face=0.95,plate=0.98 (default 0.99 each)")
    r.add_argument("--fps", default="5,10,15,30", help="comma-separated detection rates, each from 5 to 60")
    r.add_argument("--vc-redact", type=Path, default=Path("build/cpp-release/vc_redact"))
    r.add_argument("--vc-detect", type=Path, default=Path("build/cpp-release/vc_detect"))
    r.add_argument("--config", type=Path, default=Path("configs/redact.json"))
    r.add_argument(
        "--redacted-root", type=Path, default=Path("data/redacted"),
        help="where the redacted runs go; give each --config its own, or a later run overwrites the videos an earlier report links to",
    )
    r.add_argument("--overwrite", action="store_true", help="clear a non-empty --out first (it is refused otherwise)")
    return p


def _rate_text(rate) -> str:
    return f"{rate.hits}/{rate.total}" + (f" ({rate.value:.2%})" if rate.total else "")


def _write_json(path: Path | None, data: dict) -> None:
    if path is not None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fps_values(text: str) -> list[float]:
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise ValueError(f"--fps must be comma-separated numbers, got {text!r}") from None


def _coverage(args) -> int:
    from vc_eval.coverage import evaluate_coverage, parse_recall_targets
    from vc_eval.metrics import HEATMAP_MIN_TARGETS
    from vc_eval.records import EXCLUDED_REFS
    from vc_eval.run import coverage_json

    r = evaluate_coverage(args.frames, args.labels, args.masks, parse_recall_targets(args.recall_target))
    weakest = r.weakest_cell()
    cell = (
        f"P2 n/a (no heatmap cell has {HEATMAP_MIN_TARGETS} targets, so P2 is vacuously satisfied)"
        if weakest is None
        else f"weakest {weakest[0]} r{weakest[1]}c{weakest[2]} {_rate_text(weakest[3])}"
    )
    print(f"coverage {r.video}: " + ", ".join(f"{c} {_rate_text(r.classes[c].rate)}" for c in LABEL_CLASSES)
          + f"; {len(r.misses)} misses; {cell}"
          + "; low-confidence (report only): " + ", ".join(f"{c} {_rate_text(r.classes[c].uncertain)}" for c in LABEL_CLASSES)
          + f"; {'PASS' if r.passed() else 'FAIL'}")
    if (args.labels / EXCLUDED_REFS).exists():
        covered = sum(e.covered for e in r.excluded)
        print(f"excluded refs: {len(r.excluded)} excluded ({covered} had been covered), {len(r.unmatched_exclusions)} unmatched rows")
    _write_json(args.json, coverage_json(r))
    return 0 if r.passed() else 1


def _temporal(args) -> int:
    from vc_eval.run import rate_json
    from vc_eval.temporal import evaluate_temporal

    r = evaluate_temporal(args.dense, args.sparse, args.frame_height)
    print(f"temporal {r.video}: " + ", ".join(f"{c} {_rate_text(r.classes[c])}" for c in LABEL_CLASSES))
    _write_json(args.json, {c: rate_json(r.classes[c]) for c in LABEL_CLASSES})
    return 0


def _integrity(args) -> int:
    from vc_eval.integrity import check_integrity

    r = check_integrity(args.source, args.redacted)
    print("integrity: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in r.checks.items()))
    _write_json(args.json, {"checks": r.checks, "details": r.details, "passed": r.passed})
    return 0 if r.passed else 1


def _viewing(args) -> int:
    from vc_eval.viewing import evaluate_viewing, summary_line, viewing_json

    r = evaluate_viewing(args.masks, args.width, args.height, require_inside=True)
    print(summary_line(r))
    _write_json(args.json, viewing_json(r))
    return 0


def _run(args) -> int:
    from vc_eval.coverage import parse_recall_targets
    from vc_eval.run import RunConfig, run_all

    cfg = RunConfig(
        stems=[s.strip() for s in args.stems.split(",") if s.strip()], model=args.model, fps_values=_fps_values(args.fps),
        out=args.out, vc_redact=args.vc_redact, vc_detect=args.vc_detect, config=args.config, overwrite=args.overwrite,
        redacted_root=args.redacted_root, face_model=args.face_model, verifier_model=args.verifier_model,
        recall_targets=parse_recall_targets(args.recall_target),
    )
    summary = run_all(cfg)
    chosen = summary["chosen_fps"]
    print(
        f"run: candidate {chosen:g} fps (P1-P3 passed; P4: review index.html)"
        if chosen is not None
        else "run: no detection rate passed P1-P3"
    )
    print(f"run: {args.out / 'index.html'}")
    return 0 if chosen is not None else 1


def _falsepos(args) -> int:
    from vc_eval.falsepos import count_false_positives

    r = count_false_positives(args.detections, args.labels)
    for cls, c in r.items():
        share = c["unreferenced"] / c["masked"] if c["masked"] else 0.0
        print(f"{cls}: {c['unreferenced']}/{c['masked']} masked boxes match no teacher box ({share:.1%}); by source {c['by_src']}")
    _write_json(args.json, r)
    return 0


def _newmisses(args) -> int:
    from vc_eval.coverage import parse_recall_targets
    from vc_eval.newmisses import new_misses, write_crops

    misses = new_misses(args.frames, args.labels, args.baseline, args.candidate, parse_recall_targets(args.recall_target))
    write_crops(misses, args.frames, args.out)
    print(f"newmisses: {len(misses)} reference boxes newly missed; crops in {args.out}")
    return 0 if not misses else 1


def _objects(args) -> int:
    from vc_eval.objects import evaluate_objects, model_error_rate, write_report
    from vc_eval.records import read_masks
    from vc_gt.audit import AUDIT, H2_SAMPLE, read_audit
    from vc_gt.candidates import TRACKS
    from vc_gt.judge import FRAMES, OBJECTS
    from vc_gt.tracks import read_tracks
    from vc_hardneg.judgments import read_judgments, resolve
    from vc_label.contracts import read_frames_csv
    from vc_label.review import image_size

    tracks = read_tracks(args.gt / TRACKS)
    refs = read_frames_csv(args.frames)
    images = {r.frame: r.image for r in refs}
    _, masks = read_masks(args.masks)
    objects_j, frames_j = read_judgments(args.gt / OBJECTS), read_judgments(args.gt / FRAMES)
    objects = {i: j.verdict for i, j in resolve(objects_j).items()}
    frames = {i: j.verdict for i, j in resolve(frames_j).items()}
    audit = read_audit(args.gt / AUDIT)[1] if (args.gt / AUDIT).exists() else None
    results = evaluate_objects(tracks, objects, frames, masks, image_size(refs[0].image)[1], audit)
    sample = json.loads((args.gt / H2_SAMPLE).read_text()) if (args.gt / H2_SAMPLE).exists() else {}
    errors = {"问题 A（真不真、认不认得出）": model_error_rate(objects_j, set(sample.get("objects", []))),
              "问题 B（未打码帧认不认得出）": model_error_rate(frames_j, set(sample.get("frames", [])))}
    page = write_report(results, errors, args.out, tracks, images, masks)
    for cls, r in results.items():
        print(f"{cls}: {r.readable} readable real objects, {len(r.leaks)} leaked, frame recall "
              f"{r.frame_recall.hits}/{r.frame_recall.total}, passed {r.passed}")
    print(f"report: {page}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        handlers = {"coverage": _coverage, "temporal": _temporal, "integrity": _integrity, "viewing": _viewing, "run": _run,
                    "falsepos": _falsepos, "newmisses": _newmisses, "objects": _objects}
        return handlers[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"vc_eval: {e}", file=sys.stderr)
        return 2
