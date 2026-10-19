"""python -m vc_eval coverage | temporal | integrity | viewing | run (proxy-validation spec §6, spec 2026-10-19 §5)."""

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
    c.add_argument("--json", type=Path)
    t = sub.add_parser("temporal", help="sparse masks against a --detect-every-frame run")
    t.add_argument("--dense", type=Path, required=True, help="detections.jsonl of the dense run")
    t.add_argument("--sparse", type=Path, required=True, help="masks.jsonl of the sparse run")
    t.add_argument("--frame-height", type=int, required=True)
    t.add_argument("--json", type=Path)
    i = sub.add_parser("integrity", help="frame count, duration, audio and A/V sync of the output")
    i.add_argument("--source", type=Path, required=True)
    i.add_argument("--redacted", type=Path, required=True)
    i.add_argument("--json", type=Path)
    v = sub.add_parser("viewing", help="masked area, flicker and boxes per frame of a masks.jsonl (every frame)")
    v.add_argument("--masks", type=Path, required=True)
    v.add_argument("--width", type=int, required=True, help="frame width of the video in px (display size if it is rotated)")
    v.add_argument("--height", type=int, required=True, help="frame height of the video in px (display size if it is rotated)")
    v.add_argument("--json", type=Path)
    r = sub.add_parser("run", help="detection-rate sweep, summary and spot-check page")
    r.add_argument("--stems", required=True, help="comma-separated held-out stems")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument(
        "--face-model", type=Path, default=None,
        help="YuNet face model dir (python -m vc_train yunet) passed to every vc_redact run as --face-model",
    )
    r.add_argument("--out", type=Path, required=True)
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
    from vc_eval.coverage import evaluate_coverage
    from vc_eval.metrics import HEATMAP_MIN_TARGETS
    from vc_eval.run import coverage_json

    r = evaluate_coverage(args.frames, args.labels, args.masks)
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
    from vc_eval.run import RunConfig, run_all

    cfg = RunConfig(
        stems=[s.strip() for s in args.stems.split(",") if s.strip()], model=args.model, fps_values=_fps_values(args.fps),
        out=args.out, vc_redact=args.vc_redact, vc_detect=args.vc_detect, config=args.config, overwrite=args.overwrite,
        redacted_root=args.redacted_root, face_model=args.face_model,
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        handlers = {"coverage": _coverage, "temporal": _temporal, "integrity": _integrity, "viewing": _viewing, "run": _run}
        return handlers[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"vc_eval: {e}", file=sys.stderr)
        return 2
