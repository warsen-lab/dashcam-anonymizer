"""Command line: python -m vc_label {teach,fuse,export-review,import-review} ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vc_label.config import LabelingConfig
from vc_label.contracts import read_frames_csv
from vc_label.paths import RETINAFACE_WEIGHTS, YUNET_MODEL, offline_model_env, required_model_files
from vc_label.review import ensure_outside, import_review


def _load_teachers(cfg: LabelingConfig, device_name: str):
    offline_model_env()
    missing = [p for p in required_model_files() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing model files: "
            + ", ".join(str(p) for p in missing)
            + "; download them as described in py/README.md (needs approval: metered link)"
        )

    import torch

    from vc_label.pipeline import Teachers
    from vc_label.teachers.coco import CocoTeacher
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher
    from vc_label.teachers.retinaface import RetinaFaceTeacher
    from vc_label.teachers.yunet import YuNetTeacher

    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    return Teachers(
        faces=[
            RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device, score_threshold=cfg.retinaface_score),
            YuNetTeacher(str(YUNET_MODEL), score_threshold=cfg.yunet_score),
        ],
        coco=CocoTeacher(device, score_threshold=cfg.coco_score),
        plates=Owlv2PlateTeacher(device, score_threshold=cfg.owlv2_score),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vc_label", description="Teacher pre-labelling and review round trip.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--frames", type=Path, required=True, help="vc_extract output dir (with frames.csv)")
        p.add_argument("--labels", type=Path, required=True, help="label dir, outside --frames")
        p.add_argument("--config", type=Path, help="JSON overrides for LabelingConfig")

    teach = sub.add_parser("teach", help="run the teacher models (slow, resumable)")
    common(teach)
    teach.add_argument("--every", type=int, default=1, help="label every N-th frame of frames.csv")
    teach.add_argument("--limit", type=int, help="label at most N new frames in this run")
    teach.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    common(sub.add_parser("fuse", help="fuse teacher boxes, apply rules, write YOLO labels"))

    export = sub.add_parser("export-review", help="write LabelMe JSON for X-AnyLabeling")
    common(export)
    export.add_argument("--review", type=Path, required=True, help="review dir, outside --frames")
    export.add_argument("--force", action="store_true", help="overwrite existing LabelMe files, discarding human review")

    imp = sub.add_parser("import-review", help="reviewed LabelMe JSON -> gold YOLO labels")
    imp.add_argument("--frames", type=Path, required=True)
    imp.add_argument("--review", type=Path, required=True)
    imp.add_argument("--gold", type=Path, required=True, help="gold label dir, outside --frames")
    imp.add_argument("--allow-unreviewed", action="store_true", help="also import frames without the 'reviewed' flag (training videos only)")
    return parser


def _run(args: argparse.Namespace) -> str:
    for name in ("labels", "review", "gold"):
        if getattr(args, name, None) is not None:
            ensure_outside(args.frames, getattr(args, name))
    if args.command == "import-review":
        count = import_review(args.review, read_frames_csv(args.frames), args.gold, allow_unreviewed=args.allow_unreviewed)
        return f"import-review: {count} gold frames in {args.gold}"

    cfg = LabelingConfig.from_json(args.config) if args.config else LabelingConfig()
    if args.command == "teach":
        from vc_label.pipeline import run_teach

        if args.every < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("--every and --limit must be >= 1")
        count = run_teach(args.frames, args.labels, _load_teachers(cfg, args.device), cfg, args.every, args.limit)
        return f"teach: labelled {count} new frames into {args.labels}"
    if args.command == "fuse":
        from vc_label.pipeline import run_fuse

        summary, _, _ = run_fuse(args.frames, args.labels, cfg)
        return f"fuse: {summary.frames} frames, {summary.boxes} boxes ({summary.filled} temporal fills), {summary.flags} flags"
    from vc_label.pipeline import run_export_review

    written, kept = run_export_review(args.frames, args.labels, args.review, cfg, force=args.force)
    return f"export-review: {written} new LabelMe files, {kept} kept in {args.review}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(_run(args))
    except (ValueError, OSError) as e:
        print(f"vc_label: {e}", file=sys.stderr)
        return 2
    return 0
