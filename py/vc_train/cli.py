"""python -m vc_train dataset | train | export (proxy-validation spec §4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vc_label.contracts import read_frames_csv
from vc_label.review import ensure_outside


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_train")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("dataset", help="crop dataset from teacher labels (labels/<stem>/yolo)")
    d.add_argument("--stem", action="append", required=True, help="video stem; repeat for several segments")
    d.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    d.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--width", type=int, default=1920, help="model input width")
    d.add_argument("--height", type=int, default=1088, help="model input height")
    d.add_argument("--val-tail-ms", type=float, default=20000.0)

    t = sub.add_parser("train", help="fine-tune YOLOv8n offline")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--weights", type=Path, default=None, help="default: cached yolov8n.pt")
    t.add_argument("--hours", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--device", default="mps")

    e = sub.add_parser("export", help="ONNX + model.json, optional consistency check")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--weights", type=Path, default=None, help="default: <run>/train/weights/best.pt")
    e.add_argument("--dataset", type=Path, default=None, help="dataset dir whose manifest.csv is hashed into model.json")
    e.add_argument("--width", type=int, default=1920)
    e.add_argument("--height", type=int, default=1088)
    e.add_argument("--check-frames", type=Path, default=None, help="frames dir (frames.csv) for the consistency check")
    e.add_argument("--check-count", type=int, default=20)
    return p


def _dataset(args) -> int:
    from vc_train.dataset import build_dataset

    segments = []
    for stem in args.stem:
        frames_dir = args.frames_root / stem
        ensure_outside(frames_dir, args.out)
        segments.append((frames_dir, args.labels_root / stem))
    s = build_dataset(segments, args.out, args.width, args.height, args.val_tail_ms)
    print(
        f"dataset: {s.train_samples} train / {s.val_samples} val samples from "
        f"{s.train_frames} / {s.val_frames} frames, {s.boxes} boxes in {args.out}"
    )
    return 0


def _train(args) -> int:
    from vc_train.offline import YOLO_WEIGHTS
    from vc_train.train import train

    best = train(
        args.data, args.run, args.weights or YOLO_WEIGHTS,
        imgsz=args.imgsz, epochs=args.epochs, hours=args.hours, batch=args.batch, device=args.device,
    )
    print(f"train: best weights {best}")
    return 0


def _export(args) -> int:
    weights = args.weights or args.run / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} does not exist; train first or pass --weights")
    from vc_train.export import check_consistency, export_onnx

    manifest = args.dataset / "manifest.csv" if args.dataset else None
    onnx_path = export_onnx(weights, args.run, args.width, args.height, manifest=manifest, name=args.run.name)
    print(f"export: {onnx_path}")
    if args.check_frames is None:
        return 0
    refs = read_frames_csv(args.check_frames)
    step = max(1, len(refs) // max(1, args.check_count))
    images = [r.image for r in refs[::step][: args.check_count]]
    report = check_consistency(weights, onnx_path, images)
    print(f"check: {report.images} images, {report.boxes} boxes, max diff {report.max_diff_px:.3f} px, {report.unmatched} unmatched")
    return 0 if report.passed() else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"dataset": _dataset, "train": _train, "export": _export}[args.command](args)
    except (ValueError, OSError) as e:  # FileNotFoundError is an OSError
        print(f"vc_train: {e}", file=sys.stderr)
        return 2
