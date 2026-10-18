"""python -m vc_train dataset | train | export (proxy-validation spec §4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vc_label.contracts import read_frames_csv
from vc_label.review import ensure_outside
from vc_train.offline import ultralytics_offline_env


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
    e.add_argument("--check-count", type=int, default=20, help="check frames, spread evenly over frames.csv (>= 1)")
    return p


def _check_indices(total: int, count: int) -> list[int]:
    """Frame indices spread evenly over the whole list, both ends included, in order.

    For count >= 2 this is round(i * (total - 1) / (count - 1)) with halves rounded up, done in integers
    so a tie like 24.5 goes to 25 whatever the float error. count == 1 takes the middle frame.
    """
    if total <= count:
        return list(range(total))
    if count == 1:
        return [(total - 1) // 2]
    span = 2 * (count - 1)
    return list(dict.fromkeys((2 * i * (total - 1) + (count - 1)) // span for i in range(count)))


def _dataset(args) -> int:
    from vc_train.dataset import build_dataset

    # the whole frames root is off limits, including segments that are not listed with --stem
    ensure_outside(args.frames_root, args.out)
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
    for segment in s.segments:
        print(f"  {segment.name}: {segment.train_frames} train / {segment.val_frames} val frames")
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
    # everything is validated before anything is written
    if args.check_count < 1:
        raise ValueError(f"--check-count must be at least 1, got {args.check_count}")
    if args.check_frames is not None:
        # the model must not land inside a frames dir: vc_extract --overwrite deletes that whole dir
        ensure_outside(args.check_frames, args.run)
    weights = args.weights or args.run / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} does not exist; train first or pass --weights")
    images = None
    if args.check_frames is not None:
        refs = read_frames_csv(args.check_frames)
        if not refs:
            raise ValueError(f"{args.check_frames}/frames.csv lists no frames to check")
        images = [refs[i].image for i in _check_indices(len(refs), args.check_count)]
    from vc_train.export import check_consistency, export_onnx

    manifest = args.dataset / "manifest.csv" if args.dataset else None
    onnx_path = export_onnx(weights, args.run, args.width, args.height, manifest=manifest, name=args.run.name)
    print(f"export: {onnx_path}")
    if images is None:
        return 0
    report = check_consistency(weights, onnx_path, images)
    print(
        f"check: {report.images} images, {report.boxes} boxes, {report.matched} matched, max diff {report.max_diff_px:.3f} px, "
        f"{report.unmatched_confident} unmatched confident, {report.unmatched_low} unmatched low"
    )
    return 0 if report.passed() else 1


def main(argv: list[str] | None = None) -> int:
    # before any command can reach ultralytics, directly or by unpickling a checkpoint (defence in depth: train and
    # export call it again where they load a model)
    ultralytics_offline_env()
    args = _parser().parse_args(argv)
    try:
        return {"dataset": _dataset, "train": _train, "export": _export}[args.command](args)
    except (ValueError, OSError) as e:  # FileNotFoundError is an OSError
        print(f"vc_train: {e}", file=sys.stderr)
        return 2
