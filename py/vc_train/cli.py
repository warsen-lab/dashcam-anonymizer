"""python -m vc_train dataset | train | export (proxy-validation spec §4) | yunet (the face model for vc_redact) |
verifier (the verifier for vc_redact --verifier-model, spec 2026-10-21 §4)."""

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
    d.add_argument("--label-candidates", type=Path, default=None, help="vc_hardneg mine-labels candidates.jsonl")
    d.add_argument("--label-judgments", type=Path, default=None, help="their judgments.jsonl: plates judged no are left out")
    d.add_argument("--drop-tall-low-plates", action="store_true",
                   help="also leave out the low-band plate labels taller than wide (lamps, body corners)")
    d.add_argument("--keep-no-plates-narrower-than", type=float, default=0.0, metavar="PX",
                   help="keep plate labels judged no that are narrower than this (the shape rule still applies)")
    d.add_argument("--grey-dropped-labels", action="store_true",
                   help="cover the label boxes left out in grey, so they are learnt as neither plate nor background")
    d.add_argument("--hard-negatives", type=Path, default=None, help="vc_hardneg candidates.jsonl of student false positives")
    d.add_argument("--hard-negative-judgments", type=Path, default=None, help="their judgments.jsonl")
    d.add_argument("--hard-negative-share", type=float, default=0.10, help="extra samples, as a share of the others")

    t = sub.add_parser("train", help="fine-tune YOLOv8n offline")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--frames-root", type=Path, default=Path("data/frames"), help="the run dir must not be under it")
    t.add_argument("--weights", type=Path, default=None, help="default: cached yolov8n.pt")
    t.add_argument("--hours", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--device", default="mps")

    e = sub.add_parser("export", help="ONNX + model.json, optional consistency check")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--name", default=None, help="model name in model.json (default: the run dir name)")
    e.add_argument("--frames-root", type=Path, default=Path("data/frames"), help="the run dir must not be under it")
    e.add_argument("--weights", type=Path, default=None, help="default: <run>/train/weights/best.pt")
    e.add_argument("--dataset", type=Path, default=None, help="dataset dir whose manifest.csv is hashed into model.json")
    e.add_argument("--width", type=int, default=1920)
    e.add_argument("--height", type=int, default=1088)
    e.add_argument("--check-frames", type=Path, default=None, help="frames dir (frames.csv) for the consistency check")
    e.add_argument("--check-count", type=int, default=20, help="check frames, spread evenly over frames.csv (>= 1)")

    y = sub.add_parser("yunet", help="YuNet face model with a free input size for vc_redact --face-model")
    y.add_argument("--out", type=Path, required=True, help="new or empty directory for model.onnx + model.json")
    y.add_argument("--source", type=Path, default=None, help="OpenCV Zoo ONNX (default: the cached face_detection_yunet_2023mar.onnx)")
    y.add_argument("--frames-root", type=Path, default=Path("data/frames"), help="the out dir must not be under it")

    v = sub.add_parser("verifier", help="train the verifier on judged candidates (spec 2026-10-21 §4)")
    v.add_argument("--candidates", type=Path, required=True, help="vc_hardneg candidates.jsonl")
    v.add_argument("--judgments", type=Path, required=True, help="vc_hardneg judgments.jsonl")
    v.add_argument("--val-stem", action="append", required=True, help="stems held out for validation; repeat")
    v.add_argument("--out", type=Path, required=True, help="new or empty directory for model.onnx + model.json")
    v.add_argument("--epochs", type=int, default=30)
    v.add_argument("--batch", type=int, default=256)
    v.add_argument("--lr", type=float, default=1e-3)
    v.add_argument("--device", default="mps")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--keep", type=float, default=0.995,
                   help="share of the real validation crops each class's pass threshold lets through")
    v.add_argument("--onnx-batch", type=int, default=16, help="the exported model's fixed batch")
    v.add_argument("--arch", choices=("small_cnn", "mobilenet_v3_small"), default="small_cnn",
                   help="mobilenet_v3_small starts from the cached ImageNet weights (never downloaded)")
    v.add_argument("--class", dest="classes", action="append", choices=("face", "plate"),
                   help="train these classes only; repeat (default both); vc_redact verifies only the model's classes")
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
    if (args.label_candidates is None) != (args.label_judgments is None):
        raise ValueError("--label-candidates and --label-judgments go together")
    if args.drop_tall_low_plates and args.label_candidates is None:
        raise ValueError("--drop-tall-low-plates needs --label-candidates and --label-judgments")
    for flag, used in (("--grey-dropped-labels", args.grey_dropped_labels),
                       ("--keep-no-plates-narrower-than", args.keep_no_plates_narrower_than > 0)):
        if used and args.label_candidates is None:
            raise ValueError(f"{flag} needs --label-candidates and --label-judgments")
    if (args.hard_negatives is None) != (args.hard_negative_judgments is None):
        raise ValueError("--hard-negatives and --hard-negative-judgments go together")
    cleaning, hard = None, []
    if args.label_candidates is not None:
        from vc_train.cleaning import LabelCleaning

        cleaning = LabelCleaning.load(args.label_candidates, args.label_judgments, args.labels_root, args.stem,
                                      args.drop_tall_low_plates, args.keep_no_plates_narrower_than)
    if args.hard_negatives is not None:
        from vc_train.cleaning import load_hard_negatives

        hard = load_hard_negatives(args.hard_negatives, args.hard_negative_judgments)
    s = build_dataset(segments, args.out, args.width, args.height, args.val_tail_ms, cleaning=cleaning,
                      hard_negatives=hard, hard_negative_share=args.hard_negative_share,
                      grey_dropped=args.grey_dropped_labels)
    print(
        f"dataset: {s.train_samples} train / {s.val_samples} val samples from "
        f"{s.train_frames} / {s.val_frames} frames, {s.boxes} boxes in {args.out}"
    )
    for segment in s.segments:
        print(f"  {segment.name}: {segment.train_frames} train / {segment.val_frames} val frames")
    if cleaning is not None:
        grey = " (covered in grey)" if args.grey_dropped_labels else ""
        print(f"  label boxes left out by the cleaning: {s.labels_dropped}{grey}")
    if args.hard_negatives is not None:
        print(f"  hard-negative samples: {s.hard_negatives} (of {len(hard)} judged false positives)")
    return 0


def _train(args) -> int:
    # vc_extract --overwrite deletes a whole frames dir, run dir included
    ensure_outside(args.frames_root, args.run)
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
    # the model must not land inside a frames dir: vc_extract --overwrite deletes that whole dir
    ensure_outside(args.frames_root, args.run)
    if args.check_frames is not None:
        ensure_outside(args.check_frames, args.run)
    weights = args.weights or args.run / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} does not exist; train first or pass --weights")
    manifest = None
    if args.dataset is not None:
        manifest = args.dataset / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"{manifest} does not exist; --dataset must be a directory written by vc_train dataset")
    images = None
    if args.check_frames is not None:
        refs = read_frames_csv(args.check_frames)
        if not refs:
            raise ValueError(f"{args.check_frames}/frames.csv lists no frames to check")
        images = [refs[i].image for i in _check_indices(len(refs), args.check_count)]
        missing = [image for image in images if not image.is_file()]
        if missing:
            raise FileNotFoundError(f"check frame {missing[0]} does not exist ({len(missing)} of {len(images)} chosen frames are missing)")
    from vc_train.export import check_consistency, export_onnx

    onnx_path = export_onnx(weights, args.run, args.width, args.height, manifest=manifest, name=args.name or args.run.name)
    print(f"export: {onnx_path}")
    if images is None:
        return 0
    report = check_consistency(weights, onnx_path, images)
    print(
        f"check: {report.images} images, {report.boxes} boxes, {report.matched} matched, max diff {report.max_diff_px:.3f} px, "
        f"{report.unmatched_confident} unmatched confident, {report.unmatched_low} unmatched low"
    )
    return 0 if report.passed() else 1


def _yunet(args) -> int:
    # vc_extract --overwrite deletes a whole frames dir, model dir included
    ensure_outside(args.frames_root, args.out)
    from vc_label.paths import YUNET_MODEL
    from vc_train.yunet import write_yunet_model

    onnx_path = write_yunet_model(args.source or YUNET_MODEL, args.out)
    print(f"yunet: {onnx_path}")
    return 0


def _verifier(args) -> int:
    import hashlib

    from vc_train.verifier import (
        VERIFIER_CLASSES, export_verifier, load_crops, load_samples, mobilenet_weights, pass_thresholds, train_verifier,
    )

    if not 0 < args.keep <= 1:
        raise ValueError(f"--keep must be in (0, 1], got {args.keep}")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise ValueError(f"{args.out} is not empty; choose a new --out")  # before training, not after
    classes = tuple(c for c in VERIFIER_CLASSES if c in (args.classes or VERIFIER_CLASSES))
    pretrained = mobilenet_weights() if args.arch == "mobilenet_v3_small" else None
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(f"{pretrained} does not exist; put torchvision's {pretrained.name} there first")
    samples = load_samples(args.candidates, args.judgments, classes)
    crops = load_crops(samples)
    r = train_verifier(samples, crops, args.val_stem, args.epochs, args.batch, args.lr, args.device, args.seed,
                       classes, args.arch, pretrained)
    thresholds = pass_thresholds(r.val_probs, r.val_labels, r.val_cls, args.keep, classes)
    metrics = {}
    for i, name in enumerate(classes):
        m = r.val_cls == i
        real, fake = m & (r.val_labels == 1), m & (r.val_labels == 0)
        metrics[name] = {
            "val_real": int(real.sum()), "val_not_real": int(fake.sum()),
            "real_passed": float((r.val_probs[real] >= thresholds[name]).mean()) if real.any() else None,
            "not_real_rejected": float((r.val_probs[fake] < thresholds[name]).mean()) if fake.any() else None,
        }

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    meta = {"name": args.out.name, "arch": args.arch, "pass_threshold": thresholds, "metrics": metrics,
            "train": {"samples": r.train_count, "val_stems": args.val_stem, "epochs": args.epochs, "keep": args.keep,
                      "lr": args.lr, "batch": args.batch,
                      "candidates_sha256": sha(args.candidates), "judgments_sha256": sha(args.judgments)}}
    onnx_path = export_verifier(r.model, args.out, crops.shape[1], args.onnx_batch, meta, classes)
    print(f"verifier: {onnx_path}; pass thresholds {thresholds}")
    for name, m in metrics.items():
        print(f"  {name}: {m}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # before any command can reach ultralytics, directly or by unpickling a checkpoint (defence in depth: train and
    # export call it again where they load a model)
    ultralytics_offline_env()
    args = _parser().parse_args(argv)
    try:
        return {"dataset": _dataset, "train": _train, "export": _export, "yunet": _yunet,
                "verifier": _verifier}[args.command](args)
    except (ValueError, OSError) as e:  # FileNotFoundError is an OSError
        print(f"vc_train: {e}", file=sys.stderr)
        return 2
