"""Fine-tunes YOLOv8n on a vc_train dataset, offline (spec §4.3)."""

from __future__ import annotations

from pathlib import Path

from vc_train.offline import YOLO_WEIGHTS, load_yolo


def train(
    data_yaml: Path,
    run_dir: Path,
    weights: Path = YOLO_WEIGHTS,
    *,
    imgsz: int = 1280,
    epochs: int = 300,
    hours: float | None = 0.5,
    batch: int = 8,
    device: str = "mps",
    workers: int = 4,
) -> Path:
    """Trains into run_dir/train.

    `hours` is ultralytics `time`: the wall-time budget. When set it overrides `epochs` (ultralytics re-derives the
    epoch count from the measured epoch time, so the run can go past `epochs`); with hours=None `epochs` alone bounds it.
    """
    run_dir = Path(run_dir).resolve()
    if (run_dir / "train").exists():
        raise ValueError(f"{run_dir / 'train'} already exists; choose a new --run")
    model = load_yolo(weights)
    args = dict(
        data=str(Path(data_yaml).resolve()),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=str(run_dir),
        name="train",
        exist_ok=False,
        plots=False,
        seed=0,
        verbose=False,
    )
    if hours is not None:
        args["time"] = hours
    model.train(**args)
    best = run_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished without {best}")
    return best
