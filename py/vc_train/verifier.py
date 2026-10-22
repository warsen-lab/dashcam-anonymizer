"""The verifier (spec 2026-10-21 §4): a small network scoring vc_redact's dumped crops, trained on judged candidates.

Two architectures: "small_cnn" (from scratch, about 0.3M parameters) and "mobilenet_v3_small" (ImageNet weights from the
shared cache, about 1M parameters without its classifier; the 2026-10-21 run showed the small CNN overfits the few
thousand crops). Either takes the crops as they are dumped (RGB / 255); the MobileNet upsamples and normalises inside,
so vc_redact feeds both the same way."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vc_hardneg.candidates import read_candidates
from vc_hardneg.judgments import read_judgments, resolve

VERIFIER_CLASSES = ("face", "plate")
ARCHS = ("small_cnn", "mobilenet_v3_small")
MOBILENET_WEIGHTS = "mobilenet_v3_small-047dcff4.pth"  # torchvision's ImageNet weights, in $TORCH_HOME/hub/checkpoints
MOBILENET_SIDE = 128  # the MobileNet sees the crop upsampled to this
PAD_VALUE = 114
CROP_FACTOR = 2.5
MIN_CROP_PX = 32


@dataclass(frozen=True)
class Sample:
    id: str
    stem: str
    cls_index: int  # into the classes it was loaded for
    label: int  # 1 real, 0 not
    dump_dir: str
    dump_index: int


def load_samples(candidates: Path, judgments: Path, classes: tuple[str, ...] = VERIFIER_CLASSES) -> list[Sample]:
    """Candidates of `classes` whose final verdict is yes or no (unsure ones are left out), in candidates-file order."""
    final = resolve(read_judgments(judgments))
    out = []
    for c in read_candidates(candidates):
        j = final.get(c.id)
        if j is None or j.verdict == "unsure" or c.cls not in classes:
            continue
        out.append(Sample(c.id, c.stem, classes.index(c.cls), int(j.verdict == "yes"), c.dump_dir, c.dump_index))
    return out


def mobilenet_weights() -> Path:
    """Where the cached MobileNetV3-Small ImageNet weights are (vc_label.paths.offline_model_env sets TORCH_HOME)."""
    from vc_label.paths import offline_model_env

    offline_model_env()
    return Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / MOBILENET_WEIGHTS


def load_crops(samples: list[Sample]) -> np.ndarray:
    """[N, S, S, 3] uint8, the dumped bytes of each sample (all dumps must share one size)."""
    sizes, maps = {}, {}
    for d in sorted({s.dump_dir for s in samples}):
        with (Path(d) / "verify_crops.jsonl").open(encoding="utf-8") as f:
            sizes[d] = json.loads(f.readline())["size"]
        maps[d] = np.memmap(Path(d) / "verify_crops.bin", dtype=np.uint8, mode="r").reshape(-1, sizes[d], sizes[d], 3)
    if len(set(sizes.values())) > 1:
        raise ValueError(f"the dumps have different crop sizes: {sizes}")
    return np.stack([np.asarray(maps[s.dump_dir][s.dump_index]) for s in samples])


def _torch():
    import torch
    from torch import nn

    return torch, nn


def VerifierNet(n_classes: int = 2, arch: str = "small_cnn", pretrained: Path | None = None):  # noqa: N802
    """The network for `arch` (a factory: torch is imported lazily). For "mobilenet_v3_small", `pretrained` is the
    ImageNet weights file (FileNotFoundError when missing; nothing is ever downloaded) or None for random weights."""
    torch, nn = _torch()
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    if arch == "mobilenet_v3_small":
        import torchvision

        if pretrained is not None and not Path(pretrained).is_file():
            raise FileNotFoundError(f"{pretrained} does not exist; the MobileNet weights are never downloaded here")
        base = torchvision.models.mobilenet_v3_small()
        if pretrained is not None:
            base.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))

        class _Mobile(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = base.features
                self.head = nn.Linear(576, n_classes)
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x):
                x = nn.functional.interpolate(x, size=(MOBILENET_SIDE, MOBILENET_SIDE), mode="bilinear", align_corners=False)
                return self.head(self.features((x - self.mean) / self.std).mean(dim=(2, 3)))

        return _Mobile()

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers, cin = [], 3
            for cout in (16, 32, 64, 128):
                layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                           nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
                cin = cout
            self.features = nn.Sequential(*layers)
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):
            return self.head(self.features(x).mean(dim=(2, 3)))

    return _Net()


@dataclass
class TrainResult:
    model: object
    val_probs: np.ndarray  # pass probability of each validation sample (its own class's head)
    val_labels: np.ndarray
    val_cls: np.ndarray
    train_count: int


def _augment(torch, x):
    """Random horizontal flip, brightness +-15 %, shift up to 4 px with pad-value borders."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = (x * (0.85 + 0.3 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))).clamp(0, 1)
    pad = torch.nn.functional.pad(x, (4, 4, 4, 4), value=PAD_VALUE / 255)
    dx, dy = np.random.randint(0, 9, 2)
    return pad[:, :, dy : dy + x.shape[2], dx : dx + x.shape[3]]


def train_verifier(samples: list[Sample], crops: np.ndarray, val_stems: list[str], epochs: int = 30, batch: int = 256,
                   lr: float = 1e-3, device: str = "mps", seed: int = 0, classes: tuple[str, ...] = VERIFIER_CLASSES,
                   arch: str = "small_cnn", pretrained: Path | None = None) -> TrainResult:
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    val = np.array([s.stem in val_stems for s in samples])
    if val.all() or not val.any():
        raise ValueError(f"--val-stem {val_stems} must leave both training and validation samples")
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255
    y = torch.tensor([s.label for s in samples], dtype=torch.float32)
    k = torch.tensor([s.cls_index for s in samples])
    tr = np.flatnonzero(~val)
    model = VerifierNet(len(classes), arch, pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + batch - 1) // batch))
    # balance yes / no within each class: each sample weighs 1 / (count of its (class, label))
    groups = k[tr] * 2 + y[tr].long()
    weight = 1.0 / torch.bincount(groups, minlength=2 * len(classes)).float().clamp(min=1)[groups]
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        model.train()
        order = tr[torch.multinomial(weight, len(tr), replacement=True).numpy()]
        for start in range(0, len(order), batch):
            idx = torch.from_numpy(order[start : start + batch])
            xb = _augment(torch, x[idx].to(device))
            logits = model(xb).gather(1, k[idx].to(device)[:, None])[:, 0]
            loss = loss_fn(logits, y[idx].to(device)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    vi = np.flatnonzero(val)
    with torch.no_grad():
        logits = torch.cat([model(x[vi[s : s + batch]].to(device)).cpu() for s in range(0, len(vi), batch)])
    probs = torch.sigmoid(logits.gather(1, k[vi][:, None])[:, 0]).numpy()
    return TrainResult(model, probs, y[vi].numpy().astype(int), k[vi].numpy(), len(tr))


def pass_thresholds(probs: np.ndarray, labels: np.ndarray, cls_index: np.ndarray, keep: float = 0.995,
                    classes: tuple[str, ...] = VERIFIER_CLASSES) -> dict[str, float]:
    """Per class the largest threshold that still passes `keep` of its real crops (0 when it has none)."""
    out = {}
    for i, name in enumerate(classes):
        real = np.sort(probs[(cls_index == i) & (labels == 1)])
        out[name] = 0.0 if len(real) == 0 else float(real[int(np.floor((1 - keep) * len(real)))])
    return out


def export_verifier(model, out_dir: Path, size: int, batch: int, meta: dict,
                    classes: tuple[str, ...] = VERIFIER_CLASSES) -> Path:
    """model.onnx ([batch, 3, size, size] -> [batch, len(classes)]) + model.json into out_dir (absent or empty), all at
    once. meta["arch"] defaults to "small_cnn"."""
    torch, _ = _torch()
    from vc_train.ort_cpu import cpu_session

    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} is not empty; choose a new --out")
    model = model.cpu().eval()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        onnx_path = staging / "model.onnx"
        dummy = torch.rand(batch, 3, size, size)
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["crops"], output_names=["logits"], opset_version=17,
                          dynamo=False)
        session = cpu_session(onnx_path)
        if [list(i.shape) for i in session.get_inputs()] != [[batch, 3, size, size]]:
            raise ValueError("the exported verifier does not have a static input")
        with torch.no_grad():
            diff = np.abs(session.run(None, {"crops": dummy.numpy()})[0] - model(dummy).numpy()).max()
        if diff > 1e-4:
            raise ValueError(f"onnxruntime and torch disagree by {diff}")
        info = {
            "name": meta.get("name", out_dir.name), "kind": "verifier", "classes": list(classes),
            "arch": meta.get("arch", "small_cnn" if not hasattr(model, "std") else "mobilenet_v3_small"),
            "input": {"size": size, "batch": batch, "color": "RGB", "scale": 1 / 255, "pad_value": PAD_VALUE},
            "crop": {"factor": CROP_FACTOR, "min_px": MIN_CROP_PX},
            **{k: v for k, v in meta.items() if k not in ("name", "arch")},
        }
        (staging / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # empty, checked above
        os.replace(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir / "model.onnx"
