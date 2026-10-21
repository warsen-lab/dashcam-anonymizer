"""The verifier (spec 2026-10-21 §4): a small CNN scoring vc_redact's dumped crops, trained on judged candidates."""

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
PAD_VALUE = 114
CROP_FACTOR = 2.5
MIN_CROP_PX = 32


@dataclass(frozen=True)
class Sample:
    id: str
    stem: str
    cls_index: int  # into VERIFIER_CLASSES
    label: int  # 1 real, 0 not
    dump_dir: str
    dump_index: int


def load_samples(candidates: Path, judgments: Path) -> list[Sample]:
    """Candidates whose final verdict is yes or no (unsure ones are left out), in candidates-file order."""
    final = resolve(read_judgments(judgments))
    out = []
    for c in read_candidates(candidates):
        j = final.get(c.id)
        if j is None or j.verdict == "unsure" or c.cls not in VERIFIER_CLASSES:
            continue
        out.append(Sample(c.id, c.stem, VERIFIER_CLASSES.index(c.cls), int(j.verdict == "yes"), c.dump_dir, c.dump_index))
    return out


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


def VerifierNet(n_classes: int = 2):  # noqa: N802 (a class factory: torch is imported lazily)
    torch, nn = _torch()

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
                   lr: float = 1e-3, device: str = "mps", seed: int = 0) -> TrainResult:
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
    model = VerifierNet(len(VERIFIER_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + batch - 1) // batch))
    # balance yes / no within each class: each sample weighs 1 / (count of its (class, label))
    groups = k[tr] * 2 + y[tr].long()
    weight = 1.0 / torch.bincount(groups, minlength=4).float().clamp(min=1)[groups]
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


def pass_thresholds(probs: np.ndarray, labels: np.ndarray, cls_index: np.ndarray, keep: float = 0.995) -> dict[str, float]:
    """Per class the largest threshold that still passes `keep` of its real crops (0 when it has none)."""
    out = {}
    for i, name in enumerate(VERIFIER_CLASSES):
        real = np.sort(probs[(cls_index == i) & (labels == 1)])
        out[name] = 0.0 if len(real) == 0 else float(real[int(np.floor((1 - keep) * len(real)))])
    return out


def export_verifier(model, out_dir: Path, size: int, batch: int, meta: dict) -> Path:
    """model.onnx ([batch, 3, size, size] -> [batch, 2]) + model.json into out_dir (absent or empty), all at once."""
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
            "name": meta.get("name", out_dir.name), "kind": "verifier", "classes": list(VERIFIER_CLASSES),
            "input": {"size": size, "batch": batch, "color": "RGB", "scale": 1 / 255, "pad_value": PAD_VALUE},
            "crop": {"factor": CROP_FACTOR, "min_px": MIN_CROP_PX},
            **{k: v for k, v in meta.items() if k != "name"},
        }
        (staging / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # empty, checked above
        os.replace(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir / "model.onnx"
