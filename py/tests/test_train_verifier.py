import json

import numpy as np
import pytest

from vc_hardneg.candidates import Candidate, write_candidates
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments
from vc_train.verifier import (
    VERIFIER_CLASSES, VerifierNet, export_verifier, load_crops, load_samples, pass_thresholds, train_verifier,
)

SIZE = 16


def _dump(dir_, crops):
    dir_.mkdir(parents=True)
    (dir_ / "verify_crops.bin").write_bytes(np.stack(crops).astype(np.uint8).tobytes())
    (dir_ / "verify_crops.jsonl").write_text("".join(json.dumps({"index": i, "size": SIZE}) + "\n" for i in range(len(crops))))


def _toy(tmp_path, n=120):
    """Bright crops are real, dark ones are not; half plates, half faces; two stems."""
    rng = np.random.default_rng(0)
    cands, judged = [], []
    for stem in ("a", "b"):
        crops = []
        for i in range(n):
            real = i % 2 == 0
            crops.append(np.clip(rng.normal(200 if real else 50, 20, (SIZE, SIZE, 3)), 0, 255))
            cid = f"{stem}{i}"
            cands.append(Candidate(id=cid, video="v", stem=stem, frame=i, cls=VERIFIER_CLASSES[(i // 2) % 2],
                                   box=(0, 0, 1, 1), score=0.1, src="student", track=i, dump_dir=str(tmp_path / stem),
                                   dump_index=i, frame_teacher=0.0, image="", crop_origin=(0, 0)))
            judged.append(Judgment(cid, "yes" if real else "no", "teacher"))
        _dump(tmp_path / stem, crops)
    judged.append(Judgment("a0", "unsure", HUMAN))  # a person's unsure: left out
    write_candidates(tmp_path / "c.jsonl", cands)
    append_judgments(tmp_path / "j.jsonl", judged)
    return tmp_path / "c.jsonl", tmp_path / "j.jsonl"


def test_samples_take_yes_and_no_and_read_the_right_crop(tmp_path):
    c, j = _toy(tmp_path)
    samples = load_samples(c, j)
    assert len(samples) == 239 and "a0" not in {s.id for s in samples}
    crops = load_crops(samples)
    assert crops.shape == (239, SIZE, SIZE, 3) and crops.dtype == np.uint8
    first = next(i for i, s in enumerate(samples) if s.id == "a2")
    assert crops[first].mean() > 150 and samples[first].label == 1


def test_pass_thresholds_keep_the_asked_share_of_real_crops():
    probs = np.linspace(0.01, 1.0, 100)
    labels = np.ones(100, int)
    cls = np.zeros(100, int)
    t = pass_thresholds(probs, labels, cls, keep=0.95)
    assert (probs >= t["face"]).mean() >= 0.95
    assert t["plate"] == 0.0  # no real plate in the set: pass everything


def test_training_separates_the_toy_and_the_export_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    c, j = _toy(tmp_path)
    samples = load_samples(c, j)
    result = train_verifier(samples, load_crops(samples), ["b"], epochs=15, batch=32, lr=1e-2, device="cpu", seed=0)
    acc = ((result.val_probs >= 0.5).astype(int) == result.val_labels).mean()
    assert acc > 0.9
    out = tmp_path / "model"
    onnx_path = export_verifier(result.model, out, SIZE, 4, {"name": "toy", "pass_threshold": {"face": 0.5, "plate": 0.5}})
    meta = json.loads((out / "model.json").read_text())
    assert meta["kind"] == "verifier" and meta["input"] == {"size": SIZE, "batch": 4, "color": "RGB",
                                                             "scale": 1 / 255, "pad_value": 114}
    from vc_train.ort_cpu import cpu_session

    x = np.random.default_rng(1).random((4, 3, SIZE, SIZE), dtype=np.float32)
    ort_out = cpu_session(onnx_path).run(None, {"crops": x})[0]
    with torch.no_grad():
        torch_out = result.model.cpu().eval()(torch.from_numpy(x)).numpy()
    assert ort_out.shape == (4, 2) and np.abs(ort_out - torch_out).max() < 1e-4


def test_the_cli_trains_exports_and_records_thresholds_and_metrics(tmp_path):
    pytest.importorskip("torch")
    from vc_train.cli import main

    c, j = _toy(tmp_path)
    out = tmp_path / "verifier-toy"
    assert main(["verifier", "--candidates", str(c), "--judgments", str(j), "--val-stem", "b", "--out", str(out),
                 "--epochs", "2", "--batch", "32", "--device", "cpu", "--onnx-batch", "4"]) == 0
    meta = json.loads((out / "model.json").read_text())
    assert set(meta["pass_threshold"]) == {"face", "plate"} and set(meta["metrics"]) == {"face", "plate"}
    assert meta["train"]["val_stems"] == ["b"] and len(meta["train"]["judgments_sha256"]) == 64
    assert (out / "model.onnx").is_file()
    assert main(["verifier", "--candidates", str(c), "--judgments", str(j), "--val-stem", "b", "--out", str(out),
                 "--device", "cpu", "--epochs", "1000"]) == 2  # out is not empty: refused before training
