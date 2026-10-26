"""Question A against people's own verdicts (spec 2026-10-22 §11): a way of asking is good enough to scale up only when
people overturn at most 2 % of its answers. Runs in a directory of its own, so the answer key stays untouched."""

from __future__ import annotations

import json
import random
from pathlib import Path

from vc_gt.config import GtConfig
from vc_gt.crops import CROPS_INDEX, read_crop_index
from vc_gt.judge import OBJECTS, REAL, Runner, ask_per_crop
from vc_hardneg.judgments import HUMAN, read_judgments, resolve

OVERTURN_MAX = 0.02


def compare_verdicts(model: dict[str, str], people: dict[str, str]) -> dict:
    """Over the objects both judged: how many the model answered (not "unsure"), how many of those people overturned,
    and the two overturns that hide leaks: a real object called not real, a readable one called unreadable."""
    both = [i for i in people if i in model]
    answered = [i for i in both if model[i] != "unsure"]
    return {
        "judged": len(both),
        "answered": len(answered),
        "overturned": sum(model[i] != people[i] for i in answered),
        "unsure": len(both) - len(answered),
        "real_called_not_real": sum(people[i] in REAL and model[i] == "not_real" for i in answered),
        "readable_called_unreadable": sum(people[i] == "real_readable" and model[i] == "real_unreadable"
                                          for i in answered),
    }


def calibrate(gt_dir: Path, cls_of: dict[str, str], cfg: GtConfig, runner: Runner, judge: str, sample: int, seed: int,
              out_dir: Path) -> dict:
    """Asks question A (per crop, as judge_objects does) about a seeded sample of the objects people judged in
    gt_dir, writing everything to out_dir; returns compare_verdicts plus the overturn rate, also in
    out_dir/calibration.json."""
    gt_dir, out_dir = Path(gt_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    people = {i: j.verdict for i, j in resolve(read_judgments(gt_dir / OBJECTS)).items() if j.judge == HUMAN}
    ids = sorted(people)
    chosen = sorted(random.Random(seed).sample(ids, min(sample, len(ids))))
    index = read_crop_index(gt_dir / CROPS_INDEX)
    pending = {i: (cls_of[i], index[i]) for i in chosen}
    ask_per_crop(out_dir, pending, cfg, runner, judge, objects_path=out_dir / OBJECTS)
    model = {i: j.verdict for i, j in resolve(read_judgments(out_dir / OBJECTS)).items()}
    result = compare_verdicts(model, {i: people[i] for i in chosen})
    result["overturn_rate"] = result["overturned"] / result["answered"] if result["answered"] else None
    result["passed"] = result["overturn_rate"] is not None and result["overturn_rate"] <= OVERTURN_MAX
    result["rows"] = [{"id": i, "people": people[i], "model": model.get(i)} for i in chosen]
    (out_dir / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
