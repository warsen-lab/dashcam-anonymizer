"""The 100-crop pilot (spec 2026-10-21 §3.4): a stratified sample, and the gate on what the models judged "no"."""

from __future__ import annotations

import random

from vc_hardneg.candidates import Candidate
from vc_hardneg.judgments import HUMAN, Judgment, resolve

STRATA = ("plate_lt02", "plate_02_05", "face_yunet", "face_student")


def stratum(c: Candidate) -> str | None:
    if c.cls == "plate":
        return "plate_lt02" if c.score < 0.2 else "plate_02_05" if c.score < 0.5 else None
    if c.cls == "face":
        return {"yunet": "face_yunet", "student": "face_student"}.get(c.src)
    return None


def sample_pilot(cands: list[Candidate], per_stratum: int = 25, seed: int = 0) -> list[Candidate]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    for s in STRATA:
        pool = sorted((c for c in cands if stratum(c) == s), key=lambda c: c.id)
        out.extend(rng.sample(pool, min(per_stratum, len(pool))))
    return out


def _sample_key(c: Candidate) -> str | None:
    """The pilot's strata, plus "face_both": faces both detectors found but that score low."""
    return stratum(c) or ("face_both" if c.cls == "face" else None)


def sample_stratified(cands: list[Candidate], total: int, seed: int = 0, exclude: set[str] = frozenset()) -> list[Candidate]:
    """About `total` candidates spread evenly over the strata (a stratum with fewer gives all it has and the rest
    goes to the others); ids in `exclude` are left out. Reproducible for a seed."""
    pools: dict[str, list[Candidate]] = {}
    for c in sorted(cands, key=lambda c: c.id):
        key = _sample_key(c)
        if key is not None and c.id not in exclude:
            pools.setdefault(key, []).append(c)
    quota = {k: 0 for k in pools}
    left = total
    open_ = sorted(pools, key=lambda k: len(pools[k]))
    while open_ and left > 0:
        share = left // len(open_)
        small = [k for k in open_ if len(pools[k]) - quota[k] <= share]
        if small:
            for k in small:
                left -= len(pools[k]) - quota[k]
                quota[k] = len(pools[k])
                open_.remove(k)
            continue
        for k in open_:
            quota[k] += share
        left -= share * len(open_)
        for k in sorted(open_)[:left]:  # the remainder, one each
            quota[k] += 1
        left = 0
    rng = random.Random(seed)
    return [c for k in sorted(pools) for c in rng.sample(pools[k], quota[k])]


def _confusion(model: dict[str, Judgment], truth: dict[str, str], ids: list[str]) -> dict:
    rows = {v: {"yes": 0, "no": 0} for v in ("yes", "no", "unsure")}
    for i in ids:
        if i in model and i in truth:
            rows[model[i].verdict][truth[i]] += 1
    no = rows["no"]["yes"] + rows["no"]["no"]
    return {"confusion": rows, "no": no, "no_but_real": rows["no"]["yes"],
            "no_but_real_rate": rows["no"]["yes"] / no if no else 0.0}


def pilot_report(ids: list[str], judgments: list[Judgment], max_no_false: float = 0.02) -> dict:
    """Per model judge and for the models together (the last non-human word per id): how their verdicts compare
    with the people's. Passes when, of what the models together judged "no", at most max_no_false is real."""
    truth = {j.id: j.verdict for j in judgments if j.judge == HUMAN}
    judged = [i for i in ids if i in truth]
    out = {"ids": len(ids), "judged_by_people": len(judged), "max_no_false": max_no_false, "judges": {}}
    for name in sorted({j.judge for j in judgments if j.judge != HUMAN}):
        out["judges"][name] = _confusion(resolve([j for j in judgments if j.judge == name]), truth, judged)
    out["pipeline"] = _confusion(resolve(judgments, include_human=False), truth, judged)
    out["passed"] = len(judged) == len(ids) and out["pipeline"]["no_but_real_rate"] <= max_no_false
    return out
