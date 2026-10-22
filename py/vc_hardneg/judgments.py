"""judgments.jsonl (spec 2026-10-21 §3.3): one verdict per line, appended; the last per id wins, people over models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

VERDICTS = ("yes", "no", "unsure")
HUMAN = "human"


@dataclass(frozen=True)
class Judgment:
    id: str
    verdict: str  # one of VERDICTS
    judge: str  # "frame_teacher" | "teacher" | "llm:<name>" | "human"
    score: float | None = None
    detail: str = ""


def append_judgments(path: Path, items: Iterable[Judgment]) -> None:
    """Appends every item, or none when one has a verdict outside VERDICTS (ValueError)."""
    items = list(items)
    for j in items:
        if j.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {j.verdict!r} for {j.id}")
    with Path(path).open("a", encoding="utf-8") as f:
        for j in items:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")


def read_judgments(path: Path) -> list[Judgment]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Judgment(**json.loads(line)) for line in f if line.strip()]


def resolve(judgments: Iterable[Judgment], include_human: bool = True) -> dict[str, Judgment]:
    """Per id the last judgment, except that a person's last word beats any later model's."""
    out: dict[str, Judgment] = {}
    for j in judgments:
        if j.judge == HUMAN and not include_human:
            continue
        if j.id in out and out[j.id].judge == HUMAN and j.judge != HUMAN:
            continue
        out[j.id] = j
    return out
