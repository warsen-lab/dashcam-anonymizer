"""Label cleaning for the student (spec 2026-10-21 §5.1, as decided on 2026-10-21): plate label boxes judged not real are
left out of the dataset, and with drop_tall_low_plates also the low-band plate boxes taller than wide (lamps and body
corners: 1.1 % of the teachers' confident plates have that shape, 35 % of the false labels people found). A box judged
real always stays; face labels are never touched (judging them "no" was wrong 8 times in 27). labels/<stem>/yolo is
never changed."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vc_hardneg.candidates import LOW_BANDS, read_candidates
from vc_hardneg.judgments import read_judgments, resolve
from vc_label.boxes import iou
from vc_label.contracts import Box, read_jsonl

SAME_BOX_IOU = 0.9  # a label box read back from yolo/ (rounded to 6 decimals) and its merged.jsonl box
CLEANED = "plate"  # the only class whose labels are cleaned


def _matches(box: Box, others: list[Box]) -> bool:
    return any(o.cls == box.cls and iou(o, box) >= SAME_BOX_IOU for o in others)


@dataclass
class LabelCleaning:
    no: dict[tuple[str, int], list[Box]]  # (stem, frame) -> plate boxes judged not real
    yes: dict[tuple[str, int], list[Box]]  # ... judged real
    low_boxes: dict[tuple[str, int], list[Box]] = field(default_factory=dict)  # low-band teacher plate boxes
    drop_tall_low_plates: bool = False

    @staticmethod
    def load(candidates: Path, judgments: Path, labels_root: Path, stems: list[str],
             drop_tall_low_plates: bool) -> LabelCleaning:
        final = resolve(read_judgments(judgments))
        no: dict[tuple[str, int], list[Box]] = defaultdict(list)
        yes: dict[tuple[str, int], list[Box]] = defaultdict(list)
        for c in read_candidates(candidates):
            j = final.get(c.id)
            if c.cls != CLEANED or j is None or j.verdict == "unsure":
                continue
            (no if j.verdict == "no" else yes)[(c.stem, c.frame)].append(Box(c.cls, *c.box))
        low: dict[tuple[str, int], list[Box]] = defaultdict(list)
        if drop_tall_low_plates:
            lo, hi = LOW_BANDS[CLEANED]
            for stem in stems:
                for d in read_jsonl(labels_root / stem / "merged.jsonl"):
                    if d.box.cls == CLEANED and lo <= d.box.score < hi:
                        low[(stem, d.frame)].append(d.box)
        return LabelCleaning(dict(no), dict(yes), dict(low), drop_tall_low_plates)

    def clean(self, stem: str, frame: int, boxes: list[Box]) -> tuple[list[Box], int]:
        """The boxes to keep, and how many were left out."""
        key = (stem, frame)
        no, yes, low = self.no.get(key, []), self.yes.get(key, []), self.low_boxes.get(key, [])
        kept = []
        for b in boxes:
            if b.cls == CLEANED and not _matches(b, yes):
                if _matches(b, no):
                    continue
                if self.drop_tall_low_plates and b.width < b.height and _matches(b, low):
                    continue
            kept.append(b)
        return kept, len(boxes) - len(kept)


@dataclass(frozen=True)
class HardNegative:
    """A student false positive on a labelled frame (vc_hardneg, judged "no"): its region becomes an extra sample."""

    stem: str
    frame: int
    box: Box


def load_hard_negatives(candidates: Path, judgments: Path) -> list[HardNegative]:
    """The candidates whose final verdict is "no", ordered by id (a stable, content-blind order to cap them by)."""
    final = resolve(read_judgments(judgments))
    return [HardNegative(c.stem, c.frame, Box(c.cls, *c.box)) for c in sorted(read_candidates(candidates), key=lambda c: c.id)
            if final.get(c.id) is not None and final[c.id].verdict == "no"]


def hard_negative_window(box: Box, width: int, height: int, side: int = 640) -> tuple[int, int, int, int]:
    """(x0, y0, w, h): side x side frame pixels centred on the box, shifted inside the frame (the whole frame when it is
    smaller)."""
    w, h = min(side, width), min(side, height)
    cx, cy = box.center
    x0 = int(min(max(round(cx - w / 2), 0), width - w))
    y0 = int(min(max(round(cy - h / 2), 0), height - h))
    return x0, y0, w, h
