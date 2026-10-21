"""The judges of spec 2026-10-21 §3.3, tried in order: the frame's own teacher boxes, the local teachers on the
upscaled crop, an optional LLM command for what is still unsure. People come last, through review.py."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np

from vc_hardneg.candidates import Candidate, box_in_crop
from vc_hardneg.config import HardnegConfig, LlmConfig
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments, resolve
from vc_label.boxes import iou
from vc_label.contracts import Box

LLM_ENV = "VC_HARDNEG_ALLOW_LLM"
CLS_ZH = {"plate": "车牌", "face": "人脸"}
Detect = Callable[[np.ndarray], list[Box]]


def verdict_from_boxes(target: Box, found: list[Box], cfg: HardnegConfig) -> tuple[str, float]:
    """"yes" when a box at the target (IoU >= yes_iou) reaches yes_score, "no" when no box overlapping it
    (IoU >= overlap_iou) reaches no_score, else "unsure"; and the best overlapping score."""
    t = cfg.thresholds(target.cls)
    overlapping = [b for b in found if iou(b, target) >= cfg.overlap_iou]
    best = max((b.score for b in overlapping), default=0.0)
    if any(iou(b, target) >= cfg.yes_iou and b.score >= t.yes_score for b in found):
        return "yes", best
    if best < t.no_score:
        return "no", best
    return "unsure", best


class FrameTeacherJudge:
    name = "frame_teacher"

    def __init__(self, cfg: HardnegConfig) -> None:
        self.cfg = cfg

    def judge(self, c: Candidate) -> Judgment | None:
        if c.frame_teacher >= self.cfg.frame_teacher_yes:
            return Judgment(c.id, "yes", self.name, c.frame_teacher, "teacher box on the full frame")
        return None


class TeacherJudge:
    name = "teacher"

    def __init__(self, cfg: HardnegConfig, plate_detect: Detect, face_detect: Detect) -> None:
        self.cfg = cfg
        self.detect = {"plate": plate_detect, "face": face_detect}

    def judge(self, c: Candidate, crop_bgr: np.ndarray) -> Judgment:
        t = self.cfg.thresholds(c.cls)
        h, w = crop_bgr.shape[:2]
        scale = max(1.0, min(self.cfg.upscale, t.max_side / max(h, w)))
        img = cv2.resize(crop_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
        b = box_in_crop(c, self.cfg)
        target = Box(c.cls, b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale)
        found = [f for f in self.detect[c.cls](img) if f.cls == c.cls]
        verdict, best = verdict_from_boxes(target, found, self.cfg)
        return Judgment(c.id, verdict, self.name, best, f"x{scale:.2f}, {len(found)} boxes")


def make_teacher_judge(cfg: HardnegConfig, device) -> TeacherJudge:
    """The local teachers (cached weights only; offline_model_env() must have run)."""
    from vc_label.paths import RETINAFACE_WEIGHTS
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher
    from vc_label.teachers.retinaface import RetinaFaceTeacher

    owl = Owlv2PlateTeacher(device, score_threshold=min(0.05, cfg.plate.no_score))
    face = RetinaFaceTeacher(str(RETINAFACE_WEIGHTS), device, score_threshold=min(0.1, cfg.face.no_score))

    def plate_detect(img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        return owl.detect(img, [Box("vehicle", 0, 0, w, h)])

    return TeacherJudge(cfg, plate_detect, face.detect)


def _fill(text: str, fields: dict[str, str]) -> str:
    """The placeholders replaced literally: str.format would choke on the JSON braces a prompt or command holds."""
    for key, value in fields.items():
        text = text.replace(key, value)
    return text


class CommandJudge:
    """An external command (an LLM CLI) given the crop; it must print a JSON object with "verdict". It sends the
    crop wherever that command sends it, so it only runs with VC_HARDNEG_ALLOW_LLM=1."""

    def __init__(self, cfg: LlmConfig, env: Mapping[str, str] = os.environ) -> None:
        if env.get(LLM_ENV) != "1":
            raise PermissionError(f"the LLM judge sends crops to an outside service; set {LLM_ENV}=1 to allow it")
        if not cfg.command:
            raise ValueError("configs/hardneg.json llm.command is empty")
        self.cfg = cfg
        self.name = f"llm:{cfg.name}"

    def judge(self, c: Candidate, image_path: Path) -> Judgment:
        fields = {"{image}": str(image_path), "{cls}": c.cls, "{cls_zh}": CLS_ZH[c.cls]}
        fields["{prompt}"] = _fill(self.cfg.prompt, fields)
        argv = [_fill(a, fields) for a in self.cfg.command]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=self.cfg.timeout_s,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return Judgment(c.id, "unsure", self.name, None, f"timeout after {self.cfg.timeout_s:g} s")
        except OSError as e:
            return Judgment(c.id, "unsure", self.name, None, f"cannot run: {e}")
        for text in reversed(re.findall(r"\{[^{}]*\}", done.stdout)):
            try:
                answer = json.loads(text)
            except ValueError:
                continue
            if isinstance(answer, dict) and answer.get("verdict") in ("yes", "no", "unsure"):
                return Judgment(c.id, answer["verdict"], self.name, None, str(answer.get("reason", ""))[:200])
        return Judgment(c.id, "unsure", self.name, None, f"no verdict in the output (exit {done.returncode})")


def run_judges(cands: list[Candidate], judgments_path: Path, crops_root: Path, frame_judge: FrameTeacherJudge,
               teacher_judge: TeacherJudge, llm_judge=None, rejudge: bool = False) -> Counter:
    """Judges every candidate without a judgment yet (with rejudge every one people did not judge); appends each
    verdict as it comes, so an interrupted run keeps what it did. Returns counts of "<judge>:<verdict>"."""
    done = resolve(read_judgments(judgments_path))
    counts: Counter = Counter()
    for c in cands:
        prior = done.get(c.id)
        if prior is not None and (not rejudge or prior.judge == HUMAN):
            continue
        j = frame_judge.judge(c)
        if j is None:
            crop = cv2.imread(str(crops_root / c.image))
            if crop is None:
                raise FileNotFoundError(f"{crops_root / c.image} cannot be read")
            j = teacher_judge.judge(c, crop)
        append_judgments(judgments_path, [j])
        counts[f"{j.judge}:{j.verdict}"] += 1
        if j.verdict == "unsure" and llm_judge is not None:
            k = llm_judge.judge(c, crops_root / c.image)
            append_judgments(judgments_path, [k])
            counts[f"{k.judge}:{k.verdict}"] += 1
    return counts
