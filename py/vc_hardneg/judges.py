"""The judges of spec 2026-10-21 §3.3, tried in order: the frame's own teacher boxes, the local teachers on the
upscaled crop, an optional LLM command for what is still unsure. People come last, through review.py."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np

from vc_hardneg.candidates import Candidate, box_in_crop
from vc_hardneg.config import HardnegConfig, LlmConfig
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments, read_judgments, resolve
from vc_label.boxes import intersection_area, iou
from vc_label.contracts import Box

LLM_ENV = "VC_HARDNEG_ALLOW_LLM"
CLS_ZH = {"plate": "车牌", "face": "人脸"}
Detect = Callable[[np.ndarray], list[Box]]


def verdict_from_boxes(target: Box, found: list[Box], cfg: HardnegConfig, scale: float = 1.0) -> tuple[str, float]:
    """"yes" when a box at the target reaches yes_score, "no" when no box overlapping it reaches no_score, else
    "unsure"; and the best overlapping score. A box is at the target with IoU >= yes_iou, overlaps it with IoU >=
    overlap_iou, and is both when at least contain_frac of it lies inside the target (the target masks it). A
    target whose long side is below tiny_px frame pixels (`scale`: its upscaling) is never "no": the teachers
    cannot tell (the 2026-10-21 pilot)."""
    t = cfg.thresholds(target.cls)

    def inside(b: Box) -> bool:
        return b.area > 0 and intersection_area(b, target) / b.area >= cfg.contain_frac

    overlapping = [b for b in found if iou(b, target) >= cfg.overlap_iou or inside(b)]
    best = max((b.score for b in overlapping), default=0.0)
    if any((iou(b, target) >= cfg.yes_iou or inside(b)) and b.score >= t.yes_score for b in found):
        return "yes", best
    if best < t.no_score and max(target.width, target.height) / scale >= cfg.tiny_px:
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
        verdict, best = verdict_from_boxes(target, found, self.cfg, scale)
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

    def _shown(self, c: Candidate, image_path: Path) -> Path:
        """The crop enlarged to about image_side with the box drawn in red, written to <crop dir>/llm/<id>.png."""
        crop = cv2.imread(str(image_path))
        if crop is None:
            raise FileNotFoundError(f"{image_path} cannot be read")
        s = max(1.0, self.cfg.image_side / max(crop.shape[:2]))
        img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        ox, oy = c.crop_origin
        x1, y1, x2, y2 = c.box
        cv2.rectangle(img, (round((x1 - ox) * s), round((y1 - oy) * s)), (round((x2 - ox) * s), round((y2 - oy) * s)),
                      (0, 0, 255), 2)
        out = image_path.parent / "llm" / f"{c.id}.png"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), img)
        return out

    def _run(self, image: Path, prompt: str, cls: str) -> tuple[str | None, str]:
        """The command's stdout, or None and why it did not answer."""
        fields = {"{image}": str(image), "{cls}": cls, "{cls_zh}": CLS_ZH[cls], "{prompt}": prompt}
        argv = [_fill(a, fields) for a in self.cfg.command]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=self.cfg.timeout_s,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None, f"timeout after {self.cfg.timeout_s:g} s"
        except OSError as e:
            return None, f"cannot run: {e}"
        return done.stdout, f"exit {done.returncode}"

    def judge(self, c: Candidate, image_path: Path) -> Judgment:
        fields = {"{image}": "", "{cls}": c.cls, "{cls_zh}": CLS_ZH[c.cls]}
        out, why = self._run(self._shown(c, image_path), _fill(self.cfg.prompt, fields), c.cls)
        for text in reversed(re.findall(r"\{[^{}]*\}", out or "")):
            try:
                answer = json.loads(text)
            except ValueError:
                continue
            if isinstance(answer, dict) and answer.get("verdict") in ("yes", "no", "unsure"):
                return Judgment(c.id, answer["verdict"], self.name, None, str(answer.get("reason", ""))[:200])
        return Judgment(c.id, "unsure", self.name, None, why if out is None else f"no verdict in the output ({why})")

    def _grid(self, cands: list[Candidate], image_paths: list[Path]) -> Path:
        """The crops on one image, 4 per row, each enlarged to fit a tile x tile cell with its box in red and its
        number (1, 2, ...) in yellow at the top left; written to <crop dir>/llm/batch_<first id>.png."""
        t = self.cfg.tile
        cols = min(4, len(cands))
        rows = (len(cands) + cols - 1) // cols
        grid = np.full((rows * t, cols * t, 3), 40, np.uint8)
        for k, (c, path) in enumerate(zip(cands, image_paths)):
            crop = cv2.imread(str(path))
            if crop is None:
                raise FileNotFoundError(f"{path} cannot be read")
            s = (t - 4) / max(crop.shape[:2])
            img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            ox, oy = c.crop_origin
            x1, y1, x2, y2 = c.box
            cv2.rectangle(img, (round((x1 - ox) * s), round((y1 - oy) * s)), (round((x2 - ox) * s), round((y2 - oy) * s)),
                          (0, 0, 255), 2)
            r, q = divmod(k, cols)
            grid[r * t + 2 : r * t + 2 + img.shape[0], q * t + 2 : q * t + 2 + img.shape[1]] = img
            for color, width in (((0, 0, 0), 5), ((0, 255, 255), 2)):
                cv2.putText(grid, str(k + 1), (q * t + 8, r * t + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, width)
        out = image_paths[0].parent / "llm" / f"batch_{cands[0].id}.png"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), grid)
        return out

    def judge_batch(self, cands: list[Candidate], image_paths: list[Path]) -> list[Judgment]:
        """One verdict per candidate, in order: one call on a numbered grid when the config's batch is above 1 (and
        there is more than one), else one call each. A number the answer leaves out is "unsure". Raises ValueError
        for a batch of several classes."""
        if self.cfg.batch <= 1 or len(cands) == 1:
            return [self.judge(c, p) for c, p in zip(cands, image_paths)]
        if len({c.cls for c in cands}) != 1:
            raise ValueError("a batch for the LLM judge must hold one class")
        cls = cands[0].cls
        prompt = _fill(self.cfg.batch_prompt, {"{count}": str(len(cands)), "{cls_zh}": CLS_ZH[cls], "{cls}": cls})
        out, why = self._run(self._grid(cands, image_paths), prompt, cls)
        answers: dict[int, str] = {}
        for text in reversed(re.findall(r"\[[^\[\]]*\]", out or "")):
            try:
                rows = json.loads(text)
            except ValueError:
                continue
            if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
                answers = {int(r["n"]): r["verdict"] for r in rows
                           if isinstance(r.get("n"), int) and r.get("verdict") in ("yes", "no", "unsure")}
                break
        detail = f"grid of {len(cands)}" if answers else (why if out is None else f"no verdicts in the output ({why})")
        return [Judgment(c.id, answers.get(k + 1, "unsure"), self.name, None, detail) for k, c in enumerate(cands)]


def run_judges(cands: list[Candidate], judgments_path: Path, crops_root: Path, frame_judge: FrameTeacherJudge,
               teacher_judge: TeacherJudge, llm_judge=None, rejudge: bool = False,
               ask_on: tuple[str, ...] = ("no", "unsure"), llm_parallel: int = 1, llm_batch: int = 1) -> Counter:
    """Judges every candidate without a judgment yet (with rejudge every one people did not judge): the frame
    teacher, else the teacher; then the LLM about those the models judged one of `ask_on`, in batches of up to
    llm_batch of one class, llm_parallel calls at once (its verdict, appended last, is the final word). Appends
    the verdicts as they come, so an interrupted run keeps what it did. Returns counts of "<judge>:<verdict>"."""
    done = resolve(read_judgments(judgments_path))
    counts: Counter = Counter()
    ask: list[Candidate] = []
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
        if j.verdict in ask_on and llm_judge is not None:
            ask.append(c)
    batches = [same[i : i + llm_batch] for cls in sorted({c.cls for c in ask})
               for same in [[c for c in ask if c.cls == cls]] for i in range(0, len(same), llm_batch)]
    if batches:
        with ThreadPoolExecutor(max_workers=llm_parallel) as pool:
            for done_batch in pool.map(lambda b: llm_judge.judge_batch(b, [crops_root / c.image for c in b]), batches):
                append_judgments(judgments_path, done_batch)
                for k in done_batch:
                    counts[f"{k.judge}:{k.verdict}"] += 1
    return counts
