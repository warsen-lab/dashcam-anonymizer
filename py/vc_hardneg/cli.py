"""python -m vc_hardneg mine | judge | review | import-human | pilot | pilot-report (spec 2026-10-21 §3)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from vc_hardneg.candidates import CANDIDATES, mine, read_candidates
from vc_hardneg.config import DEFAULT_CONFIG, load_config
from vc_hardneg.judgments import read_judgments, resolve


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vc_hardneg")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine", help="candidates from vc_redact --dump-verify-crops runs")
    m.add_argument("--run", type=Path, action="append", required=True, help="a run dir <out>/<stem>; repeat")
    m.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    m.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name, help_ in (("judge", "judge candidates with the local teachers (and the LLM with --llm)"),
                        ("pilot", "sample the pilot, judge it and write its review page")):
        j = sub.add_parser(name, help=help_)
        j.add_argument("--candidates", type=Path, required=True)
        j.add_argument("--judgments", type=Path, required=True)
        j.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        j.add_argument("--device", default="mps")
        j.add_argument("--llm", action="store_true",
                       help="ask the LLM command about what stays unsure (needs VC_HARDNEG_ALLOW_LLM=1)")
        if name == "judge":
            j.add_argument("--ids", type=Path, help="only the ids listed in this file, one per line")
            j.add_argument("--rejudge", action="store_true",
                           help="judge again what models judged before (people's verdicts stay)")
        else:
            j.add_argument("--out", type=Path, required=True)
            j.add_argument("--per-stratum", type=int, default=25)
            j.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("review", help="local review page")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--judgments", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--only", choices=("unsure", "all"), default="unsure")
    r.add_argument("--ids", type=Path)
    i = sub.add_parser("import-human", help="append a review page's export as human judgments")
    i.add_argument("--candidates", type=Path, required=True)
    i.add_argument("--judgments", type=Path, required=True)
    i.add_argument("--file", type=Path, required=True)
    pr = sub.add_parser("pilot-report", help="compare the pilot's model verdicts with the people's")
    pr.add_argument("--pilot", type=Path, required=True)
    pr.add_argument("--judgments", type=Path, required=True)
    pr.add_argument("--max-no-false", type=float, default=0.02)
    return p


def _ids(path: Path | None) -> set[str] | None:
    return None if path is None else {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _judges(args):
    """The frame-teacher, teacher and (with --llm) LLM judges. The LLM is checked first, so that a refusal comes
    before the teacher models load."""
    from vc_hardneg.judges import CommandJudge, FrameTeacherJudge, make_teacher_judge

    cfg = load_config(args.config)
    llm = CommandJudge(cfg.llm) if args.llm else None  # PermissionError without VC_HARDNEG_ALLOW_LLM=1
    from vc_label.paths import offline_model_env

    offline_model_env()
    import torch

    return FrameTeacherJudge(cfg), make_teacher_judge(cfg, torch.device(args.device)), llm


def _by_id(judgments) -> dict[str, list]:
    out = defaultdict(list)
    for j in judgments:
        out[j.id].append(j)
    return out


def _mine(args) -> int:
    s = mine(args.run, args.labels_root, args.frames_root, args.out, load_config(args.config))
    print(f"mine: {s.rows} dumped crops, {s.masked} masked on labelled frames, {s.candidates} candidates in {args.out / CANDIDATES}")
    return 0


def _judge(args) -> int:
    from vc_hardneg.judges import run_judges

    cands = read_candidates(args.candidates)
    wanted = _ids(args.ids)
    if wanted is not None:
        cands = [c for c in cands if c.id in wanted]
    frame, teacher, llm = _judges(args)
    if llm is not None:
        print(f"judge: the LLM judge may send up to {len(cands)} crops to {llm.name}", file=sys.stderr)
    llm_cfg = load_config(args.config).llm
    counts = run_judges(cands, args.judgments, args.candidates.parent, frame, teacher, llm, args.rejudge,
                        llm_cfg.ask_on, llm_cfg.parallel)
    print("judge: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing left to judge"))
    return 0


def _review(args) -> int:
    from vc_hardneg.review import write_review_page

    cands = read_candidates(args.candidates)
    judgments = read_judgments(args.judgments)
    final = resolve(judgments)
    wanted = _ids(args.ids)
    picked = [c for c in cands if (wanted is None or c.id in wanted)
              and (args.only == "all" or final.get(c.id) is None or final[c.id].verdict == "unsure")]
    page = write_review_page(picked, args.candidates.parent, args.out, _by_id(judgments), args.out.name)
    print(f"review: {len(picked)} crops in {page}")
    return 0


def _import(args) -> int:
    from vc_hardneg.review import import_human

    n = import_human(args.judgments, args.file, {c.id for c in read_candidates(args.candidates)})
    print(f"import-human: {n} verdicts appended to {args.judgments}")
    return 0


def _pilot(args) -> int:
    from vc_hardneg.judges import run_judges
    from vc_hardneg.pilot import sample_pilot
    from vc_hardneg.review import write_review_page

    cands = sample_pilot(read_candidates(args.candidates), args.per_stratum, args.seed)
    frame, teacher, llm = _judges(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_ids.txt").write_text("".join(f"{c.id}\n" for c in cands))
    llm_cfg = load_config(args.config).llm
    counts = run_judges(cands, args.judgments, args.candidates.parent, frame, teacher, llm,
                        ask_on=llm_cfg.ask_on, llm_parallel=llm_cfg.parallel)
    page = write_review_page(cands, args.candidates.parent, args.out, _by_id(read_judgments(args.judgments)), "试判 100 张")
    print(f"pilot: {len(cands)} crops; " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"; review {page}")
    return 0


def _pilot_report(args) -> int:
    from vc_hardneg.pilot import pilot_report

    ids = sorted(_ids(args.pilot / "pilot_ids.txt"))
    r = pilot_report(ids, read_judgments(args.judgments), args.max_no_false)
    (args.pilot / "report.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p = r["pipeline"]
    print(f"pilot-report: {r['judged_by_people']}/{r['ids']} judged by people; models said no {p['no']} times, "
          f"{p['no_but_real']} of them real ({p['no_but_real_rate']:.1%}, limit {args.max_no_false:.0%}): "
          + ("PASS" if r["passed"] else "FAIL"))
    return 0 if r["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"mine": _mine, "judge": _judge, "review": _review, "import-human": _import, "pilot": _pilot,
                "pilot-report": _pilot_report}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError) as e:  # PermissionError and FileNotFoundError are OSErrors
        print(f"vc_hardneg: {e}", file=sys.stderr)
        return 2
