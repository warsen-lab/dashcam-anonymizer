"""Writes <out>/batch.json for scripts/redact.sh (spec 2026-10-20 §5): every video's result and the batch throughput.

Standard library only: it runs wherever redact.sh does (the Docker image, a bare Linux machine, macOS).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Copied from each successful run's stats.json as they are (None when the field is missing).
STATS_FIELDS = ("detector", "encoder", "pipeline", "rate_control")


def read_results(path: Path) -> list[tuple[int, int, str | None, str]]:
    """redact.sh's lines "<position>\\t<exit status>\\t<gpu>\\t<video>", in input order."""
    rows = []
    # Split on "\n" only (the file's separator): a path may hold \r, U+0085 or U+2028, which splitlines() and
    # universal newlines would cut. surrogateescape keeps a path that is not valid UTF-8.
    for line in path.read_bytes().decode("utf-8", "surrogateescape").split("\n"):
        if not line:
            continue
        position, status, gpu, video = line.split("\t", 3)
        rows.append((int(position), int(status), gpu or None, video))
    return sorted(rows)


def video_entry(out: Path, status: int, gpu: str | None, video: str) -> dict:
    """One video's entry; the stats of a failed run stay None (exit 3 leaves an earlier run's stats.json there)."""
    output_dir = out / Path(video).stem
    entry = {
        "video": video, "output_dir": str(output_dir), "gpu": gpu, "exit_status": status,
        "frames": None, "elapsed_s": None, "fps": None, **{name: None for name in STATS_FIELDS},
    }
    if status != 0:
        return entry
    try:
        stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        entry["error"] = f"cannot read stats.json: {exc}"
        return entry
    frames, elapsed = stats.get("frames"), stats.get("elapsed_s")
    entry["frames"], entry["elapsed_s"] = frames, elapsed
    entry["fps"] = frames / elapsed if frames and elapsed else None
    entry.update({name: stats.get(name) for name in STATS_FIELDS})
    return entry


def batch_summary(entries: list[dict], wall_s: float, jobs: int, gpus: list[str]) -> dict:
    """The whole batch: frames of the successful runs over the batch's wall time is its throughput."""
    ok = [e for e in entries if e["exit_status"] == 0]
    frames = sum(e["frames"] or 0 for e in ok)
    return {
        "jobs": jobs,
        "gpus": gpus,
        "processes_at_once": jobs * max(1, len(gpus)),
        "wall_s": wall_s,
        "succeeded": len(ok),
        "failed": len(entries) - len(ok),
        "frames": frames,
        "throughput_fps": frames / wall_s if wall_s > 0 else None,
        "videos": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--started", type=float, required=True, help="epoch seconds when the batch started")
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--gpus", default="")
    args = parser.parse_args(argv)
    gpus = [g for g in args.gpus.split(",") if g]
    entries = [video_entry(args.out, status, gpu, video) for _, status, gpu, video in read_results(args.results)]
    summary = batch_summary(entries, time.time() - args.started, args.jobs, gpus)
    partial = args.out / "batch.json.partial"
    partial.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, args.out / "batch.json")
    fps = summary["throughput_fps"]
    print(f"=== throughput: {fps or 0:.1f} fps ({summary['frames']} frames from {summary['succeeded']} video(s) "
          f"in {summary['wall_s']:.1f} s, {summary['processes_at_once']} at once); {args.out / 'batch.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
