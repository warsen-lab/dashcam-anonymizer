"""scripts/redact.sh with a fake vc_redact: parallel jobs, GPUs in turn, failures and batch.json (spec 2026-10-20 §5)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "redact.sh"

# Stands in for vc_redact: notes how many runs are going on and which GPU it was given, can wait until
# FAKE_BARRIER runs go on at once (exit 9 after 5 s), fails for stems starting with "bad" and otherwise writes
# a stats.json with 120 frames in 2 s.
FAKE = r"""#!/usr/bin/env bash
set -euo pipefail
while [ $# -gt 0 ]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
stem="$(basename "$input")"; stem="${stem%.*}"
state="$FAKE_STATE"
mkdir -p "$state/running"
: > "$state/running/$stem"
ls "$state/running" | wc -l | tr -d ' ' >> "$state/concurrency"
echo "$stem ${CUDA_VISIBLE_DEVICES-unset}" >> "$state/gpus"
echo "fake vc_redact on $stem"
if [ -n "${FAKE_BARRIER:-}" ]; then
  for _ in $(seq 50); do
    if [ "$(ls "$state/running" | wc -l | tr -d ' ')" -ge "$FAKE_BARRIER" ]; then break; fi
    sleep 0.1
  done
  if [ "$(ls "$state/running" | wc -l | tr -d ' ')" -lt "$FAKE_BARRIER" ]; then echo "barrier timeout" >&2; exit 9; fi
fi
sleep "${FAKE_SLEEP:-0}"
rm -f "$state/running/$stem"
case "$stem" in bad*) echo "fake failure on $stem" >&2; exit 1 ;; esac
mkdir -p "$out/$stem"
printf '{"frames": 120, "elapsed_s": 2.0, "detector": "student/CPU", "encoder": "libx264", "pipeline": "cpu-frames"}\n' \
  > "$out/$stem/stats.json"
"""


@pytest.fixture
def batch(tmp_path):
    """Runs redact.sh on named videos (created as empty files) with the fake; returns (completed process, out dir, state dir)."""
    fake = tmp_path / "vc_redact"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_text("x")
    (model / "model.json").write_text("{}")
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "out"

    def run(names, **env_extra):
        videos = []
        for name in names:
            path = tmp_path / "in" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            videos.append(str(path))
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "VC_GPUS", "VC_JOBS")}
        env.update(VC_REDACT=str(fake), VC_MODEL=str(model), VC_FACE_MODEL="", VC_SANDBOX="0",
                   VC_PYTHON=sys.executable, FAKE_STATE=str(state), **env_extra)
        done = subprocess.run([str(SCRIPT), str(out), *videos], env=env, capture_output=True, text=True, timeout=60)
        return done, out, state

    return run


def _batch_json(out):
    return json.loads((out / "batch.json").read_text(encoding="utf-8"))


def _gpu_of(state):
    return dict(line.split() for line in (state / "gpus").read_text().splitlines())


def _concurrency(state):
    return [int(n) for n in (state / "concurrency").read_text().split()]


def test_runs_every_video_in_turn_and_writes_batch_json(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4"])
    assert done.returncode == 0, done.stderr
    assert max(_concurrency(state)) == 1  # VC_JOBS defaults to 1 without VC_GPUS
    assert _gpu_of(state) == {"a": "unset", "b": "unset", "c": "unset"}
    b = _batch_json(out)
    assert [Path(v["video"]).name for v in b["videos"]] == ["a.mp4", "b.mp4", "c.mp4"]
    for v in b["videos"]:
        assert v["exit_status"] == 0 and v["gpu"] is None
        assert v["frames"] == 120 and v["elapsed_s"] == 2.0 and v["fps"] == 60.0
        assert (v["detector"], v["encoder"], v["pipeline"]) == ("student/CPU", "libx264", "cpu-frames")
        assert v["output_dir"] == str(out / Path(v["video"]).stem)
    assert (b["jobs"], b["gpus"], b["processes_at_once"]) == (1, [], 1)
    assert (b["succeeded"], b["failed"], b["frames"]) == (3, 0, 360)
    assert b["wall_s"] > 0 and b["throughput_fps"] == pytest.approx(360 / b["wall_s"])
    assert "done: 3 of 3 redacted" in done.stdout and "=== throughput:" in done.stdout


def test_a_failed_video_is_reported_and_the_rest_still_run(batch):
    done, out, _ = batch(["a.mp4", "bad.mp4", "c.mp4"])
    assert done.returncode == 1
    assert "=== FAILED (exit 1): " in done.stderr and "bad.mp4" in done.stderr
    assert "failed: " in done.stderr
    b = _batch_json(out)
    by_name = {Path(v["video"]).name: v for v in b["videos"]}
    assert by_name["bad.mp4"]["exit_status"] == 1
    assert by_name["bad.mp4"]["frames"] is None and by_name["bad.mp4"]["fps"] is None
    assert by_name["c.mp4"]["exit_status"] == 0
    assert (b["succeeded"], b["failed"], b["frames"]) == (2, 1, 240)


def test_vc_jobs_runs_that_many_at_once(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4"], VC_JOBS="3", FAKE_BARRIER="3")
    assert done.returncode == 0, done.stdout + done.stderr  # the barrier needs all three running together
    assert max(_concurrency(state)) == 3
    assert _batch_json(out)["processes_at_once"] == 3
    for stem in ("a", "b", "c"):  # each run's output is printed once, in one piece
        assert done.stdout.count(f"fake vc_redact on {stem}\n") == 1


def test_vc_gpus_hands_the_videos_out_in_turn(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4", "d.mp4"], VC_GPUS="0,1", VC_JOBS="2", FAKE_BARRIER="4")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _gpu_of(state) == {"a": "0", "b": "1", "c": "0", "d": "1"}
    b = _batch_json(out)
    assert [v["gpu"] for v in b["videos"]] == ["0", "1", "0", "1"]
    assert (b["jobs"], b["gpus"], b["processes_at_once"]) == (2, ["0", "1"], 4)
    assert "[gpu 1] " in done.stdout


def test_no_gpu_runs_more_than_vc_jobs_at_once(batch):
    done, out, state = batch(["a.mp4", "b.mp4", "c.mp4", "d.mp4"], VC_GPUS="0,1", VC_JOBS="1", FAKE_SLEEP="0.3")
    assert done.returncode == 0, done.stderr
    assert max(_concurrency(state)) <= 2
    gpus = list(_gpu_of(state).values())
    assert sorted(gpus) == ["0", "0", "1", "1"]


def test_vc_gpus_defaults_to_four_jobs_per_gpu(batch):
    done, out, _ = batch(["a.mp4"], VC_GPUS="0,1")
    assert done.returncode == 0, done.stderr
    assert (_batch_json(out)["jobs"], _batch_json(out)["processes_at_once"]) == (4, 8)


def test_a_second_video_with_the_same_stem_is_refused_without_running(batch, tmp_path):
    done, out, state = batch(["x/a.mp4", "y/a.mp4"])
    assert done.returncode == 1
    assert "=== FAILED (exit 3): " in done.stderr and "y/a.mp4" in done.stderr
    assert len((state / "gpus").read_text().splitlines()) == 1  # vc_redact ran once
    assert [v["exit_status"] for v in _batch_json(out)["videos"]] == [0, 3]


def test_cuda_visible_devices_is_left_alone_without_vc_gpus(batch):
    done, _, state = batch(["a.mp4"], CUDA_VISIBLE_DEVICES="7")
    assert done.returncode == 0, done.stderr
    assert _gpu_of(state) == {"a": "7"}


@pytest.mark.parametrize("env, message", [
    ({"VC_JOBS": "0"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_JOBS": "65"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_JOBS": "two"}, "VC_JOBS must be an integer in [1, 64]"),
    ({"VC_GPUS": "0,,1"}, "VC_GPUS must list GPU indices or UUIDs"),
    ({"VC_GPUS": "0;1"}, "VC_GPUS must list GPU indices or UUIDs"),
])
def test_bad_vc_jobs_and_vc_gpus_are_refused_before_running(batch, env, message):
    done, out, state = batch(["a.mp4"], **env)
    assert done.returncode == 2 and message in done.stderr
    assert not (state / "gpus").exists() and not (out / "batch.json").exists()
