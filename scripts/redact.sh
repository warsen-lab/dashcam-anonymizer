#!/usr/bin/env bash
# Batch redaction with the validated models: scripts/redact.sh <output dir> <video or directory>...
#
# Every video goes to <output dir>/<video stem>/ (see `vc_redact --help`). A directory argument takes the
# .mp4 / .mov / .mkv files directly inside it. A failed video is reported and the rest still run; the exit
# code is 1 when any video failed. A video whose stem an earlier one already has would share its output
# directory: it is refused (exit 3) without running. <output dir>/batch.json lists every video's result
# and the batch throughput (spec 2026-10-20 §5). Ctrl-C (SIGINT) or SIGTERM stops every running vc_redact,
# prints one line with the videos that did not finish, still writes batch.json and exits 130 / 143. A video
# that had already finished when the signal came keeps its real exit status. With a single slot (no parallel
# run) vc_redact still runs in the background of the script, with its output going straight to the terminal,
# and the script waits for it: bash runs a trap only when a foreground command ends, so this is what lets
# SIGTERM (`docker stop`, where redact.sh is PID 1) stop the batch at once. A background child ignores
# Ctrl-C, so the script's own INT trap passes SIGTERM on to it.
#
# Environment overrides:
#   VC_REDACT       vc_redact binary              (default build/cpp-release/vc_redact)
#   VC_MODEL        student model directory       (default data/models/r3)
#   VC_FACE_MODEL   YuNet face model directory    (default data/models/yunet; empty = student only)
#   VC_VERIFIER     verifier model directory      (default data/models/verifier-v1 when it exists; empty = none)
#   VC_DETECT_FPS   detections per second, 5 to 60 (default: detect_fps of configs/redact.json, 10)
#   VC_REDACT_ARGS  extra vc_redact options, e.g. "--hwaccel videotoolbox" or "--overwrite"
#   VC_GPUS         GPUs to use, e.g. "0,1": videos go to them in turn, each vc_redact runs with
#                   CUDA_VISIBLE_DEVICES set to its GPU (default: none; CUDA_VISIBLE_DEVICES is left alone)
#   VC_JOBS         vc_redact processes at once per GPU of VC_GPUS, or in all without VC_GPUS
#                   (default 4 per GPU, 1 without VC_GPUS); 1 to 64
#   VC_PYTHON       python3 that writes batch.json (default python3)
#   VC_SANDBOX      1 (default on macOS): run inside sandbox-exec with outbound network denied
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VC_REDACT:-$ROOT/build/cpp-release/vc_redact}"
MODEL="${VC_MODEL:-$ROOT/data/models/r3}"
FACE="${VC_FACE_MODEL-$ROOT/data/models/yunet}"
if [ "${VC_VERIFIER+set}" = set ]; then
  VERIFIER="$VC_VERIFIER"
elif [ -d "$ROOT/data/models/verifier-v1" ]; then
  VERIFIER="$ROOT/data/models/verifier-v1"
else
  VERIFIER=""
fi
DETECT_FPS="${VC_DETECT_FPS:-}"
GPUS_TEXT="${VC_GPUS:-}"
PYTHON="${VC_PYTHON:-python3}"
read -r -a EXTRA <<< "${VC_REDACT_ARGS:-}"
SANDBOX_PROFILE='(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'
MAX_JOBS=64
POLL_SECONDS=0.2

usage() { echo "usage: $0 <output dir> <video or directory>..." >&2; exit 2; }
fail() { echo "redact.sh: $*" >&2; exit 2; }

[ $# -ge 2 ] || usage
out="$1"; shift
[ -x "$BIN" ] || fail "no vc_redact at $BIN (build it: cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && cmake --build build/cpp-release -j 4)"
for dir in "$MODEL" ${FACE:+"$FACE"} ${VERIFIER:+"$VERIFIER"}; do
  [ -s "$dir/model.onnx" ] && [ -s "$dir/model.json" ] || fail "$dir has no model.onnx + model.json"
done

gpus=()
if [ -n "$GPUS_TEXT" ]; then
  IFS=',' read -r -a gpus <<< "$GPUS_TEXT"
  for gpu in ${gpus[@]+"${gpus[@]}"}; do
    [[ "$gpu" =~ ^[A-Za-z0-9-]+$ ]] || fail "VC_GPUS must list GPU indices or UUIDs separated by commas, got '$GPUS_TEXT'"
  done
fi
n_gpus=${#gpus[@]}
if [ "$n_gpus" -gt 0 ]; then default_jobs=4; else default_jobs=1; fi
JOBS="${VC_JOBS:-$default_jobs}"
if ! [[ "$JOBS" =~ ^[1-9][0-9]?$ ]] || [ "$JOBS" -gt "$MAX_JOBS" ]; then
  fail "VC_JOBS must be an integer in [1, $MAX_JOBS], got '$JOBS'"
fi
if [ "$n_gpus" -gt 0 ]; then total_slots=$((JOBS * n_gpus)); else total_slots=$JOBS; fi
command -v "$PYTHON" > /dev/null || fail "$PYTHON is needed to write batch.json (set VC_PYTHON)"

videos=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r -d '' f; do videos+=("$f"); done < <(
      find "$arg" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) -print0 | sort -z)
  elif [ -f "$arg" ]; then
    videos+=("$arg")
  else
    fail "$arg is neither a file nor a directory"
  fi
done
[ ${#videos[@]} -gt 0 ] || fail "no .mp4 / .mov / .mkv files found"
for video in "${videos[@]}"; do
  case "$video" in
    *$'\t'* | *$'\n'*) fail "a video path holds a tab or a newline: $video" ;;
  esac
done

runner=()
if [ "${VC_SANDBOX:-1}" = 1 ] && command -v sandbox-exec > /dev/null; then
  runner=(sandbox-exec -p "$SANDBOX_PROFILE")
fi

mkdir -p "$out"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/redact.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
started="$("$PYTHON" -c 'import time; print(f"{time.time():.3f}")')"
: > "$tmp/results.tsv"

failed=()
pids=()      # by job number, while a background job runs
job_video=()
job_gpu=()   # the GPU (as listed in VC_GPUS), empty without VC_GPUS
job_slot=()  # index into gpus, -1 without VC_GPUS
on_gpu=()    # jobs running per index into gpus
for ((g = 0; g < n_gpus; g++)); do on_gpu[g]=0; done
running=0
recorded=0
fg_job=""    # the job that runs alone (a single slot), while it runs
fg_pid=""    # its process
fg_status="" # its exit status, from when it has ended until it is recorded
done_job=()  # by job number: 1 once the job is recorded (a signal may come between that and the end of its clean-up)
next_gpu=0

# One line per video for batch.json: input position, exit status, GPU, path.
record() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$tmp/results.tsv"
  done_job[$1]=1
  recorded=$((recorded + 1))
}

write_batch_json() {
  "$PYTHON" -I "$ROOT/scripts/batch_json.py" --out "$out" --results "$tmp/results.tsv" --started "$started" \
    --jobs "$JOBS" --gpus "$GPUS_TEXT"
}

# For the subshell of a job: vc_redact sees only its GPU.
use_gpu() { if [ -n "$1" ]; then export CUDA_VISIBLE_DEVICES="$1"; fi; }

ARGS=()
build_args() {
  ARGS=(--input "$1" --out "$out" --model "$MODEL")
  if [ -n "$FACE" ]; then ARGS+=(--face-model "$FACE"); fi
  if [ -n "$VERIFIER" ]; then ARGS+=(--verifier-model "$VERIFIER"); fi
  if [ -n "$DETECT_FPS" ]; then ARGS+=(--detect-fps "$DETECT_FPS"); fi
}

# Records one ended job: prints its log (parallel runs only), lists a failure and frees its slot.
collect() {
  local job=$1 status=$2
  if [ "$total_slots" -gt 1 ] && [ -f "$tmp/$job.log" ]; then cat "$tmp/$job.log"; fi
  if [ "$status" != 0 ]; then
    echo "=== FAILED (exit $status): ${job_video[job]}" >&2
    failed+=("${job_video[job]}")
  fi
  record "$job" "$status" "${job_gpu[job]}" "${job_video[job]}"
  running=$((running - 1))
  if [ "${job_slot[job]}" -ge 0 ]; then on_gpu[job_slot[job]]=$((on_gpu[job_slot[job]] - 1)); fi
}

# With a single slot vc_redact runs in the background of the script, not as a foreground command, and the script
# `wait`s for it: a signal interrupts `wait` and its trap runs at once, whereas bash would hold the trap back until
# a foreground command ended. The output is not captured, so it reaches the terminal as it is written.
run_foreground() {
  local job=$1 video=$2 gpu=$3 status=0
  build_args "$video"
  echo "=== $(date +%T) ${gpu:+[gpu $gpu] }$video"
  fg_job=$job  # before the launch, so a signal right after it still finds the job
  (
    use_gpu "$gpu"
    exec ${runner[@]+"${runner[@]}"} "$BIN" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
  ) &
  fg_pid=$!
  wait "$fg_pid" || status=$?
  fg_status=$status
  collect "$job" "$status"
  fg_job=""
  fg_pid=""
  fg_status=""
}

# Starts vc_redact on `video` in the background; its output goes to a log that is printed when it ends, so
# that the runs do not interleave. A background job of a script ignores SIGINT (and so would its vc_redact),
# so the job's subshell passes SIGTERM on to vc_redact, which is how abort_batch stops them.
launch() {
  local job=$1 video=$2 gpu=$3
  build_args "$video"
  echo "=== $(date +%T) ${gpu:+[gpu $gpu] }$video"
  (
    terminated=0
    child=""
    # shellcheck disable=SC2329  # runs as the TERM trap
    on_term() {
      terminated=1
      if [ -n "$child" ]; then kill -TERM "$child" 2> /dev/null || true; fi
    }
    trap on_term TERM
    use_gpu "$gpu"
    exec > "$tmp/$job.log" 2>&1
    ${runner[@]+"${runner[@]}"} "$BIN" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} &
    child=$!
    if [ "$terminated" = 1 ]; then kill -TERM "$child" 2> /dev/null || true; fi
    status=0
    wait "$child" || status=$?
    while kill -0 "$child" 2> /dev/null; do wait "$child" || status=$?; done
    echo "$status" > "$tmp/$job.status.partial"
    mv "$tmp/$job.status.partial" "$tmp/$job.status"
  ) &
  pids[job]=$!
}

# Collects a background job that wrote its status.
finish() {
  local job=$1 status
  status="$(cat "$tmp/$job.status")"
  wait "${pids[job]}" || true
  collect "$job" "$status"
  unset 'pids[job]'  # after the record: a signal in between still finds the job in abort_batch
}

# Waits until a background job ends and collects it. A job whose subshell is gone without a status (it was
# killed, or could not write the file) is a failure, not something to wait for.
reap_one() {
  local job status
  while :; do
    for job in "${!pids[@]}"; do
      if [ -f "$tmp/$job.status" ]; then
        finish "$job"
        return
      fi
      if ! kill -0 "${pids[job]}" 2> /dev/null; then
        if [ -f "$tmp/$job.status" ]; then  # renamed just before the subshell exited
          finish "$job"
          return
        fi
        status=0
        wait "${pids[job]}" || status=$?
        if [ "$status" = 0 ]; then status=70; fi
        echo "redact.sh: the job for ${job_video[job]} ended without reporting (exit $status)" >&2
        collect "$job" "$status"
        unset 'pids[job]'
        return
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

# SIGINT / SIGTERM: stops the running jobs, writes batch.json for what is known and exits 130 / 143.
abort_batch() {
  local code=$1 job status names="" rest inflight
  trap '' INT TERM
  inflight=()
  if [ ${#pids[@]} -gt 0 ]; then inflight=("${!pids[@]}"); fi
  for job in ${inflight[@]+"${inflight[@]}"}; do
    if [ -z "${done_job[job]:-}" ]; then kill -TERM "${pids[job]}" 2> /dev/null || true; fi
  done
  if [ -n "$fg_pid" ] && [ -z "$fg_status" ]; then kill -TERM "$fg_pid" 2> /dev/null || true; fi
  for job in ${inflight[@]+"${inflight[@]}"}; do
    if [ -n "${done_job[job]:-}" ]; then continue; fi
    wait "${pids[job]}" 2> /dev/null || true
    status="$code"
    if [ -f "$tmp/$job.status" ]; then status="$(cat "$tmp/$job.status")"; fi
    if [ "$status" != 0 ]; then names="${names:+$names, }${job_video[job]}"; fi
    if [ "$total_slots" -gt 1 ] && [ -f "$tmp/$job.log" ]; then cat "$tmp/$job.log"; fi
    record "$job" "$status" "${job_gpu[job]}" "${job_video[job]}"
  done
  if [ -n "$fg_job" ] && [ -z "${done_job[fg_job]:-}" ]; then
    # Its real status: a video that finished before the signal stays a success; the TERM above ends the others.
    status="$fg_status"
    if [ -z "$status" ] && [ -z "$fg_pid" ]; then
      status="$code"  # the signal came before its pid was known
    elif [ -z "$status" ]; then
      status=0
      wait "$fg_pid" 2> /dev/null || status=$?
      if [ "$status" = 127 ]; then status="$code"; fi  # not a child of ours any more
    fi
    if [ "$status" != 0 ]; then names="${names:+$names, }${job_video[fg_job]}"; fi
    record "$fg_job" "$status" "${job_gpu[fg_job]}" "${job_video[fg_job]}"
  fi
  rest="$(( ${#videos[@]} - recorded )) not started"
  echo "=== interrupted: ${names:+did not finish: $names; }$rest" >&2
  write_batch_json || echo "redact.sh: could not write $out/batch.json" >&2
  exit "$code"
}
trap 'abort_batch 130' INT
trap 'abort_batch 143' TERM

# The index of the next GPU in turn with a free slot (there is one whenever fewer than total_slots run).
free_gpu() {
  local k g
  for ((k = 0; k < n_gpus; k++)); do
    g=$(((next_gpu + k) % n_gpus))
    if [ "${on_gpu[g]}" -lt "$JOBS" ]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

seen=()
for ((job = 0; job < ${#videos[@]}; job++)); do
  video="${videos[job]}"
  stem="$(basename "$video")"
  stem="${stem%.*}"
  duplicate=0
  for s in ${seen[@]+"${seen[@]}"}; do
    if [ "$s" = "$stem" ]; then duplicate=1; fi
  done
  if [ "$duplicate" = 1 ]; then
    echo "=== FAILED (exit 3): $video (an earlier video already writes $out/$stem)" >&2
    failed+=("$video")
    record "$job" 3 "" "$video"
    continue
  fi
  seen+=("$stem")
  while [ "$running" -ge "$total_slots" ]; do reap_one; done
  gpu=""
  slot=-1
  if [ "$n_gpus" -gt 0 ]; then
    slot="$(free_gpu)"
    gpu="${gpus[slot]}"
    on_gpu[slot]=$((on_gpu[slot] + 1))
    next_gpu=$(((slot + 1) % n_gpus))
  fi
  job_video[job]="$video"
  job_gpu[job]="$gpu"
  job_slot[job]=$slot
  running=$((running + 1))
  if [ "$total_slots" -gt 1 ]; then launch "$job" "$video" "$gpu"; else run_foreground "$job" "$video" "$gpu"; fi
done
while [ "$running" -gt 0 ]; do reap_one; done

echo "=== $(date +%T) done: $(( ${#videos[@]} - ${#failed[@]} )) of ${#videos[@]} redacted into $out"
if ! write_batch_json; then
  echo "redact.sh: could not write $out/batch.json" >&2
  exit 1
fi
if [ ${#failed[@]} -gt 0 ]; then
  printf '    failed: %s\n' "${failed[@]}" >&2
  exit 1
fi
