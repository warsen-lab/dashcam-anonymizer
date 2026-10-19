#!/usr/bin/env bash
# Batch redaction with the validated models: scripts/redact.sh <output dir> <video or directory>...
#
# Every video goes to <output dir>/<video stem>/ (see `vc_redact --help`). A directory argument takes the
# .mp4 / .mov / .mkv files directly inside it. A failed video is reported and the rest still run; the exit
# code is 1 when any video failed.
#
# Environment overrides:
#   VC_REDACT       vc_redact binary              (default build/cpp-release/vc_redact)
#   VC_MODEL        student model directory       (default data/models/r3)
#   VC_FACE_MODEL   YuNet face model directory    (default data/models/yunet; empty = student only)
#   VC_REDACT_ARGS  extra vc_redact options, e.g. "--hwaccel videotoolbox" or "--overwrite"
#   VC_SANDBOX      1 (default on macOS): run inside sandbox-exec with outbound network denied
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VC_REDACT:-$ROOT/build/cpp-release/vc_redact}"
MODEL="${VC_MODEL:-$ROOT/data/models/r3}"
FACE="${VC_FACE_MODEL-$ROOT/data/models/yunet}"
read -r -a EXTRA <<< "${VC_REDACT_ARGS:-}"
SANDBOX_PROFILE='(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'

usage() { echo "usage: $0 <output dir> <video or directory>..." >&2; exit 2; }
fail() { echo "redact.sh: $*" >&2; exit 2; }

[ $# -ge 2 ] || usage
out="$1"; shift
[ -x "$BIN" ] || fail "no vc_redact at $BIN (build it: cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && cmake --build build/cpp-release -j 4)"
for dir in "$MODEL" ${FACE:+"$FACE"}; do
  [ -s "$dir/model.onnx" ] && [ -s "$dir/model.json" ] || fail "$dir has no model.onnx + model.json"
done

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

runner=()
if [ "${VC_SANDBOX:-1}" = 1 ] && command -v sandbox-exec > /dev/null; then
  runner=(sandbox-exec -p "$SANDBOX_PROFILE")
fi

mkdir -p "$out"
failed=()
for video in "${videos[@]}"; do
  echo "=== $(date +%T) $video"
  args=(--input "$video" --out "$out" --model "$MODEL")
  [ -n "$FACE" ] && args+=(--face-model "$FACE")
  if ${runner[@]+"${runner[@]}"} "$BIN" "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"}; then
    :
  else
    status=$?
    echo "=== FAILED (exit $status): $video" >&2
    failed+=("$video")
  fi
done

echo "=== $(date +%T) done: $(( ${#videos[@]} - ${#failed[@]} )) of ${#videos[@]} redacted into $out"
if [ ${#failed[@]} -gt 0 ]; then
  printf '    failed: %s\n' "${failed[@]}" >&2
  exit 1
fi
