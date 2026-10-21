#!/usr/bin/env bash
# The image's entry point: `docker run ... vc-redact <input> <output>` runs scripts/redact.sh <output> <input>
# (spec 2026-10-20 §2 puts the input first). VC_JOBS, VC_GPUS and the other redact.sh variables pass through -e.
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: docker run --gpus all -v <videos>:/in -v <output>:/out -v <models>:/models -v <cache>:/cache vc-redact /in /out" >&2
  exit 2
fi
exec "${VC_REDACT_SH:-/opt/vc/scripts/redact.sh}" "$2" "$1"
