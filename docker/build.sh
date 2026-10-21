#!/usr/bin/env bash
# Builds the vc-redact image from docker/Dockerfile (spec 2026-10-20 §2) with its base images pinned by digest:
#
#   VC_ALLOW_DOWNLOAD=1 CUDA_DEVEL_DIGEST=sha256:<64 hex> CUDA_RUNTIME_DIGEST=sha256:<64 hex> \
#     TENSORRT_VERSION=<apt version> docker/build.sh [tag]
#
# The digests are those of nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 and nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
# (`docker buildx imagetools inspect <image>` prints them) and TENSORRT_VERSION the apt version of the TensorRT 10
# runtime onnxruntime 1.30.0 needs; the target-machine plan looks them up and records them. Building downloads the
# base images and the pinned packages, so it also needs VC_ALLOW_DOWNLOAD=1. The default tag is vc-redact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tag="${1:-vc-redact}"
fail() { echo "docker/build.sh: $*" >&2; exit 2; }

[ $# -le 1 ] || fail "usage: $0 [tag]"
[ "${VC_ALLOW_DOWNLOAD:-}" = 1 ] || fail "building the image downloads base images and packages; set VC_ALLOW_DOWNLOAD=1 to allow it"
for name in CUDA_DEVEL_DIGEST CUDA_RUNTIME_DIGEST; do
  value="${!name:-}"
  [[ "$value" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "$name must be the sha256:<64 hex digits> digest of its base image, got '$value'"
done
[ -n "${TENSORRT_VERSION:-}" ] || fail "TENSORRT_VERSION must name the TensorRT apt version onnxruntime 1.30.0 needs"
"$ROOT/scripts/setup_linux.sh" --check-pins > /dev/null || fail "some pinned downloads have no sha256 yet: run scripts/setup_linux.sh --check-pins"
command -v docker > /dev/null || fail "docker not found"
exec docker build -f "$ROOT/docker/Dockerfile" \
  --build-arg "CUDA_DEVEL_DIGEST=$CUDA_DEVEL_DIGEST" --build-arg "CUDA_RUNTIME_DIGEST=$CUDA_RUNTIME_DIGEST" \
  --build-arg "TENSORRT_VERSION=$TENSORRT_VERSION" -t "$tag" "$ROOT"
