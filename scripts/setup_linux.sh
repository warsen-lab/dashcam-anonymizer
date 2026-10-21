#!/usr/bin/env bash
# Sets up Ubuntu 22.04 / 24.04 with an NVIDIA GPU for vc_redact and builds it with VC_WITH_CUDA=ON
# (spec 2026-10-20 §2):
#
#   scripts/setup_linux.sh                 development machine: checks the driver (>= 550), installs the apt
#                                          packages (sudo), builds FFmpeg 8 (with libx265, see below) and vc_redact
#                                          with its tests
#   scripts/setup_linux.sh --docker-deps   docker/Dockerfile's first build step (root, no GPU, apt done by the
#                                          Dockerfile): downloads and unpacks onnxruntime, nlohmann/json and CMake,
#                                          builds FFmpeg 8 without libx265. Nothing in it depends on cpp/, so the image
#                                          layer survives C++ changes
#   scripts/setup_linux.sh --docker-build  the Dockerfile's second build step, after COPY cpp/: builds vc_redact
#                                          without tests from what --docker-deps left; downloads nothing
#   scripts/setup_linux.sh --check-pins    lists the pinned downloads; exit 1 while a checksum is empty (no network)
#
# Every mode but --check-pins is part of a build that downloads, so it refuses to run unless VC_ALLOW_DOWNLOAD=1.
# Each download is checked against its sha256 in PINS and kept in the cache all checkouts share ($VC_CACHE/downloads;
# the Dockerfile mounts a BuildKit cache there); a file there whose checksum matches is not downloaded again.
#
# libx265: the C++ test fixtures make 10-bit HEVC videos with libx265 (tenBitVideo() in cpp/tests), so the
# development machine, which builds the tests, installs libx265-dev and builds FFmpeg with --enable-libx265 (the build
# is GPL already for libx264). The Docker image builds no tests and does not read HEVC encoded by us, so its FFmpeg
# is built without libx265.
#
# Environment:
#   VC_ALLOW_DOWNLOAD=1  required for anything but --check-pins
#   VC_CACHE             cache root (default ~/.cache/video-compliance)
#   VC_FFMPEG_PREFIX     FFmpeg 8 install prefix (default $VC_CACHE/linux/ffmpeg-8.1.2)
#   VC_THIRD_PARTY_DIR   onnxruntime and nlohmann/json (default $VC_CACHE/third_party, as cpp/cmake/ThirdParty.cmake)
#   VC_BUILD_DIR         vc_redact's build directory (default build/cpp-linux)
#   TENSORRT_VERSION     apt version of the TensorRT 10 runtime libraries onnxruntime 1.30.0 needs, e.g.
#                        "10.x.y.z-1+cuda12.8" (development machine only; required there)
set -euo pipefail
# A failure inside $(...) must stop the script too (bash >= 4.4; macOS's bash 3.2 only ever reaches the refusals).
shopt -s inherit_errexit 2> /dev/null || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${VC_CACHE:-$HOME/.cache/video-compliance}"
DOWNLOADS="$CACHE/downloads"
FFMPEG_VERSION=8.1.2
NV_CODEC_VERSION=12.2.72.0   # Video Codec SDK 12.2: driver >= 550 (13.x would need 570)
CMAKE_VERSION=3.31.6         # Ubuntu 22.04's cmake 3.22 is older than the 3.24 cpp/CMakeLists.txt needs
CUDA_APT_VERSION=12-8        # the CUDA of docker/Dockerfile's nvidia/cuda:12.8.1 images
ORT_PACKAGE=onnxruntime-linux-x64-gpu-1.30.0
FFMPEG_PREFIX="${VC_FFMPEG_PREFIX:-$CACHE/linux/ffmpeg-$FFMPEG_VERSION}"
THIRD_PARTY="${VC_THIRD_PARTY_DIR:-$CACHE/third_party}"
BUILD_DIR="${VC_BUILD_DIR:-$ROOT/build/cpp-linux}"
MIN_DRIVER=550

# name|url|sha256 of every download. An empty sha256 has not been looked up yet: the target-machine plan fills it
# from the publisher's checksum and commits it, and until then the script refuses to download anything. Never put
# a checksum here that was not computed from (or published for) the real file.
PINS=(
  "$ORT_PACKAGE.tgz|https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/$ORT_PACKAGE.tgz|"
  "json.hpp|https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp|aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
  # The sha256 of Homebrew's ffmpeg 8.1.2 formula (the same tarball the macOS build comes from).
  "ffmpeg-$FFMPEG_VERSION.tar.xz|https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz|464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
  "nv-codec-headers-$NV_CODEC_VERSION.tar.gz|https://github.com/FFmpeg/nv-codec-headers/releases/download/n$NV_CODEC_VERSION/nv-codec-headers-$NV_CODEC_VERSION.tar.gz|"
  "cmake-$CMAKE_VERSION-linux-x86_64.tar.gz|https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.tar.gz|"
  "cuda-keyring-ubuntu2204.deb|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb|"
  "cuda-keyring-ubuntu2404.deb|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb|"
)

die() { echo "setup_linux.sh: $*" >&2; exit 2; }
usage() { echo "usage: $0 [--docker-deps | --docker-build | --check-pins]" >&2; exit 2; }

# pin <name> <field>: field 2 is the url, 3 the sha256.
pin() {
  local entry name url sha
  for entry in "${PINS[@]}"; do
    IFS='|' read -r name url sha <<< "$entry"
    if [ "$name" = "$1" ]; then
      if [ "$2" = 2 ]; then echo "$url"; else echo "$sha"; fi
      return 0
    fi
  done
  die "no pin named $1"
}

# Lists every pin as "ok <name> <url> <sha256>" or "missing <name> <url> ..."; returns 1 when a checksum is missing
# or not 64 hex digits.
check_pins() {
  local entry name url sha bad=0
  for entry in "${PINS[@]}"; do
    IFS='|' read -r name url sha <<< "$entry"
    if [[ "$sha" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ok $name $url $sha"
    else
      echo "missing $name $url (no sha256 yet: fill it in PINS on the target machine)"
      bad=1
    fi
  done
  return "$bad"
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# fetch <name>: the path of the verified download in $DOWNLOADS, downloading it only when it is missing or wrong.
fetch() {
  local name=$1 url sha file partial
  url="$(pin "$name" 2)"
  sha="$(pin "$name" 3)"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || die "no sha256 pinned for $name (scripts/setup_linux.sh --check-pins)"
  file="$DOWNLOADS/$name"
  if [ -f "$file" ] && [ "$(sha256_of "$file")" = "$sha" ]; then
    echo "$file"
    return 0
  fi
  mkdir -p "$DOWNLOADS"
  partial="$(mktemp "$DOWNLOADS/.$name.XXXXXX")"
  echo "setup_linux.sh: downloading $url" >&2
  if ! curl -fL --retry 3 -o "$partial" "$url"; then
    rm -f "$partial"
    die "download failed: $url"
  fi
  if [ "$(sha256_of "$partial")" != "$sha" ]; then
    rm -f "$partial"
    die "sha256 of $name does not match the pin; the file was not kept"
  fi
  mv "$partial" "$file"
  echo "$file"
}

check_os() {
  [ "$(uname -s)" = Linux ] || die "Linux only (this is $(uname -s)); macOS builds as cpp/README.md describes"
  [ "$(uname -m)" = x86_64 ] || die "x86_64 only (the onnxruntime GPU package is built for it), this is $(uname -m)"
  [ -r /etc/os-release ] || die "cannot read /etc/os-release"
  # shellcheck source=/dev/null
  . /etc/os-release
  if [ "${ID:-}" != ubuntu ] || { [ "${VERSION_ID:-}" != 22.04 ] && [ "${VERSION_ID:-}" != 24.04 ]; }; then
    die "Ubuntu 22.04 or 24.04 only, this is ${PRETTY_NAME:-unknown}"
  fi
}

check_driver() {
  command -v nvidia-smi > /dev/null || die "nvidia-smi not found: install the NVIDIA driver >= $MIN_DRIVER first"
  local version major
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  major="${version%%.*}"
  if ! [[ "$major" =~ ^[0-9]+$ ]] || [ "$major" -lt "$MIN_DRIVER" ]; then
    die "NVIDIA driver '$version' is older than $MIN_DRIVER (NVENC / NVDEC of nv-codec-headers $NV_CODEC_VERSION need it)"
  fi
}

# The apt packages of the development machine: build tools, libx264, libx265, GoogleTest, and from NVIDIA's repository the
# CUDA toolkit, cuDNN 9 and the TensorRT runtime pinned to $TENSORRT_VERSION.
install_apt_packages() {
  [ -n "${TENSORRT_VERSION:-}" ] || die "TENSORRT_VERSION must name the TensorRT apt version onnxruntime 1.30.0 needs"
  local sudo=() repo keyring
  if [ "$(id -u)" != 0 ]; then sudo=(sudo); fi
  repo="ubuntu${VERSION_ID//./}"
  keyring="$(fetch "cuda-keyring-$repo.deb")"
  "${sudo[@]}" dpkg -i "$keyring"
  "${sudo[@]}" apt-get update
  local packages=(build-essential pkg-config nasm curl ca-certificates xz-utils python3 python3-venv
                  libx264-dev libx265-dev libgtest-dev "cuda-toolkit-$CUDA_APT_VERSION" libcudnn9-cuda-12
                  "libnvinfer10=$TENSORRT_VERSION" "libnvinfer-plugin10=$TENSORRT_VERSION"
                  "libnvonnxparsers10=$TENSORRT_VERSION")
  if [ "$VERSION_ID" = 24.04 ]; then packages+=(cmake); fi
  "${sudo[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
}

# The cmake to use: Ubuntu 24.04's, or on 22.04 Kitware's pinned binary unpacked into the cache.
cmake_binary() {
  if [ "${VERSION_ID:-}" = 24.04 ] && command -v cmake > /dev/null; then
    command -v cmake
    return 0
  fi
  local dir="$CACHE/linux/cmake-$CMAKE_VERSION-linux-x86_64" tarball
  if [ ! -x "$dir/bin/cmake" ]; then
    tarball="$(fetch "cmake-$CMAKE_VERSION-linux-x86_64.tar.gz")"
    mkdir -p "$CACHE/linux"
    tar -xzf "$tarball" -C "$CACHE/linux"
  fi
  echo "$dir/bin/cmake"
}

# onnxruntime's Linux GPU package and nlohmann/json, where cpp/cmake/ThirdParty.cmake looks for them.
unpack_third_party() {
  local tarball header
  mkdir -p "$THIRD_PARTY"
  if [ ! -f "$THIRD_PARTY/$ORT_PACKAGE/include/onnxruntime_cxx_api.h" ]; then
    tarball="$(fetch "$ORT_PACKAGE.tgz")"
    tar -xzf "$tarball" -C "$THIRD_PARTY"
  fi
  if [ ! -f "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann/json.hpp" ]; then
    header="$(fetch json.hpp)"
    mkdir -p "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann"
    cp "$header" "$THIRD_PARTY/nlohmann_json-3.12.0/include/nlohmann/json.hpp"
  fi
}

# FFmpeg 8 with NVENC / NVDEC (nv-codec-headers, MIT) and libx264 (GPL: the build is GPL), shared libraries
# that find each other through their RPATH. build_ffmpeg 1 adds libx265 (the test fixtures, see the header).
build_ffmpeg() {
  local with_x265=$1 encoders
  if [ -x "$FFMPEG_PREFIX/bin/ffmpeg" ]; then
    encoders="$("$FFMPEG_PREFIX/bin/ffmpeg" -hide_banner -encoders 2> /dev/null || true)"
    if grep -q h264_nvenc <<< "$encoders" && { [ "$with_x265" != 1 ] || grep -q libx265 <<< "$encoders"; }; then
      echo "setup_linux.sh: FFmpeg $FFMPEG_VERSION is already in $FFMPEG_PREFIX"
      return 0
    fi
  fi
  local src headers ffmpeg options=()
  if [ "$with_x265" = 1 ]; then options+=(--enable-libx265); fi
  src="$(mktemp -d)"
  headers="$(fetch "nv-codec-headers-$NV_CODEC_VERSION.tar.gz")"
  ffmpeg="$(fetch "ffmpeg-$FFMPEG_VERSION.tar.xz")"
  tar -xzf "$headers" -C "$src"
  make -C "$src/nv-codec-headers-$NV_CODEC_VERSION" install PREFIX="$FFMPEG_PREFIX"
  tar -xJf "$ffmpeg" -C "$src"
  (
    cd "$src/ffmpeg-$FFMPEG_VERSION"
    PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig" ./configure --prefix="$FFMPEG_PREFIX" \
      --enable-shared --disable-static --disable-doc --enable-gpl --enable-libx264 ${options[@]+"${options[@]}"} \
      --enable-ffnvcodec --enable-nvenc --enable-nvdec --enable-cuvid \
      --extra-ldflags="-Wl,-rpath,$FFMPEG_PREFIX/lib"
    make -j "$(nproc)"
    make install
  )
  rm -rf "$src"
}

# vc_redact with the CUDA code paths; tests ON on the development machine (filtered gtest groups, cpp/README.md).
build_vc_redact() {
  local tests=$1 cmake
  cmake="$(cmake_binary)"
  "$cmake" -S "$ROOT/cpp" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DVC_WITH_CUDA=ON -DVC_BUILD_TESTS="$tests" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH="$FFMPEG_PREFIX" \
    -DVC_THIRD_PARTY_DIR="$THIRD_PARTY"
  "$cmake" --build "$BUILD_DIR" -j "$(nproc)"
  echo "setup_linux.sh: built $BUILD_DIR/vc_redact"
}

mode=dev
case "${1:-}" in
  "") ;;
  --docker-deps) mode=docker-deps ;;
  --docker-build) mode=docker-build ;;
  --check-pins)
    check_pins
    exit $?
    ;;
  *) usage ;;
esac
[ $# -le 1 ] || usage

[ "${VC_ALLOW_DOWNLOAD:-}" = 1 ] || die "this downloads onnxruntime, FFmpeg and more (see PINS); set VC_ALLOW_DOWNLOAD=1 to allow it"
check_os
check_pins > /dev/null || die "some pinned downloads have no sha256 yet: run $0 --check-pins"
case "$mode" in
  dev)
    check_driver
    install_apt_packages
    unpack_third_party
    build_ffmpeg 1
    build_vc_redact ON
    ;;
  docker-deps)
    unpack_third_party
    cmake_binary > /dev/null  # unpacks Kitware's CMake now, so that --docker-build downloads nothing
    build_ffmpeg 0
    ;;
  docker-build)
    [ -x "$FFMPEG_PREFIX/bin/ffmpeg" ] && [ -f "$THIRD_PARTY/$ORT_PACKAGE/include/onnxruntime_cxx_api.h" ] ||
      die "FFmpeg or onnxruntime is missing: run $0 --docker-deps first"
    build_vc_redact OFF
    ;;
esac
