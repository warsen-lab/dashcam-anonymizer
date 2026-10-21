# syntax=docker/dockerfile:1
# vc_redact for Linux + NVIDIA (spec 2026-10-20 §2): a build stage on the CUDA devel image, a runtime stage on the
# CUDA runtime image with cuDNN and the TensorRT runtime. Build it with docker/build.sh, which passes the base image
# digests (looked up on the target machine) and refuses without them. Run:
#   docker run --gpus all -v <videos>:/in -v <output>:/out -v <models>:/models -v <cache>:/cache vc-redact /in /out
# Models (/models/r3, /models/yunet) and caches (/cache: TensorRT engines) stay outside the image.
ARG CUDA_TAG=12.8.1
ARG CUDA_DEVEL_DIGEST
ARG CUDA_RUNTIME_DIGEST

FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04@${CUDA_DEVEL_DIGEST} AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config nasm curl ca-certificates xz-utils \
      libx264-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
ENV VC_CACHE=/opt/vc-cache VC_FFMPEG_PREFIX=/opt/ffmpeg VC_THIRD_PARTY_DIR=/opt/third_party VC_BUILD_DIR=/opt/vc/build
# Layer order: what a C++ change does not touch comes first. Only setup_linux.sh is copied before the downloads and the
# FFmpeg build, so editing cpp/ rebuilds vc_redact alone; the downloads also sit in a BuildKit cache mount, which
# survives even a change of setup_linux.sh (and its pins). Building the image is the consent to its downloads (all of
# them pinned by sha256 in setup_linux.sh).
COPY scripts/setup_linux.sh scripts/
RUN --mount=type=cache,id=vc-downloads,target=/opt/vc-cache/downloads \
    VC_ALLOW_DOWNLOAD=1 scripts/setup_linux.sh --docker-deps
COPY cpp/ cpp/
RUN VC_ALLOW_DOWNLOAD=1 scripts/setup_linux.sh --docker-build

FROM nvidia/cuda:${CUDA_TAG}-cudnn-runtime-ubuntu22.04@${CUDA_RUNTIME_DIGEST}
ARG TENSORRT_VERSION
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ -z "${TENSORRT_VERSION}" ]; then echo "TENSORRT_VERSION is required (docker/build.sh)" >&2; exit 1; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends libx264-163 python3 python3-numpy \
      "libnvinfer10=${TENSORRT_VERSION}" "libnvinfer-plugin10=${TENSORRT_VERSION}" \
      "libnvonnxparsers10=${TENSORRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*
# The same paths as in the build stage: vc_redact and FFmpeg find their libraries through their RPATH.
COPY --from=build /opt/ffmpeg/ /opt/ffmpeg/
COPY --from=build /opt/third_party/onnxruntime-linux-x64-gpu-1.30.0/lib/ /opt/third_party/onnxruntime-linux-x64-gpu-1.30.0/lib/
COPY --from=build /opt/vc/build/vc_redact /opt/vc/bin/vc_redact
COPY scripts/redact.sh scripts/batch_json.py /opt/vc/scripts/
COPY configs/ /opt/vc/configs/
COPY py/vc_eval/ /opt/vc/py/vc_eval/
COPY py/vc_label/__init__.py py/vc_label/contracts.py /opt/vc/py/vc_label/
COPY docker/entrypoint.sh /opt/vc/entrypoint.sh
# NVENC and NVDEC need libnvidia-encode and libnvcuvid, which the NVIDIA container toolkit mounts only with the
# "video" capability; without it FFmpeg silently falls back to libx264 and software decoding.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV VC_REDACT=/opt/vc/bin/vc_redact VC_MODEL=/models/r3 VC_FACE_MODEL=/models/yunet VC_CACHE=/cache VC_SANDBOX=0 \
    PATH=/opt/ffmpeg/bin:$PATH PYTHONPATH=/opt/vc/py
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/opt/vc/entrypoint.sh"]
