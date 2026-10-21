"""Static checks of the Linux deployment files (spec 2026-10-20 §2): docker/, scripts/setup_linux.sh.

Nothing here builds an image or downloads anything: the scripts are only run up to their refusals, and the
Dockerfile is read as text. shellcheck and hadolint run when they are installed and are skipped otherwise.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "scripts" / "setup_linux.sh"
BUILD = REPO / "docker" / "build.sh"
ENTRYPOINT = REPO / "docker" / "entrypoint.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile"
SHELL_SCRIPTS = [SETUP, BUILD, ENTRYPOINT, REPO / "scripts" / "redact.sh"]


def _run(cmd, **env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("VC_ALLOW_DOWNLOAD", "CUDA_DEVEL_DIGEST", "CUDA_RUNTIME_DIGEST",
                                                          "TENSORRT_VERSION")}
    env.update(env_extra)
    return subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=30)


def _pins():
    """(status, name, url, sha256 or None) of every pin, as setup_linux.sh --check-pins lists them."""
    done = _run([SETUP, "--check-pins"])
    assert done.returncode in (0, 1), done.stderr
    pins = []
    for line in done.stdout.splitlines():
        status, name, url, rest = line.split(" ", 3)
        pins.append((status, name, url, rest if status == "ok" else None))
    return pins


@pytest.mark.parametrize("mode", [[], ["--docker-deps"], ["--docker-build"]])
def test_setup_refuses_to_download_without_permission(tmp_path, mode):
    done = _run([SETUP, *mode], VC_CACHE=str(tmp_path))
    assert done.returncode == 2
    assert "set VC_ALLOW_DOWNLOAD=1" in done.stderr
    assert not any(tmp_path.iterdir())  # nothing was created, let alone downloaded


@pytest.mark.skipif(os.uname().sysname == "Linux", reason="checks the refusal off Linux")
def test_setup_refuses_anything_but_ubuntu_on_linux(tmp_path):
    done = _run([SETUP], VC_ALLOW_DOWNLOAD="1", VC_CACHE=str(tmp_path))
    assert done.returncode == 2 and "Linux only" in done.stderr
    assert not any(tmp_path.iterdir())


def test_setup_rejects_unknown_options():
    done = _run([SETUP, "--bogus"])
    assert done.returncode == 2 and "usage:" in done.stderr


def test_check_pins_fails_while_a_checksum_is_missing():
    pins = _pins()
    missing = [name for status, name, _, _ in pins if status == "missing"]
    assert _run([SETUP, "--check-pins"]).returncode == (1 if missing else 0)
    assert {status for status, _, _, _ in pins} <= {"ok", "missing"}


def test_pins_cover_the_downloads_over_https_with_real_checksums():
    pins = _pins()
    names = {name for _, name, _, _ in pins}
    assert {"onnxruntime-linux-x64-gpu-1.30.0.tgz", "json.hpp", "ffmpeg-8.1.2.tar.xz",
            "nv-codec-headers-12.2.72.0.tar.gz", "cmake-3.31.6-linux-x86_64.tar.gz"} <= names
    for status, name, url, sha in pins:
        assert url.startswith("https://"), name
        if status == "ok":
            assert re.fullmatch(r"[0-9a-f]{64}", sha), f"{name}: {sha!r} is not a sha256"
    known = {name: sha for status, name, _, sha in pins if status == "ok"}
    # Published checksums: cpp/README.md's for json.hpp, Homebrew's ffmpeg 8.1.2 formula for the FFmpeg tarball.
    assert known["json.hpp"] == "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
    assert known["ffmpeg-8.1.2.tar.xz"] == "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"


@pytest.mark.parametrize("env, message", [
    ({}, "set VC_ALLOW_DOWNLOAD=1"),
    ({"VC_ALLOW_DOWNLOAD": "1"}, "CUDA_DEVEL_DIGEST must be the sha256:<64 hex digits> digest"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:abc"}, "CUDA_DEVEL_DIGEST must be the sha256"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:" + "a" * 64}, "CUDA_RUNTIME_DIGEST must be the sha256"),
    ({"VC_ALLOW_DOWNLOAD": "1", "CUDA_DEVEL_DIGEST": "sha256:" + "a" * 64, "CUDA_RUNTIME_DIGEST": "sha256:" + "b" * 64},
     "TENSORRT_VERSION must name"),
])
def test_image_build_refuses_without_pinned_digests(env, message):
    done = _run([BUILD], **env)
    assert done.returncode == 2 and message in done.stderr


def test_dockerfile_pins_both_base_images_by_digest():
    text = DOCKERFILE.read_text(encoding="utf-8")
    froms = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2
    assert froms[0].startswith("FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04@${CUDA_DEVEL_DIGEST} AS build")
    assert froms[1] == "FROM nvidia/cuda:${CUDA_TAG}-cudnn-runtime-ubuntu22.04@${CUDA_RUNTIME_DIGEST}"
    assert "ARG CUDA_DEVEL_DIGEST\n" in text and "ARG CUDA_RUNTIME_DIGEST\n" in text  # no default digest
    assert "latest" not in text and "sha256:" not in text  # nothing made up in the file itself


def test_dockerfile_keeps_the_downloads_and_the_ffmpeg_build_before_the_cpp_sources():
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# syntax=docker/dockerfile:1"  # BuildKit's frontend: the cache mount needs it

    def at(prefix, containing=""):
        found = [i for i, line in enumerate(lines) if line.startswith(prefix) and containing in line]
        assert len(found) == 1, (prefix, containing, found)
        return found[0]

    setup_copy = at("COPY scripts/setup_linux.sh")
    deps = at("RUN --mount=type=cache", "")
    cpp_copy = at("COPY cpp/")
    build = at("RUN VC_ALLOW_DOWNLOAD=1 scripts/setup_linux.sh --docker-build")
    assert setup_copy < deps < cpp_copy < build
    # the downloads (and the FFmpeg build) are one step that survives a change of the C++ sources, with the
    # downloaded files in a cache that survives even a change of setup_linux.sh
    assert "target=/opt/vc-cache/downloads" in lines[deps]
    assert "scripts/setup_linux.sh --docker-deps" in "\n".join(lines[deps:deps + 2])
    assert "VC_ALLOW_DOWNLOAD=1" in "\n".join(lines[deps:deps + 2])
    assert not any(line.startswith("COPY cpp/") for line in lines[:deps])


def test_dockerfile_asks_for_the_video_capability_of_the_nvidia_container_toolkit():
    # without "video" libnvidia-encode / libnvcuvid are not mounted and NVENC / NVDEC silently fall back
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "\nENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video\n" in text


def test_setup_builds_ffmpeg_with_libx265_only_on_the_development_machine():
    text = SETUP.read_text(encoding="utf-8")
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert any("libx265-dev" in line for line in code)
    assert sum("--enable-libx265" in line for line in code) == 1
    assert "build_ffmpeg 1" in text and "build_ffmpeg 0" in text  # dev machine with x265, Docker image without


def test_dockerfile_copies_only_files_that_exist():
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            for source in line.split()[1:-1]:
                assert (REPO / source).exists(), source


def test_dockerignore_keeps_personal_data_and_builds_out_of_the_context():
    ignored = (REPO / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in ("data/", "build/", "**/build/", ".venv/", ".git/", ".superpowers/"):
        assert entry in ignored


def test_entrypoint_puts_the_input_first(tmp_path):
    fake = tmp_path / "redact.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    done = _run([ENTRYPOINT, "/in", "/out"], VC_REDACT_SH=str(fake))
    assert done.returncode == 0 and done.stdout.splitlines() == ["/out", "/in"]
    assert _run([ENTRYPOINT, "/in"], VC_REDACT_SH=str(fake)).returncode == 2


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_shell_scripts_pass_shellcheck():
    done = _run(["shellcheck", *SHELL_SCRIPTS])
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("hadolint") is None, reason="hadolint is not installed")
def test_dockerfile_passes_hadolint():
    done = _run(["hadolint", DOCKERFILE])
    assert done.returncode == 0, done.stdout + done.stderr
