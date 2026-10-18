"""The offline environment must be in place before anything can import ultralytics (no ultralytics needed here)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vc_train import export as export_mod
from vc_train.cli import main

PY_DIR = Path(__file__).resolve().parents[1]


class _Reached(Exception):
    """Raised by a stub standing in for the first call that could import ultralytics."""


@pytest.fixture
def online_env(monkeypatch):
    """A shell that has not been set up for offline use: YOLO_OFFLINE unset, nothing for the code under test to rely on."""
    monkeypatch.setenv("YOLO_OFFLINE", "")
    monkeypatch.delenv("YOLO_OFFLINE")


def test_export_onnx_sets_the_offline_env_before_unpickling_the_checkpoint(tmp_path, monkeypatch, online_env):
    import torch

    seen = {}

    def fake_load(*args, **kwargs):
        # unpickling a checkpoint imports ultralytics.nn.tasks, which runs its is_online() check at import time
        seen["offline"] = os.environ.get("YOLO_OFFLINE")
        raise _Reached

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(_Reached):
        export_mod.export_onnx(tmp_path / "best.pt", tmp_path / "model", 64, 64)
    assert seen["offline"] == "1"


def test_check_consistency_sets_the_offline_env_before_loading_the_model(tmp_path, monkeypatch, online_env):
    seen = {}

    def fake_load_yolo(weights):
        seen["offline"] = os.environ.get("YOLO_OFFLINE")
        raise _Reached

    monkeypatch.setattr(export_mod, "load_yolo", fake_load_yolo)
    with pytest.raises(_Reached):
        export_mod.check_consistency(tmp_path / "best.pt", tmp_path / "model.onnx", [])
    assert seen["offline"] == "1"


def test_check_consistency_sets_the_offline_env_before_importing_anything_heavy(tmp_path, monkeypatch, online_env):
    seen = {}
    real_import = __import__

    def spy_import(name, *args, **kwargs):
        if name in ("torch", "onnxruntime"):
            seen.setdefault("offline", os.environ.get("YOLO_OFFLINE"))
        return real_import(name, *args, **kwargs)

    def fake_load_yolo(weights):
        raise _Reached

    monkeypatch.setattr("builtins.__import__", spy_import)
    monkeypatch.setattr(export_mod, "load_yolo", fake_load_yolo)
    with pytest.raises(_Reached):
        export_mod.check_consistency(tmp_path / "best.pt", tmp_path / "model.onnx", [])
    assert seen["offline"] == "1"


def test_cli_main_sets_the_offline_env_first(tmp_path, online_env):
    # a failing command still ends in main(), so the env is checked after an error exit
    assert main(["export", "--run", str(tmp_path / "run")]) == 2
    assert os.environ.get("YOLO_OFFLINE") == "1"
    assert os.environ.get("YOLO_AUTOINSTALL") == "false"


def test_no_vc_train_module_imports_ultralytics_when_imported():
    code = (
        "import sys\n"
        "import vc_train.cli, vc_train.dataset, vc_train.export, vc_train.geometry, vc_train.offline, vc_train.train\n"
        "sys.exit(1 if 'ultralytics' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(PY_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
