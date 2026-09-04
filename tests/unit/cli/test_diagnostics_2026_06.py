"""Tests for ``spectramr doctor`` (cluster environment diagnostics).

2026-06-11 CLI enrichment. ``doctor`` is a cluster pre-flight tool: it prints
torch/CUDA/device/cache info and doubles as a gate (``--require-cuda``,
``--config``) that exits non-zero on a misconfigured environment.
"""

from __future__ import annotations

import argparse

from spectramr.cli import diagnostics


def test_collect_diagnostics_has_expected_shape():
    d = diagnostics.collect_diagnostics()
    for key in ("version", "python", "platform", "torch", "cache_root", "data_root", "env"):
        assert key in d, key
    assert isinstance(d["torch"], dict)
    assert "available" in d["torch"]


def test_render_diagnostics_is_human_readable():
    text = diagnostics.render_diagnostics(diagnostics.collect_diagnostics())
    assert "spectramr doctor" in text
    assert "torch" in text
    assert "cache_root" in text


def test_run_doctor_healthy_returns_zero():
    args = argparse.Namespace(config=None, require_cuda=False, json=False)
    assert diagnostics.run_doctor(args) == 0


def test_run_doctor_require_cuda_gate(monkeypatch):
    # Force the CUDA probe to report unavailable → the gate must fail.
    monkeypatch.setattr(
        diagnostics,
        "_torch_info",
        lambda: {
            "available": True,
            "version": "2.11.0",
            "cuda_compiled": "12.9",
            "cuda_available": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "devices": [],
        },
    )
    args = argparse.Namespace(config=None, require_cuda=True, json=False)
    assert diagnostics.run_doctor(args) == 1


def test_run_doctor_missing_config_fails(tmp_path):
    args = argparse.Namespace(
        config=tmp_path / "does_not_exist.yaml", require_cuda=False, json=False
    )
    assert diagnostics.run_doctor(args) == 1


def test_run_doctor_valid_config_passes():
    args = argparse.Namespace(
        config="experiments/inprogress/dummy/dummy_gan.yaml",
        require_cuda=False,
        json=False,
    )
    assert diagnostics.run_doctor(args) == 0
