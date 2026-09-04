"""Tests for the ``spectramr report`` handler (``spectramr.cli.app.report``).

Covers the --config reporting-block parity path and CLI-override precedence
without loading a full TrainingSettings YAML (stubbed) or rendering figures
(generate_report stubbed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectramr.cli import app


class _Rep:
    task = "reconstruction"
    style = "nature"
    formats = ["pdf", "png"]
    dpi = 300
    panel_labels = True
    method_name = "cfgmethod"
    figures = None
    tables = None
    metrics = None
    hyperparameters = None
    extra_runs = None
    out_subdir = "report"
    emit_manifest = True
    submission_bundle = False
    tikz = False
    qc_figures = False
    html_report = True
    cohort = None


class _Run:
    seed = 11


class _Settings:
    reporting = _Rep()
    run = _Run()
    # The RETIRED top-level spelling, kept as a decoy. `RENAMES` moved `seed`
    # to `run.seed` with a raise posture, so a real `TrainingSettings` cannot
    # carry this attribute at all -- which is why the old pin below (`== 7`,
    # "falls back to settings.seed") passed on the double while `report`
    # stamped `seed: null` on every real run. The double asserted a schema
    # that no longer exists.
    seed = 7


def _ns(exp_dir, **over):
    base = dict(exp_dir=exp_dir, config=None, task=None, method=None,
                out_subdir=None, seed=None, dataset_version=None,
                cohort_json=None, html=None, qc=None, recursive=False)
    base.update(over)
    return argparse.Namespace(**base)


def _patch(monkeypatch):
    captured: dict = {}

    def _fake_gen(exp_dir, **kw):
        captured.update(kw)
        captured["exp_dir"] = exp_dir
        return {"figures": {}, "tables": {}, "out_dir": Path(exp_dir),
                "summary": Path(exp_dir) / "s.md", "html": None}

    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_report", _fake_gen)
    monkeypatch.setattr("spectramr.config.settings.TrainingSettings.from_yaml",
                        lambda p: _Settings())
    return captured


def test_missing_dir_returns_2(tmp_path):
    assert app.report(_ns(tmp_path / "nope")) == 2


def test_config_block_is_forwarded(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp, config=tmp_path / "c.yaml"))
    assert rc == 0
    assert captured["task"] == "reconstruction"
    assert captured["method_name"] == "cfgmethod"
    assert captured["dpi"] == 300
    assert captured["qc_figures"] is False
    assert captured["html_report"] is True
    assert captured["seed"] == 11, "the seed comes from run.seed"
    assert captured["seed"] != 7, "the retired top-level spelling must not be read"


def test_cli_flags_override_config(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp, config=tmp_path / "c.yaml", task="synthesis",
                        html=False, qc=True, method="cli_method"))
    assert rc == 0
    assert captured["task"] == "synthesis"
    assert captured["html_report"] is False
    assert captured["qc_figures"] is True
    assert captured["method_name"] == "cli_method"


def test_no_config_uses_cli_defaults(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp))  # no --config
    assert rc == 0
    assert captured["task"] == "default"
    assert captured["method_name"] == "run"  # exp_dir.name fallback
    assert captured["out_subdir"] == "report"


def test_recursive_dispatches_to_batch(tmp_path, monkeypatch):
    exp = tmp_path / "cohort"; exp.mkdir()
    seen: dict = {}

    def _fake_batch(root, **kw):
        seen["root"] = root
        seen["kw"] = kw
        return {"runs": [], "index": None, "n_runs": 2, "n_ok": 2}

    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_reports", _fake_batch)
    rc = app.report(_ns(exp, recursive=True))
    assert rc == 0
    assert seen["root"] == exp
    # method_name must NOT be forwarded (batch names each run itself)
    assert "method_name" not in seen["kw"]


def test_recursive_no_runs_found(tmp_path, monkeypatch):
    exp = tmp_path / "empty"; exp.mkdir()
    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_reports",
                        lambda root, **kw: {"runs": [], "index": None,
                                            "n_runs": 0, "n_ok": 0})
    assert app.report(_ns(exp, recursive=True)) == 0
