"""Tests for batch reporting over a cohort tree.

Targets ``mriforge.infrastructure.reporting.batch``: run discovery, per-run
report generation, the linking index, and per-run failure isolation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from mriforge.infrastructure.reporting.batch import (
    discover_run_dirs,
    generate_reports,
    write_batch_index,
)


def _make_run(root, name):
    d = root / name
    (d / "logs").mkdir(parents=True)
    (d / "logs" / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    return d


def test_discover_finds_runs_and_skips_non_runs(tmp_path):
    _make_run(tmp_path, "runA")
    _make_run(tmp_path / "paradigm", "runB")
    (tmp_path / "not_a_run").mkdir()  # no markers
    found = discover_run_dirs(tmp_path)
    names = {p.name for p in found}
    assert names == {"runA", "runB"}


def test_discover_direct_run_dir_returns_itself(tmp_path):
    run = _make_run(tmp_path, "solo")
    assert discover_run_dirs(run) == [run]


def test_discover_does_not_descend_into_run_artifacts(tmp_path):
    run = _make_run(tmp_path, "runA")
    # a 'report' subdir that itself contains a marker must NOT be a second run
    (run / "report" / "logs").mkdir(parents=True)
    (run / "report" / "logs" / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n")
    found = discover_run_dirs(tmp_path)
    assert found == [run]


def test_generate_reports_produces_index_and_per_run(tmp_path):
    _make_run(tmp_path, "runA")
    _make_run(tmp_path, "runB")
    out = generate_reports(tmp_path, task="default",
                           figures=["fig_1_2_learning_curves"], tables_=[],
                           html_report=False)
    assert out["n_runs"] == 2
    assert out["n_ok"] == 2
    assert out["index"] is not None and out["index"].exists()
    doc = out["index"].read_text()
    assert "runA" in doc and "runB" in doc
    # each run got its own report dir
    assert (tmp_path / "runA" / "report").exists()
    assert (tmp_path / "runB" / "report").exists()


def test_generate_reports_isolates_per_run_failure(tmp_path, monkeypatch):
    _make_run(tmp_path, "ok_run")
    bad = _make_run(tmp_path, "bad_run")

    import mriforge.infrastructure.reporting.batch as batch_mod

    real = batch_mod.generate_report

    def _flaky(run_dir, **kw):
        if run_dir == bad:
            raise RuntimeError("boom")
        return real(run_dir, **kw)

    monkeypatch.setattr(batch_mod, "generate_report", _flaky)
    out = generate_reports(tmp_path, figures=["fig_1_2_learning_curves"],
                           tables_=[], html_report=False)
    assert out["n_runs"] == 2
    assert out["n_ok"] == 1
    fails = [r for r in out["runs"] if not r["ok"]]
    assert len(fails) == 1 and "boom" in fails[0]["error"]


def test_write_batch_index_none_when_empty(tmp_path):
    assert write_batch_index(tmp_path, []) is None


# --------------------------------------------------------------------------
# `mriforge profile` files a throwaway run under <exp>/profiles/<run_id>/run/.
# --------------------------------------------------------------------------


def test_profiling_runs_are_not_discovered_as_experiments(tmp_path):
    """A profiling run carries the same markers as a real one.

    Without `profiles` in the skip set it would be reported as an experiment —
    but only for an arm not yet trained, because `_is_run_dir` stops the descent
    at a real run. That makes the phantom appear and disappear with unrelated
    history, so it is pinned rather than left to the discovery order.
    """
    arm = tmp_path / "results" / "exp_11"
    _make_run(arm / "profiles" / "run-1", "run")
    assert discover_run_dirs(tmp_path) == []


def test_a_real_run_is_still_found_beside_its_profiles(tmp_path):
    """The skip must not swallow the arm itself — proves the test discriminates."""
    arm = tmp_path / "results" / "exp_11"
    real = _make_run(arm.parent, "exp_11")
    _make_run(arm / "profiles" / "run-1", "run")
    assert discover_run_dirs(tmp_path) == [real]
