"""Regression tests for the ``meta-evaluate`` unknown-metric validation path.

``src/mriforge/cli/app.py::meta_evaluate`` historically *warned-and-skipped*
when ``--metrics`` named a metric that was not in the registry, silently
yielding a smaller ranking that looked like a successful run. The fix makes
an unregistered metric name fail loud (CLAUDE.md pitfall #9, no silent
fallbacks): the requested set is diffed against ``list_available()`` and any
unknown name now raises :class:`ValueError` *before* the pipeline runs.

These tests isolate that single validation branch. The clean-reference loader
and the meta-evaluation pipeline are mocked out, and the registry is replaced
with a known small set via monkeypatch, so the assertion does not depend on
which metrics happen to be registered in the running environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mriforge.core.metrics as core_metrics
from mriforge.cli import app
from mriforge.cli.app import meta_evaluate

pytestmark = pytest.mark.unit


# A deterministic, registry-independent stand-in for the available metrics.
_KNOWN_METRICS = {"psnr", "ssim", "rmse", "mae"}


def _meta_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Minimal namespace ``meta_evaluate`` expects (defaults = a 2-D synthetic run).

    Mirrors the namespace built by the CLI argparse layer; the ``ranking``
    (non-``--tiers``) branch is selected (``tiers=None``) so we exercise the
    unknown-metric guard at ``app.py`` rather than the §8 harness.
    """
    base = {
        "input": None,
        "metrics": None,
        "severities": 2,
        "max_subjects": 1,
        "max_contrasts": 1,
        "contrasts": None,
        "synthetic_n": 2,
        "device": "cpu",
        "eval_mode": "2d",
        "betting": False,
        "betting_alpha": 0.05,
        "seed": 0,
        "out": tmp_path / "summary.json",
        "figures": False,
        "figures_dir": None,
        "tables": False,
        "tables_dir": None,
        "nr_battery": False,
        "tiers": None,
        "register_aggregator": False,
        "aggregator": "z",
        "screen_pool": False,
        "core_degradations": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def isolated_meta_evaluate(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the heavy downstream of ``meta_evaluate`` so only validation runs.

    * ``list_available`` (the registry the CLI consults) → fixed small set.
    * ``_synthesize_phantom_volumes`` → a lightweight sentinel (no torch work).
    * ``MetaEvaluationPipeline`` → a MagicMock whose ``.run`` we can assert was
      *never* called when validation rejects the request.

    Returns the pipeline mock so tests can assert the pipeline was not reached.
    """
    monkeypatch.setattr(core_metrics, "list_available", lambda: set(_KNOWN_METRICS))
    monkeypatch.setattr(
        app,
        "_synthesize_phantom_volumes",
        lambda n, seed: [("phantom_0", object())],
    )

    pipeline_cls = MagicMock(name="MetaEvaluationPipeline")
    monkeypatch.setattr(
        "mriforge.core.metrics.meta_evaluation.MetaEvaluationPipeline",
        pipeline_cls,
    )
    return pipeline_cls


def test_unknown_metric_raises_value_error(
    tmp_path: Path, isolated_meta_evaluate: MagicMock
) -> None:
    """A ``--metrics`` name absent from the registry must raise, not skip."""
    args = _meta_args(tmp_path, metrics=["psnr", "definitely_not_a_metric"])
    with pytest.raises(ValueError, match="definitely_not_a_metric"):
        meta_evaluate(args)


def test_unknown_metric_error_mentions_unregistered_phrase(
    tmp_path: Path, isolated_meta_evaluate: MagicMock
) -> None:
    """The error names the contract (unregistered metric) so the typo is obvious."""
    args = _meta_args(tmp_path, metrics=["bogus_metric"])
    with pytest.raises(ValueError, match="unregistered metric"):
        meta_evaluate(args)


def test_unknown_metric_aborts_before_pipeline_runs(
    tmp_path: Path, isolated_meta_evaluate: MagicMock
) -> None:
    """Validation fails loud *before* the meta-evaluation pipeline is built/run.

    A warn-and-skip regression would have constructed and run the pipeline on
    the surviving metrics. We assert the pipeline mock was never touched and
    that no summary JSON was written.
    """
    args = _meta_args(tmp_path, metrics=["psnr", "nope"])
    with pytest.raises(ValueError):
        meta_evaluate(args)
    isolated_meta_evaluate.from_defaults.assert_not_called()
    isolated_meta_evaluate.assert_not_called()
    assert not (tmp_path / "summary.json").exists()


def test_all_known_metrics_pass_validation_reaching_pipeline(
    tmp_path: Path, isolated_meta_evaluate: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: when every requested metric *is* registered, no ValueError fires.

    This fences the guard against over-rejecting — the unknown-metric branch
    must trigger only on genuinely-absent names. We stub the metric adapter so
    no real metric callables are needed, then stop the run right after the
    validation gate by raising a unique sentinel from the pipeline factory.
    """

    class _StopAfterValidation(RuntimeError):
        pass

    # build_safe_metric_set is imported lazily inside meta_evaluate from its
    # own module; patch it there so we never need real metric callables. It
    # feeds ``MetricSet(**...)``, so it must supply the required ``metrics`` key.
    monkeypatch.setattr(
        "mriforge.core.metrics.meta_evaluation.metric_adapter.build_safe_metric_set",
        lambda valid: {"metrics": {name: (lambda *a, **k: 0.0) for name in valid}},
    )
    isolated_meta_evaluate.from_defaults.side_effect = _StopAfterValidation

    args = _meta_args(tmp_path, metrics=["psnr", "ssim"])
    # All names are known → we pass the ValueError gate and reach the pipeline,
    # where our sentinel halts the run. No ValueError means the guard did not
    # falsely reject a registered metric.
    with pytest.raises(_StopAfterValidation):
        meta_evaluate(args)


def _patch_safe_metric_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mriforge.core.metrics.meta_evaluation.metric_adapter.build_safe_metric_set",
        lambda valid: {"metrics": {name: (lambda *a, **k: 0.0) for name in valid}},
    )


def test_unknown_aggregator_raises(
    tmp_path: Path, isolated_meta_evaluate: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--aggregator`` outside {z, kemeny} fails loud (no silent fallback, D2)."""
    _patch_safe_metric_set(monkeypatch)
    args = _meta_args(tmp_path, aggregator="bogus")
    with pytest.raises(ValueError, match=r"z.*kemeny"):
        meta_evaluate(args)


def test_aggregator_and_screen_pool_are_wired(
    tmp_path: Path, isolated_meta_evaluate: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--aggregator kemeny`` / ``--screen-pool`` reach the pipeline fields (D2)."""

    class _StopAtRunError(RuntimeError):
        pass

    _patch_safe_metric_set(monkeypatch)
    pipe = isolated_meta_evaluate.from_defaults.return_value
    pipe.run.side_effect = _StopAtRunError
    args = _meta_args(tmp_path, aggregator="kemeny", screen_pool=True)
    with pytest.raises(_StopAtRunError):
        meta_evaluate(args)
    assert pipe.aggregator == "kemeny"
    assert pipe.screen_pool_enabled is True


def test_default_uses_physics_degradation_ssot(
    tmp_path: Path, isolated_meta_evaluate: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By default the simulator uses the injected physics degradation SSOT (C2)."""

    class _StopAtRunError(RuntimeError):
        pass

    _patch_safe_metric_set(monkeypatch)
    pipe = isolated_meta_evaluate.from_defaults.return_value
    pipe.run.side_effect = _StopAtRunError
    args = _meta_args(tmp_path)  # no --core-degradations
    with pytest.raises(_StopAtRunError):
        meta_evaluate(args)
    # The physics operator library was injected (not the dependency-free core lib).
    assert pipe.simulator_config.operator_library is not None


def test_core_degradations_flag_uses_dependency_free_library(
    tmp_path: Path, isolated_meta_evaluate: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--core-degradations`` opts into the dependency-free core library (C2)."""

    class _StopAtRunError(RuntimeError):
        pass

    _patch_safe_metric_set(monkeypatch)
    pipe = isolated_meta_evaluate.from_defaults.return_value
    pipe.run.side_effect = _StopAtRunError
    args = _meta_args(tmp_path, core_degradations=True)
    with pytest.raises(_StopAtRunError):
        meta_evaluate(args)
    assert pipe.simulator_config.operator_library is None  # core fallback
