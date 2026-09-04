"""Unit tests for the post-training conformal/calibration dispatch hook.

Regression for the 2026-06 inert-certification finding: the conformal,
equivariance-conformal, and exchangeability strategies expose
``run_calibration(dataloader)`` but it had ZERO call sites — the pipeline ran the
standard train loop and never produced their declared certificate
(``coverage_at_alpha`` / exchangeability p-value). ``_maybe_run_calibration``
closes that seam. These tests exercise the hook in isolation (no GPU / DI).

2026-07-03 fail-loud regression (mrixfields_b17_dice_risk_calibration): the hook
used to SWALLOW a ``run_calibration`` failure into a warning while training
reported ``success: True``. Every strategy that exposes ``run_calibration`` is
certify-only (``train_step`` returns ``[]``); the certificate is its SOLE
deliverable, so a swallowed failure is a facade success (pitfall #16) and a
"passed-with-warnings" (pitfall #10). The hook now RAISES so the run reports
``success: False``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spectramr.pipelines.train import _maybe_run_calibration


class _RecordingLog:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def log_info(self, msg: str) -> None:
        self.infos.append(msg)

    def log_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _pipeline(val_loader: object) -> SimpleNamespace:
    return SimpleNamespace(data_loaders={"val": val_loader, "train": object()})


def test_hook_noop_for_strategy_without_run_calibration(tmp_path) -> None:
    """A normal recon strategy (no run_calibration) must be untouched."""
    strategy = SimpleNamespace()  # no run_calibration attribute
    log = _RecordingLog()
    _maybe_run_calibration(
        strategy, _pipeline(["batch"]), {"run_output_dir": str(tmp_path)}, log
    )
    assert not (tmp_path / "calibration_report.json").exists()
    assert log.warnings == []


def test_hook_dispatches_and_stamps_report(tmp_path) -> None:
    """A calibration strategy's run_calibration is invoked on the val loader and
    its report is stamped into the run dir (the provenance the certificate needs)."""
    called = {}

    def run_calibration(loader):
        called["loader"] = loader
        return {"conformal": {"empirical_coverage": 0.9, "target": 0.9}}

    strategy = SimpleNamespace(run_calibration=run_calibration)
    log = _RecordingLog()
    sentinel_loader = ["v0", "v1"]
    _maybe_run_calibration(
        strategy, _pipeline(sentinel_loader), {"run_output_dir": str(tmp_path)}, log
    )
    assert called["loader"] is sentinel_loader  # ran on the validation loader
    report = tmp_path / "calibration_report.json"
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["conformal"]["empirical_coverage"] == 0.9


def test_hook_raises_on_calibration_failure(tmp_path) -> None:
    """A failure inside run_calibration must FAIL LOUD, not be swallowed.

    The certificate is the certify-only arm's sole deliverable; a swallowed
    failure reports a facade ``success: True`` (the mrixfields_b17 finding). The
    hook re-raises so ``run_training_pipeline`` returns ``success: False`` and the
    original cause (e.g. "Unrecognised checkpoint format") is preserved via
    exception chaining.
    """

    def run_calibration(loader):
        raise RuntimeError("Unrecognised checkpoint format")

    strategy = SimpleNamespace(run_calibration=run_calibration)
    log = _RecordingLog()
    with pytest.raises(RuntimeError, match="Unrecognised checkpoint format") as exc:
        _maybe_run_calibration(
            strategy, _pipeline(["b"]), {"run_output_dir": str(tmp_path)}, log
        )
    # original cause chained, not masked
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert not (tmp_path / "calibration_report.json").exists()


def test_hook_raises_when_no_val_loader(tmp_path) -> None:
    """A certify-only arm with no calibration cohort cannot certify → fail loud."""
    strategy = SimpleNamespace(run_calibration=lambda loader: {})
    log = _RecordingLog()
    with pytest.raises(RuntimeError, match="no validation loader"):
        _maybe_run_calibration(
            strategy, _pipeline(None), {"run_output_dir": str(tmp_path)}, log
        )
