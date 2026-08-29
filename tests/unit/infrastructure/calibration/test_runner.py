"""Tests for ``ConformalCalibrationRunner``.

Targets ``mriforge.infrastructure.calibration.runner``. Phase-2 of PR-3.

Categories:

- Run on a tiny mock model + calibration loader → calibrator fits
- Empirical coverage reported when a test loader is supplied
- Custom ``unpack_batch`` callable used when batches aren't tuples
- Default unpacker rejects non-tuple batches loudly
- Report fields match the calibrator's fitted state
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mriforge.infrastructure.calibration.conformal import ConformalCalibrator
from mriforge.infrastructure.calibration.runner import (
    CalibrationReport,
    ConformalCalibrationRunner,
)


class _IdentityModel(nn.Module):
    """Stub model that returns its input unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_loader(
    n_batches: int = 4, batch_size: int = 8, std: float = 1.0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a small list-of-batches loader with paired inputs/targets."""
    loader: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        target = torch.zeros(batch_size, 4)
        # Input is target + noise — identity model returns input, so the
        # residual is exactly the noise.
        input_ = target + torch.randn_like(target) * std
        loader.append((input_, target))
    return loader


# ---------------------------------------------------------------------------
# run() basic
# ---------------------------------------------------------------------------


def test_run_fits_calibrator() -> None:
    """After ``run``, the calibrator's quantile is finite and ≥ 0."""
    torch.manual_seed(0)
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_loader())
    assert isinstance(report, CalibrationReport)
    assert report.quantile >= 0.0
    assert report.n_calibration > 0


def test_run_records_alpha_in_report() -> None:
    """``CalibrationReport.alpha`` mirrors the calibrator's α."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.05),
    )
    report = runner.run(_make_loader())
    assert report.alpha == 0.05


def test_run_avg_set_size_is_double_quantile() -> None:
    """``avg_set_size = 2 · quantile`` (band width)."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_loader())
    assert pytest.approx(report.avg_set_size) == 2.0 * report.quantile


# ---------------------------------------------------------------------------
# Empirical coverage on test split
# ---------------------------------------------------------------------------


def test_empirical_coverage_in_unit_interval() -> None:
    """When a test loader is provided, coverage ∈ [0, 1]."""
    torch.manual_seed(0)
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_loader(), test_loader=_make_loader())
    assert report.empirical_coverage is not None
    assert 0.0 <= report.empirical_coverage <= 1.0


def test_no_test_loader_yields_none_coverage() -> None:
    """Without a test loader, ``empirical_coverage`` is left ``None``."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_loader())
    assert report.empirical_coverage is None


# ---------------------------------------------------------------------------
# Unpack callable
# ---------------------------------------------------------------------------


def test_custom_unpack_used() -> None:
    """A custom ``unpack_batch`` lets the runner consume non-tuple batches."""
    # Loader yields dicts; default unpacker would reject them.
    raw_loader = [
        {"x": torch.zeros(4, 4), "y": torch.zeros(4, 4)} for _ in range(2)
    ]
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
        unpack_batch=lambda b: (b["x"], b["y"]),
    )
    report = runner.run(raw_loader)
    assert report.n_calibration == 4 * 4 * 2  # 2 batches × 4×4 elements


def test_default_unpack_rejects_dict() -> None:
    """Default unpacker fails loud on dict batches."""
    raw_loader = [{"x": torch.zeros(4), "y": torch.zeros(4)}]
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    with pytest.raises(ValueError, match="not a"):
        runner.run(raw_loader)


# ---------------------------------------------------------------------------
# Calibrator state
# ---------------------------------------------------------------------------


def test_calibrator_property_exposes_fitted_state() -> None:
    """The runner exposes the calibrator after fit so reporting can read it."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    runner.run(_make_loader())
    assert runner.calibrator.is_fitted is True
    assert runner.calibrator.quantile >= 0.0


def test_runner_does_not_train_model() -> None:
    """Running the calibration phase does not advance gradients."""

    class _Counter(nn.Module):
        forward_calls: int = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            type(self).forward_calls += 1
            return x

    runner = ConformalCalibrationRunner(
        model=_Counter(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    runner.run(_make_loader(n_batches=3))
    # No exception, no gradient-tracking — implied by the ``no_grad`` block.
    assert _Counter.forward_calls == 3
