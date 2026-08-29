"""Unit tests for :class:`ConformalCalibrationRunner`.

Targets ``mriforge.infrastructure.calibration.runner``. Part of the parallel
test-coverage push (Unit 14).

The runner is a thin orchestration layer that:

1. Streams batches from a calibration ``DataLoader`` through a model.
2. Fits a ``ConformalCalibrator`` on the resulting ``(pred, target)`` pairs.
3. Optionally evaluates empirical coverage on a held-out test loader.

We test:

- End-to-end run on a synthetic mock model produces a sensible report.
- Empty calibration loader raises an actionable error.
- The ``CalibrationReport`` dataclass round-trips its fields and is
  *frozen* (immutable post-construction).
- Custom ``unpack_batch`` is honoured; default unpacker fails loud on
  unexpected batch shapes.
- The runner does not mutate model weights (audited via parameter-hash
  comparison).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from torch import nn

from mriforge.infrastructure.calibration.conformal import ConformalCalibrator
from mriforge.infrastructure.calibration.runner import (
    CalibrationReport,
    ConformalCalibrationRunner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityModel(nn.Module):
    """Stub: returns input unchanged. Useful because the residual then
    equals the additive noise we put into the synthetic dataset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


class _ParameterisedIdentity(nn.Module):
    """Identity model carrying a trainable parameter (so we can test no-mutation)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_noisy_loader(
    n_batches: int = 4,
    batch_size: int = 8,
    feature_dim: int = 4,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a list-of-batches loader where the identity model has known residual."""
    gen = torch.Generator().manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        target = torch.zeros(batch_size, feature_dim)
        # Identity model returns input, so residual = standard normal noise.
        inputs = target + torch.randn(target.shape, generator=gen)
        batches.append((inputs, target))
    return batches


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def test_run_produces_calibration_report() -> None:
    """``run`` returns a :class:`CalibrationReport` with all fields populated."""
    torch.manual_seed(0)
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_noisy_loader())
    assert isinstance(report, CalibrationReport)
    assert report.quantile >= 0.0
    assert report.n_calibration > 0
    assert report.alpha == 0.1
    assert report.empirical_coverage is None  # no test loader supplied


def test_run_quantile_matches_calibrator_state() -> None:
    """``report.quantile`` mirrors ``runner.calibrator.quantile`` post-fit."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    runner = ConformalCalibrationRunner(model=_IdentityModel(), calibrator=cal)
    report = runner.run(_make_noisy_loader())
    assert report.quantile == cal.quantile
    assert report.n_calibration == cal.n_calibration


def test_run_avg_set_size_is_double_quantile() -> None:
    """``avg_set_size == 2 · quantile`` (symmetric band width)."""
    torch.manual_seed(0)
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(_make_noisy_loader())
    assert report.avg_set_size == pytest.approx(2.0 * report.quantile)


def test_run_with_test_loader_reports_empirical_coverage() -> None:
    """Supplying a test loader fills ``empirical_coverage`` ∈ [0, 1]."""
    torch.manual_seed(0)
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(
        _make_noisy_loader(seed=1), test_loader=_make_noisy_loader(seed=2)
    )
    assert report.empirical_coverage is not None
    assert 0.0 <= report.empirical_coverage <= 1.0


def test_run_orchestrates_score_and_conformal_correctly() -> None:
    """End-to-end synthetic check: marginal coverage ≈ 1 − α on fresh data.

    The runner calls the calibrator which calls the score function;
    coverage on a fresh split should still match the target within
    statistical noise (the unit-spec validity property).
    """
    torch.manual_seed(2026)
    alpha = 0.1
    n_batches, batch_size, fdim = 32, 64, 4  # ≈ 8192 calibration samples
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=alpha),
    )
    report = runner.run(
        _make_noisy_loader(n_batches=n_batches, batch_size=batch_size, feature_dim=fdim, seed=10),
        test_loader=_make_noisy_loader(n_batches=n_batches, batch_size=batch_size, feature_dim=fdim, seed=20),
    )
    target = 1.0 - alpha
    assert report.empirical_coverage is not None
    assert abs(report.empirical_coverage - target) < 0.05, (
        f"Coverage {report.empirical_coverage:.3f} far from target {target:.3f}"
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_calibration_loader_raises() -> None:
    """An empty calibration loader is rejected with an actionable error."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    with pytest.raises(ValueError, match="empty|at least one"):
        runner.run([])


# ---------------------------------------------------------------------------
# Unpacker
# ---------------------------------------------------------------------------


def test_default_unpack_rejects_non_tuple_batch() -> None:
    """The default unpacker fails loud on dict batches."""
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    bad_loader: list[Any] = [{"x": torch.zeros(4), "y": torch.zeros(4)}]
    with pytest.raises(ValueError, match="not a"):
        runner.run(bad_loader)


def test_custom_unpack_consumes_dict_batches() -> None:
    """A user-supplied unpacker lets the runner handle arbitrary batch shapes."""
    raw_loader = [
        {"x": torch.zeros(4, 3), "y": torch.zeros(4, 3)} for _ in range(3)
    ]
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
        unpack_batch=lambda b: (b["x"], b["y"]),
    )
    report = runner.run(raw_loader)
    # 3 batches × 4 × 3 = 36 flattened calibration samples.
    assert report.n_calibration == 36


def test_default_unpack_accepts_longer_tuples() -> None:
    """The default unpacker takes the first two elements of an N-tuple."""
    loader = [(torch.zeros(2, 2), torch.zeros(2, 2), "extra-metadata") for _ in range(2)]
    runner = ConformalCalibrationRunner(
        model=_IdentityModel(),
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    report = runner.run(loader)
    assert report.n_calibration == 8


# ---------------------------------------------------------------------------
# State protection
# ---------------------------------------------------------------------------


def test_runner_exposes_fitted_calibrator() -> None:
    """``runner.calibrator`` is the same object, now fitted."""
    cal = ConformalCalibrator(alpha=0.1)
    runner = ConformalCalibrationRunner(model=_IdentityModel(), calibrator=cal)
    assert runner.calibrator is cal
    runner.run(_make_noisy_loader())
    assert runner.calibrator.is_fitted is True


def test_runner_does_not_mutate_model_parameters() -> None:
    """Calibration must not change model weights (it is post-training).

    Uses a parameterised model so there *is* something to mutate; the
    runner runs under ``torch.no_grad()`` and never invokes an
    optimiser, so the weight checksum is invariant.
    """
    model = _ParameterisedIdentity()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    runner = ConformalCalibrationRunner(
        model=model,
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    runner.run(_make_noisy_loader())
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"Parameter '{k}' was mutated"


def test_runner_calls_model_forward_for_each_batch() -> None:
    """``run`` invokes the model exactly once per calibration batch."""

    class _Counter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.n_calls += 1
            return x

    model = _Counter()
    runner = ConformalCalibrationRunner(
        model=model,
        calibrator=ConformalCalibrator(alpha=0.1),
    )
    n_batches = 5
    runner.run(_make_noisy_loader(n_batches=n_batches))
    assert model.n_calls == n_batches


# ---------------------------------------------------------------------------
# CalibrationReport dataclass
# ---------------------------------------------------------------------------


def test_report_is_a_frozen_dataclass() -> None:
    """``CalibrationReport`` is ``@dataclass(frozen=True)`` — fields are immutable."""
    report = CalibrationReport(
        quantile=0.5,
        n_calibration=100,
        alpha=0.1,
        empirical_coverage=0.9,
        avg_set_size=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.quantile = 0.7  # type: ignore[misc]


def test_report_fields_preserved() -> None:
    """Constructed values round-trip on attribute access."""
    report = CalibrationReport(
        quantile=0.42,
        n_calibration=2048,
        alpha=0.05,
        empirical_coverage=0.951,
        avg_set_size=0.84,
    )
    assert report.quantile == 0.42
    assert report.n_calibration == 2048
    assert report.alpha == 0.05
    assert report.empirical_coverage == 0.951
    assert report.avg_set_size == 0.84


def test_report_allows_none_coverage() -> None:
    """``empirical_coverage`` is optional (no test loader supplied)."""
    report = CalibrationReport(
        quantile=0.1,
        n_calibration=10,
        alpha=0.1,
        empirical_coverage=None,
        avg_set_size=0.2,
    )
    assert report.empirical_coverage is None
