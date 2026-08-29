"""Tests for ``InverseBlochPhaseStrategy``.

Targets ``mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy``.

The strategy is a multi-parameter mapper plus a Tikhonov smoothness prior on a
background-removed ``phase_residual`` map. "Inverse-Bloch" is a legacy label:
the true Bloch/SPGR phase forward-model residual is NOT implemented (F5 honesty
relabel). These tests pin (a) the smoothness math and (b) the honesty note.
"""
from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy import (
    InverseBlochPhaseStrategy,
)


def test_phase_residual_smoothness_zero_for_constant_field() -> None:
    s = object.__new__(InverseBlochPhaseStrategy)
    s.device = torch.device("cpu")
    phase = torch.full((1, 1, 8, 8), 0.7)
    assert s._phase_residual_smoothness(phase).item() == 0.0


def test_phase_residual_smoothness_positive_for_varying_field() -> None:
    s = object.__new__(InverseBlochPhaseStrategy)
    s.device = torch.device("cpu")
    ramp = torch.linspace(0, 1, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8).contiguous()
    assert s._phase_residual_smoothness(ramp).item() > 0.0


def test_phase_residual_smoothness_guards_low_rank() -> None:
    s = object.__new__(InverseBlochPhaseStrategy)
    s.device = torch.device("cpu")
    assert s._phase_residual_smoothness(torch.zeros(4)).item() == 0.0


def test_docstrings_disclaim_inverse_bloch_forward_model() -> None:
    doc = (
        (InverseBlochPhaseStrategy.__doc__ or "")
        + (
            __import__(
                "mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy",
                fromlist=["x"],
            ).__doc__
            or ""
        )
    ).lower()
    assert "smoothness" in doc
    # Must state the true Bloch forward-model residual is not implemented.
    assert "not implemented" in doc or "legacy label" in doc


# --- D10: the phase-smoothness prior must not be silently skipped -----------


def _shell(monkeypatch, lam):
    """Strategy shell with the parent loss stubbed, to isolate the guard."""
    import mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy as m

    s = object.__new__(InverseBlochPhaseStrategy)
    s.device = torch.device("cpu")
    s.lambda_phase_smooth = lam
    monkeypatch.setattr(
        m.OneShotMultiParameterStrategy,
        "_compute_losses_impl",
        lambda self, **kw: {"loss_total": torch.tensor(1.0)},
        raising=False,
    )
    monkeypatch.setattr(
        m.InverseBlochPhaseStrategy,
        "_resolve_legacy_batch",
        lambda self, ib, kw: ib,
        raising=False,
    )
    return s


def test_missing_phase_residual_raises_when_the_prior_is_requested(monkeypatch) -> None:
    """The weight knob was read and INFO-logged over an unreachable term.

    PhaseResidualTransform had no way to be constructed, so the key was ALWAYS
    absent and the prior ALWAYS skipped -- pitfall #15 wrapped around #16.
    """
    s = _shell(monkeypatch, 0.01)
    with pytest.raises(ValueError) as exc:
        s._compute_losses_impl(input_batch={"input": torch.zeros(1, 1, 4, 4)})
    msg = str(exc.value)
    assert "phase_residual" in msg
    assert "lambda_phase_smooth" in msg


def test_lambda_zero_does_not_raise(monkeypatch) -> None:
    """No prior requested, nothing silently dropped -- the guard must not fire."""
    s = _shell(monkeypatch, 0.0)
    out = s._compute_losses_impl(input_batch={"input": torch.zeros(1, 1, 4, 4)})
    assert "loss_phase_smooth" not in out


def test_present_residual_still_adds_the_term(monkeypatch) -> None:
    """Guard against over-tightening: the happy path must be unchanged."""
    s = _shell(monkeypatch, 0.5)
    batch = {"phase_residual": torch.zeros(1, 1, 4, 4)}
    out = s._compute_losses_impl(input_batch=batch)
    assert "loss_phase_smooth" in out
