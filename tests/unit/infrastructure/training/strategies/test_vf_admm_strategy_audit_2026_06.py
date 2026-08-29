"""Regression tests for the 2026-06 VF-ADMM-strategy audit fixes.

Covers the BROKEN_STRATEGY_ONLY finding from the 2026-05-31 strategy audit
that is entirely contained in ``vf_admm_strategy.py``:

- **Hardcoded ADMM hyperparameters bypass config SSOT.** ``self._admm_iters``,
  ``self._admm_mu`` and ``self._admm_rho`` were literal assignments only fed
  into a ``logger.info`` line — never read by any logic (no ADMM unroll, no DC
  loop, no μ/ρ dual variables exist). They were unread/unstamped knobs
  (pitfall #15) and contradicted the docstring's "ADMM alternating ... over 8
  iters" claim. The fix deletes the three dead attributes, trims the log line
  to the two config-driven λ weights, and corrects the false ADMM-loop
  docstrings.

The setup path is exercised with monkeypatched module-level dependencies so we
do not need the full DI / simulator / loss-registry machinery — the assertion
is about which attributes the method does (and no longer does) assign.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies import (  # noqa: E402
    vf_admm_strategy as mod,
)
from mriforge.infrastructure.training.strategies.vf_admm_strategy import (  # noqa: E402
    ConcreteVFADMMStrategy,
)


class _Recon:
    lambda_prior = 0.25
    lambda_marker = 0.5


class _Losses:
    reconstruction = _Recon()


class _Config:
    losses = _Losses()


def _make_strategy() -> ConcreteVFADMMStrategy:
    """Bare instance with just the fields ``_setup_*`` reads off ``self``."""
    strat = ConcreteVFADMMStrategy.__new__(ConcreteVFADMMStrategy)
    strat.config = _Config()  # type: ignore[attr-defined]
    strat.device = torch.device("cpu")  # type: ignore[attr-defined]
    return strat


def _run_setup(monkeypatch: pytest.MonkeyPatch) -> ConcreteVFADMMStrategy:
    """Run ``_setup_strategy_specific_components`` with all heavy deps faked."""
    sentinel = object()
    monkeypatch.setattr(mod, "build_simulator_from_config", lambda *a, **k: sentinel)
    monkeypatch.setattr(mod, "create_loss", lambda *a, **k: _IdentityModule())
    monkeypatch.setattr(mod, "NoiseVarianceEstimator", _IdentityModule)
    # UnifiedReconstructionLossComputer is imported lazily inside the method.
    import mriforge.models.losses.computers.unified_diffusion_reconstruction as urc

    monkeypatch.setattr(
        urc, "UnifiedReconstructionLossComputer", lambda *a, **k: object()
    )
    strat = _make_strategy()
    strat._setup_strategy_specific_components()
    return strat


class _IdentityModule(torch.nn.Module):
    def to(self, *a, **k):  # noqa: D401 - keep .to(device) chains cheap
        return self


def test_dead_admm_knobs_are_not_assigned(monkeypatch: pytest.MonkeyPatch) -> None:
    """After setup the three dead ADMM literals must not exist on the instance."""
    strat = _run_setup(monkeypatch)
    assert not hasattr(strat, "_admm_iters")
    assert not hasattr(strat, "_admm_mu")
    assert not hasattr(strat, "_admm_rho")


def test_config_driven_lambda_weights_still_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The λ weights that ARE read from config must still be stamped."""
    strat = _run_setup(monkeypatch)
    assert strat._lambda_prior == pytest.approx(0.25)
    assert strat._lambda_marker == pytest.approx(0.5)


def test_setup_source_has_no_admm_literals() -> None:
    """Guard against the dead knobs creeping back into the setup method."""
    src = inspect.getsource(
        ConcreteVFADMMStrategy._setup_strategy_specific_components
    )
    assert "_admm_iters" not in src
    assert "_admm_mu" not in src
    assert "_admm_rho" not in src


def test_module_docstring_drops_false_admm_loop_claim() -> None:
    """The module docstring must no longer claim a multi-iter ADMM unroll/DC loop."""
    doc = mod.__doc__ or ""
    assert "ADMM alternating DC" not in doc


def test_class_docstring_drops_false_data_consistency_step() -> None:
    """The class docstring must no longer advertise a data-consistency ADMM step."""
    doc = ConcreteVFADMMStrategy.__doc__ or ""
    assert "Data consistency via" not in doc


# ── 2026-06-02: measurement (mask) is threaded into the DC-bearing model ──


def test_compute_losses_threads_undersampling_mask() -> None:
    """Train path must hand the twin's sampling mask to the generator."""
    src = inspect.getsource(ConcreteVFADMMStrategy._compute_losses_impl)
    assert "undersampling_mask_kwargs(self.simulator)" in src


def test_validation_threads_undersampling_mask() -> None:
    """Val path must thread the mask too (else val DC differs from train)."""
    src = inspect.getsource(ConcreteVFADMMStrategy.validation_step)
    assert "undersampling_mask_kwargs(self.simulator)" in src
