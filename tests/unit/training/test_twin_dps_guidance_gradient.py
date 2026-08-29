"""Unit tests for ``TwinLikelihoodDPSStrategy.compute_guidance_gradient``.

This is the mathematical core of Twin-DPS guidance: the autograd
gradient of the phase-stego marker NLL with respect to ``x_t``,
threaded through a user-supplied Tweedie estimator. The full
sampler-hook integration (folding the gradient into the reverse-SDE
drift inside ``DiffusionTrainingStrategy``) is not yet in place;
these tests cover the helper that the eventual sampler will call.

We bypass the heavy ``DiffusionTrainingStrategy.__init__`` by
constructing the instance via ``__new__`` and setting only the
attributes the helper needs --- the helper is self-contained and
does not touch the diffusion scaffolding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies.twin_dps_strategy import (  # noqa: E402
    TwinLikelihoodDPSStrategy,
)
from mriforge.models.losses.phase_stego_score import PhaseStegoScoreLoss  # noqa: E402


@pytest.fixture
def marker_template(tmp_path: Path) -> Path:
    """16x16 Fourier-domain marker template for a small probe image."""
    torch.manual_seed(0)
    template = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    # 4 random mid-band frequencies, unit magnitude.
    coords = [(2, 3), (5, 7), (9, 4), (12, 11)]
    for (yi, xi), phase in zip(coords, [0.1, 0.7, 1.3, 2.0]):
        template[0, 0, yi, xi] = torch.complex(
            torch.cos(torch.tensor(phase)), torch.sin(torch.tensor(phase))
        )
    path = tmp_path / "marker.pt"
    torch.save(template, path)
    return path


@pytest.fixture
def twin_dps_stub(marker_template: Path) -> TwinLikelihoodDPSStrategy:
    """Construct a TwinDPS strategy bypassing the heavy DiffusionTrainingStrategy init."""
    strat = TwinLikelihoodDPSStrategy.__new__(TwinLikelihoodDPSStrategy)
    strat.lambda_y = 1.0
    strat.lambda_M = 1.0
    strat.sigma_M = 0.05
    strat.phase_stego_basis = "fourier"
    strat.guidance_callback_interval = 1
    strat._marker_template_path = str(marker_template)
    strat._marker_template = None
    strat._marker_score_loss = PhaseStegoScoreLoss(
        sigma_M=strat.sigma_M, basis=strat.phase_stego_basis
    )
    return strat


def _identity_tweedie(x_t: torch.Tensor) -> torch.Tensor:
    """Trivial Tweedie estimator: at t=0 the posterior mean equals x_t."""
    return x_t


def _scaled_tweedie(x_t: torch.Tensor) -> torch.Tensor:
    """A nonlinear Tweedie estimator (light smoothing) — keeps the chain rule active."""
    return x_t * (1.0 + 0.1 * x_t.abs().mean())


class TestComputeGuidanceGradient:
    def test_gradient_is_finite(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        x_t = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        grad = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        assert torch.isfinite(grad.real).all()
        assert torch.isfinite(grad.imag).all()

    def test_gradient_shape_matches_input(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        x_t = torch.randn(3, 1, 16, 16, dtype=torch.complex64)
        grad = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        assert grad.shape == x_t.shape
        assert grad.is_complex()

    def test_gradient_nonzero_off_solution(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        """At a random point, the gradient must have nonzero norm."""
        torch.manual_seed(7)
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        grad = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        assert grad.abs().max().item() > 1e-6

    def test_lambda_M_scales_gradient(
        self, marker_template: Path
    ) -> None:
        """Doubling lambda_M doubles the returned gradient (linear in scale)."""
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)

        def make(scale: float) -> TwinLikelihoodDPSStrategy:
            strat = TwinLikelihoodDPSStrategy.__new__(TwinLikelihoodDPSStrategy)
            strat.lambda_y = 1.0
            strat.lambda_M = scale
            strat.sigma_M = 0.05
            strat.phase_stego_basis = "fourier"
            strat.guidance_callback_interval = 1
            strat._marker_template_path = str(marker_template)
            strat._marker_template = None
            strat._marker_score_loss = PhaseStegoScoreLoss(
                sigma_M=strat.sigma_M, basis=strat.phase_stego_basis
            )
            return strat

        g1 = make(1.0).compute_guidance_gradient(x_t, _identity_tweedie)
        g2 = make(2.0).compute_guidance_gradient(x_t, _identity_tweedie)
        assert torch.allclose(g2, 2.0 * g1, atol=1e-5)

    def test_sign_convention(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        r"""Returned tensor is -lambda_M * dL/dx_t, i.e. +lambda_M * d log p / d x_t.

        Adding it to the drift moves x_t in the direction that increases
        the marker log-likelihood. So one step of x_t += eps * grad must
        decrease the marker loss.
        """
        torch.manual_seed(3)
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        # Loss at current point.
        marker = twin_dps_stub._load_marker_template(x_t.device)
        loss_before = twin_dps_stub._marker_score_loss(x_t, marker).item()

        grad = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        eps = 1e-3
        x_step = x_t + eps * grad
        loss_after = twin_dps_stub._marker_score_loss(x_step, marker).item()
        assert loss_after < loss_before, (
            f"Following the guidance gradient should reduce the marker NLL; "
            f"got loss_before={loss_before:.6e}, loss_after={loss_after:.6e}."
        )

    def test_input_not_mutated(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        """The caller's x_t must not be left with grad attached."""
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        x_t_backup = x_t.clone()
        _ = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        assert torch.equal(x_t, x_t_backup)
        assert not x_t.requires_grad

    def test_chain_rule_through_nontrivial_tweedie(
        self, twin_dps_stub: TwinLikelihoodDPSStrategy
    ) -> None:
        """Autograd must trace through a nonlinear Tweedie estimator."""
        torch.manual_seed(11)
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        grad_id = twin_dps_stub.compute_guidance_gradient(x_t, _identity_tweedie)
        grad_scaled = twin_dps_stub.compute_guidance_gradient(x_t, _scaled_tweedie)
        # The two gradients must differ (different functions of x_t).
        assert not torch.allclose(grad_id, grad_scaled, atol=1e-4)
