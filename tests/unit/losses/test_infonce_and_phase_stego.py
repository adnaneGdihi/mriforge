"""Phase-2 (InfoNCECritic) + Phase-3 (PhaseStegoScoreLoss) unit tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.infonce_critic import InfoNCECritic  # noqa: E402
from spectramr.models.losses.phase_stego_score import PhaseStegoScoreLoss  # noqa: E402
from spectramr.models.losses.registry import LossRegistry  # noqa: E402


class TestInfoNCECritic:
    def test_registered(self) -> None:
        reg = LossRegistry._custom_losses
        assert "infonce_critic" in reg

    def test_in_batch_forward_shape(self) -> None:
        torch.manual_seed(0)
        critic = InfoNCECritic(temperature=0.1, critic_hidden_dim=0, feature_dim=16)
        z = torch.randn(8, 16)
        y = torch.randn(8, 16)
        loss = critic(z, y)
        assert loss.ndim == 0
        assert loss.item() > 0  # cross-entropy on random pairs > 0

    def test_paired_features_lower_loss(self) -> None:
        """When z == y (perfect alignment) the loss should be lower than random pairing."""
        torch.manual_seed(0)
        critic = InfoNCECritic(temperature=0.1, critic_hidden_dim=0, feature_dim=32)
        z = torch.randn(16, 32)
        loss_paired = critic(z, z).item()
        loss_random = critic(z, torch.randn(16, 32)).item()
        assert loss_paired < loss_random, (
            f"Aligned z=y loss ({loss_paired}) should be < random pairing ({loss_random})."
        )

    def test_temperature_changes_loss(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 16)
        y = torch.randn(8, 16)
        loss_lo = InfoNCECritic(temperature=0.05, critic_hidden_dim=0, feature_dim=16)(z, y).item()
        loss_hi = InfoNCECritic(temperature=0.5, critic_hidden_dim=0, feature_dim=16)(z, y).item()
        assert loss_lo != pytest.approx(loss_hi, abs=1e-3)

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            InfoNCECritic(temperature=-0.1)

    def test_shape_mismatch_raises(self) -> None:
        critic = InfoNCECritic(temperature=0.1, critic_hidden_dim=0, feature_dim=8)
        with pytest.raises(ValueError, match="batch sizes"):
            critic(torch.randn(4, 8), torch.randn(5, 8))

    def test_gradient_flow(self) -> None:
        critic = InfoNCECritic(temperature=0.1, critic_hidden_dim=0, feature_dim=16)
        z = torch.randn(8, 16, requires_grad=True)
        y = torch.randn(8, 16, requires_grad=True)
        loss = critic(z, y)
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0

    def test_momentum_bank_backward_across_steps(self) -> None:
        """Regression: with a momentum bank, ``z @ bank.T`` saves the bank for
        backward, but ``_enqueue`` writes ``self._bank`` in place. Without a
        clone the SECOND step (once the bank is non-empty) raised "a variable
        needed for gradient computation has been modified by an inplace
        operation" (ib_vf smoke 20260605). Two encode→loss→backward→step cycles
        must complete cleanly."""
        torch.manual_seed(0)
        critic = InfoNCECritic(
            temperature=0.1,
            critic_hidden_dim=8,
            feature_dim=16,
            negative_pool="momentum_bank",
            bank_size=8,
        )
        proj = torch.nn.Linear(16, 16)
        opt = torch.optim.SGD(list(critic.parameters()) + list(proj.parameters()), lr=0.1)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            z = proj(torch.randn(4, 16))
            y = proj(torch.randn(4, 16))
            loss = critic(z, y)
            loss.backward()  # second iteration would raise the in-place error
            opt.step()
        assert critic._bank is not None and critic._bank.numel() > 0


    def test_momentum_bank_batch_larger_than_bank(self) -> None:
        """Regression: a batch wider than ``bank_size`` must not crash the FIFO write.

        The ib_vf navigator feeds per-patch features (e.g. 576 rows) into a small
        momentum bank (64). The old ring math did ``end = (ptr + b) % bank_size``
        and wrote ``self._bank[ptr:end] = y_proj`` — for b > bank_size the slice
        collapsed and raised "expanded size (64) must match existing size (576)"
        (cluster crash exp_vf_ib_infonce, diagnostics 2026-06-26). The write must
        instead absorb only the most-recent ``bank_size`` rows.
        """
        torch.manual_seed(0)
        bank_size = 64
        critic = InfoNCECritic(
            temperature=0.1,
            critic_hidden_dim=0,
            feature_dim=16,
            negative_pool="momentum_bank",
            bank_size=bank_size,
        )
        z = torch.randn(576, 16)
        y = torch.randn(576, 16)
        loss = critic(z, y)  # would raise the size-mismatch RuntimeError pre-fix
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert critic._bank is not None
        assert critic._bank.shape[0] == bank_size  # bank stays at capacity

    def test_momentum_bank_exact_fill_wraps_cleanly(self) -> None:
        """A batch of exactly ``bank_size`` rows (ptr+b == bank_size) must fill the
        whole bank, not drop it via an ``[ptr:0]`` empty slice."""
        torch.manual_seed(0)
        bank_size = 8
        critic = InfoNCECritic(
            temperature=0.1,
            critic_hidden_dim=0,
            feature_dim=4,
            negative_pool="momentum_bank",
            bank_size=bank_size,
        )
        critic(torch.randn(bank_size, 4), torch.randn(bank_size, 4))
        assert critic._bank is not None
        # pointer wrapped back to 0 and every row was written (no NaN/zero gap)
        assert int(critic._bank_ptr.item()) == 0
        assert critic._bank.shape[0] == bank_size


class TestPhaseStegoScoreLoss:
    def test_registered(self) -> None:
        reg = LossRegistry._custom_losses
        assert "phase_stego_score" in reg

    def test_fourier_forward_shape(self) -> None:
        loss_fn = PhaseStegoScoreLoss(sigma_M=0.05, basis="fourier")
        x = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        marker = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        v = loss_fn(x, marker)
        assert v.ndim == 0
        assert v.item() >= 0

    def test_zero_residual_when_target_matches_forward(self) -> None:
        loss_fn = PhaseStegoScoreLoss(sigma_M=0.05, basis="fourier")
        x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        marker = loss_fn._phase_stego_forward(x).detach()
        v = loss_fn(x, marker)
        assert v.item() == pytest.approx(0.0, abs=1e-5)

    def test_magnitude_only_perturbation_does_not_change_score(self) -> None:
        """Score must depend only on the phase of x, not its magnitude."""
        loss_fn = PhaseStegoScoreLoss(sigma_M=0.1, basis="fourier")
        x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        marker = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        v1 = loss_fn(x, marker).item()
        # Scale magnitudes by random positive factors per voxel; preserve phase.
        magnitude_scale = (torch.rand(1, 1, 8, 8) + 0.5).to(torch.complex64)
        x_scaled = x * magnitude_scale
        v2 = loss_fn(x_scaled, marker).item()
        assert abs(v1 - v2) < 1e-3, (
            f"Phase-stego loss should be magnitude-invariant; v1={v1}, v2={v2}."
        )

    def test_sigma_M_scaling(self) -> None:
        """Halving sigma_M should quadruple the loss (1/(2 sigma^2) factor)."""
        x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        marker = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        v_a = PhaseStegoScoreLoss(sigma_M=0.05, basis="fourier")(x, marker).item()
        v_b = PhaseStegoScoreLoss(sigma_M=0.025, basis="fourier")(x, marker).item()
        assert v_b == pytest.approx(4.0 * v_a, rel=0.01)

    def test_invalid_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_M"):
            PhaseStegoScoreLoss(sigma_M=-0.01)

    def test_invalid_basis_raises(self) -> None:
        with pytest.raises(ValueError, match="basis"):
            PhaseStegoScoreLoss(basis="haar")

    def test_shape_mismatch_raises(self) -> None:
        loss_fn = PhaseStegoScoreLoss(sigma_M=0.05, basis="fourier")
        with pytest.raises(ValueError, match="does not"):
            loss_fn(
                torch.randn(1, 1, 16, 16, dtype=torch.complex64),
                torch.randn(1, 1, 8, 8, dtype=torch.complex64),
            )
