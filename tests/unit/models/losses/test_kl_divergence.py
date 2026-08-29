"""Tests for ``KLDivergenceLoss`` and ``VQKLLoss``.

Targets ``mriforge.models.losses.kl_divergence``. KL[N(μ, σ²) ∥ N(0, 1)] for
β-VAE training, optional Burgess-style capacity constraint, optional
linear annealing, and a VQ-VAE variant with code-book + commitment
losses.

Categories:

- Closed-form: prior point ``mu=0, log_var=0`` → KL = 0
- Property: KL grows with |μ| and with |log_var|
- Annealing: monotone weight ramp from start → end
- Capacity: ``|KL - C|`` branch active only when ``limit_capacity``
- VQ: commitment_cost weights commitment branch correctly
- Registry: ``kl`` and ``vq_kl`` both resolvable
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.kl_divergence import (
    KLDivergenceLoss,
    VQKLLoss,
    kl_divergence_loss,
    vq_loss,
)


# ---------------------------------------------------------------------------
# KL: closed-form
# ---------------------------------------------------------------------------


def test_kl_zero_at_prior() -> None:
    """At μ=0, log_var=0 (i.e. σ²=1), KL[N ∥ N] = 0."""
    loss = KLDivergenceLoss(beta=1.0)
    mu = torch.zeros(2, 4)
    log_var = torch.zeros(2, 4)
    out = loss(mu, log_var)
    assert pytest.approx(out.item(), abs=1e-6) == 0.0


def test_kl_grows_with_mean_offset() -> None:
    """KL increases with |μ| at fixed log_var."""
    loss = KLDivergenceLoss()
    log_var = torch.zeros(1, 4)
    kl_zero = loss(torch.zeros(1, 4), log_var).item()
    kl_far = loss(torch.full((1, 4), 3.0), log_var).item()
    assert kl_far > kl_zero


def test_kl_grows_with_logvar_offset() -> None:
    """KL increases when log_var deviates from 0 in either direction."""
    loss = KLDivergenceLoss()
    mu = torch.zeros(1, 4)
    kl_zero = loss(mu, torch.zeros(1, 4)).item()
    kl_high = loss(mu, torch.full((1, 4), 2.0)).item()
    assert kl_high > kl_zero


# ---------------------------------------------------------------------------
# Beta scaling
# ---------------------------------------------------------------------------


def test_beta_scales_loss_linearly() -> None:
    """β=2 returns exactly 2× of the β=1 result on the same input."""
    mu = torch.randn(2, 4)
    log_var = torch.randn(2, 4) * 0.1
    out_1 = KLDivergenceLoss(beta=1.0)(mu, log_var)
    out_2 = KLDivergenceLoss(beta=2.0)(mu, log_var)
    assert pytest.approx(out_2.item()) == 2.0 * out_1.item()


# ---------------------------------------------------------------------------
# Capacity branch (Burgess et al. 2018)
# ---------------------------------------------------------------------------


def test_capacity_branch_uses_abs() -> None:
    """``limit_capacity=True`` returns ``β · |KL - C|``."""
    mu = torch.zeros(1, 4)
    log_var = torch.zeros(1, 4)  # KL == 0
    loss = KLDivergenceLoss(beta=1.0, capacity=2.0, limit_capacity=True)
    # |0 - 2| = 2
    assert pytest.approx(loss(mu, log_var).item()) == 2.0


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------


def test_annealing_weight_ramps() -> None:
    """After ``annealing_steps`` calls, weight reaches ``annealing_end``."""
    loss = KLDivergenceLoss(
        beta=1.0,
        annealing_steps=4,
        annealing_start=0.0,
        annealing_end=1.0,
    )
    mu = torch.randn(1, 2)
    log_var = torch.zeros(1, 2)
    for _ in range(4):
        loss(mu, log_var)
    # After enough steps, weight saturates at annealing_end.
    loss(mu, log_var)
    assert pytest.approx(loss.get_current_weight()) == 1.0


def test_reset_annealing_restores_initial() -> None:
    """``reset_annealing`` rewinds to ``annealing_start``."""
    loss = KLDivergenceLoss(
        beta=1.0, annealing_steps=4, annealing_start=0.2, annealing_end=1.0
    )
    mu = torch.randn(1, 2)
    log_var = torch.zeros(1, 2)
    for _ in range(2):
        loss(mu, log_var)
    loss.reset_annealing()
    assert pytest.approx(loss.get_current_weight()) == 0.2
    assert loss.current_step == 0


def test_annealing_multiply_matches_item_path() -> None:
    """Tensor-buffer multiply equals the pre-fix ``.item()`` float multiply.

    Regression for the per-call GPU sync: ``annealing_weight`` now scales
    ``final_loss`` as a 0-dim tensor instead of ``.item()``.
    """
    kwargs = {
        "beta": 1.0,
        "annealing_steps": 4,
        "annealing_start": 0.0,
        "annealing_end": 1.0,
    }
    new_loss = KLDivergenceLoss(**kwargs)
    ref_loss = KLDivergenceLoss(**kwargs)
    mu = torch.randn(2, 4)
    log_var = torch.randn(2, 4) * 0.1
    for _ in range(6):
        out = new_loss(mu, log_var)
        # pre-fix reference path: same KL body, weight applied via .item()
        lv = torch.clamp(log_var, min=-20.0, max=20.0)
        m = torch.clamp(mu, min=-50.0, max=50.0)
        kl = torch.mean(-0.5 * torch.sum(1 + lv - m.pow(2) - lv.exp(), dim=1))
        ref_loss._update_annealing_weight()
        expected = ref_loss.annealing_weight.item() * (ref_loss.beta * kl)
        assert torch.allclose(out, expected)


def test_annealing_forward_never_calls_item(monkeypatch) -> None:
    """Poison ``Tensor.item``: the annealed forward path must not sync."""

    def _boom(self):
        raise AssertionError(".item() called in per-step loss path")

    loss = KLDivergenceLoss(
        beta=1.0, annealing_steps=4, annealing_start=0.0, annealing_end=1.0
    )
    mu = torch.randn(2, 4)
    log_var = torch.randn(2, 4) * 0.1
    monkeypatch.setattr(torch.Tensor, "item", _boom)
    out = loss(mu, log_var)
    monkeypatch.undo()
    assert torch.isfinite(out)
    # _update_annealing_weight semantics unchanged: one step consumed
    assert loss.current_step == 1


def test_annealing_output_stays_tensor_with_grad() -> None:
    """The annealed loss remains a differentiable 0-dim tensor."""
    loss = KLDivergenceLoss(
        beta=1.0, annealing_steps=2, annealing_start=0.5, annealing_end=1.0
    )
    mu = torch.randn(1, 4, requires_grad=True)
    log_var = torch.zeros(1, 4)
    out = loss(mu, log_var)
    assert isinstance(out, torch.Tensor) and out.dim() == 0
    out.backward()
    assert mu.grad is not None


# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------


def test_extreme_logvar_clamped() -> None:
    """Very large log_var (would otherwise overflow exp) is clamped → finite."""
    loss = KLDivergenceLoss()
    mu = torch.zeros(1, 4)
    log_var = torch.full((1, 4), 50.0)  # exp(50) overflows; clamp to 20
    out = loss(mu, log_var)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def test_functional_kl_matches_class() -> None:
    """Functional ``kl_divergence_loss`` matches ``KLDivergenceLoss``."""
    mu = torch.randn(2, 4)
    log_var = torch.randn(2, 4) * 0.1
    fn_out = kl_divergence_loss(mu, log_var, beta=0.5)
    cls_out = KLDivergenceLoss(beta=0.5)(mu, log_var)
    assert pytest.approx(fn_out.item()) == cls_out.item()


# ---------------------------------------------------------------------------
# VQ-KL
# ---------------------------------------------------------------------------


def test_vq_kl_reduces_to_vq_when_kl_disabled() -> None:
    """With ``use_kl=False``, output equals plain VQ loss."""
    quantized = torch.randn(2, 4)
    z = torch.randn(2, 4)
    out = VQKLLoss(commitment_cost=0.25, use_kl=False)(quantized, z)
    expected = vq_loss(quantized, z, commitment_cost=0.25)
    assert pytest.approx(out.item()) == expected.item()


def test_vq_kl_adds_kl_when_enabled() -> None:
    """``use_kl=True`` plus mu/log_var produces a strictly larger total."""
    quantized = torch.randn(2, 4)
    z = torch.randn(2, 4)
    mu = torch.full((2, 4), 1.0)
    log_var = torch.zeros(2, 4)
    plain = VQKLLoss(use_kl=False)(quantized, z).item()
    with_kl = VQKLLoss(use_kl=True, kl_weight=0.5)(quantized, z, mu, log_var).item()
    assert with_kl > plain


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_mu_and_logvar() -> None:
    """Backward reaches both μ and log_var."""
    loss = KLDivergenceLoss()
    mu = torch.randn(1, 4, requires_grad=True)
    log_var = torch.randn(1, 4, requires_grad=True) * 0.1
    log_var.retain_grad()
    loss(mu, log_var).backward()
    assert mu.grad is not None
    assert log_var.grad is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_kl_and_vq_kl_registered() -> None:
    """Both ``kl`` and ``vq_kl`` resolvable."""
    from mriforge.models.losses.registry import list_available

    names = list_available()
    assert "kl" in names
    assert "vq_kl" in names


# ---------------------------------------------------------------------------
# VQ gradient routing (regression: code-book / commitment stop-grad was swapped)
# ---------------------------------------------------------------------------


def _grad_mag(t: torch.Tensor) -> float:
    return 0.0 if t.grad is None else float(t.grad.abs().sum())


def test_functional_vq_loss_codebook_routes_to_codebook() -> None:
    """``vq_loss`` with ``commitment_cost=0`` must update the codebook only.

    The code-book term is ``‖z_q − sg[z]‖²`` (gradient to ``quantized``); the
    commitment term carries the stop-gradient on the codebook. Pre-fix the
    ``detach`` placement was swapped, so the only active gradient flowed to the
    encoder ``z`` instead of the codebook.
    """
    from mriforge.models.losses.kl_divergence import vq_loss

    quantized = torch.randn(4, 8, requires_grad=True)
    z = torch.randn(4, 8, requires_grad=True)

    loss = vq_loss(quantized, z, commitment_cost=0.0)
    loss.backward()

    assert _grad_mag(quantized) > 0
    assert _grad_mag(z) == pytest.approx(0.0, abs=1e-12)
