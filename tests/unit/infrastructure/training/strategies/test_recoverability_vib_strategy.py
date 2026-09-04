"""Tests for RecoverabilityVIBStrategy (B-2.3)."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
    RecoverabilityVIBStrategy,
    _kl_rate,
    compute_recoverability_vib_loss,
)
from spectramr.models.generators.recoverability_vib_net import RecoverabilityVIBNet


def _net() -> RecoverabilityVIBNet:
    return RecoverabilityVIBNet(width=24, latent_channels=8, n_downsample=2)


def _batch() -> dict:
    return {"input": torch.rand(2, 1, 32, 32), "target": torch.rand(2, 1, 32, 32)}


def test_kl_rate_zero_for_standard_normal_posterior() -> None:
    # mu=0, logvar=0 (unit variance) -> KL to N(0,I) is exactly 0.
    mu = torch.zeros(2, 4, 8, 8)
    assert float(_kl_rate(mu, torch.zeros_like(mu))) == pytest.approx(0.0, abs=1e-6)


def test_loss_keys_and_rate_positive() -> None:
    out = compute_recoverability_vib_loss(_net(), _batch())
    assert {"loss_total", "loss_recon", "rate_nats"} <= set(out)
    assert torch.isfinite(out["loss_total"]) and float(out["rate_nats"]) > 0.0


def test_higher_beta_increases_total_loss() -> None:
    torch.manual_seed(0)
    m = _net()
    batch = _batch()
    lo = compute_recoverability_vib_loss(m, batch, beta=1e-3)
    hi = compute_recoverability_vib_loss(m, batch, beta=1e-1)
    assert float(hi["loss_total"].detach()) > float(lo["loss_total"].detach())  # beta weights rate


# --- free-bits anti-collapse floor (B-2.3, b23 rate→0.002 degeneracy) ----------


class _CollapsedVIB(torch.nn.Module):
    """Stub with a fully-collapsed posterior (mu=0, logvar=0 => rate=0)."""

    def encode(self, x: torch.Tensor):
        b = x.shape[0]
        return torch.zeros(b, 8, 8, 8), torch.zeros(b, 8, 8, 8)

    def reparameterize(self, mu, logvar):
        return mu

    def decode(self, z, size):
        return torch.zeros(z.shape[0], 1, *size)


def test_free_bits_floors_penalty_below_floor() -> None:
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate_free_bits,
    )

    mu = torch.zeros(2, 4, 8, 8)  # collapsed posterior => true rate 0
    logvar = torch.zeros(2, 4, 8, 8)
    assert float(_kl_rate(mu, logvar)) == pytest.approx(0.0, abs=1e-6)
    penalty = _kl_rate_free_bits(mu, logvar, free_bits=0.5)
    assert float(penalty) == pytest.approx(0.5, abs=1e-6)  # floored at free_bits


def test_free_bits_is_noop_above_floor() -> None:
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate_free_bits,
    )

    torch.manual_seed(0)
    mu = torch.randn(2, 4, 8, 8)  # nonzero => rate > 0.01
    logvar = torch.zeros(2, 4, 8, 8)
    assert torch.allclose(_kl_rate(mu, logvar), _kl_rate_free_bits(mu, logvar, 0.01))


def test_free_bits_removes_collapse_pushdown_gradient() -> None:
    """Below the floor the free-bits penalty has zero gradient, so beta*rate can
    no longer push the encoder toward posterior collapse."""
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate_free_bits,
    )

    mu = torch.full((1, 4, 8, 8), 0.01, requires_grad=True)
    logvar = torch.zeros(1, 4, 8, 8, requires_grad=True)
    _kl_rate_free_bits(mu, logvar, free_bits=1.0).backward()  # true rate << floor
    assert float(mu.grad.abs().sum()) == 0.0


def test_loss_free_bits_penalises_but_reports_true_rate() -> None:
    batch = _batch()
    without = compute_recoverability_vib_loss(_CollapsedVIB(), batch, beta=1.0, free_bits=0.0)
    withfb = compute_recoverability_vib_loss(_CollapsedVIB(), batch, beta=1.0, free_bits=0.5)
    # true rate is 0 in BOTH (honest reported budget), regardless of free_bits
    assert float(without["rate_nats"]) == pytest.approx(0.0, abs=1e-6)
    assert float(withfb["rate_nats"]) == pytest.approx(0.0, abs=1e-6)
    # but the loss carries the beta*free_bits floor cost (breaks the free collapse)
    assert float(withfb["loss_total"]) == pytest.approx(
        float(without["loss_total"]) + 1.0 * 0.5, abs=1e-5
    )


def test_effective_beta_ramps_linearly_over_warmup() -> None:
    # Anti-posterior-collapse (#20): beta ramps 0 -> target over beta_warmup_iters.
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat._vib_beta = 0.1
    strat._vib_beta_warmup = 1000
    assert strat._effective_beta(0) == 0.0
    assert strat._effective_beta(500) == pytest.approx(0.05)
    assert strat._effective_beta(1000) == pytest.approx(0.1)
    assert strat._effective_beta(50_000) == pytest.approx(0.1)  # clamped to target


def test_effective_beta_no_warmup_is_constant() -> None:
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat._vib_beta = 0.1
    strat._vib_beta_warmup = 0
    assert strat._effective_beta(0) == 0.1
    assert strat._effective_beta(9999) == 0.1


def test_warmup_zeroes_rate_pressure_at_iteration_zero() -> None:
    # At iteration 0 with warmup active, beta_eff=0 -> total == reconstruction only (the rate
    # term contributes nothing), so recon leads before the budget tightens ('recon floor').
    torch.manual_seed(0)
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._vib_beta = 0.1
    strat._vib_beta_warmup = 1000
    strat._vib_lambda_recon = 1.0
    out = strat._compute_losses_impl(
        input_batch=_batch(), target_batch=None, epoch=0, iteration=0)
    assert float(out["loss_total"].detach()) == pytest.approx(float(out["loss_recon"]), abs=1e-6)


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = compute_recoverability_vib_loss(m, batch, beta=1e-3)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._vib_beta = 1e-3
    strat._vib_lambda_recon = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._vib_beta = 1e-3
    strat._vib_lambda_recon = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range() -> None:
    m = _net().eval()
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(torch.rand(2, 1, 32, 32), {}).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_score_recoverability_emits_rate_and_input_sensitivity() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat.logging_service = None
    out = strat._score_recoverability(torch.rand(2, 1, 32, 32))
    assert out["val_recoverability_rate"] > 0.0
    assert out["val_input_sensitivity"] > 1e-4  # a real autoencoder is input-dependent
    assert out["val_encoder_l2_norm"] > 0.0  # bottleneck collapse witness
    assert "val_encoder_logvar_mean" in out


def test_score_recoverability_warns_on_collapsed_encoder() -> None:
    # A zeroed encoder -> mu~0 (posterior collapse at the bottleneck) must WARN (pitfall #20),
    # caught directly at the source, not only via a lagging rate~0.
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.zero_()
        for p in m.to_mu.parameters():
            p.zero_()
    warnings: list[str] = []
    strat = object.__new__(RecoverabilityVIBStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat.logging_service = types.SimpleNamespace(log_warning=lambda msg: warnings.append(msg))
    out = strat._score_recoverability(torch.rand(2, 1, 32, 32))
    assert out["val_encoder_l2_norm"] < 1e-2
    assert any("encoder_l2_norm" in w for w in warnings)


def test_rate_decreases_with_higher_beta_after_training() -> None:
    # The rate-distortion mechanism: training with a TIGHTER budget (higher beta) converges to a
    # LOWER rate. (A single forward's rate is beta-independent; the response is a training effect.)
    def _converged_rate(beta: float) -> float:
        torch.manual_seed(0)
        m = _net()
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        batch = _batch()
        out = None
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            out = compute_recoverability_vib_loss(m, batch, beta=beta)
            out["loss_total"].backward()
            opt.step()
        assert out is not None
        return float(out["rate_nats"])

    assert _converged_rate(1e-1) < _converged_rate(1e-4)


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "recoverability_vib" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "recoverability_vib" in TrainingStrategyConfigSchema.model_fields


# --------------------------------------------------------------- rate_reduction (#20)
# The b23 arm collapsed on 2026-07-05 AND again on 2026-07-23 despite beta_warmup_iters
# and a free_bits floor. Root cause: `_kl_rate` SUMS over every latent dimension, so on
# the 16x91x109 = 158,704-dim latent an innocuous-looking beta=1e-3 is a per-dimension
# beta of 159 (and the ablation's 1e-1 is 15,870) — far past collapse at BOTH ends,
# which is exactly why the rate-distortion sweep was dead.


def _collapsed_posterior():
    """The posterior actually observed on 2026-07-23 (val_encoder_l2_norm 7.4e-04)."""
    import torch

    mu = torch.full((2, 4, 8, 8), 7.4e-4)
    return mu, torch.zeros_like(mu)


def test_sum_rate_scales_with_latent_dimension_count() -> None:
    """The legacy reduction makes the rate — hence the effective beta — grid-dependent."""
    import torch

    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate,
    )

    small = torch.randn(2, 4, 8, 8) * 0.5
    large = small.repeat(1, 1, 2, 2)  # 4x the dimensions, same per-dim statistics
    zeros_s, zeros_l = torch.zeros_like(small), torch.zeros_like(large)

    r_small = float(_kl_rate(small, zeros_s, "sum"))
    r_large = float(_kl_rate(large, zeros_l, "sum"))
    assert r_large == pytest.approx(4.0 * r_small, rel=1e-5)

    # mean_per_dim is invariant to the grid: beta becomes dimensionless and portable.
    m_small = float(_kl_rate(small, zeros_s, "mean_per_dim"))
    m_large = float(_kl_rate(large, zeros_l, "mean_per_dim"))
    assert m_large == pytest.approx(m_small, rel=1e-5)


def test_mean_per_dim_equals_sum_divided_by_dimensions() -> None:
    import torch

    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate,
    )

    mu = torch.randn(2, 4, 8, 8) * 0.5
    lv = torch.zeros_like(mu)
    n_dims = mu[0].numel()
    assert float(_kl_rate(mu, lv, "sum")) == pytest.approx(
        float(_kl_rate(mu, lv, "mean_per_dim")) * n_dims, rel=1e-4
    )


def test_free_bits_floor_is_meaningless_in_aggregate_units() -> None:
    """Why the 0.2 floor never protected b23: a collapsed encoder is already near it."""
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate,
    )

    mu, lv = _collapsed_posterior()
    collapsed_aggregate = float(_kl_rate(mu, lv, "sum"))
    # A fully collapsed posterior sits only a small factor below the configured floor,
    # so `clamp_min` engages immediately and the penalty is constant from the start.
    assert 0 < collapsed_aggregate < 0.2
    # In per-dimension units the same state is orders of magnitude below a 0.01 floor,
    # so the floor is a real constraint rather than a rounding artefact.
    assert float(_kl_rate(mu, lv, "mean_per_dim")) < 1e-4


def test_free_bits_clamp_has_no_gradient_below_the_floor() -> None:
    """Free bits removes DOWNWARD pressure only — it cannot rescue a collapsed encoder."""
    import torch

    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate_free_bits,
    )

    mu = torch.full((2, 4, 8, 8), 7.4e-4, requires_grad=True)
    lv = torch.zeros_like(mu, requires_grad=True)
    penalty = _kl_rate_free_bits(mu, lv, free_bits=1.0, reduction="mean_per_dim")
    penalty.backward()
    assert mu.grad is not None
    # Zero gradient: nothing pushes the rate back UP once it is under the floor.
    assert float(mu.grad.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_unknown_rate_reduction_raises_not_degrades() -> None:
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        _kl_rate,
    )

    mu, lv = _collapsed_posterior()
    with pytest.raises(ValueError, match="rate_reduction"):
        _kl_rate(mu, lv, "per_channel")


def test_compute_loss_threads_rate_reduction_through() -> None:
    """The knob must reach the objective, not just the schema (#15)."""
    import torch

    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        compute_recoverability_vib_loss,
    )

    class _Net:
        def encode(self, x):
            mu = torch.full((x.shape[0], 4, 8, 8), 0.5, requires_grad=True)
            return mu, torch.zeros_like(mu)

        @staticmethod
        def reparameterize(mu, logvar):
            return mu

        def decode(self, z, out_hw):
            return torch.zeros(z.shape[0], 1, *out_hw, requires_grad=True)

    batch = {"input": torch.zeros(2, 1, 8, 8), "target": torch.zeros(2, 1, 8, 8)}
    r_sum = compute_recoverability_vib_loss(
        _Net(), batch, beta=1.0, rate_reduction="sum"
    )["rate_nats"]
    r_dim = compute_recoverability_vib_loss(
        _Net(), batch, beta=1.0, rate_reduction="mean_per_dim"
    )["rate_nats"]
    # 4*8*8 = 256 latent dims -> the summed rate is 256x the per-dim rate.
    assert float(r_sum) == pytest.approx(float(r_dim) * 256, rel=1e-4)
