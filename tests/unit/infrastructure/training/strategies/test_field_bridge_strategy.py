"""Tests for FieldBridgeStrategy (B-3.3 field-conditioned stochastic bridge).

Not a true entropic-OT Schrödinger bridge — bridge-matching with a fixed
Brownian-bridge reference + posterior-mean re-noising sampler (see module docstring).
"""

from __future__ import annotations

import types

import torch
from torch import nn

from spectramr.infrastructure.training.strategies.field_bridge_strategy import (
    FieldBridgeStrategy,
    bridge_sample,
    compute_field_bridge_losses,
)
from spectramr.models.generators.field_velocity_unet import FieldVelocityUNet


class _OracleVelocity(nn.Module):
    """Constant per-unit-tau velocity (ignores x) — a non-denoising model."""

    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_):
        return self.w.expand_as(x)


class _DenoiserOracle(nn.Module):
    """Perfect-denoiser velocity: from any (noisy) x it points to the true endpoint,
    so x_hat_t = x + v*remaining == target regardless of accumulated noise. Isolates
    the SAMPLER's endpoint-pinning from model quality."""

    def __init__(self, target: torch.Tensor, tau_t: float) -> None:
        super().__init__()
        self.register_buffer("target", target)
        self.tau_t = tau_t

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_):
        tau_k = torch.log10(field_strength).reshape(-1, 1, 1, 1)
        remaining = (self.tau_t - tau_k).clamp_min(1e-6)
        return (self.target - x) / remaining


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 0.5]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_field_bridge_losses(FieldVelocityUNet(width=8), _batch(), sigma_bridge=0.1)
    assert {"loss_total", "loss_velocity"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_endpoint_l1_anchor_trains_the_denoiser_property() -> None:
    # Anti-degeneracy (#20): the endpoint L1 anchor |x_hat_t - x_t| is ~0 for a perfect
    # denoiser (x_hat_t == target regardless of interpolant noise) and >0 for a non-denoising
    # constant-velocity model — so upweighting it grounds the bridge in the true target.
    torch.manual_seed(0)
    x_s = torch.zeros(1, 1, 4, 4)
    x_t = torch.ones(1, 1, 4, 4)
    batch = {
        "input": x_s, "target": x_t,
        "field_strength": torch.tensor([1.0]),       # tau_s = 0
        "field_strength_target": torch.tensor([10.0]),  # tau_t = 1
    }
    denoiser = compute_field_bridge_losses(
        _DenoiserOracle(x_t, tau_t=1.0), batch, sigma_bridge=0.3, lambda_endpoint_l1=1.0)
    constant = compute_field_bridge_losses(
        _OracleVelocity(torch.ones(1, 1, 4, 4)), batch, sigma_bridge=0.3, lambda_endpoint_l1=1.0)
    assert "loss_endpoint_l1" in denoiser
    assert float(denoiser["loss_endpoint_l1"]) < 1e-5
    assert float(constant["loss_endpoint_l1"]) > float(denoiser["loss_endpoint_l1"])


def test_endpoint_l1_omitted_when_disabled() -> None:
    out = compute_field_bridge_losses(
        FieldVelocityUNet(width=8), _batch(), sigma_bridge=0.1, lambda_endpoint_l1=0.0)
    assert "loss_endpoint_l1" not in out


def test_velocity_loss_zero_for_matching_oracle() -> None:
    # Even with the stochastic interpolant, an oracle returning the secant velocity
    # u = (x_t - x_s)/span scores ~0 (the target is independent of the noise).
    x_s = torch.zeros(1, 1, 4, 4)
    x_t = torch.ones(1, 1, 4, 4)
    batch = {
        "input": x_s, "target": x_t,
        "field_strength": torch.tensor([1.0]),  # tau 0
        "field_strength_target": torch.tensor([10.0]),  # tau 1 -> span 1 -> u = 1
    }
    out = compute_field_bridge_losses(_OracleVelocity(torch.ones(1, 1, 4, 4)), batch, sigma_bridge=0.5)
    assert float(out["loss_velocity"]) < 1e-10


def test_sigma_zero_is_deterministic_sampler() -> None:
    # sigma_bridge=0 -> the sampler is deterministic (recovers the flow): two runs
    # from the same start are identical.
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=8).eval()
    x = torch.rand(1, 1, 16, 16)
    a = bridge_sample(m, x, 0.1, 7.0, n_steps=6, sigma_bridge=0.0)
    b = bridge_sample(m, x, 0.1, 7.0, n_steps=6, sigma_bridge=0.0)
    assert torch.allclose(a, b)


def test_sigma_positive_is_stochastic() -> None:
    # sigma_bridge>0 -> stochastic bridge: two samples differ (the distinction from
    # the deterministic field-flow).
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=8).eval()
    x = torch.rand(1, 1, 16, 16)
    a = bridge_sample(m, x, 0.1, 7.0, n_steps=6, sigma_bridge=0.3)
    b = bridge_sample(m, x, 0.1, 7.0, n_steps=6, sigma_bridge=0.3)
    assert not torch.allclose(a, b)


def test_deterministic_sampler_recovers_endpoint_constant_velocity() -> None:
    # sigma=0 + constant per-unit-tau velocity w -> integrate over [tau_s,tau_t] = x0+w.
    x0 = torch.zeros(1, 1, 4, 4)
    w = torch.full((1, 1, 4, 4), 0.7)
    out = bridge_sample(_OracleVelocity(w), x0, 1.0, 10.0, n_steps=8, sigma_bridge=0.0)
    assert torch.allclose(out, x0 + w, atol=1e-5)


def test_stochastic_sampler_pins_endpoint() -> None:
    # Fix A (posterior-mean re-noising): the per-step noise gamma(s)=sigma*sqrt(s(1-s))
    # is identically 0 at the final s=1, so the output equals the model's last-step
    # endpoint estimate x_hat_t with NO terminal noise. For a denoising model (x_hat_t
    # == target for any noisy x) the endpoint is therefore PINNED: across many sigma>0
    # trials the std collapses to ~0 (the old integrate-then-add-noise sampler would
    # carry the penultimate noise injection into the output -> not pinned).
    torch.manual_seed(0)
    x0 = torch.rand(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), 0.42)
    oracle = _DenoiserOracle(target, tau_t=1.0)  # b_t = 10 T -> tau_t = 1
    samples = torch.stack(
        [bridge_sample(oracle, x0, 1.0, 10.0, n_steps=64, sigma_bridge=0.5) for _ in range(16)]
    )
    assert torch.allclose(samples.mean(0), target, atol=1e-4)
    assert float(samples.std(0).max()) < 1e-5  # pinned: zero terminal noise


def test_velocity_loss_reduces() -> None:
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_field_bridge_losses(m, batch, sigma_bridge=0.1)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_velocity"].detach())
    assert out is not None and first is not None
    assert float(out["loss_velocity"].detach()) < first


def test_validation_forward_reads_fields_from_kwargs() -> None:
    strat = object.__new__(FieldBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=_OracleVelocity(torch.full((1, 1, 4, 4), 0.3)))
    strat._fb_n_steps = 8
    strat._fb_sigma = 0.0
    pred = strat._validation_forward(
        torch.zeros(1, 1, 4, 4),
        {"use_dc": False},
        field_strength=torch.tensor([1.0]),
        field_strength_target=torch.tensor([10.0]),
    )
    assert torch.allclose(pred, torch.full((1, 1, 4, 4), 0.3), atol=1e-5)


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_bridge" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "field_bridge" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=FieldVelocityUNet(width=8))

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the velocity model (train + sampler) ---


def test_contrast_id_threaded_train_and_sampling() -> None:
    # b33 enables use_contrast_conditioning on a widened FieldVelocityUNet; the bridge
    # strategy MUST forward contrast_id or the model raises at step 0 (the cluster
    # failure). Assert both the training loss and the stochastic sampler thread it.
    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_field_bridge_losses(spy, batch, sigma_bridge=0.1)
    assert seen["cid"] is cid  # training forward threads it
    seen.clear()
    bridge_sample(
        spy, torch.rand(2, 1, 8, 8), 0.1, 7.0, n_steps=1, sigma_bridge=0.0, contrast_id=cid
    )
    assert seen["cid"] is cid  # stochastic sampler threads it


def test_validation_step_declares_contrast_id_for_pipeline_seam() -> None:
    # The gated validation seam only forwards a field the signature DECLARES; without
    # contrast_id in validation_step, val-time sampling would drop it (metric on a
    # contrast-blind reconstruction, pitfall #18).
    import inspect

    params = inspect.signature(FieldBridgeStrategy.validation_step).parameters
    assert "contrast_id" in params
