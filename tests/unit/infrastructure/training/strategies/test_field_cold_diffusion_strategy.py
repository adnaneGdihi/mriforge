"""Tests for FieldColdDiffusionStrategy (B-2.4)."""

from __future__ import annotations

import torch
from torch import nn

from spectramr.infrastructure.training.strategies.field_cold_diffusion_strategy import (
    cold_diffusion_sample,
    compute_field_cold_diffusion_loss,
    field_degrade,
)


class _IdentityRestorer(nn.Module):
    """R(x_t, t) = x_t (identity) — for exact sampler/degradation tests."""

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_):
        return x


class _RecordingRestorer(nn.Module):
    """Identity restorer that records the ``contrast_id`` kwarg it last received."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_contrast: object = "UNSET"

    def forward(
        self, x: torch.Tensor, *, field_strength: torch.Tensor, contrast_id=None, **_
    ):
        self.seen_contrast = contrast_id
        return x


def test_degradation_is_progressive_lowpass() -> None:
    from spectramr.infrastructure.physics.fft_ops import fft2c

    x = torch.rand(1, 1, 16, 16)
    d0 = field_degrade(x, 0, 10, 0.25, 5.0)
    assert torch.allclose(d0, x)  # t=0 is identity
    d_mid = field_degrade(x, 5, 10, 0.25, 5.0)
    d_end = field_degrade(x, 10, 10, 0.25, 5.0)
    hf = lambda im: fft2c(im).abs()[..., :3, :3].sum()  # noqa: E731 corner band
    # more degradation (higher t) => less high-frequency energy
    assert float(hf(d_end)) <= float(hf(d_mid)) <= float(hf(x)) + 1e-4


def test_sample_recovers_start_with_identity_restorer() -> None:
    # With an identity restorer, x0_hat = x_t and the Bansal update telescopes:
    # x_{t-1} = x_t - D_t(x_t) + D_{t-1}(x_t). Starting from a sharp image the
    # sampler should return a finite image of the same shape (smoke for the loop).
    x = torch.rand(1, 1, 16, 16)
    out = cold_diffusion_sample(_IdentityRestorer(), x, n_steps=5, min_cutoff=0.25, max_cutoff=5.0)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_loss_keys_and_finite() -> None:
    out = compute_field_cold_diffusion_loss(
        _IdentityRestorer(), torch.rand(2, 1, 16, 16), train_steps=10,
        min_cutoff=0.25, max_cutoff=5.0,
    )
    assert {"loss_total", "loss_restore"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_geometric_schedule_per_step_increments_are_balanced() -> None:
    # Geometric cutoff schedule => roughly comparable per-step image-space
    # degradation (linear-in-cutoff put ~80% in the last 2 steps; review finding).
    x = torch.rand(1, 1, 32, 32)
    n = 10
    degraded = [field_degrade(x, t, n, 0.25, 5.0) for t in range(n + 1)]
    incr = [
        float((degraded[t] - degraded[t - 1]).abs().mean()) for t in range(1, n + 1)
    ]
    lo, hi = min(i for i in incr if i > 0), max(incr)
    assert hi / lo < 15.0  # was ~70x with the linear schedule


def test_loss_reduces_with_a_real_restorer() -> None:
    from spectramr.models.generators.field_velocity_unet import FieldVelocityUNet

    torch.manual_seed(0)
    net = FieldVelocityUNet(width=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    target = torch.rand(2, 1, 16, 16)
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_field_cold_diffusion_loss(net, target, train_steps=4, min_cutoff=0.5, max_cutoff=5.0)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_cold_diffusion" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "field_cold_diffusion" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter
    from spectramr.infrastructure.training.strategies.field_cold_diffusion_strategy import (
        FieldColdDiffusionStrategy,
    )
    from spectramr.models.generators.field_velocity_unet import FieldVelocityUNet

    tb = BatchAdapter.from_dict(
        {
            "input": torch.rand(2, 1, 16, 16),
            "target": torch.rand(2, 1, 16, 16),
            "field_strength": torch.tensor([0.1, 0.1]),
            "field_strength_target": torch.tensor([7.0, 1.5]),
        }
    )
    strat = object.__new__(FieldColdDiffusionStrategy)
    strat.env = types.SimpleNamespace(generator=FieldVelocityUNet(width=16))

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_loss_threads_contrast_id_to_restorer() -> None:
    rec = _RecordingRestorer()
    cid = torch.tensor([0, 1])
    compute_field_cold_diffusion_loss(
        rec,
        torch.rand(2, 1, 16, 16),
        train_steps=4,
        min_cutoff=0.25,
        max_cutoff=5.0,
        contrast_id=cid,
    )
    assert rec.seen_contrast is cid


def test_sample_threads_contrast_id_to_restorer() -> None:
    rec = _RecordingRestorer()
    cid = torch.tensor([2])
    cold_diffusion_sample(
        rec,
        torch.rand(1, 1, 16, 16),
        n_steps=3,
        min_cutoff=0.25,
        max_cutoff=5.0,
        contrast_id=cid,
    )
    assert rec.seen_contrast is cid


def test_compute_losses_forwards_batch_contrast_id() -> None:
    # The strategy must extract the batch's contrast_id and thread it to the
    # restorer, so a contrast-conditioned FieldVelocityUNet actually sees it (#15).
    import types

    from spectramr.data.batch_types import BatchAdapter
    from spectramr.infrastructure.training.strategies.field_cold_diffusion_strategy import (
        FieldColdDiffusionStrategy,
    )

    rec = _RecordingRestorer()
    tb = BatchAdapter.from_dict(
        {
            "input": torch.rand(2, 1, 16, 16),
            "target": torch.rand(2, 1, 16, 16),
            "field_strength": torch.tensor([0.1, 0.1]),
            "field_strength_target": torch.tensor([7.0, 1.5]),
            "contrast_id": torch.tensor([0, 2]),
        }
    )
    strat = object.__new__(FieldColdDiffusionStrategy)
    strat.env = types.SimpleNamespace(generator=rec)
    strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert isinstance(rec.seen_contrast, torch.Tensor)
    assert torch.equal(rec.seen_contrast, torch.tensor([0, 2]))


def test_validation_forward_threads_contrast_id() -> None:
    import types

    from spectramr.infrastructure.training.strategies.field_cold_diffusion_strategy import (
        FieldColdDiffusionStrategy,
    )

    rec = _RecordingRestorer()
    strat = object.__new__(FieldColdDiffusionStrategy)
    strat.env = types.SimpleNamespace(generator=rec)
    strat._fcd_n_steps = 3
    strat._fcd_min = 0.25
    strat._fcd_max = 5.0
    cid = torch.tensor([1])
    strat._validation_forward(
        None, {"input": torch.rand(1, 1, 16, 16), "contrast_id": cid}
    )
    assert rec.seen_contrast is cid
