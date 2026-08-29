"""Regression tests for chi_square_diffusion.

Covers two real bugs surfaced by the WS-6 audit:

1. ``dof_schedule='exponential'`` referenced the non-existent
   ``SchedulerTypes.EXPONENTIAL`` member -> AttributeError instead of building
   the log-spaced schedule (and the ``else: raise ValueError`` was unreachable).
2. ``ChiSquareColdDiffusion._degrade`` dropped the ``mask`` keyword the parent
   ``ColdDiffusion.p_sample`` passes -> TypeError on any generalized cold-sample
   path.
"""

import pytest
import torch

from mriforge.models.diffusion.chi_square_diffusion import (
    ChiSquareColdDiffusion,
    ChiSquareDiffusion,
)


def test_exponential_dof_schedule_builds():
    m = ChiSquareDiffusion(timesteps=8, dof_schedule="exponential", device="cpu")
    assert m.dof.shape[0] == 8


def test_linear_and_cosine_still_build():
    assert ChiSquareDiffusion(timesteps=8, dof_schedule="linear", device="cpu").dof.shape[0] == 8
    assert ChiSquareDiffusion(timesteps=8, dof_schedule="cosine", device="cpu").dof.shape[0] == 8


def test_unknown_dof_schedule_raises_valueerror():
    with pytest.raises(ValueError):
        ChiSquareDiffusion(timesteps=8, dof_schedule="bogus", device="cpu")


def test_cold_degrade_accepts_mask_kwarg():
    """Parent ColdDiffusion.p_sample calls self._degrade(..., mask=mask)."""
    torch.manual_seed(0)
    m = ChiSquareColdDiffusion(timesteps=4, device="cpu")
    x = torch.randn(2, 2, 8, 8)
    t = torch.tensor([1, 1])

    # Must not raise TypeError on the keyword the parent passes.
    out_with_mask = m._degrade(x, t, mask=None)
    assert out_with_mask.shape == x.shape


def test_cold_degrade_mask_is_ignored_numerically():
    """mask is accepted for contract-compatibility only; output is unchanged."""
    m = ChiSquareColdDiffusion(timesteps=4, device="cpu")
    x = torch.randn(2, 2, 8, 8)
    t = torch.tensor([1, 1])

    torch.manual_seed(123)
    out_no_mask = m._degrade(x, t)
    torch.manual_seed(123)
    out_with_mask = m._degrade(x, t, mask=torch.ones(2, 1, 8, 8))

    assert torch.allclose(out_no_mask, out_with_mask)


class _ScaledIdentity(torch.nn.Module):
    def forward(self, x, t, cond=None):
        return 0.9 * x


class TestColdPLossesRegistryRouting:
    """WS-6 plan #6: the elementary loss dispatch is registry-routed and
    numerically identical to the replaced functionals."""

    def _setup(self, monkeypatch):
        m = ChiSquareColdDiffusion(timesteps=4, device="cpu")
        # Deterministic degradation: bypass the chi-square sampling noise.
        monkeypatch.setattr(
            m.chi_square_sampler, "q_sample", lambda x_start, t, noise=None: 0.5 * x_start
        )
        torch.manual_seed(0)
        x_start = torch.randn(2, 2, 8, 8)
        t = torch.tensor([1, 2])
        return m, x_start, t

    def test_matches_replaced_functionals(self, monkeypatch):
        import torch.nn.functional as F

        m, x_start, t = self._setup(monkeypatch)
        predicted = 0.9 * (0.5 * x_start)
        expected = {
            "l1": F.l1_loss(predicted, x_start),
            "l2": F.mse_loss(predicted, x_start),
            "huber": F.smooth_l1_loss(predicted, x_start),
        }
        model = _ScaledIdentity()
        for name, exp in expected.items():
            got = m.p_losses(model, x_start, t, loss_type=name)
            assert torch.allclose(got, exp), name

    def test_unknown_loss_type_raises(self, monkeypatch):
        m, x_start, t = self._setup(monkeypatch)
        with pytest.raises(ValueError, match="Unsupported loss type"):
            m.p_losses(_ScaledIdentity(), x_start, t, loss_type="nope")
