"""Unit tests for the rank-normalize block (MCGI invariance core).

Targets ``mriforge.models.blocks.rank_normalize``.

The per-image empirical-CDF rank transform :math:`R(\\mathbf x)(v)=F_{\\mathbf
x}(\\mathbf x(v))` is *exactly* invariant to the group :math:`G_+` of strictly
increasing intensity remaps, because the empirical CDF transforms covariantly:
:math:`F_{g\\circ\\mathbf x}(g(t))=F_{\\mathbf x}(t)`. The hard-rank path is what
delivers the by-construction guarantee; the soft-rank path is the differentiable
training surrogate.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_hard_rank_exactly_invariant_to_monotone_remap() -> None:
    from mriforge.models.blocks.rank_normalize import rank_normalize

    torch.manual_seed(0)
    x = torch.rand(2, 1, 8, 8)

    # A strictly-increasing nonlinear contrast remap g.
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(2.0 * t) + 0.3 * t + 5.0

    r_x = rank_normalize(x, hard=True)
    r_gx = rank_normalize(g(x), hard=True)
    assert torch.equal(r_x, r_gx), "hard rank must be bit-exact under monotone remap"


def test_hard_rank_in_unit_interval() -> None:
    from mriforge.models.blocks.rank_normalize import rank_normalize

    x = torch.randn(3, 2, 5, 5)
    r = rank_normalize(x, hard=True)
    assert r.shape == x.shape
    assert float(r.min()) >= 0.0
    assert float(r.max()) <= 1.0


def test_rank_is_per_image_per_channel() -> None:
    """Ranks are computed within each (batch, channel) map independently."""
    from mriforge.models.blocks.rank_normalize import rank_normalize

    x = torch.zeros(2, 1, 1, 3)
    x[0, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    x[1, 0, 0] = torch.tensor([-5.0, -1.0, 0.0])
    r = rank_normalize(x, hard=True)
    # Each row maps its own min->0, max->1 regardless of the other row's scale.
    assert torch.allclose(r[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(r[1, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_soft_rank_is_differentiable() -> None:
    from mriforge.models.blocks.rank_normalize import rank_normalize

    x = torch.randn(1, 1, 4, 4, requires_grad=True)
    r = rank_normalize(x, hard=False, temperature=0.1)
    r.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_soft_rank_approaches_hard_as_temperature_drops() -> None:
    from mriforge.models.blocks.rank_normalize import rank_normalize

    torch.manual_seed(1)
    x = torch.randn(1, 1, 6, 6)
    hard = rank_normalize(x, hard=True)
    soft = rank_normalize(x, hard=False, temperature=1e-3)
    assert torch.allclose(hard, soft, atol=0.05)


def test_block_module_matches_functional() -> None:
    from mriforge.models.blocks.rank_normalize import RankNormalize2d, rank_normalize

    x = torch.randn(2, 1, 7, 7)
    block = RankNormalize2d(hard_eval=True)
    block.eval()
    assert torch.equal(block(x), rank_normalize(x, hard=True))
