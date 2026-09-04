"""#347: RiemannianMRFDiffusionStrategy must not lose its fingerprint in silence.

``kwargs.get("fingerprint")`` returned ``None`` when the batch carried no
fingerprint, and the strategy ran on as an *unconditional* Riemannian score
model on Bloch parameters. That is a reasonable model and it is not
fingerprinting: the conditioning is the entire thing that distinguishes MRF
from generic quantitative parameter modelling.

Nothing showed. ``riemannian_mrf_dsm`` stayed finite and decreased smoothly,
while ``MRFTangentScore``'s fingerprint encoder collected zero gradient for the
whole run. Pinned here because the fix is what unblocks the FINGERPRINTING tag
(#342's deliberate non-tag).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.mrf_kspace_strategies import (
    RiemannianMRFDiffusionStrategy,
)


class _Score(nn.Module):
    """Records whether it was handed a fingerprint."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(5, 5)
        self.seen_fingerprint = "not-called"

    def forward(self, theta, sigma, fingerprint=None):
        self.seen_fingerprint = fingerprint
        return self.lin(theta)


def _strategy(model):
    s = object.__new__(RiemannianMRFDiffusionStrategy)
    s.sigma_min = 1e-2
    s.sigma_max = 1.0
    s.env = types.SimpleNamespace(generator=model)
    return s


def _bloch_params(b=4):
    """[B, 5] = (T1, T2, M0, B0, B1), T2 < T1 so the chart is defined."""
    return torch.stack(
        [
            torch.full((b,), 1000.0),
            torch.full((b,), 80.0),
            torch.full((b,), 1.0),
            torch.zeros(b),
            torch.full((b,), 1.0),
        ],
        dim=-1,
    )


def test_missing_fingerprint_raises_naming_the_key() -> None:
    model = _Score()
    with pytest.raises(ValueError, match="'fingerprint' key on the batch"):
        _strategy(model)._compute_losses_impl(torch.randn(4, 5), _bloch_params(), epoch=0)


def test_the_raise_says_why_unconditional_is_not_fingerprinting() -> None:
    """A bare 'missing key' would read as a plumbing slip; it is a claim."""
    model = _Score()
    with pytest.raises(ValueError) as exc:
        _strategy(model)._compute_losses_impl(torch.randn(4, 5), _bloch_params(), epoch=0)
    message = str(exc.value)
    assert "UNCONDITIONAL" in message
    assert "declared config mode" in message


def test_a_supplied_fingerprint_reaches_the_model() -> None:
    model = _Score()
    fingerprint = torch.randn(4, 32)
    out = _strategy(model)._compute_losses_impl(
        torch.randn(4, 5), _bloch_params(), epoch=0, fingerprint=fingerprint
    )
    assert model.seen_fingerprint is fingerprint, (
        "the conditioning must arrive at the score net, not be dropped en route"
    )
    assert torch.isfinite(out["g_total_loss"])
    assert out["g_total_loss"].requires_grad


def test_wrong_bloch_width_still_raises_first() -> None:
    """The pre-existing [..., 5] guard must not be shadowed by the new one."""
    with pytest.raises(ValueError, match=r"\[\.\.\., 5\] Bloch params"):
        _strategy(_Score())._compute_losses_impl(torch.randn(4, 3), torch.randn(4, 3), epoch=0)
