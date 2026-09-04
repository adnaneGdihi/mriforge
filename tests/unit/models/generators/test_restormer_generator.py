"""Tests for the Restormer generator's upsampling tail.

The MDTA/GDFN body is exercised through the arm-level integration tests; what is
pinned here is the ``scale`` contract, because the super-resolution factor is
chosen by the PHYSICS (the measured ULF-to-3T resolution gap) and the backbone
has to be able to express it. Before 2026-07-26 the tail supported only
``{1, 2, 4}``, so a 3.26x acquisition gap had to be rounded to a power of two —
training one decimation and reporting against another.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.restormer_generator import (  # noqa: E402
    RestormerGenerator,
)


@pytest.mark.parametrize("scale", [1, 2, 3, 4, 5, 7])
def test_output_grid_matches_the_declared_scale(scale: int) -> None:
    model = RestormerGenerator(in_channels=4, out_channels=1, scale=scale, dim=16)
    out = model(torch.randn(1, 4, 16, 16))
    assert out.shape == (1, 1, 16 * scale, 16 * scale)


def test_power_of_two_tails_are_unchanged() -> None:
    """The cascaded x2 forms must keep their exact parameter count, or adding
    odd-factor support would silently re-shape every existing arm's weights.
    Both figures are read off the pre-change implementation at HEAD, not
    copied from the post-change one."""
    counts = {
        s: sum(
            p.numel()
            for p in RestormerGenerator(
                in_channels=24, out_channels=1, scale=s, dim=24
            ).parameters()
        )
        for s in (2, 4)
    }
    assert counts[2] == 1_346_544
    assert counts[4] == 1_367_280


def test_non_positive_scale_raises_rather_than_degrading() -> None:
    """No silent fallback to a default factor (CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="positive integer"):
        RestormerGenerator(in_channels=1, out_channels=1, scale=0, dim=8)


def test_odd_scale_is_differentiable_end_to_end() -> None:
    model = RestormerGenerator(in_channels=4, out_channels=1, scale=3, dim=16)
    model(torch.randn(1, 4, 8, 8)).pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
