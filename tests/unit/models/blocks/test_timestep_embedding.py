"""Tests for the canonical sinusoidal timestep embedding.

The two properties that matter are the ones the previous `nn.Linear(1, ·)`
encoding failed: neighbouring timesteps must be DISTINGUISHABLE, and the
codes must not collapse when the schedule horizon is small (T=28 here, against
a 10000-base frequency ladder built for T~1000).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding


def test_shape_and_dtype():
    emb = sinusoidal_timestep_embedding(torch.arange(5), 64, max_timesteps=28.0)
    assert emb.shape == (5, 64)
    assert emb.dtype == torch.float32


def test_odd_dim_is_padded():
    emb = sinusoidal_timestep_embedding(torch.arange(3), 65, max_timesteps=28.0)
    assert emb.shape == (3, 65)


def test_adjacent_timesteps_are_distinguishable():
    """The defect this replaces: a rank-1 map of t puts every code on one line.
    Adjacent steps must be separated by a non-trivial margin after scaling."""
    t = torch.arange(28)
    emb = sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
    diffs = (emb[1:] - emb[:-1]).norm(dim=-1)
    assert diffs.min() > 1e-3, f"adjacent codes collapse; min gap {diffs.min():.2e}"


def test_all_timesteps_are_pairwise_distinct():
    t = torch.arange(28)
    emb = sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
    d = torch.cdist(emb, emb)
    off_diag = d + torch.eye(len(t)) * 1e9
    assert off_diag.min() > 1e-3, "two distinct timesteps share an embedding"


def test_unscaled_small_horizon_still_separates():
    """Sanity: without max_timesteps the codes are still distinct for small t —
    the scaling matters for CONDITIONING quality, not for raw injectivity."""
    emb = sinusoidal_timestep_embedding(torch.arange(28), 256)
    d = torch.cdist(emb, emb) + torch.eye(28) * 1e9
    assert d.min() > 1e-3


def test_codes_span_many_dimensions_not_a_line():
    """A linear encoding of ``t`` yields a RANK-1 matrix of codes — every
    timestep on one line, which is exactly why ``nn.Linear(1, ·)`` could not
    express "which t". The sinusoidal basis must span the full set instead.

    Computed in float64 deliberately: the frequency ladder spans ~4 orders of
    magnitude, so float32's default rank tolerance truncates a mathematically
    full-rank matrix to ~4. That is a tolerance artifact, not a property of the
    encoding, and asserting on it would pin the wrong thing.
    """
    emb = sinusoidal_timestep_embedding(torch.arange(28), 256, max_timesteps=28.0)
    assert torch.linalg.matrix_rank(emb.double()) == 28


def test_float_input_accepted():
    """Acceleration factors are continuous and are a legitimate conditioning
    signal on this path, so floats must not be silently truncated."""
    a = sinusoidal_timestep_embedding(torch.tensor([2.0]), 64)
    b = sinusoidal_timestep_embedding(torch.tensor([2.5]), 64)
    assert (a - b).norm() > 1e-4


@pytest.mark.parametrize("bad_dim", [0, 1])
def test_dim_below_two_raises(bad_dim):
    """``half_dim - 1`` divides; dim<2 would emit NaN rather than fail."""
    with pytest.raises(ValueError, match="dim must be"):
        sinusoidal_timestep_embedding(torch.arange(3), bad_dim)


@pytest.mark.parametrize("bad_max", [0.0, -1.0])
def test_non_positive_max_timesteps_raises(bad_max):
    with pytest.raises(ValueError, match="max_timesteps"):
        sinusoidal_timestep_embedding(torch.arange(3), 64, max_timesteps=bad_max)


def test_scaling_changes_the_code():
    """max_timesteps must actually be read — a silently ignored argument here
    would reintroduce the collapse it exists to prevent."""
    t = torch.arange(28)
    assert (
        sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
        - sinusoidal_timestep_embedding(t, 256, max_timesteps=1000.0)
    ).norm() > 1.0
