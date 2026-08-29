"""Canary / parametrized / edge-case / expected-failure tests for mrf_dictionary.py.

Covers BlochDictionary: generate(), match(), project_signal().

NOTE: generate() drives DifferentiableBlochSimulator internally with a small
dictionary so the test remains CPU-only and finishes in a few seconds.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_mrf_dictionary_import():
    """Module imports and BlochDictionary can be instantiated."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 200.0),
        t2_range=(20.0, 40.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    assert bd is not None


@pytest.mark.physics
def test_canary_mrf_generate_small():
    """generate() with a small dictionary and 2 time points completes."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 200.0),
        t2_range=(20.0, 40.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    seq_params = [
        {"TR": 10.0, "TE": 2.0, "alpha": 10.0},
        {"TR": 15.0, "TE": 3.0, "alpha": 15.0},
    ]
    dictionary, params = bd.generate(seq_params)
    assert dictionary is not None
    assert params is not None
    assert dictionary.dim() == 2
    n_entries = dictionary.shape[0]
    assert params.shape == (n_entries, 3)


# ---------------------------------------------------------------------------
# Parametrized — generate + match
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "n_timepoints, h, w",
    [
        (2, 4, 4),
        (3, 8, 8),
        (2, 4, 8),   # rectangular
    ],
    ids=["tp2-4x4", "tp3-8x8", "tp2-4x8-rect"],
)
def test_mrf_match_output_shape(n_timepoints, h, w):
    """match() returns maps with [B, H, W] for image-format signals."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 300.0),
        t2_range=(20.0, 60.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    seq_params = [{"TR": 10.0, "TE": 2.0, "alpha": float(8 + i)} for i in range(n_timepoints)]
    bd.generate(seq_params)

    # Simulate signals as [B, T, H, W]
    B = 1
    signals = torch.randn(B, n_timepoints, h, w)
    est_t1, est_t2, est_b0 = bd.match(signals)
    assert est_t1.shape == (B, h, w)
    assert est_t2.shape == (B, h, w)
    assert est_b0.shape == (B, h, w)


@pytest.mark.physics
def test_mrf_match_1d_signals():
    """match() accepts flat [N, T] signals and returns flat [N] maps."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 300.0),
        t2_range=(20.0, 60.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    n_tp = 3
    seq_params = [{"TR": 10.0, "TE": 2.0, "alpha": float(10 + i)} for i in range(n_tp)]
    bd.generate(seq_params)
    signals = torch.randn(16, n_tp)     # 16 flat pixels
    est_t1, est_t2, est_b0 = bd.match(signals)
    assert est_t1.shape == (16,)
    assert est_t2.shape == (16,)


@pytest.mark.physics
def test_mrf_project_signal_shape():
    """project_signal() returns same shape as input."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 300.0),
        t2_range=(20.0, 60.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    n_tp = 2
    seq_params = [{"TR": 10.0, "TE": 2.0, "alpha": float(10 + i)} for i in range(n_tp)]
    bd.generate(seq_params)
    signals = torch.randn(1, n_tp, 4, 4)
    projected = bd.project_signal(signals)
    assert projected.shape == signals.shape


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_mrf_edge_match_uniform_signal():
    """Matching a uniform signal tensor returns valid T1/T2 without error."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 300.0),
        t2_range=(20.0, 60.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    bd.generate([{"TR": 10.0, "TE": 2.0, "alpha": 10.0}] * 2)
    signals = torch.ones(1, 2, 4, 4)
    est_t1, est_t2, _ = bd.match(signals)
    assert not torch.isnan(est_t1).any()
    assert not torch.isnan(est_t2).any()


@pytest.mark.physics
def test_mrf_edge_match_zero_signal():
    """Matching an all-zero signal tensor completes (norm denom is eps-guarded)."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(
        t1_range=(100.0, 300.0),
        t2_range=(20.0, 60.0),
        b0_range=(0.0, 0.0),
        t1_step=100.0,
        t2_step=20.0,
        b0_step=1.0,
        device="cpu",
    )
    bd.generate([{"TR": 10.0, "TE": 2.0, "alpha": 10.0}] * 2)
    signals = torch.zeros(1, 2, 4, 4)
    est_t1, est_t2, _ = bd.match(signals)
    # should not raise and should not produce NaN
    assert not torch.isnan(est_t1).any()


# ---------------------------------------------------------------------------
# Expected-failure tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_mrf_raises_if_match_called_before_generate():
    """match() must raise RuntimeError when called without generate()."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(device="cpu")
    signals = torch.randn(1, 2, 4, 4)
    with pytest.raises(RuntimeError, match="Dictionary not generated"):
        bd.match(signals)


@pytest.mark.physics
def test_mrf_raises_project_before_generate():
    """project_signal() must raise RuntimeError when dictionary is not built."""
    from mriforge.infrastructure.physics.mrf_dictionary import BlochDictionary
    bd = BlochDictionary(device="cpu")
    with pytest.raises(RuntimeError, match="Dictionary not generated"):
        bd.project_signal(torch.randn(1, 2, 4, 4))
