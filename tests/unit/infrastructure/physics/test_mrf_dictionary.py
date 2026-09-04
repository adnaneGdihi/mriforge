"""Tests for the MRF Bloch dictionary.

Targets ``spectramr.infrastructure.physics.mrf_dictionary``. The
``BlochDictionary`` generates a fingerprint dictionary by simulating
Bloch responses across a (T1, T2) grid, then matches observed signals
to recover quantitative maps via dot-product matching.

Categories:

- Unit: generate populates buffers; match before generate raises.
- Property: dictionary entries are L2-normalised; matched dot-product
  recovers planted (T1, T2) within grid step.
- Sanity-shape: 4D and 2D signal layouts both supported.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.mrf_dictionary import BlochDictionary


def _coarse_dict() -> BlochDictionary:
    """Small dictionary suitable for fast tests."""
    return BlochDictionary(
        t1_range=(500.0, 1500.0),
        t2_range=(40.0, 100.0),
        t1_step=200.0,
        t2_step=20.0,
    )


def _short_sequence() -> list[dict[str, float]]:
    """A 4-time-point sequence that varies BOTH flip angle (T1 contrast)
    and TE (T2 contrast).

    A pure variable-flip-angle sequence at constant TE makes the
    fingerprint depend almost entirely on T1 — different T2 values
    produce numerically identical fingerprints (``exp(-TE/T2)`` for
    short TE is ~constant), which breaks dot-product matching. Varying
    TE across timepoints gives the matcher something to work with.
    """
    return [
        {"TR": 10.0, "TE": 2.0, "alpha": 5.0},
        {"TR": 10.0, "TE": 20.0, "alpha": 15.0},
        {"TR": 10.0, "TE": 50.0, "alpha": 30.0},
        {"TR": 10.0, "TE": 100.0, "alpha": 60.0},
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_populates_dictionary_and_params_buffers() -> None:
    """generate() fills the ``dictionary`` and ``params`` buffers."""
    bd = _coarse_dict()
    assert bd.dictionary is None
    assert bd.params is None
    bd.generate(_short_sequence())
    assert bd.dictionary is not None
    assert bd.params is not None
    # Dictionary: [N_entries, T_timepoints]
    assert bd.dictionary.dim() == 2
    assert bd.dictionary.shape[1] == len(_short_sequence())


def test_dictionary_entries_are_l2_normalised() -> None:
    """Each row of the dictionary has unit ℓ₂ norm (by construction)."""
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    norms = torch.linalg.vector_norm(bd.dictionary, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_params_shape_matches_dictionary_entries() -> None:
    """params is [N_entries, 3] for (T1, T2, B0)."""
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    n_entries = bd.dictionary.shape[0]
    assert bd.params.shape == (n_entries, 3)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_before_generate_raises() -> None:
    """Calling match() before generate() fails loud."""
    bd = _coarse_dict()
    with pytest.raises(RuntimeError, match="Dictionary not generated"):
        bd.match(torch.zeros(1, 4, 8, 8))


def test_match_recovers_planted_t1_t2() -> None:
    """Feeding a dictionary entry back recovers its (T1, T2) within grid step.

    Take a known dictionary entry, scale by a positive constant (PD),
    feed to match() — the recovered T1/T2 should equal the planted ones
    exactly (the dictionary atom is at most a grid_step away from
    itself).
    """
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    # Pick a few specific dictionary entries.
    test_idx = torch.tensor([0, 5, 10, 15])
    # Each row is a unit-norm fingerprint; PD-scale it.
    pd_scale = 2.5
    signals = pd_scale * bd.dictionary[test_idx]  # [4, T]
    est_t1, est_t2, est_b0 = bd.match(signals)
    # Compare to the planted T1, T2, B0. B0 is now a load-bearing axis: feeding
    # a complex dictionary atom back recovers its off-resonance (previously b0
    # was hardcoded to 0, so this asserted est_b0==0 — the inert-knob facade).
    expected_t1 = bd.params[test_idx, 0]
    expected_t2 = bd.params[test_idx, 1]
    expected_b0 = bd.params[test_idx, 2]
    assert torch.allclose(est_t1, expected_t1, atol=1e-3)
    assert torch.allclose(est_t2, expected_t2, atol=1e-3)
    assert torch.allclose(est_b0, expected_b0, atol=1e-3)


def test_real_magnitude_signal_recovers_zero_b0() -> None:
    """A real (magnitude) signal cannot encode off-resonance, so match() must
    return est_b0≈0 — the honest answer, not a spurious field."""
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    # Magnitude of a complex atom: real, phase-stripped.
    real_sig = bd.dictionary[3].abs().unsqueeze(0)  # [1, T] real
    _, _, est_b0 = bd.match(real_sig)
    assert torch.allclose(est_b0, torch.zeros_like(est_b0), atol=1e-6)


def test_b0_axis_is_load_bearing() -> None:
    """The dictionary must actually span the B0 grid (not a single 0-Hz slice)."""
    bd = BlochDictionary(
        t1_range=(800.0, 800.0),
        t2_range=(80.0, 80.0),
        b0_range=(-40.0, 40.0),
        b0_step=20.0,
    )
    bd.generate(_short_sequence())
    unique_b0 = torch.unique(bd.params[:, 2])
    assert unique_b0.numel() == 5  # {-40,-20,0,20,40}
    assert bd.dictionary.is_complex()


def test_match_handles_4d_signals() -> None:
    """Match accepts [B, T, H, W] signals and returns [B, H, W] maps."""
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    t = bd.dictionary.shape[1]
    # Build a [1, T, 4, 4] signal where every pixel == dictionary[0].
    fingerprint = bd.dictionary[0]  # [T]
    signals = fingerprint.view(1, t, 1, 1).expand(1, t, 4, 4).clone()
    est_t1, est_t2, est_b0 = bd.match(signals)
    assert est_t1.shape == (1, 4, 4)
    assert est_t2.shape == (1, 4, 4)
    assert est_b0.shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 4), (8, 4), (1, 4, 4, 4), (2, 4, 8, 8)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_match_shape_matrix(shape: tuple[int, ...]) -> None:
    """Match handles both 2D and 4D layouts across shape matrix."""
    bd = _coarse_dict()
    bd.generate(_short_sequence())
    # Build a signal matching the dimensionality.
    if len(shape) == 2:
        n_pixels, t = shape
        signals = bd.dictionary[:n_pixels].clone().repeat((1 + n_pixels // bd.dictionary.shape[0], 1))[:n_pixels]
    else:
        b, t, h, w = shape
        fp = bd.dictionary[0]
        signals = fp.view(1, t, 1, 1).expand(b, t, h, w).clone()
    est_t1, est_t2, est_b0 = bd.match(signals)
    assert torch.isfinite(est_t1).all()
    assert torch.isfinite(est_t2).all()
    assert torch.isfinite(est_b0).all()
