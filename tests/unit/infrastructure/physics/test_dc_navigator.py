r"""Tests for ``DCNavigatorExtractor``.

Targets ``mriforge.infrastructure.physics.dc_navigator``. Pulls the DC
component :math:`k_0(t) = k(0, 0, t)` of a k-space sequence and
spectrally decomposes it into respiratory and cardiac bands.

Categories:

- ``extract_dc``: single-point and ``center_radius > 0`` averaging
- ``extract_dc`` accepts both 4-D and 5-D inputs
- ``_bandpass``: zero outside the configured band, near-pass-through
  for in-band sinusoids
- ``decompose``: at the configured fs, a synthesized respiratory +
  cardiac signal is recovered into the right buckets
- ``forward`` produces all four output keys
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.dc_navigator import DCNavigatorExtractor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction_holds_documented_bands() -> None:
    """Default respiratory and cardiac bands match the docstring."""
    nav = DCNavigatorExtractor()
    assert nav.sample_rate == 500.0
    assert nav.resp_band == (0.15, 0.5)
    assert nav.card_band == (0.8, 2.0)
    assert nav.center_radius == 0


# ---------------------------------------------------------------------------
# extract_dc
# ---------------------------------------------------------------------------


def test_extract_dc_single_point() -> None:
    """``center_radius=0`` → returns just the centre voxel of each frame."""
    T, B, C, H, W = 4, 1, 1, 8, 8
    k = torch.zeros(T, B, C, H, W, dtype=torch.complex64)
    # Set the centre to a known sequence.
    centres = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex64)
    k[:, 0, 0, H // 2, W // 2] = centres
    nav = DCNavigatorExtractor(center_radius=0)
    out = nav.extract_dc(k)
    assert out.shape == (T,)
    assert torch.allclose(out, centres)


def test_extract_dc_averages_within_radius() -> None:
    """``center_radius=1`` averages a 3×3 neighbourhood."""
    T, B, C, H, W = 1, 1, 1, 8, 8
    k = torch.zeros(T, B, C, H, W, dtype=torch.complex64)
    cy, cx = H // 2, W // 2
    # Set the 3×3 centre block to a constant.
    k[0, 0, 0, cy - 1 : cy + 2, cx - 1 : cx + 2] = 2.0
    nav = DCNavigatorExtractor(center_radius=1)
    out = nav.extract_dc(k)
    assert torch.allclose(out, torch.tensor([2.0 + 0.0j]))


def test_extract_dc_accepts_4d_input() -> None:
    """4-D ``[B, C, H, W]`` input is treated as ``T = 1``."""
    k = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
    k[0, 0, 4, 4] = 5.0 + 1j
    nav = DCNavigatorExtractor(center_radius=0)
    out = nav.extract_dc(k)
    assert out.shape == (1,)


def test_extract_dc_center_matches_fft2c_for_even_sizes() -> None:
    """For EVEN H/W, ``extract_dc`` reads the exact ``fft2c`` DC bin.

    Documented contract: the DC bin is read at ``[H // 2, W // 2]``, which
    coincides with the ``fft2c`` DC location for even grids. A uniform image
    has all spectral energy at DC, so ``extract_dc`` returns the maximum-
    magnitude bin and reading the wrong bin would return (near) zero.
    """
    from mriforge.infrastructure.physics.fft_ops import fft2c

    nav = DCNavigatorExtractor(center_radius=0)
    for n in (8, 6):
        img = torch.ones(1, 1, n, n, dtype=torch.complex64)
        ksp = fft2c(img)  # centered: all DC energy at [n//2, n//2]
        flat_argmax = ksp.abs().reshape(-1).argmax().item()
        cy, cx = divmod(flat_argmax, n)
        assert (cy, cx) == (n // 2, n // 2)  # fft2c DC bin == H//2 for even n
        dc = nav.extract_dc(ksp)
        assert torch.allclose(dc, ksp[:, :, n // 2, n // 2].mean(dim=(1,)))
        # DC magnitude == total energy of the uniform image (sanity).
        assert dc.abs().item() == pytest.approx(float(n), rel=1e-4)


def test_extract_dc_reads_the_true_dc_bin_for_odd_sizes() -> None:
    """``H // 2`` is the true ``fft2c`` DC bin at EVERY parity.

    This test previously pinned an "off-by-one limitation" of the navigator on
    odd grids and blamed its floor-divided ``H // 2`` index. The index was never
    wrong: ``fft2c`` was, applying ``fftshift`` where ``ifftshift`` was needed and
    so landing DC one bin past centre on odd axes. The old docstring asked that a
    change to the centering convention be "a deliberate, reviewed one" — this is
    it. With the SSOT fixed, the navigator reads the real DC and the whole
    uniform-image energy comes back.
    """
    from mriforge.infrastructure.physics.fft_ops import fft2c

    nav = DCNavigatorExtractor(center_radius=0)
    for n in (5, 7):
        img = torch.ones(1, 1, n, n, dtype=torch.complex64)
        ksp = fft2c(img)
        flat_argmax = ksp.abs().reshape(-1).argmax().item()
        assert divmod(flat_argmax, n) == (n // 2, n // 2)

        dc = nav.extract_dc(ksp)
        assert torch.allclose(dc, ksp[:, :, n // 2, n // 2].mean(dim=(1,)))
        assert dc.abs().item() == pytest.approx(float(n), rel=1e-4)


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------


def test_bandpass_zero_outside_band() -> None:
    """A pure-tone outside the band is suppressed to ≈ 0.

    With T=1024 samples at fs=100 Hz the FFT bin width is ≈0.098 Hz,
    so a 0.05 Hz tone sits between bins 0 and 1 and the rectangular
    window leaks ~6% of its energy into adjacent bins (including some
    inside the pass-band). The ideal brick-wall bandpass can't suppress
    that leakage further — anything below ~10% residual amplitude
    means the bandpass is doing its job. The previous 5% threshold
    was tighter than the leakage floor of the chosen test signal.
    """
    fs = 100.0
    T = 1024
    t = torch.linspace(0, T / fs, T)
    # Out-of-band signal at 0.05 Hz (below resp band).
    signal = torch.sin(2 * math.pi * 0.05 * t).to(torch.complex64)
    filtered = DCNavigatorExtractor._bandpass(signal, fs, 0.15, 0.5)
    # Most energy removed (>90% suppression vs unit-amplitude input).
    assert filtered.abs().mean().item() < 0.1


def test_bandpass_passes_in_band() -> None:
    """A pure-tone inside the band passes through (non-zero amplitude)."""
    fs = 100.0
    T = 1024
    t = torch.linspace(0, T / fs, T)
    # In-band signal at 0.3 Hz (inside resp band).
    signal = torch.sin(2 * math.pi * 0.3 * t).to(torch.complex64)
    filtered = DCNavigatorExtractor._bandpass(signal, fs, 0.15, 0.5)
    # Most energy preserved.
    assert filtered.abs().mean().item() > 0.4


# ---------------------------------------------------------------------------
# Decompose
# ---------------------------------------------------------------------------


def test_decompose_returns_three_components() -> None:
    """``decompose`` returns the three expected keys."""
    nav = DCNavigatorExtractor()
    sig = torch.zeros(64, dtype=torch.complex64)
    out = nav.decompose(sig)
    assert {"respiratory", "cardiac", "residual"} == set(out.keys())


def test_decompose_separates_synthetic_signal() -> None:
    """Resp + cardiac sinusoids land in their respective bands."""
    fs = 100.0
    T = 2048
    t = torch.linspace(0, T / fs, T)
    resp = torch.sin(2 * math.pi * 0.3 * t)  # in resp band
    card = torch.sin(2 * math.pi * 1.2 * t)  # in cardiac band
    signal = (resp + card).to(torch.complex64)

    nav = DCNavigatorExtractor(sample_rate=fs)
    out = nav.decompose(signal)
    # Each component carries non-trivial energy.
    assert out["respiratory"].abs().mean().item() > 0.3
    assert out["cardiac"].abs().mean().item() > 0.3
    # Residual is small (most energy in the two bands).
    assert out["residual"].abs().mean().item() < (out["respiratory"].abs().mean().item())


# ---------------------------------------------------------------------------
# forward (full pipeline)
# ---------------------------------------------------------------------------


def test_forward_returns_all_keys() -> None:
    """``forward`` returns dc + the three decomposed bands."""
    fs = 100.0
    T = 256
    k = torch.zeros(T, 1, 1, 8, 8, dtype=torch.complex64)
    # Inject a respiratory tone into the DC bin.
    centres = torch.sin(2 * math.pi * 0.3 * torch.linspace(0, T / fs, T)).to(torch.complex64)
    k[:, 0, 0, 4, 4] = centres

    nav = DCNavigatorExtractor(sample_rate=fs)
    out = nav(k)
    assert {"dc", "respiratory", "cardiac", "residual"} == set(out.keys())
    assert out["dc"].shape == (T,)
