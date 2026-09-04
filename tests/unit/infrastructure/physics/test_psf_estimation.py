"""Tests for fiducial-measured point-spread-function estimation.

The claim is that a KNOWN input turns a blind deconvolution into a linear solve.
The decisive tests plant a kernel, run it forward, and check it comes back — and
check that the estimator's own limits (regularisation bias, spectral
identifiability) are reported rather than hidden.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.psf_estimation import (  # noqa: E402
    apply_psf,
    estimate_psf,
    gaussian_psf,
    psf_fwhm_map,
    psf_identifiability,
)
from spectramr.infrastructure.physics.virtual_fiducial import (  # noqa: E402
    VirtualFiducial,
)

FWHM_PER_SIGMA = 2.3548


def _marker(h: int = 64, sigma: float = 0.8, jitter: float = 0.45) -> torch.Tensor:
    return VirtualFiducial(im_size=(h, h), grid_spacing=8, sigma=sigma, jitter=jitter, seed=0)(
        1
    ).real


# ── the assumed kernel ───────────────────────────────────────────────────────


def test_gaussian_psf_is_normalised_and_matches_its_declared_width() -> None:
    k = gaussian_psf(11, 1.5).reshape(1, 1, 1, 11, 11)
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(psf_fwhm_map(k)) == pytest.approx(1.5 * FWHM_PER_SIGMA, rel=0.02)


def test_gaussian_psf_rejects_an_even_kernel() -> None:
    """An even kernel has no centre, so 'the' PSF would be off by half a pixel."""
    with pytest.raises(ValueError, match="odd"):
        gaussian_psf(8, 1.0)


# ── the solve ────────────────────────────────────────────────────────────────


def test_a_planted_kernel_is_recovered() -> None:
    """The whole construction in one test: blur a KNOWN marker with a known
    kernel, then solve it back."""
    marker = _marker()
    true_k = gaussian_psf(9, 1.8).reshape(1, 1, 1, 9, 9)
    est = estimate_psf(apply_psf(marker, true_k), marker, kernel_size=9, mu=1e-3)
    assert float(psf_fwhm_map(est)) == pytest.approx(float(psf_fwhm_map(true_k)), rel=0.05)
    assert float(est.sum()) == pytest.approx(1.0, abs=1e-5)


def test_a_wrong_assumption_is_detected() -> None:
    """The point of measuring: the estimate must track the TRUE blur, not the
    one an arm assumed."""
    marker = _marker()
    observed = apply_psf(marker, gaussian_psf(9, 2.4).reshape(1, 1, 1, 9, 9))
    measured = float(psf_fwhm_map(estimate_psf(observed, marker, mu=1e-3)))
    assumed = float(psf_fwhm_map(gaussian_psf(9, 1.2).reshape(1, 1, 1, 9, 9)))
    assert measured > assumed * 1.5


def test_spatially_varying_blur_is_resolved_across_control_points() -> None:
    """A single kernel is itself an assumption; a real low-field PSF varies."""
    h = 64
    marker = _marker(h)
    left = apply_psf(marker, gaussian_psf(9, 0.8).reshape(1, 1, 1, 9, 9))
    right = apply_psf(marker, gaussian_psf(9, 2.5).reshape(1, 1, 1, 9, 9))
    mixed = torch.cat([left[..., : h // 2], right[..., h // 2 :]], dim=-1)
    fwhm = psf_fwhm_map(
        estimate_psf(mixed, marker, kernel_size=9, mu=1e-3, control_grid=(1, 2))
    ).flatten()
    assert float(fwhm[1]) > float(fwhm[0])


def test_regularisation_trades_bias_against_null_amplification() -> None:
    """mu is not a free parameter. Too large over-smooths, too small amplifies
    the marker's spectral nulls; 1e-3 is the swept default."""
    marker = _marker()
    true_k = gaussian_psf(9, 1.8).reshape(1, 1, 1, 9, 9)
    observed = apply_psf(marker, true_k)
    truth = float(psf_fwhm_map(true_k))
    over = float(psf_fwhm_map(estimate_psf(observed, marker, mu=1e-1)))
    good = float(psf_fwhm_map(estimate_psf(observed, marker, mu=1e-3)))
    under = float(psf_fwhm_map(estimate_psf(observed, marker, mu=1e-6)))
    assert over < truth  # over-smoothed
    assert abs(good - truth) < abs(over - truth)
    assert abs(good - truth) < abs(under - truth)


def test_zero_regularisation_is_rejected() -> None:
    """At a spectral null the kernel is genuinely unidentifiable, and an
    unregularised solve there returns noise scaled by 1/0."""
    marker = _marker()
    with pytest.raises(ValueError, match="unidentifiable"):
        estimate_psf(marker, marker, mu=0.0)


# ── identifiability, reported not hoped for ──────────────────────────────────


def test_a_narrower_marker_excites_more_of_the_band() -> None:
    """The kernel is only recoverable where the marker has energy, so marker
    width is the lever on how much of the estimate is measurement rather than
    interpolation."""
    broad = float(psf_identifiability(_marker(sigma=1.5, jitter=0.35)))
    narrow = float(psf_identifiability(_marker(sigma=0.8, jitter=0.45)))
    assert narrow > broad
    assert 0.0 <= broad <= 1.0 and narrow > 0.8


# ── contracts ────────────────────────────────────────────────────────────────


def test_mismatched_grids_are_rejected() -> None:
    """The operator measured has to be the one inverted."""
    with pytest.raises(ValueError, match="must share a grid"):
        estimate_psf(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 64, 64))


def test_control_grid_must_divide_the_image() -> None:
    """Ragged patches would give control points different amounts of evidence
    while the interpolation weights them as equal."""
    with pytest.raises(ValueError, match="divide"):
        estimate_psf(_marker(64), _marker(64), control_grid=(1, 5))


def test_kernel_wider_than_its_patch_is_rejected() -> None:
    """A kernel wider than its evidence is extrapolation."""
    with pytest.raises(ValueError, match="exceeds"):
        estimate_psf(_marker(64), _marker(64), kernel_size=21, control_grid=(8, 8))


def test_apply_psf_preserves_shape_and_is_differentiable() -> None:
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    k = gaussian_psf(5, 1.0).reshape(1, 1, 1, 5, 5).expand(2, 1, 1, 5, 5)
    out = apply_psf(x, k)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_apply_psf_blends_smoothly_across_control_points() -> None:
    """Blockwise application would make the operator discontinuous, and a
    network trained to invert a discontinuous operator learns the
    discontinuity."""
    x = torch.ones(1, 1, 32, 32)
    kernels = torch.stack([gaussian_psf(5, 0.5), gaussian_psf(5, 2.0)], dim=0).reshape(
        1, 1, 2, 5, 5
    )
    out = apply_psf(x, kernels)
    # a constant field stays constant under any partition of unity of
    # unit-sum kernels: no seam, no gain
    assert float(out.std()) < 1e-5
    assert float(out.mean()) == pytest.approx(1.0, abs=1e-5)


def test_fwhm_of_a_delta_kernel_is_near_zero() -> None:
    k = torch.zeros(1, 1, 1, 5, 5)
    k[..., 2, 2] = 1.0
    assert float(psf_fwhm_map(k)) < 0.1


def test_fwhm_scales_with_voxel_size() -> None:
    k = gaussian_psf(9, 1.5).reshape(1, 1, 1, 9, 9)
    assert float(psf_fwhm_map(k, voxel_mm=2.0)) == pytest.approx(
        2.0 * float(psf_fwhm_map(k, voxel_mm=1.0)), rel=1e-5
    )
