"""Measure the acquisition's point-spread function from a known fiducial.

Every super-resolution arm here assumes a forward operator: the anatomy is
blurred and decimated, and the network inverts that. The blur is normally
*assumed* — a Gaussian of some declared width — and if the assumption is wrong
the network spends capacity correcting a model error it was never told about,
while the reported degradation and the real one quietly differ.

The effective PSF of a real low-field acquisition is neither Gaussian nor
spatially uniform: gradient non-linearity, off-resonance and coil profile all
vary across the field of view. It is also unobservable on anatomy, because
solving for a blur kernel from a blurred image alone is a blind deconvolution.

The fiducial removes the blindness. Its un-blurred form is known exactly, so
observing what the acquisition does to it turns a blind problem into a linear
one: given input :math:`m` and output :math:`y = h * m`, solve for :math:`h`.
Doing it on a coarse control grid and interpolating with a partition of unity
recovers the *spatial variation* as well.

The estimator is ridge-regularised in the Fourier domain,

.. math::

    \\hat H(f) = \\frac{\\overline{M(f)}\\, Y(f)}{|M(f)|^2 + \\mu},

which is the Wiener/Tikhonov solution to
:math:`\\min_h \\|h * m - y\\|^2 + \\mu \\|h\\|^2`. The regulariser is not
cosmetic: the marker has spectral nulls, and at those frequencies the kernel is
genuinely unidentifiable. :math:`\\mu` decides whether that shows up as a
bounded, slightly biased estimate or as amplified noise.

One bias is worth stating because it is not obvious. When the estimate is made
on a POOLED grid -- which it must be, or the pooling and the interpolation used
to undo it are folded into the kernel -- pooling and convolution do not commute,
so the LR-domain effective kernel of an HR-domain blur is not exactly the
downscaled kernel. The residual reads as a POSITIVE FWHM bias, large when the
blur is comparable to the pooling factor and small when it dominates: +0.63 px
at a simulated sigma of 1.5 against +0.05 px at 2.4, both at a pooling factor of
2. Measure where the blur is well resolved, and do not read a small positive
error as a detected model error.

Because the marker's spectrum is known, the *identifiability* of the estimate is
computable rather than hoped for — see :func:`psf_identifiability`, which
reports the fraction of frequency support the marker actually excites.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import interpolate

from spectramr.infrastructure.physics.subpixel_registration import (
    centred_fft,
    centred_ifft,
)

__all__ = [
    "apply_psf",
    "estimate_psf",
    "gaussian_psf",
    "psf_fwhm_map",
    "psf_identifiability",
]


def _crop_centre(x: Tensor, size: int) -> Tensor:
    """Central ``size x size`` crop of a ``[..., H, W]`` tensor."""
    h, w = x.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"kernel_size={size} exceeds the {h}x{w} grid")
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top : top + size, left : left + size]


def gaussian_psf(kernel_size: int, sigma_px: float, device=None, dtype=None) -> Tensor:
    """The ASSUMED kernel every arm falls back to: an isotropic Gaussian.

    Kept here rather than inline so the assumed and measured operators are
    built by the same code path and an arm can swap one for the other without
    changing anything else.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and >= 1, got {kernel_size}")
    if sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be > 0, got {sigma_px}")
    ax = torch.arange(kernel_size, device=device, dtype=dtype or torch.float32)
    ax = ax - (kernel_size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma_px**2))
    k = torch.outer(g, g)
    return k / k.sum()


def estimate_psf(
    observed: Tensor,
    known: Tensor,
    *,
    kernel_size: int = 9,
    mu: float = 1e-3,
    control_grid: tuple[int, int] | None = None,
) -> Tensor:
    """Solve for the blur kernel that maps ``known`` to ``observed``.

    Args:
        observed: ``[B, 1, H, W]`` the fiducial as the acquisition rendered it.
        known: ``[B, 1, H, W]`` the fiducial as it truly is. Same grid — an
            arm that pools the marker must upsample it back before calling
            this, so the operator being measured is the one being inverted.
        kernel_size: Odd spatial extent of the estimated kernel.
        mu: Ridge weight. The marker has spectral nulls where the kernel is
            genuinely unidentifiable; ``mu`` decides whether that appears as a
            bounded biased estimate or as amplified noise. Too small is worse
            than too large.
        control_grid: ``(rows, cols)`` of patches for a SPATIALLY VARYING
            estimate. ``None`` gives one global kernel. A real low-field PSF
            varies across the field of view, so a single kernel is itself an
            assumption.

    Returns:
        ``[B, R, C, kernel_size, kernel_size]`` kernels, normalised to unit sum
        so they are averaging operators and cannot smuggle in a gain. ``R = C =
        1`` when ``control_grid`` is ``None``. Differentiable in both inputs.
    """
    if observed.shape != known.shape:
        raise ValueError(
            f"observed {tuple(observed.shape)} and known {tuple(known.shape)} "
            "must share a grid: the operator measured has to be the one inverted"
        )
    if observed.ndim != 4 or observed.shape[1] != 1:
        raise ValueError(f"expected [B, 1, H, W], got {tuple(observed.shape)}")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"kernel_size must be odd and >= 1, got {kernel_size}")
    if mu <= 0.0:
        raise ValueError(
            f"mu must be > 0, got {mu}. At a spectral null of the marker the "
            "kernel is unidentifiable, and an unregularised solve there returns "
            "noise scaled by 1/0."
        )

    rows, cols = control_grid or (1, 1)
    b, _, h, w = observed.shape
    if h % rows or w % cols:
        raise ValueError(
            f"control_grid {(rows, cols)} must divide the {h}x{w} grid evenly; "
            "ragged patches would give control points different amounts of "
            "evidence and the interpolation would weight them as if equal."
        )
    ph, pw = h // rows, w // cols
    if kernel_size > min(ph, pw):
        raise ValueError(
            f"kernel_size={kernel_size} exceeds the {ph}x{pw} patch implied by "
            f"control_grid {(rows, cols)}. Use fewer control points or a "
            "smaller kernel: a kernel wider than its evidence is extrapolation."
        )

    kernels = []
    for r in range(rows):
        for c in range(cols):
            y = observed[..., r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            m = known[..., r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            fm = centred_fft(m)
            fy = centred_fft(y)
            # Wiener/Tikhonov: argmin ||h*m - y||^2 + mu ||h||^2
            h_hat = centred_ifft((fm.conj() * fy) / (fm.abs() ** 2 + mu)).real
            k = _crop_centre(h_hat, kernel_size)
            kernels.append(k / k.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8))
    return torch.stack(kernels, dim=1).reshape(b, rows, cols, kernel_size, kernel_size)


def psf_identifiability(known: Tensor, *, threshold: float = 1e-3) -> Tensor:
    """Fraction of frequency support the marker actually excites, per sample.

    The kernel is only recoverable where the marker has energy. This is the
    honest companion to any measured PSF: an estimate from a marker that excites
    20% of the band is an interpolation over the other 80%, and reporting the
    kernel without reporting this would hide that.

    Args:
        known: ``[B, 1, H, W]`` the true fiducial.
        threshold: Magnitude floor, relative to the marker's peak.

    Returns:
        ``[B]`` in ``[0, 1]``.
    """
    mag = centred_fft(known).abs()
    peak = mag.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return (mag / peak > threshold).float().mean(dim=(-3, -2, -1))


def apply_psf(x: Tensor, kernels: Tensor) -> Tensor:
    """Blur ``x`` with a spatially varying kernel field, by partition of unity.

    Each control point's kernel is applied to the whole image and the results
    are blended with bilinear weights, so the effective kernel varies smoothly
    rather than jumping at patch boundaries. Blockwise application would make
    the operator discontinuous, and a network trained to invert a discontinuous
    operator learns the discontinuity.

    Args:
        x: ``[B, C, H, W]``.
        kernels: ``[B, R, C_grid, k, k]`` from :func:`estimate_psf`.

    Returns:
        ``[B, C, H, W]``. Differentiable in both.
    """
    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    if kernels.ndim != 5:
        raise ValueError(f"expected [B, R, C, k, k] kernels, got {tuple(kernels.shape)}")
    b, ch, hh, ww = x.shape
    _, rows, cols, k, _ = kernels.shape
    pad = k // 2
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")

    out = torch.zeros_like(x)
    for r in range(rows):
        for c in range(cols):
            # One-hot control field, bilinearly upsampled: the partition of unity
            weight_grid = torch.zeros(1, 1, rows, cols, device=x.device, dtype=x.dtype)
            weight_grid[0, 0, r, c] = 1.0
            w = (
                interpolate(weight_grid, size=(hh, ww), mode="bilinear", align_corners=True)
                if rows > 1 or cols > 1
                else torch.ones(1, 1, hh, ww, device=x.device, dtype=x.dtype)
            )
            kern = kernels[:, r, c].reshape(b, 1, 1, k, k).expand(b, ch, 1, k, k)
            blurred = torch.nn.functional.conv2d(
                xp.reshape(1, b * ch, hh + 2 * pad, ww + 2 * pad),
                kern.reshape(b * ch, 1, k, k),
                groups=b * ch,
            ).reshape(b, ch, hh, ww)
            out = out + w * blurred
    return out


def psf_fwhm_map(kernels: Tensor, *, voxel_mm: float = 1.0) -> Tensor:
    """Per-control-point FWHM, from the kernel's second moment.

    A scalar summary the network can be conditioned on, and the number an arm
    should quote: "the measured PSF is 2.8 mm here and 4.1 mm there" is a claim
    a reader can check, where a 9x9 kernel field is not.

    Args:
        kernels: ``[B, R, C, k, k]``, unit sum.
        voxel_mm: In-plane spacing, so the result is in millimetres.

    Returns:
        ``[B, R, C]`` FWHM in mm, isotropised as the geometric mean of the two
        axis widths.
    """
    k = kernels.shape[-1]
    ax = torch.arange(k, device=kernels.device, dtype=kernels.dtype) - (k - 1) / 2.0
    p = kernels.clamp(min=0.0)
    p = p / p.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    mean_y = (p.sum(dim=-1) * ax).sum(dim=-1)
    mean_x = (p.sum(dim=-2) * ax).sum(dim=-1)
    var_y = (p.sum(dim=-1) * ax**2).sum(dim=-1) - mean_y**2
    var_x = (p.sum(dim=-2) * ax**2).sum(dim=-1) - mean_x**2
    sigma = (var_y.clamp(min=0.0) * var_x.clamp(min=0.0)).clamp(min=1e-12) ** 0.25
    return 2.0 * (2.0 * torch.log(torch.tensor(2.0))) ** 0.5 * sigma * voxel_mm
