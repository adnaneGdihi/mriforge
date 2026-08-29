"""Channelized Hotelling Observer (CHO) for model-observer image-quality validation.

Implements the linear *channelized* Hotelling observer used as a principled,
task-based substitute for human Likert scoring in the no-reference metric
battery (spec §8.3). The observer projects each image realisation onto a fixed,
small bank of spatial-frequency channels :math:`\\mathbf{U}` (a Gabor / DDOG
bank), estimates the channel-domain second-order statistics, and reports the
signal-detection figures of merit:

.. math::

    \\mathbf{g}_1 = \\mathbf{U}^{\\top}\\mathbf{f}_{\\text{present}},\\qquad
    \\mathbf{g}_0 = \\mathbf{U}^{\\top}\\mathbf{f}_{\\text{absent}}

    \\mathbf{K}_{\\mathbf{g}} = \\tfrac12\\big(\\operatorname{cov}(\\mathbf{g}_1) + \\operatorname{cov}(\\mathbf{g}_0)\\big),\\qquad
    \\Delta\\bar{\\mathbf{g}} = \\bar{\\mathbf{g}}_1 - \\bar{\\mathbf{g}}_0

    d'^{2} = \\Delta\\bar{\\mathbf{g}}^{\\top}\\,\\mathbf{K}_{\\mathbf{g}}^{-1}\\,\\Delta\\bar{\\mathbf{g}},\\qquad
    \\mathbf{w}_{\\mathrm{CHO}} = \\mathbf{K}_{\\mathbf{g}}^{-1}\\Delta\\bar{\\mathbf{g}},\\qquad
    \\mathrm{AUC} = \\Phi\\!\\big(d'/\\sqrt{2}\\big)

where :math:`\\Phi` is the standard-normal CDF.

Channels are *fixed* spatial templates (data-independent), so the observer is
deterministic given fixed inputs; any stochasticity (e.g. sampling internal
observer noise) is driven by a caller-supplied :class:`torch.Generator`.

Conventions
-----------
- ``channels`` (the bank :math:`\\mathbf{U}`) is ``[n_pixels, n_channels]``.
- ``present`` / ``absent`` ensembles are ``[n_pixels, n_samples]`` (each column
  is one flattened image realisation under the signal-present / signal-absent
  hypothesis).
- A small ridge ``ridge`` is added to :math:`\\mathbf{K}_{\\mathbf{g}}` before the
  solve so the inverse is well-conditioned with few samples.

Note on ``w_cho``
-----------------
The spec writes :math:`\\mathbf{w}_{\\mathrm{CHO}}=(\\mathbf{U}^{\\top}\\mathbf{K}_{\\mathbf{g}}\\mathbf{U})^{-1}\\mathbf{U}^{\\top}\\Delta\\bar{\\mathbf{g}}`,
mixing pixel-space (:math:`\\mathbf{U}^{\\top}\\Delta\\bar{\\mathbf{g}}`) and
channel-space (:math:`\\mathbf{K}_{\\mathbf{g}}`) quantities. The dimensionally-consistent
and standard (Barrett & Myers) channel-space template that actually realises
:math:`d'^2 = \\Delta\\bar{\\mathbf{g}}^{\\top}\\mathbf{w}_{\\mathrm{CHO}}` is
:math:`\\mathbf{w}_{\\mathrm{CHO}} = \\mathbf{K}_{\\mathbf{g}}^{-1}\\Delta\\bar{\\mathbf{g}}`,
which is what :pyattr:`ChannelizedHotellingObserver.w_cho` returns.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ChannelizedHotellingObserver",
    "ddog_channel_bank",
    "gabor_channel_bank",
]


def _normal_cdf(x: torch.Tensor | float) -> float:
    """Standard-normal CDF :math:`\\Phi(x)` via the error function (scalar)."""
    if isinstance(x, torch.Tensor):
        x = float(x.detach().reshape(-1)[0].item())
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _radial_grid(
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (yy, xx, radius) centered coordinate grids for ``image_shape``."""
    h, w = image_shape
    ys = torch.arange(h, dtype=torch.float64) - (h - 1) / 2.0
    xs = torch.arange(w, dtype=torch.float64) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius = torch.sqrt(yy * yy + xx * xx)
    return yy, xx, radius


def _orthonormalise(columns: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Centre + L2-normalise each channel column; drop degenerate (near-zero) columns.

    Centering removes the DC component so channels respond to *structure*, not
    mean level. Columns whose post-centering norm is below ``eps`` are dropped.
    """
    cols = columns - columns.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(cols, dim=0)
    keep = norms > eps
    cols = cols[:, keep]
    norms = norms[keep]
    if cols.shape[1] == 0:
        raise ValueError("All channel columns are degenerate after centering.")
    return cols / norms.unsqueeze(0)


def gabor_channel_bank(
    image_shape: tuple[int, int],
    n_channels: int = 10,
    *,
    n_orientations: int = 4,
    sigma_frac: float = 0.15,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed (data-independent) cosine-Gabor channel bank, flattened.

    Builds ``n_channels`` even-symmetric (cosine) Gabor channels spanning a
    log-spaced set of radial frequencies and a set of orientations, each
    multiplied by a Gaussian window. The bank is centred + L2-normalised
    column-wise.

    Args:
        image_shape: ``(H, W)`` of the images the observer will score.
        n_channels: number of channels to return (truncated/cycled from the
            ``n_radial x n_orientations`` grid).
        n_orientations: orientations per radial band.
        sigma_frac: Gaussian-window std as a fraction of ``min(H, W)``.
        dtype: dtype of the returned bank.

    Returns:
        ``[H*W, n_channels]`` real tensor (the bank :math:`\\mathbf{U}`).
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    h, w = image_shape
    yy, xx, _ = _radial_grid(image_shape)
    sigma = sigma_frac * float(min(h, w))
    window = torch.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))

    n_radial = max(1, math.ceil(n_channels / max(1, n_orientations)))
    # Frequencies in cycles/pixel, log-spaced (avoid DC).
    freqs = torch.logspace(
        math.log10(1.0 / min(h, w)), math.log10(0.45), n_radial, dtype=torch.float64
    )
    orientations = torch.linspace(0.0, math.pi, n_orientations + 1, dtype=torch.float64)[
        :n_orientations
    ]

    cols: list[torch.Tensor] = []
    for f in freqs:
        for theta in orientations:
            kx = 2.0 * math.pi * float(f) * math.cos(float(theta))
            ky = 2.0 * math.pi * float(f) * math.sin(float(theta))
            carrier = torch.cos(kx * xx + ky * yy)
            cols.append((window * carrier).reshape(-1))
            if len(cols) >= n_channels:
                break
        if len(cols) >= n_channels:
            break

    bank = torch.stack(cols, dim=1)
    bank = _orthonormalise(bank)
    return bank.to(dtype)


def ddog_channel_bank(
    image_shape: tuple[int, int],
    n_channels: int = 10,
    *,
    sigma0_frac: float = 0.02,
    alpha: float = 1.4,
    q: float = 1.67,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed difference-of-difference-of-Gaussians (DDoG) radial channel bank.

    Each channel is a rotationally-symmetric band-pass profile built as a
    difference of two difference-of-Gaussians at successive scales, the classic
    Abbey/Barrett DDoG observer channels. The bank is centred + L2-normalised
    column-wise and flattened.

    Args:
        image_shape: ``(H, W)``.
        n_channels: number of radial channels.
        sigma0_frac: smallest Gaussian std as a fraction of ``min(H, W)``.
        alpha: ratio between the two Gaussians inside each DoG.
        q: geometric scale ratio between successive channels.
        dtype: dtype of the returned bank.

    Returns:
        ``[H*W, n_channels]`` real tensor (the bank :math:`\\mathbf{U}`).
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    h, w = image_shape
    _, _, radius = _radial_grid(image_shape)
    r2 = radius * radius
    sigma0 = sigma0_frac * float(min(h, w))
    if sigma0 <= 0:
        raise ValueError(f"sigma0 must be > 0, got {sigma0}")

    def _gauss(sigma: float) -> torch.Tensor:
        return torch.exp(-r2 / (2.0 * sigma * sigma))

    cols: list[torch.Tensor] = []
    for j in range(n_channels):
        s = sigma0 * (q**j)
        # difference-of-difference-of-Gaussians: ((G_s - G_{a*s}) - (G_{a*s} - G_{a*a*s}))
        dog_inner = _gauss(s) - _gauss(alpha * s)
        dog_outer = _gauss(alpha * s) - _gauss(alpha * alpha * s)
        cols.append((dog_inner - dog_outer).reshape(-1))

    bank = torch.stack(cols, dim=1)
    bank = _orthonormalise(bank)
    return bank.to(dtype)


class ChannelizedHotellingObserver:
    """Linear channelized Hotelling observer.

    Args:
        channels: fixed channel bank :math:`\\mathbf{U}` of shape
            ``[n_pixels, n_channels]`` (e.g. from :func:`gabor_channel_bank` or
            :func:`ddog_channel_bank`).
        ridge: ridge added to the channel covariance diagonal before inversion.
    """

    def __init__(self, channels: torch.Tensor, *, ridge: float = 1e-6) -> None:
        if channels.ndim != 2:
            raise ValueError(
                f"channels must be [n_pixels, n_channels], got shape {tuple(channels.shape)}"
            )
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        self.U = channels
        self.ridge = float(ridge)
        self._w_cho: torch.Tensor | None = None

    # ----------------------------------------------------------------- internals
    def _project(self, ensemble: torch.Tensor) -> torch.Tensor:
        """U^T @ ensemble -> channel responses ``[n_channels, n_samples]``."""
        if ensemble.ndim != 2:
            raise ValueError(
                f"ensemble must be [n_pixels, n_samples], got shape {tuple(ensemble.shape)}"
            )
        if ensemble.shape[0] != self.U.shape[0]:
            raise ValueError(
                f"ensemble n_pixels={ensemble.shape[0]} != channels n_pixels={self.U.shape[0]}"
            )
        u = self.U.to(ensemble.dtype)
        return u.transpose(0, 1) @ ensemble

    def _channel_stats(
        self, present: torch.Tensor, absent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Δḡ, K_g + ridge·I) in channel space, in float64 for stability.

        ``K_g`` is the **average within-class** channel covariance (each class
        mean-centred before pooling), i.e. the noise covariance under the
        equal-covariance model. This is the statistically-correct CHO estimator
        (Barrett & Myers): a naive ``cov(cat([g1, g0]))`` over the concatenated
        ensembles confounds the between-class mean separation into ``K_g`` and
        caps ``d'^2`` at a constant (~4 for equal class sizes) regardless of the
        true noise level. We therefore deviate from the spec's literal one-liner
        here so that ``d'^2`` scales as ``1/noise^2`` as it must.
        """
        g1 = self._project(present).to(torch.float64)
        g0 = self._project(absent).to(torch.float64)
        d = g1.mean(dim=1) - g0.mean(dim=1)
        if g1.shape[1] + g0.shape[1] < 2:
            raise ValueError("Need >= 2 total samples to estimate a channel covariance.")
        n_ch = g1.shape[0]

        # Pooled within-class covariance (mean-centre each class separately).
        def _within_cov(g: torch.Tensor) -> tuple[torch.Tensor, int]:
            n = g.shape[1]
            if n < 2:
                return torch.zeros(n_ch, n_ch, dtype=g.dtype, device=g.device), 0
            gc = g - g.mean(dim=1, keepdim=True)
            return gc @ gc.transpose(0, 1), n - 1

        s1, df1 = _within_cov(g1)
        s0, df0 = _within_cov(g0)
        df = df1 + df0
        if df <= 0:
            raise ValueError("Need >= 2 samples in at least one class for a covariance.")
        k = (s1 + s0) / df
        if k.ndim == 0:  # single channel safety
            k = k.reshape(1, 1)
        # ``device=`` is load-bearing: without it the ridge lands on CPU while ``k`` is
        # on the accelerator, ``torch.linalg.solve`` raises, and ``detectability``'s
        # ``except RuntimeError -> nan`` turns a device bug into a silent NaN. That is
        # why ``task_based_detectability`` was ``never_computed`` on all 600 cells of
        # the 2026-07-27 GPU run while computing fine in every CPU test.
        k = k + self.ridge * torch.eye(n_ch, dtype=k.dtype, device=k.device)
        return d, k

    # ------------------------------------------------------------------- public
    def detectability(self, present: torch.Tensor, absent: torch.Tensor) -> float:
        r"""Squared detectability :math:`d'^2 = \Delta\bar{\mathbf{g}}^\top K_g^{-1} \Delta\bar{\mathbf{g}}`.

        Also caches the CHO template :pyattr:`w_cho` (``K_g^{-1} Δḡ``).
        Returns ``float('nan')`` if the inputs are degenerate (rather than
        raising) so it is safe inside a metric sweep.
        """
        try:
            d, k = self._channel_stats(present, absent)
            w = torch.linalg.solve(k, d)
            self._w_cho = w.to(self.U.dtype)
            d_prime_sq = float(torch.dot(d, w).item())
        except (RuntimeError, ValueError):
            self._w_cho = None
            return float("nan")
        if not math.isfinite(d_prime_sq):
            return float("nan")
        # d'^2 is a Mahalanobis quadratic form -> non-negative up to numerical noise.
        return max(0.0, d_prime_sq)

    def auc(self, present: torch.Tensor, absent: torch.Tensor) -> float:
        r"""Area under the ROC: :math:`\mathrm{AUC} = \Phi(d'/\sqrt 2)`.

        For the equal-covariance Gaussian model the CHO ROC depends only on
        :math:`d'`; ``AUC = 0.5`` when there is no detectable signal and
        :math:`\to 1` for a strong signal. Returns ``float('nan')`` on
        degenerate input.
        """
        d_prime_sq = self.detectability(present, absent)
        if not math.isfinite(d_prime_sq):
            return float("nan")
        d_prime = math.sqrt(max(0.0, d_prime_sq))
        return _normal_cdf(d_prime / math.sqrt(2.0))

    @property
    def w_cho(self) -> torch.Tensor:
        """Last-computed CHO template weights :math:`\\mathbf{w}_{\\mathrm{CHO}}=K_g^{-1}\\Delta\\bar{\\mathbf{g}}`.

        Raises:
            RuntimeError: if no successful :meth:`detectability` / :meth:`auc`
                call has populated it yet.
        """
        if self._w_cho is None:
            raise RuntimeError(
                "w_cho is unavailable: call detectability()/auc() with valid inputs first."
            )
        return self._w_cho
