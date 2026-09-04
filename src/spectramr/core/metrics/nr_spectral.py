"""No-reference *spectral* MRI quality metrics (spec §3 ``nr_spectral``).

Four self-consistency probes on the spectrum of a reconstruction
:math:`\\hat{\\mathbf{x}}`. They need no reference image; some need the
acquisition mask / foreground region carried by a
:class:`~spectramr.core.metrics.context.MetricContext`.

==============  ==========================================================  ===========
key             quantity                                                    direction
==============  ==========================================================  ===========
``csr``         Conjugate-Symmetry Residual after smooth-phase demodulation  lower
``sher``        Structured block-Hankel effective rank (deviation-from-r*)   lower
``bpci``        Bicoherence phase-coupling index                             lower
``ssel``        Slepian subspace energy leakage                              lower
==============  ==========================================================  ===========

Design notes
------------
* All four return a Python ``float`` and *never* raise ``ValueError``
  (``core/metrics/computer.py`` re-raises ``ValueError`` and aborts the
  whole validation). On a degenerate / missing input they return
  ``float('nan')``.
* ``ssel`` depends on ``spectramr.infrastructure.physics.prolate`` (a sibling
  module created concurrently). That import is **lazy** inside ``__call__``;
  if it is not yet importable the metric returns ``nan`` rather than killing
  every registration in this module at walk-discovery time.
* Physics SSOT: all FFT round-trips go through ``fft2c``/``ifft2c`` and the
  conjugate-mirror convention is the one in
  ``infrastructure.physics.hermitian._cyclic_mirror`` (centered k-space,
  ``flip`` + ``roll`` by +1), never a raw ``torch.flip``.

References
----------
[15] K. H. Jin, D. Lee, J. C. Ye, "A general framework for compressed
     sensing and parallel MRI using annihilating filter based low-rank
     Hankel matrix," *IEEE TCI*, 2(4), 2016.  (SHER block-Hankel.)
[23] D. Slepian, H. O. Pollak, "Prolate spheroidal wave functions, …—I,"
     *Bell Syst. Tech. J.*, 40(1), 1961.  (SSEL Slepian subspace.)
Nikias & Petropulu, *Higher-Order Spectra Analysis*, 1993. (BPCI bicoherence.)
"""

from __future__ import annotations

import math

import torch

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.hallucination_metrics import _to_magnitude
from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.fft_ops import _to_complex, fft2c
from spectramr.infrastructure.physics.hermitian import _cyclic_mirror


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _as_complex_image(prediction: torch.Tensor) -> torch.Tensor:
    """Coerce ``prediction`` to a complex image ``[B, 1, H, W]``.

    Accepts ``[B, C, H, W]`` (C even ⇒ interleaved real/imag, taken as the
    first complex channel), ``[B, 1, H, W]`` real, complex tensors, or
    unbatched ``[C, H, W]`` / ``[H, W]``. Returns ``complex64``.
    """
    x = prediction
    if x.dim() == 2:  # [H, W]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:  # [C, H, W]
        x = x.unsqueeze(0)
    # x is now [B, C, H, W] (or [B, C, H, W] complex)
    z = _to_complex(x)  # complex [B, C', H, W]
    if z.dim() == 4 and z.shape[1] > 1:
        z = z[:, :1]  # first complex channel
    elif z.dim() == 3:
        z = z.unsqueeze(1)
    return z.to(torch.complex64)


def _phase_lowpass_unitvector(
    phase: torch.Tensor, *, sigma: float, truncate: float = 3.0
) -> torch.Tensor:
    """Wrap-safe low-pass of a phase map via the unit-vector (cos/sin) trick.

    Low-passing the raw angle is wrong across the ±π branch cut; instead
    smooth ``(cos φ, sin φ)`` and recover ``atan2``. Reuses the depthwise
    separable Gaussian from ``high_frequency_residual_loss`` so the kernel
    matches the rest of the codebase.

    Args:
        phase: real ``[B, 1, H, W]`` angle map.
        sigma: Gaussian sigma in pixels.
        truncate: kernel half-width in units of sigma.

    Returns:
        Smoothed phase ``[B, 1, H, W]``.
    """
    from spectramr.models.losses.high_frequency_residual_loss import _gaussian_lowpass

    cos_s = _gaussian_lowpass(torch.cos(phase), sigma=sigma, truncate=truncate)
    sin_s = _gaussian_lowpass(torch.sin(phase), sigma=sigma, truncate=truncate)
    return torch.atan2(sin_s, cos_s)


# ---------------------------------------------------------------------------
# csr — Conjugate-Symmetry Residual after Smooth-Phase Demodulation
# ---------------------------------------------------------------------------
@register_metric("csr", aliases=["conjugate_symmetry_residual"], requires_reference=False)
class ConjugateSymmetryResidual:
    r"""Conjugate-Symmetry Residual after smooth-phase demodulation (↓).

    A real object has Hermitian-symmetric k-space, :math:`Y(\mathbf{k}) =
    Y^{*}(-\mathbf{k})`. A *smooth* image phase only shifts/modulates that
    symmetry and can be removed; the *residual* anti-symmetry that survives
    after demodulating a low-pass phase estimate flags non-physical
    (network-fabricated) phase structure.

    .. math::

        \hat\varphi = \mathrm{LP}_\sigma(\angle\hat{\mathbf{x}}),\quad
        \tilde{Y} = \mathbf{F}\!\left(e^{-i\hat\varphi}\hat{\mathbf{x}}\right),\quad
        \mathrm{CSR} = \frac{\lVert\tilde{Y}(\mathbf{k}) - \tilde{Y}^{*}(-\mathbf{k})\rVert_2}
                            {\lVert\tilde{Y}(\mathbf{k}) + \tilde{Y}^{*}(-\mathbf{k})\rVert_2 + \varepsilon}

    Anchor: ``≈ 0`` for a real / smooth-phase phantom; rises with
    non-smooth phase corruption.

    Args:
        sigma: Gaussian sigma (px) for the low-pass phase estimate.
        eps: denominator stabiliser.
    """

    def __init__(self, sigma: float = 6.0, eps: float = 1e-8) -> None:
        self.sigma = float(sigma)
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "csr"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        del target, context  # CSR needs only the complex prediction
        try:
            x = _as_complex_image(prediction)  # [B, 1, H, W] complex
            phase = torch.angle(x)
            phi_hat = _phase_lowpass_unitvector(phase, sigma=self.sigma)
            x_demod = x * torch.exp(-1j * phi_hat.to(x.dtype))
            y = fft2c(x_demod)  # complex [B, 1, H, W]
            y_mirror = _cyclic_mirror(y).conj()
            anti = (y - y_mirror).abs()
            sym = (y + y_mirror).abs()
            # ||·||_2 per image, then mean over batch.
            num = torch.sqrt((anti**2).flatten(1).sum(dim=1))
            den = torch.sqrt((sym**2).flatten(1).sum(dim=1)) + self.eps
            return float((num / den).mean().item())
        except Exception:  # NR metrics must never crash validation
            return float("nan")


# ---------------------------------------------------------------------------
# sher — Structured Block-Hankel Effective Rank
# ---------------------------------------------------------------------------
@register_metric(
    "sher",
    aliases=["structured_hankel_effective_rank", "block_hankel_effective_rank"],
    requires_reference=False,
)
class StructuredHankelEffectiveRank:
    r"""Structured block-Hankel effective rank, deviation from a target (↓).

    The annihilating-filter theory of [15] says a finite-support /
    smooth-phase image has a *low-rank* block-Hankel matrix built from its
    k-space patches. The (entropy) effective rank

    .. math::

        p_i = \frac{\sigma_i}{\sum_j \sigma_j},\qquad
        r_{\mathrm{eff}} = \exp\!\Big(-\sum_i p_i \ln p_i\Big)

    estimates that rank without a threshold. Noise inflates it (the spectrum
    fattens) and over-smoothing collapses it. We score the **absolute
    deviation** :math:`|r_{\mathrm{eff}} - r^{\star}|` from a target rank
    ``r_star`` so that *both* directions of departure are penalised and
    ``higher_is_better`` is unambiguous (``False``).

    When ``r_star`` is ``None`` the raw effective rank is returned (and the
    direction convention then reads "the smaller the model's rank inflation,
    the better").

    Args:
        kernel_size: side of the square k-space neighbourhood (Hankel block).
        r_star: target effective rank :math:`r^{\star}`; ``None`` ⇒ raw rank.
        max_rows: cap on Hankel rows (memory guard), mirrors HankelLowRankLoss.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        r_star: float | None = None,
        max_rows: int = 4096,
    ) -> None:
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.r_star = None if r_star is None else float(r_star)
        self.max_rows = int(max_rows)

    @property
    def name(self) -> str:
        return "sher"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _effective_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy effective rank of the block-Hankel of ``F x`` (per batch)."""
        k = fft2c(x)  # [B, 1, H, W] complex
        b, _c, h, w = k.shape
        ks = self.kernel_size
        if h < ks or w < ks:
            return torch.full((b,), float("nan"), device=k.device)
        # Sliding ks*ks patches → Hankel rows (same construction as
        # HankelLowRankLoss.forward).
        patches = k.unfold(2, ks, 1).unfold(3, ks, 1)  # [B, C, n_h, n_w, ks, ks]
        b2, c2, n_h, n_w, _, _ = patches.shape
        rows = patches.contiguous().view(b2, c2, n_h * n_w, ks * ks)
        if rows.shape[2] > self.max_rows:
            rows = rows[:, :, : self.max_rows, :]
        rows = rows[:, 0]  # single complex channel → [B, n, ks*ks]
        sv = torch.linalg.svdvals(rows)  # [B, min(n, ks*ks)]
        total = sv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p = sv / total
        # -sum p ln p  with the 0 ln 0 = 0 convention.
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
        return torch.exp(ent)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        del target, context
        try:
            x = _as_complex_image(prediction)
            r_eff = self._effective_rank(x)  # [B]
            if self.r_star is not None:
                r_eff = (r_eff - self.r_star).abs()
            return float(r_eff.mean().item())
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# bpci — Bicoherence Phase-Coupling Index
# ---------------------------------------------------------------------------
@register_metric("bpci", aliases=["bicoherence_phase_coupling"], requires_reference=False)
class BicoherencePhaseCouplingIndex:
    r"""Bicoherence phase-coupling index over a coarse low-frequency grid (↓).

    Coherent artefacts (ghosts, structured aliasing, hallucinated texture)
    introduce *quadratic phase coupling* between spectral components — a
    third-order (bispectral) signature absent in Gaussian noise (whose odd
    cumulants vanish). The normalised bicoherence

    .. math::

        b^2(\mathbf{k}_1,\mathbf{k}_2) =
          \frac{\big|\mathbb{E}[Y(\mathbf{k}_1)Y(\mathbf{k}_2)Y^{*}(\mathbf{k}_1{+}\mathbf{k}_2)]\big|^2}
               {\mathbb{E}|Y(\mathbf{k}_1)Y(\mathbf{k}_2)|^2\;\mathbb{E}|Y(\mathbf{k}_1{+}\mathbf{k}_2)|^2}

    is bounded in ``[0, 1]``, and the expectation must run over an **ensemble
    of realisations**. That is the whole content of the estimator: with a
    single realisation the numerator is
    :math:`|Y_1 Y_2 Y_3^{*}|^2 = |Y_1|^2|Y_2|^2|Y_3|^2`, which is *exactly*
    the denominator, so :math:`b^2 \equiv 1` for every ``(k1, k2)`` and every
    image — Cauchy-Schwarz at equality. Averaging those over the ``(k1, k2)``
    grid still gives 1, so the metric returned a constant ``1.0`` on all
    inputs and could not respond to any degradation (verified on white noise,
    a structured triple, and a near-zero triple).

    The ensemble here is therefore built by **Welch-style segment averaging**:
    the image is tiled into ``n_segments**2`` patches, each is transformed
    independently, and the expectations run over patches. Quadratic phase
    coupling is precisely the property that survives that average — a coherent
    ghost holds the same phase relation in every patch so the triple products
    add constructively (:math:`b^2 \to 1`), whereas white noise has an
    independent random phase per patch so they cancel to the estimator's noise
    floor :math:`\approx 1/S` for ``S`` segments. This restores the documented
    anchor, which the previous form made unreachable.

    Anchor: ``≈ 1/S`` on a Gaussian-noise phantom (``S`` segments); rises
    toward 1 with coherent ghost/aliasing.

    Args:
        band: half-width (bins) of the low-frequency square around DC that
            ``k1, k2`` range over (keeps cost O(band^4)).
        n_segments: tiles per axis; the ensemble is ``n_segments**2`` patches.
            Must be >= 2 — at 1 there is no ensemble and ``b^2`` degenerates
            to the constant 1 described above.
        eps: denominator stabiliser.
    """

    def __init__(self, band: int = 4, n_segments: int = 4, eps: float = 1e-12) -> None:
        if band < 1:
            raise ValueError(f"band must be >= 1, got {band}")
        if n_segments < 2:
            raise ValueError(
                f"n_segments must be >= 2 (bicoherence needs an ensemble; at 1 "
                f"it is identically 1 for every input), got {n_segments}"
            )
        self.n_segments = int(n_segments)
        self.band = int(band)
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "bpci"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        del target, context
        try:
            x = _as_complex_image(prediction)  # [B, 1, H, W] complex
            b, _c, h, w = x.shape
            band = self.band

            if b >= 2:
                # The batch IS the ensemble — the estimator's intended form, and
                # what the analytic anchors below are calibrated against. Left
                # exactly as it was.
                y = fft2c(x)[:, 0]  # [B, H, W]
            else:
                # B == 1: every expectation would be over a single sample, which
                # makes the numerator identically equal to the denominator and
                # b^2 == 1 for EVERY (k1, k2) and every input. sim2rank scores
                # one image at a time, so the metric returned a constant 1.0 on
                # all 30 axes and was recorded "dead". Tile into patches to
                # recover an ensemble.
                n = self.n_segments
                # Each segment must hold the (k1, k2, k1+k2) support: indices
                # reach 2*band from DC, so a patch needs >= 4*band+1 bins/axis.
                while n > 1 and (h // n < 4 * band + 1 or w // n < 4 * band + 1):
                    n -= 1
                if n < 2:
                    # Too small to segment and no batch: bicoherence is not
                    # estimable. NaN is the honest answer — 1.0 would be a
                    # fabricated "perfectly coupled" reading.
                    return float("nan")
                ph, pw = h // n, w // n
                segs = (
                    x[:, :, : ph * n, : pw * n]
                    .reshape(b, 1, n, ph, n, pw)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(b * n * n, 1, ph, pw)
                )
                y = fft2c(segs)[:, 0]  # [S, ph, pw], S = n*n
            _s, hh, ww = y.shape
            cy, cx = hh // 2, ww // 2
            if hh < 4 * band + 1 or ww < 4 * band + 1:
                return float("nan")

            # --- flat (di, dj) offset grid, shared by k1 and k2 ------------
            offs = torch.arange(-band, band + 1, device=y.device)
            di, dj = torch.meshgrid(offs, offs, indexing="ij")
            di, dj = di.reshape(-1), dj.reshape(-1)  # [N], N = (2*band+1)^2

            y1 = y[:, cy + di, cx + dj]  # [S, N]  Y(k1)
            y2 = y1  # same grid for k2
            # k3 = k1 + k2 -> outer sum of the offset grids.
            i3 = cy + di[:, None] + di[None, :]  # [N, N]
            j3 = cx + dj[:, None] + dj[None, :]
            y3 = y[:, i3, j3]  # [S, N, N]  Y(k1+k2)

            prod12 = y1[:, :, None] * y2[:, None, :]  # [S, N, N]
            # Expectations over the SEGMENT ensemble (dim 0) — the whole point.
            e_triple = (prod12 * y3.conj()).mean(dim=0)  # [N, N]
            e_p12 = prod12.abs().pow(2).mean(dim=0)  # [N, N]
            e_p3 = y3.abs().pow(2).mean(dim=0)  # [N, N]

            b2 = e_triple.abs().pow(2) / (e_p12 * e_p3 + self.eps)
            return float(b2.mean().item())
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# ssel — Slepian Subspace Energy Leakage
# ---------------------------------------------------------------------------
@register_metric(
    "ssel",
    aliases=["slepian_subspace_energy_leakage"],
    requires_reference=False,
    needs=("foreground_mask", "mask"),
)
class SlepianSubspaceEnergyLeakage:
    r"""Slepian subspace energy leakage (↓).

    The leading prolate-spheroidal (Slepian) functions of a
    space-limit→band-limit→space-limit operator span the subspace that an
    optimally concentrated, band-limited, finite-support signal can occupy.
    Energy of :math:`\hat{\mathbf{x}}` that *leaks* outside that subspace is
    ringing / aliasing spread:

    .. math::

        \mathrm{SSEL} = 1 -
          \frac{\sum_{i:\lambda_i > 1-\epsilon}\lvert\langle\hat{\mathbf{x}},\psi_i\rangle\rvert^2}
               {\lVert\hat{\mathbf{x}}\rVert_2^2}

    We apply the 1-D DPSS separably along image rows (restricted to the
    foreground extent), via ``slepian_concentration`` from
    ``infrastructure.physics.prolate``. ``→ 0`` at full bandwidth/support;
    rises with ringing/aliasing.

    The metric reads ``foreground_mask`` (for the support extent) and
    ``mask`` (for the bandwidth ``W = n_acquired / n``) from the context,
    falling back to a magnitude-threshold foreground and a full-band default
    when they are absent — and to ``nan`` if ``prolate`` is not importable.

    Args:
        n_tapers: number of leading Slepian tapers (subspace dimension).
        bandwidth: time-bandwidth product (``NW``) when ``mask`` is absent.
        eps: ``λ`` selection slack (kept for spec fidelity / docs).
    """

    def __init__(
        self,
        n_tapers: int = 8,
        bandwidth: float = 4.0,
        eps: float = 1e-3,
    ) -> None:
        if n_tapers < 1:
            raise ValueError(f"n_tapers must be >= 1, got {n_tapers}")
        self.n_tapers = int(n_tapers)
        self.bandwidth = float(bandwidth)
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "ssel"

    @property
    def higher_is_better(self) -> bool:
        return False

    @staticmethod
    def _foreground_rowsupport(mag: torch.Tensor, fg: torch.Tensor | None) -> torch.Tensor:
        """Boolean foreground ``[B, H, W]`` from context mask or |x| threshold."""
        b = mag.shape[0]
        if fg is not None:
            f = fg
            while f.dim() > 3:
                f = f[:, 0] if f.shape[1] == 1 else f.squeeze(1)
                if f.dim() == 4:
                    f = f[:, 0]
            if f.dim() == 2:
                f = f.unsqueeze(0).expand(b, -1, -1)
            return f.to(torch.bool)
        # Otsu-lite: threshold at 5% of per-image max.
        m = mag[:, 0]
        thresh = 0.05 * m.flatten(1).max(dim=1).values.view(b, 1, 1)
        return m > thresh

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        del target
        ctx = resolve_context(context, kwargs)
        try:
            from spectramr.infrastructure.physics.prolate import slepian_concentration
        except Exception:  # prolate created concurrently
            return float("nan")
        try:
            x = _as_complex_image(prediction)  # [B, 1, H, W] complex
            mag = _to_magnitude(x)  # [B, 1, H, W] real
            b, _, h, w = mag.shape
            fg = self._foreground_rowsupport(mag, ctx.foreground_mask)

            # Bandwidth from the acquisition mask if present.
            nw = self.bandwidth
            if ctx.mask is not None:
                msk = ctx.mask
                while msk.dim() > 2:
                    msk = msk[0]
                frac = float(msk.float().mean().clamp(1e-3, 1.0).item())
                # NW ~ half-bandwidth * n / 2 ; keep small & valid.
                nw = max(2.0, min(self.bandwidth, frac * (max(h, w) / 4.0)))

            n_tapers = self.n_tapers
            # Per-image, average the row-wise concentration over foreground rows.
            leak_per_image = []
            for bi in range(b):
                rows_keep = fg[bi].any(dim=1).nonzero(as_tuple=True)[0]
                if rows_keep.numel() == 0:
                    leak_per_image.append(float("nan"))
                    continue
                concs = []
                # Subsample rows to keep the eigensolve count bounded.
                step = max(1, rows_keep.numel() // 32)
                for r in rows_keep[::step].tolist():
                    sig = mag[bi, 0, r]  # [W] real
                    energy = float((sig**2).sum().item())
                    if energy < 1e-12:
                        continue
                    nt = min(n_tapers, max(1, sig.numel() - 1))
                    # ``nw`` is the time-half-bandwidth PRODUCT (NW); prolate's
                    # ``slepian_concentration`` wants the normalised half-bandwidth
                    # W = NW / n in the open interval (0, 0.5). Passing NW directly
                    # raised ValueError on every call -> the metric was always NaN.
                    w_norm = min(0.49, max(nw, 1e-3) / float(sig.numel()))
                    conc = slepian_concentration(sig, nt, w_norm)
                    concs.append(float(conc))
                if not concs:
                    leak_per_image.append(float("nan"))
                    continue
                # concentration ∈ [0,1] is the *retained* fraction; leakage = 1 - mean.
                mean_conc = sum(concs) / len(concs)
                leak_per_image.append(1.0 - mean_conc)

            vals = [v for v in leak_per_image if not math.isnan(v)]
            if not vals:
                return float("nan")
            return float(sum(vals) / len(vals))
        except Exception:
            return float("nan")


__all__ = [
    "BicoherencePhaseCouplingIndex",
    "ConjugateSymmetryResidual",
    "SlepianSubspaceEnergyLeakage",
    "StructuredHankelEffectiveRank",
]
