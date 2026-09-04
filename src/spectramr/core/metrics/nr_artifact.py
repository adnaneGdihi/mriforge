"""No-reference *artifact* quality metrics for MRI reconstructions (spec §3).

This module scores a reconstruction :math:`\\hat{\\mathbf{x}}` for the
acquisition-specific artifacts that a full-reference PSNR/SSIM cannot see
because they have *no clean target*: phase-encode ghosts, residual
parallel-imaging aliasing replicas, Gibbs ringing at sharp edges, phase
corruption, and resolution anisotropy. Every metric is no-reference
(``requires_reference=False``) and reduces over the batch to a single
``float``.

All metrics follow the house contract: a plain class decorated with
:func:`~spectramr.core.metrics.registry.register_metric`, a self-declared
``higher_is_better`` property, and a ``__call__`` that reads its
measurement/geometry context through
:func:`~spectramr.core.metrics.context.resolve_context`. A missing or
degenerate input returns ``float('nan')`` (never raises ``ValueError`` —
the computer re-raises that and aborts the whole validation).

Metrics
-------
- ``pegc`` — Phase-Encode Ghost Coherence (↓)
- ``are``  — Aliasing-Replica Energy (↓)
- ``gri``  — Gibbs-Ringing Index (→, target ≈ 0.09; lower-ringing ⇒ better)
- ``cpge`` — Complex Phase-Gradient Entropy (↓, more negative ⇒ better)
- ``rar``  — Resolution-Anisotropy Ratio (→, target 1.0; deviation ⇒ worse)
"""

from __future__ import annotations

import math

import torch

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.hallucination_metrics import _to_magnitude
from spectramr.core.metrics.registry import register_metric
from spectramr.core.quantile import robust_quantile

# ---------------------------------------------------------------------------
# Shared layout / mask helpers (kept local — dependency-free on infrastructure)
# ---------------------------------------------------------------------------


def _as_bchw_magnitude(prediction: torch.Tensor) -> torch.Tensor:
    """Coerce any accepted input layout to a real magnitude ``[B, 1, H, W]``.

    Accepts ``[B, C, H, W]`` (C even ⇒ interleaved real/imag), ``[B, 1, H, W]``,
    ``[C, H, W]``, ``[H, W]``, or complex. Uses the de-facto magnitude helper
    :func:`spectramr.core.metrics.hallucination_metrics._to_magnitude` for the
    real/imag collapse, then reduces any remaining coil channels by RSS.
    """
    x = prediction
    if x.dim() == 2:  # [H, W]
        x = x[None, None]
    elif x.dim() == 3:  # [C, H, W] or complex [C, H, W]
        x = x[None]
    mag = _to_magnitude(x)  # complex/2-ch -> [B, K, H, W] real
    if mag.shape[1] > 1:  # residual coil channels -> RSS over channel
        mag = torch.sqrt((mag.float() ** 2).sum(dim=1, keepdim=True) + 1e-12)
    # These metrics reduce to a Python ``float`` (non-differentiable), so hold no
    # autograd graph: detaching at the boundary matches the tissue-segmentation
    # segmenter and silences the "requires_grad -> scalar" UserWarning the SFA
    # autograd probe would otherwise trip on every call.
    return mag.float().detach()


def _as_bhw_complex(prediction: torch.Tensor) -> torch.Tensor | None:
    """Return a *complex* ``[B, H, W]`` view, or ``None`` if input is real-only.

    Required by ``cpge`` which needs the phase. Interleaved real/imag (even C)
    and native complex are honoured; a single real channel has no phase and
    returns ``None`` (the metric then yields ``nan``).
    """
    x = prediction
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    if x.is_complex():
        c = x
        if c.dim() == 4 and c.shape[1] > 1:
            # coil-combine complex by the first channel proxy (phase is per-coil
            # arbitrary; take channel 0 to keep a well-defined phase map).
            c = c[:, 0]
        elif c.dim() == 4:
            c = c[:, 0]
        return c.detach()
    # real layout: only meaningful if interleaved real/imag (even channels)
    if x.dim() == 4 and x.shape[1] % 2 == 0 and x.shape[1] >= 2:
        real = x[:, 0]
        imag = x[:, 1]
        return torch.complex(real.float(), imag.float()).detach()
    return None


def _foreground_mask(
    ctx: MetricContext, mag_bchw: torch.Tensor, *, rel: float = 0.05
) -> torch.Tensor:
    """Boolean foreground mask :math:`\\Omega` as ``[B, 1, H, W]``.

    Prefers ``ctx.foreground_mask`` (broadcast/cropped to match), else derives
    a relative-threshold mask ``|x̂| > rel * max|x̂|`` per image. Never imports a
    physics module (spec instruction).
    """
    if ctx.foreground_mask is not None:
        fg = ctx.foreground_mask
        if fg.dim() == 2:
            fg = fg[None, None]
        elif fg.dim() == 3:
            fg = fg[None]
        if fg.shape[-2:] == mag_bchw.shape[-2:]:
            # collapse any channel dim to one boolean plane
            fg = fg[:, :1]
            if fg.shape[0] != mag_bchw.shape[0] and fg.shape[0] == 1:
                fg = fg.expand(mag_bchw.shape[0], -1, -1, -1)
            return fg.to(torch.bool)
    per_image_max = mag_bchw.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    per_image_max = per_image_max.clamp_min(1e-12)
    return mag_bchw > (rel * per_image_max)


def _pe_axis(ctx: MetricContext) -> int:
    """Resolve the phase-encode spatial axis index into a 4-D ``[B,C,H,W]``.

    Honours ``ctx.nr_params['pe_axis']`` (-1/-2 or 2/3); defaults to the last
    axis (W, ``-1``). Readout is the complementary spatial axis.
    """
    if ctx.nr_params is not None:
        ax = ctx.nr_params.get("pe_axis")
        if ax in (-1, 3):
            return -1
        if ax in (-2, 2):
            return -2
    return -1


# ---------------------------------------------------------------------------
# pegc — Phase-Encode Ghost Coherence
# ---------------------------------------------------------------------------


@register_metric(
    "pegc",
    aliases=["phase_encode_ghost_coherence", "ghost_coherence"],
    requires_reference=False,
    needs=("foreground_mask",),
)
class PhaseEncodeGhostCoherence:
    r"""Phase-Encode Ghost Coherence (↓ — lower is better).

    Motion / off-resonance ghosts replicate the object shifted along the
    phase-encode (PE) axis, depositing coherent signal in the *background*
    :math:`\Omega^c` that correlates with the foreground :math:`\Omega`
    shifted by the ghost displacement :math:`\Delta`. PEGC is the worst-case
    such correlation over candidate displacements:

    .. math::

        \mathrm{PEGC}=\max_{\Delta\in\mathcal{D}}
        \frac{\langle |\hat{\mathbf{x}}|\cdot\mathbf{1}_{\Omega^c},\,
        T_\Delta(|\hat{\mathbf{x}}|\cdot\mathbf{1}_{\Omega})\rangle}
        {\lVert |\hat{\mathbf{x}}|\cdot\mathbf{1}_{\Omega}\rVert_2^2}

    :math:`T_\Delta` is a cyclic shift along the PE axis. ``≈ 0`` for a clean
    image (the shifted foreground lands on more foreground, not background);
    rises with simulated ghosting.
    """

    def __init__(self, num_shifts: int = 16, min_shift_frac: float = 0.1) -> None:
        self.num_shifts = int(num_shifts)
        self.min_shift_frac = float(min_shift_frac)

    @property
    def name(self) -> str:
        return "pegc"

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
        ctx = resolve_context(context, kwargs)
        mag = _as_bchw_magnitude(prediction)
        if mag.numel() == 0 or not torch.isfinite(mag).all():
            return float("nan")
        fg = _foreground_mask(ctx, mag)
        pe = _pe_axis(ctx)
        n_pe = mag.shape[pe]

        x_fg = mag * fg.float()  # |x| · 1_Omega
        x_bg = mag * (~fg).float()  # |x| · 1_Omega^c
        denom = (x_fg**2).flatten(1).sum(dim=1) + 1e-12  # [B]

        lo = max(1, int(self.min_shift_frac * n_pe))
        # candidate displacements span PE axis (exclude trivial 0-shift)
        cands = torch.linspace(
            lo, n_pe - 1, steps=min(self.num_shifts, n_pe - lo), dtype=torch.float32
        )
        cand_ints = sorted({round(c.item()) for c in cands if round(c.item()) >= lo})
        if not cand_ints:
            return float("nan")

        best = torch.zeros(mag.shape[0], dtype=torch.float32, device=mag.device)
        for delta in cand_ints:
            shifted = torch.roll(x_fg, shifts=delta, dims=pe)
            corr = (x_bg * shifted).flatten(1).sum(dim=1) / denom  # [B]
            best = torch.maximum(best, corr)
        return float(best.mean())


# ---------------------------------------------------------------------------
# are — Aliasing-Replica Energy
# ---------------------------------------------------------------------------


@register_metric(
    "are",
    aliases=["aliasing_replica_energy"],
    requires_reference=False,
    needs=("acceleration", "foreground_mask"),
)
class AliasingReplicaEnergy:
    r"""Aliasing-Replica Energy (↓ — lower is better).

    Under-sampled parallel imaging deposits an aliased replica of the object
    shifted by :math:`\Delta_R=N_{\mathrm{PE}}/R`. Residual (un-cancelled)
    aliasing shows up as foreground energy at the replica location landing in
    the background:

    .. math::

        \mathrm{ARE}=\frac{\lVert(\mathbf{I}-\mathbf{P}_\Omega)\,
        T_{\Delta_R}(|\hat{\mathbf{x}}|\cdot\mathbf{1}_\Omega)\rVert_2}
        {\lVert |\hat{\mathbf{x}}|\cdot\mathbf{1}_\Omega\rVert_2}

    :math:`\mathbf{P}_\Omega` projects onto the foreground (multiply by
    :math:`\mathbf{1}_\Omega`). ``→ 0`` as :math:`R\to 1` (the replica
    coincides with the object, so the off-foreground residual vanishes); rises
    with residual undersampling aliasing.
    """

    @property
    def name(self) -> str:
        return "are"

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
        ctx = resolve_context(context, kwargs)
        if ctx.acceleration is None:
            return float("nan")
        mag = _as_bchw_magnitude(prediction)
        if mag.numel() == 0 or not torch.isfinite(mag).all():
            return float("nan")
        r = float(ctx.acceleration)
        if not math.isfinite(r) or r < 1.0:
            return float("nan")
        fg = _foreground_mask(ctx, mag)
        pe = _pe_axis(ctx)
        n_pe = mag.shape[pe]

        x_fg = mag * fg.float()
        denom = torch.sqrt((x_fg**2).flatten(1).sum(dim=1) + 1e-12)  # [B]

        delta_r = round(n_pe / r) % n_pe  # R=1 -> 0 (replica==object)
        shifted = torch.roll(x_fg, shifts=delta_r, dims=pe)
        # (I - P_Omega): keep only what lands OUTSIDE the foreground
        residual = shifted * (~fg).float()
        num = torch.sqrt((residual**2).flatten(1).sum(dim=1) + 1e-12)  # [B]
        return float((num / denom).mean())


# ---------------------------------------------------------------------------
# gri — Gibbs-Ringing Index
# ---------------------------------------------------------------------------


@register_metric(
    "gri",
    aliases=["gibbs_ringing_index"],
    requires_reference=False,
)
class GibbsRingingIndex:
    r"""Gibbs-Ringing Index (→, target ≈ 0.09; lower-ringing ⇒ better).

    Truncating k-space convolves the image with a sinc whose first side-lobe
    overshoots a step edge by the Gibbs constant ≈ 9 % of the step height,
    regardless of edge contrast. Along the gradient-normal direction at each
    strong edge we read the first overshoot :math:`o_e` past the post-edge
    asymptote and the step height :math:`h_e`:

    .. math::

        \mathrm{GRI}=\operatorname{median}_{e\in\mathcal{E}}\frac{o_e}{h_e}

    A faithful truncated reconstruction yields ``≈ 0.09``; **over-smoothing**
    (windowing / blur) suppresses the overshoot toward ``0``. Direction:
    less ringing is preferable, so ``higher_is_better=False`` (lower GRI =
    smoother = better-behaved); the spec target is informational.
    """

    def __init__(self, edge_quantile: float = 0.9, profile_len: int = 12) -> None:
        self.edge_quantile = float(edge_quantile)
        self.profile_len = int(profile_len)

    @property
    def name(self) -> str:
        return "gri"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _grad(self, mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Central-difference gradients (gy, gx) for ``[B,1,H,W]``."""
        gy = torch.zeros_like(mag)
        gx = torch.zeros_like(mag)
        gy[..., 1:-1, :] = 0.5 * (mag[..., 2:, :] - mag[..., :-2, :])
        gx[..., :, 1:-1] = 0.5 * (mag[..., :, 2:] - mag[..., :, :-2])
        return gy, gx

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        resolve_context(context, kwargs)  # GRI needs no context, keep contract
        mag = _as_bchw_magnitude(prediction)
        if mag.numel() == 0 or not torch.isfinite(mag).all():
            return float("nan")
        b, _, h, w = mag.shape
        gy, gx = self._grad(mag)
        grad_mag = torch.sqrt(gy**2 + gx**2 + 1e-12)

        ratios: list[float] = []
        wlen = self.profile_len
        for bi in range(b):
            gm = grad_mag[bi, 0]
            img = mag[bi, 0]
            # strong-edge threshold (quantile of the gradient magnitude)
            thr = robust_quantile(gm.flatten(), self.edge_quantile)
            ys, xs = torch.where(gm > thr)
            if ys.numel() == 0:
                continue
            # subsample edges for speed/stability
            if ys.numel() > 64:
                idx = torch.linspace(0, ys.numel() - 1, 64).long()
                ys, xs = ys[idx], xs[idx]
            gyb, gxb = gy[bi, 0], gx[bi, 0]
            for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=False):
                # gradient-normal unit direction at this edge
                ny, nx = gyb[y0, x0].item(), gxb[y0, x0].item()
                norm = math.hypot(ny, nx)
                if norm < 1e-8:
                    continue
                ny, nx = ny / norm, nx / norm
                # sample the profile a few px each side of the edge
                lo_vals, hi_vals = [], []
                for d in range(1, wlen + 1):
                    yl = round(y0 - d * ny)
                    xl = round(x0 - d * nx)
                    yh = round(y0 + d * ny)
                    xh = round(x0 + d * nx)
                    if 0 <= yl < h and 0 <= xl < w:
                        lo_vals.append(img[yl, xl].item())
                    if 0 <= yh < h and 0 <= xh < w:
                        hi_vals.append(img[yh, xh].item())
                if len(lo_vals) < 4 or len(hi_vals) < 4:
                    continue
                # Robust asymptote per side: median of the OUTER half of the
                # profile (past the ringing zone). The Gibbs overshoot lives
                # near the edge, so the outer median is the true plateau.
                lo_asym = float(torch.tensor(lo_vals[len(lo_vals) // 2 :]).median())
                hi_asym = float(torch.tensor(hi_vals[len(hi_vals) // 2 :]).median())
                bright_side = hi_vals if hi_asym >= lo_asym else lo_vals
                bright_asym = max(hi_asym, lo_asym)
                dark_asym = min(hi_asym, lo_asym)
                step = bright_asym - dark_asym
                if step < 1e-6:
                    continue
                # overshoot = first peak past the bright plateau
                overshoot = max(bright_side) - bright_asym
                if overshoot <= 0:
                    continue
                ratios.append(overshoot / step)
        if not ratios:
            return float("nan")
        return float(torch.tensor(ratios).median())


# ---------------------------------------------------------------------------
# cpge — Complex Phase-Gradient Entropy
# ---------------------------------------------------------------------------


@register_metric(
    "cpge",
    aliases=["complex_phase_gradient_entropy"],
    requires_reference=False,
    needs=("foreground_mask",),
)
class ComplexPhaseGradientEntropy:
    r"""Complex Phase-Gradient Entropy (↓ — more negative is better).

    A faithful reconstruction has a *smooth* phase inside the object
    :math:`\Omega` (low entropy of phase gradients) but random phase in the
    air background :math:`\Omega^c` (high entropy). The difference is strongly
    negative; phase corruption (wraps, motion) roughens the foreground phase
    and drives the difference toward 0:

    .. math::

        \mathrm{CPGE}=H\!\big(h_\Omega(\nabla\angle\hat{\mathbf{x}})\big)
        -H\!\big(h_{\Omega^c}(\nabla\angle\hat{\mathbf{x}})\big)

    :math:`\nabla\angle` is a wrap-aware finite difference of the phase. Needs
    a *complex* prediction (interleaved real/imag or native complex); a
    real-only input returns ``nan``.
    """

    def __init__(self, bins: int = 64) -> None:
        self.bins = int(bins)

    @property
    def name(self) -> str:
        return "cpge"

    @property
    def higher_is_better(self) -> bool:
        return False

    @staticmethod
    def _wrap(d: torch.Tensor) -> torch.Tensor:
        """Wrap phase differences into ``(-pi, pi]``."""
        return torch.remainder(d + math.pi, 2 * math.pi) - math.pi

    def _entropy(self, vals: torch.Tensor) -> float:
        if vals.numel() < 2:
            return 0.0
        hist = torch.histc(vals, bins=self.bins, min=-math.pi, max=math.pi)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        if p.numel() == 0:
            return 0.0
        return float(-(p * torch.log(p)).sum())

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        cplx = _as_bhw_complex(prediction)  # [B, H, W] complex or None
        if cplx is None:
            return float("nan")
        if cplx.numel() == 0 or not torch.isfinite(cplx.abs()).all():
            return float("nan")
        mag = cplx.abs()[:, None]  # [B,1,H,W] for the foreground mask
        fg = _foreground_mask(ctx, mag)[:, 0]  # [B, H, W] bool
        phase = torch.angle(cplx)  # [B, H, W]
        # wrap-aware gradient magnitude proxy of the phase
        dy = torch.zeros_like(phase)
        dx = torch.zeros_like(phase)
        dy[:, 1:, :] = self._wrap(phase[:, 1:, :] - phase[:, :-1, :])
        dx[:, :, 1:] = self._wrap(phase[:, :, 1:] - phase[:, :, :-1])
        grad = self._wrap(dy + dx)  # combined, kept in (-pi, pi]

        out: list[float] = []
        for bi in range(cplx.shape[0]):
            fmask = fg[bi]
            bmask = ~fmask
            g = grad[bi]
            h_fg = self._entropy(g[fmask])
            h_bg = self._entropy(g[bmask])
            out.append(h_fg - h_bg)
        if not out:
            return float("nan")
        return float(torch.tensor(out).mean())


# ---------------------------------------------------------------------------
# rar — Resolution-Anisotropy Ratio
# ---------------------------------------------------------------------------


@register_metric(
    "rar",
    aliases=["resolution_anisotropy_ratio"],
    requires_reference=False,
)
class ResolutionAnisotropyRatio:
    r"""Resolution-Anisotropy Ratio (→, target 1.0; deviation ⇒ worse).

    Directional blur (common when the phase-encode axis is under-resolved)
    makes edges sharper along readout (RO) than phase-encode (PE). We measure
    a directional edge acutance — the strongest signed directional gradient
    normalised by the local step height — along each spatial axis (the
    Comprehensive Image Edge Acutance, CIEA, restricted to one axis):

    .. math::

        \mathrm{RAR}=\frac{\mathrm{CIEA}_{\mathrm{RO}}}{\mathrm{CIEA}_{\mathrm{PE}}}

    ``≈ 1`` for isotropic resolution; deviation (``> 1`` or ``< 1``) signals
    anisotropic blur. The reported score is :math:`|\mathrm{RAR} - 1|`, the
    absolute deviation from the isotropic target ``1.0`` (``0`` = best), so
    ``higher_is_better = False`` (closer to isotropic ⇒ smaller score ⇒ ranked
    higher). It is a target-value (→ 1.0) metric.
    """

    def __init__(self, edge_quantile: float = 0.85) -> None:
        self.edge_quantile = float(edge_quantile)

    @property
    def name(self) -> str:
        return "rar"

    @property
    def higher_is_better(self) -> bool:
        # Target-value metric (→ 1.0). We report the *deviation* score
        # |RAR - 1| so that "closer to isotropic ⇒ smaller ⇒ ranked higher".
        return False

    def _directional_acutance(self, mag: torch.Tensor, axis: int) -> torch.Tensor:
        """Directional edge acutance = peak directional gradient / step height.

        Acutance is the *sharpness* of the steepest transition along ``axis``;
        directional blur lowers the peak gradient on the blurred axis while
        leaving the orthogonal axis intact, so the RO/PE ratio departs from 1.
        Using a high percentile of the *forward* first difference (rather than
        the mean over a quantile set) makes the measure robust to the blur
        lowering the whole gradient field.

        ``axis`` is a spatial axis of ``[B,1,H,W]`` (-2 = rows, -1 = cols).
        Returns ``[B]``.
        """
        g = (
            mag[..., 1:, :] - mag[..., :-1, :]  # row (vertical) forward diff
            if axis == -2
            else mag[..., :, 1:] - mag[..., :, :-1]  # col (horizontal) forward diff
        )
        ag = g.abs()
        b = mag.shape[0]
        out = torch.zeros(b, dtype=torch.float32)
        for bi in range(b):
            a = ag[bi, 0].flatten()
            if a.numel() == 0:
                continue
            # peak directional slope (robust high quantile, not the absolute max
            # which is sensitive to a single discretisation spike).
            peak = torch.quantile(a, 0.999)
            img = mag[bi, 0]
            step = (img.max() - img.min()).clamp_min(1e-6)
            out[bi] = peak / step
        return out

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_bchw_magnitude(prediction)
        if mag.numel() == 0 or not torch.isfinite(mag).all():
            return float("nan")
        pe = _pe_axis(ctx)  # -1 or -2
        ro = -1 if pe == -2 else -2
        a_ro = self._directional_acutance(mag, ro)
        a_pe = self._directional_acutance(mag, pe)
        ratio = a_ro / (a_pe + 1e-8)  # [B]
        valid = torch.isfinite(ratio) & (a_pe > 1e-8)
        if not bool(valid.any()):
            return float("nan")
        rar = ratio[valid].mean()
        # report deviation from the isotropic target (1.0); 0 ⇒ best
        return float((rar - 1.0).abs())


__all__ = [
    "AliasingReplicaEnergy",
    "ComplexPhaseGradientEntropy",
    "GibbsRingingIndex",
    "PhaseEncodeGhostCoherence",
    "ResolutionAnisotropyRatio",
]
