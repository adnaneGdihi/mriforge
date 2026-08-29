r"""No-reference *stability* metric battery (spec §3 ``nr_stability`` + PEDU).

These three metrics interrogate the **reconstruction operator** :math:`\mathcal{R}`
rather than the reconstructed image alone. They are reference-free: each
answers "how stable / self-consistent is this reconstructor at the test
measurement?" using the :class:`~mriforge.core.metrics.context.MetricContext`
seam (a frozen reconstructor, the acquired k-space, the mask, an RNG).

* ``lrjs`` — Local Reconstruction-Jacobian Spectral Norm. The local Lipschitz
  constant :math:`\lVert\partial\mathcal{R}/\partial\mathbf{y}\rVert_2`,
  estimated by power iteration on :math:`\mathbf{J}^H\mathbf{J}` using
  ``torch.func`` JVP/VJP. Large ⇒ an ill-conditioned reconstructor that
  amplifies measurement noise. *Lower is better.*
* ``fmed`` — Forward-Model Equivariance Defect. RMS commutator of
  :math:`\mathcal{R}` with a rotation group, delegated to the existing
  :func:`mriforge.core.metrics.equivariance_defect.equivariance_defect_score`.
  Zero for the trivial group and for an exact-adjoint reconstructor; rises with
  null-space hallucination. *Lower is better.*
* ``pedu`` — Posterior/Ensemble Disagreement Uncertainty. The mean
  foreground variance across stochastic forward samples (multi-mask SSDU-style,
  or MC-dropout if the reconstructor exposes it). Epistemic uncertainty grows in
  the under-determined regime (higher acceleration). *Lower is better.*

All three return a single Python ``float``; on a missing/degenerate context
they return ``float('nan')`` so the sim2rank sweep skips them cleanly rather
than aborting the whole validation (``core/metrics/computer.py`` re-raises
``ValueError``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from mriforge.core.metrics.context import MetricContext, resolve_context
from mriforge.core.metrics.registry import register_metric

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _flat_real(x: Tensor) -> Tensor:
    """Flatten an arbitrary tensor to a 1-D *real* vector.

    Complex tensors are split into stacked real/imag so the power-iteration and
    variance reductions operate in a well-defined real inner-product space.
    """
    if x.is_complex():
        return torch.cat([x.real.reshape(-1), x.imag.reshape(-1)])
    return x.reshape(-1)


def _to_mag_image(x: Tensor) -> Tensor:
    """Coerce ``[B,C,H,W]`` (complex / interleaved r-i / single) to magnitude.

    Mirrors :func:`mriforge.core.metrics.hallucination_metrics._to_magnitude`
    semantics but kept local so this module stays import-light (the auto-
    discovery walker swallows ImportError, so a fragile cross-import would
    silently kill every registration in this file).
    """
    if x.is_complex():
        return x.abs()
    if x.dim() == 4 and x.shape[1] % 2 == 0 and x.shape[1] >= 2:
        c = x.shape[1] // 2
        re, im = x[:, :c], x[:, c:]
        return torch.sqrt(re.pow(2) + im.pow(2) + 1e-12)
    return x.abs()


def _foreground_mask(prediction: Tensor, ctx: MetricContext) -> Tensor:
    """Return a boolean foreground mask ``Omega`` over ``prediction``.

    Uses ``ctx.foreground_mask`` when supplied, else a magnitude threshold at
    ``5%`` of the per-image max (an Otsu-lite proxy, per the spec's fallback
    note). Returns ``[B,1,H,W]`` boolean aligned with the magnitude image.
    """
    mag = _to_mag_image(prediction)
    if ctx.foreground_mask is not None:
        fg = ctx.foreground_mask
        if fg.shape == mag.shape:
            return fg.bool()
        # Broadcast a coarse mask to the magnitude grid if shapes differ.
        return fg.reshape(fg.shape[0], 1, *fg.shape[-2:]).bool()
    peak = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return mag > (0.05 * peak)


# ----------------------------------------------------------------------------
# lrjs — Local Reconstruction-Jacobian Spectral Norm
# ----------------------------------------------------------------------------


@register_metric(
    "lrjs",
    aliases=["jacobian_spectral_norm", "recon_jacobian_norm"],
    requires_reference=False,
    needs=("reconstructor", "y_kspace"),
)
class LocalReconstructionJacobianSpectralNorm:
    r"""Local Reconstruction-Jacobian Spectral Norm — lower is better.

    The largest singular value of the reconstruction Jacobian
    :math:`\mathbf{J}=\partial\mathcal{R}/\partial\mathbf{y}` evaluated at the
    test measurement, i.e. the local Lipschitz constant of the reconstructor:

    .. math::

        \mathrm{LRJS}=\lVert\mathbf{J}\rVert_2
            =\sqrt{\lambda_{\max}(\mathbf{J}^{H}\mathbf{J})},\qquad
        \mathbf{v}\leftarrow\frac{\mathbf{J}^{H}\mathbf{J}\,\mathbf{v}}
            {\lVert\mathbf{J}^{H}\mathbf{J}\,\mathbf{v}\rVert}.

    Estimated by power iteration on :math:`\mathbf{J}^H\mathbf{J}` using
    ``torch.func`` forward-mode (``jvp``) for :math:`\mathbf{J}\mathbf{v}` and
    reverse-mode (``vjp``) for :math:`\mathbf{J}^H u`. A large spectral norm
    flags a reconstructor that amplifies measurement perturbations (noise,
    motion), so smaller is more stable. Returns ``nan`` when no differentiable
    reconstructor / measurement is in context.

    Args:
        num_iters: Power-iteration steps (default 8).
        eps: Numerical floor for normalisation.
    """

    def __init__(self, num_iters: int = 8, eps: float = 1e-8) -> None:
        self.num_iters = int(num_iters)
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "lrjs"

    @property
    def higher_is_better(self) -> bool:
        # An ill-conditioned reconstructor (large ||J||) amplifies noise — bad.
        return False

    def _call_recon(self, recon: Callable[..., Tensor], y: Tensor, ctx: MetricContext) -> Tensor:
        if ctx.coil_maps is not None:
            return recon(y, ctx.coil_maps)
        return recon(y)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("reconstructor", "y_kspace"):
            return float("nan")
        recon = ctx.reconstructor
        y0 = ctx.y_kspace
        assert recon is not None and y0 is not None

        n_iters = self.num_iters
        if ctx.nr_params and "lrjs_num_iters" in ctx.nr_params:
            n_iters = int(ctx.nr_params["lrjs_num_iters"])

        # Single forward function R(y) -> image, with smaps curried in.
        def fwd(y: Tensor) -> Tensor:
            return self._call_recon(recon, y, ctx)

        try:
            y = y0.detach().clone().requires_grad_(False)
            # Seed direction: deterministic ones (no global-seed dependence).
            v = torch.ones_like(y)
            v = v / (_flat_real(v).norm() + self.eps)
            # vjp closure: J^H u for a given primal y.
            _out0, vjp_fn = torch.func.vjp(fwd, y)
            sigma = torch.tensor(0.0)
            for _ in range(max(1, n_iters)):
                # Jv via forward-mode JVP.
                _, jv = torch.func.jvp(fwd, (y,), (v,))
                # J^H (Jv) via the cached reverse-mode closure.
                (jhjv,) = vjp_fn(jv)
                nrm = _flat_real(jhjv).norm()
                if not torch.isfinite(nrm) or nrm < self.eps:
                    return float("nan")
                v = jhjv / nrm
                # Rayleigh quotient gives lambda_max(J^H J) = sigma_max^2.
                sigma = nrm  # ||J^H J v|| with unit v converges to lambda_max
            spectral = math.sqrt(max(float(sigma.item()), 0.0))
        except (RuntimeError, TypeError):
            # Non-differentiable reconstructor (e.g. numpy/closure that breaks
            # functorch) — degrade to nan, never raise (CLAUDE.md pitfall #9 on
            # the *metric* side: a missing capability skips, not aborts).
            return float("nan")

        if not math.isfinite(spectral):
            return float("nan")
        return float(spectral)


# ----------------------------------------------------------------------------
# fmed — Forward-Model Equivariance Defect
# ----------------------------------------------------------------------------


@register_metric(
    "fmed",
    aliases=["forward_model_equivariance_defect"],
    requires_reference=False,
    needs=("reconstructor",),
)
class ForwardModelEquivarianceDefect:
    r"""Forward-Model Equivariance Defect — lower is better.

    The root-mean-square commutator of the reconstructor with a rotation group
    :math:`\mathcal{G}`:

    .. math::

        \mathrm{FMED}=\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}
            \frac{\lVert T_g\hat{\mathbf{x}}
                  -\mathcal{R}(\mathbf{A}\,T_g\hat{\mathbf{x}})\rVert_2}
                 {\lVert\hat{\mathbf{x}}\rVert_2}.

    A perfectly equivariant reconstructor commutes with the group action, so the
    defect is :math:`0`; an exact adjoint reconstructor on a fully-sampled
    measurement is equivariant and also scores :math:`0`. The defect rises with
    null-space hallucination (the operator behaves differently on rotated
    inputs). This is a thin wrapper over the existing
    :func:`mriforge.core.metrics.equivariance_defect.equivariance_defect_score`
    — the group machinery (sampling, rotation, commutator) is **not**
    reimplemented here. Returns ``nan`` when no reconstructor is in context.

    Args:
        num_rotations: ``K`` group elements in the Monte-Carlo estimate.
        group: ``"SO2"`` or ``"SE2"``; honoured by the underlying scorer.
        seed: Per-sample seed for the deterministic rotation sampler.
    """

    def __init__(
        self,
        num_rotations: int = 8,
        group: str = "SO2",
        seed: int = 0,
    ) -> None:
        self.num_rotations = int(num_rotations)
        self.group = str(group)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "fmed"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("reconstructor"):
            return float("nan")
        recon = ctx.reconstructor
        assert recon is not None

        # Operate on the measurement when available, else on the reconstruction
        # itself (the scorer rotates the *input* to the reconstructor).
        y = ctx.y_kspace if ctx.y_kspace is not None else prediction
        if y is None:
            return float("nan")

        num_rotations = self.num_rotations
        group = self.group
        if ctx.nr_params:
            num_rotations = int(ctx.nr_params.get("fmed_num_rotations", num_rotations))
            group = str(ctx.nr_params.get("fmed_group", group))

        # Lazy import keeps this module light and lets discovery survive even if
        # the equivariance module ever grows a heavy dependency.
        from mriforge.core.metrics.equivariance_defect import (
            SUPPORTED_GROUPS,
            equivariance_defect_score,
        )

        # No silent fallback on an unknown group — but on the *metric* path a
        # bad nr_param should skip, not crash the sweep; surface via nan.
        if group not in SUPPORTED_GROUPS:
            return float("nan")

        if y.dim() != 4:
            return float("nan")

        try:
            per_sample = equivariance_defect_score(
                recon,
                y,
                num_rotations=num_rotations,
                group=group,
                seed=self.seed,
                coil_sensitivities=ctx.coil_maps,
            )
        except (RuntimeError, ValueError, TypeError):
            return float("nan")

        val = float(per_sample.mean().item())
        if not math.isfinite(val):
            return float("nan")
        return val


# ----------------------------------------------------------------------------
# pedu — Posterior/Ensemble Disagreement Uncertainty
# ----------------------------------------------------------------------------


@register_metric(
    "pedu",
    aliases=["posterior_disagreement", "ensemble_disagreement_uncertainty"],
    requires_reference=False,
    needs=("reconstructor", "generator"),
)
class PosteriorEnsembleDisagreementUncertainty:
    r"""Posterior/Ensemble Disagreement Uncertainty — lower is better.

    The mean foreground variance over a set of stochastic forward samples
    :math:`\{\hat{\mathbf{x}}^{(s)}\}`:

    .. math::

        \mathrm{PEDU}=\operatorname{mean}_\Omega\operatorname{Var}_s
            [\hat{\mathbf{x}}^{(s)}].

    The stochastic source is, in order of preference: (1) MC-dropout, if the
    reconstructor exposes a ``train()``/``eval()`` toggle and dropout layers;
    (2) multi-mask SSDU — re-running the reconstructor on several random
    sub-masks of ``ctx.mask`` (or freshly drawn random masks when no mask is in
    context). All randomness is drawn from ``ctx.generator`` so the metric is
    deterministic for a fixed seed. Epistemic uncertainty rises in the
    under-determined regime, so PEDU grows with acceleration; smaller PEDU means
    a more decisive (better-posed) reconstruction. Returns ``nan`` when neither
    stochastic source is available.

    Args:
        n_samples: Number of stochastic forward samples (default 8).
        submask_keep: Fraction of the *acquired* lines to keep per SSDU sample.
        eps: Numerical floor.
    """

    def __init__(
        self,
        n_samples: int = 8,
        submask_keep: float = 0.6,
        eps: float = 1e-8,
    ) -> None:
        self.n_samples = int(n_samples)
        self.submask_keep = float(submask_keep)
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "pedu"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _call_recon(self, recon: Callable[..., Tensor], y: Tensor, ctx: MetricContext) -> Tensor:
        if ctx.coil_maps is not None:
            return recon(y, ctx.coil_maps)
        return recon(y)

    def _has_dropout(self, recon: Any) -> bool:
        if not hasattr(recon, "modules") or not hasattr(recon, "train"):
            return False
        try:
            return any(isinstance(m, torch.nn.modules.dropout._DropoutNd) for m in recon.modules())
        except (AttributeError, TypeError):
            return False

    def _random_submask(self, mask: Tensor, gen: torch.Generator) -> Tensor:
        """Keep a random ``submask_keep`` fraction of the acquired entries."""
        keep = torch.rand(mask.shape, generator=gen, device=mask.device)
        sub = (mask > 0) & (keep <= self.submask_keep)
        return sub.to(mask.dtype)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("reconstructor", "generator"):
            return float("nan")
        recon = ctx.reconstructor
        gen = ctx.generator
        assert recon is not None and gen is not None

        n_samples = self.n_samples
        if ctx.nr_params and "pedu_n_samples" in ctx.nr_params:
            n_samples = int(ctx.nr_params["pedu_n_samples"])
        n_samples = max(2, n_samples)

        y0 = ctx.y_kspace
        samples: list[Tensor] = []

        try:
            if self._has_dropout(recon) and y0 is not None:
                # (1) MC-dropout: enable dropout, re-run with per-sample RNG.
                was_training = bool(getattr(recon, "training", False))
                recon.train()  # type: ignore[attr-defined]
                try:
                    # Fork the global RNG so the per-sample ``torch.manual_seed``
                    # below cannot leak into the surrounding process RNG — one
                    # metric's RNG consumption must not cascade into another
                    # metric's result (the contract-test drift, WS-2 6.3).
                    # ``devices=[]`` skips the CUDA-state fork; the per-sample
                    # seed is drawn from the LOCAL ``gen`` (unaffected by the
                    # fork), so the sampled values are byte-identical.
                    with torch.random.fork_rng(devices=[]):
                        for _ in range(n_samples):
                            # Seed the *global* dropout RNG deterministically
                            # from ctx.generator so MC-dropout is reproducible.
                            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
                            torch.manual_seed(seed)
                            out = self._call_recon(recon, y0, ctx)
                            samples.append(_to_mag_image(out).detach())
                finally:
                    recon.train(was_training)  # type: ignore[attr-defined]
            elif ctx.mask is not None and y0 is not None:
                # (2) multi-mask SSDU: random sub-masks of the acquired k-space.
                for _ in range(n_samples):
                    sub = self._random_submask(ctx.mask, gen)
                    y_sub = y0 * sub
                    out = self._call_recon(recon, y_sub, ctx)
                    samples.append(_to_mag_image(out).detach())
            else:
                # No stochastic source we can drive deterministically.
                return float("nan")
        except (RuntimeError, TypeError):
            return float("nan")

        if len(samples) < 2:
            return float("nan")

        stack = torch.stack(samples, dim=0)  # [S, B, 1, H, W]
        var = stack.var(dim=0, unbiased=False)  # [B, 1, H, W]
        fg = _foreground_mask(prediction, ctx)
        if fg.shape != var.shape:
            # Align coil/channel: variance is single-channel magnitude.
            fg = fg.reshape(var.shape) if fg.numel() == var.numel() else fg[:, :1]
        denom = fg.sum().clamp_min(1.0)
        pedu = (var * fg.to(var.dtype)).sum() / denom
        val = float(pedu.item())
        if not math.isfinite(val):
            return float("nan")
        return val


__all__ = [
    "ForwardModelEquivarianceDefect",
    "LocalReconstructionJacobianSpectralNorm",
    "PosteriorEnsembleDisagreementUncertainty",
]
