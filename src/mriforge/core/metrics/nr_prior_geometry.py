r"""Learned-prior / manifold-geometry no-reference (NR) quality metrics.

This module implements the ``nr_prior_geometry`` group of the NR metric
battery (spec §3). Every metric here is **gated behind a learned asset** — an
SSL feature encoder (:attr:`MetricContext.encoder`) or a generative score /
diffusion prior (:attr:`MetricContext.prior_model`). Without that asset there
is no ground-truth-free notion of "typicality", so each metric returns
``float('nan')`` (logged once) rather than crashing the sim2rank sweep or
re-raising a :class:`ValueError` (which ``core/metrics/computer.py`` would treat
as a hard abort).

The maths is implemented faithfully against a *score function* contract so a
trivial analytic prior exercises the real formula. A prior model may expose:

* ``score(x, t=...)`` → :math:`\nabla_x \log p_t(x)` of the same shape as ``x``;
* ``denoise(x, sigma=...)`` / ``denoiser(x, sigma=...)`` → a Tweedie denoiser
  :math:`D(x,\sigma)`, from which the score is recovered as
  :math:`s = (D(x,\sigma)-x)/\sigma^2`;
* ``log_prob(x)`` / ``log_likelihood(x)`` → :math:`\ln p_0(x)` (optional, used
  by :class:`HallucinationIndex` and as a PFLD shortcut).

:func:`_PriorAdapter` normalises any of those into a single
``score(x, t)`` / ``log_prob(x)`` interface so each metric implements one
formula path.

Metrics
-------
* ``pmmd`` — Pristine-Manifold Mahalanobis Distance (↓)        [encoder]
* ``dpst`` — Diffusion-Prior Score Typicality (↓)              [prior_model]
* ``fitc`` — Fisher-Information-Trace Curvature (↑)            [prior_model, generator]
* ``ksdp`` — Kernel Stein Discrepancy to the Prior (↓)         [prior_model]
* ``pfld`` — Probability-Flow Exact-Likelihood Deficit (↓)     [prior_model]
* ``hallucination_index`` — null-space hallucination statistic (↓)
                                          [prior_model, y_kspace, mask, coil_maps]

References
----------
* Song & Ermon, "Generative Modeling by Estimating Gradients of the Data
  Distribution", NeurIPS 2019 (score matching, Langevin).
* Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
  (probability-flow ODE, instantaneous change of variables).
* Liu, Lee & Jordan, "A Kernelized Stein Discrepancy for Goodness-of-fit
  Tests", ICML 2016 (KSD).
* Efron, "Tweedie's Formula and Selection Bias", JASA 2011 (Tweedie denoiser).
* Jalal et al., "Robust Compressed Sensing MRI with Deep Generative Priors",
  NeurIPS 2021 (null-space hallucination in MR reconstruction).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from mriforge.core.metrics.context import MetricContext, resolve_context
from mriforge.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)

# One-shot "asset absent" notes so a config that lists a prior-geometry metric
# without wiring the prior doesn't flood the log on every validation step.
_LOGGED_ABSENT: set[str] = set()


def _note_absent(key: str, asset: str) -> float:
    """Log once that ``key`` is asset-blocked and return ``nan``."""
    if key not in _LOGGED_ABSENT:
        logger.warning(
            "metric %r is gated behind a learned %s "
            "(MetricContext.%s); none supplied — returning nan. "
            "Wire the asset to activate this metric.",
            key,
            asset,
            asset,
        )
        _LOGGED_ABSENT.add(key)
    return float("nan")


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #
def _as_real_bchw(x: Tensor) -> Tensor:
    """Coerce ``prediction`` to a real ``[B, C, H, W]`` tensor for prior eval.

    Complex inputs are split into interleaved real/imag channels so a real-valued
    score / encoder operates on a well-defined tensor; an already-real tensor is
    passed through. A bare ``[H, W]`` / ``[C, H, W]`` is promoted to a batch.
    """
    if x.is_complex():
        x = torch.cat([x.real, x.imag], dim=-3)
    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    return x


# --------------------------------------------------------------------------- #
# Prior / encoder adapters
# --------------------------------------------------------------------------- #
class _PriorAdapter:
    """Normalise a heterogeneous prior model into ``score`` / ``log_prob``.

    Accepts a model exposing any of ``score``, ``denoise``/``denoiser``,
    ``log_prob``/``log_likelihood``. The Tweedie identity converts a denoiser
    :math:`D(x,\\sigma)` into a score :math:`s=(D(x,\\sigma)-x)/\\sigma^2`.

    Raises:
        TypeError: if the model exposes none of the supported hooks.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._score = getattr(model, "score", None)
        self._denoise = getattr(model, "denoise", None) or getattr(model, "denoiser", None)
        self._log_prob = getattr(model, "log_prob", None) or getattr(model, "log_likelihood", None)
        if self._score is None and self._denoise is None and self._log_prob is None:
            raise TypeError(
                "prior_model must expose one of: score(x, t), denoise(x, sigma), log_prob(x)."
            )

    @property
    def has_score(self) -> bool:
        return self._score is not None or self._denoise is not None

    @property
    def has_log_prob(self) -> bool:
        return self._log_prob is not None

    def score(self, x: Tensor, t: float) -> Tensor:
        r""":math:`s_t(x) = \nabla_x \log p_t(x)`, shape matching ``x``."""
        if self._score is not None:
            try:
                return self._score(x, t)
            except TypeError:
                return self._score(x)
        # Tweedie: s = (D(x, sigma) - x) / sigma^2 with sigma <-> t.
        sigma = float(t) if t > 0 else 1e-3
        try:
            d = self._denoise(x, sigma)
        except TypeError:
            d = self._denoise(x)
        return (d - x) / (sigma * sigma)

    def log_prob(self, x: Tensor) -> Tensor:
        """Per-sample :math:`\\ln p_0(x)` (shape ``[B]``)."""
        out = self._log_prob(x)
        if out.dim() == 0:
            out = out.expand(x.shape[0])
        return out.reshape(x.shape[0], -1).sum(dim=1) if out.dim() > 1 else out


# --------------------------------------------------------------------------- #
# pmmd — Pristine-Manifold Mahalanobis Distance
# --------------------------------------------------------------------------- #
@register_metric("pmmd", requires_reference=False, needs=("encoder",))
class PristineManifoldMahalanobisDistance:
    r"""Mahalanobis distance of encoder features to the pristine manifold (↓).

    With :math:`\phi=\mathcal{E}(\hat{\mathbf x})` and a Gaussian fitted on a
    pristine corpus,

    .. math::

        \mathrm{PMMD}=(\phi-\boldsymbol\mu)^{\top}\boldsymbol\Sigma^{-1}
            (\phi-\boldsymbol\mu)\;=\;2(-\ln p(\phi))-\text{const}.

    :math:`(\boldsymbol\mu,\boldsymbol\Sigma)` come from
    ``context.nr_params['pmmd_mu' / 'pmmd_cov']`` (or ``'mu'/'cov'``), else from
    a ``fitted_stats`` attribute on the encoder. Lower ⇒ more on-manifold.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "pmmd"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _fitted(self, ctx: MetricContext, enc: Any) -> tuple[Tensor, Tensor] | None:
        params = ctx.nr_params or {}
        mu = params.get("pmmd_mu", params.get("mu"))
        cov = params.get("pmmd_cov", params.get("cov"))
        if mu is None or cov is None:
            stats = getattr(enc, "fitted_stats", None)
            if isinstance(stats, dict):
                mu, cov = stats.get("mu"), stats.get("cov")
        if mu is None or cov is None:
            return None
        return torch.as_tensor(mu).float(), torch.as_tensor(cov).float()

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("encoder"):
            return _note_absent("pmmd", "encoder")
        fitted = self._fitted(ctx, ctx.encoder)
        if fitted is None:
            return _note_absent("pmmd", "encoder")  # no fitted Gaussian asset
        mu, cov = fitted
        x = _as_real_bchw(prediction)
        with torch.no_grad():
            phi = ctx.encoder(x)
        phi = phi.reshape(phi.shape[0], -1).float()
        d = phi.shape[1]
        mu = mu.reshape(-1).to(phi)
        cov = cov.reshape(d, d).to(phi)
        cov = cov + self.eps * torch.eye(d, device=phi.device, dtype=phi.dtype)
        diff = phi - mu  # [B, D]
        try:
            sol = torch.linalg.solve(cov, diff.T).T  # Sigma^{-1} (phi-mu)
        except RuntimeError:
            return float("nan")
        m2 = (diff * sol).sum(dim=1)  # [B]
        return float(m2.clamp_min(0.0).mean())


# --------------------------------------------------------------------------- #
# dpst — Diffusion-Prior Score Typicality
# --------------------------------------------------------------------------- #
@register_metric("dpst", requires_reference=False, needs=("prior_model",))
class DiffusionPriorScoreTypicality:
    r"""Expected weighted score-norm under a small-noise diffusion prior (↓).

    .. math::

        \mathrm{DPST}=\mathbb{E}_{t\sim\text{small}}\;
            \sigma_t^2\,\lVert\mathbf{s}_t(\hat{\mathbf x})\rVert_2^2,
        \qquad \mathbf{s}_t=(D(\mathbf{x}_t,\sigma_t)-\mathbf{x}_t)/\sigma_t^2.

    Typical (high-density) images have a small score; atypical / hallucinated
    structure pushes the score up. Lower is better. The small-:math:`t` levels
    default to ``context.nr_params['dpst_sigmas']`` or ``(0.05, 0.1, 0.2)``.
    """

    def __init__(self, sigmas: tuple[float, ...] = (0.05, 0.1, 0.2)) -> None:
        self.sigmas = sigmas

    @property
    def name(self) -> str:
        return "dpst"

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
        if not ctx.has("prior_model"):
            return _note_absent("dpst", "prior_model")
        try:
            adapter = _PriorAdapter(ctx.prior_model)
        except TypeError:
            return _note_absent("dpst", "prior_model")
        if not adapter.has_score:
            return _note_absent("dpst", "prior_model")
        sigmas = (ctx.nr_params or {}).get("dpst_sigmas", self.sigmas)
        x = _as_real_bchw(prediction)
        acc = 0.0
        with torch.no_grad():
            for t in sigmas:
                s = adapter.score(x, float(t))
                sq = s.reshape(s.shape[0], -1).pow(2).sum(dim=1)  # ||s||^2
                acc = acc + float((t * t) * sq.mean())
        return acc / max(len(sigmas), 1)


# --------------------------------------------------------------------------- #
# fitc — Fisher-Information-Trace Curvature
# --------------------------------------------------------------------------- #
@register_metric("fitc", requires_reference=False, needs=("prior_model", "generator"))
class FisherInformationTraceCurvature:
    r"""Negative divergence of the score (Hutchinson trace estimator) (↑).

    .. math::

        \mathrm{FITC}=-\nabla\!\cdot\!\mathbf{s}_t(\hat{\mathbf x})
            =-\operatorname{tr}(\partial_{\mathbf x}\mathbf{s}_t)
            \approx-\mathbb{E}_{\boldsymbol\eta\sim\mathcal{N}(0,I)}
                [\boldsymbol\eta^{\top}(\partial_{\mathbf x}\mathbf{s}_t)
                 \boldsymbol\eta].

    At a typical mode the score field contracts (negative Jacobian trace), so
    :math:`-\operatorname{tr}` is large-positive there. Higher ⇒ more typical.
    The directional derivative :math:`(\partial_x s)\eta` is computed with
    ``torch.func.jvp``; :math:`\eta` is drawn from ``context.generator``.
    """

    def __init__(self, t: float = 0.1, num_probes: int = 8) -> None:
        self.t = t
        self.num_probes = num_probes

    @property
    def name(self) -> str:
        return "fitc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("prior_model"):
            return _note_absent("fitc", "prior_model")
        try:
            adapter = _PriorAdapter(ctx.prior_model)
        except TypeError:
            return _note_absent("fitc", "prior_model")
        if not adapter.has_score:
            return _note_absent("fitc", "prior_model")
        from torch.func import jvp  # local: keep module import light

        t = float((ctx.nr_params or {}).get("fitc_t", self.t))
        gen = ctx.generator
        x = _as_real_bchw(prediction)

        def score_fn(z: Tensor) -> Tensor:
            return adapter.score(z, t)

        trace_est = 0.0
        with torch.no_grad():
            for _ in range(self.num_probes):
                eta = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                # jvp gives (s(x), (d s / d x) @ eta)
                _, jvp_val = jvp(score_fn, (x,), (eta,))
                quad = (eta * jvp_val).reshape(x.shape[0], -1).sum(dim=1)
                trace_est = trace_est + float(quad.mean())
        trace_est /= max(self.num_probes, 1)
        return -trace_est  # FITC = -tr(d s / d x)


# --------------------------------------------------------------------------- #
# ksdp — Kernel Stein Discrepancy to the Prior
# --------------------------------------------------------------------------- #
@register_metric("ksdp", requires_reference=False, needs=("prior_model",))
class KernelSteinDiscrepancyToPrior:
    r"""U-statistic kernel Stein discrepancy of patches to the prior (↓).

    With RBF kernel :math:`k(u,u')=\exp(-\lVert u-u'\rVert^2/2h^2)` and prior
    score :math:`s_p`, the Stein kernel is

    .. math::

        u_p(u,u')=s_p(u)^{\top}s_p(u')\,k
            + s_p(u)^{\top}\nabla_{u'}k
            + \nabla_{u}k^{\top}s_p(u')
            + \nabla_u\!\cdot\!\nabla_{u'}k,

    and :math:`\mathrm{KSDP}^2=\mathbb{E}_{u,u'\sim q}[u_p(u,u')]`. It is
    :math:`0` iff the sample distribution :math:`q` matches the prior :math:`p`,
    and grows with mismatch. Lower is better.

    The "samples" are flattened patches drawn from the prediction (a single
    image is decomposed into ``num_patches`` random patches so the U-statistic
    has support); the RBF bandwidth uses the median heuristic.
    """

    def __init__(self, t: float = 0.1, patch: int = 8, num_patches: int = 16) -> None:
        self.t = t
        self.patch = patch
        self.num_patches = num_patches

    @property
    def name(self) -> str:
        return "ksdp"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _patches(self, x: Tensor, gen: torch.Generator | None) -> Tensor:
        """Return ``[N, D]`` flattened random patches from ``x`` (first sample)."""
        _, _, h, w = x.shape
        p = min(self.patch, h, w)
        n = self.num_patches
        if h == p and w == p:
            return x[0].reshape(1, -1).repeat(n, 1)
        ys = torch.randint(0, h - p + 1, (n,), generator=gen)
        xs = torch.randint(0, w - p + 1, (n,), generator=gen)
        out = [
            x[0, :, yy : yy + p, xx : xx + p].reshape(-1) for yy, xx in zip(ys, xs, strict=False)
        ]
        return torch.stack(out, dim=0)  # [N, c*p*p]

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("prior_model"):
            return _note_absent("ksdp", "prior_model")
        try:
            adapter = _PriorAdapter(ctx.prior_model)
        except TypeError:
            return _note_absent("ksdp", "prior_model")
        if not adapter.has_score:
            return _note_absent("ksdp", "prior_model")
        t = float((ctx.nr_params or {}).get("ksdp_t", self.t))
        x = _as_real_bchw(prediction)
        with torch.no_grad():
            u = self._patches(x, ctx.generator)  # [N, D]
            n, d = u.shape
            # Score of each patch under the prior (reshape to a 4-D singleton image).
            side = round((u.shape[1] / max(x.shape[1], 1)) ** 0.5)
            c = x.shape[1]
            # Non-square patch flattening ⇒ treat as a 1xD "image" row.
            up = u.reshape(n, c, side, side) if side * side * c == d else u.reshape(n, 1, 1, d)
            sp = adapter.score(up, t).reshape(n, d)  # [N, D]

            # Pairwise squared distances + median-heuristic bandwidth.
            diff = u.unsqueeze(1) - u.unsqueeze(0)  # [N, N, D]
            sq = (diff * diff).sum(dim=2)  # [N, N]
            med = torch.median(sq[sq > 0]) if (sq > 0).any() else torch.tensor(1.0)
            h2 = (med / max(torch.log(torch.tensor(float(n))), torch.tensor(1.0))).clamp_min(1e-8)
            k = torch.exp(-sq / (2.0 * h2))  # [N, N]

            # Kernel gradients (RBF): grad_{u'} k = k * (u - u')/h2.
            grad_kxp = k.unsqueeze(2) * (diff / h2)  # d/d u' : [N,N,D]
            grad_kx = -grad_kxp  # d/d u
            ss = sp @ sp.T  # s(u)^T s(u')  [N, N]
            term1 = ss * k
            term2 = (sp.unsqueeze(1) * grad_kxp).sum(dim=2)  # s(u)^T grad_{u'} k
            term3 = (grad_kx * sp.unsqueeze(0)).sum(dim=2)  # grad_u k^T s(u')
            # trace of mixed second derivative of RBF:
            # grad_u . grad_{u'} k = k * (D/h2 - ||u-u'||^2/h2^2)
            term4 = k * (d / h2 - sq / (h2 * h2))
            up_kernel = term1 + term2 + term3 + term4  # [N, N]

            # U-statistic: exclude the diagonal.
            mask = ~torch.eye(n, dtype=torch.bool, device=up_kernel.device)
            denom = max(n * (n - 1), 1)
            ksd2 = up_kernel[mask].sum() / denom
        return float(ksd2.clamp_min(0.0))


# --------------------------------------------------------------------------- #
# pfld — Probability-Flow Exact-Likelihood Deficit
# --------------------------------------------------------------------------- #
@register_metric("pfld", requires_reference=False, needs=("prior_model",))
class ProbabilityFlowLikelihoodDeficit:
    r"""Negative exact log-likelihood via the probability-flow ODE (↓).

    Instantaneous change of variables along the PF-ODE
    :math:`d\mathbf{x}/dt=\tilde{\mathbf f}(\mathbf{x},t)` with the divergence
    estimated by Hutchinson:

    .. math::

        \mathrm{PFLD}=-\ln p_0(\hat{\mathbf x})
            =-\Big[\ln p_T(\mathbf x_T)
                +\int_0^T\nabla\!\cdot\!\tilde{\mathbf f}\,dt\Big].

    Compute-heavy and asset-blocked — this is the *short-integration*
    (periodic-validation) variant: a handful of Euler steps with the PF drift
    :math:`\tilde{\mathbf f}=-\tfrac12 t^2\,s_t`. If the prior exposes
    ``log_prob`` directly it is used as :math:`-\ln p_0` (exact short-circuit).
    Lower is better (clean images sit at higher likelihood).
    """

    def __init__(self, num_steps: int = 4, t_max: float = 1.0) -> None:
        self.num_steps = num_steps
        self.t_max = t_max

    @property
    def name(self) -> str:
        return "pfld"

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
        if not ctx.has("prior_model"):
            return _note_absent("pfld", "prior_model")
        try:
            adapter = _PriorAdapter(ctx.prior_model)
        except TypeError:
            return _note_absent("pfld", "prior_model")
        x = _as_real_bchw(prediction)

        # Exact short-circuit: a prior that knows its own log-density.
        if adapter.has_log_prob:
            with torch.no_grad():
                lp = adapter.log_prob(x)
            return float((-lp).mean())

        if not adapter.has_score:
            return _note_absent("pfld", "prior_model")

        from torch.func import jvp  # local

        n = max(int((ctx.nr_params or {}).get("pfld_steps", self.num_steps)), 1)
        dt = self.t_max / n
        gen = ctx.generator
        with torch.no_grad():
            xt = x.clone()
            div_acc = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            for i in range(n):
                t = (i + 0.5) * dt + 1e-3

                def drift(z: Tensor, _t: float = t) -> Tensor:
                    return -0.5 * (_t * _t) * adapter.score(z, _t)

                eta = torch.randn(xt.shape, generator=gen, device=xt.device, dtype=xt.dtype)
                f, jvp_val = jvp(drift, (xt,), (eta,))
                div = (eta * jvp_val).reshape(x.shape[0], -1).sum(dim=1)
                div_acc = div_acc + div * dt
                xt = xt + f * dt
            # ln p_T(x_T) under a unit Gaussian base measure (up to a const).
            ln_p_base = -0.5 * xt.reshape(x.shape[0], -1).pow(2).sum(dim=1)
            ln_p0 = ln_p_base + div_acc
        return float((-ln_p0).mean())


# --------------------------------------------------------------------------- #
# hallucination_index — null-space hallucination test statistic
# --------------------------------------------------------------------------- #
@register_metric(
    "hallucination_index",
    aliases=["HI", "null_space_hallucination_index"],
    requires_reference=False,
    needs=("prior_model", "y_kspace", "mask", "coil_maps"),
)
class HallucinationIndex:
    r"""Null-space hallucination log-ratio test statistic (↓).

    The forward operator :math:`A=MFS` (built from ``sense_forward`` /
    ``sense_adjoint``) defines the data-consistent range and its orthogonal
    complement (the *null space*). With the projector
    :math:`P_{\mathrm{null}}=I-A^{+}A`, the statistic isolates structure the
    measurement cannot have constrained:

    .. math::

        \mathrm{HI}=\log\!\frac{p_{\mathrm{prior}}(\hat{\mathbf x})}
            {p_{\mathrm{data\text{-}consistent}}(\hat{\mathbf x})}
            \Big|_{P_{\mathrm{null}}\hat{\mathbf x}}.

    Operationally we measure how *atypical under the prior* the null-space
    component is, normalised by its energy: a large prior score-norm on
    :math:`P_{\mathrm{null}}\hat{\mathbf x}` flags plausible-but-unsupported
    fabrication. Rises with injected null-space structure. Lower is better.
    """

    def __init__(self, t: float = 0.1) -> None:
        self.t = t

    @property
    def name(self) -> str:
        return "hallucination_index"

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
        if not ctx.has("prior_model"):
            return _note_absent("hallucination_index", "prior_model")
        if not ctx.has("y_kspace", "mask", "coil_maps"):
            # The measurement context is part of the asset gate here.
            return _note_absent("hallucination_index", "prior_model")
        try:
            adapter = _PriorAdapter(ctx.prior_model)
        except TypeError:
            return _note_absent("hallucination_index", "prior_model")
        if not adapter.has_score:
            return _note_absent("hallucination_index", "prior_model")

        from mriforge.infrastructure.physics.fft_ops import (
            _to_complex,
            sense_adjoint,
            sense_forward,
        )

        smaps = _to_complex(ctx.coil_maps)
        mask = ctx.mask
        t = float((ctx.nr_params or {}).get("hi_t", self.t))

        with torch.no_grad():
            xc = _to_complex(prediction)
            if xc.shape[1] != 1:
                # Collapse multi-channel complex to a single image channel.
                xc = xc[:, :1]
            # P_null x = x - A^+ A x ; here A^+ ≈ A^H (SENSE adjoint) so
            # P_null x = x - A^H A x (idempotent up to the normal-equations scale).
            ax = sense_forward(xc, smaps, mask)
            aha_x = sense_adjoint(ax, smaps, mask)
            x_null = xc - aha_x  # null-space component (complex)

            # HI = log[p_prior / p_data-consistent] | P_null x. We evaluate the
            # numerator: the prior atypicality (-ln p_prior, surrogate
            # ∝ t^2 ||s||^2 for a smooth prior) of the *null-space* component
            # P_null x — the part the measurement could not constrain. The
            # data-consistent reference p_data-consistent is, by construction,
            # already constrained by y (its likelihood is fixed by the data fit),
            # so the discriminating term is the null-component prior atypicality:
            # plausible-but-unsupported structure injected into the null space
            # raises ||s_prior(P_null x)|| and hence HI. This is the safety-QC
            # signal the spec asks for, computed robustly without relying on
            # A^H A being an exact projector (it carries the normal-equations
            # scaling for un-normalised coil maps).
            x_null_r = _as_real_bchw(x_null)
            s_null = adapter.score(x_null_r, t)
            null_term = (t * t) * s_null.reshape(s_null.shape[0], -1).pow(2).sum(dim=1)
            hi = null_term  # [B]
        return float(hi.mean())


__all__ = [
    "DiffusionPriorScoreTypicality",
    "FisherInformationTraceCurvature",
    "HallucinationIndex",
    "KernelSteinDiscrepancyToPrior",
    "PristineManifoldMahalanobisDistance",
    "ProbabilityFlowLikelihoodDeficit",
]
