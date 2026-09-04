r"""Certified global Lipschitz-bound metrics (A-8.3 *CertRob*).

The framework already carries a *local* robustness certificate — ``lrjs``
(:class:`~spectramr.core.metrics.nr_stability.LocalReconstructionJacobianSpectralNorm`)
estimates :math:`\lVert\partial\mathcal{R}/\partial\mathbf{y}\rVert_2` at *one*
test measurement by power-iterating the Jacobian. That is the Lipschitz constant
*linearised around a point*; it says nothing global. This module adds the
complementary **global** certificate: a provable upper bound on the network's
Lipschitz constant that holds for *every* input.

For a **feed-forward** network :math:`f = L_n\circ\phi\circ\cdots\circ\phi\circ L_1`
with 1-Lipschitz activations :math:`\phi` (ReLU and LeakyReLU with slope
:math:`\le 1` are exactly 1-Lipschitz; tanh is 1-Lipschitz; **sigmoid is
0.25-Lipschitz** and GELU/SiLU are :math:`\approx 1.1`-Lipschitz, so they are
NOT 1-Lipschitz) and weight layers :math:`L_i`,

.. math::

    \mathrm{Lip}(f)\;\le\;\prod_{i=1}^{n}\sigma_{\max}(W_i),

because :math:`\sigma_{\max}(AB)\le\sigma_{\max}(A)\,\sigma_{\max}(B)` and a
1-Lipschitz activation cannot increase the operator norm. The product
:math:`\prod_i\sigma_i` is therefore a certificate: for any perturbation
:math:`\delta`, :math:`\lVert f(y+\delta)-f(y)\rVert \le (\prod_i\sigma_i)\,
\lVert\delta\rVert`. Two metrics expose it:

* ``composed_spectral_norm_bound`` — the product :math:`\prod_i\sigma_i`. The
  headline global Lipschitz upper bound. *Lower is better* (a smaller bound = a
  reconstructor that cannot amplify a worst-case perturbation as much).
* ``max_layer_spectral_norm`` — :math:`\max_i\sigma_i`. A diagnostic that
  pinpoints the layer dominating the bound (the one a Lipschitz penalty should
  push down first). *Lower is better.*

**Honest scope (the sigma estimator).** Per layer, :math:`\sigma_i` is the largest
singular value of the weight reshaped to 2-D ``[out, -1]`` — the *exact*
operator norm for a ``Linear`` layer, and the quantity
:func:`torch.nn.utils.spectral_norm` constrains. For a ``Conv`` layer this
reshape-2-D norm is the standard spectral-norm-regularisation surrogate
(Miyato et al., SN-GAN); it is **not** the exact circular-convolution operator
norm (that needs the Sedghi et al. 2019 FFT-of-kernel method, which depends on
the input spatial size). So the product of per-layer norms is a **rigorous
certificate for fully-connected stacks** and a **spectral-norm-consistent
estimate** for convolutional backbones — the oracle test pins the rigorous case
(a Linear MLP), and the conv boundary is documented rather than overclaimed. Norm
/ attention layers are not included in the product; the bound is over the
conv/linear backbone (document, don't silently assume).

**Architecture-validity scope (when is this a STRICT certificate?).** The product
is a strict Lipschitz upper bound only for a **feed-forward** stack of weight
layers + 1-Lipschitz activations. Two common backbone features break strictness
and are NOT captured by the product:

* **Residual skips** — a block :math:`f(x)=\phi(g(x)+\alpha x)` has
  :math:`\mathrm{Lip}(f)\le \mathrm{Lip}(g)+\alpha`, not :math:`\prod` over
  :math:`g`'s layers. The product *under*-counts when skips are active.
* **Learned-affine norm layers** (BatchNorm/GroupNorm/LayerNorm with
  :math:`\gamma`) can scale by :math:`|\gamma|/\sqrt{\mathrm{var}}>1`, which the
  product ignores.

For such backbones (e.g. a residual, normalized U-Net) the product is best read
as a **Lipschitz regularisation target / robustness proxy** — minimising it
provably shrinks the *dominant multiplicative term* of the true bound, and the
companion penalty is sound regardless — but it is not a strict certificate. Use a
feed-forward, 1-Lipschitz-activation, non-residual, norm-light architecture when a
strict certificate is required. The metric does not silently claim strictness it
cannot deliver; this boundary is the claim.

Both metrics read the live :class:`torch.nn.Module` from ``kwargs['model']``
(precedent:
:class:`~spectramr.core.metrics.robustness_metrics.AdversarialPSNRDropMetric`),
or from ``ctx.reconstructor`` when that is itself an ``nn.Module``. With no
module in scope they return ``float('nan')`` — a bound of ``0`` would be a false
claim of perfect contraction, so "cannot compute" is reported as ``nan``, never
fabricated (CLAUDE.md pitfall #9 / #20).

References
----------
[1] T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, "Spectral Normalization for
    Generative Adversarial Networks," *ICLR*, 2018.
[2] H. Sedghi, V. Gupta, P. M. Long, "The Singular Values of Convolutional
    Layers," *ICLR*, 2019.
[3] V. Antun et al., "On instabilities of deep learning in image
    reconstruction," *PNAS*, 117(48), 2020.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.registry import register_metric

# Weight-bearing linear maps whose reshape-2-D spectral norm enters the product.
# (Norm layers, activations, attention have separate Lipschitz behaviour and are
# intentionally excluded — see the module docstring's honest-scope note.)
_WEIGHTED_LAYERS = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def _layer_sigma(weight: torch.Tensor) -> float:
    """Largest singular value of ``weight`` reshaped to 2-D ``[out, -1]``.

    Exact operator norm for a ``Linear`` weight; the Miyato reshape-2-D
    surrogate for a ``Conv`` weight. Computed in double precision (real) or
    ``complex128`` (complex weights) so the SVD is accurate.
    """
    w2d = weight.detach().reshape(weight.shape[0], -1)
    w2d = w2d.to(torch.complex128) if w2d.is_complex() else w2d.to(torch.float64)
    return float(torch.linalg.matrix_norm(w2d, ord=2))


def layer_spectral_norms(model: nn.Module) -> list[tuple[str, float]]:
    """``[(qualified_name, sigma), …]`` for every weight-bearing layer.

    Empty when the module has no ``Linear``/``Conv`` layers (e.g. a pure
    activation stack), in which case the bound metrics report ``nan``.
    """
    out: list[tuple[str, float]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, _WEIGHTED_LAYERS):
            w = getattr(mod, "weight", None)
            if w is None or w.numel() == 0:
                continue
            out.append((name, _layer_sigma(w)))
    return out


def _resolve_module(kwargs: dict[str, Any], ctx: MetricContext) -> nn.Module | None:
    """Find the live ``nn.Module`` to certify, or ``None``.

    Order: ``kwargs['model']`` -> ``kwargs['module']`` -> ``kwargs['reconstructor']``
    -> ``ctx.reconstructor`` (only when it is itself an ``nn.Module``; the context
    field is typed as a bare callable, which a spectral-norm scan cannot use).
    """
    for key in ("model", "module", "reconstructor"):
        cand = kwargs.get(key)
        if isinstance(cand, nn.Module):
            return cand
    if isinstance(ctx.reconstructor, nn.Module):
        return ctx.reconstructor
    return None


@register_metric(
    "composed_spectral_norm_bound",
    aliases=[
        "lipschitz_bound",
        "global_lipschitz_bound",
        "spectral_norm_product",
    ],
    requires_reference=False,
    direction="lower",
)
class ComposedSpectralNormBound:
    r"""Global certified Lipschitz upper bound :math:`\prod_i\sigma_i` — lower is better.

    A provable bound on :math:`\mathrm{Lip}(f)` for a **feed-forward** net with
    1-Lipschitz activations: :math:`\lVert f(y+\delta)-f(y)\rVert\le(\prod_i
    \sigma_i)\lVert\delta\rVert` for *all* inputs. Complements the local ``lrjs``.
    Reads the live module from ``kwargs`` / ``ctx.reconstructor``; returns ``nan``
    when no module (or no weight-bearing layer) is in scope. Rigorous for Linear
    stacks; a spectral-norm-consistent estimate for conv backbones; a Lipschitz
    *regularisation target* (not a strict certificate) for residual / affine-norm
    backbones — see the module docstring's architecture-validity scope.
    """

    @property
    def name(self) -> str:
        return "composed_spectral_norm_bound"

    @property
    def higher_is_better(self) -> bool:
        # A large Lipschitz bound = a reconstructor that can amplify a worst-case
        # perturbation more (less robust) -> smaller is better.
        return False

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        model = _resolve_module(kwargs, ctx)
        if model is None:
            return float("nan")
        sigmas = layer_spectral_norms(model)
        if not sigmas:
            return float("nan")
        # Product in python float (double). For a deep, ill-conditioned net this
        # may overflow to +inf — an honest "astronomically large bound" signal,
        # which direction='lower' ranks as worst.
        bound = 1.0
        for _name, s in sigmas:
            bound *= s
        if not math.isfinite(bound) and bound > 0:
            return float("inf")
        return float(bound)


@register_metric(
    "max_layer_spectral_norm",
    aliases=["max_layer_lipschitz", "worst_layer_spectral_norm"],
    requires_reference=False,
    direction="lower",
)
class MaxLayerSpectralNorm:
    r"""Diagnostic :math:`\max_i\sigma_i` — the layer dominating the bound. Lower is better.

    Pinpoints which weight layer a Lipschitz penalty should push down first.
    Returns ``nan`` when no module / no weight-bearing layer is in scope.
    """

    @property
    def name(self) -> str:
        return "max_layer_spectral_norm"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        model = _resolve_module(kwargs, ctx)
        if model is None:
            return float("nan")
        sigmas = layer_spectral_norms(model)
        if not sigmas:
            return float("nan")
        return float(max(s for _name, s in sigmas))


# ``core/metrics/__init__`` walks every submodule on package import
# (pkgutil.walk_packages), so the ``@register_metric`` decorators above fire
# without any edit to ``__init__.py``.

__all__ = [
    "ComposedSpectralNormBound",
    "MaxLayerSpectralNorm",
    "layer_spectral_norms",
]
